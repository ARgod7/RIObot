"""
ws_server.py
------------
WebSocket server on port 8765.
Browser sends transcript → Python pipeline runs → response sent back to browser.

Messages FROM browser:
  {"type": "transcript", "text": "..."}
  {"type": "audio", "data": "<base64>"}   ← raw mic audio for voice emotion

Messages TO browser:
  {"type": "transcript_ack", "text": "..."}
  {"type": "rio_response", "text": "...", "expression": "calm"}
  {"type": "play_audio", "url": "/audio/response.mp3"}
  {"type": "emotion_update", "emotions": {...}}
"""

import asyncio
import base64
import json
import logging
import queue
import threading
from pathlib import Path
import numpy as np

import websockets

logger = logging.getLogger("ws_server")

# ── Shared state ──────────────────────────────────────────────────────────
_transcript_queue: queue.Queue = queue.Queue()
_response_queue:   queue.Queue = queue.Queue()   # main.py puts responses here
_connected_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None

# ── Public API (called from main.py) ─────────────────────────────────────

def get_latest_transcript() -> str | None:
    """Pop the latest transcript from the browser. Returns None if empty."""
    try:
        return _transcript_queue.get_nowait()
    except queue.Empty:
        return None


def send_response_to_browser(
    response_text: str,
    expression: str,
    audio_url: str,
    rio_state: dict | None = None,
    intervention_intent: dict | str | None = None,
    pipeline_feedback: dict | None = None,
) -> None:
    """
    Push a RIO response to all connected browser clients.
    Call this from main.py after the pipeline runs.
    """
    normalized_audio_url = audio_url if str(audio_url).startswith("/audio/") else "/audio/response.mp3"
    _response_queue.put({
        "response_text": response_text,
        "expression": expression,
        "audio_url": normalized_audio_url,
        "rio_state": rio_state or {},
        "intervention_intent": intervention_intent or {},
        "pipeline_feedback": pipeline_feedback or {},
    })
    # Wake up the async loop to flush the queue
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_flush_response_queue_sync)


def send_emotion_update(emotions: dict) -> None:
    """Push live emotion vector to browser (for details panel)."""
    _response_queue.put({"__type": "emotion_update", "emotions": emotions})
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_flush_response_queue_sync)


def _flush_response_queue_sync():
    """Schedules async flush from sync context."""
    if _loop:
        asyncio.ensure_future(_flush_response_queue(), loop=_loop)


# ── WebSocket handler ─────────────────────────────────────────────────────

async def _handler(websocket):
    _connected_clients.add(websocket)
    logger.info(f"[WS] Client connected. Total: {len(_connected_clients)}")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                await _handle_message(websocket, msg)
            except json.JSONDecodeError:
                logger.warning("[WS] Non-JSON message received")
    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        _connected_clients.discard(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(_connected_clients)}")


async def _handle_message(websocket, msg: dict):
    msg_type = msg.get("type")

    if msg_type == "transcript":
        text = msg.get("text", "").strip()
        if text:
            logger.info(f"[WS] Transcript received: {text[:60]}")
            _transcript_queue.put(text)
            # Ack back so browser can show it
            await websocket.send(json.dumps({
                "type": "transcript_ack",
                "text": "You: " + text
            }))

    elif msg_type == "audio":
        # Raw PCM16 audio bytes for voice emotion detection
        try:
            audio_bytes = base64.b64decode(msg.get("data", ""))
            sample_rate = int(msg.get("sampleRate", 16000))
            _feed_audio_to_voice_detector(audio_bytes, sample_rate)
        except Exception as e:
            logger.warning(f"[WS] Audio decode error: {e}")


def _feed_audio_to_voice_detector(audio_bytes: bytes, sample_rate: int):
    """Pass PCM16 audio to voice_detector if it supports feed_audio_chunk."""
    try:
        detector = _get_voice_detector()
        if detector and hasattr(detector, "feed_audio_chunk"):
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            detector.feed_audio_chunk(audio_np, sample_rate=sample_rate)
    except Exception as e:
        logger.debug(f"[WS] Voice detector feed skipped: {e}")


_voice_detector_instance = None

def _get_voice_detector():
    global _voice_detector_instance
    if _voice_detector_instance is None:
        try:
            from perception.voice_detector import VoiceEmotionDetector
            _voice_detector_instance = VoiceEmotionDetector(use_transformer=True)
            logger.info("[WS] Voice detector created for browser audio feed")
        except Exception as e:
            logger.warning(f"[WS] Voice detector init failed: {e}")
            _voice_detector_instance = None
    return _voice_detector_instance

def set_voice_detector(detector):
    """Call this from main.py after creating the voice detector."""
    global _voice_detector_instance
    _voice_detector_instance = detector


# ── Broadcast helpers ─────────────────────────────────────────────────────

async def _broadcast(payload: dict):
    if not _connected_clients:
        return
    message = json.dumps(payload)
    # Send to all, ignore disconnected
    dead = set()
    for client in _connected_clients:
        try:
            await client.send(message)
        except Exception:
            dead.add(client)
    _connected_clients.difference_update(dead)


async def _flush_response_queue():
    """Drain response queue and broadcast to all clients."""
    while not _response_queue.empty():
        try:
            item = _response_queue.get_nowait()
        except queue.Empty:
            break

        if item.get("__type") == "emotion_update":
            await _broadcast({
                "type": "emotion_update",
                "emotions": item["emotions"]
            })
        else:
            await _broadcast({
                "type": "rio_response",
                "text": item["response_text"],
                "expression": item["expression"],
                "rio_state": item.get("rio_state", {}),
                "intervention_intent": item.get("intervention_intent", {}),
                "pipeline_feedback": item.get("pipeline_feedback", {}),
            })
            await _broadcast({
                "type": "play_audio",
                "url": item["audio_url"],
            })


# ── Server startup ────────────────────────────────────────────────────────

async def _run_server():
    global _loop
    _loop = asyncio.get_running_loop()
    logger.info("[WS] WebSocket server listening on ws://localhost:8765")
    async with websockets.serve(_handler, "localhost", 8765):
        await asyncio.Future()   # run forever


def start_ws_server() -> None:
    """Start WebSocket server in a background daemon thread."""
    def _thread():
        asyncio.run(_run_server())

    t = threading.Thread(target=_thread, daemon=True, name="ws-server")
    t.start()
    logger.info("[WS] Server thread started (daemon=True)")
"""
RIO Main Loop — Perception → Engine → Dialogue → TTS → Animation.

Orchestrates the full emotionally intelligent robot companion system.
"""

import json
import subprocess
import threading
import time
import logging
import sys
import traceback
import os
import re
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from rio_bridge.ws_server import send_emotion_update
import httpx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose library logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("crewai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("comtypes").setLevel(logging.ERROR)
logging.getLogger("EmotionFusion").setLevel(logging.ERROR)
logging.getLogger("PerceptionLoop").setLevel(logging.WARNING)
logging.getLogger("VoiceDetector").setLevel(logging.WARNING)
logging.getLogger("cognition").setLevel(logging.WARNING)
logging.getLogger("rio_bridge").setLevel(logging.WARNING)

# Imports
from perception.perception_loop import (
    run as run_perception,
    get_latest_stimulus,
    get_latest_perception_debug,
)
from rio_bridge.ws_server import get_latest_transcript, start_ws_server
from memory.short_term_memory import get_summary, add_entry, hydrate_memory_from_file
from cognition.orchestration.pipeline_manager import run_pipeline
from cognition.agents.servo_agent import stop_alive_animation, close_serial
from rio_bridge.rio_client import call_rio_engine, send_expression, send_audio_to_browser
from tts.gtts_wrapper import speak
from rio_bridge.ws_server import send_response_to_browser
from config import EMOTION_WS_BROADCAST_MIN_S
from memory.short_term_memory import reset_memory
from memory.persistent_memory import set_user_name
from config import CLEAR_MEMORY_ON_START

# Logger endpoint for HTTP logger server (Node)
LOGGER_URL = os.getenv("LOGGER_URL", "http://127.0.0.1:4000").rstrip('/')
LOGGER_TIMEOUT = float(os.getenv("LOGGER_TIMEOUT", "1.0"))


def _send_log(endpoint: str, payload: dict) -> None:
    """Send a JSON log to the external logger HTTP server.

    Non-critical: failures are logged at debug level and do not raise.
    """
    try:
        url = f"{LOGGER_URL}/{endpoint.lstrip('/') }"
        httpx.post(url, json=payload, timeout=LOGGER_TIMEOUT)
    except Exception as e:
        logger.debug(f"Failed to POST to logger {endpoint}: {e}")


def has_hindi(text: str) -> bool:
    """Check if text contains Hindi/Devanagari characters."""
    return any('\u0900' <= char <= '\u097f' for char in text)


_EKMAN_KEYS = ("joy", "sadness", "fear", "disgust", "anger", "surprise")

_NAME_TOKEN = r"(?:[A-Z]|[a-z])(?:[A-Z]|[a-z]|['-])*"

_NAME_PATTERNS = [
    re.compile(rf"\bmy name is\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?)\b", re.IGNORECASE),
    re.compile(rf"\bi am\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?)\b", re.IGNORECASE),
    re.compile(rf"\bi'm\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?)\b", re.IGNORECASE),
    re.compile(rf"\bcall me\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?)\b", re.IGNORECASE),
]

# Reject "I'm just very angry" → "Just Very" false positives
_NAME_BLOCKLIST = frozenset({
    "just", "very", "really", "quite", "so", "not", "only", "also", "still", "even",
    "feeling", "angry", "mad", "upset", "sad", "happy", "fine", "okay", "ok", "sorry",
    "trying", "doing", "being", "getting", "little", "bit", "kind", "sort",
})


def _neutral_emotions() -> dict:
    """Six-way neutral vector for placeholders (details UI only counts Ekman keys)."""
    return {k: 0.0 for k in _EKMAN_KEYS}


def _vector_payload(source_status: dict, fallback: dict | None = None) -> dict:
    """Normalize source vector payload for details websocket message."""
    if source_status and source_status.get("emotions"):
        return {
            **source_status.get("emotions", {}),
            "confidence": float(source_status.get("confidence", 0.0)),
            "age_seconds": source_status.get("age_seconds"),
            "stale": bool(source_status.get("stale", False)),
        }
    payload = dict(fallback or {})
    if "confidence" not in payload:
        payload["confidence"] = 0.0
    return payload


def _broadcast_live_perception(
    stimulus_dict: dict,
    perception_debug: dict,
    *,
    fused_confidence: float,
) -> None:
    """Push fused + source vectors to browser /details (must run even when idle)."""
    source_status = perception_debug.get("sources", {})
    emotion_dynamics = perception_debug.get("emotion_dynamics", {})
    payload = {
        "face": _vector_payload(source_status.get("face", {})),
        "posture": _vector_payload(source_status.get("posture", {})),
        "voice": _vector_payload(source_status.get("voice", {})),
        "fused": _vector_payload(
            {"emotions": stimulus_dict.get("emotions", {}), "confidence": fused_confidence},
            fallback=stimulus_dict.get("emotions", {}),
        ),
        "raw_emotions": emotion_dynamics.get("raw_fused", {}),
        "smoothed_emotions": emotion_dynamics.get("smoothed_fused", {}),
        "camera": perception_debug.get("camera", {}),
        "voice_status": perception_debug.get("voice", {}),
        "loop": perception_debug.get("loop", {}),
        "emotion_dynamics": {
            "update_count": emotion_dynamics.get("update_count", 0),
            "significant_changes": emotion_dynamics.get("significant_changes", 0),
            "fps": emotion_dynamics.get("fps", 0),
        },
        "stimulus_meta": {
            "label": stimulus_dict.get("label"),
            "trust": stimulus_dict.get("trust"),
            "likeness": stimulus_dict.get("likeness"),
            "timesOccurred": stimulus_dict.get("timesOccurred"),
            "timestamp": stimulus_dict.get("timestamp"),
        },
    }
    send_emotion_update(payload)
    try:
        _send_log("emotion_update", payload)
    except Exception:
        logger.debug("Logger POST failed for emotion_update")


def start_web_server(port: int = 8000):
    project_root = Path(__file__).parent
    web_dir = project_root / "rio_web"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def do_GET(self):
            request_path = urlparse(self.path).path
            if request_path == "/details":
                self.path = "/details.html"
                request_path = "/details.html"
            # Audio must be FIRST, before super()
            if request_path.startswith("/audio/"):
                audio_path = project_root / "rio_js" / "public" / "audio" / Path(request_path).name
                if audio_path.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(audio_path.read_bytes())
                    return

            if request_path.startswith("/assets/"):
                rel = request_path[len("/assets/"):]
                asset_path = project_root / "rio_js" / "public" / "facial_expressions" / rel
                if asset_path.exists():
                    self.send_response(200)
                    ext = request_path.split(".")[-1].lower()
                    ctype = "image/png" if ext == "png" else "image/gif"
                    self.send_header("Content-Type", ctype)
                    self.end_headers()
                    self.wfile.write(asset_path.read_bytes())
                    return

            super().do_GET()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info(f"Web server started on http://0.0.0.0:{port}")
    return t


def _dominant_from_vector(vec: dict) -> tuple[str, float]:
    if not vec:
        return ("neutral", 0.0)
    dom, val = max(vec.items(), key=lambda item: item[1])
    return (str(dom), float(val))


def _fallback_rio_state(stimulus_dict: dict) -> dict:
    emotions = stimulus_dict.get("emotions", {})
    dominant, _ = _dominant_from_vector(emotions)
    return {
        "emotion_vector": dict(emotions),
        "dominant_emotion": dominant,
        "trust": stimulus_dict.get("trust", 0.0),
        "likeness": stimulus_dict.get("likeness", 0.0),
        "stagnation_counter": 0,
    }


def _extract_user_name(text: str) -> str | None:
    if not text:
        return None
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip()
            if not name:
                return None
            parts = name.split()
            if len(parts) > 2:
                parts = parts[:2]
            lowered = [p.lower() for p in parts]
            if any(p in _NAME_BLOCKLIST for p in lowered):
                return None
            if all(p in _NAME_BLOCKLIST for p in lowered):
                return None
            return " ".join(p.capitalize() for p in parts)
    return None


def _remember_user_name(text: str) -> None:
    name = _extract_user_name(text)
    if name:
        set_user_name("default_user", name)


def main():
    """
    Main loop: Perception → RIO Engine → Dialogue → TTS → Animation.
    """
    node_process = None
    perception_thread = None
    stop_event = threading.Event()

    try:
        # ============================================================
        # Step 1: Start Node.js server (old facial engine - index.js)
        # ============================================================
        logger.info("Starting Node.js facial engine (index.js on port 5000)...")
        project_root = Path(__file__).parent
        rio_js_path = project_root / "rio_js"

        # Set up environment for Node server
        env = dict(os.environ)
        env['PORT'] = '5000'  # index.js listens on port 5000
        env['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')

        node_process = subprocess.Popen(
            ["node", "index.js"],
            cwd=rio_js_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )
        logger.info("Node.js facial engine started on PID: %s (port 5000)", node_process.pid)
        time.sleep(3)

        # ============================================================
        # Step 1b: Start server.js (new LLM emotional engine on port 3001)
        # ============================================================
        logger.info("Starting Node.js LLM engine (server.js on port 3001)...")
        env['RIO_PORT'] = '3001'  # server.js on port 3001
        
        llm_engine_process = subprocess.Popen(
            ["node", "server.js"],
            cwd=rio_js_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )
        logger.info("Node.js LLM engine started on PID: %s (port 3001)", llm_engine_process.pid)
        time.sleep(2)

        # ============================================================
        # Step 1c: Start web server on port 8000 (new LLM + green dot UI)
        # ============================================================
        logger.info("Starting web server for LLM UI on http://0.0.0.0:8000...")
        web_server_thread = start_web_server(port=8000)
        time.sleep(1)


        # ============================================================
        # Step 2: Start perception in background thread
        # ============================================================
        # ============================================================
        if CLEAR_MEMORY_ON_START:
            logger.info("Clearing short-term memory on startup...")
            reset_memory()
        else:
            hydrate_memory_from_file()

        logger.info("Starting perception loop...")
        perception_thread = threading.Thread(
            target=run_perception,
            args=(stop_event,),
            daemon=True,
        )
        perception_thread.start()
        logger.info("Perception thread started")
        time.sleep(1)

        # ============================================================
        # Step 2b: Start WebSocket server for browser microphone
        # ============================================================
        logger.info("Starting WebSocket server for browser microphone (ws://0.0.0.0:8765)...")
        start_ws_server()
        time.sleep(1)  # Wait for WebSocket server to boot


        # ============================================================
        # Step 3: Main loop
        # ============================================================
        logger.info("Entering main loop...")
        tick_count = 0
        last_emotion_ws_ts = 0.0

        while not stop_event.is_set():
            tick_count += 1

            try:
                user_transcript = (get_latest_transcript() or "").strip()
                if user_transcript:
                    _remember_user_name(user_transcript)

                # Always publish live perception → /details WebSocket (not only after transcript).
                stimulus = get_latest_stimulus()
                if stimulus is None:
                    stimulus_dict = {
                        "label": "neutral",
                        "emotions": _neutral_emotions(),
                        "trust": 0.0,
                        "likeness": 0.0,
                        "timesOccurred": 0,
                        "timestamp": time.time(),
                    }
                else:
                    stimulus_dict = stimulus if isinstance(stimulus, dict) else stimulus.to_dict()

                perception_debug = get_latest_perception_debug() or {}
                now_ts = time.time()
                if user_transcript or (
                    now_ts - last_emotion_ws_ts >= EMOTION_WS_BROADCAST_MIN_S
                ):
                    _broadcast_live_perception(
                        stimulus_dict,
                        perception_debug,
                        fused_confidence=1.0 if stimulus is not None else 0.0,
                    )
                    last_emotion_ws_ts = now_ts

                # Skip dialogue pipeline until the user speaks (but emotion UI already updated).
                if not user_transcript:
                    logger.debug("[RIO] Waiting for user input...")
                    if tick_count % 25 == 0:
                        print("[RIO] Waiting for user input...")
                    time.sleep(
                        max(0.02, EMOTION_WS_BROADCAST_MIN_S - (time.time() - now_ts))
                    )
                    continue

                # Step 3c: Call Rio engine via rio_client
                rio_response = call_rio_engine(stimulus_dict)
                try:
                    _send_log("stimulus", {"stimulus": stimulus_dict, "rio_response": rio_response})
                except Exception:
                    logger.debug("Logger POST failed for stimulus")
                intervention_intent = rio_response.get("intervention_intent", "validation")
                emotion_before = stimulus_dict.get("emotions", {})

                if not rio_response.get("rio_state"):
                    rio_response["rio_state"] = _fallback_rio_state(stimulus_dict)

                # Step 3d: Get memory context
                memory_context = get_summary() or ""

                # Step 3e: Run pipeline
                logger.debug(f"[Tick {tick_count}] Running pipeline...")
                pipeline_result = run_pipeline(
                    stimulus=stimulus_dict,
                    intervention_intent=intervention_intent,
                    user_transcript=user_transcript,
                    memory_context=memory_context,
                    emotion_before=emotion_before,
                )

                response_text = pipeline_result.get("response_text", "")
                expression_intent = pipeline_result.get("expression_intent", "calm")
                tts_params = pipeline_result.get("tts_params", {"pitch": 1.0, "speed": 0.95})

                # Step 3f: Speak response
                if response_text:
                    lang = "hi" if has_hindi(response_text) else "en"
                    logger.info(f"Speaking ({lang}): {response_text[:60]}...")
                    audio_url = speak(
                        text=response_text,
                        lang=lang,
                        pitch=tts_params.get("pitch", 1.0),
                        speed=tts_params.get("speed", 0.95),
                    )


                    # Step 3f-audio: Play audio in browser
                    response_audio_file = project_root / "rio_js" / "public" / "audio" / Path(audio_url).name if audio_url else project_root / "rio_js" / "public" / "audio" / "response.mp3"
                    if not response_audio_file.exists():
                        logger.warning(f"Expected browser audio file missing: {response_audio_file}")
                    send_response_to_browser(
                        response_text=response_text,
                        expression=expression_intent,
                        audio_url=audio_url or "/audio/response.mp3",
                        rio_state=rio_response.get("rio_state", {}),
                        intervention_intent=intervention_intent,
                        pipeline_feedback=pipeline_result.get("feedback", {}),
                    )
                    try:
                        _send_log("rio_response", {
                            "response_text": response_text,
                            "expression": expression_intent,
                            "audio_url": audio_url or "/audio/response.mp3",
                            "rio_state": rio_response.get("rio_state", {}),
                            "intervention_intent": intervention_intent,
                            "pipeline_feedback": pipeline_result.get("feedback", {}),
                        })
                    except Exception:
                        logger.debug("Logger POST failed for rio_response")

                # Step 3g: Send expression to Node face
                logger.debug(f"Sending expression: {expression_intent}")
                send_expression(expression_intent)

                # Step 3h: Save to short-term memory
                _intent = intervention_intent
                if isinstance(_intent, dict):
                    _intent = json.dumps(_intent)
                add_entry(
                    user_transcript=user_transcript,
                    response_text=response_text,
                    expression_intent=expression_intent,
                    intervention_intent=str(_intent),
                    emotion_before=emotion_before,
                    activity_used=pipeline_result.get("activity_used"),
                )

                # Step 3i: Debug print
                print(f"[RIO] {expression_intent} | {response_text[:60]}...")

            except Exception as e:
                logger.error(f"Loop tick error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(2)
                continue

            # Sleep before next tick
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")

    finally:
        # ============================================================
        # Cleanup
        # ============================================================
        logger.info("Cleaning up...")

        # Stop servo animation and close serial port
        stop_alive_animation()
        close_serial()

        # Signal perception thread to stop
        stop_event.set()

        # Wait for perception thread
        if perception_thread and perception_thread.is_alive():
            perception_thread.join(timeout=2)
            logger.info("Perception thread stopped")

        # Terminate Node process
        if node_process:
            node_process.terminate()
            try:
                node_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                node_process.kill()
            logger.info("Node.js server stopped")

        logger.info("RIO shutdown complete")


if __name__ == "__main__":
    main()


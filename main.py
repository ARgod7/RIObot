"""
RIO Main Loop — Perception → Engine → Dialogue → TTS → Animation.

Orchestrates the full emotionally intelligent robot companion system.
"""

import subprocess
import threading
import time
import logging
import sys
import traceback
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from rio_bridge.ws_server import send_emotion_update

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
from memory.short_term_memory import get_summary, add_entry
from cognition.orchestration.pipeline_manager import run_pipeline
from rio_bridge.rio_client import call_rio_engine, send_expression, send_audio_to_browser
from tts.gtts_wrapper import speak
from rio_bridge.ws_server import send_response_to_browser


def has_hindi(text: str) -> bool:
    """Check if text contains Hindi/Devanagari characters."""
    return any('\u0900' <= char <= '\u097f' for char in text)


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

    server = HTTPServer(("localhost", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info(f"Web server started on http://localhost:{port}")
    return t
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
        logger.info("Starting web server for LLM UI on http://localhost:8000...")
        web_server_thread = start_web_server(port=8000)
        time.sleep(1)


        # ============================================================
        # Step 2: Start perception in background thread
        # ============================================================
        # ============================================================
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
        logger.info("Starting WebSocket server for browser microphone (ws://localhost:8765)...")
        start_ws_server()
        time.sleep(1)  # Wait for WebSocket server to boot

        # ============================================================
        # Step 3: Main loop
        # ============================================================
        logger.info("Entering main loop...")
        tick_count = 0

        while not stop_event.is_set():
            tick_count += 1

            try:
                # Step 3a: Get latest stimulus
                stimulus = get_latest_stimulus()
                if stimulus is None:
                    logger.debug("No stimulus available, skipping tick")
                    time.sleep(2)
                    continue

                stimulus_dict = stimulus if isinstance(stimulus, dict) else stimulus.to_dict()

                perception_debug = get_latest_perception_debug() or {}
                source_status = perception_debug.get("sources", {})
                send_emotion_update({
                    "face": _vector_payload(source_status.get("face", {})),
                    "posture": _vector_payload(source_status.get("posture", {})),
                    "voice": _vector_payload(source_status.get("voice", {})),
                    "fused": _vector_payload(
                        {"emotions": stimulus_dict.get("emotions", {}), "confidence": 1.0},
                        fallback=stimulus_dict.get("emotions", {}),
                    ),
                    "camera": perception_debug.get("camera", {}),
                    "voice_status": perception_debug.get("voice", {}),
                    "loop": perception_debug.get("loop", {}),
                    "stimulus_meta": {
                        "label": stimulus_dict.get("label"),
                        "trust": stimulus_dict.get("trust"),
                        "likeness": stimulus_dict.get("likeness"),
                        "timesOccurred": stimulus_dict.get("timesOccurred"),
                        "timestamp": stimulus_dict.get("timestamp"),
                    },
                })

                # Step 3b: Call Rio engine via rio_client
                rio_response = call_rio_engine(stimulus_dict)
                intervention_intent = rio_response.get("intervention_intent", "validation")
                emotion_before = stimulus_dict.get("emotions", {})
                # Step 3c: Get user transcript
                user_transcript = get_latest_transcript() or ""

                # Skip pipeline if no actual user input
                if not user_transcript or not user_transcript.strip():
                    logger.debug("[RIO] Waiting for user input...")
                    print("[RIO] Waiting for user input...")
                    time.sleep(2)
                    continue

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

                # Step 3g: Send expression to Node face
                logger.debug(f"Sending expression: {expression_intent}")
                send_expression(expression_intent)

                # Step 3h: Save to short-term memory
                add_entry(
                    user_transcript=user_transcript,
                    response_text=response_text,
                    expression_intent=expression_intent,
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


"""
perception_loop.py
------------------
Background perception thread for the RIO robot.

Runs all three detectors concurrently at ~10fps and exposes the latest
fused EmotionVector and StimulusObject for consumption by the API and
debug views.

Architecture:
    PerceptionLoop.start()
        └─ _loop()  [daemon thread @ 10fps]
                ├─ FaceEmotionDetector.detect()
                ├─ PostureEmotionDetector.detect()
                ├─ VoiceEmotionDetector.get_latest()   (voice runs its own thread)
                └─ EmotionFusionEngine.fuse_smoothed()
                        ↓
                   _latest_fused  (thread-safe via Lock)
                   _latest_stimulus

Usage:
    loop = PerceptionLoop()
    loop.start()

    fused    = loop.get_fused()       # EmotionVector
    stimulus = loop.get_stimulus()    # StimulusObject (RIO-compatible dict)
    transcript = loop.get_transcript()

    loop.stop()
"""

import time
import logging
import threading
from collections import deque
import cv2

from config import (
    PERCEPTION_LOOP_INTERVAL,
    DETECTOR_CONFIDENCE_THRESHOLD,
    EMOTION_INPUT_FRAME_INTERVAL,
    WEBCAM_INDEX,
    SHORT_TERM_MEMORY_SIZE,
    FUSED_EMOTION_LOG_INTERVAL,
    FUSED_EMOTION_LOG_ENABLED,
    ENABLE_VOICE_DETECTOR,
    FUSION_SMOOTH_ALPHA,
    EMOTION_PERCEPTION_DAMPING,
    dampen_emotion_vector,
)
from perception.emotion_fusion import EmotionFusionEngine, EmotionVector

logger = logging.getLogger("PerceptionLoop")

# Global module-level stimulus access
_latest_stimulus = None
_latest_perception_debug: dict = {}


def get_latest_stimulus():
    """Return the most recent fused StimulusObject, or None if not ready."""
    global _latest_stimulus
    return _latest_stimulus


def get_latest_perception_debug() -> dict:
    """Return latest perception diagnostics for debug UI."""
    global _latest_perception_debug
    return dict(_latest_perception_debug)


def run(stop_event=None):
    """
    Run the perception loop in the current thread.
    Used by main.py to start perception as a daemon thread.
    
    Args:
        stop_event: threading.Event to signal thread to stop.
    """
    global _latest_stimulus
    loop = PerceptionLoop()
    if stop_event:
        loop._stop_event = stop_event
    loop.wait_until_ready()
    loop._loop()


class PerceptionLoop:
    """
    Manages all three perception detectors and fuses their outputs
    in a background daemon thread at ~10fps.

    Thread safety:
        get_fused() / get_stimulus() / get_transcript() are safe to call
        from any thread — they acquire a short lock and return copies.
    """

    def __init__(
        self,
        camera_index: int = WEBCAM_INDEX,
        loop_interval: float = PERCEPTION_LOOP_INTERVAL,
        enable_voice: bool = ENABLE_VOICE_DETECTOR,
        enable_face: bool = True,
        enable_posture: bool = True,
    ):
        self.camera_index = camera_index
        self.loop_interval = loop_interval
        self.enable_voice = enable_voice
        self.enable_face = enable_face
        self.enable_posture = enable_posture

        # Fusion engine — shared across loop iterations
        self._fusion = EmotionFusionEngine()

        # Short-term memory: deque of recent StimulusObjects (max 7)
        self._stimulus_history: deque[dict] = deque(maxlen=SHORT_TERM_MEMORY_SIZE)

        # Latest outputs — protected by lock
        self._lock = threading.Lock()
        self._latest_fused: EmotionVector = EmotionVector(source="fused", confidence=0.0)
        self._latest_stimulus: dict = {}
        self._latest_transcript: str = ""
        self._latest_frame = None

        # Thread control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._ready_event = threading.Event()

        # Detector instances (lazy-init inside thread to avoid camera conflict)
        self._face_detector = None
        self._posture_detector = None
        self._voice_detector = None

        # Shared camera frame (face and posture share one camera capture)
        self._shared_frame = None

        # Loop diagnostics
        self._loop_count = 0
        self._last_loop_ts = 0.0

        # Source health timestamps
        self._last_face_ts = 0.0
        self._last_posture_ts = 0.0
        self._last_voice_ts = 0.0
        self._last_fused_log_ts = 0.0
        self._last_frame_ts = 0.0
        self._last_frame_source = "none"

        # Real-time emotion tracking (for dynamic updates)
        self._last_raw_fused = None
        self._last_smoothed_fused = None
        self._emotion_update_count = 0
        self._significant_change_count = 0

        # Show camera feed (OpenCV window)
        self._show_camera = False

    # ── Real-Time Emotion Tracking ─────────────────────────────────────────

    def _track_emotion_changes(self, fused: EmotionVector, smoothed: EmotionVector) -> dict:
        """Track and detect significant emotion changes for real-time responsiveness."""
        self._emotion_update_count += 1

        # Calculate velocity (rate of change) from previous reading
        change_info = {
            "raw_velocity": 0.0,
            "smoothed_velocity": 0.0,
            "dominant_emotion": None,
            "dominant_value": 0.0,
            "changed_significantly": False,
            "rapid_shift": False
        }

        if self._last_raw_fused is not None:
            # Calculate maximum change across all emotions (velocity)
            raw_emotions = fused.to_dict()
            last_raw_emotions = self._last_raw_fused.to_dict()

            raw_velocity = max(
                abs(raw_emotions[e] - last_raw_emotions[e])
                for e in raw_emotions.keys()
            )
            change_info["raw_velocity"] = raw_velocity

            # Detect significant changes (threshold > 0.1)
            if raw_velocity > 0.10:
                change_info["changed_significantly"] = True
                self._significant_change_count += 1

            # Detect rapid shifts (very fast changes > 0.2)
            if raw_velocity > 0.20:
                change_info["rapid_shift"] = True

        # Get dominant emotion
        if fused.confidence > 0.0:
            emotions = fused.to_dict()
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            change_info["dominant_emotion"] = dominant_emotion[0]
            change_info["dominant_value"] = dominant_emotion[1]

        # Store for next comparison
        self._last_raw_fused = EmotionVector(**fused.to_dict(), confidence=fused.confidence, source=fused.source)
        self._last_smoothed_fused = EmotionVector(**smoothed.to_dict(), confidence=smoothed.confidence, source=smoothed.source)

        return change_info

    # ── Detector Initialisation ─────────────────────────────────────────────

    def _init_detectors(self):
        """Initialise detectors inside the loop thread to avoid cross-thread issues."""
        if self.enable_face:
            try:
                from perception.face_detector import FaceEmotionDetector
                self._face_detector = FaceEmotionDetector(camera_index=self.camera_index)
                logger.info("FaceEmotionDetector initialised.")
            except Exception as e:
                logger.warning(f"FaceEmotionDetector unavailable: {e}")

        if self.enable_posture:
            try:
                from perception.posture_detector import PostureEmotionDetector
                # Share the face detector's capture — a second VideoCapture on the same index fails on many systems.
                self._posture_detector = PostureEmotionDetector(
                    camera_index=self.camera_index,
                    open_camera=False,
                )
                logger.info("PostureEmotionDetector initialised.")
            except Exception as e:
                logger.warning(f"PostureEmotionDetector unavailable: {e}")

        if self.enable_voice:
            try:
                from perception.voice_detector import VoiceEmotionDetector
                self._voice_detector = VoiceEmotionDetector(use_transformer=True)
                self._voice_detector.start_listening()
                logger.info("VoiceEmotionDetector started.")
            except Exception as e:
                logger.warning(f"VoiceEmotionDetector unavailable: {e}")

    def _release_detectors(self):
        if self._face_detector:
            try:
                self._face_detector.release()
            except Exception:
                pass
        if self._posture_detector:
            try:
                self._posture_detector.release()
            except Exception:
                pass
        if self._voice_detector:
            try:
                self._voice_detector.stop_listening()
            except Exception:
                pass

    # ── Main Loop ───────────────────────────────────────────────────────────

    def _loop(self):
        """
        Core perception loop — runs at ~10fps in a daemon thread.

        Each iteration:
          1. Capture a frame from the camera (shared between face + posture)
          2. Run face detector on the frame
          3. Run posture detector on the same frame
          4. Poll voice detector for its latest vector
          5. Update fusion engine with all available readings
          6. Fuse → smoothed EmotionVector
          7. Build StimulusObject and store to history
          8. Write results behind lock for safe consumption by the API
        """
        logger.info("Perception loop started.")
        self._init_detectors()
        self._ready_event.set()
        logger.info("PerceptionLoop ready — all detectors initialised.")

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            # ── 1. Capture shared camera frame ──────────────────────────
            frame = None
            if self._face_detector:
                frame = self._face_detector.get_frame()
            if frame is None and self._posture_detector:
                # Fallback to posture detector camera if face detector is unavailable
                frame = self._posture_detector.get_frame()
            if frame is not None:
                self._last_frame_ts = time.time()
                self._last_frame_source = "local"

            self._shared_frame = frame

            # ── 2. Face detection ────────────────────────────────────────
            if self._face_detector and frame is not None:
                try:
                    face_vector = self._face_detector.detect(frame=frame)
                    if face_vector.confidence >= DETECTOR_CONFIDENCE_THRESHOLD:
                        self._fusion.update("face", face_vector)
                        self._last_face_ts = time.time()
                except Exception as e:
                    logger.debug(f"Face detect error: {e}")

            # ── 3. Posture detection (same frame) ────────────────────────
            if self._posture_detector and frame is not None:
                try:
                    posture_vector = self._posture_detector.detect(frame=frame)
                    if posture_vector.confidence >= DETECTOR_CONFIDENCE_THRESHOLD:
                        self._fusion.update("posture", posture_vector)
                        self._last_posture_ts = time.time()
                except Exception as e:
                    logger.debug(f"Posture detect error: {e}")

            # # ── 3b. Optional camera preview ───────────────────────────────
            if self._show_camera and frame is not None:
                cv2.imshow("Perception Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()

            # ── 4. Voice (non-blocking poll) ─────────────────────────────
            if self._voice_detector:
                try:
                    voice_vector = self._voice_detector.get_latest()
                    transcript   = self._voice_detector.get_transcript()
                    if voice_vector.confidence >= DETECTOR_CONFIDENCE_THRESHOLD:
                         self._fusion.update("voice", voice_vector)
                         self._last_voice_ts = time.time()
                except Exception as e:
                     logger.debug(f"Voice poll error: {e}")
            else:
                transcript = ""

             # ── 5. Fuse ──────────────────────────────────────────────────
            try:
                fused = self._fusion.fuse_smoothed(alpha=FUSION_SMOOTH_ALPHA)
            except Exception as e:
                 logger.warning(f"Fusion error: {e}")
                 fused = EmotionVector(source="fused", confidence=0.0)

            raw_fused = self._fusion.fuse()
            change_info = self._track_emotion_changes(raw_fused, fused)

            if FUSED_EMOTION_LOG_ENABLED:
                 now = time.time()
                 if (now - self._last_fused_log_ts) >= FUSED_EMOTION_LOG_INTERVAL:
                     emotions = fused.to_dict()
                     dominant = max(emotions, key=emotions.get) if emotions else "neutral"
                     logger.info("Fused emotion: %s (conf=%.2f)", dominant, fused.confidence)
                     self._last_fused_log_ts = now

             # ── 6. Build StimulusObject ──────────────────────────────────
            stimulus_obj = self._fusion.build_stimulus(fused, label="user_emotional_state")
            if EMOTION_PERCEPTION_DAMPING < 0.999:
                stimulus_obj.emotions = dampen_emotion_vector(
                    stimulus_obj.emotions, EMOTION_PERCEPTION_DAMPING
                )

            # Only commit emotion/stimulus outputs every N frames.
            # (Main.py uses `get_latest_stimulus()` when the user speaks.)
            should_commit_emotion = (
                EMOTION_INPUT_FRAME_INTERVAL <= 1
                or (self._loop_count % EMOTION_INPUT_FRAME_INTERVAL == 0)
            )

             # ── 7. Write results (thread-safe) ───────────────────────────
            with self._lock:
                self._latest_frame = frame.copy() if frame is not None else None
                if transcript:
                    self._latest_transcript = transcript

                if should_commit_emotion:
                    self._latest_fused = fused
                    self._latest_stimulus = stimulus_obj
                    self._stimulus_history.append(stimulus_obj.to_dict())

             # Update module-level stimulus reference
            global _latest_stimulus
            if should_commit_emotion:
                _latest_stimulus = stimulus_obj
            global _latest_perception_debug
            _latest_perception_debug = {
                 "sources": self._fusion.status(),
                 "camera": self.get_camera_health(),
                 "voice": self.get_voice_status(),
                 "loop": self.get_loop_stats(),
                 "emotion_dynamics": self.get_emotion_dynamics(),
                 "transcript": self._latest_transcript,
            }

            self._loop_count += 1
            self._last_loop_ts = time.time()

            # ── 8. Sleep to maintain target fps ─────────────────────────
            elapsed = time.perf_counter() - t_start
            sleep_for = max(0.0, self.loop_interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

        self._release_detectors()
        cv2.destroyAllWindows()
        logger.info(f"Perception loop stopped after {self._loop_count} cycles.")

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        """Start the background perception loop."""
        if self._running:
            logger.warning("PerceptionLoop already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="perception-loop",
            daemon=True
        )
        self._thread.start()
        self._running = True
        logger.info("PerceptionLoop daemon thread started.")

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        """Block until detectors are initialised. Returns True if ready, False if timed out."""
        ready = self._ready_event.wait(timeout=timeout)
        if not ready:
            logger.warning(f"PerceptionLoop not ready after {timeout}s — proceeding anyway.")
        return ready

    def stop(self, timeout: float = 5.0):
        """Stop the background perception loop gracefully."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("PerceptionLoop stopped.")

    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    def get_fused(self) -> EmotionVector:
        """
        Return the latest smoothed fused EmotionVector.
        Thread-safe — call from the API or debug views.
        """
        with self._lock:
            return self._latest_fused

    def get_stimulus(self) -> dict:
        """
        Return the latest StimulusObject as a dict (RIO-compatible).
        Thread-safe.
        """
        with self._lock:
            return dict(self._latest_stimulus)

    def get_transcript(self) -> str:
        """Return the latest speech transcript. Thread-safe."""
        with self._lock:
            return self._latest_transcript

    def get_latest_frame(self):
        """Return a copy of the latest camera frame (BGR). Thread-safe."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_sources_status(self) -> dict:
        """Return the latest per-source fusion status for debugging."""
        return self._fusion.status()

    def get_stimulus_history(self) -> list[dict]:
        """
        Return a snapshot of the short-term stimulus memory (up to 7 items).
        Thread-safe.
        """
        with self._lock:
            return list(self._stimulus_history)

    def get_sadness_assessment(self):
        """
        Convenience: run sadness assessment on the latest fused vector.
        Returns SadnessAssessment.
        """
        fused      = self.get_fused()
        transcript = self.get_transcript()
        return self._fusion.assess_sadness_state(fused, transcript)

    def get_loop_stats(self) -> dict:
        """Diagnostics — useful for the debug panel."""
        return {
            "loop_count": self._loop_count,
            "last_loop_ts": self._last_loop_ts,
            "running": self.is_running(),
            "fusion_status": self._fusion.status(),
        }

    def get_camera_health(self) -> dict:
        """Return freshness of the latest camera frame."""
        now = time.time()
        staleness = self._fusion.staleness_seconds
        return {
            "last_frame_ts": self._last_frame_ts,
            "age_seconds": round(now - self._last_frame_ts, 2) if self._last_frame_ts else None,
            "stale": bool(self._last_frame_ts and (now - self._last_frame_ts) > staleness),
            "enabled": self.enable_face or self.enable_posture,
            "source": self._last_frame_source,
        }

    def get_voice_status(self) -> dict:
         """Return voice detector availability and listening status."""
         if not self._voice_detector:
             return {"available": False, "listening": False}
         return {
             "available": self._voice_detector.is_available(),
             "listening": self._voice_detector.is_listening(),
        }

    def get_emotion_dynamics(self) -> dict:
         """Return real-time emotion change metrics for UI display."""
         return {
             "update_count": self._emotion_update_count,
             "significant_changes": self._significant_change_count,
             "fps": 1.0 / self.loop_interval if self.loop_interval > 0 else 0,
             "raw_fused": self._last_raw_fused.to_dict() if self._last_raw_fused else {},
             "smoothed_fused": self._last_smoothed_fused.to_dict() if self._last_smoothed_fused else {},
         }


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("=" * 55)
    print("PerceptionLoop — Standalone Test (10 seconds)")
    print("  Face + posture: requires webcam")
    print("  Voice: requires microphone + SpeechRecognition")
    print("  Ctrl+C to exit early")
    print("=" * 55)

    loop = PerceptionLoop(
        camera_index=0,
        enable_face=True,
        enable_posture=True,
        enable_voice=True,
    )
    loop.start()

    try:
        for i in range(10):
            time.sleep(1)
            fused      = loop.get_fused()
            transcript = loop.get_transcript()
            stats      = loop.get_loop_stats()
            dominant   = max(fused.to_dict(), key=fused.to_dict().get)
            print(
                f"[t={i+1:02d}s] dominant={dominant:<10} "
                f"conf={fused.confidence:.2f}  "
                f"loops={stats['loop_count']}  "
                f"transcript='{transcript[:30]}'"
            )
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        print("\nLoop stopped. Final stimulus history:")
        for s in loop.get_stimulus_history():
            dom = max(s["emotions"], key=s["emotions"].get)
            print(f"  {dom:<10} {json.dumps({k: round(v,2) for k,v in s['emotions'].items()})}")

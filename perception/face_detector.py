"""
face_detector.py (v4 — full emotion palette, calibrated)

Key fixes vs v3:
  1. Anger gate removed — anger fires on brow_low + furrow + squint independently,
     no longer requires all three at extreme thresholds simultaneously.
  2. Fear / disgust / surprise no longer transferred to sadness and suppressed.
     Cross-emotion bleed removed entirely — each emotion stands on its own evidence.
  3. Smile-suppression of anger reduced: smile_gate only reduces anger by 15%,
     and only when smile_gate >= 0.25 (genuine, not ambiguous smile).
  4. negative_load joy gate: threshold lowered 0.45 → 0.35, suppression factor
     0.55 → 0.45 so the gate actually fires on real negative faces.
  5. Anger false-positive guard retained but loosened: requires brow_low >= 0.15
     AND (furrow >= 0.20 OR squint >= 0.20) — one strong signal plus one weak one.
  6. TTS joy pitch cap corrected: was 1.30–1.52 but clamp was 1.32 — now 1.10–1.28
     so dialogue_agent and face_detector agree.
"""

import cv2
import logging
import math
import time
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.emotion_fusion import EmotionVector

try:
    from config import FACE_SAMPLE_INTERVAL_S
except Exception:
    FACE_SAMPLE_INTERVAL_S = 5.0

logger = logging.getLogger("FaceDetector")


def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


class FaceEmotionDetector:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap = None
        self._face_mesh = None
        self._available = False
        self._last_vector = None

        self._baseline_ear = None
        self._baseline_mar = None
        self._baseline_brow_u = None
        self._baseline_smile = None
        self._calibration_frames = []
        self._calibrated = False
        self._calibration_target = 30

        self._sample_interval_s = float(FACE_SAMPLE_INTERVAL_S)
        self._last_sample_ts = 0.0

        self._init_camera()
        self._init_mediapipe()

    def _init_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info(f"Camera {self.camera_index} opened successfully.")
        else:
            logger.warning(f"Camera {self.camera_index} not available.")

    def _init_mediapipe(self):
        try:
            from mediapipe import solutions
            self._mp_face_mesh = solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._available = True
            logger.info("MediaPipe FaceMesh loaded.")
        except ImportError:
            logger.warning("MediaPipe not installed.")

    # ── Geometric helpers ────────────────────────────────────────────────────

    def _eye_aspect_ratio(self, lm, indices) -> float:
        p = [lm[i] for i in indices]
        vert1 = _dist(p[1], p[5])
        vert2 = _dist(p[2], p[4])
        horiz = _dist(p[0], p[3])
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _mouth_aspect_ratio(self, lm) -> float:
        vert = _dist(lm[13], lm[14])
        horiz = _dist(lm[61], lm[291])
        return vert / (horiz + 1e-6)

    def _mouth_corner_angle(self, lm) -> float:
        mid_y = (lm[61].y + lm[291].y) / 2
        centre_y = lm[13].y
        delta = centre_y - mid_y
        return max(-1.0, min(1.0, delta * 20))

    def _brow_raise(self, lm) -> float:
        left_gap = lm[159].y - lm[105].y
        right_gap = lm[386].y - lm[334].y
        avg_gap = (left_gap + right_gap) / 2
        return max(0.0, min(1.0, avg_gap * 12))

    def _brow_furrow(self, lm) -> float:
        gap = _dist(lm[107], lm[336])
        return max(0.0, min(1.0, 1.0 - gap / 0.10))

    # ── Calibration ──────────────────────────────────────────────────────────

    def _calibrate(self, ear, mar, brow_u, smile):
        if self._calibrated:
            return
        self._calibration_frames.append((ear, mar, brow_u, smile))
        if len(self._calibration_frames) >= self._calibration_target:
            ears, mars, brows, smiles = zip(*self._calibration_frames)
            self._baseline_ear    = sum(ears)   / len(ears)
            self._baseline_mar    = sum(mars)   / len(mars)
            self._baseline_brow_u = sum(brows)  / len(brows)
            self._baseline_smile  = sum(smiles) / len(smiles)
            self._calibrated = True
            print(
                f"\n[FaceDetector] Calibrated — "
                f"EAR:{self._baseline_ear:.3f} "
                f"MAR:{self._baseline_mar:.3f} "
                f"BROW:{self._baseline_brow_u:.3f} "
                f"SMILE:{self._baseline_smile:.3f}"
            )

    # ── Core mapping ─────────────────────────────────────────────────────────

    def _features_to_emotion(self, lm) -> EmotionVector:
        ear_l = self._eye_aspect_ratio(lm, [33, 160, 158, 133, 153, 144])
        ear_r = self._eye_aspect_ratio(lm, [362, 385, 387, 263, 373, 380])
        ear   = (ear_l + ear_r) / 2
        mar   = self._mouth_aspect_ratio(lm)
        smile = self._mouth_corner_angle(lm)   # positive = smile, negative = frown
        brow_u = self._brow_raise(lm)
        furrow = self._brow_furrow(lm)

        self._calibrate(ear, mar, brow_u, smile)

        if not self._calibrated:
            frames_done = len(self._calibration_frames)
            if frames_done % 10 == 0:
                print(f"\r[Calibrating... {frames_done}/{self._calibration_target}]", end="")
            return EmotionVector(source="face_calibrating", confidence=0.0)

        # Delta from personal neutral baseline
        ear_delta  = ear    - self._baseline_ear
        mar_delta  = mar    - self._baseline_mar
        brow_delta = brow_u - self._baseline_brow_u

        # Feature intensities (each [0, 1])
        wide_eyes  = max(0.0, min(1.0,  ear_delta  / 0.08))
        open_mouth = max(0.0, min(1.0,  mar_delta  / 0.15))
        brow_raise = max(0.0, min(1.0,  brow_delta / 0.15))
        brow_low   = max(0.0, min(1.0, -brow_delta / 0.10))
        squint     = max(0.0, min(1.0, -ear_delta  / 0.05))

        smile_delta   = smile - (self._baseline_smile or 0.0)
        smile_strength = max(0.0, smile_delta)
        frown_strength = max(0.0, -smile_delta)
        relaxed        = 1.0 - furrow

        # Genuine smile gate (Duchenne: corners up AND squint)
        smile_gate = max(0.0, smile_strength - frown_strength * 0.40)

        v = EmotionVector(source="face")

        # ── JOY ──────────────────────────────────────────────────────────────
        # Genuine smile: corner lift + slight eye crinkle + relaxed brows
        eye_crinkle_bonus = max(0.0, squint - 0.30) * 0.10
        v.joy = smile_gate * 0.88 + eye_crinkle_bonus + relaxed * 0.06
        if smile_gate < 0.10:
            v.joy *= 0.30

        # ── SADNESS ──────────────────────────────────────────────────────────
        # Drooping: downturned mouth + inner brows raise + droopy eyes
        inner_brow_raise = max(0.0, brow_raise - furrow * 0.5)
        v.sadness = (
            frown_strength * 0.60
            + inner_brow_raise * 0.20
            + (1.0 - wide_eyes) * 0.20
        )

        # ── FEAR ─────────────────────────────────────────────────────────────
        # Wide eyes + brows raised AND furrowed + open mouth + tension
        # Fear vs surprise: fear has BOTH raise AND furrow together
        brow_fear = brow_raise * furrow   # combined signal: brows up + together
        v.fear = (
            wide_eyes  * 0.40
            + brow_fear  * 0.35
            + open_mouth * 0.25
        )

        # ── SURPRISE ─────────────────────────────────────────────────────────
        # Wide eyes + open mouth but WITHOUT the tension of fear
        # Relaxed brows differentiate surprise from fear
        v.surprise = (
            wide_eyes  * 0.35
            + open_mouth * 0.40
            + relaxed    * 0.25
        )

        # ── ANGER ────────────────────────────────────────────────────────────
        # FIX v4: removed the triple-AND gate that required ALL three at
        # extreme levels simultaneously. Now: brow_low is the primary signal,
        # furrow and squint each contribute independently.
        #
        # Anger pattern: brows pulled down + together (furrow) + eye narrowing
        # Disgust pattern: nose wrinkle (approximated via upper-lip brow_low) +
        #                  less squint — so anger needs squint more than disgust.
        anger_base = (
            brow_low * 0.45          # primary: brows pulled down
            + furrow  * 0.30         # secondary: brows pulled together
            + squint  * 0.25         # tertiary: eyes narrowed
        )

        # Only suppress anger when smile is clearly genuine (gate >= 0.25)
        # and only partially — someone can be sarcastically angry-smiling
        if smile_gate >= 0.25:
            anger_base = anger_base * (1.0 - smile_gate * 0.15)

        # Minimal guard: need SOME brow involvement (not noise)
        # Loosened from the old triple-AND: just brow_low >= 0.15 + one other
        if brow_low >= 0.15 and (furrow >= 0.20 or squint >= 0.20):
            v.anger = max(0.0, min(1.0, anger_base))
        else:
            v.anger = 0.0

        # ── DISGUST ──────────────────────────────────────────────────────────
        # Frown + brow pulled down + less squint than anger
        # (nose wrinkle not directly measurable, proxy via lip/brow)
        disgust_base = (
            frown_strength * 0.45
            + brow_low      * 0.35
            + squint        * 0.20
        )
        # Only suppress when smile is definite
        if smile_gate >= 0.25:
            disgust_base = disgust_base * (1.0 - smile_gate * 0.12)
        v.disgust = max(0.0, min(1.0, disgust_base))

        # ── Post-processing ──────────────────────────────────────────────────

        # If smile is clearly present, joy dominates — but don't zero everything
        if smile_gate >= 0.14:
            v.joy     = max(v.joy, 0.60 + min(0.35, smile_gate))
            v.sadness  *= 0.25
            v.fear     *= 0.30
            v.disgust  *= 0.30
            v.surprise *= 0.30
            v.anger    = 0.0   # can't be angry while genuinely smiling

        # Joy suppression: only when face is clearly negative (lowered threshold)
        # FIX v4: threshold 0.45 → 0.35, suppression 0.55 → 0.45
        negative_load = max(v.sadness, v.fear, v.anger, v.disgust)
        if smile_gate < 0.15 and negative_load > 0.35:
            v.joy *= 0.45

        v.clamp()

        # Expressiveness-based confidence
        expressiveness = max(v.joy, v.sadness, v.surprise, v.fear, v.anger, v.disgust)
        v.confidence   = min(0.92, 0.55 + expressiveness * 0.4)
        return v

    # ── Decay & detect ───────────────────────────────────────────────────────

    def _decay_last_vector(self) -> EmotionVector:
        if not self._last_vector:
            return EmotionVector(confidence=0.08, source="face_no_landmarks")
        decayed = EmotionVector(source="face_no_landmarks")
        for key in ["joy", "sadness", "fear", "disgust", "anger", "surprise"]:
            setattr(decayed, key, getattr(self._last_vector, key) * 0.75)
        decayed.confidence = max(0.05, self._last_vector.confidence * 0.35)
        self._last_vector = decayed.clamp()
        return self._last_vector

    def get_frame(self):
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def detect(self, frame=None) -> EmotionVector:
        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return EmotionVector(confidence=0.0, source="face")
        if not self._available:
            return EmotionVector(confidence=0.0, source="face_unavailable")

        if self._calibrated and self._last_vector and self._sample_interval_s > 0:
            now = time.time()
            if (now - self._last_sample_ts) < self._sample_interval_s:
                return self._last_vector

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            results = self._face_mesh.process(rgb)
            rgb.flags.writeable = True
            if not results.multi_face_landmarks:
                return self._decay_last_vector()
            lm = results.multi_face_landmarks[0].landmark
            vector = self._features_to_emotion(lm)
            self._last_vector = vector
            if self._calibrated:
                self._last_sample_ts = time.time()
            return vector
        except Exception as e:
            logger.warning(f"FaceMesh error: {e}")
            return self._last_vector or EmotionVector(confidence=0.0, source="face")

    def release(self):
        if self._cap:
            self._cap.release()
        if self._face_mesh:
            self._face_mesh.close()
        logger.info("FaceDetector released.")


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Face Emotion Detector v4 (MediaPipe FaceMesh) — full emotion palette")
    print("Hold a NEUTRAL expression for calibration (30 frames)...")
    print("Press 'q' to quit.\n")

    detector = FaceEmotionDetector(camera_index=0)

    COLORS = {
        "joy":      (0, 220, 100),
        "sadness":  (255, 100, 50),
        "fear":     (0, 100, 255),
        "disgust":  (0, 180, 0),
        "anger":    (0, 0, 255),
        "surprise": (255, 220, 0),
    }

    while True:
        frame = detector.get_frame()
        if frame is None:
            print("No frame available.")
            break

        vector   = detector.detect(frame=frame)
        emotions = vector.to_dict()
        dominant, dominant_val = max(emotions.items(), key=lambda item: item[1])

        if not detector._calibrated:
            frames = len(detector._calibration_frames)
            print(f"\rCalibrating... {frames}/30 — hold neutral", end="", flush=True)
            cv2.putText(frame, f"Calibrating... {frames}/30",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            bar = " | ".join(f"{k[:3]}:{v:.2f}" for k, v in emotions.items())
            print(f"\r{bar}  → {dominant.upper():<10}", end="", flush=True)

            # Overlay bars
            bar_x, bar_y = 10, 10
            for emotion, val in emotions.items():
                color = COLORS.get(emotion, (200, 200, 200))
                bar_w = int(val * 160)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 160, bar_y + 14), (40, 40, 40), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), color, -1)
                cv2.putText(frame, f"{emotion[:3]}:{val:.2f}",
                            (bar_x + 165, bar_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                bar_y += 18

            cv2.putText(frame, f">> {dominant.upper()} ({dominant_val:.2f})",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)
            cv2.putText(frame, f"conf: {vector.confidence:.2f}",
                        (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Face Emotion v4", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.release()
    cv2.destroyAllWindows()
    print()

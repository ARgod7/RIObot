"""
face_detector.py (v3 — calibrated, no DeepFace, no TensorFlow)
"""

import cv2
import logging
import math
from perception.emotion_fusion import EmotionVector

logger = logging.getLogger("FaceDetector")


def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


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

        self._init_camera()
        self._init_mediapipe()

    def _init_camera(self):
        # Try Windows DirectShow backend first (most reliable on Windows)
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Fallback to default if DirectShow fails
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap.isOpened():
            # Set camera properties for better performance
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            self._cap.set(cv2.CAP_PROP_FPS, 30)       # Request 30 FPS
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
                min_tracking_confidence=0.5
            )
            self._available = True
            logger.info("MediaPipe FaceMesh loaded — no TensorFlow needed.")
        except ImportError:
            logger.warning("MediaPipe not installed.")

    def _eye_aspect_ratio(self, lm, indices) -> float:
        p = [lm[i] for i in indices]
        vert1 = _dist(p[1], p[5])
        vert2 = _dist(p[2], p[4])
        horiz = _dist(p[0], p[3])
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _mouth_aspect_ratio(self, lm) -> float:
        vert  = _dist(lm[13], lm[14])
        horiz = _dist(lm[61], lm[291])
        return vert / (horiz + 1e-6)

    def _mouth_corner_angle(self, lm) -> float:
        mid_y    = (lm[61].y + lm[291].y) / 2
        centre_y = lm[13].y
        delta    = centre_y - mid_y
        return max(-1.0, min(1.0, delta * 20))

    def _brow_raise(self, lm) -> float:
        left_gap  = lm[159].y - lm[105].y
        right_gap = lm[386].y - lm[334].y
        avg_gap   = (left_gap + right_gap) / 2
        return max(0.0, min(1.0, avg_gap * 12))

    def _brow_furrow(self, lm) -> float:
        gap = _dist(lm[107], lm[336])
        return max(0.0, min(1.0, 1.0 - gap / 0.10))

    def _calibrate(self, ear, mar, brow_u, smile):
        if self._calibrated:
            return
        self._calibration_frames.append((ear, mar, brow_u, smile))
        if len(self._calibration_frames) >= self._calibration_target:
            ears, mars, brows, smiles = zip(*self._calibration_frames)
            self._baseline_ear    = sum(ears)  / len(ears)
            self._baseline_mar    = sum(mars)  / len(mars)
            self._baseline_brow_u = sum(brows) / len(brows)
            self._baseline_smile  = sum(smiles) / len(smiles)
            self._calibrated = True
            print(f"\n[FaceDetector] Calibrated — EAR:{self._baseline_ear:.3f} "
                  f"MAR:{self._baseline_mar:.3f} BROW:{self._baseline_brow_u:.3f} "
                  f"SMILE:{self._baseline_smile:.3f}")

    def _features_to_emotion(self, lm) -> EmotionVector:
        ear_l  = self._eye_aspect_ratio(lm, [33,160,158,133,153,144])
        ear_r  = self._eye_aspect_ratio(lm, [362,385,387,263,373,380])
        ear    = (ear_l + ear_r) / 2
        mar    = self._mouth_aspect_ratio(lm)
        smile  = self._mouth_corner_angle(lm)  # positive = smile, negative = frown
        brow_u = self._brow_raise(lm)
        furrow = self._brow_furrow(lm)

        self._calibrate(ear, mar, brow_u, smile)

        if not self._calibrated:
            frames_done = len(self._calibration_frames)
            if frames_done % 10 == 0:
                print(f"\r[Calibrating... {frames_done}/{self._calibration_target}]", end="")
            return EmotionVector(source="face_calibrating", confidence=0.0)

        # All scores relative to personal neutral baseline
        ear_delta  = ear    - self._baseline_ear
        mar_delta  = mar    - self._baseline_mar
        brow_delta = brow_u - self._baseline_brow_u

        # Calculate feature intensities
        wide_eyes  = max(0.0, min(1.0,  ear_delta  / 0.08))     # Eyes wide = surprise/fear
        open_mouth = max(0.0, min(1.0,  mar_delta  / 0.15))     # Mouth open = surprise/fear
        brow_raise = max(0.0, min(1.0,  brow_delta / 0.15))     # Brows up = surprise/fear
        brow_low   = max(0.0, min(1.0, -brow_delta / 0.10))     # Brows down = anger/sad
        squint     = max(0.0, min(1.0, -ear_delta  / 0.05))     # Eyes squinted = joy/anger
        
        # Mouth curve relative to personal neutral baseline.
        smile_delta = smile - (self._baseline_smile or 0.0)
        smile_strength = max(0.0, smile_delta)
        frown_strength = max(0.0, -smile_delta)
        relaxed = 1.0 - furrow                  # Relaxed brows = joy/sadness (not anger)

        # Glasses can create artificial eye-squint readings, so require real smile evidence.
        smile_gate = max(0.0, smile_strength - frown_strength * 0.50)

        v = EmotionVector(source="face")

        # Reference-based emotion mapping (from user provided facial action units)
        
        # JOY: smile + eye involvement (slightly squinted cheeks raised)
        # Key: mouth curve positive + squint + relaxed
        v.joy = smile_gate * 0.90 + max(0.0, squint - 0.35) * 0.06 + relaxed * 0.04
        if smile_gate < 0.10:
            v.joy *= 0.22

        # SADNESS: drooping features (downturned mouth + inner brows raised + droopy eyes)
        # Key: mouth curve negative + slight brow raise (inner) + low energy
        inner_brow_raise = max(0.0, brow_raise - furrow * 0.5)
        v.sadness = frown_strength * 0.60 + inner_brow_raise * 0.20 + (1.0 - wide_eyes) * 0.20
        
        # FEAR: wide eyes + alert tension (brows up AND together + open mouth + tension)
        # Key: wide eyes + brows raised + slight tension
        brow_fear = brow_raise * furrow  # Both raised AND furrowed
        v.fear = wide_eyes * 0.40 + brow_fear * 0.35 + open_mouth * 0.25
        
        # SURPRISE: wide eyes + relaxed openness (no tension)
        # Key: wide eyes + open mouth + NO tension
        v.surprise = wide_eyes * 0.35 + open_mouth * 0.35 + relaxed * 0.30
        
        # ANGER: tension + narrowed brows + tight mouth
        # Key: brows down + together + squint + tense
        v.anger = brow_low * 0.40 + (furrow * squint) * 0.35 + (1.0 - open_mouth) * 0.25
        
        # DISGUST: nose wrinkled effect + upper lip raised (mouth corners down + frown) + narrowed brows
        # Key: frown + brow lowered + eye narrowing
        v.disgust = frown_strength * 0.40 + brow_low * 0.30 + squint * 0.30

        # If face is overall negative and smile evidence is weak, suppress false-positive joy.
        negative_load = max(v.sadness, v.fear, v.anger, v.disgust)
        if smile_gate < 0.15 and negative_load > 0.25:
            v.joy *= 0.35

        v.clamp()
        expressiveness = max(v.joy, v.sadness, v.surprise, v.fear, v.anger, v.disgust)
        v.confidence   = min(0.92, 0.55 + expressiveness * 0.4)
        return v

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


if __name__ == "__main__":
    print("Face Emotion Detector (MediaPipe FaceMesh) — no TensorFlow needed")
    print("Hold a NEUTRAL expression for calibration (30 frames)...")
    print("Press 'q' to quit.\n")

    detector = FaceEmotionDetector(camera_index=0)

    while True:
        frame = detector.get_frame()
        if frame is None:
            print("No frame available."); break

        vector   = detector.detect(frame=frame)
        emotions = vector.to_dict()
        dominant, dominant_val = max(emotions.items(), key=lambda item: item[1])

        if not detector._calibrated:
            frames = len(detector._calibration_frames)
            print(f"\rCalibrating... {frames}/30 — hold neutral expression", end="", flush=True)
            cv2.putText(frame, f"Calibrating... {frames}/30",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        else:
            bar = " | ".join(f"{k[:3]}:{v:.2f}" for k, v in emotions.items())
            print(f"\r{bar}  → {dominant.upper():<10}", end="", flush=True)
            cv2.putText(frame, f"{dominant.upper()} ({dominant_val:.2f})",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,150), 2)
            cv2.putText(frame, f"conf: {vector.confidence:.2f}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("Face Emotion (FaceMesh)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()
    print()
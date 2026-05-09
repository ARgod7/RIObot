"""
posture_detector.py
-------------------
Posture-based emotion detection module for the RIO robot perception pipeline.

Uses MediaPipe Pose to extract 33 body landmarks, then applies
rule-based heuristics (grounded in the paper's references [74-81])
to map posture → Ekman emotion vector.

Key posture-emotion mappings (from RIO paper references):
    Slouched / head-down     → sadness, fear
    Upright / open shoulders → joy, confidence (low negative)
    Arms crossed / hunched   → disgust, anger
    Startled / jerky motion  → surprise, fear
    Leaning forward          → interest (joy + surprise)
    Withdrawn / curled       → sadness, fear

Install:
    pip install mediapipe opencv-python

MediaPipe Pose landmark indices (key ones we use):
    0  = nose
    11 = left shoulder,   12 = right shoulder
    13 = left elbow,      14 = right elbow
    15 = left wrist,      16 = right wrist
    23 = left hip,        24 = right hip
    11,12 relative to 23,24 → slouch detection
"""

import cv2
import math
import logging
import time
from perception.emotion_fusion import EmotionVector

logger = logging.getLogger("PostureDetector")


def _angle(a, b, c) -> float:
    """
    Compute the angle at point b (in degrees), formed by a-b-c.
    Each point is a (x, y) tuple.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) or 1e-6
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) or 1e-6
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class PostureEmotionDetector:
    """
    Detects posture-based emotions from a webcam using MediaPipe Pose.

    Usage:
        detector = PostureEmotionDetector()
        vector = detector.detect()
        detector.release()
    """

    def __init__(
        self,
        camera_index: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5
    ):
        self.camera_index = camera_index
        self._cap = None
        self._mp_pose = None
        self._pose = None
        self._mp_drawing = None
        self._mediapipe_available = False

        self._init_camera()
        self._init_mediapipe(min_detection_confidence, min_tracking_confidence)

    def _init_camera(self):
        # Try Windows DirectShow backend first
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Fallback to default
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info(f"Camera {self.camera_index} opened for posture detection.")
        else:
            logger.warning(f"Camera {self.camera_index} not available for posture.")

    def _init_mediapipe(self, det_conf, track_conf):
        try:
            from mediapipe import solutions
            self._mp_pose = solutions.pose
            self._pose = self._mp_pose.Pose(
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf
            )
            self._mp_drawing = solutions.drawing_utils
            self._mediapipe_available = True
            logger.info("MediaPipe Pose loaded successfully.")
        except ImportError:
            logger.warning("MediaPipe not installed. Run: pip install mediapipe")

    def _landmarks_to_xy(self, landmarks, indices: list, frame_w: int, frame_h: int) -> dict:
        """Extract (x, y) pixel coords for the requested landmark indices."""
        result = {}
        for idx in indices:
            lm = landmarks[idx]
            result[idx] = (lm.x * frame_w, lm.y * frame_h)
        return result

    def _analyse_posture(self, landmarks, frame_w: int, frame_h: int) -> EmotionVector:
        """
        Rule-based heuristics mapping landmark geometry → emotion scores.

        Rules (each contributes a partial score):
          1. Shoulder drop / slouch   → sadness
          2. Head drop (nose below shoulders midpoint) → sadness, fear
          3. Elbow position (arms close to body vs open) → disgust/anger vs joy
          4. Shoulder width ratio (open vs closed) → joy vs sadness
          5. Spine straightness (shoulder-hip alignment) → positive vs negative
        """
        key_indices = [0, 11, 12, 13, 14, 23, 24]
        pts = self._landmarks_to_xy(landmarks, key_indices, frame_w, frame_h)

        nose          = pts[0]
        l_shoulder    = pts[11]
        r_shoulder    = pts[12]
        l_elbow       = pts[13]
        r_elbow       = pts[14]
        l_hip         = pts[23]
        r_hip         = pts[24]

        emotion_scores = {k: 0.0 for k in ["joy","sadness","fear","disgust","anger","surprise"]}
        evidence_count = 0

        # ── Rule 1: Shoulder droop ──────────────────────────────────────────
        # If one shoulder is significantly lower than the other → sadness
        shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1]) / (frame_h or 1)
        if shoulder_tilt > 0.04:                      # >4% of frame height
            sadness_from_tilt = min(1.0, shoulder_tilt * 8)
            emotion_scores["sadness"] += sadness_from_tilt * 0.5
            evidence_count += 1

        # ── Rule 2: Head drop ───────────────────────────────────────────────
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        head_drop = (nose[1] - shoulder_mid_y) / (frame_h or 1)
        # In image coords, y increases downward. Nose above shoulders = negative value.
        # Extreme head bow = large positive head_drop
        if head_drop > 0.05:
            sadness_from_head = min(1.0, head_drop * 5)
            emotion_scores["sadness"] += sadness_from_head * 0.3
            emotion_scores["fear"]    += sadness_from_head * 0.15
            evidence_count += 1
        elif head_drop < -0.15:
            # Head is well above shoulders (upright) → positive
            emotion_scores["joy"] += 0.3
            evidence_count += 1

        # ── Rule 3: Shoulder width (openness) ──────────────────────────────
        # Wide shoulders relative to hip width → open/positive posture
        shoulder_width = _dist(l_shoulder, r_shoulder)
        hip_width      = _dist(l_hip, r_hip)
        ratio = shoulder_width / (hip_width or 1)

        if ratio > 1.2:       # open, expansive
            emotion_scores["joy"] += 0.35
            evidence_count += 1
        elif ratio < 0.85:    # closed, hunched
            emotion_scores["sadness"] += 0.25
            emotion_scores["fear"]    += 0.1
            evidence_count += 1

        # ── Rule 4: Elbow proximity to body ────────────────────────────────
        # Arms crossed / elbows inward → defensive → disgust/anger
        l_elbow_in = l_elbow[0] - l_shoulder[0]   # negative = elbow crosses body
        r_elbow_in = r_shoulder[0] - r_elbow[0]
        arms_crossed_score = max(0, l_elbow_in) / (frame_w or 1) + max(0, r_elbow_in) / (frame_w or 1)
        if arms_crossed_score > 0.05:
            emotion_scores["disgust"] += min(0.4, arms_crossed_score * 3) * 0.5
            emotion_scores["anger"]   += min(0.3, arms_crossed_score * 2) * 0.3
            evidence_count += 1

        # ── Rule 5: Overall spine alignment ────────────────────────────────
        # If shoulder midpoint and hip midpoint are well aligned vertically → upright
        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_mid_x      = (l_hip[0] + r_hip[0]) / 2
        spine_offset   = abs(shoulder_mid_x - hip_mid_x) / (frame_w or 1)

        if spine_offset < 0.05:   # good alignment
            emotion_scores["joy"] += 0.2
            evidence_count += 1
        elif spine_offset > 0.12:  # leaning/slouching sideways
            emotion_scores["sadness"] += 0.2
            evidence_count += 1

        # ── Normalise: clamp each score to [0, 1] ──────────────────────────
        for key in emotion_scores:
            emotion_scores[key] = min(1.0, emotion_scores[key])

        # Confidence scales with how many rules fired and landmark visibility
        confidence = min(0.9, 0.4 + evidence_count * 0.1) if evidence_count > 0 else 0.3

        return EmotionVector(
            **emotion_scores,
            confidence=confidence,
            source="posture"
        ).clamp()

    def detect(self, frame=None) -> EmotionVector:
        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return EmotionVector(confidence=0.0, source="posture")
        if not self._mediapipe_available:
            return self._mock_detect()

        # MediaPipe needs a fresh RGB copy every time — don't reuse arrays
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # perf hint to MediaPipe

        try:
            results = self._pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if not results.pose_landmarks:
                return EmotionVector(confidence=0.1, source="posture")

            frame_h, frame_w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark
            vector = self._analyse_posture(landmarks, frame_w, frame_h)

             # Draw skeleton on the original frame (in-place)
            if self._mp_drawing is not None:
                self._mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    self._mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self._mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            return vector

        except Exception as e:
            logger.warning(f"Pose analysis error: {e}")
            return EmotionVector(confidence=0.0, source="posture")

    def get_frame(self):
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def _mock_detect(self) -> EmotionVector:
        return EmotionVector(sadness=0.5, confidence=0.4, source="posture_mock")

    def release(self):
        if self._cap:
            self._cap.release()
        if self._pose:
            self._pose.close()
        logger.info("Posture detector released.")


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Posture Emotion Detector — Live Test")
    print("Press 'q' to quit.\n")

    detector = PostureEmotionDetector(camera_index=0)

    while True:
        frame = detector.get_frame()
        if frame is None:
            print("No frame available.")
            break

        # detect() now draws the skeleton onto frame in-place
        vector = detector.detect(frame=frame)

        emotions = vector.to_dict()
        dominant = max(emotions, key=emotions.get)
        bar = " | ".join(f"{k[:3]}:{v:.2f}" for k, v in emotions.items())
        print(f"\r{bar}  → {dominant.upper():<10}", end="", flush=True)

        # Overlay emotion bars on the frame
        h, w = frame.shape[:2]
        colors = {
            "joy": (0,255,100), "sadness": (255,100,0), "fear": (0,100,255),
            "disgust": (0,200,0), "anger": (0,0,255), "surprise": (255,255,0)
        }
        bar_x, bar_y = 10, 20
        for emotion, val in emotions.items():
            color = colors.get(emotion, (200,200,200))
            bar_w = int(val * 150)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+150, bar_y+14), (50,50,50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+14), color, -1)
            cv2.putText(frame, f"{emotion[:3]}:{val:.2f}",
                        (bar_x+155, bar_y+11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
            bar_y += 18

        cv2.putText(frame, f">> {dominant.upper()} ({emotions[dominant]:.2f})",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)

        cv2.imshow("Posture Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()
    print()

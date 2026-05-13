"""
face_detector.py (v5 — hybrid learned model + landmark geometry)

Architecture:
  PRIMARY  : ONNX emotion model (MobileNet fine-tuned on RAF-DB / AffectNet)
             → stable, generalises across glasses/lighting/skin tone
  SECONDARY: MediaPipe FaceMesh landmark geometry (v4 rules)
             → personal-baseline calibrated, no model needed, used as blend/fallback

Fusion:
  model_confidence >= 0.45  →  80% model  / 20% geometry
  model_confidence  0.25–0.45 → 55% model  / 45% geometry
  model_confidence  < 0.25  →  25% model  / 75% geometry
  model unavailable          → 100% geometry

Temporal smoothing:
  EMA (alpha=0.35) applied per-frame.
  Hysteresis (margin=0.12): dominant label only switches when new leader
  exceeds current leader by 0.12 — prevents rapid flicker.

Model:
  Downloaded on first run from Hugging Face Hub (~12 MB ONNX).
  Cached at ~/.rio_models/face_emotion.onnx
  Falls back gracefully to geometry-only if download fails.
  Requires: pip install huggingface_hub onnxruntime
"""

import cv2
import logging
import math
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.emotion_fusion import EmotionVector

try:
    from config import FACE_SAMPLE_INTERVAL_S
except Exception:
    FACE_SAMPLE_INTERVAL_S = 3.0

logger = logging.getLogger("FaceDetector")

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_DIR  = Path(__file__).parent / "rio_models"
MODEL_PATH = MODEL_DIR / "face_emotion.onnx"

HF_SOURCES = [
    (
        "onnxmodelzoo/emotion-ferplus-8",
        "model/emotion-ferplus-8.onnx",
        ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"],
    ),
]

LABEL_TO_FIELD: Dict[str, str] = {
    "happiness": "joy",   "happy": "joy",      "joy": "joy",
    "sadness":   "sadness", "sad": "sadness",
    "anger":     "anger",   "angry": "anger",
    "fear":      "fear",    "fearful": "fear",
    "surprise":  "surprise","surprised": "surprise",
    "disgust":   "disgust", "disgusted": "disgust", "contempt": "disgust",
    "neutral":   "neutral", "calm": "neutral",
}

MODEL_INPUT_SIZE = (64, 64)   # emotion-ferplus-8 expects 64×64 grayscale

# ── Smoothing config ──────────────────────────────────────────────────────────

EMA_ALPHA         = 0.35
HYSTERESIS_MARGIN = 0.12


# ── Geometry helper ───────────────────────────────────────────────────────────

def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ── Model downloader ──────────────────────────────────────────────────────────

def _download_model() -> Optional[Tuple[Path, List[str]]]:
    import urllib.request

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for url, labels in MODEL_URLS:
        try:
            logger.info(f"Downloading model from {url} ...")
            dest = MODEL_DIR / "face_emotion.onnx"
            tmp  = MODEL_DIR / "face_emotion.onnx.tmp"
            req = urllib.request.Request(url, headers={"User-Agent": "rio-face-detector/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            (MODEL_DIR / "face_emotion_labels.txt").write_text("\n".join(labels))
            logger.info(f"Model saved to {dest}")
            return dest, labels
        except Exception as e:
            logger.warning(f"Download failed ({url}): {e}")

    logger.warning("All model sources failed — geometry-only mode.")
    return None


def _load_cached_labels() -> Optional[List[str]]:
    p = MODEL_DIR / "face_emotion_labels.txt"
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(face_bgr) -> "np.ndarray":
    import numpy as np
    h, w = face_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    cropped = face_bgr[y0:y0+s, x0:x0+s]
    resized = cv2.resize(cropped, MODEL_INPUT_SIZE)
    # FERPlus expects raw grayscale pixel values (0-255), shape (1, 1, 64, 64)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype("float32")
    return gray[None, None]  # shape: (1, 1, 64, 64)


def _softmax(x) -> "np.ndarray":
    import numpy as np
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── ONNX wrapper ──────────────────────────────────────────────────────────────

class ONNXEmotionModel:
    def __init__(self, path: Path, labels: List[str]):
        import onnxruntime as ort
        self.labels = [l.lower().strip() for l in labels]
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"ONNX model ready: {path.name}  labels={self.labels}")

    def predict(self, face_bgr) -> Optional[Dict[str, float]]:
        try:
            inp = _preprocess(face_bgr)
            probs = _softmax(self.session.run(None, {self.input_name: inp})[0][0])

            scores = {k: 0.0 for k in ["joy","sadness","fear","disgust","anger","surprise"]}
            neutral = 0.0
            for i, label in enumerate(self.labels):
                field = LABEL_TO_FIELD.get(label)
                if field == "neutral":
                    neutral = float(probs[i])
                elif field in scores:
                    scores[field] += float(probs[i])

            # Neutral → small joy baseline (face is just relaxed)
            if neutral > 0.0:
                scores["joy"] = max(scores["joy"], neutral * 0.25)

            scores["_confidence"] = min(0.95, max(0.30, 1.0 - neutral * 0.60))
            return scores
        except Exception as e:
            logger.warning(f"ONNX inference error: {e}")
            return None


# ── Temporal smoother ─────────────────────────────────────────────────────────

class EmotionSmoother:
    KEYS = ["joy","sadness","fear","disgust","anger","surprise"]

    def __init__(self, alpha: float = EMA_ALPHA, hysteresis: float = HYSTERESIS_MARGIN):
        self.alpha = alpha
        self.hysteresis = hysteresis
        self._s = {k: 0.0 for k in self.KEYS}
        self._dominant: Optional[str] = None
        self._init = False

    def update(self, raw: Dict[str, float]) -> Dict[str, float]:
        if not self._init:
            self._s = {k: raw.get(k, 0.0) for k in self.KEYS}
            self._init = True
            self._dominant = max(self._s, key=self._s.get)
            return dict(self._s)

        for k in self.KEYS:
            self._s[k] = self.alpha * raw.get(k, 0.0) + (1 - self.alpha) * self._s[k]

        best = max(self._s, key=self._s.get)
        if self._dominant is None:
            self._dominant = best
        elif best != self._dominant:
            if self._s[best] - self._s[self._dominant] >= self.hysteresis:
                self._dominant = best

        return dict(self._s)

    def reset(self):
        self._s = {k: 0.0 for k in self.KEYS}
        self._dominant = None
        self._init = False


# ── Main detector ─────────────────────────────────────────────────────────────

class FaceEmotionDetector:

    def __init__(self, camera_index: int = 0, auto_download_model: bool = True):
        self.camera_index = camera_index
        self._cap = None
        self._face_mesh = None
        self._mp_available = False
        self._last_vector = None
        self._model: Optional[ONNXEmotionModel] = None
        self._smoother = EmotionSmoother()

        # Geometry calibration state
        self._calibration_frames = []
        self._calibrated = False
        self._calibration_target = 30
        self._baseline_ear = self._baseline_mar = None
        self._baseline_brow_u = self._baseline_smile = None

        self._sample_interval_s = float(FACE_SAMPLE_INTERVAL_S)
        self._last_sample_ts = 0.0

        self._init_camera()
        self._init_mediapipe()
        self._init_model(auto_download_model)

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info(f"Camera {self.camera_index} opened.")
        else:
            logger.warning(f"Camera {self.camera_index} unavailable.")

    def _init_mediapipe(self):
        try:
            from mediapipe import solutions
            self._mp_face_mesh = solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
            self._mp_available = True
            logger.info("MediaPipe FaceMesh loaded.")
        except ImportError:
            logger.warning("MediaPipe not installed.")

    def _init_model(self, auto_download: bool):
        if MODEL_PATH.exists():
            labels = _load_cached_labels()
            if labels:
                try:
                    self._model = ONNXEmotionModel(MODEL_PATH, labels)
                    return
                except Exception as e:
                    logger.warning(f"Cached model load failed: {e} — re-downloading.")
                    MODEL_PATH.unlink(missing_ok=True)

        if auto_download:
            result = _download_model()
            if result:
                try:
                    self._model = ONNXEmotionModel(result[0], result[1])
                except Exception as e:
                    logger.warning(f"Post-download model init failed: {e}")
        else:
            logger.info("auto_download=False — geometry-only.")

    # ── Geometry layer ────────────────────────────────────────────────────────

    def _ear(self, lm, idx) -> float:
        p = [lm[i] for i in idx]
        return (_dist(p[1],p[5]) + _dist(p[2],p[4])) / (2.0*_dist(p[0],p[3]) + 1e-6)

    def _mar(self, lm) -> float:
        return _dist(lm[13],lm[14]) / (_dist(lm[61],lm[291]) + 1e-6)

    def _smile(self, lm) -> float:
        mid_y = (lm[61].y + lm[291].y) / 2
        # In MediaPipe, Y increases downward. Mouth corners rise (lower Y) when smiling,
        # so (mid_y - lm[13].y) is positive for a smile.
        return max(-1.0, min(1.0, (mid_y - lm[13].y) * 20))

    def _brow_raise(self, lm) -> float:
        return max(0.0, min(1.0, ((lm[159].y - lm[105].y + lm[386].y - lm[334].y) / 2) * 12))

    def _brow_furrow(self, lm) -> float:
        return max(0.0, min(1.0, 1.0 - _dist(lm[107], lm[336]) / 0.10))

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
            print(f"\n[FaceDetector] Calibrated EAR:{self._baseline_ear:.3f} "
                  f"MAR:{self._baseline_mar:.3f}")

    def _geometry_scores(self, lm) -> Optional[Dict[str, float]]:
        ear    = (self._ear(lm,[33,160,158,133,153,144]) + self._ear(lm,[362,385,387,263,373,380])) / 2
        mar    = self._mar(lm)
        smile  = self._smile(lm)
        brow_u = self._brow_raise(lm)
        furrow = self._brow_furrow(lm)

        self._calibrate(ear, mar, brow_u, smile)
        if not self._calibrated:
            return None

        ed = ear    - self._baseline_ear
        md = mar    - self._baseline_mar
        bd = brow_u - self._baseline_brow_u

        wide_eyes  = max(0.0, min(1.0,  ed / 0.08))
        open_mouth = max(0.0, min(1.0,  md / 0.15))
        brow_raise = max(0.0, min(1.0,  bd / 0.15))
        brow_low   = max(0.0, min(1.0, -bd / 0.10))
        squint     = max(0.0, min(1.0, -ed / 0.05))
        relaxed    = 1.0 - furrow

        sd = smile - (self._baseline_smile or 0.0)
        smile_gate = max(0.0, sd - max(0.0, -sd) * 0.40)
        frown      = max(0.0, -sd)

        joy      = smile_gate * 0.88 + max(0.0, squint-0.30)*0.10 + relaxed*0.06
        if smile_gate < 0.10: joy *= 0.30

        # Sadness: frown OR inner brow raise (oblique brows), soft eyes, no smile
        # Don't require deep frown — even flat/slightly-down corners count
        sadness  = frown*0.45 + brow_raise*0.30 + (1-wide_eyes)*0.15 + (1.0-smile_gate)*0.10
        sadness  = max(0.0, sadness - smile_gate*0.60)  # smile suppresses sadness

        # Fear: wide eyes are the primary cue; brow raise + open mouth boost it
        # Lower the bar — wide eyes alone at 0.5 should read as fear
        fear     = wide_eyes*0.50 + brow_raise*0.30 + open_mouth*0.20
        fear     = max(0.0, fear - smile_gate*0.80)  # smile kills fear

        surprise = wide_eyes*0.35 + open_mouth*0.40 + relaxed*0.25

        # Anger: needs brow involvement
        ab = brow_low*0.45 + furrow*0.30 + squint*0.25
        if smile_gate >= 0.25: ab *= (1 - smile_gate*0.15)
        anger = max(0.0, min(1.0, ab)) if (brow_low >= 0.15 and (furrow >= 0.20 or squint >= 0.20)) else 0.0

        db = frown*0.45 + brow_low*0.35 + squint*0.20
        if smile_gate >= 0.25: db *= (1 - smile_gate*0.12)
        disgust = max(0.0, min(1.0, db))

        if smile_gate >= 0.14:
            joy = max(joy, 0.60 + min(0.35, smile_gate))
            sadness *= 0.20; fear *= 0.20; disgust *= 0.30; surprise *= 0.30; anger = 0.0

        if smile_gate < 0.15 and max(sadness, fear, anger, disgust) > 0.30:
            joy *= 0.40

        return {k: min(1.0, v) for k, v in
                [("joy",joy),("sadness",sadness),("fear",fear),
                 ("disgust",disgust),("anger",anger),("surprise",surprise)]}

    # ── Face ROI ──────────────────────────────────────────────────────────────

    def _face_roi(self, frame, lm):
        h, w = frame.shape[:2]
        xs = [int(l.x*w) for l in lm]; ys = [int(l.y*h) for l in lm]
        x1 = max(0, min(xs)-20); x2 = min(w, max(xs)+20)
        y1 = max(0, min(ys)-20); y2 = min(h, max(ys)+20)
        roi = frame[y1:y2, x1:x2]
        return roi if roi.shape[0] >= 32 and roi.shape[1] >= 32 else None

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(self, model: Optional[Dict], geo: Optional[Dict]) -> Tuple[Dict, float]:
        KEYS = ["joy","sadness","fear","disgust","anger","surprise"]
        if model is None and geo is None:
            return {k:0.0 for k in KEYS}, 0.0
        if model is None:
            return geo, 0.55
        if geo is None:
            c = model.get("_confidence", 0.70)
            return {k: model.get(k,0.0) for k in KEYS}, c

        mc = model.get("_confidence", 0.50)
        if   mc >= 0.45: mw, gw = 0.80, 0.20
        elif mc >= 0.25: mw, gw = 0.55, 0.45
        else:            mw, gw = 0.25, 0.75

        # FERPlus has a happiness bias — dampen model joy when geo doesn't agree
        model_scores = {k: model.get(k, 0.0) for k in KEYS}
        if model_scores.get("joy", 0) > 0.35 and geo.get("joy", 0) < 0.20:
            model_scores["joy"] *= 0.50

        fused = {k: mw*model_scores.get(k,0.0) + gw*geo.get(k,0.0) for k in KEYS}
        conf  = min(0.92, 0.50 + mc*0.40 + max(fused.values())*0.10)
        return fused, conf

    # ── Decay ─────────────────────────────────────────────────────────────────

    def _decay(self) -> EmotionVector:
        if not self._last_vector:
            return EmotionVector(confidence=0.08, source="face_no_landmarks")
        d = EmotionVector(source="face_no_landmarks")
        for k in ["joy","sadness","fear","disgust","anger","surprise"]:
            setattr(d, k, getattr(self._last_vector, k) * 0.75)
        d.confidence = max(0.05, self._last_vector.confidence * 0.35)
        self._last_vector = d.clamp()
        return self._last_vector

    # ── Public API ────────────────────────────────────────────────────────────

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
        if not self._mp_available:
            return EmotionVector(confidence=0.0, source="face_unavailable")

        if self._calibrated and self._last_vector and self._sample_interval_s > 0:
            if (time.time() - self._last_sample_ts) < self._sample_interval_s:
                return self._last_vector

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            results = self._face_mesh.process(rgb)
            rgb.flags.writeable = True
            if not results.multi_face_landmarks:
                return self._decay()

            lm = results.multi_face_landmarks[0].landmark

            geo = self._geometry_scores(lm)
            if not self._calibrated:
                return EmotionVector(source="face_calibrating", confidence=0.0)

            model_out = None
            if self._model:
                roi = self._face_roi(frame, lm)
                if roi is not None:
                    model_out = self._model.predict(roi)

            raw, confidence = self._fuse(model_out, geo)
            smoothed = self._smoother.update(raw)

            src = "face_hybrid" if model_out is not None else "face_geometry"
            vector = EmotionVector(
                joy=smoothed["joy"], sadness=smoothed["sadness"],
                fear=smoothed["fear"], disgust=smoothed["disgust"],
                anger=smoothed["anger"], surprise=smoothed["surprise"],
                confidence=confidence, source=src,
            ).clamp()

            self._last_vector    = vector
            self._last_sample_ts = time.time()
            return vector

        except Exception as e:
            logger.warning(f"detect() error: {e}")
            return self._last_vector or EmotionVector(confidence=0.0, source="face")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def release(self):
        if self._cap:      self._cap.release()
        if self._face_mesh: self._face_mesh.close()
        logger.info("FaceDetector released.")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Face Emotion Detector v5 (ONNX + geometry hybrid)")
    print("Hold NEUTRAL for 30 frames calibration, then express freely.")
    print("Press 'q' to quit.\n")

    det = FaceEmotionDetector(camera_index=0, auto_download_model=True)
    COLORS = {"joy":(0,220,100),"sadness":(255,100,50),"fear":(0,100,255),
              "disgust":(0,180,0),"anger":(0,0,255),"surprise":(255,220,0)}

    while True:
        frame = det.get_frame()
        if frame is None:
            print("No frame."); break

        v   = det.detect(frame=frame)
        em  = v.to_dict()
        dom, dv = max(em.items(), key=lambda x: x[1])

        if not det.calibrated:
            n = len(det._calibration_frames)
            print(f"\rCalibrating {n}/30 — hold neutral", end="", flush=True)
            cv2.putText(frame, f"Calibrating {n}/30", (10,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        else:
            bar = " | ".join(f"{k[:3]}:{val:.2f}" for k, val in em.items())
            print(f"\r[{v.source.replace('face_','')}] {bar} → {dom.upper():<10}",
                  end="", flush=True)
            bx, by = 10, 10
            for emo, val in em.items():
                bw = int(val * 160)
                cv2.rectangle(frame,(bx,by),(bx+160,by+14),(40,40,40),-1)
                cv2.rectangle(frame,(bx,by),(bx+bw, by+14),COLORS.get(emo,(200,200,200)),-1)
                cv2.putText(frame,f"{emo[:3]}:{val:.2f}",(bx+165,by+11),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
                by += 18
            cv2.putText(frame,f">> {dom.upper()} ({dv:.2f})  [{v.source}]",
                        (10,frame.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,200),2)

        cv2.imshow("RIO Face Emotion v5", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    det.release()
    cv2.destroyAllWindows()
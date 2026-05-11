"""
emotion_fusion.py  (v2 — calibration patch)
--------------------------------------------
Central emotion fusion engine for the RIO Elderly Care Robot.

Changes from v1 → v2 (Day 2 calibration patch):
  1. sadness_focus_multiplier raised  1.15 → 1.35
  2. sadness_threshold lowered        0.42 → 0.40
  3. sustained_cycles_required lowered   3 → 2  (catch depression faster)
  4. Per-source EMOTION_BIAS offsets applied before fusion (from config.py)
  5. Anger↔sadness disambiguation:
       if posture=sad AND voice does NOT confirm anger → transfer 50% anger→sadness
  6. Joy suppression gate:
       if sadness > 0.40 AND joy < 0.35 → dynamically cap joy

Everything else (weights, StimulusObject, SadnessAssessment, helpers) unchanged.

Architecture:
  EmotionVector (dataclass)
      ↓
  EmotionFusionEngine.fuse()
      ↓
  StimulusObject (dict, emotion data container)
"""

import time
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Ensure config.py in project root is discoverable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmotionFusion")


# ─── Data Structures ────────────────────────────────────────────────────────

EMOTION_KEYS = ["joy", "sadness", "fear", "disgust", "anger", "surprise"]


@dataclass
class EmotionVector:
    """
    A 6D Ekman emotion vector.
    Each value ∈ [0.0, 1.0].
    confidence ∈ [0.0, 1.0] — how reliable this reading is.
    source: which detector produced this (for logging).
    """
    joy:      float = 0.0
    sadness:  float = 0.0
    fear:     float = 0.0
    disgust:  float = 0.0
    anger:    float = 0.0
    surprise: float = 0.0
    confidence: float = 1.0   # not part of RIO vector, used for fusion weighting
    source: str = "unknown"   # "face" | "posture" | "voice"

    def clamp(self) -> "EmotionVector":
        """Clamp all emotion values to [0, 1]."""
        for key in EMOTION_KEYS:
            setattr(self, key, max(0.0, min(1.0, getattr(self, key))))
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in EMOTION_KEYS}

    def __repr__(self):
        vals = ", ".join(f"{k}={getattr(self,k):.2f}" for k in EMOTION_KEYS)
        return f"EmotionVector({vals}, conf={self.confidence:.2f}, src={self.source})"


@dataclass
class StimulusObject:
    """
    RIO Stimulus Object — matches the structure expected by the RIO engine.
    See paper Section 5.1 for the full specification.
    """
    label: str
    emotions: dict                          # 6D Ekman dict
    personality: dict = field(default_factory=lambda: {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "agreeableness": 0.5,
        "extraversion": 0.5,
        "neuroticism": 0.5
    })
    trust: float = 0.7                      # affective trust α ∈ [0,1]
    likeness: float = 0.7                   # l ∈ [0,1]
    times_occurred: int = 1                 # ρ
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "emotions": self.emotions,
            "personality": self.personality,
            "trust": self.trust,
            "likeness": self.likeness,
            "timesOccurred": self.times_occurred,
            "timestamp": self.timestamp
        }


@dataclass
class SadnessAssessment:
    """Decision object used by the therapeutic sadness-first policy."""
    sadness_score: float
    confidence: float
    dominant_emotion: str
    dominant_value: float
    sustained: bool
    support_recommended: bool
    escalation_needed: bool
    keyword_hits: list[str] = field(default_factory=list)


# ─── Fusion Weights ─────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "face":    0.45,   # most reliable real-time signal
    "posture": 0.30,   # good for overall emotional posture
    "voice":   0.25,   # higher latency, used after speech ends
}


# ─── Fusion Engine ──────────────────────────────────────────────────────────

class EmotionFusionEngine:
    """
    Combines up to 3 emotion vectors (face, posture, voice) into
    a single fused 6D Ekman vector using weighted averaging.

    v2 changes:
      - EMOTION_BIAS offsets applied per-source before fusion
      - Anger↔sadness disambiguation post-fusion
      - Joy suppression gate post-fusion
      - Tuned sadness sensitivity parameters

    Usage:
        engine = EmotionFusionEngine()
        engine.update("face",    face_vector)
        engine.update("posture", posture_vector)
        engine.update("voice",   voice_vector)
        fused = engine.fuse()
        stimulus = engine.build_stimulus(fused, label="user_emotional_state")
    """

    def __init__(
        self,
        weights: dict = None,
        staleness_seconds: float = 5.0,
        min_confidence: float = 0.1,
        # v2: raised sadness sensitivity; tune down slightly to avoid over-triggering sadness.
        sadness_focus_multiplier: float = 1.25,
        # v2: trigger/support thresholds (tuned to reduce sadness being selected too often).
        sadness_threshold: float = 0.42,
        sadness_high_threshold: float = 0.64,
        # v2: lowered from 3 → 2 (catch depression faster)
        sustained_cycles_required: int = 2,
        support_cooldown_seconds: float = 18.0,
        emotion_bias: dict = None,          # v2: per-source bias offsets
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.staleness_seconds = staleness_seconds
        self.min_confidence = min_confidence
        self.sadness_focus_multiplier = sadness_focus_multiplier
        self.sadness_threshold = sadness_threshold
        self.sadness_high_threshold = sadness_high_threshold
        self.sustained_cycles_required = sustained_cycles_required
        self.support_cooldown_seconds = support_cooldown_seconds

        # v2: load bias from config.py if not passed directly
        if emotion_bias is not None:
            self.emotion_bias = emotion_bias
        else:
            try:
                import importlib.util
                config_path = Path(__file__).parent.parent / "config.py"
                if config_path.exists():
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    self.emotion_bias = getattr(config_module, "EMOTION_BIAS", {})
                else:
                    logger.warning("config.py not found — EMOTION_BIAS not applied.")
                    self.emotion_bias = {}
            except Exception:
                logger.warning("config.py not found — EMOTION_BIAS not applied.")
                self.emotion_bias = {}

        # Stores the latest EmotionVector per source
        self._readings: dict[str, EmotionVector] = {}
        self._last_update: dict[str, float] = {}

        # Running history for smoothing (last N fused vectors)
        self._history: list[dict] = []
        self._history_max = 5

        # Sadness-first therapeutic state
        self._sadness_streak = 0
        self._last_support_emit_ts = 0.0

    SADNESS_KEYWORDS = {
        "sad", "down", "lonely", "alone", "hopeless", "empty",
        "cry", "grief", "tired", "exhausted"
    }
    CRISIS_KEYWORDS = {
        "suicide", "kill myself", "end my life", "self harm",
        "self-harm", "want to die", "die"
    }

    def update(self, source: str, vector: EmotionVector) -> None:
        """
        Update a single source's emotion reading.
        v2: applies per-source EMOTION_BIAS offsets before storing.

        Args:
            source: "face" | "posture" | "voice"
            vector: EmotionVector from that detector
        """
        if source not in self.weights:
            logger.warning(f"Unknown source '{source}', ignoring.")
            return

        vector.source = source
        vector.clamp()

        # ── v2 fix 4: apply per-source bias offsets ──────────────────────
        bias = self.emotion_bias.get(source, {})
        if bias:
            for emotion_key, offset in bias.items():
                if emotion_key in EMOTION_KEYS:
                    current = getattr(vector, emotion_key)
                    setattr(vector, emotion_key, current + offset)
            vector.clamp()   # re-clamp after offsets
            logger.debug(f"Bias applied to {source}: {bias}")

        self._readings[source] = vector
        self._last_update[source] = time.time()
        logger.debug(f"Updated {source}: {vector}")

    def _get_active_readings(self) -> dict[str, EmotionVector]:
        """Return readings that are recent enough and confident enough."""
        now = time.time()
        active = {}
        for source, vector in self._readings.items():
            age = now - self._last_update.get(source, 0)
            if age > self.staleness_seconds:
                logger.debug(f"Dropping stale reading from {source} (age={age:.1f}s)")
                continue
            if vector.confidence < self.min_confidence:
                logger.debug(
                    f"Dropping low-confidence reading from {source} "
                    f"(conf={vector.confidence:.2f})"
                )
                continue
            active[source] = vector
        return active

    def fuse(self) -> EmotionVector:
        """
        Compute the fused emotion vector from all active source readings.

        v2 additions (applied after weighted average):
          - Anger↔sadness disambiguation  (fix 5)
          - Joy suppression gate           (fix 6)
          - Sadness focus multiplier       (fix 1, raised)

        Returns:
            EmotionVector with source="fused"
        """
        active = self._get_active_readings()

        if not active:
            logger.warning("No active emotion readings — returning neutral vector.")
            return EmotionVector(source="fused", confidence=0.0)

        # Effective weight = base_weight × confidence
        effective_weights = {
            s: self.weights.get(s, 0.0) * v.confidence
            for s, v in active.items()
        }
        total_weight = sum(effective_weights.values()) or 1e-6

        fused_emotions = {k: 0.0 for k in EMOTION_KEYS}
        for source, vector in active.items():
            norm_w = effective_weights[source] / total_weight
            for key in EMOTION_KEYS:
                fused_emotions[key] += getattr(vector, key) * norm_w

        fused_confidence = sum(
            v.confidence * (effective_weights[s] / total_weight)
            for s, v in active.items()
        )

        fused = EmotionVector(
            joy=fused_emotions["joy"],
            sadness=fused_emotions["sadness"],
            fear=fused_emotions["fear"],
            disgust=fused_emotions["disgust"],
            anger=fused_emotions["anger"],
            surprise=fused_emotions["surprise"],
            confidence=fused_confidence,
            source="fused"
        ).clamp()

        # ── v2 fix 5: anger↔sadness disambiguation ───────────────────────
        # Problem: downturned mouth triggers anger in face_detector.
        # Rule: if posture says sadness dominates AND voice does NOT confirm
        #       anger → transfer 50% of the fused anger score to sadness.
        posture_vector = active.get("posture")
        voice_vector   = active.get("voice")

        posture_sad_dominant = (
            posture_vector is not None
            and posture_vector.sadness >= posture_vector.anger
        )
        voice_confirms_anger = (
            voice_vector is not None
            and voice_vector.anger >= 0.30   # voice clearly angry
        )

        if posture_sad_dominant and not voice_confirms_anger and fused.anger > 0.05:
            transfer = fused.anger * 0.50
            fused.anger   = max(0.0, fused.anger - transfer)
            fused.sadness = min(1.0, fused.sadness + transfer)
            logger.debug(
                f"Anger→Sadness disambiguation: transferred {transfer:.3f} "
                f"(posture_sad={posture_sad_dominant}, voice_anger={voice_confirms_anger})"
            )

        # ── v2 fix 6: joy suppression gate ───────────────────────────────
        # Problem: joy over-triggers even when sadness is clearly dominant.
        # Rule: if sadness is elevated AND joy is low → cap joy dynamically.
        # Slightly higher sadness cutoff reduces how often sadness dominates.
        if fused.sadness > 0.45 and fused.joy < 0.35:
            # Scale joy down proportionally: the higher sadness, the lower joy cap
            joy_cap = max(0.05, 0.35 - (fused.sadness - 0.45) * 0.60)
            if fused.joy > joy_cap:
                logger.debug(
                    f"Joy gate: capping joy {fused.joy:.3f} → {joy_cap:.3f} "
                    f"(sadness={fused.sadness:.3f})"
                )
                fused.joy = joy_cap

        # ── v2 fix 1: raised sadness focus multiplier ─────────────────────
        # Applied last so disambiguation and gate work on raw values.
        fused.sadness = min(1.0, fused.sadness * self.sadness_focus_multiplier)
        fused.clamp()

        # Store in history for smoothing
        self._history.append(fused.to_dict())
        if len(self._history) > self._history_max:
            self._history.pop(0)

        logger.debug(f"Fused emotion: {fused}")
        return fused

    def _extract_keyword_hits(self, transcript: str) -> tuple[list[str], bool]:
        text = (transcript or "").lower()
        hits = [k for k in self.SADNESS_KEYWORDS if k in text]
        crisis = any(k in text for k in self.CRISIS_KEYWORDS)
        return sorted(hits), crisis

    def assess_sadness_state(
        self, fused: EmotionVector, transcript: str = ""
    ) -> SadnessAssessment:
        """
        Evaluate whether the current fused state indicates therapeutic sadness support.
        Uses tuned sadness_threshold and sustained_cycles_required to decide when to support sadness.
        """
        dominant, dominant_value = self.get_dominant_emotion(fused)
        keyword_hits, crisis = self._extract_keyword_hits(transcript)

        sadness_evidence = (
            fused.sadness >= self.sadness_threshold
            or (dominant == "sadness" and dominant_value >= self.sadness_threshold * 0.9)
            or bool(keyword_hits)
        )

        if sadness_evidence:
            self._sadness_streak += 1
        else:
            self._sadness_streak = max(0, self._sadness_streak - 1)

        sustained = self._sadness_streak >= self.sustained_cycles_required
        support_recommended = (
            (fused.sadness >= self.sadness_high_threshold
             and fused.confidence >= self.min_confidence)
            or sustained
            or (bool(keyword_hits) and fused.confidence >= self.min_confidence)
        )

        return SadnessAssessment(
            sadness_score=fused.sadness,
            confidence=fused.confidence,
            dominant_emotion=dominant,
            dominant_value=dominant_value,
            sustained=sustained,
            support_recommended=support_recommended,
            escalation_needed=crisis,
            keyword_hits=keyword_hits,
        )

    def can_emit_support_prompt(self) -> bool:
        """Rate-limit therapeutic prompts so the robot does not spam interventions."""
        return (time.time() - self._last_support_emit_ts) >= self.support_cooldown_seconds

    def mark_support_prompt_emitted(self) -> None:
        self._last_support_emit_ts = time.time()

    def fuse_smoothed(self, alpha: float = 0.55) -> EmotionVector:
        """
        Exponential moving average smoothing over recent fused vectors.
        REDUCED alpha (0.55 instead of 0.65) for faster real-time updates.

        alpha=1.0 means no smoothing (just latest), alpha→0 means very smooth.
        Lower alpha = more responsive to changes, but noisier.
        """
        current = self.fuse()
        if len(self._history) < 2:
            return current

        smoothed = {k: 0.0 for k in EMOTION_KEYS}
        weight = 1.0
        total = 0.0
        for past in reversed(self._history):
            for key in EMOTION_KEYS:
                smoothed[key] += past[key] * weight
            total += weight
            weight *= (1 - alpha)

        for key in EMOTION_KEYS:
            smoothed[key] /= total

        return EmotionVector(
            **smoothed, confidence=current.confidence, source="fused_smoothed"
        ).clamp()


    def get_dominant_emotion(self, vector: EmotionVector) -> tuple[str, float]:
        """Returns the emotion with the highest intensity and its value."""
        emotions = vector.to_dict()
        dominant, value = max(emotions.items(), key=lambda item: item[1])
        return dominant, value

    def build_stimulus(
        self,
        fused: EmotionVector,
        label: str = "user_emotional_state",
        trust: float = 0.7,
        likeness: float = 0.7
    ) -> StimulusObject:
        """Package a fused EmotionVector into a RIO-compatible StimulusObject."""
        return StimulusObject(
            label=label,
            emotions=fused.to_dict(),
            trust=trust,
            likeness=likeness
        )

    def status(self) -> dict:
        """Returns current state of all sources — useful for debugging."""
        now = time.time()
        status = {}
        for source in self.weights:
            if source in self._readings:
                age = now - self._last_update.get(source, now)
                status[source] = {
                    "emotions": self._readings[source].to_dict(),
                    "confidence": self._readings[source].confidence,
                    "age_seconds": round(age, 2),
                    "stale": age > self.staleness_seconds
                }
            else:
                status[source] = {"status": "no reading yet"}
        return status


# ─── Convenience: Mock Detectors for Testing ────────────────────────────────

def mock_face_vector(dominant: str = "sadness", intensity: float = 0.7) -> EmotionVector:
    v = EmotionVector(confidence=0.9, source="face")
    setattr(v, dominant, intensity)
    return v

def mock_posture_vector(dominant: str = "sadness", intensity: float = 0.5) -> EmotionVector:
    v = EmotionVector(confidence=0.75, source="posture")
    setattr(v, dominant, intensity)
    return v

def mock_voice_vector(dominant: str = "sadness", intensity: float = 0.4) -> EmotionVector:
    v = EmotionVector(confidence=0.65, source="voice")
    setattr(v, dominant, intensity)
    return v


# ─── Self Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RIO Emotion Fusion Engine v2 — Self Test")
    print("=" * 60)

    # Use explicit bias so the test runs even without config.py on the path
    TEST_BIAS = {
        "face":    {"joy": -0.15, "sadness": +0.10, "anger": -0.10, "fear": +0.05, "disgust": 0.0, "surprise": 0.0},
        "posture": {"joy": -0.05, "sadness": +0.10, "anger": -0.05, "fear": +0.05, "disgust": 0.0, "surprise": 0.0},
        "voice":   {"joy":  0.00, "sadness": +0.05, "anger":  0.00, "fear":  0.00, "disgust": 0.0, "surprise": 0.0},
    }

    # ── Test 1: downturned mouth → should be SADNESS, not anger ─────────
    print("\n[TEST 1] Downturned mouth scenario — must be dominant=sadness (not anger)")
    engine = EmotionFusionEngine(emotion_bias=TEST_BIAS)
    # Simulate: face picks up anger from frown, posture says sad, voice quiet
    engine.update("face",    EmotionVector(anger=0.55, sadness=0.30, confidence=0.85, source="face"))
    engine.update("posture", EmotionVector(sadness=0.60, anger=0.10, confidence=0.75, source="posture"))
    engine.update("voice",   EmotionVector(sadness=0.40, anger=0.05, confidence=0.60, source="voice"))
    fused = engine.fuse()
    dominant, val = engine.get_dominant_emotion(fused)
    status = "✅ PASS" if dominant == "sadness" else "❌ FAIL"
    print(f"  {status} dominant={dominant} ({val:.3f})")
    print(f"  Full vector: {fused.to_dict()}")

    # ── Test 2: joy suppression gate ─────────────────────────────────────
    print("\n[TEST 2] Joy gate — joy should be suppressed when sadness > 0.40")
    engine2 = EmotionFusionEngine(emotion_bias=TEST_BIAS)
    engine2.update("face",    EmotionVector(joy=0.30, sadness=0.50, confidence=0.85, source="face"))
    engine2.update("posture", EmotionVector(sadness=0.55, confidence=0.70, source="posture"))
    engine2.update("voice",   EmotionVector(sadness=0.45, confidence=0.60, source="voice"))
    fused2 = engine2.fuse()
    dominant2, val2 = engine2.get_dominant_emotion(fused2)
    status2 = "✅ PASS" if dominant2 == "sadness" else "❌ FAIL"
    print(f"  {status2} dominant={dominant2} ({val2:.3f})  joy={fused2.joy:.3f}")

    # ── Test 3: genuine joy should still pass ─────────────────────────────
    print("\n[TEST 3] Genuine joy — should NOT be suppressed")
    engine3 = EmotionFusionEngine(emotion_bias=TEST_BIAS)
    engine3.update("face",    EmotionVector(joy=0.80, confidence=0.90, source="face"))
    engine3.update("posture", EmotionVector(joy=0.65, confidence=0.75, source="posture"))
    engine3.update("voice",   EmotionVector(joy=0.70, confidence=0.80, source="voice"))
    fused3 = engine3.fuse()
    dominant3, val3 = engine3.get_dominant_emotion(fused3)
    status3 = "✅ PASS" if dominant3 == "joy" else "❌ FAIL"
    print(f"  {status3} dominant={dominant3} ({val3:.3f})")

    # ── Test 4: sadness streak → sustain detection ────────────────────────
    print("\n[TEST 4] Sadness streak — sustained=True after 2 cycles")
    engine4 = EmotionFusionEngine(emotion_bias=TEST_BIAS)
    for i in range(2):
        engine4.update("face", EmotionVector(sadness=0.65, confidence=0.85, source="face"))
        fused4 = engine4.fuse()
        assessment = engine4.assess_sadness_state(fused4)
    status4 = "✅ PASS" if assessment.sustained else "❌ FAIL"
    print(f"  {status4} sustained={assessment.sustained} "
          f"support_recommended={assessment.support_recommended}")

    print("\n" + "=" * 60)
    print("v2 self-test complete.")
    print("=" * 60)

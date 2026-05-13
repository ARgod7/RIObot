"""
Pipeline Manager — Orchestrates dialogue, servo, and feedback agents sequentially.

Flow:
  1. DIALOGUE AGENT: Generates response & emotion/expression intent based on stimulus
  2. SERVO AGENT: Applies emotion-specific servo poses from servo_controls/poses.json
                  Uses smooth transitions and alive animation while TTS speaks
  3. FEEDBACK AGENT: Assesses intervention effectiveness

v3 changes vs v2:
  - Emotion mirroring now covers ALL 6 Ekman emotions, not just the NEGATIVE_EMOTIONS set.
    Previously only sadness/fear/anger/disgust were mirrored — joy and surprise were ignored.
  - Mirror threshold lowered 0.20 → 0.15 so moderate emotions are reflected.
  - Sadness keyword override removed: it was forcing expression_intent="sadness" based on
    transcript words even when the fused vector showed a different dominant emotion (e.g.
    anger or fear). The fused vector is the authoritative signal.
  - TTS baselines corrected: anger was 1.36 speed (way too fast); now 0.88 (measured, firm).
    Joy pitch corrected to 1.10–1.20 range (was 1.15/1.10 but dialogue_agent clamp is 1.32).
  - merge_tts_for_therapist blend ratio adjusted 55/45 → 45/55 (LLM hint weighted slightly
    more so dialogue_agent's nuanced prosody choices are better preserved).
"""

import logging
from typing import Dict, Any

from cognition.agents.dialogue_agent import run_dialogue
from cognition.agents.servo_agent import run_servo, resolve_emotion
from cognition.agents.feedback_agent import run_feedback

logger = logging.getLogger(__name__)

# All 6 Ekman emotions that should be mirrored before uplifting
ALL_EMOTIONS = {"joy", "sadness", "fear", "anger", "disgust", "surprise"}

# Minimum dominant value to trigger emotion mirroring on RIO's face/servo
MIRROR_THRESHOLD = 0.15


def _dominant_from_stimulus(stimulus: Dict[str, Any]) -> tuple[str, float]:
    emotions = stimulus.get("emotions") or {}
    if not emotions:
        return ("neutral", 0.0)
    dom, val = max(emotions.items(), key=lambda item: item[1])
    return (str(dom), float(val))


def merge_tts_for_therapist(expression_intent: str, tts: Dict[str, Any]) -> Dict[str, float]:
    """
    Blend LLM TTS hints with therapist-style baselines per facial expression.

    Baselines:
      sadness:  soft, slow — lower pitch, reduced speed
      anger:    measured, firm — NOT loud or fast (therapist mirrors calmly)
      fear:     steady, reassuring
      joy:      warm, bright — moderate uplift
      surprise: engaged, curious
      calm:     steady, warm
      disgust:  neutral-to-calm (mirror then redirect)

    Blend: 45% baseline + 55% LLM hint (v3: LLM weighted slightly more).
    """
    ex = (expression_intent or "calm").lower()
    baselines = {
        "sadness": {"pitch": 0.86, "speed": 0.84},
        "anger":   {"pitch": 0.92, "speed": 0.88},   # FIX: was speed=1.36 (far too fast)
        "fear":    {"pitch": 0.93, "speed": 0.88},
        "joy":     {"pitch": 1.15, "speed": 1.06},
        "surprise":{"pitch": 1.08, "speed": 1.04},
        "disgust": {"pitch": 0.95, "speed": 0.90},
        "calm":    {"pitch": 0.98, "speed": 0.90},
    }
    base = baselines.get(ex, baselines["calm"])
    lp = float(tts.get("pitch", base["pitch"]))
    ls = float(tts.get("speed", base["speed"]))

    # v3: 45% baseline + 55% LLM (was 55/45)
    p = 0.45 * base["pitch"] + 0.55 * lp
    s = 0.45 * base["speed"] + 0.55 * ls

    # Clamp to safe hardware range
    p = max(0.78, min(1.32, p))
    s = max(0.82, min(1.12, s))
    return {"pitch": p, "speed": s}


def _emotion_intensity_index(emotion: str, stimulus: Dict[str, Any]) -> int:
    resolved = resolve_emotion(emotion)
    value = float((stimulus.get("emotions") or {}).get(resolved, 0.0))
    return max(0, min(4, int(round(value * 4))))


def run_pipeline(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str,
    emotion_before: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run the full dialogue → servo → feedback pipeline sequentially.

    Args:
        stimulus: StimulusObject.to_dict() (emotion vectors, metadata).
        intervention_intent: e.g., "deflect_sadness", "reinforce_joy".
        user_transcript: What the user just said.
        memory_context: Summary of recent interactions (plain text).
        emotion_before: Emotion vector before intervention.

    Returns:
        Dict with response_text, expression_intent, tts_params, and feedback.
    """
    dialogue_output = None
    servo_output = None
    feedback_output = None

    # ── Step 1: Run dialogue agent ───────────────────────────────────────────
    try:
        dialogue_output = run_dialogue(
            stimulus=stimulus,
            intervention_intent=intervention_intent,
            user_transcript=user_transcript,
            memory_context=memory_context,
        )
        logger.info(f"Dialogue: {dialogue_output['expression_intent']}")
    except Exception as e:
        logger.error(f"Dialogue agent failed: {e}", exc_info=True)
        if not (memory_context or "").strip():
            dialogue_output = {
                "response_text": "Hello! I'm RIO. Welcome back. How has your day been so far?",
                "expression_intent": "calm",
                "tts_params": {"pitch": 1.0, "speed": 0.98},
            }
        else:
            dialogue_output = {
                "response_text": "I'm here for you. What would feel a little easier right now?",
                "expression_intent": "calm",
                "tts_params": {"pitch": 0.98, "speed": 0.90},
            }

    # ── Step 2: Emotion mirroring ────────────────────────────────────────────
    # Mirror the user's dominant emotion on RIO's face/servo BEFORE the verbal
    # uplift response. This shows empathy — "I see how you feel" — before
    # attempting to shift mood.
    #
    # v3 fix: covers ALL 6 emotions (was only NEGATIVE_EMOTIONS set in v2).
    # v3 fix: removed transcript keyword override — fused vector is authoritative.
    dominant_user, dominant_value = _dominant_from_stimulus(stimulus)
    if dominant_value >= MIRROR_THRESHOLD:
        # Mirror whatever the user is feeling, including joy and surprise
        dialogue_output["expression_intent"] = dominant_user
        logger.info(
            f"Mirroring user emotion: {dominant_user} ({dominant_value:.2f}) "
            f"→ expression_intent overridden"
        )
    # If dominant_value < threshold, trust the dialogue agent's chosen expression_intent
    # (it may choose a slightly positive expression as a gentle lead)

    # Apply TTS prosody blend
    dialogue_output["tts_params"] = merge_tts_for_therapist(
        dialogue_output.get("expression_intent", "calm"),
        dialogue_output.get("tts_params") or {},
    )

    # ── Step 3: Run servo agent ──────────────────────────────────────────────
    try:
        servo_output = run_servo(
            emotion=dialogue_output["expression_intent"],
            speaking=True,
            transition=True,
            intensity=_emotion_intensity_index(dialogue_output["expression_intent"], stimulus),
        )
        logger.info(f"Servo: {servo_output['action']} → {servo_output['resolved_emotion']}")
    except Exception as e:
        logger.error(f"Servo agent failed: {e}", exc_info=True)
        servo_output = {
            "action": "error",
            "emotion": dialogue_output.get("expression_intent", "unknown"),
        }

    # ── Step 4: Run feedback agent ───────────────────────────────────────────
    try:
        emotion_after = stimulus.get("emotions", {})
        feedback_output = run_feedback(emotion_before, emotion_after)
        logger.info(f"Feedback: {feedback_output['note']}")
    except Exception as e:
        logger.error(f"Feedback agent failed: {e}", exc_info=True)
        feedback_output = {"improved": False, "delta": 0.0, "note": "error"}

    # ── Step 5: Return combined output ──────────────────────────────────────
    return {
        "response_text":    dialogue_output["response_text"],
        "expression_intent": dialogue_output["expression_intent"],
        "tts_params":       dialogue_output["tts_params"],
        "feedback":         feedback_output,
    }

"""
pipeline_manager.py (v4)

The core fix: this file now writes structured markers into the pipeline result
so main.py can append them to memory_context before the next call.

Without these markers every [activity:], [promise:], [rio_said:] tracking
block in dialogue_agent.py is completely dead — the model has no idea what
it already did, what it promised, or what it just said.

Markers written per turn:
  [activity:tag]       — which activity was used (for rotation)
  [promise:description]— what RIO offered but hasn't delivered (for commitment)
  [rio_said:summary]   — 8-word summary of RIO's response (for loop detection)

main.py is responsible for appending these to memory_context each turn.
See main.py — the `_build_enriched_memory` function handles this.

Other fixes vs v2/v3:
  - Emotion mirroring covers all 6 Ekman emotions (not just NEGATIVE_EMOTIONS)
  - SADNESS_KEYWORDS transcript override removed (fused vector is authoritative)
  - Anger TTS speed corrected: was 1.36 (frantic), now 0.88 (measured/firm)
  - merge_tts blend 55/45 → 45/55 (LLM hint weighted more, preserves nuance)
"""

import logging
from typing import Dict, Any

from cognition.agents.dialogue_agent import run_dialogue
from cognition.agents.servo_agent import run_servo, resolve_emotion
from cognition.agents.feedback_agent import run_feedback

logger = logging.getLogger(__name__)

MIRROR_THRESHOLD = 0.15  # minimum dominant value to mirror on RIO's face


def _dominant_from_stimulus(stimulus: Dict[str, Any]):
    emotions = stimulus.get("emotions") or {}
    if not emotions:
        return "neutral", 0.0
    dom, val = max(emotions.items(), key=lambda x: x[1])
    return str(dom), float(val)


def merge_tts_for_therapist(expression_intent: str, tts: Dict[str, Any]) -> Dict[str, float]:
    """
    Blend dialogue agent's TTS hint with therapist-style baseline.
    45% baseline + 55% LLM hint — preserves model's nuanced prosody.
    """
    ex = (expression_intent or "calm").lower()
    baselines = {
        "sadness":  {"pitch": 0.86, "speed": 0.84},
        "anger":    {"pitch": 0.92, "speed": 0.88},   # was 1.36 — fixed
        "fear":     {"pitch": 0.93, "speed": 0.88},
        "joy":      {"pitch": 1.15, "speed": 1.06},
        "surprise": {"pitch": 1.08, "speed": 1.04},
        "disgust":  {"pitch": 0.95, "speed": 0.90},
        "calm":     {"pitch": 0.98, "speed": 0.90},
    }
    base = baselines.get(ex, baselines["calm"])
    lp = float(tts.get("pitch", base["pitch"]))
    ls = float(tts.get("speed", base["speed"]))
    p = max(0.78, min(1.32, 0.45 * base["pitch"] + 0.55 * lp))
    s = max(0.82, min(1.12, 0.45 * base["speed"] + 0.55 * ls))
    return {"pitch": p, "speed": s}


def _emotion_intensity_index(emotion: str, stimulus: Dict[str, Any]) -> int:
    resolved = resolve_emotion(emotion)
    value = float((stimulus.get("emotions") or {}).get(resolved, 0.0))
    return max(0, min(4, int(round(value * 4))))


def _short_summary(text: str, max_words: int = 8) -> str:
    """Produce a short summary of RIO's response for the rio_said marker."""
    words = (text or "").split()
    summary = " ".join(words[:max_words])
    return summary + ("..." if len(words) > max_words else "")


def run_pipeline(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str,
    emotion_before: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run dialogue → servo → feedback sequentially.

    Returns:
        response_text       : str
        expression_intent   : str
        tts_params          : dict
        feedback            : dict
        activity_used       : str   ← NEW: main.py appends [activity:X] to memory
        pending_promise     : str|None ← NEW: main.py appends [promise:X] if set
        rio_said_marker     : str   ← NEW: main.py appends [rio_said:X] to memory
    """
    # ── Step 1: Dialogue agent ────────────────────────────────────────────────
    try:
        dialogue_output = run_dialogue(
            stimulus=stimulus,
            intervention_intent=intervention_intent,
            user_transcript=user_transcript,
            memory_context=memory_context,
        )
        logger.info(f"Dialogue: {dialogue_output['expression_intent']} | activity={dialogue_output.get('activity_used')}")
    except Exception as e:
        logger.error(f"Dialogue agent failed: {e}", exc_info=True)
        is_first = not (memory_context or "").strip()
        dialogue_output = {
            "response_text":    "Hello! I'm RIO. How has your day been?" if is_first
                                else "I'm here with you. What's on your mind?",
            "expression_intent":"calm",
            "activity_used":    "curiosity_question",
            "pending_promise":  None,
            "tts_params":       {"pitch": 1.10, "speed": 1.00},
        }

    # ── Step 2: Emotion mirroring (all 6 Ekman) ───────────────────────────────
    # Mirror whatever the user is feeling on RIO's face before the verbal uplift.
    # Covers all emotions — including joy and surprise, not just negative ones.
    dominant_user, dominant_value = _dominant_from_stimulus(stimulus)
    if dominant_value >= MIRROR_THRESHOLD:
        dialogue_output["expression_intent"] = dominant_user

    dialogue_output["tts_params"] = merge_tts_for_therapist(
        dialogue_output.get("expression_intent", "calm"),
        dialogue_output.get("tts_params") or {},
    )

    # ── Step 3: Servo agent ───────────────────────────────────────────────────
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
        servo_output = {"action": "error", "emotion": dialogue_output.get("expression_intent", "unknown")}

    # ── Step 4: Feedback agent ────────────────────────────────────────────────
    try:
        emotion_after  = stimulus.get("emotions", {})
        feedback_output = run_feedback(emotion_before, emotion_after)
        logger.info(f"Feedback: {feedback_output['note']}")
    except Exception as e:
        logger.error(f"Feedback agent failed: {e}", exc_info=True)
        feedback_output = {"improved": False, "delta": 0.0, "note": "error"}

    # ── Step 5: Build markers for main.py to write into next memory_context ───
    response_text  = dialogue_output["response_text"]
    activity_used  = dialogue_output.get("activity_used", "unknown")
    pending_promise = dialogue_output.get("pending_promise")
    rio_said_marker = _short_summary(response_text)

    return {
        "response_text":     response_text,
        "expression_intent": dialogue_output["expression_intent"],
        "tts_params":        dialogue_output["tts_params"],
        "feedback":          feedback_output,
        # ── Tracking markers (main.py appends these to memory_context) ────────
        "activity_used":     activity_used,
        "pending_promise":   pending_promise,
        "rio_said_marker":   rio_said_marker,
    }

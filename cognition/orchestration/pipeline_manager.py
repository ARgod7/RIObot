"""
Pipeline Manager — Orchestrates dialogue, servo, and feedback agents sequentially.

Flow:
  1. DIALOGUE AGENT: Generates response & emotion/expression intent based on stimulus
  2. SERVO AGENT: Applies emotion-specific servo poses from servo_controls/poses.json
                  Uses smooth transitions and alive animation while TTS speaks
  3. FEEDBACK AGENT: Assesses intervention effectiveness

The servo positions are dynamically applied based on the expression_intent (emotion)
chosen by the dialogue agent, creating embodied emotional responses synchronized
with the robot's speech.
"""

import logging
from typing import Dict, Any

from cognition.agents.dialogue_agent import run_dialogue
from cognition.agents.servo_agent import run_servo
from cognition.agents.feedback_agent import run_feedback

logger = logging.getLogger(__name__)


def merge_tts_for_therapist(expression_intent: str, tts: Dict[str, Any]) -> Dict[str, float]:
    """
    Blend LLM TTS hints with therapist-style baselines per facial expression.
    Sad: softer, slower. Anger: firm and measured (not loud/fast). Calm: steady, slightly slow.
    """
    ex = (expression_intent or "calm").lower()
    baselines = {
        "sadness": {"pitch": 0.86, "speed": 0.84},
        "anger": {"pitch": 0.90, "speed": 0.86},
        "fear": {"pitch": 0.93, "speed": 0.88},
        "joy": {"pitch": 1.05, "speed": 0.98},
        "surprise": {"pitch": 1.04, "speed": 0.95},
        "calm": {"pitch": 0.98, "speed": 0.90},
    }
    base = baselines.get(ex, baselines["calm"])
    lp = float(tts.get("pitch", base["pitch"]))
    ls = float(tts.get("speed", base["speed"]))
    p = 0.55 * base["pitch"] + 0.45 * lp
    s = 0.55 * base["speed"] + 0.45 * ls
    p = max(0.78, min(1.32, p))
    s = max(0.82, min(1.12, s))
    return {"pitch": p, "speed": s}


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

    # Step 1: Run dialogue agent
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
        dialogue_output = {
            "response_text": "I'm here for you.",
            "expression_intent": "calm",
            "tts_params": {"pitch": 1.0, "speed": 0.95},
        }

    dialogue_output["tts_params"] = merge_tts_for_therapist(
        dialogue_output.get("expression_intent", "calm"),
        dialogue_output.get("tts_params") or {},
    )

    # Step 2: Run servo agent with emotion-based positions
    try:
        # expression_intent is the emotion from the dialogue agent
        servo_output = run_servo(
            emotion=dialogue_output["expression_intent"],
            speaking=True,  # Robot is about to speak, so use alive animation
            transition=True  # Smooth transition between poses
        )
        logger.info(f"Servo: {servo_output['action']} → {servo_output['resolved_emotion']}")
    except Exception as e:
        logger.error(f"Servo agent failed: {e}", exc_info=True)
        servo_output = {"action": "error", "emotion": dialogue_output.get("expression_intent", "unknown")}

    # Step 3: Run feedback agent
    try:
        emotion_after = stimulus.get("emotions", {})
        feedback_output = run_feedback(emotion_before, emotion_after)
        logger.info(f"Feedback: {feedback_output['note']}")
    except Exception as e:
        logger.error(f"Feedback agent failed: {e}", exc_info=True)
        feedback_output = {"improved": False, "delta": 0.0, "note": "error"}

    # Step 4: Return final combined output
    return {
        "response_text": dialogue_output["response_text"],
        "expression_intent": dialogue_output["expression_intent"],
        "tts_params": dialogue_output["tts_params"],
        "feedback": feedback_output,
    }


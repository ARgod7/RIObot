"""
Pipeline Manager — Orchestrates dialogue, servo, and feedback agents sequentially.
"""

import logging
from typing import Dict, Any

from cognition.agents.dialogue_agent import run_dialogue
from cognition.agents.servo_agent import run_servo
from cognition.agents.feedback_agent import run_feedback

logger = logging.getLogger(__name__)


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

    # Step 2: Run servo agent (stub)
    try:
        servo_output = run_servo(
            dialogue_output["expression_intent"],
            stimulus.get("emotions", {}),
        )
        logger.info(f"Servo: {servo_output}")
    except Exception as e:
        logger.error(f"Servo agent failed: {e}", exc_info=True)
        servo_output = {"action": "stub", "intent": dialogue_output["expression_intent"]}

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


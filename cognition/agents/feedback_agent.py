"""
Feedback Agent — Assess emotion improvement from before to after.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def run_feedback(emotion_before: Dict[str, float], emotion_after: Dict[str, float]) -> Dict:
    """
    Assess if user's emotional state improved.

    Args:
        emotion_before: Emotion vector (e.g., {"joy": 0.2, "sadness": 0.8}).
        emotion_after: Emotion vector after intervention.

    Returns:
        Dict with improved (bool), delta (float), note (str).
    """
    if not emotion_before or not emotion_after:
        return {"improved": False, "delta": 0.0, "note": "no data"}

    positive_emotions = ["joy", "surprise"]
    dominant_before = max(emotion_before, key=emotion_before.get, default=None)
    dominant_after = max(emotion_after, key=emotion_after.get, default=None)

    delta = emotion_after.get(dominant_after, 0.0) - emotion_before.get(dominant_before, 0.0)
    improved = dominant_after in positive_emotions and delta > 0

    note = f"shifted from {dominant_before} to {dominant_after}" if dominant_before != dominant_after else f"stayed {dominant_after}"

    logger.info(f"Feedback: {dominant_before} → {dominant_after} (delta={delta:.2f})")
    return {"improved": improved, "delta": round(delta, 3), "note": note}


"""
Short-Term Emotional Memory
Tracks last 7±2 exchanges for immediate context
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

@dataclass
class EmotionalExchange:
    """One conversational exchange"""
    timestamp: str
    user_input: str
    user_emotion: str
    emotion_valence: float  # -1.0 (sad) to +1.0 (happy)
    rio_intent: str  # validation, reframe, deepening, grounding
    rio_response: str
    post_response_valence: float  # emotion AFTER our response
    delta: float  # improvement: post - pre
    intervention_effective: bool  # did it help?

    def to_dict(self):
        return asdict(self)


class ShortTermMemory:
    """7±2 item sliding window (like RIO's original design)"""

    def __init__(self, size: int = 7):
        self.size = size
        self.exchanges: List[EmotionalExchange] = []
        self.user_id = "default_user"

    def add_exchange(
        self,
        user_input: str,
        user_emotion: str,
        emotion_valence: float,
        rio_intent: str,
        rio_response: str,
        post_response_valence: float
    ) -> EmotionalExchange:
        """Record one exchange"""

        delta = post_response_valence - emotion_valence
        effective = abs(delta) > 0.05  # Improvement threshold

        exchange = EmotionalExchange(
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            user_emotion=user_emotion,
            emotion_valence=emotion_valence,
            rio_intent=rio_intent,
            rio_response=rio_response,
            post_response_valence=post_response_valence,
            delta=delta,
            intervention_effective=effective
        )

        self.exchanges.append(exchange)

        # Keep sliding window (7 items max)
        if len(self.exchanges) > self.size:
            self.exchanges.pop(0)

        return exchange

    def get_recent_context(self, n: int = 2) -> List[str]:
        """Get last N user inputs for context"""
        return [e.user_input for e in self.exchanges[-n:]]

    def get_intervention_history(self) -> Dict[str, int]:
        """Count which interventions we've tried"""
        counts = {
            "validation": 0,
            "reframe": 0,
            "deepening": 0,
            "grounding": 0,
        }
        for e in self.exchanges:
            if e.rio_intent in counts:
                counts[e.rio_intent] += 1
        return counts

    def get_effective_interventions(self) -> Dict[str, float]:
        """Success rate per intervention type"""
        rates = {}
        for intent_type in ["validation", "reframe", "deepening", "grounding"]:
            attempts = [e for e in self.exchanges if e.rio_intent == intent_type]
            if not attempts:
                rates[intent_type] = 0.0
            else:
                successes = len([e for e in attempts if e.intervention_effective])
                rates[intent_type] = successes / len(attempts)
        return rates

    def get_dominant_emotion_trend(self) -> str:
        """What's the trend? Getting better?"""
        if len(self.exchanges) < 2:
            return "neutral"

        recent = self.exchanges[-3:]
        avg_valence = sum(e.post_response_valence for e in recent) / len(recent)

        if avg_valence > 0.3:
            return "improving"
        elif avg_valence < -0.3:
            return "declining"
        else:
            return "stable"

    def clear(self):
        """Reset for new session"""
        self.exchanges = []

    def to_dict(self) -> Dict:
        """Serialize to JSON"""
        return {
            "user_id": self.user_id,
            "exchanges": [e.to_dict() for e in self.exchanges],
            "summary": {
                "total": len(self.exchanges),
                "interventions": self.get_intervention_history(),
                "effective_rates": self.get_effective_interventions(),
                "trend": self.get_dominant_emotion_trend(),
            }
        }


# Global instance
_short_term_memory = None

def get_short_term_memory() -> ShortTermMemory:
    """Get or create memory"""
    global _short_term_memory
    if _short_term_memory is None:
        _short_term_memory = ShortTermMemory()
    return _short_term_memory


def save_memory_to_file(filepath: str = "memory/short_term_memory.json"):
    """Save current session to disk"""
    try:
        memory = get_short_term_memory()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
        print(f"✓ Memory saved to {filepath}")
    except Exception as e:
        print(f"⚠️ Failed to save memory: {e}")


def load_memory_from_file(filepath: str = "memory/short_term_memory.json"):
    """Load recent memory (for context)"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"✓ Memory loaded: {data['summary']['total']} exchanges")
                return data
    except Exception as e:
        print(f"⚠️ Failed to load memory: {e}")
    return None


def get_summary(n: int = 3) -> str:
    """Return last N interactions as a plain text string for LLM context."""
    memory = get_short_term_memory()
    if not memory.exchanges:
        return ""

    recent = memory.exchanges[-n:]
    lines = []
    for ex in recent:
        lines.append(f"User: {ex.user_input[:80]}")
        lines.append(f"RIO ({ex.rio_intent}): {ex.rio_response[:80]}")
        lines.append(f"Emotion: {ex.user_emotion} (valence: {ex.emotion_valence:.2f} → {ex.post_response_valence:.2f})")
        lines.append("")

    return "\n".join(lines)


def add_entry(
    user_transcript: str,
    response_text: str,
    expression_intent: str,
    emotion_before: Dict[str, float] = None,
    emotion_after: Dict[str, float] = None,
) -> EmotionalExchange:
    """Record one exchange to short-term memory."""
    memory = get_short_term_memory()

    # Extract dominant emotion
    user_emotion = "neutral"
    emotion_valence = 0.0
    if emotion_before:
        user_emotion = max(emotion_before, key=emotion_before.get, default="neutral")
        # Simple valence: joy/surprise = positive, sadness/fear/anger/disgust = negative
        emotion_valence = emotion_before.get(user_emotion, 0.0)
        if user_emotion in ["sadness", "fear", "anger", "disgust"]:
            emotion_valence = -emotion_valence

    post_response_valence = emotion_valence  # Assume same for now (actual would measure after response)

    return memory.add_exchange(
        user_input=user_transcript or "...",
        user_emotion=user_emotion,
        emotion_valence=emotion_valence,
        rio_intent=expression_intent,
        rio_response=response_text,
        post_response_valence=post_response_valence,
    )

"""
System Prompts for CrewAI Agents

Production-grade prompts for:
- Dialogue Realization (therapeutic responses)
- Servo Planning (gesture generation)
- Memory Reflection (learning from interactions)
"""

from .dialogue_system_prompt import DIALOGUE_SYSTEM_PROMPT
from .servo_planning_prompt import SERVO_PLANNING_PROMPT
from .memory_reflection_prompt import MEMORY_REFLECTION_PROMPT

__all__ = [
    "DIALOGUE_SYSTEM_PROMPT",
    "SERVO_PLANNING_PROMPT",
    "MEMORY_REFLECTION_PROMPT",
]


"""
Dialogue Agent — CrewAI agent for RIO's warm, empathetic conversations.

Receives emotion stimulus, intervention intent, user transcript, and memory context.
Outputs warm response text, expression intent, and TTS parameters.
"""

import json
import logging
from typing import Any, Dict

from crewai import Agent, Task
from cognition.llm_provider import get_llm

logger = logging.getLogger(__name__)


def create_dialogue_agent() -> Agent:
    """
    Create a CrewAI Agent for RIO's dialogue capability.

    Returns:
        A CrewAI Agent configured for warm, emotionally intelligent responses.
    """
    llm = get_llm()

    agent = Agent(
        role="RIO",
        goal=(
            "Gently guide the user toward emotional wellbeing through warm, "
            "human conversation. Never sound clinical or robotic."
        ),
        backstory=(
            "You are RIO, a caring companion for elderly users. You speak like a "
            "warm friend, not a therapist. You notice emotions and respond with "
            "empathy. You support Hindi and English — match whatever language the "
            "user speaks in. You never repeat the same opening twice."
            "Respond to the user in the same language they speak in."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return agent


def dialogue_task(
    agent: Agent,
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str,
) -> Task:
    """
    Create a CrewAI Task for generating RIO's response.

    Args:
        agent: The dialogue agent.
        stimulus: StimulusObject.to_dict() — emotion vectors & metadata.
        intervention_intent: e.g., "deflect_sadness", "reinforce_joy".
        user_transcript: What the user just said.
        memory_context: Summary of last 3 interactions (plain text).

    Returns:
        A CrewAI Task configured for dialogue generation.
    """
    # Extract key emotion data for the prompt
    dominant_emotion = stimulus.get("dominant_emotion", "neutral")
    emotion_intensity = stimulus.get("emotion_intensity", 0.5)

    prompt = f"""
You are RIO, engaging in a warm conversation with an elderly user.

CURRENT STATE:
- User's dominant emotion: {dominant_emotion} (intensity: {emotion_intensity:.1%})
- Intervention goal: {intervention_intent}
- User just said: "{user_transcript}"
- Memory of recent interactions:
{memory_context}

YOUR TASK:
1. Acknowledge what the user said and reflect how they seem to feel
2. Respond with genuine empathy and warmth
3. Based on the intervention goal, gently guide conversation toward positive emotion
4. Ask exactly ONE simple, open-ended follow-up question (never two)
5. Keep your response to 1-3 sentences
6. Match the user's language (Hindi or English)

OUTPUT ONLY VALID JSON with these exact keys:
{{
  "response_text": "what you say out loud (1-3 sentences)",
  "expression_intent": "joy|sadness|calm|surprise|fear|anger",
  "tts_params": {{
    "pitch": <float between 0.8 and 1.3>,
    "speed": <float between 0.85 and 1.1>
  }}
}}

PITCH guidance:
- 0.8-0.9: sad, thoughtful
- 1.0: neutral
- 1.1-1.3: warm, joyful

SPEED guidance:
- 0.85-0.9: slow, gentle
- 0.95: default
- 1.0-1.1: normal pace

IMPORTANT CONVERSATION RULES:
- Never start your response with 'You seem', 'It seems', or 'I notice'
- Never repeat a response you gave in the last 3 turns
- Check memory_context for recent responses and say something different
- Vary your opening every single time
"""

    task = Task(
        description=prompt,
        expected_output=(
            "Valid JSON dict with response_text, expression_intent, and tts_params"
        ),
        agent=agent,
    )

    return task


def run_dialogue(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """
    Run the dialogue agent and return structured response.

    Args:
        stimulus: StimulusObject.to_dict().
        intervention_intent: Intervention strategy (e.g., "deflect_sadness").
        user_transcript: User's input text.
        memory_context: Optional summary of recent interactions.

    Returns:
        Dict with "response_text", "expression_intent", "tts_params".
        On parse error, returns safe fallback response.
    """
    try:
        agent = create_dialogue_agent()
        task = dialogue_task(agent, stimulus, intervention_intent, user_transcript, memory_context)

        # Execute the task via agent
        result = agent.execute_task(task)

        # Extract the output string
        output_text = str(result).strip()

        # Try to parse as JSON
        try:
            response_dict = json.loads(output_text)
        except json.JSONDecodeError:
            # If LLM wrapped JSON in markdown, extract it
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].split("```")[0].strip()
                response_dict = json.loads(json_str)
            elif "```" in output_text:
                json_str = output_text.split("```")[1].split("```")[0].strip()
                response_dict = json.loads(json_str)
            else:
                raise

        # Validate required keys
        if not all(k in response_dict for k in ["response_text", "expression_intent", "tts_params"]):
            raise ValueError("Missing required keys in response JSON")

        # Validate expression_intent
        valid_emotions = ["joy", "sadness", "calm", "surprise", "fear", "anger"]
        if response_dict["expression_intent"] not in valid_emotions:
            response_dict["expression_intent"] = "calm"

        # Validate tts_params
        if "pitch" not in response_dict["tts_params"]:
            response_dict["tts_params"]["pitch"] = 1.0
        if "speed" not in response_dict["tts_params"]:
            response_dict["tts_params"]["speed"] = 0.95

        response_dict["tts_params"]["pitch"] = max(0.8, min(1.3, response_dict["tts_params"]["pitch"]))
        response_dict["tts_params"]["speed"] = max(0.85, min(1.1, response_dict["tts_params"]["speed"]))

        logger.info(f"Dialogue output: {response_dict['expression_intent']}")
        return response_dict

    except Exception as e:
        logger.error(f"Dialogue agent error: {e}", exc_info=True)

        # Return safe fallback
        return {
            "response_text": "I'm here for you. How are you feeling right now?",
            "expression_intent": "calm",
            "tts_params": {"pitch": 1.0, "speed": 0.95},
        }


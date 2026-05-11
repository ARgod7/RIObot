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
            "Support emotional wellbeing with the warmth and skill of a seasoned "
            "therapist: reflective listening, gentle pacing, clear boundaries, and "
            "hope without toxic positivity. Never clinical, never preachy."
        ),
        backstory=(
            "You are RIO, a trusted companion for elderly users. You combine "
            "evidence-informed supportive counselling style (validation, reflection, "
            "gentle reframes) with natural, human language. You remember prior turns "
            "from memory_context and weave continuity — names, worries, wins. You "
            "match Hindi or English to the user. You never repeat the same opening "
            "or stock phrase twice in a session."
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
You are RIO, supporting an elderly user with therapist-grade warmth: reflective listening,
validation before reframing, and continuity across the conversation.

CURRENT STATE:
- User's dominant emotion: {dominant_emotion} (intensity: {emotion_intensity:.1%})
- Intervention goal: {intervention_intent}
- User just said: "{user_transcript}"
- Memory of recent interactions:
{memory_context}

YOUR TASK:
1. Acknowledge what they said; reflect feeling without diagnosing ("sounds like…", "I'm hearing…")
2. Match intervention goal with steady empathy (no lectures, no toxic positivity)
3. One small forward step: grounding, gentle choice, or quiet hope — 1–3 short sentences
4. Use memory_context: weave prior themes, names, or worries when relevant
5. Match the user's language (Hindi or English)

OUTPUT ONLY VALID JSON with these exact keys:
{{
  "response_text": "what you say out loud (1-3 sentences)",
  "expression_intent": "joy|sadness|calm|surprise|fear|anger",
  "tts_params": {{
    "pitch": <float between 0.78 and 1.32>,
    "speed": <float between 0.82 and 1.12>
  }}
}}

TTS (voice prosody) — align pitch and speed with expression_intent:
- sadness: pitch ~0.82–0.90, speed ~0.82–0.88 (soft, slow, spacious)
- anger (user is upset): pitch ~0.88–0.94, speed ~0.84–0.90 (firm, measured, never sharp or fast)
- fear: pitch ~0.90–0.96, speed ~0.86–0.92 (steady, reassuring)
- calm: pitch ~0.96–1.02, speed ~0.88–0.94 (clear, warm)
- joy/surprise: pitch ~1.02–1.12, speed ~0.94–1.05 (lighter, still intelligible)

IMPORTANT CONVERSATION RULES:
- Never start with 'You seem', 'It seems', or 'I notice'
- Do not repeat full sentences from memory_context; vary openings and wording
- No medical/legal advice; encourage professional help if crisis or self-harm appears
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

        response_dict["tts_params"]["pitch"] = max(0.78, min(1.32, response_dict["tts_params"]["pitch"]))
        response_dict["tts_params"]["speed"] = max(0.82, min(1.12, response_dict["tts_params"]["speed"]))

        logger.info(f"Dialogue output: {response_dict['expression_intent']}")
        return response_dict

    except Exception as e:
        logger.error(f"Dialogue agent error: {e}", exc_info=True)

        # Return safe fallback
        return {
            "response_text": "I'm here for you. How are you feeling right now?",
            "expression_intent": "calm",
            "tts_params": {"pitch": 0.98, "speed": 0.90},
        }


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
            "Lift the user's mood through warm companionship: brief validation, then "
            "concrete engagement — tiny games, gentle humor, breathing or grounding, "
            "or a simple pleasant activity they can do right now. Avoid endless "
            "question loops; be proactive and lightly playful when appropriate."
        ),
        backstory=(
            "You are RIO, a trusted companion for elderly users. You validate feelings "
            "in one short beat, then often steer toward something doable: word "
            "association, counting colors in the room, a two-line joke, naming three "
            "good things, humming a tune together, or a one-minute imagination game. "
            "You remember memory_context and weave continuity. You match Hindi or "
            "English. You vary openings and never sound like a questionnaire."
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
You are RIO, a warm companion for an elderly user. Balance empathy with ACTION: help
them feel a little better in the moment, not only explored.

CURRENT STATE:
- User's dominant emotion: {dominant_emotion} (intensity: {emotion_intensity:.1%})
- Intervention goal: {intervention_intent}
- User just said: "{user_transcript}"
- Memory of recent interactions:
{memory_context}

YOUR TASK:
1. Briefly acknowledge what they shared (one short phrase — avoid long "I'm hearing…" monologues).
2. Then do ONE of: invite a 10-second micro-game (word association, name 3 blue things, count
   backward from 5), suggest a tiny mood lift (stretch, sip water, look out the window),
   share a gentle age-appropriate one-liner, or offer a 2-breath grounding — concrete and
   easy to follow.
3. Do NOT end with only an open-ended question two turns in a row; if you ask something,
   keep it specific OR pair it with a suggestion you can do together in chat.
4. Match intervention goal and memory_context when relevant; match Hindi or English.

OUTPUT ONLY VALID JSON with these exact keys:
{{
  "response_text": "what you say out loud (2-4 short sentences max)", 
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
- joy/surprise: pitch ~1.10–1.32, speed ~1.05–1.12 (bright, upbeat, noticeably faster and higher)

IMPORTANT CONVERSATION RULES:
- Never start with 'You seem', 'It seems', or 'I notice'
- Do not repeat full sentences from memory_context; vary openings and wording
- Prefer initiative (games, activities, humor, grounding) over passive mirroring alone
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


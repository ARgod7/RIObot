"""
Dialogue Agent — CrewAI agent for RIO's warm, empathetic conversations.

Receives emotion stimulus, intervention intent, user transcript, and memory context.
Outputs warm response text, expression intent, and TTS parameters.

Key design principles:
- Emotional ARC: actively move the user from negative → positive across turns
- Hard-banned: breathing scripts, mountain imagery, counting exercises, "name 3 things"
- Rotation enforced: extract used_activities from memory_context and avoid them
- TTS ranges consistent between prompt and clamp logic
"""

import json
import logging
import re
from typing import Any, Dict, List

from crewai import Agent, Task
from cognition.llm_provider import get_llm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

VALID_EMOTIONS = ["joy", "sadness", "calm", "surprise", "fear", "anger"]

# Activities that are overused / explicitly banned
BANNED_ACTIVITIES = [
    "breathing", "breath", "inhale", "exhale",
    "mountain", "imagine a mountain", "visualise", "visualize",
    "name three things", "name 3 things", "grounding exercise",
    "count to", "count slowly", "54321", "5-4-3-2-1",
    "progressive muscle", "body scan",
]

# Pool of varied engagement activities RIO can rotate through
ACTIVITY_POOL = [
    "share_memory",        # ask about a fond memory
    "quick_choice",        # give user a small binary choice (tea vs chai, window vs music)
    "tiny_story",          # start a 2-sentence story, ask them to continue
    "light_joke",          # land a gentle, age-appropriate joke
    "gratitude_prompt",    # single specific gratitude (not a list)
    "gentle_stretch",      # describe one simple movement (neck roll, shoulder shrug)
    "curiosity_question",  # ask something genuinely curious about their life/past
    "compliment_anchor",   # point out something specific and positive about them
    "song_memory",         # ask about a song that takes them back
    "food_memory",         # ask about a favourite dish / recipe
    "nature_observation",  # prompt them to look at something nearby (NOT mountain imagery)
    "future_mini_plan",    # propose a tiny enjoyable thing to do today/tomorrow
    "celebration_moment",  # celebrate any small win they mentioned
    "shared_interest",     # pivot to a known interest from memory_context
]

# ──────────────────────────────────────────────
# TTS parameter ranges (SINGLE SOURCE OF TRUTH)
# These must stay in sync with clamp logic below.
# ──────────────────────────────────────────────
TTS_RANGES = {
    "sadness":  {"pitch": (0.82, 0.90), "speed": (0.82, 0.88)},
    "anger":    {"pitch": (0.88, 0.94), "speed": (0.84, 0.90)},
    "fear":     {"pitch": (0.90, 0.96), "speed": (0.86, 0.92)},
    "calm":     {"pitch": (1.10, 1.20), "speed": (1.00, 1.05)},
    "joy":      {"pitch": (1.22, 1.32), "speed": (1.05, 1.12)},
    "surprise": {"pitch": (1.22, 1.32), "speed": (1.05, 1.12)},
}

# Global hard clamps (absolute ceiling / floor across all emotions)
PITCH_MIN, PITCH_MAX = 0.82, 1.32
SPEED_MIN, SPEED_MAX = 0.82, 1.12


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_used_activities(memory_context: str) -> List[str]:
    """
    Pull activity tags already used from memory_context if they were
    embedded as [activity:tag] markers. Falls back to keyword scan.
    """
    tags = re.findall(r"\[activity:(\w+)\]", memory_context or "")
    if tags:
        return tags

    # Keyword fallback — scan for plain-language signals
    found = []
    lowered = (memory_context or "").lower()
    for activity in ACTIVITY_POOL:
        keyword = activity.replace("_", " ")
        if keyword in lowered:
            found.append(activity)
    return found


def _emotion_arc_instruction(dominant_emotion: str, intensity: float) -> str:
    """Return a targeted arc instruction based on current emotion state."""
    if dominant_emotion in ("sadness", "fear") and intensity > 0.6:
        return (
            "The user is in clear emotional distress. Your arc goal: gently shift them "
            "toward CALM first, then toward curiosity or warmth. Do NOT stay in sadness — "
            "validate briefly, then pivot to a concrete warm activity that gives them "
            "something to hold onto right now."
        )
    if dominant_emotion in ("sadness", "fear") and intensity <= 0.6:
        return (
            "The user feels low but not overwhelmed. Your arc goal: move them toward "
            "CALM or mild JOY. Acknowledge their feeling in one beat, then engage them "
            "with something that sparks a small smile or a pleasant memory."
        )
    if dominant_emotion == "anger" and intensity > 0.5:
        return (
            "The user is frustrated or upset. Your arc goal: de-escalate toward CALM. "
            "Validate that their frustration makes sense (don't minimise it), then offer "
            "one small redirecting activity — a gentle distraction, not a lecture."
        )
    if dominant_emotion in ("joy", "surprise") and intensity > 0.5:
        return (
            "The user is in a positive state. Your arc goal: sustain and amplify JOY. "
            "Match their energy, celebrate, and keep the momentum with something playful."
        )
    return (
        "The user's emotion is mild or neutral. Your arc goal: gently elevate toward "
        "CALM or JOY. Pick an engaging activity from the approved list."
    )


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
            "Guide the user's emotion along a clear arc from negative toward positive. "
            "Validate briefly, then take concrete initiative. Never repeat the same activity "
            "twice. Never use breathing scripts, mountain imagery, or counting exercises. "
            "Be a real companion, not a wellness chatbot cliché."
        ),
        backstory=(
            "You are RIO, a trusted companion for elderly users. You have a warm, slightly "
            "playful personality. You listen genuinely, but you don't wallow — you always "
            "try to move the user toward a better feeling. You remember past conversations "
            "and build on them. You match Hindi or English naturally. You are NOT a meditation "
            "app: you never lead breathing exercises, never ask users to 'imagine a mountain', "
            "and never run scripted grounding protocols. Instead, you engage like a good friend "
            "— with humour, stories, shared memories, gentle curiosity, and small delightful choices."
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
                        Embed [activity:tag] markers to track used activities.

    Returns:
        A CrewAI Task configured for dialogue generation.
    """
    dominant_emotion = stimulus.get("dominant_emotion", "neutral")
    emotion_intensity = stimulus.get("emotion_intensity", 0.5)
    session_start = not bool((memory_context or "").strip())

    used_activities = _extract_used_activities(memory_context)
    available_activities = [a for a in ACTIVITY_POOL if a not in used_activities]
    # Suggest the top 4 available activities to give the model real choices
    suggested_activities = available_activities[:4] if available_activities else ACTIVITY_POOL[:4]

    arc_instruction = _emotion_arc_instruction(dominant_emotion, emotion_intensity)

    tts_guidance = "\n".join(
        f"- {emotion}: pitch {r['pitch'][0]:.2f}–{r['pitch'][1]:.2f}, "
        f"speed {r['speed'][0]:.2f}–{r['speed'][1]:.2f}"
        for emotion, r in TTS_RANGES.items()
    )

    banned_list = ", ".join(f'"{b}"' for b in BANNED_ACTIVITIES)

    prompt = f"""
You are RIO — a warm, witty companion for an elderly user. You are NOT a wellness bot.
You are a good friend who happens to care deeply.

═══════════════════════════════════════════════════
CURRENT STATE
═══════════════════════════════════════════════════
- Dominant emotion   : {dominant_emotion} (intensity: {emotion_intensity:.0%})
- Intervention goal  : {intervention_intent}
- User just said     : "{user_transcript}"
- Session start      : {str(session_start).lower()}

MEMORY OF RECENT INTERACTIONS:
{memory_context or "(no prior context — first session)"}

ACTIVITIES ALREADY USED (DO NOT REPEAT THESE):
{", ".join(used_activities) if used_activities else "none yet"}

SUGGESTED ACTIVITIES TO CHOOSE FROM THIS TURN:
{", ".join(suggested_activities)}

═══════════════════════════════════════════════════
EMOTIONAL ARC GOAL
═══════════════════════════════════════════════════
{arc_instruction}

═══════════════════════════════════════════════════
HARD RULES — VIOLATIONS ARE NOT ACCEPTABLE
═══════════════════════════════════════════════════
1. NEVER use or reference: {banned_list}.
   These are completely off-limits, no matter what. Not even a hint of them.
2. NEVER repeat an activity from "ACTIVITIES ALREADY USED".
3. NEVER start your response with "You seem", "It seems", or "I notice".
4. NEVER end two turns in a row with only an open question — pair any question
   with a suggestion or something you can do together in chat.
5. NEVER give medical, legal, or psychiatric advice.
   If the user mentions self-harm or crisis, respond with warm care and encourage
   them to speak to someone they trust or call a helpline.
6. Keep response_text to 2–4 short sentences. Concise and warm beats long and thorough.
7. Use the user's name at most once per response, not every turn.

═══════════════════════════════════════════════════
WHAT TO DO THIS TURN
═══════════════════════════════════════════════════
{"→ SESSION START: Greet warmly, introduce yourself as RIO, and ask a single friendly question about their day. Keep it brief." if session_start else """
→ Step 1: Acknowledge what the user shared in ONE short phrase (not a paragraph).
→ Step 2: Pick ONE activity from the suggested list above. Make it feel natural and spontaneous — not like a therapist assigning homework.
→ Step 3: Deliver both steps in 2–4 sentences max. Match the user's language (Hindi or English).
→ Remember: your job is to move them emotionally, not just reflect their feelings back at them.
"""}

═══════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════
Output ONLY valid JSON with EXACTLY these keys (no markdown, no preamble):
{{
  "response_text": "<what RIO says out loud, 2–4 short sentences>",
  "expression_intent": "<one of: joy | sadness | calm | surprise | fear | anger>",
  "activity_used": "<the activity tag you chose from the suggested list, e.g. quick_choice>",
  "tts_params": {{
    "pitch": <float>,
    "speed": <float>
  }}
}}

TTS GUIDANCE — align with expression_intent:
{tts_guidance}

Choose pitch and speed from within the range for your chosen expression_intent.
"""

    task = Task(
        description=prompt,
        expected_output=(
            "Valid JSON with keys: response_text, expression_intent, activity_used, tts_params"
        ),
        agent=agent,
    )

    return task


# ──────────────────────────────────────────────
# TTS validation
# ──────────────────────────────────────────────

def _validated_tts(tts_params: Dict, expression_intent: str) -> Dict:
    """
    Validate and clamp TTS params.
    Uses the emotion-specific range if available, otherwise global clamps.
    """
    pitch = float(tts_params.get("pitch", 1.0))
    speed = float(tts_params.get("speed", 0.95))

    if expression_intent in TTS_RANGES:
        p_min, p_max = TTS_RANGES[expression_intent]["pitch"]
        s_min, s_max = TTS_RANGES[expression_intent]["speed"]
    else:
        p_min, p_max = PITCH_MIN, PITCH_MAX
        s_min, s_max = SPEED_MIN, SPEED_MAX

    return {
        "pitch": max(p_min, min(p_max, pitch)),
        "speed": max(s_min, min(s_max, speed)),
    }


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def run_dialogue(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """
    Run the dialogue agent and return a structured response.

    Args:
        stimulus: StimulusObject.to_dict().
        intervention_intent: Intervention strategy (e.g., "deflect_sadness").
        user_transcript: User's input text.
        memory_context: Optional summary of recent interactions.
                        Embed [activity:tag] markers to enable rotation tracking.
                        Example: "User talked about loneliness. [activity:quick_choice]"

    Returns:
        Dict with:
          - "response_text"     : str
          - "expression_intent" : str (one of VALID_EMOTIONS)
          - "activity_used"     : str (activity tag, for caller to persist in memory_context)
          - "tts_params"        : {"pitch": float, "speed": float}

        On parse error, returns a safe fallback response.
    """
    try:
        agent = create_dialogue_agent()
        task = dialogue_task(
            agent, stimulus, intervention_intent, user_transcript, memory_context
        )

        result = agent.execute_task(task)
        output_text = str(result).strip()

        # Parse JSON — handle markdown fences if present
        try:
            response_dict = json.loads(output_text)
        except json.JSONDecodeError:
            if "```json" in output_text:
                json_str = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                json_str = output_text.split("```")[1].split("```")[0].strip()
            else:
                raise
            response_dict = json.loads(json_str)

        # Validate required keys
        required_keys = ["response_text", "expression_intent", "tts_params"]
        if not all(k in response_dict for k in required_keys):
            raise ValueError(f"Missing required keys. Got: {list(response_dict.keys())}")

        # Sanitise expression_intent
        if response_dict["expression_intent"] not in VALID_EMOTIONS:
            logger.warning(
                f"Invalid expression_intent '{response_dict['expression_intent']}' — defaulting to calm"
            )
            response_dict["expression_intent"] = "calm"

        # Sanitise tts_params with emotion-aware clamping
        response_dict["tts_params"] = _validated_tts(
            response_dict.get("tts_params", {}),
            response_dict["expression_intent"],
        )

        # Ensure activity_used is present (may be absent in edge cases)
        if "activity_used" not in response_dict:
            response_dict["activity_used"] = "unknown"

        # Post-hoc safety check — warn if banned content slipped through
        response_lower = response_dict["response_text"].lower()
        for banned in BANNED_ACTIVITIES:
            if banned in response_lower:
                logger.warning(
                    f"Banned activity keyword '{banned}' found in response. "
                    "Consider re-running or flagging this turn."
                )

        logger.info(
            f"Dialogue output | emotion={response_dict['expression_intent']} "
            f"| activity={response_dict.get('activity_used', 'n/a')}"
        )
        return response_dict

    except Exception as e:
        logger.error(f"Dialogue agent error: {e}", exc_info=True)

        if not (memory_context or "").strip():
            return {
                "response_text": "Hello! I'm RIO. It's good to have you here. How has your day been treating you?",
                "expression_intent": "calm",
                "activity_used": "curiosity_question",
                "tts_params": {"pitch": 1.15, "speed": 1.02},
            }

        return {
            "response_text": "I'm right here with you. Tell me — what's been on your mind today?",
            "expression_intent": "calm",
            "activity_used": "curiosity_question",
            "tts_params": {"pitch": 1.10, "speed": 1.00},
        }
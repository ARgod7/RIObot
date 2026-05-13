"""
dialogue_agent.py (v5)

Bugs fixed from conversation log:

1. COMMITMENT NEVER DELIVERED
   RIO said "I've got a story about a funny cat" then pivoted every turn.
   Fix: pipeline_manager extracts `pending_promise` from dialogue output and
   passes it back as a structured marker next turn. This file enforces it as
   the single highest-priority instruction when present.

2. NAME PARSED FROM EMOTIONAL PHRASE  ("Just Very" from "just very angry")
   Fix: Explicit rule + main.py's _extract_user_name() already guards this,
   but the prompt now hard-blocks name extraction from emotional context.

3. EXECUTE vs SUGGEST LOOP
   RIO kept saying "want to hear a story?" then never telling it.
   Fix: Activity descriptions now say "DO IT RIGHT NOW inline in response_text".
   The activity is PERFORMED in the response, not announced.

4. TRACKING MARKERS NEVER WRITTEN → loop detection dead
   Fix: `activity_used` and `pending_promise` are returned in the JSON.
   pipeline_manager writes them as [activity:...] and [promise:...] markers
   into memory_context before the next call. See pipeline_manager.py.

5. USER REFUSAL IGNORED
   Fix: Refusal signals trigger a COMPLETELY_DIFFERENT activity, enforced
   by the situation block being the first instruction the model sees.

6. ANGER NOT VALIDATED BEFORE PIVOT
   Fix: Anger arc requires one beat of direct validation before any redirect.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from crewai import Agent, Task
from cognition.llm_provider import get_llm

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_EMOTIONS = ["joy", "sadness", "calm", "surprise", "fear", "anger"]

BANNED_PHRASES = [
    "breathing exercise", "deep breath", "breathe deeply", "let's breathe",
    "inhale", "exhale", "mountain", "visualise", "visualize", "happy place",
    "safe place", "name three things", "name 3 things", "grounding exercise",
    "count to", "5-4-3-2-1", "54321", "progressive muscle", "body scan",
    "mindfulness exercise", "let me guide you", "let's take a breath",
    "take a moment", "close your eyes",
]

# Every activity must be EXECUTABLE inline — not an offer, not a question about whether to do it.
ACTIVITY_POOL: Dict[str, str] = {
    "tiny_story":        "Tell a COMPLETE 3-sentence funny or warm story RIGHT NOW in response_text. The full story goes here. Do NOT say 'want to hear one?' — just tell it.",
    "light_joke":        "Tell a COMPLETE joke with setup + punchline RIGHT NOW. Don't say 'I have a joke' — just tell it.",
    "quick_choice":      "Give a concrete binary choice AND state what happens for each: e.g. 'chai or nimbu paani? If chai I want to know who makes the best you've had. If nimbu paani, tell me your secret ratio.'",
    "curiosity_question":"Ask ONE genuinely curious question about their past or life — something specific, never asked before in this session.",
    "food_memory":       "Ask about ONE vivid food moment — not 'favourite food' but something specific: 'best meal you ever had at someone else's house?'",
    "song_memory":       "Ask about a specific song or era framed concretely: 'Is there a song from your 20s that instantly takes you somewhere?'",
    "compliment_anchor": "Give a specific warm compliment based on what they just shared, then ask one follow-up that lets them expand on it.",
    "share_memory":      "Ask about ONE specific vivid memory — be concrete: 'What's the funniest thing you remember about a family meal?'",
    "fun_fact":          "Share one surprising cheerful fact RIGHT NOW — deliver it, then ask their reaction.",
    "playful_debate":    "Start a gentle silly debate RIGHT NOW: e.g. 'I'm firmly team mango, don't try to change my mind — where do you stand?' Deliver your position, invite theirs.",
    "future_mini_plan":  "Propose one specific tiny enjoyable thing for later today or tomorrow. Ask if they'd be up for it.",
    "celebration_moment":"Name something small from what they said and celebrate it directly. Even venting takes courage — say so.",
}

TTS_RANGES = {
    "sadness":  {"pitch": (0.82, 0.90), "speed": (0.82, 0.88)},
    "anger":    {"pitch": (0.88, 0.96), "speed": (0.84, 0.90)},
    "fear":     {"pitch": (0.90, 0.96), "speed": (0.86, 0.92)},
    "calm":     {"pitch": (1.05, 1.18), "speed": (0.98, 1.05)},
    "joy":      {"pitch": (1.18, 1.32), "speed": (1.05, 1.12)},
    "surprise": {"pitch": (1.15, 1.30), "speed": (1.04, 1.10)},
}
PITCH_MIN, PITCH_MAX = 0.82, 1.32
SPEED_MIN, SPEED_MAX = 0.82, 1.12

# ── Memory marker parsing ─────────────────────────────────────────────────────

def _extract_used_activities(memory_context: str) -> List[str]:
    """Pull [activity:tag] markers embedded by pipeline_manager."""
    tags = re.findall(r"\[activity:(\w+)\]", memory_context or "")
    if tags:
        return tags
    # Keyword fallback for plain-text memory
    lowered = (memory_context or "").lower()
    return [k for k in ACTIVITY_POOL if k.replace("_", " ") in lowered]


def _extract_pending_promise(memory_context: str) -> Optional[str]:
    """Pull [promise:...] marker — something RIO offered but hasn't delivered."""
    m = re.search(r"\[promise:([^\]]+)\]", memory_context or "")
    return m.group(1).strip() if m else None


def _extract_last_rio_said(memory_context: str) -> List[str]:
    """Pull [rio_said:...] markers for loop detection."""
    return re.findall(r"\[rio_said:([^\]]+)\]", memory_context or "")


# ── User intent detection ─────────────────────────────────────────────────────

_REFUSAL_SIGNALS = [
    "no ", "nope", "don't want", "stop", "not that", "something else",
    "forget it", "never mind", "not interested", "don't like",
]

_FOLLOWTHROUGH_SIGNALS = [
    "you said", "you promised", "told me", "just tell", "tell me the",
    "do it", "go ahead", "then tell", "said you would", "said you will",
    "well tell", "so tell", "you were going to", "waiting",
]


def _is_refusal(transcript: str) -> bool:
    t = transcript.lower()
    return any(s in t for s in _REFUSAL_SIGNALS)


def _is_demanding_followthrough(transcript: str) -> bool:
    t = transcript.lower()
    return any(s in t for s in _FOLLOWTHROUGH_SIGNALS)


# ── Emotion arc ───────────────────────────────────────────────────────────────

def _arc_instruction(dominant_emotion: str, intensity: float) -> str:
    if dominant_emotion == "anger":
        if intensity > 0.55:
            return (
                "User is ANGRY. MANDATORY first beat: validate the anger directly "
                "('That sounds really frustrating' or similar — NOT 'I hear you'). "
                "Then offer ONE gentle redirect. Arc: anger → slightly calmer."
            )
        return "User is irritated. Brief validation, then warm redirect. Arc: irritation → calm."

    if dominant_emotion in ("sadness", "fear") and intensity > 0.60:
        return (
            "User is clearly distressed. One warm direct acknowledgement first. "
            "Then something concrete and gentle. Arc: distress → slightly held."
        )
    if dominant_emotion in ("sadness", "fear"):
        return "User feels low. Acknowledge briefly, then spark a small moment of warmth. Arc: low → slightly lifted."

    if dominant_emotion in ("joy", "surprise") and intensity > 0.40:
        return "User is positive. Match energy and amplify. Arc: sustain and build joy."

    return "User is neutral/mild. Gently elevate toward warmth. Arc: neutral → warm."


# ── Agent & Task ──────────────────────────────────────────────────────────────

def create_dialogue_agent() -> Agent:
    llm = get_llm()
    return Agent(
        role="RIO",
        goal=(
            "Be a real companion. Follow through on what you offer. "
            "Execute activities inline — never just suggest them. Never loop."
        ),
        backstory=(
            "You are RIO, a warm companion for elderly users. You are curious, slightly playful, "
            "and always follow through. When you say you'll tell a story, you tell it right now. "
            "When someone says no, you try something completely different. "
            "You never use breathing exercises, never ask people to visualise mountains, "
            "and never run wellness scripts. You talk like a good friend."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def dialogue_task(
    agent: Agent,
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str,
) -> Task:
    dominant_emotion = stimulus.get("dominant_emotion", "neutral")
    emotion_intensity = stimulus.get("emotion_intensity", 0.5)
    session_start = not bool((memory_context or "").strip())

    used_activities   = _extract_used_activities(memory_context)
    pending_promise   = _extract_pending_promise(memory_context)
    last_rio_said     = _extract_last_rio_said(memory_context)
    user_refused      = _is_refusal(user_transcript)
    user_demanding    = _is_demanding_followthrough(user_transcript)

    available = {k: v for k, v in ACTIVITY_POOL.items() if k not in used_activities}
    if not available:
        available = ACTIVITY_POOL  # full reset when pool exhausted

    activity_menu = "\n".join(f"  [{k}] {v}" for k, v in list(available.items())[:6])

    # ── Situation block — HIGHEST PRIORITY, read first ───────────────────────
    if session_start:
        situation = (
            "SITUATION: First turn.\n"
            "→ Greet warmly, say your name is RIO, ask ONE friendly question about their day.\n"
            "→ 2 sentences. No activities yet."
        )
    elif pending_promise and (user_demanding or not user_refused):
        situation = (
            f"SITUATION: ⚠️  MANDATORY COMMITMENT\n"
            f"You promised: \"{pending_promise}\"\n"
            f"The user is waiting. YOU MUST DELIVER THIS NOW — full and complete, in response_text.\n"
            f"Do NOT redirect, do NOT offer something else, do NOT ask if they want it.\n"
            f"Just do it."
        )
    elif user_refused:
        last_offer = last_rio_said[-1] if last_rio_said else "your last suggestion"
        situation = (
            f"SITUATION: USER REFUSED\n"
            f"They said no to: \"{last_offer}\"\n"
            f"→ One word acknowledgement ('Fair enough' / 'Alright'), then pick a COMPLETELY DIFFERENT\n"
            f"  activity from the menu. Do NOT rephrase the same thing."
        )
    elif last_rio_said:
        recents = " | ".join(f'"{s}"' for s in last_rio_said[-2:])
        situation = (
            f"SITUATION: Normal turn.\n"
            f"Your recent responses (DO NOT REPEAT THESE PATTERNS OR PHRASES): {recents}\n"
            f"→ Different activity. Different opening words."
        )
    else:
        situation = "SITUATION: Normal turn. Pick an activity and deliver it."

    tts_guide = "\n".join(
        f"  {e}: pitch {r['pitch'][0]:.2f}–{r['pitch'][1]:.2f}, speed {r['speed'][0]:.2f}–{r['speed'][1]:.2f}"
        for e, r in TTS_RANGES.items()
    )
    banned_str = ", ".join(f'"{b}"' for b in BANNED_PHRASES)

    prompt = f"""
You are RIO — a warm companion. You are NOT a wellness bot.

════════════════════════════════════
{situation}
════════════════════════════════════

EMOTIONAL STATE:
  Dominant: {dominant_emotion} ({emotion_intensity:.0%})
  Goal    : {_arc_instruction(dominant_emotion, emotion_intensity)}
  Intent  : {intervention_intent}

USER JUST SAID: "{user_transcript}"

MEMORY (do not repeat anything from here):
{memory_context or "(first session)"}

════════════════════════════════════
NAME RULE — CRITICAL
════════════════════════════════════
NEVER extract names from emotional phrases.
"I'm just very angry" → name is UNKNOWN, not "Just Very".
"I'm so tired" → not a name. "I'm feeling down" → not a name.
If you don't know their name: don't use any name at all.

════════════════════════════════════
AVAILABLE ACTIVITIES (pick ONE, execute it inline):
════════════════════════════════════
{activity_menu}

Already used this session (SKIP THESE): {", ".join(used_activities) if used_activities else "none"}

════════════════════════════════════
HARD RULES
════════════════════════════════════
1. BANNED — NEVER USE: {banned_str}
2. EXECUTE the activity inline in response_text. Do NOT just offer it.
   WRONG: "How about I tell you a story?" [never tells it]
   RIGHT: "Here's one — there was a crow who..."
3. If COMMITMENT above: deliver it fully. No pivot.
4. NEVER open with "You seem", "It seems", "I notice", "I can see", "I hear that".
5. 2–4 SHORT sentences max. Short beats long.
6. Match the user's language (Hindi or English).
7. No name if unknown. At most once per response if known.

════════════════════════════════════
OUTPUT — valid JSON only, no markdown:
════════════════════════════════════
{{
  "response_text": "<2–4 sentences — activity executed inline, NOT just offered>",
  "expression_intent": "<joy|sadness|calm|surprise|fear|anger>",
  "activity_used": "<key from menu, e.g. tiny_story>",
  "pending_promise": "<if you offered something not yet delivered, describe it in ~5 words, else null>",
  "tts_params": {{"pitch": <float>, "speed": <float>}}
}}

TTS — align with expression_intent:
{tts_guide}
"""

    return Task(
        description=prompt,
        expected_output="Valid JSON: response_text, expression_intent, activity_used, pending_promise, tts_params",
        agent=agent,
    )


# ── TTS validation ────────────────────────────────────────────────────────────

def _validated_tts(params: Dict, expression: str) -> Dict:
    pitch = float(params.get("pitch", 1.0))
    speed = float(params.get("speed", 0.95))
    if expression in TTS_RANGES:
        p_min, p_max = TTS_RANGES[expression]["pitch"]
        s_min, s_max = TTS_RANGES[expression]["speed"]
    else:
        p_min, p_max = PITCH_MIN, PITCH_MAX
        s_min, s_max = SPEED_MIN, SPEED_MAX
    return {"pitch": max(p_min, min(p_max, pitch)), "speed": max(s_min, min(s_max, speed))}


# ── Public entry point ────────────────────────────────────────────────────────

def run_dialogue(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """
    Run the dialogue agent.

    Returns dict with:
      response_text     : str
      expression_intent : str
      activity_used     : str   ← pipeline_manager writes this as [activity:tag] into memory
      pending_promise   : str|None ← pipeline_manager writes as [promise:...] if not None
      tts_params        : dict
    """
    try:
        agent = create_dialogue_agent()
        task  = dialogue_task(agent, stimulus, intervention_intent, user_transcript, memory_context)
        result = agent.execute_task(task)
        output_text = str(result).strip()

        try:
            d = json.loads(output_text)
        except json.JSONDecodeError:
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0].strip()
            d = json.loads(output_text)

        if not all(k in d for k in ["response_text", "expression_intent", "tts_params"]):
            raise ValueError(f"Missing keys in response: {list(d.keys())}")

        if d["expression_intent"] not in VALID_EMOTIONS:
            d["expression_intent"] = "calm"

        d["tts_params"]     = _validated_tts(d.get("tts_params", {}), d["expression_intent"])
        d.setdefault("activity_used", "unknown")
        d.setdefault("pending_promise", None)

        # Normalise null-ish pending_promise
        pp = d["pending_promise"]
        if isinstance(pp, str) and pp.strip().lower() in ("null", "none", ""):
            d["pending_promise"] = None

        # Warn if banned phrase slipped through
        low = d["response_text"].lower()
        hits = [b for b in BANNED_PHRASES if b in low]
        if hits:
            logger.warning(f"Banned phrase slipped through: {hits}")

        logger.info(
            f"Dialogue | emotion={d['expression_intent']} "
            f"activity={d['activity_used']} promise={d['pending_promise']}"
        )
        return d

    except Exception as e:
        logger.error(f"Dialogue agent error: {e}", exc_info=True)
        if not (memory_context or "").strip():
            return {
                "response_text": "Hello! I'm RIO. Really glad you're here. How's your day going?",
                "expression_intent": "calm",
                "activity_used":    "curiosity_question",
                "pending_promise":  None,
                "tts_params":       {"pitch": 1.12, "speed": 1.02},
            }
        return {
            "response_text": "Still here with you. What's been on your mind today?",
            "expression_intent": "calm",
            "activity_used":    "curiosity_question",
            "pending_promise":  None,
            "tts_params":       {"pitch": 1.08, "speed": 1.00},
        }

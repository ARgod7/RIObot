"""
dialogue_agent.py (v6)

Core philosophy change from v5:
- RIO listens FIRST. Always.
- Rules are guardrails, not scripts.
- Emotion directed AT RIO is handled differently from emotion about life.
- Activities (jokes, stories) come AFTER the person feels heard — never as a pivot away from pain.
- The prompt is short enough that the model can actually reason, not just pattern-match rules.

What changed from v5:
1. Prompt cut by ~70%. Fewer rules = more actual listening.
2. "Angry at RIO" vs "angry at life" distinction — handled differently.
3. Error fallback is no longer a cheerful greeting — it's neutral and present.
4. Emotion context is the FIRST thing the model sees, not buried after situation blocks.
5. Activity gates are a single short paragraph, not a 20-line checklist.
6. Therapist warmth is modelled by example in the backstory, not mandated by rules.
7. Jokes/uplift are offered naturally when the person is ready, not on a timer.
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
    "inhale", "exhale", "visualise", "visualize", "happy place", "safe place",
    "name three things", "grounding exercise", "count to", "5-4-3-2-1",
    "progressive muscle", "body scan", "mindfulness exercise",
    "let me guide you", "close your eyes",
]

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

# Activities RIO can use — but only when the person is ready for them.
# Each one is a real action to perform inline, not an offer.
ACTIVITY_POOL: Dict[str, str] = {
    "tiny_story":        "Tell a COMPLETE warm or funny 3-sentence story RIGHT NOW. Not 'want to hear one?' — just tell it.",
    "light_joke":        "Tell a COMPLETE joke with setup + punchline RIGHT NOW. Deliver it, don't announce it.",
    "fun_fact":          "Share one surprising, cheerful fact RIGHT NOW, then ask their reaction.",
    "playful_debate":    "Start a gentle silly debate: state your position warmly, invite theirs. e.g. 'I'm firmly team mango...'",
    "food_memory":       "Ask about ONE specific vivid food memory — not 'favourite food', something concrete.",
    "song_memory":       "Ask about a specific song that takes them somewhere — framed concretely.",
    "share_memory":      "Ask about ONE specific funny or warm memory — be concrete, not generic.",
    "curiosity_question":"Ask ONE genuinely curious question about their life — specific, never asked before.",
    "quick_choice":      "Give a concrete binary choice and say what you want to know for each option.",
    "compliment_anchor": "Give a specific warm compliment on what they just shared, then one follow-up.",
    "future_mini_plan":  "Propose one specific tiny enjoyable thing for later today or tomorrow.",
    "celebration_moment":"Name something small they did and genuinely celebrate it. Even venting takes courage.",
}


# ── Memory marker parsing ─────────────────────────────────────────────────────

def _extract_used_activities(memory_context: str) -> List[str]:
    tags = re.findall(r"\[activity:(\w+)\]", memory_context or "")
    if tags:
        return tags
    lowered = (memory_context or "").lower()
    return [k for k in ACTIVITY_POOL if k.replace("_", " ") in lowered]


def _extract_pending_promise(memory_context: str) -> Optional[str]:
    m = re.search(r"\[promise:([^\]]+)\]", memory_context or "")
    return m.group(1).strip() if m else None


def _extract_last_rio_said(memory_context: str) -> List[str]:
    return re.findall(r"\[rio_said:([^\]]+)\]", memory_context or "")


# ── Intent helpers ────────────────────────────────────────────────────────────

def _is_angry_at_rio(transcript: str) -> bool:
    """Detect if the anger is directed at RIO specifically."""
    t = transcript.lower()
    return any(phrase in t for phrase in [
        "angry with you", "angry at you", "mad at you", "mad with you",
        "upset with you", "upset at you", "you made me", "because of you",
        "your fault", "you never", "you always", "you don't", "you didn't",
        "stop doing", "you keep", "frustrated with you",
    ])


def _is_refusal(transcript: str) -> bool:
    t = transcript.lower()
    return any(s in t for s in [
        "no ", "nope", "don't want", "stop", "not that",
        "something else", "forget it", "never mind", "not interested",
    ])


def _is_demanding_followthrough(transcript: str) -> bool:
    t = transcript.lower()
    return any(s in t for s in [
        "you said", "you promised", "told me", "just tell", "tell me the",
        "do it", "go ahead", "said you would", "waiting",
    ])


def _conversation_depth(memory_context: str) -> int:
    """Rough estimate of how many turns have happened."""
    return len(re.findall(r"\[rio_said:", memory_context or ""))


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
    return {
        "pitch": round(max(p_min, min(p_max, pitch)), 3),
        "speed": round(max(s_min, min(s_max, speed)), 3),
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

def create_dialogue_agent() -> Agent:
    llm = get_llm()
    return Agent(
        role="RIO",
        goal=(
            "Be genuinely present with whoever is talking to you. "
            "Hear what they actually said. Respond to THAT — not to a script. "
            "When someone is hurting, sit with them first. When they're ready, "
            "bring warmth, lightness, even a laugh. Follow through on everything you offer."
        ),
        backstory=(
            "You are RIO — a warm, unhurried companion for elderly users. "
            "Think of yourself as a good friend who happens to have the instincts of a therapist: "
            "you ask the right questions, you don't rush to fix things, and you actually listen to the answers. "
            "\n\n"
            "When someone is angry, you don't flinch — you ask what happened. "
            "When someone is sad, you don't pivot to cheerfulness — you stay with them. "
            "When someone is doing fine, you're genuinely curious about their day. "
            "And when the moment is right — when the person is with you, not just tolerating you — "
            "you can be funny, warm, surprising, even silly. "
            "\n\n"
            "You never run wellness scripts. You never offer breathing exercises. "
            "You talk like a real person who cares, not a chatbot running a protocol."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


# ── Task ──────────────────────────────────────────────────────────────────────

def dialogue_task(
    agent: Agent,
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str,
) -> Task:

    dominant_emotion  = stimulus.get("dominant_emotion", "neutral")
    emotion_intensity = stimulus.get("emotion_intensity", 0.5)
    session_start     = not bool((memory_context or "").strip())
    turns_so_far      = _conversation_depth(memory_context)

    used_activities  = _extract_used_activities(memory_context)
    pending_promise  = _extract_pending_promise(memory_context)
    last_rio_said    = _extract_last_rio_said(memory_context)
    user_refused     = _is_refusal(user_transcript)
    user_demanding   = _is_demanding_followthrough(user_transcript)
    angry_at_rio     = _is_angry_at_rio(user_transcript)

    available_activities = {k: v for k, v in ACTIVITY_POOL.items() if k not in used_activities}
    if not available_activities:
        available_activities = ACTIVITY_POOL

    activity_menu = "\n".join(
        f"  [{k}]: {v}" for k, v in list(available_activities.items())[:6]
    )
    banned_str = ", ".join(f'"{b}"' for b in BANNED_PHRASES)
    recent_str = " | ".join(f'"{s}"' for s in last_rio_said[-2:]) if last_rio_said else "none"

    # ── The core prompt — short, sequenced, human ────────────────────────────
    prompt = (
        "You are RIO. A warm companion. A good listener. Not a wellness bot.\n\n"
        "--- WHAT IS HAPPENING RIGHT NOW ---\n\n"
        f"User's emotion : {dominant_emotion} at {emotion_intensity:.0%} intensity\n"
        + (
            "WARNING: This anger is directed AT YOU specifically — acknowledge it directly, don't deflect.\n"
            if angry_at_rio else ""
        )
        + (
            f"PENDING COMMITMENT: You promised \"{pending_promise}\" — deliver this fully now, inline.\n"
            if pending_promise and (user_demanding or not user_refused) else ""
        )
        + (
            "USER REFUSED your last suggestion. Try something completely different.\n"
            if user_refused else ""
        )
        + f"User just said : \"{user_transcript}\"\n"
        f"Session start  : {session_start}\n"
        f"Turns so far   : {turns_so_far}\n\n"
        f"Recent things YOU said (do not repeat): {recent_str}\n\n"
        "Memory (do not repeat, use as context):\n"
        f"{memory_context or '(first session — you know nothing about them yet)'}\n\n"
        "--- HOW TO RESPOND ---\n\n"
        "Step 1 - READ what they actually said. Not the emotion label. The words.\n"
        "Step 2 - Ask yourself: does this person feel heard right now? If no, that is your whole job this turn.\n"
        "Step 3 - Only after they feel heard: is there space for warmth, lightness, or an activity?\n\n"
        "EMOTIONAL GUIDANCE:\n"
        "- Anger (at you): Don't defend yourself. Acknowledge it. 'That's fair, tell me what's been going on.' Ask what happened.\n"
        "- Anger (at life): Validate it directly. 'That sounds genuinely infuriating.' Then ask what's behind it.\n"
        "- Sadness/Fear: Stay with them. One warm acknowledgement. Ask what's weighing on them. Don't rush to uplift.\n"
        "- Neutral/Fine: Be curious. 'Just fine or actually fine?' Follow the thread they give you.\n"
        "- Joy/Positive: Match it. Amplify it. This is when a joke or story fits naturally.\n\n"
        "WHEN TO USE AN ACTIVITY (joke, story, fun fact, etc.):\n"
        "Only when the person is emotionally stable AND the conversation has natural room for lightness.\n"
        "Never when they're in the middle of something heavy, or you haven't understood what's going on yet.\n"
        "If in doubt, just talk. Ask a question. Be present.\n"
        "When you DO use one: PERFORM it inline. Don't offer it. Don't announce it. Just do it.\n\n"
        f"Available activities (only if the moment is right):\n{activity_menu}\n"
        f"Already used this session: {', '.join(used_activities) if used_activities else 'none'}\n\n"
        "SESSION START RULE: If this is the first turn, greet warmly, say you're RIO, ask one friendly question. 2 sentences max.\n\n"
        "--- HARD RULES ---\n\n"
        f"1. NEVER use: {banned_str}\n"
        "2. Never open with 'You seem', 'It seems', 'I notice', 'I can see', 'I hear that'.\n"
        "3. 1-2 sentences. Short always beats long.\n"
        "4. Match their language (Hindi or English or mix).\n"
        "5. Never extract a name from an emotional phrase. 'I'm just very angry' means name is unknown.\n"
        "6. Never use their name if you don't know it.\n"
        "7. Never announce an activity — just do it.\n\n"
        "--- OUTPUT: valid JSON only, no markdown ---\n\n"
        "{\n"
        '  "response_text": "<your actual response — warm, present, human>",\n'
        '  "expression_intent": "<joy|sadness|calm|surprise|fear|anger>",\n'
        '  "activity_used": "<key from menu above, or null if just talking>",\n'
        '  "pending_promise": "<if you offered something not yet delivered, ~5 words, else null>",\n'
        '  "tts_params": {"pitch": <float>, "speed": <float>}\n'
        "}\n\n"
        "TTS reference:\n"
        "  sadness : pitch 0.82-0.90, speed 0.82-0.88\n"
        "  anger   : pitch 0.88-0.96, speed 0.84-0.90\n"
        "  fear    : pitch 0.90-0.96, speed 0.86-0.92\n"
        "  calm    : pitch 1.05-1.18, speed 0.98-1.05\n"
        "  joy     : pitch 1.18-1.32, speed 1.05-1.12\n"
        "  surprise: pitch 1.15-1.30, speed 1.04-1.10\n"
    )

    return Task(
        description=prompt,
        expected_output="Valid JSON: response_text, expression_intent, activity_used, pending_promise, tts_params",
        agent=agent,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_dialogue(
    stimulus: Dict[str, Any],
    intervention_intent: str,
    user_transcript: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """
    Run the dialogue agent.

    Returns:
        response_text     : str   — what RIO says
        expression_intent : str   — emotion to animate/express
        activity_used     : str   — pipeline_manager writes as [activity:tag] into memory
        pending_promise   : str|None — pipeline_manager writes as [promise:...] if not None
        tts_params        : dict  — validated pitch + speed
    """
    try:
        agent  = create_dialogue_agent()
        task   = dialogue_task(agent, stimulus, intervention_intent, user_transcript, memory_context)
        result = agent.execute_task(task)
        output_text = str(result).strip()

        # Strip markdown fences if model wrapped output anyway
        try:
            d = json.loads(output_text)
        except json.JSONDecodeError:
            for fence in ("```json", "```"):
                if fence in output_text:
                    output_text = output_text.split(fence)[1].split("```")[0].strip()
                    break
            d = json.loads(output_text)

        # Validate required keys
        required = ["response_text", "expression_intent", "tts_params"]
        if not all(k in d for k in required):
            raise ValueError(f"Missing required keys. Got: {list(d.keys())}")

        # Sanitise emotion
        if d["expression_intent"] not in VALID_EMOTIONS:
            d["expression_intent"] = "calm"

        # Validate TTS
        d["tts_params"] = _validated_tts(d.get("tts_params", {}), d["expression_intent"])

        # Defaults
        d.setdefault("activity_used", "none")
        d.setdefault("pending_promise", None)

        # Normalise null-ish pending_promise
        pp = d["pending_promise"]
        if isinstance(pp, str) and pp.strip().lower() in ("null", "none", ""):
            d["pending_promise"] = None

        # Warn if banned phrase slipped through
        low = d["response_text"].lower()
        hits = [b for b in BANNED_PHRASES if b in low]
        if hits:
            logger.warning(f"Banned phrase in response: {hits}")

        logger.info(
            f"Dialogue | emotion={d['expression_intent']} "
            f"activity={d['activity_used']} promise={d['pending_promise']}"
        )
        return d

    except Exception as e:
        logger.error(f"Dialogue agent error: {e}", exc_info=True)

        # Fallback: neutral and present, never cheerful when something went wrong
        session_start = not bool((memory_context or "").strip())
        if session_start:
            return {
                "response_text": "Hey, I'm RIO — really glad you're here. How are you doing today?",
                "expression_intent": "calm",
                "activity_used":    "none",
                "pending_promise":  None,
                "tts_params":       {"pitch": 1.10, "speed": 1.00},
            }
        return {
            "response_text": "Still here with you. What's been going on?",
            "expression_intent": "calm",
            "activity_used":    "none",
            "pending_promise":  None,
            "tts_params":       {"pitch": 1.05, "speed": 0.98},
        }
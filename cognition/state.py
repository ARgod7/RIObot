"""
state.py — RIOGraphState TypedDict (updated Day 4)

All fields that flow through the LangGraph graph live here.
Nodes read from and return a copy of this dict.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional, TypedDict


class ToneParams(TypedDict, total=False):
    warmth: float   # 0.0–1.0
    energy: float   # 0.0–1.0
    pace: str       # "slow" | "medium" | "fast"


class RIOGraphState(TypedDict, total=False):
    # ── Perception ─────────────────────────────────────────────────────────────
    stimulus: dict[str, Any]          # raw StimulusObject from emotion_fusion
    transcript: str                   # latest ASR text from voice_detector

    # ── RIO JS bridge ──────────────────────────────────────────────────────────
    rio_state: dict[str, Any]         # response from /bridge/process_stimulus
    intervention_intent: str          # e.g. "validation", "reframe", "grounding"

    # ── Dialogue (Day 4) ───────────────────────────────────────────────────────
    response_text: str                # spoken response from LLM
    tone_params: ToneParams           # TTS / prosody parameters
    expression_intent: str            # SVG face expression target
    intervention_type: str            # what class of intervention was used
    llm_provider_used: str            # "groq" | "gemini" | "error"
    dialogue_error: Optional[str]     # None on success, error string on failure

    # ── Feedback / stagnation ──────────────────────────────────────────────────
    emotional_delta: float            # post-minus-pre valence delta
    stagnation_counter: int           # increments when delta near zero
    pre_response_valence: float       # captured before dialogue node fires

    # ── Memory ─────────────────────────────────────────────────────────────────
    short_term_memory: Deque[dict]    # deque(maxlen=7), each entry is a dict

    # ── Action ─────────────────────────────────────────────────────────────────
    servo_pose: dict[str, Any]        # pan/tilt/roll targets for servo
    audio_path: Optional[str]         # path to synthesised audio file (Day 6)

    # ── Meta ───────────────────────────────────────────────────────────────────
    session_id: str
    tick: int                         # graph invocation counter


def make_initial_state(session_id: str = "default") -> RIOGraphState:
    """Return a clean starting state for a new session."""
    return RIOGraphState(
        stimulus={},
        transcript="",
        rio_state={},
        intervention_intent="general_support",
        response_text="",
        tone_params=ToneParams(warmth=0.8, energy=0.4, pace="slow"),
        expression_intent="neutral",
        intervention_type="validation",
        llm_provider_used="",
        dialogue_error=None,
        emotional_delta=0.0,
        stagnation_counter=0,
        pre_response_valence=0.0,
        short_term_memory=deque(maxlen=7),
        servo_pose={},
        audio_path=None,
        session_id=session_id,
        tick=0,
    )

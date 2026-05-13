"""
Servo Agent — RIO Embodied Motion Controller
Drives all 11 servo joints from poses.json based on emotional state.
Supports:
  - LLM/agent-chosen emotion → pose lookup
  - Right-arm "talking" gesture while speaking (TTS / mouth moving): rsv=0,
    rsh oscillates 150°–180°, re oscillates 90°–180°; left arm/head/base follow
    the emotion pose. When speech stops, servos return to that emotion pose.
  - Smooth transitions between poses
  - Full 11-joint control: head, ears, shoulders, elbows, base

Servo slot mapping (8-slot serial protocol):
  Slot 0: head horizontal (hh)
  Slot 1: right shoulder vertical (rsv)
  Slot 2: left shoulder vertical (lsv)
  Slot 3: right shoulder horizontal (rsh)
  Slot 4: left shoulder horizontal (lsh)
  Slot 5: right elbow (re)
  Slot 6: left elbow (le)
  Slot 7: base

  NOTE: hv, lear, rear are stored in poses.json but the current
  8-slot protocol has no spare slots for them. Set ENABLE_EXTENDED_SERVOS=True
  below when you upgrade to a 12-slot controller — the builder functions
  already compute them so no other changes needed.
"""

import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import serial  # type: ignore
except ImportError:
    serial = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — change these to match your setup
# ---------------------------------------------------------------------------
SERVO_COM_PORT: int = int(os.getenv("SERVO_COM_PORT", "5"))
SERVO_BAUD_RATE: int = int(os.getenv("SERVO_BAUD_RATE", "9600"))
SERVO_TIMEOUT_S: float = float(os.getenv("SERVO_TIMEOUT_S", "1"))
# Full device path overrides COM/tty defaults (e.g. /dev/ttyUSB0, /dev/serial/by-id/...).
SERVO_SERIAL_DEVICE: str = os.getenv("SERVO_SERIAL_DEVICE", "").strip()
# On Linux/macOS when SERVO_SERIAL_DEVICE is unset, try this (often /dev/ttyACM0 on Pi).
SERVO_LINUX_TTY: str = os.getenv("SERVO_LINUX_TTY", "/dev/ttyACM0").strip()

# Path to poses.json — resolved from project root
POSES_FILE: Path = Path(__file__).parent.parent.parent / "servo_controls" / "poses_generated.json"

# Set True when controller is upgraded to 12 slots (adds hv, lear, rear)
ENABLE_EXTENDED_SERVOS: bool = False

# Talking gesture (while speaking=True) — right arm only; rest from emotion pose
TALKING_TICK_S: float = 0.05
TALKING_RSV: int = 0
TALKING_RSH_LO: int = 150
TALKING_RSH_HI: int = 180
TALKING_RE_LO: int = 90
TALKING_RE_HI: int = 180
TALKING_RSH_OMEGA: float = 5.0   # oscillation speed (rad/s)
TALKING_RE_OMEGA: float = 4.1
TALKING_RE_PHASE: float = 0.65  # offset vs rsh so joints do not lockstep

# Transition step size per tick (smaller = smoother but slower)
TRANSITION_STEP: int = 4
TRANSITION_TICK_S: float = 0.04

# Emotion fallback if unknown emotion string is received
DEFAULT_EMOTION: str = "joy"

# ---------------------------------------------------------------------------
# Pose library — loaded once at import
# ---------------------------------------------------------------------------

def _load_poses(path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load and normalise poses.json. Fixes typo 'sadnesss' → 'sadness'."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        logger.error("Failed to load poses.json: %s", exc)
        return {}

    normalised: Dict[str, Dict[str, Dict[str, int]]] = {}
    for key, variants in raw.items():
        clean_key = key.strip().lower().rstrip("s") if key == "sadnesss" else key.strip().lower()
        if not isinstance(variants, dict):
            continue
        # Support both {variant: {joints}} and a flat {joints} map
        if all(not isinstance(v, dict) for v in variants.values()):
            variant_map = {"0": {k: int(v) for k, v in variants.items()}}
        else:
            variant_map = {
                str(k): {jk: int(jv) for jk, jv in v.items()}
                for k, v in variants.items()
                if isinstance(v, dict)
            }
        normalised[clean_key] = variant_map
    return normalised


POSES: Dict[str, Dict[str, Dict[str, int]]] = _load_poses(POSES_FILE)

EMOTION_ALIASES: Dict[str, str] = {
    "happy": "joy",
    "excited": "joy",
    "sad": "sadness",
    "depressed": "sadness",
    "scared": "fear",
    "anxious": "fear",
    "disgusted": "disgust",
    "angry": "anger",
    "frustrated": "anger",
    "surprised": "surprise",
    "calm": "joy",       # no neutral pose — fall to gentle joy
    "neutral": "joy",
}


def resolve_emotion(emotion: str) -> str:
    """Map any emotion string to a key that exists in POSES."""
    e = emotion.strip().lower()
    if e in POSES:
        return e
    if e in EMOTION_ALIASES and EMOTION_ALIASES[e] in POSES:
        return EMOTION_ALIASES[e]
    logger.warning("Unknown emotion '%s', falling back to '%s'", emotion, DEFAULT_EMOTION)
    return DEFAULT_EMOTION


def _intensity_to_variant(intensity: Optional[float]) -> str:
    """Map intensity in [0, 1] or [0, 4] to variant key "0"–"4"."""
    if intensity is None:
        return "0"
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        return "0"
    if value > 1.0:
        idx = int(round(value))
        return str(max(0, min(4, idx)))
    clamped = max(0.0, min(1.0, value))
    idx = int(round(clamped * 4))
    return str(idx)


def _select_pose(emotion: str, intensity: Optional[float]) -> Dict[str, int]:
    variants = POSES.get(emotion, {})
    variant_key = _intensity_to_variant(intensity)
    return dict(variants.get(variant_key) or variants.get("0") or next(iter(variants.values()), {}))


# ---------------------------------------------------------------------------
# Serial connection
# ---------------------------------------------------------------------------

_SERIAL_PORT: Optional["serial.Serial"] = None
_SERIAL_LOCK = threading.Lock()


def _serial_device_name() -> str:
    """Serial port path: SERVO_SERIAL_DEVICE, else COM{n} on Windows, else SERVO_LINUX_TTY."""
    if SERVO_SERIAL_DEVICE:
        return SERVO_SERIAL_DEVICE
    if sys.platform == "win32":
        return f"COM{SERVO_COM_PORT}"
    return SERVO_LINUX_TTY


def _get_serial_port() -> Optional["serial.Serial"]:
    global _SERIAL_PORT
    if serial is None:
        logger.warning("pyserial not installed — servo control disabled")
        return None
    with _SERIAL_LOCK:
        if _SERIAL_PORT and _SERIAL_PORT.is_open:
            return _SERIAL_PORT
        device = _serial_device_name()
        try:
            _SERIAL_PORT = serial.Serial(device, SERVO_BAUD_RATE, timeout=SERVO_TIMEOUT_S)
            logger.info("Servo serial connected on %s", device)
            return _SERIAL_PORT
        except Exception as exc:
            logger.warning("Failed to open servo serial port %s: %s", device, exc)
            return None


def close_serial():
    """Call on shutdown to cleanly close the serial port."""
    global _SERIAL_PORT
    with _SERIAL_LOCK:
        if _SERIAL_PORT and _SERIAL_PORT.is_open:
            _SERIAL_PORT.close()
            logger.info("Servo serial port closed")


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

# Canonical slot order for the 8-slot protocol
_SLOT_ORDER_8 = ["hh", "rsv", "lsv", "rsh", "lsh", "re", "le", "base"]

# Extended 12-slot order (future — when hardware supports it)
_SLOT_ORDER_12 = ["hh", "hv", "lear", "rear", "rsv", "lsv", "rsh", "lsh", "re", "le", "base", "spare"]


def _clamp(v: int) -> int:
    return max(0, min(180, int(v)))


def _build_payload(joint_values: Dict[str, int]) -> str:
    """
    Build serial payload string.
    Each slot value is divided by 10, floored, zero-padded to 2 digits.
    """
    slot_order = _SLOT_ORDER_12 if ENABLE_EXTENDED_SERVOS else _SLOT_ORDER_8
    payload = ""
    for joint in slot_order:
        raw = joint_values.get(joint, 90)
        reduced = _clamp(raw) // 10
        payload += f"{reduced:02d}"
    return payload


def _send_joints(port: "serial.Serial", joint_values: Dict[str, int]) -> bool:
    payload = _build_payload(joint_values)
    try:
        with _SERIAL_LOCK:
            port.write(payload.encode())
        logger.debug("Sent payload: %s | joints: %s", payload, joint_values)
        return True
    except Exception as exc:
        logger.warning("Serial write failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Transition engine — smoothly interpolate between two poses
# ---------------------------------------------------------------------------

def _interpolate(start: Dict[str, int], end: Dict[str, int], t: float) -> Dict[str, int]:
    """Linear interpolation between two joint dicts. t in [0, 1]."""
    all_keys = set(start) | set(end)
    return {
        k: int(start.get(k, 90) + (end.get(k, 90) - start.get(k, 90)) * t)
        for k in all_keys
    }


def _transition_to(
    port: "serial.Serial",
    from_joints: Dict[str, int],
    to_joints: Dict[str, int],
    steps: int = 20,
) -> None:
    """Smoothly move from one pose to another over `steps` ticks."""
    for i in range(1, steps + 1):
        t = i / steps
        frame = _interpolate(from_joints, to_joints, t)
        _send_joints(port, frame)
        time.sleep(TRANSITION_TICK_S)


# ---------------------------------------------------------------------------
# Talking animation — background thread while speaking (TTS)
# ---------------------------------------------------------------------------

_alive_thread: Optional[threading.Thread] = None
_alive_stop_event = threading.Event()
_current_pose: Dict[str, int] = {}
# Full emotion pose to restore when speech ends (same as when talking started)
_emotion_restore_pose: Dict[str, int] = {}
_talking_timer: Optional[threading.Timer] = None
_talking_timer_lock = threading.Lock()


def _talking_loop(port: "serial.Serial", emotion_base: Dict[str, int]) -> None:
    """
    Merge emotion pose for non-right-arm joints with a fixed talking gesture
    on the right arm: rsv=0, rsh sweeps 150–180°, re sweeps 90–180°.
    """
    global _current_pose
    rsh_mid = (TALKING_RSH_LO + TALKING_RSH_HI) / 2.0
    rsh_amp = (TALKING_RSH_HI - TALKING_RSH_LO) / 2.0
    re_mid = (TALKING_RE_LO + TALKING_RE_HI) / 2.0
    re_amp = (TALKING_RE_HI - TALKING_RE_LO) / 2.0
    t = 0.0
    while not _alive_stop_event.is_set():
        frame = dict(emotion_base)
        frame["rsv"] = TALKING_RSV
        frame["rsh"] = _clamp(
            int(round(rsh_mid + rsh_amp * math.sin(t * TALKING_RSH_OMEGA)))
        )
        frame["re"] = _clamp(
            int(round(re_mid + re_amp * math.sin(t * TALKING_RE_OMEGA + TALKING_RE_PHASE)))
        )
        _send_joints(port, frame)
        _current_pose = frame
        time.sleep(TALKING_TICK_S)
        t += TALKING_TICK_S


def start_alive_animation(emotion: str, intensity: Optional[float] = None) -> None:
    """
    Start the talking right-arm animation for the given emotion pose.
    Call when TTS / mouth movement begins. ``intensity`` maps 0–1 to
    pose variants 0–4.
    """
    global _alive_thread, _emotion_restore_pose
    stop_alive_animation()

    port = _get_serial_port()
    if port is None:
        return

    resolved = resolve_emotion(emotion)
    base_pose = _select_pose(resolved, intensity)
    if not base_pose:
        return

    _emotion_restore_pose = dict(base_pose)
    _alive_stop_event.clear()
    _alive_thread = threading.Thread(
        target=_talking_loop, args=(port, base_pose), daemon=True
    )
    _alive_thread.start()
    logger.info("Talking animation started for emotion: %s", resolved)


def stop_alive_animation() -> None:
    """Stop talking animation and return to the emotion pose held when speech began."""
    global _alive_thread, _current_pose, _emotion_restore_pose
    _alive_stop_event.set()
    if _alive_thread and _alive_thread.is_alive():
        _alive_thread.join(timeout=2.0)
    _alive_thread = None

    port = _get_serial_port()
    if port and _emotion_restore_pose:
        target = dict(_emotion_restore_pose)
        if _current_pose:
            _transition_to(port, _current_pose, target)
        else:
            _send_joints(port, target)
        _current_pose = target
        _emotion_restore_pose.clear()
    elif _emotion_restore_pose:
        # No serial — still clear bookkeeping
        _emotion_restore_pose.clear()

    logger.info("Talking animation stopped")


def start_talking_for_text(
    emotion: str,
    text: str,
    intensity: Optional[float] = None,
    speed: float = 1.0,
    wpm: int = 150,
) -> None:
    """Start talking animation and auto-stop after estimated speech duration."""
    if not text or not text.strip():
        return
    safe_speed = max(0.6, min(1.6, float(speed or 1.0)))
    words = len(text.split())
    duration_s = max(1.2, (words / (wpm * safe_speed)) * 60.0)

    start_alive_animation(emotion, intensity=intensity)
    with _talking_timer_lock:
        global _talking_timer
        if _talking_timer and _talking_timer.is_alive():
            _talking_timer.cancel()
        _talking_timer = threading.Timer(duration_s, stop_alive_animation)
        _talking_timer.daemon = True
        _talking_timer.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_servo(
    emotion: str,
    speaking: bool = False,
    transition: bool = True,
    intensity: Optional[float] = None,
) -> Dict[str, str]:
    """
    Main entry point for the servo agent.

    Args:
        emotion:    Emotion string — any Ekman emotion or alias.
                    The agent resolves it to the closest pose in poses.json.
        speaking:   If True, starts the talking right-arm animation (rsv=0,
                    rsh 150–180°, re 90–180°) while keeping the rest of the body
                    on the emotion pose. If False, holds the pose still.
        transition: If True, smoothly interpolates from current pose to new pose.
        intensity:  Optional; maps 0–1 to pose variants 0–4.

    Returns:
        dict with 'action', 'emotion', 'resolved_emotion', 'payload'
    """
    global _current_pose

    resolved = resolve_emotion(emotion)
    target_pose = _select_pose(resolved, intensity)

    if not target_pose:
        logger.warning("No pose found for emotion '%s'", resolved)
        return {"action": "stub", "emotion": emotion, "resolved_emotion": resolved, "payload": ""}

    port = _get_serial_port()
    if port is None:
        return {"action": "stub", "emotion": emotion, "resolved_emotion": resolved, "payload": ""}

    # Transition smoothly if we have a previous pose
    if transition and _current_pose:
        _transition_to(port, _current_pose, target_pose)
    else:
        _send_joints(port, target_pose)

    _current_pose = target_pose
    payload = _build_payload(target_pose)

    if speaking:
        start_alive_animation(resolved, intensity=intensity)
    else:
        stop_alive_animation()
        _send_joints(port, target_pose)  # hold still

    logger.info(
        "Pose set: %s → %s | speaking=%s | intensity=%s | payload=%s",
        emotion,
        resolved,
        speaking,
        intensity,
        payload,
    )
    return {
        "action": "pose_set",
        "emotion": emotion,
        "resolved_emotion": resolved,
        "payload": payload,
    }


def neutral() -> Dict[str, str]:
    """Return robot to neutral (all joints 90°)."""
    global _current_pose
    stop_alive_animation()
    neutral_pose = {j: 90 for j in ["hh", "hv", "lear", "rear", "rsv", "lsv", "rsh", "lsh", "re", "le", "base"]}
    port = _get_serial_port()
    if port:
        if _current_pose:
            _transition_to(port, _current_pose, neutral_pose)
        else:
            _send_joints(port, neutral_pose)
        _current_pose = dict(neutral_pose)
    else:
        _current_pose = dict(neutral_pose)
    return {"action": "neutral", "emotion": "neutral", "payload": _build_payload(neutral_pose)}


# ---------------------------------------------------------------------------
# CrewAI tool wrapper
# ---------------------------------------------------------------------------

def get_crewai_tool():
    """
    Returns a CrewAI-compatible Tool that agents can call.

    Usage in your crew:
        from servo_agent import get_crewai_tool
        servo_tool = get_crewai_tool()
    """
    try:
        from crewai.tools import tool  # type: ignore

        @tool("ServoMotionTool")
        def servo_motion_tool(
            emotion: str, speaking: bool = False, intensity: Optional[float] = None
        ) -> str:
            """
            Drive RIO's servo motors to match an emotional state.
            Use this tool whenever RIO starts speaking or transitions emotion.

            Args:
                emotion:  The emotion to express. One of: joy, sadness, fear,
                          disgust, anger, surprise, calm, neutral, happy, excited,
                          sad, scared, anxious, disgusted, angry, surprised, frustrated.
                speaking: Set to True while TTS is playing so the right arm uses
                    the talking gesture; False to hold the pose still.

            Returns:
                JSON string describing the action taken.
            """
            result = run_servo(emotion=emotion, speaking=speaking, intensity=intensity)
            return json.dumps(result)

        return servo_motion_tool

    except ImportError:
        logger.warning("CrewAI not installed — returning plain callable instead")
        return run_servo


# ---------------------------------------------------------------------------
# Quick test (run directly: python servo_agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loaded poses:", list(POSES.keys()))
    print()

    emotions_to_test = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]
    for em in emotions_to_test:
        print(f"→ Testing pose: {em}")
        result = run_servo(em, speaking=True, transition=True)
        print(f"  Result: {result}")
        time.sleep(3)
        stop_alive_animation()
        time.sleep(0.5)

    print("→ Returning to neutral")
    neutral()
    print("Done.")

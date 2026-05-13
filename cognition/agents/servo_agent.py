"""
Servo Agent — RIO Embodied Motion Controller
Drives all 11 servo joints from poses_generated.json based on emotional state.
Supports:
  - LLM/agent-chosen emotion → pose lookup
  - Micro-movement "alive" animation while speaking
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

  NOTE: hv, lear, rear are stored in poses files but the current
  8-slot protocol has no spare slots for them. Set ENABLE_EXTENDED_SERVOS=True
  below when you upgrade to a 12-slot controller — the builder functions
  already compute them so no other changes needed.
"""

import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Path to poses files — resolved from project root
POSES_FILE: Path = Path(__file__).parent.parent.parent / "servo_controls" / "poses_generated.json"
POSES_FALLBACK_FILE: Path = Path(__file__).parent.parent.parent / "servo_controls" / "poses.json"

# Set True when controller is upgraded to 12 slots (adds hv, lear, rear)
ENABLE_EXTENDED_SERVOS: bool = False

# Micro-movement config while speaking
ALIVE_INTERVAL_S: float = 0.8        # how often to nudge joints
ALIVE_AMPLITUDE: int = 6             # max degrees of random sway
ALIVE_JOINTS = ["rsv", "lsv", "rsh", "lsh", "re", "le", "hh"]
ALIVE_JOINT_AMPLITUDE = {
    "hh": 6,
    "rsv": 5,
    "lsv": 5,
    "rsh": 4,
    "lsh": 4,
    "re": 3,
    "le": 3,
}

# Transition step size per tick (smaller = smoother but slower)
TRANSITION_STEP: int = 4
TRANSITION_TICK_S: float = 0.04

# Emotion fallback if unknown emotion string is received
DEFAULT_EMOTION: str = "joy"

# ---------------------------------------------------------------------------
# Pose library — loaded once at import
# ---------------------------------------------------------------------------

def _load_poses(path: Path) -> Dict[str, Dict[int, Dict[str, int]]]:
    """Load and normalise poses with intensity variants (0-4)."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        logger.error("Failed to load poses from %s: %s", path, exc)
        return {}

    normalised: Dict[str, Dict[int, Dict[str, int]]] = {}
    for key, variants in raw.items():
        clean_key = "sadness" if key.strip().lower() == "sadnesss" else key.strip().lower()
        variant_map: Dict[int, Dict[str, int]] = {}
        for variant_key, variant_data in (variants or {}).items():
            try:
                idx = int(variant_key)
            except (TypeError, ValueError):
                continue
            variant_map[idx] = {k: int(v) for k, v in (variant_data or {}).items()}

        if not variant_map and isinstance(variants, dict):
            # Single-frame file fallback: treat entire dict as intensity 0
            variant_map[0] = {k: int(v) for k, v in variants.items()}

        if variant_map:
            normalised[clean_key] = variant_map
    return normalised


POSES: Dict[str, Dict[int, Dict[str, int]]] = _load_poses(POSES_FILE) or _load_poses(POSES_FALLBACK_FILE)

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


def _coerce_intensity(intensity: Optional[int]) -> int:
    if intensity is None:
        return 2
    try:
        return max(0, min(4, int(intensity)))
    except (TypeError, ValueError):
        return 2


def _pick_pose(emotion: str, intensity: Optional[int]) -> Dict[str, int]:
    intensity_idx = _coerce_intensity(intensity)
    variants = POSES.get(emotion, {})
    if not variants:
        return {}
    if intensity_idx in variants:
        return dict(variants[intensity_idx])
    closest = min(variants.keys(), key=lambda k: abs(k - intensity_idx))
    return dict(variants[closest])


# ---------------------------------------------------------------------------
# Serial connection
# ---------------------------------------------------------------------------

_SERIAL_PORT: Optional["serial.Serial"] = None
_SERIAL_LOCK = threading.Lock()


def _get_serial_port() -> Optional["serial.Serial"]:
    global _SERIAL_PORT
    if serial is None:
        logger.warning("pyserial not installed — servo control disabled")
        return None
    with _SERIAL_LOCK:
        if _SERIAL_PORT and _SERIAL_PORT.is_open:
            return _SERIAL_PORT
        try:
            _SERIAL_PORT = serial.Serial(
                f"COM{SERVO_COM_PORT}", SERVO_BAUD_RATE, timeout=SERVO_TIMEOUT_S
            )
            logger.info("Servo serial connected on COM%s", SERVO_COM_PORT)
            return _SERIAL_PORT
        except Exception as exc:
            logger.warning("Failed to open servo serial port: %s", exc)
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
# Alive micro-movement — runs in a background thread while speaking
# ---------------------------------------------------------------------------

_alive_thread: Optional[threading.Thread] = None
_alive_stop_event = threading.Event()
_current_pose: Dict[str, int] = {}


def _alive_loop(port: "serial.Serial", base_pose: Dict[str, int]) -> None:
    """
    Holds the emotion pose but adds subtle sinusoidal sway to selected joints
    so the robot looks alive while speaking.
    """
    t = 0.0
    while not _alive_stop_event.is_set():
        nudged = dict(base_pose)
        for i, joint in enumerate(ALIVE_JOINTS):
            if joint in nudged:
                # Each joint gets a slightly different phase so they don't all move together
                phase = i * (math.pi / len(ALIVE_JOINTS))
                amp = ALIVE_JOINT_AMPLITUDE.get(joint, ALIVE_AMPLITUDE)
                sway = int(amp * math.sin(t * 1.5 + phase))
                nudged[joint] = _clamp(nudged[joint] + sway)
        _send_joints(port, nudged)
        t += ALIVE_INTERVAL_S
        time.sleep(ALIVE_INTERVAL_S)


def start_alive_animation(emotion: str, intensity: Optional[int] = None) -> None:
    """
    Start the background alive animation for the given emotion pose.
    Call this when TTS starts speaking.
    """
    global _alive_thread, _current_pose
    stop_alive_animation()  # stop any running animation first

    port = _get_serial_port()
    if port is None:
        return

    resolved = resolve_emotion(emotion)
    base_pose = _pick_pose(resolved, intensity)
    if not base_pose:
        return

    _current_pose = base_pose
    _alive_stop_event.clear()
    _alive_thread = threading.Thread(
        target=_alive_loop, args=(port, base_pose), daemon=True
    )
    _alive_thread.start()
    logger.info("Alive animation started for emotion: %s", resolved)


def stop_alive_animation() -> None:
    """Stop the alive background animation. Call this when TTS finishes."""
    global _alive_thread
    _alive_stop_event.set()
    if _alive_thread and _alive_thread.is_alive():
        _alive_thread.join(timeout=2.0)
    _alive_thread = None
    logger.info("Alive animation stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_servo(
    emotion: str,
    speaking: bool = False,
    transition: bool = True,
    intensity: Optional[int] = None,
) -> Dict[str, str]:
    """
    Main entry point for the servo agent.

    Args:
        emotion:    Emotion string — any Ekman emotion or alias.
                    The agent resolves it to the closest pose in poses files.
        speaking:   If True, starts the alive micro-movement animation.
                    If False, just holds the pose.
        transition: If True, smoothly interpolates from current pose to new pose.

    Returns:
        dict with 'action', 'emotion', 'resolved_emotion', 'payload'
    """
    global _current_pose

    resolved = resolve_emotion(emotion)
    target_pose = _pick_pose(resolved, intensity)

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
        "Pose set: %s → %s | intensity=%s | speaking=%s | payload=%s",
        emotion,
        resolved,
        _coerce_intensity(intensity),
        speaking,
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
    stop_alive_animation()
    neutral_pose = {j: 90 for j in ["hh", "hv", "lear", "rear", "rsv", "lsv", "rsh", "lsh", "re", "le", "base"]}
    port = _get_serial_port()
    if port:
        if _current_pose:
            _transition_to(port, _current_pose, neutral_pose)
        else:
            _send_joints(port, neutral_pose)
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
        def servo_motion_tool(emotion: str, speaking: bool = False, intensity: int = 2) -> str:
            """
            Drive RIO's servo motors to match an emotional state.
            Use this tool whenever RIO starts speaking or transitions emotion.

            Args:
                emotion:  The emotion to express. One of: joy, sadness, fear,
                          disgust, anger, surprise, calm, neutral, happy, excited,
                          sad, scared, anxious, disgusted, angry, surprised, frustrated.
                emotion:  Set to True when TTS is actively playing so the robot
                speaking: does subtle alive movements. False to hold pose still.

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
        result = run_servo(em, speaking=True, transition=True, intensity=2)
        print(f"  Result: {result}")
        time.sleep(3)
        stop_alive_animation()
        time.sleep(0.5)

    print("→ Returning to neutral")
    neutral()
    print("Done.")
"""
Servo Agent — Serial motor control based on expression intent.

Uses the same packet protocol as test.py/test2.py:
- 8 servo slots
- each angle reduced via integer division by 10
- each reduced value padded to 2 digits
- concatenated payload sent over serial
"""

import logging
import os
import time
from typing import Dict, Optional

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None

logger = logging.getLogger(__name__)

_SERIAL_PORT: Optional["serial.Serial"] = None

SERVO_COM_PORT: int = int(os.getenv("SERVO_COM_PORT", "5"))
SERVO_BAUD_RATE: int = int(os.getenv("SERVO_BAUD_RATE", "9600"))
SERVO_TIMEOUT_S: float = float(os.getenv("SERVO_TIMEOUT_S", "1"))


def _clamp_angle(angle: int) -> int:
    """Clamp angle into servo-safe [0, 180] range."""
    return max(0, min(180, int(angle)))


def _build_payload(pose: list[int]) -> str:
    """Build controller payload string from 8-servo pose."""
    send_data = ""
    for value in pose:
        reduced_value = _clamp_angle(value) // 10
        send_data += f"{reduced_value:02d}"
    return send_data


def _get_serial_port() -> Optional["serial.Serial"]:
    """Get or initialize persistent serial connection."""
    global _SERIAL_PORT

    if serial is None:
        logger.warning("pyserial not installed; servo control disabled")
        return None

    if _SERIAL_PORT and _SERIAL_PORT.is_open:
        return _SERIAL_PORT

    try:
        _SERIAL_PORT = serial.Serial(
            f"COM{SERVO_COM_PORT}",
            SERVO_BAUD_RATE,
            timeout=SERVO_TIMEOUT_S,
        )
        logger.info("Servo serial connected on COM%s", SERVO_COM_PORT)
        return _SERIAL_PORT
    except Exception as exc:
        logger.warning("Failed to open servo serial port: %s", exc)
        return None


def _send_pose(
    port: "serial.Serial",
    rsv: int = 90,
    lsv: int = 90,
    rsh: int = 90,
    lsh: int = 90,
    re: int = 90,
    le: int = 90,
) -> bool:
    """
    Send full 8-servo pose using the arm mapping from test2.py.

    Servo slots:
    - 1: right shoulder vertical (rsv)
    - 2: left shoulder vertical (lsv)
    - 3: right shoulder horizontal (rsh)
    - 4: left shoulder horizontal (lsh)
    - 5: right elbow (re)
    - 6: left elbow (le)
    """
    pose = [90] * 8
    pose[1] = rsv
    pose[2] = lsv
    pose[3] = rsh
    pose[4] = lsh
    pose[5] = re
    pose[6] = le

    payload = _build_payload(pose)

    try:
        port.write(payload.encode())
        return True
    except Exception as exc:
        logger.warning("Servo write failed: %s", exc)
        return False


def _smooth_wave(port: "serial.Serial", cycles: int = 2, delay_s: float = 0.45, amplitude: float = 1.0) -> bool:
    """Run bilateral arm wave sequence from test2.py style."""
    try:
        amp = max(0.3, min(1.0, amplitude))
        shoulder_up = int(90 + (60 * amp))
        shoulder_down = int(90 - (60 * amp))
        elbow_out_r = int(90 - (90 * amp))
        elbow_out_l = int(90 + (90 * amp))

        ok = _send_pose(
            port,
            rsv=shoulder_up,
            lsv=shoulder_down,
            rsh=90,
            lsh=90,
            re=90,
            le=90,
        )
        if not ok:
            return False
        time.sleep(0.6)

        for _ in range(cycles):
            if not _send_pose(
                port,
                rsv=shoulder_up,
                lsv=shoulder_down,
                rsh=90,
                lsh=90,
                re=elbow_out_r,
                le=elbow_out_l,
            ):
                return False
            time.sleep(delay_s)
            if not _send_pose(
                port,
                rsv=shoulder_up,
                lsv=shoulder_down,
                rsh=90,
                lsh=90,
                re=90,
                le=90,
            ):
                return False
            time.sleep(delay_s)

        _send_pose(port, rsv=90, lsv=90, rsh=90, lsh=90, re=90, le=90)
        return True
    except Exception as exc:
        logger.warning("Wave action failed: %s", exc)
        return False


def run_servo(expression_intent: str, emotion_vector: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """
    Drive servo behavior from expression intent.

    - joy/surprise: short wave
    - sadness/fear: low-energy posture
    - anger: guarded posture
    - calm/default: neutral posture
    """
    intent = (expression_intent or "calm").strip().lower()
    emotion_vector = emotion_vector or {}
    intensity = float(max(emotion_vector.values(), default=0.5))
    intensity = max(0.2, min(1.0, intensity))
    port = _get_serial_port()
    if port is None:
        return {"action": "stub", "intent": intent}

    try:
        if intent in {"joy", "surprise"}:
            cycles = 2 if intensity < 0.7 else 3
            delay_s = max(0.22, 0.55 - (0.25 * intensity))
            success = _smooth_wave(port, cycles=cycles, delay_s=delay_s, amplitude=intensity)
            return {"action": "wave" if success else "stub", "intent": intent}

        if intent in {"sadness", "fear"}:
            droop = int(20 + (30 * intensity))
            elbow_fold = int(100 + (40 * intensity))
            _send_pose(
                port,
                rsv=90 + droop,
                lsv=90 - droop,
                rsh=90,
                lsh=90,
                re=elbow_fold,
                le=elbow_fold,
            )
            return {"action": "comfort_pose", "intent": intent}

        if intent == "anger":
            spread = int(12 + (28 * intensity))
            elbow_tension = int(80 - (30 * intensity))
            _send_pose(
                port,
                rsv=90 + spread,
                lsv=90 - spread,
                rsh=90 + spread,
                lsh=90 - spread,
                re=elbow_tension,
                le=180 - elbow_tension,
            )
            return {"action": "guard_pose", "intent": intent}

        _send_pose(port, rsv=90, lsv=90, rsh=90, lsh=90, re=90, le=90)
        return {"action": "neutral_pose", "intent": intent}
    except Exception as exc:
        logger.warning("Servo action failed for intent '%s': %s", intent, exc)
        return {"action": "stub", "intent": intent}


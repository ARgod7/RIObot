"""
rio_bridge/rio_client.py
Async Python client that calls the real RIO JS engine (server.js) via HTTP.
All cognition layer code imports from here — never calls the bridge directly.
"""

import asyncio
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger(__name__)

# New LLM emotional engine runs on port 3001 (server.js)
# Old facial engine runs on port 5000 (index.js)
RIO_BRIDGE_URL = "http://0.0.0.0:3001"
RIO_BRIDGE_TIMEOUT = 5.0
RIO_BRIDGE_RETRIES = 3


class RIOBridgeError(Exception):
    """Raised when the RIO JS bridge returns an error or is unreachable."""
    pass


class RIOClient:
    """
    Async client for the RIO JS emotional engine bridge.
    Use as a singleton — one instance per app lifecycle.

    Usage:
        rio = RIOClient()
        await rio.check_health()  # call at startup
        result = await rio.process_stimulus(fused_vector, speaker="user")
    """

    def __init__(self):
        self._base = RIO_BRIDGE_URL
        self._timeout = RIO_BRIDGE_TIMEOUT
        # httpx AsyncClient is reused across calls for connection pooling
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ─────────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────────

    async def check_health(self) -> dict:
        """
        Call at app startup to confirm RIO JS bridge is running.
        Raises RIOBridgeError if unreachable.
        """
        client = await self._get_client()
        try:
            r = await client.get(f"{self._base}/bridge/health")
            r.raise_for_status()
            data = r.json()
            print(f"[RIO] Bridge healthy — engine_started={data.get('engine_started')}, "
                  f"port={data.get('port')}")
            return data
        except httpx.ConnectError:
            raise RIOBridgeError(
                f"Cannot reach RIO bridge at {self._base}. "
                f"Start it with: cd rio_js && node server.js"
            )
        except Exception as e:
            raise RIOBridgeError(f"RIO health check failed: {e}")

    # ─────────────────────────────────────────────
    # PROCESS STIMULUS
    # Called every perception loop cycle with the fused emotion vector
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(RIO_BRIDGE_RETRIES),
        wait=wait_fixed(0.2),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def process_stimulus(
        self,
        stimulus_vector: dict,
        speaker: str = "user",
        likeness: float | None = None,
        label_prefix: str = "perception_stimulus",
    ) -> dict:
        """
        Send fused emotion vector to RIO engine.
        Returns dict with keys: rio_state, intervention_intent

        stimulus_vector must be:
        {
            "joy": float,      # 0.0–1.0
            "sadness": float,
            "fear": float,
            "disgust": float,
            "anger": float,
            "surprise": float
        }
        """
        self._validate_vector(stimulus_vector)

        payload = {
            "stimulus_vector": stimulus_vector,
            "speaker": speaker,
            "label_prefix": label_prefix,
        }
        if likeness is not None:
            payload["likeness"] = float(likeness)

        client = await self._get_client()
        try:
            r = await client.post(f"{self._base}/bridge/process_stimulus", json=payload)
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                raise RIOBridgeError(f"RIO process_stimulus error: {data.get('error')}")
            return {
                "rio_state": data["rio_state"],
                "intervention_intent": data["intervention_intent"],
            }
        except httpx.TransportError as e:
            raise  # let tenacity retry
        except RIOBridgeError:
            raise
        except Exception as e:
            raise RIOBridgeError(f"process_stimulus failed: {e}")

    # ─────────────────────────────────────────────
    # VERBAL INPUT
    # Called when user speaks — runs full NLU + emotion pipeline
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(RIO_BRIDGE_RETRIES),
        wait=wait_fixed(0.2),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def verbal_input(
        self,
        text: str,
        speaker: str = "user",
    ) -> dict:
        """
        Send user's transcribed speech to RIO engine for full processing.
        Returns:
        {
            "agent_response": str,    # RIO's decided response text
            "emotions_after": dict,   # 6D vector after processing
            "face_hint": str,         # e.g. "sad", "joy_big"
            "pitch": float,
            "rate": float
        }
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        client = await self._get_client()
        try:
            r = await client.post(
                f"{self._base}/bridge/verbal_input",
                json={"text": text, "speaker": speaker},
                timeout=10.0,   # verbal processing can be slower (LLM call inside)
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                raise RIOBridgeError(f"RIO verbal_input error: {data.get('error')}")
            return {
                "agent_response": data.get("agent_response", ""),
                "emotions_after": data.get("emotions_after", {}),
                "face_hint": data.get("face_hint", "neutral"),
                "pitch": data.get("pitch", 1.0),
                "rate": data.get("rate", 1.0),
            }
        except httpx.TransportError:
            raise
        except RIOBridgeError:
            raise
        except Exception as e:
            raise RIOBridgeError(f"verbal_input failed: {e}")

    # ─────────────────────────────────────────────
    # FEEDBACK
    # Called by feedback_node after measuring emotional delta
    # ─────────────────────────────────────────────

    async def send_feedback(
        self,
        emotional_delta: float,
        intervention_succeeded: bool,
        speaker: str = "user",
    ) -> dict:
        """
        Update RIO's trust/likeness based on intervention outcome.
        emotional_delta: positive = user improved, negative = worsened.
        """
        client = await self._get_client()
        try:
            r = await client.post(
                f"{self._base}/bridge/feedback",
                json={
                    "emotional_delta": float(emotional_delta),
                    "intervention_succeeded": bool(intervention_succeeded),
                    "speaker": speaker,
                },
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            # Feedback failure is non-critical — log and continue
            print(f"[RIO] feedback update failed (non-critical): {e}")
            return {}

    # ─────────────────────────────────────────────
    # GET CURRENT STATE (for debug panel)
    # ─────────────────────────────────────────────

    async def get_state(self) -> dict:
        """Returns current RIO emotional state — used by debug panel."""
        client = await self._get_client()
        try:
            r = await client.get(f"{self._base}/state")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _validate_vector(vec: dict):
        required = ["joy", "sadness", "fear", "disgust", "anger", "surprise"]
        for k in required:
            v = vec.get(k)
            if not isinstance(v, (int, float)):
                raise ValueError(f"stimulus_vector.{k} must be a number, got {type(v)}")
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"stimulus_vector.{k} must be 0.0–1.0, got {v}")

    @staticmethod
    def dominant_emotion(emotion_vector: dict) -> tuple[str, float]:
        """Returns (emotion_name, value) for the strongest emotion."""
        if not emotion_vector:
            return ("neutral", 0.0)
        return max(emotion_vector.items(), key=lambda x: x[1])


# ─────────────────────────────────────────────
# SINGLETON — import this in all other modules
# ─────────────────────────────────────────────
rio_client = RIOClient()


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to test the bridge
# python -m rio_bridge.rio_client
# ─────────────────────────────────────────────
async def _test():
    print("\n── RIO Client Test ──")
    try:
        await rio_client.check_health()

        print("\n[Test 1] process_stimulus with sadness-heavy vector")
        result = await rio_client.process_stimulus({
            "joy": 0.1, "sadness": 0.75, "fear": 0.2,
            "disgust": 0.0, "anger": 0.1, "surprise": 0.05
        })
        print(f"  Dominant: {result['rio_state']['dominant_emotion']}")
        print(f"  Intent:   {result['intervention_intent']}")
        print(f"  Stagnating: {result['rio_state']['is_stagnating']}")

        print("\n[Test 2] verbal_input")
        vresult = await rio_client.verbal_input("I feel so lonely today")
        print(f"  Agent response: {vresult['agent_response']}")
        print(f"  Face hint: {vresult['face_hint']}")

        print("\n[Test 3] feedback")
        await rio_client.send_feedback(emotional_delta=0.1, intervention_succeeded=True)
        print("  Feedback sent OK")

        print("\n✓ All tests passed")
    except RIOBridgeError as e:
        print(f"\n✗ Bridge error: {e}")
    finally:
        await rio_client.close()


if __name__ == "__main__":
    asyncio.run(_test())


# ─────────────────────────────────────────────
# SYNCHRONOUS WRAPPERS for main.py compatibility
# ─────────────────────────────────────────────

def _sanitize(d: dict) -> dict:
    """Replace None values with 0.0 for numeric emotion fields."""
    emotion_keys = {"joy","sadness","fear","disgust","anger","surprise"}
    result = {}
    for k, v in d.items():
        if k in emotion_keys:
            result[k] = float(v) if v is not None else 0.0
        elif isinstance(v, dict):
            result[k] = _sanitize(v)
        else:
            result[k] = v
    return result


def call_rio_engine(stimulus_dict: dict) -> dict:
    """
    POST stimulus to RIO engine, return rio_state + intervention_intent.
    
    Reformats StimulusObject to the exact JSON shape expected by Node.js server:
    {
        "stimulus_vector": { "joy": 0.0, "sadness": 0.0, ... },
        "trust": 0.7,
        "likeness": 0.7
    }

    Args:
        stimulus_dict: StimulusObject dict with "emotions" and other fields

    Returns:
        Dict with rio_state and intervention_intent
    """
    try:
        # Extract emotions from the StimulusObject
        emotions = stimulus_dict.get("emotions", {})
        
        # Build the payload in the exact format Node.js server expects
        payload = {
            "stimulus_vector": {
                "joy":      float(emotions.get("joy", 0.0)),
                "sadness":  float(emotions.get("sadness", 0.0)),
                "fear":     float(emotions.get("fear", 0.0)),
                "disgust":  float(emotions.get("disgust", 0.0)),
                "anger":    float(emotions.get("anger", 0.0)),
                "surprise": float(emotions.get("surprise", 0.0)),
            },
            "trust":    float(stimulus_dict.get("trust", 0.7)),
            "likeness": float(stimulus_dict.get("likeness", 0.7)),
        }
        
        # POST to Node.js RIO engine via the bridge
        r = httpx.post(f"{RIO_BRIDGE_URL}/process_stimulus", json=payload, timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"RIO engine call failed: {e}")
        return {"rio_state": {}, "intervention_intent": "validation"}


def send_expression(expression_intent: str) -> None:
    """
    Send expression intent to RIO engine.
    Note: index.js does not have a dedicated /expression endpoint.
    This is a placeholder for future expression handling.

    Args:
        expression_intent: Emotion expression (joy, sadness, calm, surprise, fear, anger)
    """
    try:
        # Note: Currently no /expression endpoint in index.js
        # This could be handled via /stimulate endpoint if needed
        client = httpx.Client(timeout=2.0)
        # For now, this is non-critical and can be expanded later
        client.close()
    except Exception as e:
        logger.debug(f"Expression send failed (non-critical): {e}")


def send_audio_to_browser(filename: str) -> None:
    """
    Send audio file to browser for playback via socket.io on port 3001.
    Makes POST request to http://0.0.0.0:3001/play-audio
    
    Args:
        filename: Relative path to audio file (e.g., "audio/response.mp3")
    """
    try:
        url = f"{RIO_BRIDGE_URL}/play-audio"
        payload = {"file": filename}
        
        client = httpx.Client(timeout=2.0)
        response = client.post(url, json=payload)
        response.raise_for_status()
        
        logger.debug(f"✓ Audio sent to browser: {filename}")
        client.close()
    except Exception as e:
        logger.warning(f"Failed to send audio to browser: {e}")




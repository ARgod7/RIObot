import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LEGACY API KEYS (OPTIONAL - not used in CrewAI architecture)
# ============================================================
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

GROQ_MODEL: str = "llama-3.3-70b-versatile"
GEMINI_MODEL: str = "gemini-2.5-flash"

# ============================================================
# LLM PROVIDER TOGGLE (Easy switch between Groq, Gemini, Ollama)
# ============================================================
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" | "gemini" | "ollama"

# Groq API (FREE - Fast, Cloud) - uncomment if available
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.1-70b-versatile"  # Fallback if switching back to Groq

# Gemini API (Cloud)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash"

# Ollama (FREE - Local, No API, Always Works!)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://0.0.0.0:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b")  # ~4.1GB, bilingual Hindi/English
OLLAMA_TIMEOUT: float = 10.0  # Ollama needs a bit more time
OLLAMA_STREAM: bool = False

RIO_PORT: int = int(os.getenv("RIO_PORT", "5000"))
RIO_BASE_URL: str = f"http://0.0.0.0:{RIO_PORT}"

# Rio Bridge (JS engine) — used by rio_client.py
RIO_BRIDGE_URL: str = RIO_BASE_URL          # http://0.0.0.0:{RIO_PORT}
RIO_BRIDGE_TIMEOUT: float = 5.0
RIO_BRIDGE_RETRIES: int = 3
DIALOGUE_TIMEOUT_S: float = 12.0
MAX_TOKENS_DIALOGUE: int = 256


# ============================================================
# LOGGING & STATUS OUTPUT
# ============================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")
FUSED_EMOTION_LOG_INTERVAL: float = float(os.getenv("FUSED_EMOTION_LOG_INTERVAL", "5"))
FUSED_EMOTION_LOG_ENABLED: bool = os.getenv("FUSED_EMOTION_LOG_ENABLED", "true").lower() == "true"

PERCEPTION_FPS: int = int(os.getenv("PERCEPTION_FPS", "8"))  # lower = calmer UI, less CPU

# Perception loop
PERCEPTION_LOOP_INTERVAL: float = 1.0 / PERCEPTION_FPS
DETECTOR_CONFIDENCE_THRESHOLD: float = 0.4

# Only update the emotion stimulus seen by the LLM/engine every N perception frames.
# Lowered 30 → 12 (at 8fps = ~1.5s refresh). The old value meant ~3.75s between
# LLM updates, which compounded with the 5s face sample interval to create very
# sticky emotion readings. 12 frames = ~1.5s which is responsive without being noisy.
EMOTION_INPUT_FRAME_INTERVAL: int = int(os.getenv("EMOTION_INPUT_FRAME_INTERVAL", "12"))

# Face detector sampling (seconds between fresh face emotion reads).
# Lowered 5s → 1.5s so a changed expression updates the fused vector
# within a couple of seconds instead of being stale for up to 5s.
FACE_SAMPLE_INTERVAL_S: float = float(os.getenv("FACE_SAMPLE_INTERVAL_S", "1.5"))

# Fusion smoothing: lower alpha = smoother, slower-moving fused vector (see EmotionFusionEngine.fuse_smoothed)
# Raised 0.28 → 0.55 so the fused vector responds to real expression changes within ~1-2s.
# At 0.28 the vector was so sluggish that any emotion (especially sadness) became sticky.
# If the output feels too jumpy, lower toward 0.40; don't go below 0.35.
FUSION_SMOOTH_ALPHA: float = float(os.getenv("FUSION_SMOOTH_ALPHA", "0.55"))

# Min seconds between emotion WebSocket pushes from main.py (details dashboard)
EMOTION_WS_BROADCAST_MIN_S: float = float(os.getenv("EMOTION_WS_BROADCAST_MIN_S", "0.15"))

# ── Emotion expressiveness tuning (lower = calmer / subtler) ─────────────────
# Perception: blend fused emotion scores toward a flat mix (1.0 = detectors as-is,
# 0.5 = halfway toward uniform, 0.0 = fully uniform — rarely useful).
EMOTION_PERCEPTION_DAMPING: float = float(os.getenv("EMOTION_PERCEPTION_DAMPING", "0.75"))

# Dialogue: scale the dominant-value used for arc instructions & prompt intensity (0–1).
# <1.0 makes the model treat states as milder (fewer “high distress” arc branches).
EMOTION_ARC_INTENSITY_SCALE: float = float(os.getenv("EMOTION_ARC_INTENSITY_SCALE", "0.8"))

# TTS: 1.0 = full pitch/speed ranges; 0.0 = locked to calm reference (pitch≈1.0, speed≈0.95).
TTS_EXPRESSIVENESS: float = float(os.getenv("TTS_EXPRESSIVENESS", "1.0"))

# Servo / pose intensity index is 0–4 from user emotion; scale down for gentler motion.
SERVO_INTENSITY_SCALE: float = float(os.getenv("SERVO_INTENSITY_SCALE", "1.0"))

# Minimum fused dominant value before pipeline overrides expression with user mirroring.
# Higher = mirror less often (RIO stays more “neutral lead”).
EMOTION_MIRROR_THRESHOLD: float = float(os.getenv("EMOTION_MIRROR_THRESHOLD", "0.3"))


def dampen_emotion_vector(emotions: dict, damping: float) -> dict:
    """
    Pull emotion scores toward a uniform mix so peaks read as milder.

    damping=1.0 → unchanged (only clamped to [0, 1]).
    damping=0.0 → every key becomes 1/n (flat).
    """
    if not emotions:
        return {}
    d = max(0.0, min(1.0, float(damping)))
    keys = list(emotions.keys())
    uniform = 1.0 / len(keys)
    if d >= 0.999:
        return {k: max(0.0, min(1.0, float(v))) for k, v in emotions.items()}
    return {
        k: max(0.0, min(1.0, uniform + d * (float(emotions[k]) - uniform)))
        for k in keys
    }
CLEAR_MEMORY_ON_START: bool = os.getenv("CLEAR_MEMORY_ON_START", "true").lower() == "true"
WEBCAM_INDEX: int = 0
MIC_INDEX: int = int(os.getenv("MIC_INDEX", "0"))  # Auto-detect default microphone. Set MIC_INDEX=1 if you have multiple mics
ENABLE_VOICE_DETECTOR: bool = os.getenv("ENABLE_VOICE_DETECTOR", "true").lower() == "true"
SHORT_TERM_MEMORY_SIZE: int = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "10"))

STAGNATION_THRESHOLD: float = 0.05
STAGNATION_REROUTE_AFTER: int = 3

# ============================================================
# TEXT-TO-SPEECH (gTTS + fallback pyttsx3)
# ============================================================
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "gtts")  # "gtts" or "pyttsx3"
TTS_CACHE_DIR: str = "tts/cache"
TTS_LANGUAGE_DETECT: bool = True  # Auto-detect Hindi vs English
TTS_SPEED: float = 1.0

# ============================================================
# CREW AI ORCHESTRATION
# ============================================================
CREW_VERBOSE: bool = False  # Set True for debug output
CREW_EXECUTOR_BACKEND: str = "threaded"  # threaded, sequential, or process

# ============================================================
# MEMORY CONFIGURATION
# ============================================================
MEMORY_SHORT_TERM_FILE: str = "memory/short_term_memory.json"
MEMORY_PERSISTENT_FILE: str = "memory/mock_data/user_profiles.json"
MEMORY_MAX_EXCHANGES: int = 7

# ============================================================
# ANIMATION & STREAMLIT
# ============================================================
ANIMATION_ASSETS_DIR: str = "rio_js/public/facial_expressions"
STREAMLIT_LAYOUT: str = "wide"
STREAMLIT_THEME: str = "dark"  # therapeutic dark theme
EMOTION_TRANSITION_MS: int = 500

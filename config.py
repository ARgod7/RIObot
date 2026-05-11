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
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b")  # ~4.1GB, bilingual Hindi/English
OLLAMA_TIMEOUT: float = 10.0  # Ollama needs a bit more time
OLLAMA_STREAM: bool = False

RIO_PORT: int = int(os.getenv("RIO_PORT", "5000"))
RIO_BASE_URL: str = f"http://localhost:{RIO_PORT}"

# Rio Bridge (JS engine) — used by rio_client.py
RIO_BRIDGE_URL: str = RIO_BASE_URL          # http://localhost:{RIO_PORT}
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

# Face detector sampling (seconds between fresh face emotion reads)
FACE_SAMPLE_INTERVAL_S: float = float(os.getenv("FACE_SAMPLE_INTERVAL_S", "5"))

# Fusion smoothing: lower alpha = smoother, slower-moving fused vector (see EmotionFusionEngine.fuse_smoothed)
FUSION_SMOOTH_ALPHA: float = float(os.getenv("FUSION_SMOOTH_ALPHA", "0.28"))

# Min seconds between emotion WebSocket pushes from main.py (details dashboard)
EMOTION_WS_BROADCAST_MIN_S: float = float(os.getenv("EMOTION_WS_BROADCAST_MIN_S", "0.15"))
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

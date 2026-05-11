"""
Google Translate TTS Wrapper
Free, cached text-to-speech with Hindi/English support
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import json
from datetime import datetime
import subprocess
import sys
import shutil
import time

logger = logging.getLogger(__name__)

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = Path("tts/cache")
CACHE_METADATA = CACHE_DIR / "metadata.json"
MAX_CACHE_SIZE_MB = 500  # Keep cache under 500MB
DEFAULT_LANG = "en"
TTS_SPEED = 1.0  # 1.0 = normal speed

# Language mapping
LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "english": "en",
    "hindi": "hi",
}

# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:
    """
    Detect language from text (Hindi/English)

    Returns: "hi" or "en" (defaults to "en")
    """
    if not text or not text.strip():
        return "en"

    if not LANGDETECT_AVAILABLE:
        # Fallback: check for Devanagari characters
        if any('\u0900' <= c <= '\u097f' for c in text):
            return "hi"
        return "en"

    try:
        detected = detect(text)
        if detected in ["hi", "en"]:
            return detected
        # Map other codes
        return LANG_MAP.get(detected[:2], "en")
    except (LangDetectException, Exception):
        # Fallback to Devanagari check
        if any('\u0900' <= c <= '\u097f' for c in text):
            return "hi"
        return "en"


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def get_cache_hash(text: str, lang: str) -> str:
    """Generate unique hash for cached audio"""
    key = f"{text}_{lang}"
    return hashlib.md5(key.encode()).hexdigest()


def load_cache_metadata() -> dict:
    """Load cache metadata"""
    if CACHE_METADATA.exists():
        try:
            with open(CACHE_METADATA, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache_metadata(metadata: dict):
    """Save cache metadata"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_METADATA, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"⚠️ Warning: Failed to save cache metadata: {e}")


def cleanup_cache_if_needed():
    """Remove old cache files if cache > MAX_SIZE"""
    try:
        cache_size_mb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.mp3")) / (1024 * 1024)

        if cache_size_mb > MAX_CACHE_SIZE_MB:
            print(f"🧹 Cleaning cache ({cache_size_mb:.1f}MB > {MAX_CACHE_SIZE_MB}MB)")

            # Get all mp3 files with modification time
            mp3_files = sorted(
                CACHE_DIR.glob("*.mp3"),
                key=lambda f: f.stat().st_mtime
            )

            # Remove oldest files until under limit
            for mp3_file in mp3_files:
                mp3_file.unlink()
                cache_size_mb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.mp3")) / (1024 * 1024)

                if cache_size_mb < MAX_CACHE_SIZE_MB * 0.8:
                    break
    except Exception as e:
        print(f"⚠️ Warning: Cache cleanup failed: {e}")


# ============================================================
# TTS ENGINE
# ============================================================

class GTTSEngine:
    """Google Translate TTS wrapper with caching"""

    def __init__(self, cache_dir: str = "tts/cache", enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not GTTS_AVAILABLE:
            print("⚠️ gTTS not available. Install: pip install gTTS")

    def text_to_speech(
        self,
        text: str,
        lang: Optional[str] = None,
        output_file: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[bool, str, str]:
        """
        Convert text to speech

        Args:
            text: Text to convert
            lang: Language code ("en", "hi", or auto-detect)
            output_file: Save to file path (optional)
            use_cache: Use cached audio if available

        Returns:
            (success: bool, audio_file_path: str, error_message: str)
        """

        if not text or not text.strip():
            return False, "", "Empty text"

        # Detect language if not specified
        if lang is None:
            lang = detect_language(text)
        else:
            lang = LANG_MAP.get(lang.lower(), "en")

        # Generate cache hash
        cache_hash = get_cache_hash(text, lang)
        cache_file = self.cache_dir / f"{cache_hash}.mp3"

        # Check cache first
        if use_cache and cache_file.exists():
            print(f"✓ Cache hit: {cache_hash[:8]}...")
            return True, str(cache_file), ""

        # Generate TTS
        if not GTTS_AVAILABLE:
            return False, "", "gTTS not installed"

        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)

            # Save to cache
            tts.save(str(cache_file))
            print(f"✓ TTS generated: {lang} ({len(text)} chars) → {cache_hash[:8]}...")

            # Update metadata
            metadata = load_cache_metadata()
            metadata[cache_hash] = {
                "text": text[:100],  # Store first 100 chars
                "lang": lang,
                "created": datetime.now().isoformat(),
                "file": str(cache_file),
            }
            save_cache_metadata(metadata)

            # Cleanup if needed
            cleanup_cache_if_needed()

            return True, str(cache_file), ""

        except Exception as e:
            return False, "", f"TTS error: {str(e)}"

    def play_audio(self, audio_file: str) -> bool:
        """
        Play audio file (cross-platform)

        Returns: True if played, False if failed
        """
        if not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return False

        try:
            if sys.platform == "win32":
                # Windows
                os.startfile(audio_file)
            elif sys.platform == "darwin":
                # macOS
                subprocess.run(["afplay", audio_file])
            else:
                # Linux
                subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file])

            print(f"🔊 Playing: {audio_file}")
            return True
        except Exception as e:
            print(f"❌ Playback error: {str(e)}")
            return False

    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB"""
        try:
            return sum(f.stat().st_size for f in self.cache_dir.glob("*.mp3")) / (1024 * 1024)
        except Exception:
            return 0.0

    def clear_cache(self):
        """Clear all cached audio"""
        try:
            for mp3_file in self.cache_dir.glob("*.mp3"):
                mp3_file.unlink()
            CACHE_METADATA.unlink(missing_ok=True)
            print("🗑️ TTS cache cleared")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")


# ============================================================
# SIMPLE API
# ============================================================

_engine = None

def init_tts_engine(cache_dir: str = "tts/cache"):
    """Initialize global TTS engine"""
    global _engine
    _engine = GTTSEngine(cache_dir=cache_dir)
    return _engine

def tts_generate(text: str, lang: Optional[str] = None) -> Tuple[bool, str]:
    """
    Simple API: Convert text to speech

    Returns: (success, audio_file_path)
    """
    global _engine
    if _engine is None:
        _engine = init_tts_engine()

    success, audio_file, error = _engine.text_to_speech(text, lang=lang)

    if not success:
        print(f"❌ TTS failed: {error}")
        return False, ""

    return True, audio_file

def tts_play(audio_file: str) -> bool:
    """Simple API: Play audio file"""
    global _engine
    if _engine is None:
        _engine = init_tts_engine()

    return _engine.play_audio(audio_file)

def tts_get_cache_size() -> float:
    """Get cache size in MB"""
    global _engine
    if _engine is None:
        _engine = init_tts_engine()

    return _engine.get_cache_size_mb()

def tts_clear_cache():
    """Clear cache"""
    global _engine
    if _engine is None:
        _engine = init_tts_engine()

    _engine.clear_cache()


def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def _ffmpeg_supports_rubberband(ffmpeg_exe: str) -> bool:
    try:
        r = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-h", "filter=rubberband"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0 and "rubberband" in (r.stdout or "").lower()
    except Exception:
        return False


def _apply_prosody(src: Path, dst: Path, pitch: float, speed: float) -> bool:
    """
    Adjust pitch/speed on MP3 via ffmpeg. Prefers rubberband for independent control.
    Falls back to atempo + small pitch→tempo nudge if rubberband is unavailable.
    """
    ffmpeg_exe = _ffmpeg_bin()
    if not ffmpeg_exe or not src.exists():
        return False

    pitch = max(0.78, min(1.32, float(pitch)))
    speed = max(0.82, min(1.12, float(speed)))

    def _chain_atempo(factor: float) -> str:
        """atempo must stay in (0.5, 2.0); chain filters if needed."""
        parts = []
        remaining = factor
        while remaining < 0.5:
            parts.append("atempo=0.5")
            remaining /= 0.5
        while remaining > 2.0:
            parts.append("atempo=2.0")
            remaining /= 2.0
        parts.append(f"atempo={remaining:.6f}")
        return ",".join(parts)

    try:
        if _ffmpeg_supports_rubberband(ffmpeg_exe):
            # libavfilter rubberband: pitch and tempo are scale factors (1.0 = unchanged)
            filt = f"rubberband=pitch={pitch:.5f}:tempo={speed:.6f}:pitchq=quality"
        else:
            # Without rubberband: primary control is speed; nudge tempo from pitch (lower pitch → slightly slower)
            combined = speed * (1.0 + (1.0 - pitch) * 0.45)
            combined = max(0.82, min(1.12, combined))
            filt = _chain_atempo(combined)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-af",
            filt,
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            logger.warning("ffmpeg prosody failed: %s", (r.stderr or r.stdout or "")[:200])
            return False
        return dst.exists() and dst.stat().st_size > 0
    except Exception as e:
        logger.warning("ffmpeg prosody error: %s", e)
        return False


def speak(text: str, lang: str = "en", pitch: float = 1.0, speed: float = 1.0) -> str:
    """
    Speak text using gTTS, then optional ffmpeg prosody (pitch/speed) for browser playback.

    Saves audio under rio_js/public/audio/. Requires ffmpeg on PATH for prosody;
    if missing or processing fails, falls back to raw gTTS MP3.
    """
    if not text or not text.strip():
        return ""

    global _engine
    if _engine is None:
        _engine = init_tts_engine()

    success, audio_file = tts_generate(text, lang=lang)
    if success and audio_file:
        try:
            project_root = Path(__file__).parent.parent  # rio/
            audio_dir = project_root / "rio_js" / "public" / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            response_file = audio_dir / f"response_{timestamp}.mp3"
            latest_file = audio_dir / "response.mp3"
            src_path = Path(audio_file)
            tmp_proc = audio_dir / f"_proc_{timestamp}.mp3"

            use_raw = abs(pitch - 1.0) < 0.02 and abs(speed - 1.0) < 0.02
            if use_raw:
                shutil.copy2(src_path, response_file)
            elif _apply_prosody(src_path, tmp_proc, pitch, speed):
                shutil.move(str(tmp_proc), str(response_file))
            else:
                shutil.copy2(src_path, response_file)
                if not use_raw:
                    logger.info("Prosody skipped (no ffmpeg or filter failed); using raw TTS")

            shutil.copy2(response_file, latest_file)

            logger.info(f"✓ Audio saved to browser: {response_file}")
            return f"/audio/{response_file.name}"
        except Exception as e:
            logger.error(f"Failed to copy audio to public directory: {e}")
            return ""
    else:
        logger.warning(f"TTS speak() failed for: {text[:50]}")
        return ""


# ============================================================
# INIT ON IMPORT
# ============================================================

init_tts_engine()

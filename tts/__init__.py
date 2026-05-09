"""
Text-to-Speech Layer

Provides gTTS + pyttsx3 wrapper for:
- English (fast, gTTS)
- Hindi (pyttsx3 fallback)
- Caching to avoid re-rendering
"""

from .gtts_wrapper import tts_generate, tts_play, detect_language, GTTSEngine
from .pyttsx3_fallback import pyttsx3_speak, Pyttsx3Engine

# Unified interface
class TTSEngine:
    """
    Unified TTS with fallback chain: gTTS (cached) → pyttsx3 (offline) → silent
    """

    def __init__(self, prefer_online: bool = True):
        self.prefer_online = prefer_online
        self.gtts_engine = GTTSEngine()
        self.pyttsx3_engine = Pyttsx3Engine()
        self.last_audio_file = None

    def speak(self, text: str, lang=None, play: bool = True):
        """Convert text to speech with fallback"""
        if not text:
            return False, ""

        if lang is None:
            lang = detect_language(text)

        # Try gTTS first
        success, audio_file, _ = self.gtts_engine.text_to_speech(text, lang=lang, use_cache=True)
        if success:
            self.last_audio_file = audio_file
            if play:
                self.gtts_engine.play_audio(audio_file)
            return True, audio_file

        # Fallback to pyttsx3
        success, _ = self.pyttsx3_engine.text_to_speech(text, lang=lang)
        if success:
            return True, ""

        print(f"❌ TTS failed: {text[:30]}...")
        return False, ""

__all__ = ["TTSEngine", "speak", "tts_generate", "detect_language"]

# Global instance
_engine = None

def get_tts_engine():
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine

def speak(text: str, lang=None, play: bool = True):
    """Simple API: speak(text, lang='en')"""
    engine = get_tts_engine()
    return engine.speak(text, lang=lang, play=play)


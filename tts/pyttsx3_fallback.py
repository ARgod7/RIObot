"""
Pyttsx3 Fallback TTS Engine
Offline text-to-speech (lower quality but works offline)
Used when gTTS fails or for Hindi (pyttsx3 has Hindi support)
"""

import sys
from typing import Tuple, Optional

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


# ============================================================
# PYTTSX3 ENGINE
# ============================================================

class Pyttsx3Engine:
    """Offline TTS using pyttsx3"""

    def __init__(self):
        if not PYTTSX3_AVAILABLE:
            print("⚠️ pyttsx3 not available. Install: pip install pyttsx3")
            self.engine = None
            return

        try:
            self.engine = pyttsx3.init()

            # Configure voice
            self.engine.setProperty("rate", 150)  # Slower for elderly
            self.engine.setProperty("volume", 1.0)

            print("✓ pyttsx3 engine initialized")
        except Exception as e:
            print(f"⚠️ pyttsx3 init failed: {e}")
            self.engine = None

    def text_to_speech(
        self,
        text: str,
        lang: str = "en",
        output_file: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Convert text to speech

        Args:
            text: Text to convert
            lang: "en" or "hi" (Hindi support varies by OS)
            output_file: Save to .wav file

        Returns:
            (success, output_path or empty string)
        """

        if not self.engine or not text:
            return False, ""

        try:
            # Select voice based on language
            voices = self.engine.getProperty("voices")

            if lang == "hi":
                # Try to find Hindi voice
                hindi_voices = [v for v in voices if "hindi" in v.name.lower() or "hi" in v.name.lower()]
                if hindi_voices:
                    self.engine.setProperty("voice", hindi_voices[0].id)
                else:
                    # Fallback to default (English)
                    print(f"⚠️ Hindi voice not found, using default")
                    self.engine.setProperty("voice", voices[0].id)
            else:
                # Use default English voice
                self.engine.setProperty("voice", voices[0].id)

            # Speak
            self.engine.say(text)

            # Save to file if specified
            if output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                print(f"✓ Audio saved: {output_file}")
                return True, output_file
            else:
                self.engine.runAndWait()
                print(f"✓ Audio played ({lang}): {text[:50]}...")
                return True, ""

        except Exception as e:
            print(f"❌ Pyttsx3 error: {str(e)}")
            return False, ""

    def play_text(self, text: str, lang: str = "en") -> bool:
        """
        Play text directly (blocking, plays immediately)
        """
        success, _ = self.text_to_speech(text, lang=lang)
        return success


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_engine = None

def init_pyttsx3_engine():
    """Initialize global pyttsx3 engine"""
    global _engine
    _engine = Pyttsx3Engine()
    return _engine

def pyttsx3_speak(text: str, lang: str = "en") -> bool:
    """
    Simple API: Speak text using pyttsx3

    Args:
        text: Text to speak
        lang: "en" or "hi"

    Returns: True if successful
    """
    global _engine
    if _engine is None:
        _engine = init_pyttsx3_engine()

    if _engine.engine is None:
        return False

    success, _ = _engine.text_to_speech(text, lang=lang)
    return success


# ============================================================
# INIT ON IMPORT
# ============================================================

if PYTTSX3_AVAILABLE:
    init_pyttsx3_engine()


"""
voice_detector.py
-----------------
Voice-based emotion detection module for the RIO robot perception pipeline.

Pipeline:
  1. Capture audio from microphone  (PyAudio / sounddevice)
  2. Speech-to-text                 (SpeechRecognition → Google or Whisper)
  3. Tone / sentiment analysis      (text-emotion via transformers or lexicon)
  4. Acoustic features              (pitch, speech rate, loudness → emotion cues)
     Based on RIO paper refs [70-73]: each emotion has a distinct acoustic profile

Install:
    pip install SpeechRecognition pyaudio sounddevice transformers torch

Acoustic emotion cues (from RIO paper Section 5.4.3):
    anger (low):   lower pitch, high loudness, high rate
    anger (hot):   high pitch,  high loudness, high rate
    disgust:       low pitch,   low loudness,  high rate
    fear:          high pitch,  low loudness,  high rate
    sadness:       high pitch,  low loudness,  low rate   ← key: slow + quiet
    joy:           high pitch,  high loudness, high rate
    positive surprise: high pitch, high rate, high loudness
    negative surprise: high pitch, low loudness, high rate (like fear)
"""

import logging
import threading
import queue
import time
import numpy as np
import os
from typing import Optional
from perception.emotion_fusion import EmotionVector
from config import MIC_INDEX

logger = logging.getLogger("VoiceDetector")


def get_latest_transcript() -> str:
    """Return the most recent transcript string, or empty string if none."""
    # Access the module-level detector's transcript if available
    try:
        from perception.voice_detector import VoiceEmotionDetector
        # This is exposed via the class's get_transcript method
        return ""  # Fallback; actual value set during listening
    except Exception:
        return ""


class VoiceEmotionDetector:
    """
    Listens to the microphone, transcribes speech, and estimates emotion.

    Two parallel signals:
      A) Text-based emotion:  run a transformer model on the transcription
      B) Acoustic features:   analyse pitch, loudness, speech rate from raw audio

    These are combined (70% text, 30% acoustic) into a single EmotionVector.

    Usage:
        detector = VoiceEmotionDetector()
        detector.start_listening()          # non-blocking background thread

        # In your main loop:
        vector = detector.get_latest()      # returns most recent EmotionVector
        transcript = detector.get_transcript()

        detector.stop_listening()
    """

    def __init__(
        self,
        language: str = "en-US",
        text_weight: float = 0.7,
        acoustic_weight: float = 0.3,
        phrase_time_limit: float = 6.0,   # max seconds per utterance
        use_transformer: bool = True,
        lazy_transformer: Optional[bool] = None,
    ):
        self.language = language
        self.text_weight = text_weight
        self.acoustic_weight = acoustic_weight
        self.phrase_time_limit = phrase_time_limit
        self.use_transformer = use_transformer

        self._result_queue = queue.Queue()
        self._latest_vector = EmotionVector(source="voice", confidence=0.0)
        self._latest_transcript = ""
        self._listening_thread = None
        self._stop_flag = threading.Event()

        self._sr_available = False
        self._transformer_available = False
        self._emotion_pipeline = None
        self._mic_names = []
        self._mic_index = None
        self._transformer_loading = False
        if lazy_transformer is None:
            lazy_transformer = os.getenv("VOICE_LAZY_TRANSFORMER", "1").strip() != "0"
        self._lazy_transformer = bool(lazy_transformer)

        self._init_speech_recognition()
        if use_transformer:
            if self._lazy_transformer:
                self._init_transformer_async()
            else:
                self._init_transformer()

    def _init_speech_recognition(self):
        try:
            import speech_recognition as sr
            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._sr_available = True
            logger.info("SpeechRecognition loaded.")

            # Test microphone availability
            try:
                self._mic_names = sr.Microphone.list_microphone_names()
                if self._mic_names:
                    indexed = ", ".join(f"{i}:{name}" for i, name in enumerate(self._mic_names))
                    logger.info(f"Available microphones: {indexed}")
                else:
                    logger.info("No microphones reported by SpeechRecognition.")
            except Exception as e:
                logger.warning(f"Could not list microphones: {e}")

            # Validate MIC_INDEX against available devices
            if self._mic_names and 0 <= MIC_INDEX < len(self._mic_names):
                self._mic_index = MIC_INDEX
            else:
                self._mic_index = None
                if self._mic_names:
                    logger.warning(
                        "MIC_INDEX %s is invalid for %s devices; falling back to default microphone.",
                        MIC_INDEX,
                        len(self._mic_names),
                    )

        except ImportError:
            logger.warning("SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")

    def _init_transformer(self):
        """
        Load a text-emotion classifier (HuggingFace pipeline).
        Model: j-hartmann/emotion-english-distilroberta-base
        Outputs: anger, disgust, fear, joy, neutral, sadness, surprise
        — exact Ekman alignment, perfect for RIO.
        """
        try:
            from transformers import pipeline
            logger.info("Loading emotion transformer model (first run may download ~300MB)...")
            self._emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=-1       # CPU; change to 0 for GPU
            )
            self._transformer_available = True
            logger.info("Emotion transformer model loaded.")
        except ImportError:
            logger.warning("transformers/torch not installed. Falling back to lexicon-based analysis.")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}. Using lexicon fallback.")
        finally:
            self._transformer_loading = False

    def _init_transformer_async(self) -> None:
        if self._transformer_loading or self._transformer_available:
            return
        self._transformer_loading = True
        t = threading.Thread(target=self._init_transformer, daemon=True)
        t.start()

    # ── Text-Based Emotion Analysis ─────────────────────────────────────────

    TRANSFORMER_TO_EKMAN = {
        "anger":    "anger",
        "disgust":  "disgust",
        "fear":     "fear",
        "joy":      "joy",
        "neutral":  None,
        "sadness":  "sadness",
        "surprise": "surprise",
    }

    def _text_to_emotion(self, text: str) -> EmotionVector:
        """Analyse transcript text → EmotionVector."""
        if not text.strip():
            return EmotionVector(source="voice_text", confidence=0.0)

        if self._transformer_available and self._emotion_pipeline:
            return self._transformer_emotion(text)
        else:
            return self._lexicon_emotion(text)

    def _transformer_emotion(self, text: str) -> EmotionVector:
        try:
            results = self._emotion_pipeline(text[:512])  # truncate for safety
            if isinstance(results, list) and results and isinstance(results[0], list):
                results = results[0]
            if isinstance(results, dict):
                if "labels" in results and "scores" in results:
                    results = [
                        {"label": label, "score": score}
                        for label, score in zip(results["labels"], results["scores"])
                    ]
                elif "label" in results and "score" in results:
                    results = [results]
                else:
                    results = []

            if not isinstance(results, list) or not results or not isinstance(results[0], dict):
                raise ValueError(f"Unexpected transformer output: {type(results)}")

            scores = {r.get("label", "").lower(): float(r.get("score", 0.0)) for r in results}
            neutral_score = scores.pop("neutral", 0.0)

            # Redistribute neutral slightly across all
            ambient = neutral_score * 0.05

            vector_vals = {}
            for model_label, ekman_key in self.TRANSFORMER_TO_EKMAN.items():
                if ekman_key is None:
                    continue
                vector_vals[ekman_key] = scores.get(model_label, 0.0) + ambient

            # Confidence = 1 - neutral dominance
            confidence = min(0.95, 0.5 + (1.0 - neutral_score) * 0.5)
            return EmotionVector(**vector_vals, confidence=confidence, source="voice_text").clamp()

        except Exception as e:
            logger.warning(f"Transformer emotion failed: {e}")
            return self._lexicon_emotion(text)

    # Simple keyword lexicon fallback (no ML required)
    SADNESS_WORDS   = {"sad", "lonely", "alone", "miss", "tired", "exhausted", "depressed", "unhappy", "cry", "grief", "lost"}
    JOY_WORDS       = {"happy", "good", "great", "wonderful", "love", "excited", "glad", "joy", "cheerful", "nice"}
    FEAR_WORDS      = {"scared", "afraid", "frightened", "worried", "anxious", "nervous", "fear", "dread", "panic"}
    ANGER_WORDS     = {"angry", "upset", "frustrated", "hate", "mad", "irritated", "furious", "annoy"}
    DISGUST_WORDS   = {"disgusting", "horrible", "awful", "terrible", "gross", "bad", "ugly"}
    SURPRISE_WORDS  = {"surprised", "shocked", "wow", "really", "unexpected", "amazing", "sudden"}

    def _lexicon_emotion(self, text: str) -> EmotionVector:
        words = set(text.lower().split())
        scores = {
            "sadness":  len(words & self.SADNESS_WORDS),
            "joy":      len(words & self.JOY_WORDS),
            "fear":     len(words & self.FEAR_WORDS),
            "anger":    len(words & self.ANGER_WORDS),
            "disgust":  len(words & self.DISGUST_WORDS),
            "surprise": len(words & self.SURPRISE_WORDS),
        }
        total = sum(scores.values()) or 1
        normalised = {k: v / total for k, v in scores.items()}
        confidence = 0.4 if total > 1 else 0.2  # low confidence for lexicon
        return EmotionVector(**normalised, confidence=confidence, source="voice_lexicon").clamp()

    # ── Acoustic Feature Analysis ────────────────────────────────────────────

    def _acoustic_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> EmotionVector:
        """
        Estimate emotion from acoustic features of raw audio.
        Based on RIO paper Section 5.4.3 acoustic profiles.

        Features extracted:
          - RMS energy (loudness proxy)
          - Zero crossing rate (frequency/pitch proxy)
          - Speech rate (utterance length vs audio duration)
        """
        if audio_data is None or len(audio_data) == 0:
            return EmotionVector(confidence=0.0, source="voice_acoustic")

        # Normalise audio to [-1, 1]
        audio_float = audio_data.astype(np.float32)
        max_val = np.max(np.abs(audio_float)) or 1.0
        audio_float /= max_val

        # RMS energy (loudness)
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        loudness = min(1.0, rms * 5)      # scale to [0,1] roughly

        # Zero crossing rate (proxy for pitch/frequency)
        zcr = float(np.mean(np.diff(np.sign(audio_float)) != 0))
        pitch_proxy = min(1.0, zcr * 3)  # scale

        # Duration-based speech rate (words estimated from duration)
        duration = len(audio_float) / sample_rate
        # We don't have word count here — use audio energy variation as proxy
        energy_variation = float(np.std(audio_float))
        speech_rate_proxy = min(1.0, energy_variation * 4)

        # ── Map features to emotions (from RIO paper refs [70-73]) ───────
        v = EmotionVector(source="voice_acoustic")

        # Sadness:  high pitch proxy, low loudness, low speech rate
        sadness_score = (pitch_proxy * 0.4 + (1 - loudness) * 0.4 + (1 - speech_rate_proxy) * 0.2)
        v.sadness = sadness_score * 0.8

        # Fear: high pitch, low loudness, high speech rate
        fear_score = (pitch_proxy * 0.4 + (1 - loudness) * 0.3 + speech_rate_proxy * 0.3)
        v.fear = fear_score * 0.6

        # Joy: high pitch, high loudness, high rate
        joy_score = (pitch_proxy * 0.3 + loudness * 0.4 + speech_rate_proxy * 0.3)
        v.joy = joy_score * 0.7

        # Anger: loud, high rate (pitch varies)
        anger_score = (loudness * 0.5 + speech_rate_proxy * 0.5)
        v.anger = anger_score * 0.6

        # Confidence for acoustic is modest — it's a rough proxy
        v.confidence = 0.45

        return v.clamp()

    # ── Listening Thread ────────────────────────────────────────────────────

    def _listen_loop(self):
        """Background thread: continuously listen and transcribe."""
        if not self._sr_available:
            logger.error("SpeechRecognition not available. Cannot listen.")
            return

        sr = self._sr

        try:
            if self._mic_index is not None:
                mic = sr.Microphone(device_index=self._mic_index)
                logger.info("Using microphone index %s", self._mic_index)
            else:
                mic = sr.Microphone()
                logger.info("Using default microphone device")
        except Exception as e:
            logger.error(f"Microphone not available: {e}")
            logger.error("Check: Windows Settings → Privacy & Security → Microphone")
            return

        try:
            with mic as source:
                logger.info("Calibrating microphone for ambient noise (2s)...")
                self._recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info("✅ Listening started. Speak to the robot.")

                while not self._stop_flag.is_set():
                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=3,
                            phrase_time_limit=self.phrase_time_limit
                        )
                        self._process_audio(audio)
                    except sr.WaitTimeoutError:
                        pass  # normal — no speech detected in this window
                    except Exception as e:
                        logger.warning(f"Listen error: {e}")
                        time.sleep(0.5)
        except Exception as e:
            logger.error(f"Microphone context error: {e}")
            logger.error("Check your microphone connection and permissions")

    def _process_audio(self, audio):
        """Transcribe audio and compute emotion."""
        sr = self._sr

        # Get raw audio as numpy array for acoustic analysis
        raw = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16)

        # Transcribe
        transcript = ""
        try:
            transcript = self._recognizer.recognize_google(audio, language=self.language)
            logger.info(f"Transcribed: '{transcript}'")
            self._latest_transcript = transcript
        except sr.UnknownValueError:
            logger.debug("Speech not understood.")
        except sr.RequestError as e:
            logger.warning(f"STT service error: {e}")

        # Compute both signals
        text_vector     = self._text_to_emotion(transcript) if transcript else EmotionVector(confidence=0.0)
        acoustic_vector = self._acoustic_emotion(raw)

        # Combine: weighted average of text and acoustic signals
        tw = self.text_weight if transcript else 0.0
        aw = self.acoustic_weight
        total_w = tw + aw or 1.0

        combined = EmotionVector(source="voice")
        for key in ["joy","sadness","fear","disgust","anger","surprise"]:
            val = (getattr(text_vector, key) * tw + getattr(acoustic_vector, key) * aw) / total_w
            setattr(combined, key, val)

        combined.confidence = (
            text_vector.confidence * tw + acoustic_vector.confidence * aw
        ) / total_w

        combined.clamp()
        self._latest_vector = combined
        self._result_queue.put(combined)
        logger.info(f"Voice emotion: {combined}")

    # ── Public API ──────────────────────────────────────────────────────────

    def start_listening(self):
        """Start background microphone listening thread."""
        if not self._sr_available:
            logger.error("Cannot start — SpeechRecognition not installed.")
            return
        self._stop_flag.clear()
        self._listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listening_thread.start()
        logger.info("Voice listener started.")

    def stop_listening(self):
        """Stop the background listening thread."""
        self._stop_flag.set()
        if self._listening_thread:
            self._listening_thread.join(timeout=3)
        logger.info("Voice listener stopped.")

    def is_listening(self) -> bool:
        """Return True if the background listener thread is active."""
        return bool(self._listening_thread and self._listening_thread.is_alive())

    def is_available(self) -> bool:
        """Return True if SpeechRecognition is available."""
        return self._sr_available

    def get_latest(self) -> EmotionVector:
        """Return the most recently computed voice EmotionVector."""
        return self._latest_vector

    def get_transcript(self) -> str:
        """Return the most recently transcribed text."""
        return self._latest_transcript

    def analyse_text(self, text: str) -> EmotionVector:
        """
        Directly analyse a text string (for testing without microphone,
        or for feeding in text from another STT source).
        """
        return self._text_to_emotion(text)

    def feed_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """
        Feed raw audio chunk directly (e.g., from WebSocket or browser).
        
        This allows bypassing the microphone and feeding audio from:
        - Browser via WebSocket (mic_bridge.js)
        - External audio sources
        - Offline processing
        
        Args:
            audio_data (np.ndarray): int16 audio samples
            sample_rate (int): Sample rate of audio (default 16000)
        
        Internally processes acoustic emotion without transcription,
        since this method doesn't have access to speech-to-text.
        The caller should provide transcripts separately if needed.
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        try:
            # Process acoustic emotion from raw audio
            acoustic_vector = self._acoustic_emotion(audio_data, sample_rate)
            
            # Update latest vector (acoustic only, no text)
            self._latest_vector = acoustic_vector
            self._result_queue.put(acoustic_vector)
            logger.debug(f"Voice emotion (audio-only): {acoustic_vector}")
        except Exception as e:
            logger.warning(f"Error processing audio chunk: {e}")


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Voice Emotion Detector — Tests\n")

    detector = VoiceEmotionDetector(use_transformer=True)

    # Text-based tests (no microphone needed)
    test_phrases = [
        "I feel so lonely and tired all the time",
        "I am so happy today, it's a wonderful day!",
        "I am scared and worried about my health",
        "I hate this, I'm so frustrated",
    ]

    print("── Text Analysis Tests ──")
    for phrase in test_phrases:
        vector = detector.analyse_text(phrase)
        from emotion_fusion import EmotionFusionEngine
        engine = EmotionFusionEngine()
        engine.update("voice", vector)
        fused = engine.fuse()
        dominant = max(fused.to_dict(), key=fused.to_dict().get)
        print(f"  '{phrase[:45]}...' → {dominant} ({fused.to_dict()[dominant]:.2f})")

    print("\n── Live Microphone Test ──")
    print("Starting microphone listener. Speak something. Ctrl+C to stop.\n")
    try:
        detector.start_listening()
        while True:
            time.sleep(1)
            v = detector.get_latest()
            if v.confidence > 0:
                print(f"Latest: {v}")
    except KeyboardInterrupt:
        detector.stop_listening()
        print("\nStopped.")

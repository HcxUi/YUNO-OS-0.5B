"""
YUNO Voice Module
==================
Provides Text-to-Speech (TTS) voice synthesis and Speech-to-Text (STT) voice
recognition for YUNO OS.

Dependencies:
    - pyttsx3 for offline, cross-platform Text-To-Speech synthesis
    - SpeechRecognition for microphone audio transcription and audio file speech-to-text

Usage:
    from yuno_llm.voice import YunoVoice
    voice = YunoVoice(config)

    # Speak text aloud
    voice.speak("Namaste! Main YUNO hoon.")

    # Transcribe audio file
    text = voice.transcribe_audio_file("sample.wav")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("yuno_llm.voice")


class YunoVoice:
    """
    Core YUNO OS Voice engine.

    Handles Text-To-Speech (TTS) synthesis and Speech-To-Text (STT) audio transcription.
    """

    def __init__(self, config=None):
        self.config = config
        self.enabled: bool = True
        self.tts_engine_name: str = "pyttsx3"
        self.speech_rate: int = 175
        self.volume: float = 1.0
        self.auto_speak: bool = False

        if config and hasattr(config, "voice"):
            v = config.voice
            self.enabled = getattr(v, "enabled", True)
            self.tts_engine_name = getattr(v, "tts_engine", "pyttsx3")
            self.speech_rate = getattr(v, "speech_rate", 175)
            self.volume = getattr(v, "volume", 1.0)
            self.auto_speak = getattr(v, "auto_speak", False)

        self._tts_engine = None

    # ── 1. Text-To-Speech (TTS) ───────────────────────────────────────────────

    def speak(self, text: str) -> bool:
        """
        Synthesize and speak text aloud using pyttsx3.

        Args:
            text: Text to speak aloud.

        Returns:
            True if speech synthesis succeeded, False otherwise.
        """
        if not text or not text.strip():
            return False

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"[Voice] TTS synthesis failed: {e}")
            return False

    # ── 2. Speech-To-Text (STT) Audio File Transcription ─────────────────────

    def transcribe_audio_file(self, audio_path: str, language: str = "en-IN") -> str:
        """
        Transcribe an audio file (.wav, .flac, .aiff) using SpeechRecognition.

        Args:
            audio_path: Path to audio file.
            language: Speech recognition language locale (default: "en-IN").

        Returns:
            Transcribed text string or error message.
        """
        path = Path(audio_path)
        if not path.exists():
            return f"[ERROR] Audio file not found: {audio_path}"

        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            with sr.AudioFile(str(path)) as source:
                audio_data = recognizer.record(source)

            # Recognize using Google Speech Recognition API / Sphinx fallback
            try:
                text = recognizer.recognize_google(audio_data, language=language)
                return text
            except sr.UnknownValueError:
                return "[Voice] Could not understand audio speech."
            except sr.RequestError as e:
                # Offline Sphinx fallback if internet is unavailable
                try:
                    text = recognizer.recognize_sphinx(audio_data)
                    return f"[Offline Sphinx] {text}"
                except Exception:
                    return f"[ERROR] Speech recognition request failed: {e}"

        except Exception as e:
            logger.error(f"[Voice] Audio transcription error: {e}")
            return f"[ERROR transcribing audio file: {e}]"

    # ── 3. Microphone Listening (Optional HITL trigger) ───────────────────────

    def listen(self, timeout: int = 5, language: str = "en-IN") -> str:
        """
        Listen to microphone audio input and transcribe to text.

        Args:
            timeout: Max seconds to wait for speech start.
            language: Language locale.

        Returns:
            Transcribed spoken input string.
        """
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                logger.info("[Voice] Listening to microphone input...")
                audio = recognizer.listen(source, timeout=timeout)

            text = recognizer.recognize_google(audio, language=language)
            return text
        except Exception as e:
            logger.warning(f"[Voice] Microphone listening unavailable/failed: {e}")
            return f"[ERROR microphone input: {e}]"

    def __repr__(self) -> str:
        return f"YunoVoice(enabled={self.enabled}, tts={self.tts_engine_name!r})"

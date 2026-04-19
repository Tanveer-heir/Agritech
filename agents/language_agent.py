"""
Language Agent
==============
Detects the farmer's language from their message and translates
the English diagnosis response into that language.

Supported languages (ISO 639-1):
  hi - Hindi      ta - Tamil     te - Telugu
  bn - Bengali    en - English (no translation needed)

Translation engine: IndicTrans2 (local) or Claude API (fallback/MVP).
"""

import os
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 42

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "en": "English",
}

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "hi")


def detect_language(text: str) -> str:
    """
    Detect the language of an input string.

    Returns ISO 639-1 code. Falls back to DEFAULT_LANGUAGE if
    detection fails or language is not supported.
    """
    try:
        detected = detect(text)
        return detected if detected in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    except Exception:
        return DEFAULT_LANGUAGE


def translate_to(text: str, target_language: str) -> str:
    """
    Translate an English text into target_language.

    Args:
        text: English text to translate
        target_language: ISO 639-1 code (e.g. "hi")

    Returns:
        Translated string
    """
    if target_language == "en":
        return text

    if os.getenv("USE_GEMINI_FOR_TRANSLATION", "true").lower() == "true":
        return _translate_via_gemini(text, target_language)
    else:
        return _translate_via_indictrans(text, target_language)


def _translate_via_gemini(text: str, target_language: str) -> str:
    """Translate using Gemini API — no local model required."""
    import google.generativeai as genai
    # TODO: Implement Gemini translation with a language-specific system prompt
    raise NotImplementedError("_translate_via_gemini() not yet implemented")


def _translate_via_indictrans(text: str, target_language: str) -> str:
    """Translate using local IndicTrans2 model."""
    # TODO: Load IndicTransToolkit model
    # TODO: Run inference
    raise NotImplementedError("_translate_via_indictrans() not yet implemented")

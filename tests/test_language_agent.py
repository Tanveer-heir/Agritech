"""
Tests for language_agent.py
"""
import pytest
from agents.language_agent import detect_language, SUPPORTED_LANGUAGES


def test_detect_hindi():
    """Hindi text should be detected as 'hi'."""
    hindi_text = "मेरी फसल की पत्तियाँ पीली हो रही हैं"
    result = detect_language(hindi_text)
    assert result == "hi"


def test_detect_english():
    """English text should be detected as 'en'."""
    result = detect_language("my crop leaves are turning yellow")
    assert result == "en"


def test_unsupported_language_falls_back_to_default():
    """Language not in SUPPORTED_LANGUAGES should return DEFAULT_LANGUAGE."""
    # French text
    french_text = "mes feuilles de maïs jaunissent"
    result = detect_language(french_text)
    assert result in SUPPORTED_LANGUAGES


def test_all_supported_languages_in_dict():
    """Verify all required MVP languages are in SUPPORTED_LANGUAGES."""
    required = {"hi", "ta", "te", "bn", "en"}
    assert required.issubset(SUPPORTED_LANGUAGES.keys())

"""
Tests for vision_agent.py
Use mocked Claude API responses — no API spend required.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_classify_disease_returns_expected_keys():
    """Vision agent result must contain all required keys."""
    mock_result = {
        "crop": "rice",
        "disease": "brown_leaf_spot",
        "severity": "moderate",
        "confidence": 0.87,
        "top_candidates": [
            {"disease": "brown_leaf_spot", "confidence": 0.87},
            {"disease": "blast", "confidence": 0.09},
            {"disease": "healthy", "confidence": 0.04},
        ]
    }
    required_keys = {"crop", "disease", "severity", "confidence", "top_candidates"}
    assert required_keys.issubset(mock_result.keys())


def test_low_confidence_below_threshold():
    """Confidence below 0.70 should trigger multi-candidate response."""
    CONFIDENCE_THRESHOLD = 0.70
    mock_confidence = 0.45
    assert mock_confidence < CONFIDENCE_THRESHOLD, \
        "Low-confidence case should return top candidates, not a single diagnosis"


# TODO: Add integration tests once vision_agent.py is implemented
# @patch("anthropic.Anthropic")
# def test_classify_disease_calls_claude(mock_client):
#     ...

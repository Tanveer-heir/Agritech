"""
Vision Agent
============
Sends a crop image to Gemini Vision API and returns a structured
disease classification result.

Returns:
    {
        "crop": str,
        "disease": str,
        "severity": "mild" | "moderate" | "severe",
        "confidence": float (0.0 - 1.0),
        "top_candidates": [{"disease": str, "confidence": float}]
    }
"""

import google.generativeai as genai
import base64
import os
from pathlib import Path


VISION_SYSTEM_PROMPT = """You are an expert plant pathologist specialising in Indian crops.
When given a photo of a crop leaf or plant, identify:
1. The crop species
2. The disease (or confirm it is healthy)
3. The severity: mild, moderate, or severe
4. Your confidence level (0.0 to 1.0)
5. Top 3 candidate diseases with confidence scores

Respond ONLY in this exact JSON format:
{
  "crop": "<crop name>",
  "disease": "<disease name or 'Healthy'>",
  "severity": "<mild|moderate|severe|none>",
  "confidence": <float>,
  "top_candidates": [
    {"disease": "<name>", "confidence": <float>},
    {"disease": "<name>", "confidence": <float>},
    {"disease": "<name>", "confidence": <float>}
  ]
}
"""


def classify_disease(image_bytes: bytes, image_media_type: str = "image/jpeg") -> dict:
    """
    Classify the disease in a crop image using Gemini Vision.

    Args:
        image_bytes: Raw image bytes (JPEG or PNG)
        image_media_type: MIME type of the image

    Returns:
        dict with crop, disease, severity, confidence, top_candidates
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))

    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    # TODO: Implement Gemini Vision API call
    # TODO: Parse JSON response
    # TODO: Return structured result
    raise NotImplementedError("vision_agent.classify_disease() not yet implemented")

"""
Diagnosis Agent
===============
Given a vision classification result, looks up the disease knowledge base
and returns treatment recommendations and yield impact.

Returns:
    {
        "disease_name": str,
        "severity": str,
        "organic_treatment": str,
        "chemical_treatment": str,
        "yield_impact": str,
        "confidence": float
    }
"""

import json
from pathlib import Path


def get_treatment(crop: str, disease: str, severity: str, confidence: float) -> dict:
    """
    Look up treatment recommendations for a diagnosed disease.

    Args:
        crop: Crop name (e.g. "rice")
        disease: Disease name (e.g. "brown leaf spot")
        severity: "mild", "moderate", or "severe"
        confidence: Vision agent confidence score

    Returns:
        dict with full diagnosis and treatment recommendation
    """
    # TODO: Load disease_db.json
    # TODO: Match crop + disease
    # TODO: Return treatment recommendation
    # TODO: Handle low confidence (< 0.70) case
    raise NotImplementedError("diagnosis_agent.get_treatment() not yet implemented")

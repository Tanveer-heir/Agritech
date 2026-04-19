"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel
from typing import Optional, List


class DiagnoseRequest(BaseModel):
    text: Optional[str] = None
    language: str = "hi"
    pin_code: Optional[str] = None


class CandidateDisease(BaseModel):
    disease: str
    confidence: float


class DiagnoseResponse(BaseModel):
    crop: str
    disease: str
    severity: str
    confidence: float
    top_candidates: List[CandidateDisease]
    organic_treatment: str
    chemical_treatment: str
    yield_impact: str
    kvk_name: Optional[str] = None
    kvk_phone: Optional[str] = None
    response_text: str  # Formatted response in farmer's language

"""
Crop Doctor Agents
==================
LangGraph-based multi-agent pipeline for crop disease diagnosis.

Agents:
  vision_agent     → Claude Vision API: image → disease classification
  diagnosis_agent  → Knowledge base: disease → treatment recommendations
  language_agent   → IndicTrans2: English response → farmer's language
  location_agent   → KVK directory: PIN code → nearest KVK contact
"""

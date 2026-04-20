"""
agents/diagnosis_agent.py — Crop Doctor Diagnosis Pipeline
===========================================================

Receives a VisionResult from vision_agent.py and returns a fully assembled
DiagnosisResult containing treatment plan, yield impact, urgency, and
confidence metadata — in a single unified object that language_agent.py
can translate and the API layer can send to Twilio.

Data flow:
    VisionResult (from vision_agent)
        │
        ▼
    _build_db_key()          ← normalise crop+disease → "Crop___Disease_name"
        │
        ├─ key found in disease_db.json ──► _build_from_db()
        │                                       │
        │                                       └─► DiagnosisResult ✓
        │
        └─ key NOT in DB ──► GeminiDiagnosisFallback.fetch()
                                   │
                                   └─► DiagnosisResult (from Gemini JSON)

Usage (standalone test):
    python agents/diagnosis_agent.py --crop Tomato --disease "Early Blight" --confidence 0.94
    python agents/diagnosis_agent.py --crop Banana --disease "Panama Disease" --source gemini_only

Environment variables (.env):
    DISEASE_DB_PATH=./data/disease_db.json   (default)
    GOOGLE_API_KEY=...                        (for Gemini fallback)

Dependencies:
    pip install google-generativeai python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed — Gemini fallback disabled.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosisResult:
    """
    Single output object from DiagnosisAgent.
    language_agent.py receives this and translates the text fields.
    api/routes/webhook.py serialises this via .to_dict().
    """

    # ── identity ──────────────────────────────────────────────────────────────
    crop:             str          # "Tomato"
    disease:          str          # "Early Blight"  or  "Healthy"
    is_healthy:       bool

    # ── confidence chain ──────────────────────────────────────────────────────
    vision_confidence:  float      # raw from VisionResult
    diagnosis_source:   str        # "db" | "db+fuzzy" | "gemini_fallback"
    db_hit:             bool       # True if disease_db had an entry

    # ── clinical picture ──────────────────────────────────────────────────────
    disease_type:       str        # "Fungal" | "Bacterial" | "Viral" | "healthy" | "Unknown"
    symptoms:           str
    conditions_favoring: str
    severity:           str        # "none" | "mild" | "moderate" | "severe"
    urgency:            str        # "none" | "monitor" | "act_this_week" | "act_today"
    affected_parts:     list[str]
    yield_loss_pct_max: int        # 0–100

    # ── treatment ─────────────────────────────────────────────────────────────
    organic_treatment:   dict      # {method, preparation, frequency, notes}
    chemical_treatment:  dict      # {primary: {...}, alternative: {...}}  (may be {})
    prevention:          str

    # ── safe harvest / legal ──────────────────────────────────────────────────
    safe_to_harvest_days: int      # 0 = no restriction / N/A for healthy
    seasons_affected:     list[str]

    # ── optional enrichment ───────────────────────────────────────────────────
    scientific_name:    str = ""
    hindi_name:         str = ""
    government_scheme:  str = ""
    icar_ref:           str = ""

    # ── meta ──────────────────────────────────────────────────────────────────
    latency_ms:         float = 0.0
    top3_candidates:    list[dict] = field(default_factory=list)
    visual_evidence:    str = ""   # forwarded from VisionResult
    gemini_reasoning:   str = ""   # forwarded from VisionResult (if Gemini ran vision)

    # ── fallback flags ────────────────────────────────────────────────────────
    low_confidence:           bool = False  # True if vision_confidence < threshold
    show_multiple_candidates: bool = False  # True if bot should list top-3 to farmer

    def urgency_emoji(self) -> str:
        return {
            "act_today":     "🔴",
            "act_this_week": "🟡",
            "monitor":       "🟢",
            "none":          "✅",
        }.get(self.urgency, "⚠️")

    def to_dict(self) -> dict:
        return {
            "crop":                   self.crop,
            "disease":                self.disease,
            "is_healthy":             self.is_healthy,
            "vision_confidence":      round(self.vision_confidence, 4),
            "diagnosis_source":       self.diagnosis_source,
            "db_hit":                 self.db_hit,
            "disease_type":           self.disease_type,
            "symptoms":               self.symptoms,
            "conditions_favoring":    self.conditions_favoring,
            "severity":               self.severity,
            "urgency":                self.urgency,
            "urgency_emoji":          self.urgency_emoji(),
            "affected_parts":         self.affected_parts,
            "yield_loss_pct_max":     self.yield_loss_pct_max,
            "organic_treatment":      self.organic_treatment,
            "chemical_treatment":     self.chemical_treatment,
            "prevention":             self.prevention,
            "safe_to_harvest_days":   self.safe_to_harvest_days,
            "seasons_affected":       self.seasons_affected,
            "scientific_name":        self.scientific_name,
            "hindi_name":             self.hindi_name,
            "government_scheme":      self.government_scheme,
            "icar_ref":               self.icar_ref,
            "latency_ms":             round(self.latency_ms, 1),
            "top3_candidates":        self.top3_candidates,
            "visual_evidence":        self.visual_evidence,
            "gemini_reasoning":       self.gemini_reasoning,
            "low_confidence":         self.low_confidence,
            "show_multiple_candidates": self.show_multiple_candidates,
        }

    def format_whatsapp_reply(self) -> str:
        """
        Produce the English WhatsApp reply string.
        language_agent.py translates this string into the farmer's language.
        Kept short enough to read on a basic Android screen.
        """
        if self.is_healthy:
            return (
                f"✅ *{self.crop} — Healthy*\n"
                f"Your crop looks healthy. No disease detected.\n\n"
                f"🌱 *Tip:* {self.prevention}"
            )

        if self.show_multiple_candidates:
            candidates_str = "\n".join(
                f"  {i+1}. {c['disease']} ({c['confidence']:.0%})"
                for i, c in enumerate(self.top3_candidates[:3])
            )
            return (
                f"⚠️ *{self.crop} — Uncertain Diagnosis*\n"
                f"I can see disease symptoms but I need a clearer photo.\n\n"
                f"*Possible conditions:*\n{candidates_str}\n\n"
                f"📸 Please send a closer photo of the affected leaf in good light."
            )

        lines = [
            f"{self.urgency_emoji()} *{self.crop} — {self.disease}*",
            f"Severity: {self.severity.title()}  |  Type: {self.disease_type}",
            "",
            f"*What you see:* {self.symptoms}",
        ]

        if self.yield_loss_pct_max > 0:
            lines.append(f"*Risk:* Up to {self.yield_loss_pct_max}% yield loss if untreated.")

        lines += [
            "",
            f"🌿 *Organic option:* {self.organic_treatment.get('method', 'N/A')}",
            f"   {self.organic_treatment.get('preparation', '')}",
            f"   Apply: {self.organic_treatment.get('frequency', '')}",
        ]

        if note := self.organic_treatment.get("notes", ""):
            lines.append(f"   ⚠ {note}")

        if primary := self.chemical_treatment.get("primary", {}):
            lines += [
                "",
                f"💊 *Chemical option:* {primary.get('pesticide', 'N/A')}",
                f"   Dose: {primary.get('dose', '')}",
                f"   Frequency: {primary.get('frequency', '')}",
            ]

        if self.safe_to_harvest_days > 0:
            lines.append(
                f"\n⏱ Do not harvest for *{self.safe_to_harvest_days} days* after spraying."
            )

        lines += ["", f"🛡 *Prevention:* {self.prevention}"]

        if self.government_scheme:
            lines += ["", f"🏛 *Govt scheme:* {self.government_scheme}"]

        if self.icar_ref:
            lines.append(f"📋 Source: {self.icar_ref}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# KEY NORMALISATION
# Converts (crop, disease) strings from VisionResult into disease_db.json keys.
# The DB uses PlantVillage folder naming: "Tomato___Early_blight"
# VisionResult gives already-parsed human strings: crop="Tomato", disease="Early Blight"
# ══════════════════════════════════════════════════════════════════════════════

# Manual alias table for common mismatches between Gemini output and DB keys.
# Format: ("crop_lower", "disease_lower_approx") → exact DB key
_ALIAS_TABLE: dict[tuple[str, str], str] = {
    ("corn",       "gray leaf spot"):                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ("maize",      "gray leaf spot"):                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ("corn",       "cercospora leaf spot"):             "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ("maize",      "cercospora leaf spot"):             "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ("corn",       "common rust"):                     "Corn_(maize)___Common_rust_",
    ("maize",      "common rust"):                     "Corn_(maize)___Common_rust_",
    ("corn",       "northern leaf blight"):            "Corn_(maize)___Northern_Leaf_Blight",
    ("maize",      "northern leaf blight"):            "Corn_(maize)___Northern_Leaf_Blight",
    ("corn",       "healthy"):                         "Corn_(maize)___healthy",
    ("maize",      "healthy"):                         "Corn_(maize)___healthy",
    ("cherry",     "powdery mildew"):                  "Cherry_(including_sour)___Powdery_mildew",
    ("cherry",     "healthy"):                         "Cherry_(including_sour)___healthy",
    ("orange",     "greening"):                        "Orange___Haunglongbing_(Citrus_greening)",
    ("orange",     "huanglongbing"):                   "Orange___Haunglongbing_(Citrus_greening)",
    ("orange",     "citrus greening"):                 "Orange___Haunglongbing_(Citrus_greening)",
    ("orange",     "hlb"):                             "Orange___Haunglongbing_(Citrus_greening)",
    ("grape",      "black measles"):                   "Grape___Esca_(Black_Measles)",
    ("grape",      "esca"):                            "Grape___Esca_(Black_Measles)",
    ("grape",      "isariopsis leaf spot"):            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    ("grape",      "leaf blight"):                     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    ("pepper",     "bacterial spot"):                  "Pepper,_bell___Bacterial_spot",
    ("bell pepper","bacterial spot"):                  "Pepper,_bell___Bacterial_spot",
    ("tomato",     "target spot"):                     "Tomato___Target_Spot",
    ("tomato",     "yellow leaf curl"):                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ("tomato",     "tylcv"):                           "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ("tomato",     "mosaic virus"):                    "Tomato___Tomato_mosaic_virus",
    ("tomato",     "tomato mosaic virus"):             "Tomato___Tomato_mosaic_virus",
    ("wheat",      "leaf rust"):                       "Wheat___Leaf_rust",
    ("wheat",      "brown rust"):                      "Wheat___Leaf_rust",
    ("wheat",      "stripe rust"):                     "Wheat___Stripe_rust",
    ("wheat",      "yellow rust"):                     "Wheat___Stripe_rust",
    ("rice",       "leaf blast"):                      "Rice___Leaf_blast",
    ("rice",       "blast"):                           "Rice___Leaf_blast",
    ("rice",       "brown spot"):                      "Rice___Brown_spot",
    ("rice",       "neck blast"):                      "Rice___Neck_blast",
    ("rice",       "bacterial blight"):                "Rice___Bacterial_leaf_blight",
    ("rice",       "bacterial leaf blight"):           "Rice___Bacterial_leaf_blight",
    ("rice",       "hispa"):                           "Rice___Hispa",
    ("cotton",     "alternaria leaf spot"):            "Cotton___Alternaria_Leaf_Spot",
    ("cotton",     "bacterial blight"):                "Cotton___Bacterial_Blight",
    ("sugarcane",  "red rot"):                         "Sugarcane___Red_Rot",
    ("sugarcane",  "smut"):                            "Sugarcane___Smut",
    ("banana",     "panama disease"):                  "Banana___Panama_Disease",
    ("banana",     "sigatoka"):                        "Banana___Yellow_Sigatoka",
    ("banana",     "yellow sigatoka"):                 "Banana___Yellow_Sigatoka",
    ("chilli",     "anthracnose"):                     "Chilli___Anthracnose",
    ("chilli",     "powdery mildew"):                  "Chilli___Powdery_Mildew",
}


def _to_snake(s: str) -> str:
    """'Early Blight' → 'Early_blight'  (PlantVillage-style title+snake)"""
    words = s.strip().split()
    if not words:
        return ""
    return words[0].capitalize() + "_" + "_".join(w.lower() for w in words[1:])


def _build_db_key(crop: str, disease: str) -> str:
    """
    Attempt #1 — canonical PlantVillage key reconstruction.
    'Tomato', 'Early Blight' → 'Tomato___Early_blight'
    """
    crop_part    = crop.strip().replace(" ", "_")
    disease_part = _to_snake(disease)
    return f"{crop_part}___{disease_part}"


def _fuzzy_lookup(crop: str, disease: str, db: dict) -> Optional[str]:
    """
    Three-stage fuzzy lookup against all DB keys.

    Stage 1: alias table (hand-curated, zero-cost)
    Stage 2: normalised string match — strip punctuation, lowercase, compare
    Stage 3: substring match — disease words inside key
    Returns matching DB key or None.
    """
    crop_l    = crop.lower().strip()
    disease_l = disease.lower().strip()

    # Stage 1: alias table
    alias_key = _ALIAS_TABLE.get((crop_l, disease_l))
    if alias_key and alias_key in db:
        logger.debug("Alias hit: (%s, %s) → %s", crop_l, disease_l, alias_key)
        return alias_key

    # Stage 2: normalised match
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    norm_crop    = _norm(crop)
    norm_disease = _norm(disease)

    for key in db:
        norm_key = _norm(key)
        if norm_crop in norm_key and norm_disease in norm_key:
            logger.debug("Fuzzy (normalised) hit: (%s, %s) → %s", crop, disease, key)
            return key

    # Stage 3: substring — crop match + any disease word overlap (words > 3 chars)
    disease_words = [w for w in disease_l.split() if len(w) > 3]
    for key in db:
        norm_key = _norm(key)
        if norm_crop not in norm_key:
            continue
        for word in disease_words:
            if _norm(word) in norm_key:
                logger.debug("Fuzzy (substring) hit: (%s, %s) → %s", crop, disease, key)
                return key

    logger.info("No DB match for: crop=%r  disease=%r", crop, disease)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI DIAGNOSIS FALLBACK
# Called only when disease_db has no entry.
# Returns a structured dict that mirrors the disease_db schema exactly.
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_DIAGNOSIS_SYSTEM_PROMPT = """\
You are a senior agronomist and plant pathologist specialising in smallholder
farming in India. A crop disease AI has detected a disease that is NOT in its
knowledge base and needs your expert guidance.

Respond ONLY with a valid JSON object. No markdown, no preamble, no trailing text.

Required JSON format:
{
  "common_name": "Human-readable disease name",
  "scientific_name": "Latin binomial or empty string",
  "hindi_name": "Hindi name or empty string",
  "disease_type": "Fungal" | "Bacterial" | "Viral" | "Pest" | "Nutritional" | "Unknown",
  "symptoms": "2-3 sentence description of visible symptoms",
  "conditions_favoring": "Temperature/humidity/season conditions that make this worse",
  "urgency": "act_today" | "act_this_week" | "monitor" | "none",
  "severity_on_detection": "none" | "mild" | "moderate" | "severe",
  "yield_loss_pct_max": integer 0-100,
  "organic_treatment": {
    "method": "Short name of organic/cultural approach",
    "preparation": "Specific preparation with doses (e.g. neem oil 5ml per litre)",
    "frequency": "How often to apply",
    "notes": "Any safety warnings or important caveats"
  },
  "chemical_treatment": {
    "primary": {
      "pesticide": "Name and formulation (e.g. Mancozeb 75% WP)",
      "dose": "Dose per litre of water or per hectare",
      "frequency": "Application schedule",
      "icar_ref": "ICAR or CIB&RC reference if known, else empty string"
    }
  },
  "prevention": "2-3 specific preventive measures",
  "safe_to_harvest_days": integer days to wait after chemical spray before harvest,
  "seasons_affected": list of strings e.g. ["kharif"] or ["rabi"] or ["all"],
  "affected_plant_parts": list of strings e.g. ["leaves", "fruit", "stem"],
  "government_scheme": "Any relevant government scheme or empty string"
}

Rules:
- All pesticide recommendations must follow CIB&RC-approved labels for India.
- If the disease has no chemical cure (e.g. viral), set primary.pesticide to
  "No effective chemical - see organic treatment" and dose/frequency to "N/A".
- Be conservative: if uncertain, lower urgency rather than escalating.
- Doses must be in per-litre or per-hectare format (not "as needed").
- Focus on treatments available to smallholder farmers in rural India.
"""


class GeminiDiagnosisFallback:
    """
    Called when disease_db.json has no entry for the detected crop+disease.
    Asks Gemini to act as an agronomist and generate a treatment plan that
    matches the disease_db schema, so DiagnosisAgent can use it identically
    to a DB hit.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed.\n"
                "Run: pip install google-generativeai"
            )

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Add GOOGLE_API_KEY=... to your .env file."
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=_GEMINI_DIAGNOSIS_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,       # near-deterministic for medical-like advice
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        logger.info("GeminiDiagnosisFallback initialised (gemini-1.5-flash)")

    def fetch(
        self,
        crop: str,
        disease: str,
        vision_confidence: float,
        visual_evidence: str = "",
        affected_parts: list[str] | None = None,
    ) -> dict:
        """
        Fetch a treatment plan from Gemini for an unknown crop+disease.

        Args:
            crop:               e.g. "Banana"
            disease:            e.g. "Panama Disease"
            vision_confidence:  confidence from VisionResult
            visual_evidence:    what the vision model saw in the image
            affected_parts:     plant parts observed by vision model

        Returns:
            dict matching disease_db.json schema — same keys, same structure.
        """
        parts_str = ", ".join(affected_parts or []) or "not specified"
        conf_pct  = f"{vision_confidence:.0%}"

        prompt = (
            f"Crop: {crop}\n"
            f"Disease detected: {disease}\n"
            f"Vision model confidence: {conf_pct}\n"
            f"Affected plant parts observed: {parts_str}\n"
        )
        if visual_evidence:
            prompt += f"Visual evidence from image: {visual_evidence}\n"

        prompt += (
            "\nProvide a complete agronomic treatment plan for this disease "
            "in the JSON format specified in your system instructions.\n"
            "Focus on treatments available to smallholder farmers in India."
        )

        logger.info(
            "Calling Gemini diagnosis fallback: crop=%r  disease=%r",
            crop, disease,
        )

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            # Strip markdown fences if Gemini adds them despite response_mime_type
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            result = json.loads(raw_text)
            logger.info(
                "Gemini diagnosis fallback: success  keys=%s",
                list(result.keys()),
            )
            return result

        except json.JSONDecodeError as e:
            logger.error("Gemini diagnosis returned non-JSON: %s", e)
            return _emergency_unknown_fallback(crop, disease)
        except Exception as e:
            logger.error("Gemini diagnosis API error: %s", e)
            return _emergency_unknown_fallback(crop, disease)


def _emergency_unknown_fallback(crop: str, disease: str) -> dict:
    """
    Last-resort fallback when both DB lookup and Gemini fail.
    Returns a safe, honest 'seek expert help' response rather than crashing.
    """
    return {
        "common_name":           f"{disease} (unconfirmed)",
        "scientific_name":       "",
        "hindi_name":            "",
        "disease_type":          "Unknown",
        "symptoms":              (
            "Disease symptoms detected but specific identification failed. "
            "Please send a clearer photo or contact your nearest KVK."
        ),
        "conditions_favoring":   "Unknown — consult an agronomist.",
        "urgency":               "act_this_week",
        "severity_on_detection": "moderate",
        "yield_loss_pct_max":    30,
        "organic_treatment": {
            "method":      "Isolate affected plants",
            "preparation": "Remove and bag affected leaves. Do not compost.",
            "frequency":   "Immediately",
            "notes":       "Seek expert advice before applying any treatment.",
        },
        "chemical_treatment": {},
        "prevention":            (
            "Contact your nearest Krishi Vigyan Kendra (KVK) for a field visit."
        ),
        "safe_to_harvest_days":  0,
        "seasons_affected":      [],
        "affected_plant_parts":  [],
        "government_scheme":     "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULT BUILDERS
# These functions assemble DiagnosisResult from either DB data or Gemini data,
# always combining with metadata forwarded from VisionResult.
# ══════════════════════════════════════════════════════════════════════════════

def _build_from_db(
    db_entry:         dict,
    vision_result:    object,
    db_key:           str,
    diagnosis_source: str,
) -> DiagnosisResult:
    """
    Build a DiagnosisResult from a disease_db.json entry + VisionResult.

    VisionResult fields used:
        .crop, .disease, .is_healthy, .confidence, .severity
        .visual_evidence, .top3, .gemini_reasoning, .affected_parts

    Merging rules:
        severity    → take the HIGHER of DB severity vs vision severity (err cautious)
        affected_parts → union of vision_result.affected_parts + DB affected_plant_parts
        urgency     → always from DB (vision model has no urgency signal)
    """
    severity_rank = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    db_sev   = db_entry.get("severity_on_detection", "mild")
    vis_sev  = getattr(vision_result, "severity", "mild")
    severity = (
        vis_sev
        if severity_rank.get(vis_sev, 0) >= severity_rank.get(db_sev, 0)
        else db_sev
    )

    # Merge affected parts: vision-observed first (specific), then DB defaults
    db_parts   = db_entry.get("affected_plant_parts", [])
    vis_parts  = getattr(vision_result, "affected_parts", [])
    merged_parts = list(dict.fromkeys(vis_parts + db_parts))

    icar_ref = (
        db_entry
        .get("chemical_treatment", {})
        .get("primary", {})
        .get("icar_ref", "")
    )

    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    confidence = getattr(vision_result, "confidence", 1.0)
    low_conf   = confidence < confidence_threshold
    top3       = getattr(vision_result, "top3", [])
    show_top3  = low_conf and len(top3) >= 2

    return DiagnosisResult(
        crop=                db_entry.get("crop", vision_result.crop),
        disease=             db_entry.get("common_name", vision_result.disease),
        is_healthy=          getattr(vision_result, "is_healthy", False),
        vision_confidence=   confidence,
        diagnosis_source=    diagnosis_source,
        db_hit=              True,
        disease_type=        db_entry.get("type", "Unknown"),
        symptoms=            db_entry.get("symptoms", ""),
        conditions_favoring= db_entry.get("conditions_favoring", ""),
        severity=            severity,
        urgency=             db_entry.get("urgency", "monitor"),
        affected_parts=      merged_parts,
        yield_loss_pct_max=  int(db_entry.get("yield_loss_pct_max", 0)),
        organic_treatment=   db_entry.get("organic_treatment", {}),
        chemical_treatment=  db_entry.get("chemical_treatment", {}),
        prevention=          db_entry.get("prevention", ""),
        safe_to_harvest_days=int(db_entry.get("safe_to_harvest_days", 0)),
        seasons_affected=    db_entry.get("seasons_affected", []),
        scientific_name=     db_entry.get("scientific_name", ""),
        hindi_name=          db_entry.get("hindi_name", ""),
        government_scheme=   db_entry.get("government_scheme", ""),
        icar_ref=            icar_ref,
        top3_candidates=     top3,
        visual_evidence=     getattr(vision_result, "visual_evidence", ""),
        gemini_reasoning=    getattr(vision_result, "gemini_reasoning", ""),
        low_confidence=      low_conf,
        show_multiple_candidates= show_top3,
    )


def _build_from_gemini(
    gemini_data:   dict,
    vision_result: object,
) -> DiagnosisResult:
    """
    Build a DiagnosisResult from a Gemini diagnosis response.
    Gemini is expected to return the same schema as disease_db entries.
    """
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    confidence = getattr(vision_result, "confidence", 0.0)
    low_conf   = confidence < confidence_threshold
    top3       = getattr(vision_result, "top3", [])
    show_top3  = low_conf and len(top3) >= 2

    gem_parts  = gemini_data.get("affected_plant_parts", [])
    vis_parts  = getattr(vision_result, "affected_parts", [])
    merged_parts = list(dict.fromkeys(vis_parts + gem_parts))

    return DiagnosisResult(
        crop=                getattr(vision_result, "crop", "Unknown"),
        disease=             gemini_data.get("common_name", getattr(vision_result, "disease", "Unknown")),
        is_healthy=          getattr(vision_result, "is_healthy", False),
        vision_confidence=   confidence,
        diagnosis_source=    "gemini_fallback",
        db_hit=              False,
        disease_type=        gemini_data.get("disease_type", "Unknown"),
        symptoms=            gemini_data.get("symptoms", ""),
        conditions_favoring= gemini_data.get("conditions_favoring", ""),
        severity=            gemini_data.get("severity_on_detection", getattr(vision_result, "severity", "mild")),
        urgency=             gemini_data.get("urgency", "act_this_week"),
        affected_parts=      merged_parts,
        yield_loss_pct_max=  int(gemini_data.get("yield_loss_pct_max", 0)),
        organic_treatment=   gemini_data.get("organic_treatment", {}),
        chemical_treatment=  gemini_data.get("chemical_treatment", {}),
        prevention=          gemini_data.get("prevention", ""),
        safe_to_harvest_days=int(gemini_data.get("safe_to_harvest_days", 0)),
        seasons_affected=    gemini_data.get("seasons_affected", []),
        scientific_name=     gemini_data.get("scientific_name", ""),
        hindi_name=          gemini_data.get("hindi_name", ""),
        government_scheme=   gemini_data.get("government_scheme", ""),
        icar_ref=            gemini_data.get("chemical_treatment", {})
                                        .get("primary", {})
                                        .get("icar_ref", ""),
        top3_candidates=     top3,
        visual_evidence=     getattr(vision_result, "visual_evidence", ""),
        gemini_reasoning=    getattr(vision_result, "gemini_reasoning", ""),
        low_confidence=      low_conf,
        show_multiple_candidates= show_top3,
    )


def _build_healthy(vision_result: object) -> DiagnosisResult:
    """
    Healthy plants skip the DB entirely.
    Returns a pre-filled DiagnosisResult with no treatment needed.
    """
    return DiagnosisResult(
        crop=                getattr(vision_result, "crop", "Unknown"),
        disease=             "Healthy",
        is_healthy=          True,
        vision_confidence=   getattr(vision_result, "confidence", 1.0),
        diagnosis_source=    "db",
        db_hit=              True,
        disease_type=        "healthy",
        symptoms=            "No disease symptoms detected.",
        conditions_favoring= "",
        severity=            "none",
        urgency=             "none",
        affected_parts=      [],
        yield_loss_pct_max=  0,
        organic_treatment= {
            "method":      "No treatment needed",
            "preparation": "",
            "frequency":   "",
            "notes":       "Continue regular crop monitoring and weekly scouting.",
        },
        chemical_treatment=  {},
        prevention=          "Weekly scouting. Balanced fertilization. Crop rotation every season.",
        safe_to_harvest_days=0,
        seasons_affected=    [],
        visual_evidence=     getattr(vision_result, "visual_evidence", ""),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DiagnosisAgent:
    """
    Receives a VisionResult from vision_agent.py, performs DB lookup,
    calls Gemini if the disease is unknown, and returns a DiagnosisResult.

    The agent is stateless after __init__ — safe for concurrent FastAPI requests.

    Example (from api/routes/webhook.py):
        agent = DiagnosisAgent()
        diagnosis = agent.diagnose(vision_result)
        reply_text = diagnosis.format_whatsapp_reply()
    """

    def __init__(
        self,
        db_path:        Optional[str] = None,
        google_api_key: Optional[str] = None,
        disable_gemini: bool = False,
    ):
        """
        Args:
            db_path:        Path to disease_db.json. Falls back to $DISEASE_DB_PATH
                            then ./data/disease_db.json.
            google_api_key: Overrides $GOOGLE_API_KEY.
            disable_gemini: True disables the Gemini fallback entirely (for testing).
        """
        resolved = db_path or os.getenv("DISEASE_DB_PATH") or "./data/disease_db.json"
        self._db_path = Path(resolved)

        if not self._db_path.exists():
            raise FileNotFoundError(
                f"disease_db.json not found at '{self._db_path}'.\n"
                "Expected location: ./data/disease_db.json\n"
                "Make sure setup_repo.sh has been run."
            )

        with open(self._db_path, encoding="utf-8") as f:
            raw = json.load(f)

        # Strip _meta — pure data dict, keyed by "Crop___Disease"
        self._db: dict = {k: v for k, v in raw.items() if not k.startswith("_")}

        logger.info(
            "DiagnosisAgent loaded: %d disease entries  path=%s",
            len(self._db), self._db_path,
        )

        # Gemini fallback — graceful init failure is non-fatal
        self._gemini: Optional[GeminiDiagnosisFallback] = None
        if not disable_gemini:
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if api_key and _GEMINI_AVAILABLE:
                try:
                    self._gemini = GeminiDiagnosisFallback(api_key=api_key)
                except Exception as e:
                    logger.warning(
                        "GeminiDiagnosisFallback init failed (non-fatal): %s", e
                    )
            else:
                logger.info(
                    "Gemini diagnosis fallback disabled "
                    "(GOOGLE_API_KEY not set or package not installed)."
                )

    # ── main entry point ──────────────────────────────────────────────────────

    def diagnose(self, vision_result: object) -> DiagnosisResult:
        """
        Core method. Accepts a VisionResult from vision_agent.py.

        The lookup strategy is:
          1. Healthy fast path (no DB needed)
          2. Canonical PlantVillage key match  → DB
          3. Fuzzy lookup (alias + normalised + substring)  → DB
          4. Gemini agronomist fallback  → generated treatment plan
          5. Emergency static fallback  → 'contact KVK' message

        Args:
            vision_result: VisionResult object (or any object with .crop,
                           .disease, .is_healthy, .confidence, .severity,
                           .visual_evidence, .affected_parts, .top3,
                           .gemini_reasoning).

        Returns:
            DiagnosisResult — fully populated, ready for language_agent.
        """
        t_start = time.perf_counter()

        # ── 0. Healthy fast path ──────────────────────────────────────────────
        if getattr(vision_result, "is_healthy", False):
            result = _build_healthy(vision_result)
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "DiagnosisAgent: healthy  crop=%r  latency=%.1fms",
                getattr(vision_result, "crop", "?"), result.latency_ms,
            )
            return result

        crop    = getattr(vision_result, "crop",    "Unknown")
        disease = getattr(vision_result, "disease", "Unknown")

        # ── 1. Canonical key (PlantVillage naming) ────────────────────────────
        canonical = _build_db_key(crop, disease)
        db_entry  = self._db.get(canonical)

        if db_entry:
            result = _build_from_db(db_entry, vision_result, canonical, "db")
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "DiagnosisAgent: DB hit (canonical)  key=%r  latency=%.1fms",
                canonical, result.latency_ms,
            )
            return result

        # ── 2. Fuzzy lookup (alias → normalised → substring) ──────────────────
        fuzzy_key = _fuzzy_lookup(crop, disease, self._db)
        if fuzzy_key:
            db_entry = self._db[fuzzy_key]
            result   = _build_from_db(db_entry, vision_result, fuzzy_key, "db+fuzzy")
            result.latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "DiagnosisAgent: DB hit (fuzzy)  key=%r  latency=%.1fms",
                fuzzy_key, result.latency_ms,
            )
            return result

        # ── 3. DB miss → Gemini fallback ──────────────────────────────────────
        logger.info(
            "DiagnosisAgent: DB miss  crop=%r  disease=%r  → Gemini fallback",
            crop, disease,
        )

        if self._gemini:
            gemini_data = self._gemini.fetch(
                crop=              crop,
                disease=           disease,
                vision_confidence= getattr(vision_result, "confidence", 0.0),
                visual_evidence=   getattr(vision_result, "visual_evidence", ""),
                affected_parts=    getattr(vision_result, "affected_parts", []),
            )
        else:
            # ── 4. No Gemini → emergency static fallback ──────────────────────
            logger.warning(
                "Gemini not configured — emergency fallback for %r / %r",
                crop, disease,
            )
            gemini_data = _emergency_unknown_fallback(crop, disease)

        result = _build_from_gemini(gemini_data, vision_result)
        result.latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "DiagnosisAgent: Gemini fallback complete  latency=%.1fms",
            result.latency_ms,
        )
        return result

    def diagnose_from_dict(self, vision_dict: dict) -> DiagnosisResult:
        """
        Convenience wrapper — accepts the dict output of VisionResult.to_dict()
        instead of the VisionResult object itself. Useful when crossing API
        boundaries (e.g. vision service and diagnosis service are separate).
        """

        class _Proxy:
            def __init__(self, d: dict):
                self.crop             = d.get("crop", "Unknown")
                self.disease          = d.get("disease", "Unknown")
                self.is_healthy       = d.get("is_healthy", False)
                self.confidence       = float(d.get("confidence", 0.0))
                self.severity         = d.get("severity", "mild")
                self.visual_evidence  = d.get("visual_evidence", "")
                self.affected_parts   = d.get("affected_parts", [])
                self.top3             = d.get("top3", [])
                self.gemini_reasoning = d.get("gemini_reasoning", "")

        return self.diagnose(_Proxy(vision_dict))

    def list_known_diseases(self) -> list[dict]:
        """Return all DB entries as a summary list — useful for debugging."""
        return [
            {
                "key":         k,
                "crop":        v.get("crop", ""),
                "common_name": v.get("common_name", ""),
                "urgency":     v.get("urgency", ""),
                "type":        v.get("type", ""),
            }
            for k, v in self._db.items()
        ]


# ══════════════════════════════════════════════════════════════════════════════
# CLI — quick test without the full FastAPI server
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="Test diagnosis_agent.py from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (DB hits):
  python agents/diagnosis_agent.py --crop Tomato --disease "Early Blight" --confidence 0.94
  python agents/diagnosis_agent.py --crop Wheat  --disease "Stripe Rust"  --confidence 0.97
  python agents/diagnosis_agent.py --crop Rice   --disease Blast          --confidence 0.91
  python agents/diagnosis_agent.py --crop Tomato --disease Healthy        --confidence 0.99

Examples (Gemini fallback — crops not in DB):
  python agents/diagnosis_agent.py --crop Banana     --disease "Panama Disease" --confidence 0.82
  python agents/diagnosis_agent.py --crop Sugarcane  --disease "Red Rot"        --confidence 0.77

List all known diseases:
  python agents/diagnosis_agent.py --list-diseases
        """,
    )
    parser.add_argument("--crop",           default="Tomato")
    parser.add_argument("--disease",        default="Early Blight")
    parser.add_argument("--confidence",     type=float, default=0.88)
    parser.add_argument("--severity",       default="moderate")
    parser.add_argument("--evidence",       default="")
    parser.add_argument("--affected-parts", nargs="*", default=[])
    parser.add_argument("--db",             default=None)
    parser.add_argument("--no-gemini",      action="store_true")
    parser.add_argument("--list-diseases",  action="store_true")
    parser.add_argument("--json",           action="store_true", help="Print full JSON output")
    parser.add_argument("--verbose", "-v",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    agent = DiagnosisAgent(db_path=args.db, disable_gemini=args.no_gemini)

    if args.list_diseases:
        diseases = agent.list_known_diseases()
        print(f"\n📋 Known diseases ({len(diseases)} entries):\n")
        for d in diseases:
            urgency_icons = {
                "act_today":     "🔴",
                "act_this_week": "🟡",
                "monitor":       "🟢",
                "none":          "✅",
            }
            icon = urgency_icons.get(d["urgency"], "⚠️")
            print(f"  {icon}  {d['crop']:<15} {d['common_name']:<40}  [{d['type']}]")
        return

    class _MockVisionResult:
        crop             = args.crop
        disease          = args.disease
        is_healthy       = "healthy" in args.disease.lower()
        confidence       = args.confidence
        severity         = args.severity
        visual_evidence  = args.evidence
        affected_parts   = args.affected_parts or []
        top3             = []
        gemini_reasoning = ""

    result = agent.diagnose(_MockVisionResult())

    print(f"\n{'─'*60}")
    print(f"  Crop           : {result.crop}")
    print(f"  Disease        : {result.disease}")
    print(f"  Type           : {result.disease_type}")
    print(f"  Severity       : {result.severity}")
    print(f"  Urgency        : {result.urgency_emoji()}  {result.urgency}")
    print(f"  Yield risk     : up to {result.yield_loss_pct_max}% loss")
    print(f"  DB hit         : {result.db_hit}")
    print(f"  Source         : {result.diagnosis_source}")
    print(f"  Vision conf.   : {result.vision_confidence:.1%}")
    print(f"  Low confidence : {result.low_confidence}")
    print(f"  Latency        : {result.latency_ms:.1f}ms")
    if result.scientific_name:
        print(f"  Scientific     : {result.scientific_name}")
    if result.hindi_name:
        print(f"  Hindi name     : {result.hindi_name}")
    print(f"{'─'*60}")

    print("\n📱 WhatsApp reply (English):\n")
    print(result.format_whatsapp_reply())

    if args.json:
        print("\n\n📦 Full JSON:\n")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()

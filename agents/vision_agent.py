"""
agents/vision_agent.py — Crop Doctor Vision Pipeline
=====================================================
Architecture:
    image bytes
        │
        ▼
    EfficientNetV2-S (local, fast, ~5ms on GPU)
        │
        ├─ confidence ≥ 0.85 ──► structured result (done, no API cost)
        │
        └─ confidence < 0.85 ──► Gemini Vision (gemini-1.5-flash)
                                     │
                                     └─► structured result + visual reasoning

Usage (standalone test):
    python agents/vision_agent.py --image path/to/leaf.jpg
    python agents/vision_agent.py --image path/to/leaf.jpg --force-gemini

Environment variables required (in .env):
    GOOGLE_API_KEY=...          # for Gemini fallback
    MODEL_PATH=./models/efficientnetv2_crop_doctor.pth
    CLASS_MAPPING_PATH=./models/class_mapping.json
    CONFIDENCE_THRESHOLD=0.85   # optional override

Dependencies:
    pip install torch torchvision timm pillow python-dotenv google-generativeai
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── optional imports — fail gracefully so tests without API keys still work ──
try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    logging.warning("timm not installed. Run: pip install timm")

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Run: pip install google-generativeai")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS  ──  single contract for both EfficientNet + Gemini paths
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisionResult:
    """
    Unified output from either EfficientNetV2 or Gemini Vision.
    Downstream agents (diagnosis_agent, language_agent) only depend on this class.
    """
    # Core detection
    crop: str                       # e.g. "Tomato"
    disease: str                    # e.g. "Early Blight"  /  "Healthy"
    is_healthy: bool                # shortcut flag
    confidence: float               # 0.0 – 1.0
    severity: str                   # "none" | "mild" | "moderate" | "severe"

    # Routing metadata
    source: str                     # "efficientnet" | "gemini" | "gemini_only"
    needs_gemini: bool              # True if EfficientNet escalated
    latency_ms: float               # end-to-end time

    # Rich fields (populated by Gemini; partial from EfficientNet)
    visual_evidence: str = ""       # what the model actually saw
    affected_parts: list[str] = field(default_factory=list)
    top3: list[dict] = field(default_factory=list)

    # Gemini-only fields
    gemini_reasoning: str = ""      # raw Gemini explanation
    is_unknown_crop: bool = False   # Gemini detected something outside PlantVillage

    def to_dict(self) -> dict:
        return {
            "crop":              self.crop,
            "disease":           self.disease,
            "is_healthy":        self.is_healthy,
            "confidence":        round(self.confidence, 4),
            "severity":          self.severity,
            "source":            self.source,
            "needs_gemini":      self.needs_gemini,
            "latency_ms":        round(self.latency_ms, 1),
            "visual_evidence":   self.visual_evidence,
            "affected_parts":    self.affected_parts,
            "top3":              self.top3,
            "gemini_reasoning":  self.gemini_reasoning,
            "is_unknown_crop":   self.is_unknown_crop,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENTNETV2-S MODEL  ──  must match architecture in train.py exactly
# ══════════════════════════════════════════════════════════════════════════════

class CropDoctorNet(nn.Module):
    """
    Identical architecture to train.py — required for state_dict loading.
    Do NOT change this class without retraining.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required. Run: pip install timm")

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,        # weights loaded from .pth — no download
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features  # 1280

        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  ──  must match VAL_TRANSFORMS in train.py
# ══════════════════════════════════════════════════════════════════════════════

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENTNET PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class EfficientNetPredictor:
    """
    Loads the fine-tuned EfficientNetV2-S and runs fast local inference.
    Thread-safe — model weights are read-only after loading.
    """

    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load class mapping
        with open(class_mapping_path) as f:
            mapping = json.load(f)

        self.class_names: list[str]  = mapping["class_names"]
        self.num_classes: int        = mapping["num_classes"]
        self.img_size: int           = mapping["img_size"]
        self.threshold: float        = float(
            os.getenv("CONFIDENCE_THRESHOLD", mapping["confidence_threshold"])
        )

        # Build model and load weights
        self.model = CropDoctorNet(num_classes=self.num_classes).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(
            "EfficientNetV2-S loaded — %d classes  threshold=%.2f  device=%s",
            self.num_classes, self.threshold, self.device,
        )
        print(
            f"✅ EfficientNetV2-S loaded\n"
            f"   Classes   : {self.num_classes}\n"
            f"   Threshold : {self.threshold}\n"
            f"   Device    : {self.device}\n"
            f"   Model     : {model_path}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def parse_class_name(raw: str) -> tuple[str, str]:
        """'Tomato___Early_blight' → ('Tomato', 'Early Blight')"""
        parts   = raw.split("___")
        crop    = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Healthy"
        return crop, disease

    @staticmethod
    def severity_from_confidence(confidence: float, disease: str) -> str:
        """Heuristic: visible disease → high confidence → more severe."""
        if "healthy" in disease.lower():
            return "none"
        if confidence > 0.92:
            return "severe"
        if confidence > 0.80:
            return "moderate"
        return "mild"

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _INFERENCE_TRANSFORM(img).unsqueeze(0)  # (1, C, H, W)
        return tensor.to(self.device)

    # ── main predict ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """
        Returns:
            {
                crop, disease, confidence, needs_gemini,
                severity, top3: [{crop, disease, confidence}],
                is_healthy
            }
        """
        tensor = self.preprocess(image_bytes)

        with torch.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda"),
        ):
            logits = self.model(tensor)

        probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top3_idx = np.argsort(probs)[::-1][:3]

        top3 = []
        for idx in top3_idx:
            crop, disease = self.parse_class_name(self.class_names[idx])
            top3.append({
                "class":      self.class_names[idx],
                "crop":       crop,
                "disease":    disease,
                "confidence": float(probs[idx]),
            })

        best       = top3[0]
        confidence = best["confidence"]

        return {
            "crop":         best["crop"],
            "disease":      best["disease"],
            "confidence":   confidence,
            "needs_gemini": confidence < self.threshold,
            "severity":     self.severity_from_confidence(confidence, best["disease"]),
            "top3":         top3,
            "is_healthy":   "healthy" in best["disease"].lower(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI VISION FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM_PROMPT = """You are an expert agricultural plant pathologist with deep knowledge
of crop diseases in India, Africa, and SE Asia.

A farmer has sent a photo because an AI system was uncertain about the diagnosis.
Analyze the image carefully and respond ONLY with a valid JSON object — no markdown,
no explanation outside the JSON, no trailing text.

Required JSON format:
{
  "crop_detected": "crop common name (e.g. rice, cotton, tomato, wheat) or 'unknown'",
  "disease_name": "specific disease name or 'Healthy' if no disease visible",
  "disease_detected": true or false,
  "confidence": a float between 0.0 and 1.0,
  "severity": "none" | "mild" | "moderate" | "severe",
  "affected_parts": ["leaves", "stem", "fruit", "roots"] — include only visible parts,
  "visual_evidence": "2 sentences describing exactly what you see in the image that led to your diagnosis",
  "is_unknown_crop": true if the crop is not a common agricultural plant,
  "reasoning": "1 sentence explaining your diagnostic logic"
}

Rules:
- If the image is not a plant, set disease_detected: false, crop_detected: "not_a_plant".
- If you see multiple possible diseases, pick the most likely one and lower confidence.
- Be conservative — a wrong diagnosis could cost a farmer their harvest.
- confidence should reflect your genuine uncertainty, not just politeness."""


class GeminiVisionFallback:
    """
    Calls Gemini 1.5 Flash for zero-shot diagnosis on:
      - low-confidence EfficientNet predictions
      - crops not in PlantVillage training set
      - images flagged as ambiguous

    Uses gemini-1.5-flash (not pro) — cheaper and still excellent for vision.
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
                "GOOGLE_API_KEY not set. Add it to your .env file:\n"
                "  GOOGLE_API_KEY=your_key_here\n"
                "Get one at: https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=_GEMINI_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,       # near-deterministic for medical-like diagnosis
                max_output_tokens=512,
                response_mime_type="application/json",  # forces JSON output
            ),
        )
        logger.info("Gemini 1.5 Flash fallback initialised")

    def predict(self, image_bytes: bytes, efficientnet_hint: Optional[dict] = None) -> dict:
        """
        Run Gemini Vision on the image.

        Args:
            image_bytes: raw image bytes
            efficientnet_hint: optional dict from EfficientNetPredictor.predict()
                               — passed to Gemini as context to help it focus

        Returns:
            Parsed JSON dict matching _GEMINI_SYSTEM_PROMPT schema.
        """
        # Detect mime type
        img        = Image.open(io.BytesIO(image_bytes))
        mime_type  = "image/jpeg" if img.format in (None, "JPEG") else f"image/{img.format.lower()}"

        # Build prompt — include EfficientNet hint if available
        user_text = "Please diagnose the disease in this crop image."
        if efficientnet_hint:
            top3_str = ", ".join(
                f"{t['disease']} ({t['confidence']:.2f})"
                for t in efficientnet_hint.get("top3", [])
            )
            user_text = (
                f"Please diagnose the disease in this crop image.\n\n"
                f"Context: Our local model was uncertain. Its top-3 guesses were:\n"
                f"  {top3_str}\n\n"
                f"Use your own judgment — do not simply agree with these guesses."
            )

        image_part = {
            "mime_type": mime_type,
            "data":      base64.b64encode(image_bytes).decode("utf-8"),
        }

        try:
            response = self.model.generate_content([user_text, image_part])
            raw_text = response.text.strip()

            # Strip markdown code fences if Gemini adds them despite response_mime_type
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            result = json.loads(raw_text)
            return result

        except json.JSONDecodeError as e:
            logger.error("Gemini returned non-JSON: %s", str(e))
            # Return a safe fallback rather than crashing
            return {
                "crop_detected":   "unknown",
                "disease_name":    "Unable to parse response",
                "disease_detected": False,
                "confidence":       0.0,
                "severity":         "none",
                "affected_parts":   [],
                "visual_evidence":  "Gemini returned an unparseable response.",
                "is_unknown_crop":  True,
                "reasoning":        str(e),
            }
        except Exception as e:
            logger.error("Gemini API error: %s", str(e))
            raise


# ══════════════════════════════════════════════════════════════════════════════
# VISION AGENT  ──  the single entry point used by all other agents
# ══════════════════════════════════════════════════════════════════════════════

class VisionAgent:
    """
    Orchestrates the EfficientNet → (optional) Gemini pipeline.

    Initialization is lazy — models are loaded on first .analyze() call
    (or call .warmup() explicitly to pre-load on startup).

    Example:
        agent = VisionAgent()
        agent.warmup()

        with open("leaf.jpg", "rb") as f:
            result = agent.analyze(f.read())

        print(result.to_dict())
    """

    def __init__(
        self,
        model_path:          Optional[str] = None,
        class_mapping_path:  Optional[str] = None,
        google_api_key:      Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device:              Optional[str] = None,
    ):
        self._model_path     = model_path         or os.getenv("MODEL_PATH",         "./models/efficientnetv2_crop_doctor.pth")
        self._mapping_path   = class_mapping_path or os.getenv("CLASS_MAPPING_PATH", "./models/class_mapping.json")
        self._google_api_key = google_api_key     or os.getenv("GOOGLE_API_KEY")
        self._threshold_override = confidence_threshold
        self._device         = device

        self._eff_predictor: Optional[EfficientNetPredictor] = None
        self._gemini:        Optional[GeminiVisionFallback]  = None
        self._loaded         = False

    # ── lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return

        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self._model_path}'.\n"
                "Run train.py first:\n"
                "  python train.py --data ./data/plantvillage"
            )
        if not Path(self._mapping_path).exists():
            raise FileNotFoundError(
                f"Class mapping not found at '{self._mapping_path}'.\n"
                "Run train.py first."
            )

        self._eff_predictor = EfficientNetPredictor(
            model_path=self._model_path,
            class_mapping_path=self._mapping_path,
            device=self._device,
        )

        # Override threshold if provided
        if self._threshold_override is not None:
            self._eff_predictor.threshold = self._threshold_override

        # Gemini is optional — if key missing, we still work for high-conf cases
        if self._google_api_key and _GEMINI_AVAILABLE:
            try:
                self._gemini = GeminiVisionFallback(api_key=self._google_api_key)
            except Exception as e:
                logger.warning("Gemini init failed (non-fatal): %s", e)
                self._gemini = None
        else:
            logger.info("Gemini not configured — low-confidence images will return a warning flag.")

        self._loaded = True

    def warmup(self) -> None:
        """Pre-load models at startup to avoid cold-start latency on first request."""
        self._load()
        # Warm up CUDA kernels with a dummy forward pass
        if self._eff_predictor and self._eff_predictor.device.type == "cuda":
            dummy = torch.zeros(1, 3, 224, 224, device=self._eff_predictor.device)
            with torch.no_grad():
                self._eff_predictor.model(dummy)
            print("✅ GPU kernels warmed up")

    # ── main entry point ──────────────────────────────────────────────────────

    def analyze(
        self,
        image_bytes: bytes,
        force_gemini: bool = False,
    ) -> VisionResult:
        """
        Analyze an image and return a VisionResult.

        Args:
            image_bytes:  raw bytes of the image (JPEG, PNG, WebP)
            force_gemini: bypass EfficientNet and go straight to Gemini
                          (useful for testing or explicitly unknown crops)

        Returns:
            VisionResult — unified result from either model path
        """
        self._load()
        t_start = time.perf_counter()

        # ── Path A: Gemini only (force flag) ──────────────────────────────────
        if force_gemini:
            return self._run_gemini_only(image_bytes, t_start)

        # ── Path B: EfficientNet first ────────────────────────────────────────
        eff_result = self._eff_predictor.predict(image_bytes)

        if not eff_result["needs_gemini"]:
            # High confidence — EfficientNet is sufficient
            latency = (time.perf_counter() - t_start) * 1000
            return VisionResult(
                crop=              eff_result["crop"],
                disease=           eff_result["disease"],
                is_healthy=        eff_result["is_healthy"],
                confidence=        eff_result["confidence"],
                severity=          eff_result["severity"],
                source=            "efficientnet",
                needs_gemini=      False,
                latency_ms=        latency,
                top3=              eff_result["top3"],
                visual_evidence=   (
                    f"EfficientNetV2-S classified as {eff_result['crop']} — "
                    f"{eff_result['disease']} with {eff_result['confidence']:.1%} confidence."
                ),
                affected_parts=    [],   # EfficientNet doesn't localize
            )

        # ── Path C: Low confidence → escalate to Gemini ───────────────────────
        logger.info(
            "Low confidence (%.3f < %.3f) — escalating to Gemini Vision",
            eff_result["confidence"],
            self._eff_predictor.threshold,
        )

        if self._gemini is None:
            # Gemini not configured — return EfficientNet result with warning flag
            latency = (time.perf_counter() - t_start) * 1000
            return VisionResult(
                crop=            eff_result["crop"],
                disease=         eff_result["disease"],
                is_healthy=      eff_result["is_healthy"],
                confidence=      eff_result["confidence"],
                severity=        eff_result["severity"],
                source=          "efficientnet_low_conf",
                needs_gemini=    True,
                latency_ms=      latency,
                top3=            eff_result["top3"],
                visual_evidence= (
                    f"Low-confidence prediction ({eff_result['confidence']:.1%}). "
                    "Gemini fallback not configured — set GOOGLE_API_KEY to enable."
                ),
            )

        return self._run_gemini_escalation(image_bytes, eff_result, t_start)

    # ── private helpers ───────────────────────────────────────────────────────

    def _run_gemini_escalation(
        self,
        image_bytes: bytes,
        eff_result: dict,
        t_start: float,
    ) -> VisionResult:
        """EfficientNet was uncertain — call Gemini with its top-3 as context."""
        gemini_raw = self._gemini.predict(image_bytes, efficientnet_hint=eff_result)
        latency    = (time.perf_counter() - t_start) * 1000

        crop    = str(gemini_raw.get("crop_detected", "Unknown")).replace("_", " ").title()
        disease = str(gemini_raw.get("disease_name", "Unknown")).replace("_", " ").title()

        return VisionResult(
            crop=              crop,
            disease=           disease,
            is_healthy=        "healthy" in disease.lower(),
            confidence=        float(gemini_raw.get("confidence", 0.5)),
            severity=          str(gemini_raw.get("severity", "mild")),
            source=            "gemini",
            needs_gemini=      True,
            latency_ms=        latency,
            visual_evidence=   str(gemini_raw.get("visual_evidence", "")),
            affected_parts=    list(gemini_raw.get("affected_parts", [])),
            top3=              eff_result["top3"],   # keep EfficientNet top3 for reference
            gemini_reasoning=  str(gemini_raw.get("reasoning", "")),
            is_unknown_crop=   bool(gemini_raw.get("is_unknown_crop", False)),
        )

    def _run_gemini_only(self, image_bytes: bytes, t_start: float) -> VisionResult:
        """Direct Gemini path — no EfficientNet involved."""
        if self._gemini is None:
            raise RuntimeError(
                "force_gemini=True but Gemini is not configured.\n"
                "Set GOOGLE_API_KEY in your .env file."
            )
        gemini_raw = self._gemini.predict(image_bytes, efficientnet_hint=None)
        latency    = (time.perf_counter() - t_start) * 1000

        crop    = str(gemini_raw.get("crop_detected", "Unknown")).replace("_", " ").title()
        disease = str(gemini_raw.get("disease_name", "Unknown")).replace("_", " ").title()

        return VisionResult(
            crop=              crop,
            disease=           disease,
            is_healthy=        "healthy" in disease.lower(),
            confidence=        float(gemini_raw.get("confidence", 0.5)),
            severity=          str(gemini_raw.get("severity", "mild")),
            source=            "gemini_only",
            needs_gemini=      True,
            latency_ms=        latency,
            visual_evidence=   str(gemini_raw.get("visual_evidence", "")),
            affected_parts=    list(gemini_raw.get("affected_parts", [])),
            top3=              [],
            gemini_reasoning=  str(gemini_raw.get("reasoning", "")),
            is_unknown_crop=   bool(gemini_raw.get("is_unknown_crop", False)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI — quick test without running the full FastAPI server
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="Test vision_agent.py from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agents/vision_agent.py --image ./test_images/tomato_leaf.jpg
  python agents/vision_agent.py --image ./test_images/unknown_crop.jpg --force-gemini
  python agents/vision_agent.py --image leaf.jpg --threshold 0.70
        """,
    )
    parser.add_argument("--image",     required=True, help="Path to image file")
    parser.add_argument("--model",     default=None,  help="Override MODEL_PATH")
    parser.add_argument("--mapping",   default=None,  help="Override CLASS_MAPPING_PATH")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override confidence threshold (default: from class_mapping.json)")
    parser.add_argument("--force-gemini", action="store_true",
                        help="Skip EfficientNet and go straight to Gemini")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    agent = VisionAgent(
        model_path=args.model,
        class_mapping_path=args.mapping,
        confidence_threshold=args.threshold,
    )
    agent.warmup()

    print(f"\n🌿 Analysing: {img_path.name}")
    print("─" * 50)

    result = agent.analyze(image_bytes, force_gemini=args.force_gemini)

    print(f"  Crop          : {result.crop}")
    print(f"  Disease       : {result.disease}")
    print(f"  Healthy       : {result.is_healthy}")
    print(f"  Confidence    : {result.confidence:.4f}")
    print(f"  Severity      : {result.severity}")
    print(f"  Source        : {result.source}")
    print(f"  Latency       : {result.latency_ms:.1f} ms")
    print(f"  Visual note   : {result.visual_evidence}")
    if result.top3:
        print(f"\n  Top-3 (EfficientNet):")
        for i, t in enumerate(result.top3, 1):
            print(f"    {i}. {t['crop']} — {t['disease']:<30} {t['confidence']:.4f}")
    if result.gemini_reasoning:
        print(f"\n  Gemini reasoning: {result.gemini_reasoning}")
    if result.is_unknown_crop:
        print("\n  ⚠  Unknown crop — not in PlantVillage training set")

    print("\n  Full dict:")
    print(json.dumps(result.to_dict(), indent=4))


if __name__ == "__main__":
    _cli()

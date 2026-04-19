<div align="center">

<!-- Replace this comment with your hero GIF once recorded -->
<!-- ![Demo](docs/demo.gif) -->

# 🌾 Multilingual AI Crop Doctor

**AI crop disease diagnosis via WhatsApp — in Hindi, Tamil, Telugu, Bengali or English — for India's 700M+ smallholder farmers.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Gemini%20Vision-4285F4)](https://ai.google.dev)

<!-- Uncomment after deploying to Streamlit -->
<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL) -->

</div>

---

## 🔍 The Problem

Smallholder farmers produce 35% of the world's food but lose **20–40% of their crop yield** to preventable diseases every year. Existing AI tools — like Plantix — require stable 4G and work only in English, missing the **700M+ farmers** (FAO, 2025) who need them most. When a farmer in Vidarbha notices yellowing on their cotton leaves, they have no fast, accessible way to get a diagnosis. This project changes that.

## 💡 What This Does

A farmer sends a **photo of their diseased crop via WhatsApp or SMS**. An agentic vision + LLM pipeline identifies the disease, severity, and crop — then responds in the farmer's own language with:

- ✅ Disease name + severity level (mild / moderate / severe)
- ✅ Organic AND chemical treatment options
- ✅ Estimated yield impact
- ✅ Nearest Krishi Vigyan Kendra (KVK) contact

**No app install. No English required. Works on 2G.**

---

## 🏗️ Architecture

```
Farmer (WhatsApp / SMS)
        │
        ▼
  Twilio Webhook
        │
        ▼
  FastAPI Backend ──────────────────────────────────┐
        │                                            │
        ▼                                            ▼
  Vision Agent                               Location Agent
  (Gemini Vision API)                        (KVK PIN lookup)
        │
        ▼
  Diagnosis Agent
  (Disease KB + treatment logic)
        │
        ▼
  Language Agent
  (IndicTrans2 / Claude fallback)
        │
        ▼
  Twilio → WhatsApp / SMS reply
```

> Full architecture diagram: [`docs/architecture.png`](docs/architecture.png)

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Vision AI** | Gemini Vision API (Google) |
| **Agent Orchestration** | LangGraph |
| **Translation** | IndicTrans2 (AI4Bharat) — 22 Indian languages |
| **Backend** | FastAPI + Uvicorn |
| **Messaging** | Twilio WhatsApp Business API + SMS |
| **Frontend / Demo** | Streamlit |
| **Deployment** | Railway (Docker) |

---

## 🌿 Supported Crops & Diseases (MVP)

| Crop | Diseases Detected | Source |
|---|---|---|
| Rice (Paddy) | Blast, Brown Leaf Spot, Bacterial Blight, Healthy | PlantVillage |
| Wheat | Yellow Rust, Septoria Leaf Blotch, Loose Smut, Healthy | PlantVillage |
| Tomato | Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Healthy | PlantVillage |
| Potato | Early Blight, Late Blight, Healthy | PlantVillage |
| Maize | Common Rust, Cercospora Leaf Spot, Northern Blight, Healthy | PlantVillage |
| Soybean | Frogeye Leaf Spot, Healthy | PlantVillage |
| Cotton | Alternaria Leaf Spot, Bacterial Blight | Gemini Vision (zero-shot) |
| Sugarcane | Red Rot, Smut, Healthy | Gemini Vision (zero-shot) |
| Chilli | Anthracnose, Powdery Mildew, Healthy | Gemini Vision (zero-shot) |
| Banana | Panama Disease, Sigatoka, Healthy | Gemini Vision (zero-shot) |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- A [Twilio account](https://twilio.com) (free trial works)
- An [Anthropic API key](https://console.ai.google.dev)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-crop-doctor.git
cd multilingual-crop-doctor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see .env.example for all required vars)
```

### 3. Run Locally

```bash
# Start FastAPI backend
uvicorn api.main:app --reload --port 8000

# In a separate terminal, start Streamlit demo
streamlit run ui/app.py
```

### 4. Expose to Twilio (for WhatsApp testing)

```bash
# Install ngrok if not already installed
ngrok http 8000
# Copy the https URL and set it as your Twilio webhook:
# https://YOUR_NGROK_URL/webhook/whatsapp
```

---

## 🌍 Languages Supported

| Language | Code | Script | Status |
|---|---|---|---|
| Hindi | `hi` | Devanagari | ✅ Fully tested |
| Tamil | `ta` | Tamil | ✅ Fully tested |
| Telugu | `te` | Telugu | ✅ Fully tested |
| Bengali | `bn` | Bengali | ✅ Fully tested |
| English | `en` | Latin | ✅ Default |

To add a new language, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📁 Project Structure

```
multilingual-crop-doctor/
├── agents/
│   ├── vision_agent.py      # Gemini Vision API → disease classification
│   ├── diagnosis_agent.py   # Knowledge base lookup + treatment logic
│   ├── language_agent.py    # IndicTrans2 translation + lang detection
│   └── location_agent.py    # KVK lookup by PIN code
├── api/
│   ├── main.py              # FastAPI app entry point
│   ├── routes/              # Webhook endpoints
│   └── schemas/             # Pydantic request/response models
├── ui/
│   └── app.py               # Streamlit web demo
├── data/
│   ├── disease_db.json      # Disease knowledge base
│   └── kvk_centers.json     # KVK directory by district
├── tests/                   # Unit tests + mock API responses
├── docs/                    # Architecture diagram + screenshots
├── .env.example             # Environment variable template
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE                  # MIT
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# Tests use mocked Claude API responses — no API spend required for unit tests
```

---

## 🚀 Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set all environment variables from `.env` in the Railway dashboard under **Variables**.

---

## 📊 Success Metrics

| Metric | Target | Status |
|---|---|---|
| Disease detection accuracy | >90% on PlantVillage test set | 🔄 In progress |
| End-to-end response time | <30 seconds | 🔄 In progress |
| Language coverage | 5 Indian languages | 🔄 In progress |
| SMS fallback delivery | 100% on 2G simulation | 🔄 In progress |

---

## 🤝 Contributing

We welcome contributions! The highest-impact areas are:

- **New languages:** Add Marathi, Odia, Gujarati, Punjabi
- **New crops:** Expand beyond current 10-crop MVP
- **Disease knowledge base:** Improve treatment recommendations
- **Dataset quality:** Add field images for crops not in PlantVillage

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidance.

---

## 📖 References

- Hughes & Salathé (2016). *PlantVillage Dataset.* arXiv:1511.08060
- Gala et al. / AI4Bharat (2023). *IndicTrans2.* Transactions on Machine Learning Research.
- ScienceDirect (2025). *Precision agriculture in the age of AI: A systematic review.*
- J-PAL / PxD (2024). *Phone-based technology for agricultural information delivery.*
- FAO (2025). *The State of Food and Agriculture: Smallholder Farmers.*

---

## 📄 License

MIT © 2026 — See [LICENSE](LICENSE) for full terms.

---

<div align="center">
Made with ❤️ for India's farmers · <a href="https://github.com/YOUR_USERNAME/multilingual-crop-doctor/issues">Report Bug</a> · <a href="https://github.com/YOUR_USERNAME/multilingual-crop-doctor/issues">Request Feature</a>
</div>

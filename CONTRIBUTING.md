# Contributing to Multilingual AI Crop Doctor

Thank you for wanting to help bring AI-powered crop disease diagnosis to India's farmers. 🌾

## Ways to Contribute

### 1. Add a New Language
The translation layer uses IndicTrans2, which supports all 22 scheduled Indian languages.

**Steps:**
1. Open `agents/language_agent.py`
2. Add your language code to `SUPPORTED_LANGUAGES` dict (use ISO 639-1 codes)
3. Add 10 test sentences in the new language to `tests/fixtures/language_samples.json`
4. Run `pytest tests/test_language_agent.py -v` and confirm all pass
5. Update the Languages table in `README.md`
6. Open a PR with title: `feat: add [Language Name] support`

Supported codes reference: [IndicTrans2 language list](https://github.com/AI4Bharat/IndicTrans2#supported-languages)

### 2. Add a New Crop
**Steps:**
1. Collect at least 20 real field images of the crop (healthy + diseased)
2. Add disease entries to `data/disease_db.json` (see format below)
3. If the crop is in PlantVillage, add the class mapping to `agents/vision_agent.py`
4. If NOT in PlantVillage, add a zero-shot prompt entry in `agents/vision_agent.py`
5. Write at least 5 unit tests in `tests/test_vision_agent.py`
6. Open a PR with title: `feat: add [Crop Name] disease detection`

**disease_db.json format:**
```json
{
  "crop_name": {
    "disease_name": {
      "symptoms": "Description of visible symptoms",
      "organic_treatment": "Recommended organic remedy",
      "chemical_treatment": "Recommended pesticide (generic name)",
      "yield_impact": "Estimated % yield loss if untreated",
      "severity_markers": {
        "mild": "Early signs",
        "moderate": "Spread to 30%+ of leaves",
        "severe": "Widespread infection, wilting"
      }
    }
  }
}
```

### 3. Improve Treatment Recommendations
Treatment recommendations should align with ICAR (Indian Council of Agricultural Research) guidelines.

**Steps:**
1. Find the ICAR recommendation for the disease at [icar.org.in](https://icar.org.in)
2. Update `data/disease_db.json` with the correct recommendation
3. Add a citation comment in the JSON entry: `"source": "ICAR Bulletin [year]"`
4. Open a PR with title: `fix: improve treatment recommendation for [Disease]`

### 4. Report a Bug or Wrong Diagnosis
Open a GitHub Issue using the **Bug Report** template. Please include:
- The crop + disease you tested
- The language used
- What the bot said vs. what the correct diagnosis is
- (Optional) A sample test image (make sure it contains no identifying information)

### 5. Improve Documentation
Even small improvements to README, docstrings, or in-code comments are very welcome.

---

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-crop-doctor.git
cd multilingual-crop-doctor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your keys
pytest tests/ -v      # Confirm all tests pass before making changes
```

## Pull Request Guidelines

- One feature or fix per PR
- All new code must have at least one unit test
- Run `pytest tests/` before submitting — all tests must pass
- Keep PR descriptions concise: What does it change? Why?
- If adding a new language, include a short video or screenshot of the bot responding in that language

## Code Style

- Python: follow PEP 8; use `black` for formatting (`pip install black && black .`)
- Docstrings: Google style
- Type hints: required for all function signatures

## Questions?

Open a GitHub Discussion or email the maintainer (see GitHub profile).

---

*All contributions are released under the project's [MIT License](LICENSE).*

"""
Streamlit Web Demo — Multilingual AI Crop Doctor
================================================
A web interface for:
  - Farmers to upload a crop photo and get a diagnosis
  - KVK extension workers to review recent diagnoses
  - Demo and hackathon presentation

Run: streamlit run ui/app.py
"""

import streamlit as st
from PIL import Image
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Crop Doctor 🌾",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 Multilingual AI Crop Doctor")
st.markdown(
    "Upload a photo of your diseased crop and get a diagnosis in your language."
)

# Language selector
language = st.selectbox(
    "Choose your language / अपनी भाषा चुनें",
    options=["Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Bengali (বাংলা)", "English"],
    index=0,
)
lang_code = {"Hindi (हिंदी)": "hi", "Tamil (தமிழ்)": "ta",
             "Telugu (తెలుగు)": "te", "Bengali (বাংলা)": "bn", "English": "en"}[language]

# Image uploader
uploaded_file = st.file_uploader(
    "Upload crop photo / फसल की फोटो अपलोड करें",
    type=["jpg", "jpeg", "png"],
)

# Optional text description
text_description = st.text_area(
    "Describe the problem (optional) / समस्या बताएं (वैकल्पिक)",
    placeholder="e.g. पत्तियाँ पीली हो रही हैं / Leaves turning yellow",
)

# PIN code for KVK lookup
pin_code = st.text_input("PIN Code (for nearest KVK) / पिन कोड", max_chars=6)

if st.button("🔍 Diagnose / निदान करें", type="primary"):
    if uploaded_file is None and not text_description:
        st.error("Please upload an image or describe the problem.")
    else:
        with st.spinner("Analysing your crop... / फसल का विश्लेषण हो रहा है..."):
            # TODO: Call /diagnose endpoint
            # TODO: Display structured result
            st.warning("⚙️ Diagnosis pipeline not yet implemented. Come back soon!")

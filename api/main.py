"""
FastAPI Backend — Multilingual AI Crop Doctor
=============================================
Endpoints:
  POST /webhook/whatsapp   - Twilio WhatsApp webhook
  POST /webhook/sms        - Twilio SMS webhook
  POST /diagnose           - Direct API for Streamlit UI
  GET  /health             - Health check
"""

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import PlainTextResponse
import os

app = FastAPI(
    title="Multilingual AI Crop Doctor",
    description="AI crop disease diagnosis via WhatsApp for Indian farmers",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Receive and handle incoming WhatsApp messages from Twilio.

    Twilio sends a POST with form fields:
      - From: Farmer's WhatsApp number (whatsapp:+91XXXXXXXXXX)
      - Body: Text message content
      - MediaUrl0: URL of attached image (if any)
      - MediaContentType0: MIME type of attached image
    """
    # TODO: Parse Twilio form data
    # TODO: Download image if present
    # TODO: Run vision pipeline (vision_agent → diagnosis_agent → language_agent → location_agent)
    # TODO: Format response
    # TODO: Send reply via Twilio
    # TODO: Return TwiML 200 OK
    return PlainTextResponse("<?xml version='1.0'?><Response></Response>",
                              media_type="text/xml")


@app.post("/webhook/sms")
async def sms_webhook(request: Request):
    """
    Receive and handle incoming SMS messages (text-only fallback path).
    LLM reasoning only — no image, no vision agent.
    """
    # TODO: Parse Twilio SMS form data
    # TODO: Run text-only LLM diagnosis
    # TODO: Return 160-char first SMS + follow-up SMS
    return PlainTextResponse("<?xml version='1.0'?><Response></Response>",
                              media_type="text/xml")


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(None),
    text: str = Form(None),
    language: str = Form("hi"),
):
    """
    Direct diagnosis endpoint for the Streamlit web UI.

    Accepts either an image or a text description (or both).
    Returns structured JSON diagnosis.
    """
    # TODO: Validate input (image or text required)
    # TODO: Run pipeline
    # TODO: Return structured response
    return {
        "status": "not_implemented",
        "message": "Diagnosis pipeline not yet implemented",
    }

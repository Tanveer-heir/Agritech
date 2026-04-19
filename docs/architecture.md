# Architecture Notes

## System Diagram

Create the architecture diagram at `docs/architecture.png` using:
- **Recommended tool:** [Excalidraw](https://excalidraw.com) (free, exportable to PNG)
- **Alternative:** draw.io / diagrams.net

## Flow to Diagram

```
Farmer (WhatsApp / SMS)
        │
        ▼
  Twilio Webhook ──────────────────► FastAPI /webhook/whatsapp
        │
        ├─[image present]──────────► vision_agent.classify_disease()
        │                                    │
        │                                    ▼
        │                            diagnosis_agent.get_treatment()
        │                                    │
        ├─[no image / SMS]─────────► LLM text-only reasoning
        │                                    │
        │                            ◄───────┘
        │
        ▼
  language_agent.translate_to()
        │
        ▼
  location_agent.find_nearest_kvk()
        │
        ▼
  Twilio → WhatsApp / SMS reply to farmer
```

## Tech Stack Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vision model | Claude Vision API | Zero-shot, no training, handles Indian crops not in PlantVillage |
| Translation | IndicTrans2 / Claude fallback | Open source, supports all 22 Indian languages |
| Agent framework | LangGraph | Native support for multi-agent pipelines with state |
| Messaging | Twilio | Proven at scale; free sandbox for development |
| Deployment | Railway | Free tier sufficient for demo; Docker-native |

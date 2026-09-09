# PoliEmotiFusion Frontend

Instructions and context for automated agents working on the PoliEmotiFusion frontend.

## Guidelines
- This application uses TanStack Start, React 19, Tailwind CSS v4, and Radix UI.
- Keep components modular and type-safe.
- Text analysis connects to the FastAPI backend at `http://127.0.0.1:8000/api/text/analyze`.
- Unfinished modalities (Image, Video, Audio) remain in a clean unavailable state until their backends are implemented.

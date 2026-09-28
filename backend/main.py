import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Robust import handling across different execution contexts (package vs direct module)
try:
    from backend.model_service import model_service
    from backend.speech_intelligence import speech_pipeline
    from backend.schemas import (
        HealthResponse,
        SpeechIntelligenceResponse,
        TextAnalysisRequest,
        TextAnalysisResponse,
    )
except ImportError:
    try:
        from .model_service import model_service
        from .speech_intelligence import speech_pipeline
        from .schemas import (
            HealthResponse,
            SpeechIntelligenceResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
        )
    except ImportError:
        from model_service import model_service
        from speech_intelligence import speech_pipeline
        from schemas import (
            HealthResponse,
            SpeechIntelligenceResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models asynchronously in background threads so server starts instantly."""
    asyncio.create_task(asyncio.to_thread(model_service.load_model))
    asyncio.create_task(asyncio.to_thread(speech_pipeline.load_model))
    yield


app = FastAPI(
    title="PoliEmotiFusion Emotion Analysis Backend",
    version="1.1.0",
    description="FastAPI service for political multimodal emotion analysis using Hugging Face Transformers.",
    lifespan=lifespan,
)

# Enable CORS for frontend applications (supports Vite, TanStack Start, Next, local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format validation errors with HTTP 400 and a helpful detail message."""
    errors = exc.errors()
    error_messages = []
    for err in errors:
        loc = " -> ".join(str(item) for item in err.get("loc", []) if item != "body")
        msg = err.get("msg", "Invalid input")
        error_messages.append(f"{loc}: {msg}" if loc else msg)

    detail = "; ".join(error_messages) if error_messages else "Invalid request payload."
    return JSONResponse(status_code=400, content={"detail": detail})


@app.get("/")
async def root():
    """Root endpoint welcoming users and pointing to docs."""
    return {
        "title": "PoliEmotiFusion Emotion Intelligence API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "modalities": ["text", "audio"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning server and model readiness status."""
    return {
        "status": "ok",
        "models": {
            "text": model_service.is_loaded(),
            "speech_intelligence": speech_pipeline.is_loaded(),
        },
    }


@app.post("/api/text/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze the emotion of the submitted political text.
    Validates that stripped text length is between 20 and 1000 characters.
    Returns HTTP 400 without invoking the model for invalid input.
    """
    raw_text = request.text
    if not isinstance(raw_text, str):
        raise HTTPException(
            status_code=400,
            detail="The 'text' field must be a string.",
        )

    stripped_text = raw_text.strip()
    text_len = len(stripped_text)

    if text_len < 20 or text_len > 1000:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid text length: stripped text is {text_len} characters. "
                "Must be between 20 and 1000 characters."
            ),
        )

    # Inference with preloaded model (run in thread pool to not block event loop)
    prediction = await asyncio.to_thread(model_service.predict, stripped_text)
    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    # 60-character preview (with '...' if truncated) plus ' - N chars'
    preview = f"{stripped_text[:60]}..." if text_len > 60 else stripped_text
    input_label = f"{preview} - {text_len} chars"

    # Required summary format
    summary = f"The submitted text is predominantly {dominant_emotion}."

    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_id = f"an-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

    return TextAnalysisResponse(
        id=analysis_id,
        modality="text",
        emotion=dominant_emotion,
        confidence=confidence,
        probabilities=probabilities,
        timestamp=timestamp,
        category=request.category if request.category else "Other",
        model=model_service.model_name,
        inputLabel=input_label,
        summary=summary,
        demo=False,
    )


def _format_bytes(size: int) -> str:
    """Format byte size into human readable string."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


@app.post("/api/audio/analyze", response_model=SpeechIntelligenceResponse)
@app.post("/api/speech/analyze", response_model=SpeechIntelligenceResponse)
async def analyze_speech(
    file: UploadFile = File(...),
    category: str = Form("Other"),
):
    """
    Multilingual Political Speech Intelligence & Translation.
    Transcribes spoken audio in original language (Hindi, English, Marathi, Hinglish, etc.)
    and produces fluent English translations preserving rhetorical meaning, context, and intent.
    Zero emotion predictions or sentiment classifications are performed.
    """
    filename = file.filename or "audio_recording"
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    allowed_extensions = {"wav", "mp3", "m4a", "flac", "ogg"}

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '.{ext}'. Supported formats: WAV, MP3, M4A, FLAC, OGG.",
        )

    # Read uploaded file bytes
    try:
        content = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read audio file: {read_err}",
        )

    file_size = len(content)
    max_size = 100 * 1024 * 1024  # 100 MB
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({_format_bytes(file_size)}). Maximum size is 100 MB.",
        )

    if file_size < 100:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty or corrupted.",
        )

    # Execute speech intelligence pipeline in worker thread
    try:
        result = await asyncio.to_thread(
            speech_pipeline.process,
            content,
            filename=filename,
            file_size_formatted=_format_bytes(file_size),
            category=category,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=400,
            detail=str(val_err),
        )
    except Exception as proc_err:
        raise HTTPException(
            status_code=500,
            detail=f"Speech intelligence processing error: {str(proc_err)}",
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_id = f"sp-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    duration = result["duration"]
    input_label = f"{filename} - {duration:.1f}s ({_format_bytes(file_size)})"

    return SpeechIntelligenceResponse(
        id=analysis_id,
        modality="audio",
        status="completed",
        language=result["language"],
        languageName=result["languageName"],
        languageConfidence=result["languageConfidence"],
        transcript=result["transcript"],
        translation=result["translation"],
        isTranslationNeeded=result["isTranslationNeeded"],
        duration=duration,
        formattedDuration=result["formattedDuration"],
        chunkCount=result["chunkCount"],
        chunks=result["chunks"],
        audioStats=result["audioStats"],
        waveform=result["waveform"],
        timestamp=timestamp,
        category=category if category else "Other",
        inputLabel=input_label,
        model=result["model"],
        processingTime=result["processingTime"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

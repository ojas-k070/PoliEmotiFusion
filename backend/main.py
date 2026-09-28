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
    from backend.image_config import ALLOWED_EXTENSIONS, MAX_IMAGE_BYTES
    from backend.image_model_service import image_model_service
    from backend.model_service import model_service
    from backend.speech_intelligence import speech_pipeline
    from backend.schemas import (
        HealthResponse,
        SpeechIntelligenceResponse,
    from backend.political_service import political_service
    from backend.schemas import (
        HealthResponse,
        ImageAnalysisResponse,
        TextAnalysisRequest,
        TextAnalysisResponse,
    )
except ImportError:
    try:
        from .image_config import ALLOWED_EXTENSIONS, MAX_IMAGE_BYTES
        from .image_model_service import image_model_service
        from .model_service import model_service
        from .speech_intelligence import speech_pipeline
        from .schemas import (
            HealthResponse,
            SpeechIntelligenceResponse,
        from .political_service import political_service
        from .schemas import (
            HealthResponse,
            ImageAnalysisResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
        )
    except ImportError:
        from image_config import ALLOWED_EXTENSIONS, MAX_IMAGE_BYTES
        from image_model_service import image_model_service
        from model_service import model_service
        from speech_intelligence import speech_pipeline
        from schemas import (
            HealthResponse,
            SpeechIntelligenceResponse,
        from political_service import political_service
        from schemas import (
            HealthResponse,
            ImageAnalysisResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at application startup."""
    await asyncio.to_thread(model_service.load_model)
    await asyncio.to_thread(political_service.load_model)
    await asyncio.to_thread(speech_pipeline.load_model)
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
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Format validation errors with HTTP 400 and a helpful detail message."""
    errors = exc.errors()
    error_messages = []

    for err in errors:
        loc = " -> ".join(
            str(item)
            for item in err.get("loc", [])
            if item != "body"
        )
        msg = err.get("msg", "Invalid input")
        error_messages.append(
            f"{loc}: {msg}" if loc else msg
        )

    detail = (
        "; ".join(error_messages)
        if error_messages
        else "Invalid request payload."
    )

    return JSONResponse(
        status_code=400,
        content={"detail": detail},
    )


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
    Analyze political text.

    First checks whether the submitted text is political.
    Non-political text is rejected from emotion analysis.
    Political text is passed to the existing emotion model.
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

    # ---------------------------------------------------------
    # POLITICAL DOMAIN GATE
    # ---------------------------------------------------------
    is_pol, pol_score = political_service.is_political(stripped_text)

    # 60-character preview plus character count
    preview = (
        f"{stripped_text[:60]}…"
        if text_len > 60
        else stripped_text
    )

    input_label = f"{preview} — {text_len} chars"

    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_id = (
        f"an-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    )

    # If text is NOT political, do not run the emotion model
    if not is_pol:
        return TextAnalysisResponse(
            id=analysis_id,
            modality="text",
            emotion="Non-Political",
            confidence=round((1 - pol_score) * 100, 1),
            probabilities={
                "Non-Political": 100.0
            },
            timestamp=timestamp,
            category=(
                request.category
                if request.category
                else "Other"
            ),
            model=political_service.model_name,
            inputLabel=input_label,
            summary="The input does not appear to be political content.",
            demo=False,
        )

    # ---------------------------------------------------------
    # EXISTING TEXT EMOTION MODEL
    # ---------------------------------------------------------
    prediction = await asyncio.to_thread(
        model_service.predict,
        stripped_text
    )
    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    # 60-character preview (with '…' if truncated) plus character count
    preview = (
        f"{stripped_text[:60]}…" if text_len > 60 else stripped_text
    )
    input_label = f"{preview} — {text_len} chars"

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
        category=(
            request.category
            if request.category
            else "Other"
        ),
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

@app.post("/api/image/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(
        ...,
        description="Political image file to analyze",
    ),
    category: str = Form(
        default="Other",
        description="Political category",
    ),
):
    """
    Score the overall scene emotion in the submitted political image using CLIP.

    Validates image file extension, size (<= 10MB), and readable image data.
    Returns HTTPException 400 for invalid, unreadable, or oversized images.
    """

    filename = file.filename or "uploaded_image.jpg"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(ALLOWED_EXTENSIONS).upper()

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Allowed formats: {allowed_str}."
            ),
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image file is empty.",
        )

    if len(contents) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES // (1024 * 1024)

        raise HTTPException(
            status_code=400,
            detail=(
                f"File exceeds maximum allowed size of "
                f"{max_mb} MB."
            ),
        )

    try:
        prediction = image_model_service.predict(contents)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred during "
                f"image emotion analysis: {str(e)}"
            ),
        )

    # Image political-content gate
    if prediction.get("status") == "not_political_content":
        return JSONResponse(
            status_code=200,
            content={
                "status": "not_political_content",
                "political_score": prediction.get(
                    "political_score"
                ),
                "threshold": prediction.get(
                    "threshold"
                ),
                "message": prediction.get(
                    "message"
                ),
            },
        )

    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    meta = prediction.get("metadata", {})

    width = meta.get("width", 0)
    height = meta.get("height", 0)
    fmt = meta.get(
        "format",
        ext.replace(".", "").upper(),
    )

    input_label = (
        f"{filename} — {width}x{height} ({fmt})"
    )

    summary = (
        f"The model's top predicted emotion is "
        f"{dominant_emotion}."
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    analysis_id = (
        f"an-{int(time.time() * 1000)}-"
        f"{uuid.uuid4().hex[:6]}"
    )

    return ImageAnalysisResponse(
        id=analysis_id,
        modality="image",
        emotion=dominant_emotion,
        confidence=confidence,
        probabilities=probabilities,
        timestamp=timestamp,
        category=category if category else "Other",
        model=image_model_service.model_name,
        inputLabel=input_label,
        summary=summary,
        facesDetected=None,
        demo=False,

    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

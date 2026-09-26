from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import uuid
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Robust import handling across different execution contexts
# (package vs direct module)
try:
    from backend.model_service import model_service
    from backend.political_service import political_service
    from backend.schemas import (
        HealthResponse,
        TextAnalysisRequest,
        TextAnalysisResponse,
        VideoAnalysisResponse,
    )
    from backend.video.model.videomae_adapter import VideoMAEAdapter
    from backend.video.inference.predictor import create_video_predictor
except ImportError:
    try:
        from .model_service import model_service
        from .political_service import political_service
        from .schemas import (
            HealthResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
            VideoAnalysisResponse,
        )
        from .video.model.videomae_adapter import VideoMAEAdapter
        from .video.inference.predictor import create_video_predictor
    except ImportError:
        from model_service import model_service
        from political_service import political_service
        from schemas import (
            HealthResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
            VideoAnalysisResponse,
        )
        from video.model.videomae_adapter import VideoMAEAdapter
        from video.inference.predictor import create_video_predictor


# -------------------------------------------------------------------
# VIDEO CONFIGURATION
# -------------------------------------------------------------------

VIDEO_ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

MAX_VIDEO_BYTES = 100 * 1024 * 1024


# -------------------------------------------------------------------
# VIDEO MODEL + PREDICTOR
# -------------------------------------------------------------------

video_model = VideoMAEAdapter()

video_predictor = create_video_predictor(
    model=video_model,
    sample_rate=1,
    max_frames=32,
    max_duration_seconds=None,
    target_resolution=(224, 224),
    start_time_seconds=0.0,
    normalize=True,
    temporal_window_size=4,
)


# -------------------------------------------------------------------
# APPLICATION LIFESPAN
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once when FastAPI application starts."""

    # Existing text models
    model_service.load_model()
    political_service.load_model()

    # Video model
    video_model.load_model()

    yield


# -------------------------------------------------------------------
# FASTAPI APPLICATION
# -------------------------------------------------------------------

app = FastAPI(
    title="Political Multimodal Emotion Analysis Backend",
    version="1.0.0",
    description=(
        "FastAPI service for political emotion analysis "
        "using text and video modalities."
    ),
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# VALIDATION ERROR HANDLER
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning server status."""

    return {"status": "ok"}


# -------------------------------------------------------------------
# TEXT ANALYSIS
# -------------------------------------------------------------------

@app.post(
    "/api/text/analyze",
    response_model=TextAnalysisResponse,
)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze the emotion of the submitted political text.

    First checks whether the text is political.
    Non-political text is rejected from emotion analysis.
    Political text is passed to the existing emotion model.

    Validates that stripped text length is between
    20 and 1000 characters.
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
                f"Invalid text length: stripped text is "
                f"{text_len} characters. "
                "Must be between 20 and 1000 characters."
            ),
        )

    # ---------------------------------------------------------
    # POLITICAL DOMAIN GATE
    # ---------------------------------------------------------

    is_pol, pol_score = political_service.is_political(
        stripped_text
    )

    preview = (
        f"{stripped_text[:60]}..."
        if text_len > 60
        else stripped_text
    )

    input_label = f"{preview} — {text_len} chars"

    timestamp = datetime.now(timezone.utc).isoformat()

    analysis_id = (
        f"an-{int(time.time() * 1000)}-"
        f"{uuid.uuid4().hex[:6]}"
    )

    # ---------------------------------------------------------
    # NON-POLITICAL INPUT
    # ---------------------------------------------------------

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
            category=request.category
            if request.category
            else "Other",
            model=political_service.model_name,
            inputLabel=input_label,
            summary=(
                "The input does not appear to be "
                "political content."
            ),
            demo=False,
        )

    # ---------------------------------------------------------
    # EXISTING EMOTION MODEL
    # ---------------------------------------------------------

    prediction = model_service.predict(
        stripped_text
    )

    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    summary = (
        f"The submitted text is predominantly "
        f"{dominant_emotion}."
    )

    return TextAnalysisResponse(
        id=analysis_id,
        modality="text",
        emotion=dominant_emotion,
        confidence=confidence,
        probabilities=probabilities,
        timestamp=timestamp,
        category=request.category
        if request.category
        else "Other",
        model=model_service.model_name,
        inputLabel=input_label,
        summary=summary,
        demo=False,
    )


# -------------------------------------------------------------------
# VIDEO ANALYSIS
# -------------------------------------------------------------------

@app.post(
    "/api/video/analyze",
    response_model=VideoAnalysisResponse,
)
async def analyze_video(
    file: UploadFile = File(...),
    category: str = Form(default="Other"),
):
    """
    Analyze emotion from an uploaded video.

    The video is temporarily stored on disk, processed by the
    video emotion pipeline, and removed after processing.
    """

    # ---------------------------------------------------------
    # FILE EXTENSION VALIDATION
    # ---------------------------------------------------------

    original_filename = file.filename or "uploaded_video"

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in VIDEO_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format '{extension}'. "
                f"Allowed formats: "
                f"{', '.join(sorted(VIDEO_ALLOWED_EXTENSIONS))}"
            ),
        )

    # ---------------------------------------------------------
    # READ FILE
    # ---------------------------------------------------------

    try:
        video_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded video: {exc}",
        ) from exc

    if not video_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    # ---------------------------------------------------------
    # FILE SIZE VALIDATION
    # ---------------------------------------------------------

    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Video file is too large. "
                "Maximum allowed size is 100 MB."
            ),
        )

    # ---------------------------------------------------------
    # TEMPORARY FILE
    # ---------------------------------------------------------

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = Path(
                temporary_file.name
            )

        # -----------------------------------------------------
        # VIDEO PREDICTION
        # -----------------------------------------------------

        result = video_predictor.predict_video(
            temporary_path
        )

        result_dict = result.to_dict()

        # -----------------------------------------------------
        # ADD API METADATA
        # -----------------------------------------------------

        result_dict["category"] = (
            category if category else "Other"
        )

        result_dict["timestamp"] = (
            datetime.now(timezone.utc).isoformat()
        )

        result_dict["id"] = (
            f"vid-{int(time.time() * 1000)}-"
            f"{uuid.uuid4().hex[:6]}"
        )

        return result_dict

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Video analysis failed: "
                f"{exc}"
            ),
        ) from exc

    finally:
        # -----------------------------------------------------
        # CLEANUP TEMPORARY FILE
        # -----------------------------------------------------

        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# -------------------------------------------------------------------
# LOCAL DEVELOPMENT ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
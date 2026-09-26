from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Robust import handling across different execution contexts
try:
    from backend.image_config import ALLOWED_EXTENSIONS, MAX_IMAGE_BYTES
    from backend.image_model_service import image_model_service
    from backend.model_service import model_service
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
        from schemas import (
            HealthResponse,
            ImageAnalysisResponse,
            TextAnalysisRequest,
            TextAnalysisResponse,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the text model at startup; the gated vision model loads on demand."""
    model_service.load_model()
    yield


app = FastAPI(
    title="Political Emotion Analysis Backend",
    version="1.0.0",
    description="FastAPI service for multimodal political emotion analysis (Text & Image) using PyTorch & Hugging Face Transformers.",
    lifespan=lifespan,
)

# Enable CORS for frontend applications
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning server status."""
    return {"status": "ok"}


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

    # Inference with preloaded model
    prediction = model_service.predict(stripped_text)
    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    # 60-character preview (with '…* if truncated) plus ' — N chars'
    preview = f"{stripped_text[:60]}…" if text_len > 60 else stripped_text
    input_label = f"{preview} — {text_len} chars"

    # Required summary format
    summary = f"The submitted text is predominantly {dominant_emotion}."

    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_id = f"in-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

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


@app.post("/api/image/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Political image file to analyze"),
    category: str = Form(default="Other", description="Political category"),
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
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_str}.",
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
            detail=f"File exceeds maximum allowed size of {max_mb} MB.",
        )

    try:
        prediction = image_model_service.predict(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during image emotion analysis: {str(e)}",
        )

    if prediction.get("status") == "not_political_content":
        return JSONResponse(
            status_code=200,
            content={
                "status": "not_political_content",
                "political_score": prediction.get("political_score"),
                "threshold": prediction.get("threshold"),
                "message": prediction.get("message"),
            },
        )

    dominant_emotion = prediction["emotion"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]
    meta = prediction.get("metadata", {})
    width = meta.get("width", 0)
    height = meta.get("height", 0)
    fmt = meta.get("format", ext.replace(".", "").upper())

    input_label = f"{filename} — {width}x{height} ({fmt})"
    summary = f"The model's top predicted emotion is {dominant_emotion}."

    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_id = f"an-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

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

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

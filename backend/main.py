from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Robust import handling across different execution contexts (package vs direct module)
try:
    from backend.model_service import model_service
    from backend.schemas import HealthResponse, TextAnalysisRequest, TextAnalysisResponse
except ImportError:
    try:
        from .model_service import model_service
        from .schemas import HealthResponse, TextAnalysisRequest, TextAnalysisResponse
    except ImportError:
        from model_service import model_service
        from schemas import HealthResponse, TextAnalysisRequest, TextAnalysisResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer once when FastAPI application starts."""
    model_service.load_model()
    yield


app = FastAPI(
    title="Political Text Emotion Analysis Backend",
    version="1.0.0",
    description="FastAPI service for political text emotion analysis using Hugging Face Transformers.",
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

    # 60-character preview (with '…' if truncated) plus ' — N chars'
    preview = f"{stripped_text[:60]}…" if text_len > 60 else stripped_text
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
        category=request.category if request.category else "Other",
        model=model_service.model_name,
        inputLabel=input_label,
        summary=summary,
        demo=False,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

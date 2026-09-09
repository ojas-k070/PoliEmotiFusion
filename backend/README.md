# Political Text Emotion Analysis Backend

FastAPI backend service for detecting emotions in political text using Hugging Face Transformers and PyTorch with the pre-trained `j-hartmann/emotion-english-distilroberta-base` model.

## Features

- **FastAPI Framework**: High performance, asynchronous endpoints with automatic OpenAPI documentation.
- **Model Loading on Startup**: Tokenizer and model are loaded once into memory during application lifespan.
- **Transformer Model**: Uses `j-hartmann/emotion-english-distilroberta-base` with max length 512 and truncation.
- **Target Emotion Classes**: Anger, Joy, Sadness, Fear, Surprise, Disgust, Neutral.
- **Strict Probabilities**: Normalized to one-decimal percentages strictly totaling 100.0%.
- **Validation & Error Handling**: Rejects text with stripped length under 20 or over 1000 characters with HTTP 400 and helpful messages without calling the model.
- **CORS Configured**: Allows origins `http://localhost:5173` and `http://localhost:8080`.

---

## Windows Setup Instructions

### 1. Prerequisites
- Windows 10 / 11
- PowerShell or Command Prompt
- Existing project virtual environment at `E:\TY\Deep learning\CP\.dlenv`

### 2. Navigate to Project Root
Open PowerShell or Command Prompt and change directory to the project root:
```powershell
cd "E:\TY\Deep learning\CP"
```

### 3. Activate the Existing Project Environment

**In PowerShell:**
```powershell
# If script execution is restricted, run this once in your session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.dlenv\Scripts\Activate.ps1
```

**In Command Prompt (cmd.exe):**
```cmd
.dlenv\Scripts\activate.bat
```

### 4. Install Dependencies
```powershell
pip install -r backend\requirements.txt
```

---

## Running the Server

Start the Uvicorn ASGI server from the project root with hot-reload enabled:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Once running:
- Server: `http://localhost:8000`
- Swagger UI Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

## Testing the API on Windows

You can test the endpoints using PowerShell, Command Prompt (`curl.exe`), or your web browser.

### 1. Health Check (`GET /health`)

**Using PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**Using curl (cmd or PowerShell):**
```bash
curl.exe http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok"
}
```

---

### 2. Analyze Text (`POST /api/text/analyze`)

**Using PowerShell:**
```powershell
$body = @{
    text = "For too long our citizens have been asked to accept less while being promised more. Today we say clearly: that arrangement ends."
    category = "Political Speech"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/text/analyze" -Method Post -Body $body -ContentType "application/json"
```

**Using curl.exe:**
```bash
curl.exe -X POST "http://localhost:8000/api/text/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"For too long our citizens have been asked to accept less while being promised more. Today we say clearly: that arrangement ends.\", \"category\": \"Political Speech\"}"
```

**Sample Response:**
```json
{
  "id": "an-1725712800000-a1b2c3",
  "modality": "text",
  "emotion": "Anger",
  "confidence": 76.4,
  "probabilities": {
    "Anger": 76.4,
    "Disgust": 8.2,
    "Fear": 4.1,
    "Joy": 1.2,
    "Neutral": 4.5,
    "Sadness": 4.1,
    "Surprise": 1.5
  },
  "timestamp": "2026-09-07T13:15:00.000000+00:00",
  "category": "Political Speech",
  "model": "j-hartmann/emotion-english-distilroberta-base",
  "inputLabel": "For too long our citizens have been asked to accept less whil… — 132 chars",
  "summary": "The submitted text is predominantly Anger.",
  "demo": false
}
```

---

### 3. Testing Validation Errors (HTTP 400)

If the submitted text is shorter than 20 characters (after whitespace stripping) or longer than 1000 characters, the API immediately returns HTTP 400 without invoking the model.

**PowerShell test for short text:**
```powershell
$body = @{
    text = "Too short"
    category = "Other"
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://localhost:8000/api/text/analyze" -Method Post -Body $body -ContentType "application/json"
} catch {
    $stream = $_.Exception.Response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($stream)
    $reader.ReadToEnd()
}
```

**Expected Error Response (HTTP 400):**
```json
{
  "detail": "Invalid text length: stripped text is 9 characters. Must be between 20 and 1000 characters."
}
```

---

## API Specification Reference

### Endpoint: `POST /api/text/analyze`

#### Request Body
| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | Yes | — | Input text (stripped length between 20 and 1000 characters). |
| `category` | string | No | `"Other"` | Political category context. |

#### Response Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | Unique analysis ID formatted as `an-<timestamp>-<hash>`. |
| `modality` | string | Always `"text"`. |
| `emotion` | string | Dominant emotion (`Anger`, `Joy`, `Sadness`, `Fear`, `Surprise`, `Disgust`, `Neutral`). |
| `confidence` | float | Percentage confidence of the dominant emotion (1 decimal). |
| `probabilities` | object | All 7 emotion percentages with 1 decimal place totaling 100.0. |
| `timestamp` | string | ISO 8601 UTC timestamp. |
| `category` | string | Category passed in request or default `"Other"`. |
| `model` | string | `"j-hartmann/emotion-english-distilroberta-base"`. |
| `inputLabel` | string | 60-character preview (with `…` if truncated) plus ` — N chars`. |
| `summary` | string | `The submitted text is predominantly {emotion}.` |
| `demo` | boolean | Always `false`. |

---

## Frontend Integration

The backend is configured to accept cross-origin requests from:
- `http://localhost:5173` (Vite dev server default)
- `http://localhost:8080` (Alternative local frontend port)

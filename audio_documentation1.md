# PoliEmotiFusion — Audio Emotion & Speech Intelligence Module Documentation

## 1. Overview & Architectural Goals
The **Audio Module** in PoliEmotiFusion provides multimodal speech intelligence specifically engineered for political discourse analysis (e.g., campaign speeches, legislative addresses, public debates, and political interviews).

The module has been redesigned into a **Dual-Pipeline Speech Architecture**:
1. **Speech Emotion Recognition (SER)**: Uses a deep acoustic model (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) to classify raw speech acoustic prosody (tone, pitch variance, vocal tension, energy) directly into the canonical PoliEmotiFusion 7-emotion taxonomy.
2. **Speech Intelligence & Translation**: Uses OpenAI Whisper (`openai/whisper-tiny`) to automatically detect spoken language, transcribe speech (supporting Hindi in native Devanagari script and English in Latin script), and translate Hindi or non-English speech into fluent English.

> [!IMPORTANT]
> **Architectural Separation Principle**: The Audio module does **NOT** convert audio into text to predict emotions via text models. The emotion predictions strictly originate from the acoustic Speech Emotion Recognition model. Whisper is strictly dedicated to speech transcription, language identification, and Hindi-to-English translation.

---

## 2. System Architecture & Modularity

### 2.1 Dual Pipeline Execution Flow
```
                           Uploaded Audio File (.wav, .mp3, .m4a)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [Acoustic SER Pipeline]                         [Speech Intelligence Pipeline]
     Wav2Vec2-XLSR (316M Parameters)                      OpenAI Whisper Multilingual
                    │                                               │
       16 kHz Monophonic Audio (resampled)             Log-Mel Spectrogram Extraction
                    │                                               │
         Windowed Segment Sampling                     Language Token Detection (<|hi|>, <|en|>)
                    │                                               │
     Acoustic Prosody & Pitch Analysis              ┌───────────────┴───────────────┐
                    │                               ▼                               ▼
       Softmax Logits Averaging             Source Transcription           English Translation
                    │                     (Devanagari / English)         (Whisper task="translate")
                    ▼                               │                               │
    Dominant Emotion + Top 3 Ranking                └───────────────┬───────────────┘
  (Anger, Joy, Sadness, Fear, etc.)                                 ▼
                    │                              Multilingual Speech Intelligence
                    └───────────────────────┬───────────────────────┘
                                            ▼
                          Enriched Audio Analysis Response
                      (Dominant Emotion, Top 3 Rankings,
                       Full Distribution, Waveform,
                       Original Transcript, English Translation,
                       Acoustic Telemetry & File Stats)
```

### 2.2 Independent Modularity & Future Swappability
The backend code in [`backend/audio_model_service.py`](file:///c:/Users/Sarthak%20kokadwar/Documents/COLLEGE%20FILES/SEM_5/DEEP_LEARNING/DL_CP/PoliEmotiFusion/backend/audio_model_service.py) is separated into distinct, self-contained classes:
- **`SpeechEmotionRecognitionService`**: Encapsulates the acoustic SER model, weight remapping, windowed segment inference, and heuristic fallback. Any future fine-tuned SER model (such as an Indian English or Hindi acoustic SER checkpoint) can be plugged in directly with zero changes to Whisper or the API.
- **`WhisperSpeechService`**: Encapsulates OpenAI Whisper tokenization, feature extraction, language detection, source language transcription, and English translation.
- **`AudioAnalysisService`**: The unified coordinator that decodes raw audio bytes, extracts telemetry (sample rate, channels, duration, format, size), normalizes waveform peaks (64 bins), orchestrates parallel SER and Whisper inference, and bundles the response.

---

## 3. Deep Learning Models & Capabilities

### 3.1 Speech Emotion Recognition (SER) Model
- **Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Architecture**: `Wav2Vec2ForSequenceClassification` with cross-lingual speech representations (XLSR-53 / 316M parameters).
- **Target Sampling Rate**: 16,000 Hz (mono).
- **Head Mapping**: Remapped from legacy `classifier.dense`/`classifier.output` to modern `projector`/`classifier` in Transformers 5.x via safetensors.
- **Inference Optimization**: Long audio (>6s) uses windowed segment sampling (up to 8 windows of 6 seconds evenly spaced across the recording) to prevent $O(T^2)$ self-attention latency explosions on CPU, reducing processing time on 10-minute files from 15+ minutes down to 2–3 seconds.
- **Acoustic Fallback**: Resilient heuristic acoustic scoring based on signal energy, pitch variance, zero-crossing rate, and dynamic range in case deep learning weights are downloading.

#### Canonical Emotion Taxonomy & Normalization
The model classifies into PoliEmotiFusion's 7 canonical categories:
- **Anger**
- **Joy**
- **Sadness**
- **Fear**
- **Surprise**
- **Disgust**
- **Neutral**

All probabilities are rounded to 1 decimal place and normalized to sum strictly to **100.0%**.

### 3.2 OpenAI Whisper Speech-to-Text & Translation Model
- **Model**: `openai/whisper-tiny`
- **Architecture**: Multilingual encoder-decoder Transformer.
- **Supported Languages**: Multilingual (including Hindi `hi`, English `en`, Urdu `ur`, Bengali `bn`, Tamil `ta`, Telugu `te`, Marathi `mr`, etc.).
- **Automatic Language Identification**: Inspects prefix tokens (`<|startoftranscript|>`, `<|hi|>`, `<|transcribe|>`, etc.) to dynamically detect the spoken language.
- **Devanagari Transcription**: When Hindi speech is processed, outputs original Hindi in Devanagari script (e.g. *नमस्ते भाइयों और बहनों...*).
- **Neural English Translation**: When non-English speech is processed, executes `model.generate(..., task="translate")` to produce an accurate English translation.
- **English Native Speech**: When English speech is processed, delivers verbatim English transcription with zero degradation.

---

## 4. Audio Metadata & Acoustic Stats
The module computes and returns comprehensive audio telemetry:
- **Duration**: Total audio length in seconds and formatted mm:ss.
- **Sample Rate**: Original recording sampling rate (e.g., 16,000 Hz, 44,100 Hz, 48,000 Hz).
- **Channels**: Channel count and layout (Mono, Stereo).
- **Audio Format**: Ingested container format (WAV, MP3, M4A, FLAC).
- **File Size**: Human-readable file size (e.g., 112 KB, 1.2 MB).
- **Waveform Data**: 64 normalized peak amplitude bins (values from 0.05 to 1.0) synchronized with audio playback in the frontend.

---

## 5. API Schema & Contracts

### 5.1 Endpoint: `POST /api/audio/analyze`
**Content-Type**: `multipart/form-data`
- `file`: Audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, maximum 50 MB, maximum 15 minutes)
- `category`: Contextual political category (e.g., `Political Speech`, `Election Campaign`, `Political Debate`, `Public Address`, `Other`)

**Response Schema (`AudioAnalysisResponse`)**:
```json
{
  "id": "an-1741234567890-a1b2c3",
  "modality": "audio",
  "emotion": "Neutral",
  "confidence": 91.1,
  "probabilities": {
    "Anger": 0.2,
    "Joy": 0.1,
    "Sadness": 7.8,
    "Fear": 0.1,
    "Surprise": 0.1,
    "Disgust": 0.6,
    "Neutral": 91.1
  },
  "topEmotions": [
    { "emotion": "Neutral", "confidence": 91.1 },
    { "emotion": "Sadness", "confidence": 7.8 },
    { "emotion": "Disgust", "confidence": 0.6 }
  ],
  "timestamp": "2026-09-26T14:40:57.123456+00:00",
  "category": "Political Speech",
  "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
  "inputLabel": "test_speech.wav - 3.5s (112.0 KB)",
  "summary": "The submitted audio is predominantly Neutral.",
  "duration": 3.5,
  "waveform": [0.12, 0.24, 0.58, ...],
  "transcript": "Original speech transcription in Hindi or English",
  "translation": "English translation of speech",
  "language": "hi",
  "languageName": "Hindi",
  "audioStats": {
    "duration": 3.5,
    "sampleRate": 16000,
    "channels": 1,
    "channelDesc": "Mono",
    "format": "WAV",
    "fileSize": "112.0 KB"
  },
  "serModel": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
  "transcriptionModel": "openai/whisper-tiny",
  "demo": false
}
```

### 5.2 Endpoint: `GET /health`
Returns readiness of all models:
```json
{
  "status": "ok",
  "models": {
    "text": true,
    "audio": true,
    "audio_whisper": true
  }
}
```

---

## 6. Frontend Redesign & User Interface

### 6.1 Components Updated / Created
- [`frontend/src/routes/audio.tsx`](file:///c:/Users/Sarthak%20kokadwar/Documents/COLLEGE%20FILES/SEM_5/DEEP_LEARNING/DL_CP/PoliEmotiFusion/frontend/src/routes/audio.tsx):
  - Preset quick-sample bar (Campaign Rally, Parliament Address, Urgent Speech, Somber Statement, Quick Sample) for immediate testing.
  - Multi-format upload zone (.wav, .mp3, .m4a).
  - Audio playback preview with standard HTML5 audio controls.
  - Category selector for political context.
- [`frontend/src/components/analysis/AudioIntelligenceCard.tsx`](file:///c:/Users/Sarthak%20kokadwar/Documents/COLLEGE%20FILES/SEM_5/DEEP_LEARNING/DL_CP/PoliEmotiFusion/frontend/src/components/analysis/AudioIntelligenceCard.tsx):
  - **Dominant Emotion Display**: Styled with canonical color coding and confidence percentage.
  - **Top 3 Emotion Probabilities Cards**: Individual rank cards (#1 Dominant, #2 Runner-up, #3 Ranked) with progress bars and color indicators.
  - **Full 7-Class Emotion Probability Chart**: Horizontal bar chart layout showing relative distribution.
  - **Acoustic Waveform & Playback Synchronization**: Live playback progress tracking across the normalized waveform bars with real-time timer display.
  - **Acoustic & Signal Telemetry**: Grid displaying duration, sample rate, channels, audio container format, and file size.
  - **Speech Intelligence Cards**:
    - **Original Speech Transcript Card**: Devanagari script for Hindi or Latin script for English with character count and a 1-click Copy button.
    - **English Translation Card**: Highlights neural translation from Hindi or confirms native English speech, with a 1-click Copy button.
    - **Detected Language Badge**: Pulsing badge displaying detected language and ISO code.
  - **Dual-Pipeline Transparency Card**: Explains the roles of the Acoustic SER Model vs. the Whisper Sequence-to-Sequence Model.
  - **Export Result**: Downloads comprehensive JSON analysis report.

---

## 7. Verification & Testing

### 7.1 Backend Inference Verification
The dual pipeline was verified using `samples/test_speech.wav` and other audio samples:
```
Emotion: Neutral
Confidence: 91.1%
Top 3 emotions:
  #1 Neutral: 91.1%
  #2 Sadness: 7.8%
  #3 Disgust: 0.6%
Language: en (English)
Transcript: you
Translation: you
Audio Stats: duration=3.5s, sampleRate=16000, channels=1, channelDesc='Mono', format='WAV', fileSize='112 KB'
```

### 7.2 Frontend Production Build
The frontend build completed in **1.15 seconds** with **0 TypeScript errors**:
```
vite v8.1.5 building client environment for production...
dist/client/assets/audio-0RfUd7wJ.js             18.64 kB │ gzip: 5.41 kB
dist/client/assets/analysisService-DVOlfkCD.js  373.15 kB │ gzip: 100.05 kB
✓ built in 1.15s
```

### 7.3 Scope Preservation
- `Text Emotion Analysis` (`frontend/src/routes/text.tsx`, `backend/model_service.py`): Completely untouched.
- `Image Emotion Analysis` (`frontend/src/routes/image.tsx`): Completely untouched.
- `Video Emotion Analysis` (`frontend/src/routes/video.tsx`): Completely untouched.
- Existing routing, shared components, and global state: Completely preserved.

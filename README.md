# Video Analysis Module

This folder contains a portable, reusable Video Analysis Module for the
PoliEmotiFusion project.

It is designed to be copied later into the parent PoliEmotiFusion
repository as:

```text
PoliEmotiFusion/video/
```

The module does not depend on files outside this folder.

The current implementation provides the foundation for processing political
videos and preparing them for a future VideoMAE-based emotion model.

---

## Purpose

The module provides:

- video validation and metadata extraction
- configurable frame sampling
- frame resizing and preprocessing
- centralized emotion-label configuration
- emotion aggregation utilities
- temporal emotion timeline utilities
- a model interface for future video models
- a VideoMAE adapter scaffold
- dataset metadata structures
- YouTube metadata collection scaffolding
- a framework-neutral video analysis service
- lightweight tests that do not require a trained model

The module currently does **not** contain a trained video emotion model.

---

## Current Status

### Currently Implemented

- reusable video reader and validation logic
- configurable frame sampling
- maximum frame-count control
- optional duration limits
- configurable starting timestamp
- target frame resolution
- frame preprocessing and normalization
- centralized seven-class emotion configuration
- emotion-distribution aggregation
- temporal emotion timeline and change detection
- VideoMAE adapter scaffold
- dataset metadata schema
- YouTube metadata collection scaffold
- framework-neutral API/service boundary
- lightweight automated tests

### Not Yet Implemented

- final political-video dataset
- final emotion annotations
- VideoMAE model loading
- VideoMAE fine-tuning
- trained emotion classification head
- trained model checkpoint
- final model evaluation
- production backend endpoint
- frontend upload integration
- multimodal fusion with text, image, and audio

---

## Emotion Labels

The current emotion taxonomy contains seven labels:

```text
anger
joy
sadness
fear
surprise
disgust
neutral
```

These labels represent **emotion**, not event or political context.

For example:

```text
category      = protest
emotion_label = anger
```

The `category` describes the context of the video, while
`emotion_label` represents the target emotion.

An unannotated video should have no emotion label rather than being
automatically assigned `neutral`.

---

## Architecture

The module is intentionally modular and self-contained:

```text
config/
    Shared constants, emotion labels, environment configuration

preprocessing/
    Video metadata reading
    Frame sampling
    Frame preprocessing

model/
    Abstract model contract
    Future VideoMAE adapter

inference/
    Future model inference wrapper

analysis/
    Emotion aggregation
    Temporal emotion analysis

dataset/
    Dataset metadata schema
    YouTube metadata collection scaffold

api/
    Framework-neutral video analysis service

tests/
    Lightweight module tests

examples/
    Example usage
```

---

## Folder Structure

```text
video/

├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── frontend_integration.md
├── TRAINING_PIPELINE.md
├── __init__.py
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   ├── metadata/
│   │   └── dataset_schema.csv
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
│
├── preprocessing/
│   ├── __init__.py
│   ├── frame_sampler.py
│   ├── preprocess.py
│   └── video_reader.py
│
├── model/
│   ├── __init__.py
│   ├── model_interface.py
│   └── videomae_adapter.py
│
├── inference/
│   ├── __init__.py
│   └── predictor.py
│
├── analysis/
│   ├── __init__.py
│   ├── emotion_aggregator.py
│   └── temporal_analysis.py
│
├── api/
│   ├── __init__.py
│   └── video_api.py
│
├── dataset/
│   ├── __init__.py
│   ├── metadata_schema.py
│   ├── youtube_metadata_collector.py
│   └── dataset_schema.csv
│
├── tests/
│   └── test_video_module.py
│
└── examples/
    └── example_usage.py
```

The `.venv/`, `__pycache__/`, `.pytest_cache/`, and other generated
development files should remain excluded through `.gitignore` and should
not be committed to the repository.

The module intentionally avoids imports that reference the parent
PoliEmotiFusion repository or unrelated backend/frontend folders.

---

## Installation

From inside the `video/` directory:

### Windows PowerShell

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Create a local `.env` file from `.env.example`.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux/macOS

```bash
cp .env.example .env
```

The module can read configuration values such as:

```text
YOUTUBE_API_KEY
VIDEO_MODEL_NAME
VIDEO_MODEL_PATH
VIDEO_INPUT_DIR
VIDEO_OUTPUT_DIR
VIDEO_SAMPLE_RATE
VIDEO_MAX_DURATION_SECONDS
VIDEO_START_TIME_SECONDS
VIDEO_MODEL_DEVICE
```

The default VideoMAE architecture name is:

```text
MCG-NJU/videomae-base
```

However, the actual trained/fine-tuned checkpoint is not currently
configured or loaded.

---

## Video Input Support

The video reader currently supports common formats including:

```text
.mp4
.avi
.mov
.mkv
.wmv
.webm
.flv
```

Before processing a video, the reader validates:

- file existence
- file type
- file readability
- frame count
- frame rate
- frame dimensions

---

## Video Preprocessing

The preprocessing pipeline currently consists of:

```text
Video
  ↓
VideoReader
  ↓
FrameSampler
  ↓
Frame resizing
  ↓
Frame sequence
  ↓
NumPy preprocessing array
  ↓
Future model
```

The current default target resolution is:

```text
224 × 224
```

The frame sampler supports:

- configurable sample rate
- configurable maximum number of frames
- optional maximum duration
- configurable starting timestamp
- temporal ordering preservation

The current default maximum frame count is:

```text
32 frames
```

The final training configuration may use a different value after the
dataset and computational requirements are evaluated.

---

## Preprocessing Example

A basic preprocessing workflow is:

```python
from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import FrameSampler, FrameSampleConfig
from preprocessing.preprocess import preprocess_video_frames

video_path = "path/to/sample.mp4"

reader = VideoReader()

metadata = reader.read_metadata(video_path)

print(metadata.to_dict())

config = FrameSampleConfig(
    sample_rate=2,
    max_frames=16,
    target_resolution=(224, 224),
)

frames = FrameSampler(config).sample(video_path)

processed = preprocess_video_frames(
    video_path,
    sample_rate=2,
    max_frames=16,
    target_resolution=(224, 224),
    normalize=True,
)

print(processed.shape)
```

The resulting `processed` object is currently a NumPy array containing the
preprocessed frame sequence.

The exact tensor conversion required by VideoMAE will be implemented when
the real model pipeline is connected.

---

## Dataset Preparation

The dataset foundation separates event/context information from emotion
annotations.

Example context categories include:

```text
speech
parliamentary_debate
rally
protest
press_conference
confrontation
strike
public_event
```

The metadata schema is available in:

```text
dataset/metadata_schema.py
```

A sample CSV structure is available in:

```text
data/metadata/dataset_schema.csv
```

A newly collected video may initially contain:

```text
emotion_label = empty / unassigned
```

The video should receive a valid emotion annotation before entering the
supervised training dataset.

---

## YouTube Metadata Collection

The planned workflow is:

```text
Approved source
      ↓
YouTube Data API
      ↓
Video IDs + metadata
      ↓
CSV/JSON metadata
      ↓
Manual selection and review
      ↓
Video acquisition according to applicable permissions/terms
      ↓
Emotion annotation
      ↓
Preprocessing
      ↓
Training dataset
```

The scaffold in:

```text
dataset/youtube_metadata_collector.py
```

currently provides:

- playlist ID input
- API-key configuration
- pagination placeholders
- metadata extraction structure
- CSV export

The current implementation does **not** perform browser scraping,
mass downloading, or automatic emotion annotation.

---

## VideoMAE Integration

The planned base architecture is:

```text
MCG-NJU/videomae-base
```

The adapter is located at:

```text
model/videomae_adapter.py
```

It implements the project's generic:

```text
VideoModelInterface
```

The adapter currently stores:

- model name
- model path
- device configuration
- model state
- processor state

It does not currently download or load VideoMAE.

Calling inference before a trained model has been configured results in an
explicit model-not-configured error.

The module intentionally does not fabricate:

- emotion labels
- confidence values
- probability distributions
- model predictions

---

## Inference

The inference wrapper is located at:

```text
inference/predictor.py
```

Its intended future workflow is:

```text
Video path
    ↓
Frame sampling
    ↓
Preprocessing
    ↓
Trained VideoMAE model
    ↓
Emotion predictions
    ↓
Emotion aggregation
    ↓
Temporal analysis
    ↓
VideoAnalysisResult
```

The predictor also provides a frame-based prediction interface for future
model integration.

The current predictor returns a `model_not_configured` result when no model
has been supplied.

---

## Temporal Emotion Analysis

Political videos may contain changing emotional states throughout a clip.

The temporal analysis module is located at:

```text
analysis/temporal_analysis.py
```

It provides timestamped emotion points such as:

```text
0.0 sec  → neutral
5.0 sec  → joy
10.0 sec → anger
```

The timeline utilities can:

- store timestamped emotion predictions
- preserve temporal ordering
- identify emotion transitions
- return timeline data in dictionary/list form

The final trained inference pipeline can use these structures to produce an
emotion timeline for the complete video.

---

## Emotion Aggregation

The emotion aggregation module is located at:

```text
analysis/emotion_aggregator.py
```

It can aggregate multiple emotion probability distributions into:

- final emotion distribution
- dominant emotion
- dominant-emotion confidence

Example conceptual output:

```text
{
    "emotions": {
        "anger": 0.62,
        "joy": 0.08,
        "sadness": 0.10,
        "fear": 0.05,
        "surprise": 0.04,
        "disgust": 0.06,
        "neutral": 0.05
    },
    "dominant_emotion": "anger",
    "confidence": 0.62
}
```

This example is illustrative only and does not represent an actual model
prediction.

---

## Future Training Pipeline

See:

```text
TRAINING_PIPELINE.md
```

for the detailed planned training workflow.

The intended future process is:

```text
1. Collect political video metadata
2. Select and review videos
3. Acquire permitted video clips
4. Annotate emotion labels
5. Create train/validation/test splits
6. Sample frames
7. Preprocess frame sequences
8. Load pretrained VideoMAE
9. Attach seven-class emotion head
10. Fine-tune on the political-video dataset
11. Evaluate the trained model
12. Save the best checkpoint
13. Connect the checkpoint to the inference adapter
14. Generate video-level and temporal predictions
```

The exact hyperparameters will be selected after the actual dataset and
available hardware have been evaluated.

---

## API / Backend Integration

This folder does not contain a production backend.

Instead, it provides a framework-neutral service in:

```text
api/video_api.py
```

The service currently:

1. validates the supplied video
2. reads its metadata
3. returns a structured `model_not_configured` result

Once the trained model is available, this service can be extended to call
the real inference pipeline.

The service can later be integrated into the main PoliEmotiFusion backend,
for example through FastAPI or Flask.

---

## Future Frontend Integration

See:

```text
frontend_integration.md
```

for the planned frontend/API contract.

The intended user flow is:

```text
Upload Video
      ↓
Analyze
      ↓
Processing
      ↓
Video Emotion Results
      ↓
Emotion Distribution
      ↓
Temporal Emotion Timeline
```

The frontend should not assume that a valid emotion prediction exists until
the backend has successfully loaded a trained model.

---

## Future Multimodal Fusion

This module is intentionally limited to the video modality.

The eventual PoliEmotiFusion system can combine the video results with:

```text
Text
Image
Audio
Video
```

through the project's multimodal fusion layer.

The video module should therefore expose structured outputs that can be
consumed by the larger fusion system without depending on the
implementation details of the other modalities.

---

## Testing

Run the tests from inside the `video/` directory:

```powershell
pytest -q
```

The current foundation test suite contains eight tests and is designed to
run without:

- a trained VideoMAE model
- GPU hardware
- downloaded model checkpoints
- a YouTube API key
- a large political-video dataset

The expected current result is:

```text
8 passed
```

---

## Repository Integration

The module is designed to be copied later into:

```text
PoliEmotiFusion/video/
```

After copying, the parent repository can import the module without requiring
the video implementation to depend on unrelated parent-project files.

Before committing the module:

- do not commit `.venv/`
- do not commit `__pycache__/`
- do not commit `.pytest_cache/`
- do not commit large raw video files
- do not commit API keys
- do not commit private credentials
- do not commit large model checkpoints unless the repository explicitly
  requires them

The `.env` file should remain local and should not be committed.

---

## Current Limitations

The current module is a **video-processing and integration foundation**.

It does not yet claim to perform real political emotion classification.

Specifically, the following remain future work:

- creation of the final annotated dataset
- validation of annotation quality
- VideoMAE loading
- model fine-tuning
- trained classification head
- model checkpoint selection
- quantitative model evaluation
- production API integration
- frontend integration
- multimodal fusion

No model accuracy, F1-score, or other performance number is claimed at this
stage.

---

## Project Principle

The module follows a simple development principle:

```text
Build and test the data pipeline first.
Train the real model only after the dataset is prepared.
Connect the trained model only after its behavior has been evaluated.
```

This keeps the current module reproducible, testable, and independent of a
future trained checkpoint.
# Frontend integration notes

This document describes the future integration contract for the UI that will upload and display video analysis results.

## Expected user flow

```text
Upload Video
  -> Analyze
  -> Loading
  -> Results
  -> Emotion Distribution
  -> Temporal Emotion Timeline
```

## Expected API request

```http
POST /analyze/video
Content-Type: multipart/form-data
```

The request should include a video file upload:

```json
{
  "video": "<binary upload>"
}
```

## Expected API response

```json
{
  "modality": "video",
  "dominant_emotion": "neutral",
  "confidence": 0.0,
  "emotions": {
    "anger": 0.0,
    "joy": 0.0,
    "sadness": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "disgust": 0.0,
    "neutral": 0.0
  },
  "frames_processed": 0,
  "duration_seconds": 0.0
}
```

## Temporal timeline example

```json
[
  {
    "timestamp": 0.0,
    "emotion": "neutral",
    "confidence": 0.0
  },
  {
    "timestamp": 5.0,
    "emotion": "anger",
    "confidence": 0.0
  }
]
```

## UI requirements

The frontend should display:

- upload field for a video file
- loading state while the backend analyzes the clip
- final dominant emotion
- overall distribution bar chart or list
- a time-based timeline for emotion values across the clip

## Integration notes

- this folder does not contain a production frontend
- the future app should call the backend video analysis endpoint
- the frontend should present results in a user-friendly way without assuming a trained model exists yet
- no hard-coded API keys or fake predictions should be embedded in the UI

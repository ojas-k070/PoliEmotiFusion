from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.emotion_aggregator import EmotionAggregator
from analysis.temporal_analysis import TemporalEmotionTimeline
from config.config import EMOTION_LABELS, VideoAnalysisResult, get_module_settings, load_module_config
from dataset.metadata_schema import DatasetMetadataSchema
from preprocessing.frame_sampler import FrameSampleConfig
from preprocessing.video_reader import VideoReader


def test_video_reader_rejects_missing_file():
    reader = VideoReader()
    with pytest.raises(FileNotFoundError):
        reader.read_metadata("does_not_exist.mp4")


def test_video_reader_rejects_unsupported_extension():
    reader = VideoReader()
    with pytest.raises(ValueError):
        reader.read_metadata("sample.txt")


def test_frame_sample_config_rejects_bad_values():
    with pytest.raises(ValueError):
        FrameSampleConfig(sample_rate=0)
    with pytest.raises(ValueError):
        FrameSampleConfig(max_frames=0)


def test_emotion_aggregator_works_for_valid_distribution():
    aggregator = EmotionAggregator()
    result = aggregator.aggregate([
        {"anger": 0.9, "joy": 0.1, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0},
        {"anger": 0.7, "joy": 0.2, "sadness": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0},
    ])
    assert result["dominant_emotion"] == "anger"
    assert result["confidence"] > 0.0
    assert set(result["emotions"]) == set(EMOTION_LABELS)


def test_temporal_result_structure():
    timeline = TemporalEmotionTimeline()
    timeline.add_event(0.0, "neutral", 0.5)
    timeline.add_event(5.0, "joy", 0.6)
    data = timeline.to_list()
    assert data[0]["timestamp"] == 0.0
    assert data[1]["emotion"] == "joy"
    assert len(data) == 2


def test_module_settings_include_expected_fields():
    settings = get_module_settings()
    assert "emotion_labels" in settings
    assert "supported_video_extensions" in settings
    assert "default_target_resolution" in settings


def test_dataset_schema_fields():
    schema = DatasetMetadataSchema()
    payload = schema.to_dict()
    assert "video_id" in payload
    assert payload["category"] == "speech"
    assert payload["split"] == "train"


def test_video_analysis_result_round_trip():
    result = VideoAnalysisResult(
        dominant_emotion="joy",
        confidence=0.8,
        emotions={label: 0.0 for label in EMOTION_LABELS},
        frames_processed=12,
        duration_seconds=24.0,
    )
    payload = result.to_dict()
    restored = VideoAnalysisResult.from_dict(payload)
    assert restored.dominant_emotion == "joy"
    assert restored.frames_processed == 12

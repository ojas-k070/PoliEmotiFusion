from __future__ import annotations

from pathlib import Path

from analysis.emotion_aggregator import EmotionAggregator
from preprocessing.frame_sampler import FrameSampleConfig, FrameSampler
from preprocessing.video_reader import VideoReader


def main() -> None:
    video_path = Path("data/raw/example_video.mp4")

    if not video_path.exists():
        print("Example video not found. Add a sample video to data/raw/ to test preprocessing.")
        return

    reader = VideoReader()
    metadata = reader.read_metadata(video_path)
    print(metadata.to_dict())

    config = FrameSampleConfig(sample_rate=2, max_frames=16, target_resolution=(224, 224))
    frames = FrameSampler(config).sample(video_path)
    print(f"Sampled frames: {len(frames)}")

    aggregator = EmotionAggregator()
    sample_predictions = [
        {"anger": 0.5, "joy": 0.1, "sadness": 0.1, "fear": 0.1, "surprise": 0.1, "disgust": 0.0, "neutral": 0.1},
        {"anger": 0.6, "joy": 0.1, "sadness": 0.1, "fear": 0.0, "surprise": 0.1, "disgust": 0.0, "neutral": 0.1},
    ]
    result = aggregator.aggregate(sample_predictions)
    print(result)


if __name__ == "__main__":
    main()

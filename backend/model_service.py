from typing import Any, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Target exact labels required by the application
TARGET_EMOTIONS = [
    "Anger",
    "Joy",
    "Sadness",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]

LABEL_MAPPING: Dict[str, str] = {
    "anger": "Anger",
    "joy": "Joy",
    "sadness": "Sadness",
    "fear": "Fear",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "neutral": "Neutral",
}


class EmotionModelService:
    """Service to load and run inference on the emotion classification model."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base") -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> None:
        """Load tokenizer and model onto target device."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on the provided text string.
        Returns:
            dict containing:
                - emotion (str): dominant emotion
                - confidence (float): confidence percentage of dominant emotion (1 decimal)
                - probabilities (Dict[str, float]): 1-decimal percentages totaling 100
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()

        # Tokenize with truncation=True and max_length=512
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Map model class indices to canonical labels and raw percentages
        raw_percentages: Dict[str, float] = {}
        for idx, prob in enumerate(raw_probs):
            raw_label = self.model.config.id2label.get(idx, str(idx)).lower()
            canonical_label = LABEL_MAPPING.get(raw_label, raw_label.capitalize())
            raw_percentages[canonical_label] = prob * 100.0

        # Guarantee all 7 labels are present
        for label in TARGET_EMOTIONS:
            if label not in raw_percentages:
                raw_percentages[label] = 0.0

        # Initial 1-decimal rounding
        probabilities: Dict[str, float] = {
            label: round(raw_percentages[label], 1) for label in TARGET_EMOTIONS
        }

        # Adjust the dominant class so probabilities strictly sum to 100.0
        current_sum = round(sum(probabilities.values()), 1)
        diff = round(100.0 - current_sum, 1)
        if diff != 0:
            dominant_key = max(probabilities, key=probabilities.get)
            probabilities[dominant_key] = round(probabilities[dominant_key] + diff, 1)

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# Singleton model service instance
model_service = EmotionModelService()

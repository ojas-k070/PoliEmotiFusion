import torch
from transformers import pipeline

class PoliticalGateService:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.model_name = "typeform/distilbert-base-uncased-mnli"
        self.pipeline = None
        self.device = 0 if torch.cuda.is_available() else -1

    def load_model(self):
        if self.pipeline is None:
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device
            )

    def is_political(self, text: str) -> tuple[bool, float]:
        if self.pipeline is None:
            self.load_model()
            
        result = self.pipeline(text, candidate_labels=["political", "non-political"])
        
        political_score = 0.0
        for label, score in zip(result['labels'], result['scores']):
            if label == "political":
                political_score = score
                break
                
        return political_score >= self.threshold, political_score

political_service = PoliticalGateService()

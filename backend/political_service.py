import re
import torch
from transformers import pipeline

# Common political domain terms to reinforce zero-shot detection on borderline texts
POLITICAL_KEYWORDS = re.compile(
    r'\b(colleague|budget|slogan|senator|bill|congress|president|presidential|election|vote|voted|voter|voting|policy|policymaker|prime minister|parliament|parliamentary|government|governor|democrat|republican|campaign|candidate|legislation|legislative|lawmaker|tax|amendment|constitution|diplomacy|sanction|tariff|party|ballot|administration|bipartisan|lobby|political|politics|debate|public|nation|national|state|regulations?|reform|council|mayor|minister)\b',
    re.IGNORECASE
)


class PoliticalGateService:
    def __init__(self, threshold: float = 0.50):
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

        result = self.pipeline(
            text,
            candidate_labels=["political", "non-political"],
            hypothesis_template="This text is {}."
        )

        political_score = 0.0
        for label, score in zip(result['labels'], result['scores']):
            if label == "political":
                political_score = score
                break

        # Check for domain keywords to reinforce classification on borderline political text
        keyword_matches = len(POLITICAL_KEYWORDS.findall(text))
        if keyword_matches > 0:
            # Apply a boost per political keyword match (max +0.25)
            boost = min(keyword_matches * 0.10, 0.25)
            political_score = min(political_score + boost, 1.0)

        return political_score >= self.threshold, round(political_score, 4)


political_service = PoliticalGateService()



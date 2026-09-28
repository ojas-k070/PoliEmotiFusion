from types import SimpleNamespace

from PIL import Image
import torch
from fastapi.testclient import TestClient

import backend.image_model_service as image_service_module
from backend.image_config import EMOTION_CLASSES
from backend.image_model_service import PoliticalImageEmotionModelService
from backend.main import app, model_service


def test_predict_rejects_non_political_content(monkeypatch):
    service = PoliticalImageEmotionModelService()

    def fake_gate(_image_bytes):
        return False, 0.12, 0.30

    monkeypatch.setattr(service, "check_domain_gate", fake_gate)

    result = service.predict(b"not-used")

    assert result["status"] == "not_political_content"
    assert "emotion" not in result
    assert result["political_score"] == 0.12
    assert result["threshold"] == 0.30


def test_predict_scores_overall_scene_with_emotion_prompts(monkeypatch):
    service = PoliticalImageEmotionModelService()
    prompt_count = sum(
        len(prompts)
        for prompts in image_service_module.POLITICAL_EMOTION_PROMPTS.values()
    )
    encoded_prompts = []

    def fake_processor(**kwargs):
        encoded_prompts.extend(kwargs["text"])
        return {"input_ids": torch.zeros((prompt_count, 1))}

    class FakeClipModel:
        logit_scale = torch.tensor(0.0)

        def __call__(self, **_inputs):
            logits = torch.zeros((1, prompt_count))
            logits[0, :2] = 10.0
            return SimpleNamespace(logits_per_image=logits)

    service.clip_processor = fake_processor
    service.clip_model = FakeClipModel()
    monkeypatch.setattr(service, "check_domain_gate", lambda _image: (True, 0.91, 0.30))
    monkeypatch.setattr(
        service,
        "validate_and_load_image",
        lambda _image: (Image.new("RGB", (32, 20)), "PNG", (32, 20)),
    )

    result = service.predict(b"image bytes")

    assert result["emotion"] == EMOTION_CLASSES[0]
    assert list(result["probabilities"]) == list(EMOTION_CLASSES)
    assert round(sum(result["probabilities"].values()), 1) == 100.0
    assert result["metadata"] == {"format": "PNG", "width": 32, "height": 20}
    assert encoded_prompts == [
        prompt
        for prompts in image_service_module.POLITICAL_EMOTION_PROMPTS.values()
        for prompt in prompts
    ]


def test_image_api_returns_gate_rejection(monkeypatch):
    monkeypatch.setattr(model_service, "load_model", lambda: None)
    monkeypatch.setattr(
        image_service_module.image_model_service,
        "predict",
        lambda _image: {
            "status": "not_political_content",
            "political_score": 0.12,
            "threshold": 0.30,
            "message": "The uploaded image does not appear to contain political content.",
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/image/analyze",
            files={"file": ("sample.jpg", b"test image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "not_political_content"
    assert response.json()["political_score"] == 0.12


def test_image_api_describes_the_top_prediction(monkeypatch):
    monkeypatch.setattr(
        image_service_module.image_model_service,
        "predict",
        lambda _image: {
            "emotion": "Fear",
            "confidence": 65.4,
            "probabilities": {label: 0.0 for label in EMOTION_CLASSES},
            "metadata": {"format": "JPEG", "width": 32, "height": 20},
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/image/analyze",
            files={"file": ("sample.jpg", b"test image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["summary"] == "The model's top predicted emotion is Fear."

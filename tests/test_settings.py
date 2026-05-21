from fastapi.testclient import TestClient

from backend.main import app


def test_model_guide_endpoint_exposes_cards():
    client = TestClient(app)
    response = client.get("/api/model-guide")

    assert response.status_code == 200
    payload = response.json()
    keys = {model["key"] for model in payload["models"]}
    assert "llava_onevision_qwen2_05b_image_only" in keys
    assert "qwen25_vl_3b_image_only" in keys
    sample = payload["models"][0]
    assert "capabilities" in sample
    assert "limitation" in sample


def test_settings_endpoint_does_not_return_plain_token():
    client = TestClient(app)
    response = client.get("/api/settings")

    assert response.status_code == 200
    payload = response.json()
    assert "huggingface" in payload
    assert "token" not in payload["huggingface"]

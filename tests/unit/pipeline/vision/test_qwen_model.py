"""Unit tests for pipeline/qwen_model.py.

All tests mock the HTTP layer or test pure functions — no model loading required.
Compatible with: pip install openai httpx pillow pytest
"""

import base64
import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from selfsuvis.pipeline.vision.qwen import (
    QwenModel,
    _encode_image_base64,
    _health_check_ollama,
    _health_check_vllm,
    _parse_qwen_response,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def disabled_settings(monkeypatch):
    import selfsuvis.pipeline.vision.qwen as qm

    monkeypatch.setattr(qm.settings, "GEMMA_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_CLIP_THRESHOLD", 0.25)
    yield


@pytest.fixture
def small_image() -> Image.Image:
    return Image.new("RGB", (10, 10), color=(128, 64, 32))


# ── Pure function tests: _parse_qwen_response ─────────────────────────────────


def test_parse_qwen_response_valid():
    payload = {
        "vehicle_groups": [{"type": "truck", "count": 2, "color": "green", "position": "front"}],
        "road_surface": "asphalt",
        "road_condition": "clear",
        "scene_summary": "Two trucks on a clear asphalt road.",
    }
    result = _parse_qwen_response(json.dumps(payload))
    assert result["vehicle_groups"] == payload["vehicle_groups"]
    assert result["road_surface"] == "asphalt"
    assert result["road_condition"] == "clear"
    assert result["scene_summary"] == "Two trucks on a clear asphalt road."
    # Exactly the 4 expected keys
    assert set(result.keys()) == {"vehicle_groups", "road_surface", "road_condition", "scene_summary"}


def test_parse_qwen_response_empty_vehicles():
    payload = {
        "vehicle_groups": [],
        "road_surface": "gravel",
        "road_condition": "wet",
        "scene_summary": "Empty road.",
    }
    result = _parse_qwen_response(json.dumps(payload))
    assert result["vehicle_groups"] == []
    assert "parse_error" not in result


def test_parse_qwen_response_markdown_fenced():
    payload = {
        "vehicle_groups": [],
        "road_surface": "dirt",
        "road_condition": "unknown",
        "scene_summary": "Dirt track.",
    }
    raw = "```json\n" + json.dumps(payload) + "\n```"
    result = _parse_qwen_response(raw)
    assert "parse_error" not in result
    assert result["road_surface"] == "dirt"


def test_parse_qwen_response_markdown_fenced_no_lang():
    payload = {
        "vehicle_groups": [],
        "road_surface": "concrete",
        "road_condition": "clear",
        "scene_summary": "Empty concrete road.",
    }
    raw = "```\n" + json.dumps(payload) + "\n```"
    result = _parse_qwen_response(raw)
    assert "parse_error" not in result
    assert result["road_surface"] == "concrete"


def test_parse_qwen_response_invalid_json():
    result = _parse_qwen_response("this is not json at all")
    assert result.get("parse_error") is True
    assert "raw" in result


def test_parse_qwen_response_not_a_dict():
    # A valid JSON array — not a dict
    result = _parse_qwen_response(json.dumps(["truck", "car"]))
    assert result.get("parse_error") is True
    assert "raw" in result


def test_parse_qwen_response_raw_truncated_at_500():
    long_text = "x" * 1000
    result = _parse_qwen_response(long_text)
    assert result.get("parse_error") is True
    assert len(result["raw"]) <= 500


# ── Pure function tests: _encode_image_base64 ─────────────────────────────────


def test_encode_image_base64_returns_string(small_image):
    result = _encode_image_base64(small_image)
    assert isinstance(result, str)
    assert len(result) > 0


def test_encode_image_base64_is_valid_base64(small_image):
    result = _encode_image_base64(small_image)
    # Should not raise
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_encode_image_base64_resizes_large_image(monkeypatch):
    import selfsuvis.pipeline.vision.qwen as qm

    monkeypatch.setattr(qm.settings, "QWEN_IMAGE_MAX_SIDE", 32)
    large = Image.new("RGB", (128, 64), color=(1, 2, 3))
    encoded = _encode_image_base64(large)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert max(decoded.size) <= 32


# ── QwenModel disabled tests ──────────────────────────────────────────────────


def test_qwen_model_disabled_when_no_api_url(disabled_settings):
    model = QwenModel()
    assert model.is_enabled() is False


def test_qwen_model_disabled_returns_disabled_dict(disabled_settings, small_image):
    model = QwenModel()
    result = model.extract_frame_facts(small_image)
    assert result == {"disabled": True}


def test_qwen_model_extract_batch_disabled(disabled_settings, small_image):
    model = QwenModel()
    results = model.extract_batch([small_image, small_image])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(r == {"disabled": True} for r in results)


def test_qwen_extract_batch_parallel_preserves_order(monkeypatch, small_image):
    import selfsuvis.pipeline.vision.qwen as qm

    monkeypatch.setattr(qm.settings, "GEMMA_API_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(qm.settings, "QWEN_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_SIDECAR_CONCURRENCY", 3)

    model = QwenModel()
    model._healthy = True

    def fake_extract(image, subtitle_text=None, ocr_text=None, extra_context=None, domain_hint=None):
        idx = int(subtitle_text)
        time.sleep(0.03 * (3 - idx))
        return {"scene_summary": f"frame-{idx}"}

    with patch.object(model, "_extract_frame_facts_with_context", side_effect=fake_extract):
        results = model.extract_batch(
            [small_image, small_image, small_image],
            subtitle_texts=["0", "1", "2"],
        )

    assert [r["scene_summary"] for r in results] == ["frame-0", "frame-1", "frame-2"]


# ── Health check tests ────────────────────────────────────────────────────────


def test_health_check_vllm_ok():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        result = _health_check_vllm("http://qwen:8000/v1", timeout=5)
    assert result is True
    mock_get.assert_called_once_with("http://qwen:8000/health", timeout=5)


def test_health_check_vllm_fail():
    with patch("httpx.get", side_effect=Exception("connection refused")):
        result = _health_check_vllm("http://qwen:8000/v1", timeout=5)
    assert result is False


def test_health_check_ollama_ok():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        result = _health_check_ollama("http://ollama:11434/v1", timeout=5)
    assert result is True
    mock_get.assert_called_once_with("http://ollama:11434/api/tags", timeout=5)


def test_health_check_ollama_connection_refused():
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    with patch("httpx.get", return_value=mock_resp):
        result = _health_check_ollama("http://ollama:11434/v1", timeout=5)
    assert result is False

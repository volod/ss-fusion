"""Tests for the Qwen→Gemma migration in pipeline/qwen_model.py.

Validates that QwenModel now prefers GEMMA_API_URL/GEMMA_API_MODEL/GEMMA_API_BACKEND
over the legacy QWEN_* settings, while remaining backward-compatible when only
QWEN_API_URL is set.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import selfsuvis.pipeline.vision.qwen as qm

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(10, 20, 30))


@pytest.fixture
def gemma_settings(monkeypatch):
    """Configure GEMMA_API_URL (Gemma active, Qwen absent)."""
    monkeypatch.setattr(qm.settings, "GEMMA_API_URL", "http://gemma-host:11434/v1")
    monkeypatch.setattr(qm.settings, "GEMMA_API_MODEL", "gemma4:e4b")
    monkeypatch.setattr(qm.settings, "GEMMA_API_BACKEND", "ollama")
    monkeypatch.setattr(qm.settings, "GEMMA_API_TIMEOUT_SEC", 60)
    monkeypatch.setattr(qm.settings, "QWEN_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_MODEL", "qwen2.5vl:7b")
    monkeypatch.setattr(qm.settings, "QWEN_TIMEOUT_SEC", 30)
    monkeypatch.setattr(qm.settings, "QWEN_CLIP_THRESHOLD", 0.0)
    yield


@pytest.fixture
def qwen_only_settings(monkeypatch):
    """Legacy configuration: only QWEN_API_URL set (backward compat)."""
    monkeypatch.setattr(qm.settings, "GEMMA_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_API_URL", "http://qwen-host:8000/v1")
    monkeypatch.setattr(qm.settings, "QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setattr(qm.settings, "QWEN_TIMEOUT_SEC", 30)
    monkeypatch.setattr(qm.settings, "QWEN_BACKEND", "vllm")
    monkeypatch.setattr(qm.settings, "QWEN_CLIP_THRESHOLD", 0.0)
    yield


@pytest.fixture
def both_disabled(monkeypatch):
    monkeypatch.setattr(qm.settings, "GEMMA_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_API_URL", "")
    monkeypatch.setattr(qm.settings, "QWEN_CLIP_THRESHOLD", 0.0)
    yield


# ── is_enabled ────────────────────────────────────────────────────────────────

def test_is_enabled_with_gemma_url(gemma_settings):
    model = qm.QwenModel()
    assert model.is_enabled() is True


def test_is_enabled_with_qwen_only_url(qwen_only_settings):
    model = qm.QwenModel()
    assert model.is_enabled() is True


def test_is_enabled_when_both_disabled(both_disabled):
    model = qm.QwenModel()
    assert model.is_enabled() is False


def test_disabled_returns_disabled_dict(both_disabled, small_image):
    model = qm.QwenModel()
    assert model.extract_frame_facts(small_image) == {"disabled": True}


# ── Health check uses Gemma settings ─────────────────────────────────────────

def test_health_check_uses_gemma_url(gemma_settings):
    """_check_health hits the Gemma URL, not the legacy Qwen URL."""
    model = qm.QwenModel()

    with patch("selfsuvis.pipeline.vision.qwen._health_check_ollama", return_value=True) as mock_hc:
        model._check_health()

    mock_hc.assert_called_once()
    url_arg = mock_hc.call_args[0][0]
    assert url_arg == "http://gemma-host:11434/v1"


def test_health_check_falls_back_to_qwen_url(qwen_only_settings):
    """_check_health falls back to Qwen URL when GEMMA_API_URL is empty."""
    model = qm.QwenModel()

    with patch("selfsuvis.pipeline.vision.qwen._health_check_vllm", return_value=True) as mock_hc:
        model._check_health()

    mock_hc.assert_called_once()
    url_arg = mock_hc.call_args[0][0]
    assert "qwen-host" in url_arg


# ── API calls use Gemma model name ────────────────────────────────────────────

def _make_fake_openai_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_extract_frame_facts_sends_gemma_model(gemma_settings, small_image):
    """extract_frame_facts sends GEMMA_API_MODEL to the chat completions endpoint."""
    model = qm.QwenModel()
    model._healthy = True  # bypass health check

    fake_response = _make_fake_openai_response(
        '{"vehicle_groups": [], "road_surface": "asphalt", '
        '"road_condition": "clear", "scene_summary": "empty road"}'
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        model.extract_frame_facts(small_image)

    # Verify model name sent is the Gemma model
    call_kwargs = fake_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gemma4:e4b"
    # Result should be parsed successfully
    assert "road_surface" in result


def test_extract_frame_facts_uses_qwen_model_in_legacy_mode(qwen_only_settings, small_image):
    """In legacy mode (no GEMMA_API_URL), QWEN_MODEL is used."""
    model = qm.QwenModel()
    model._healthy = True

    fake_response = _make_fake_openai_response(
        '{"vehicle_groups": [], "road_surface": "dirt", '
        '"road_condition": "clear", "scene_summary": "field"}'
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        result = model.extract_frame_facts(small_image)

    call_kwargs = fake_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_extract_frame_facts_timeout_returns_timeout_dict(gemma_settings, small_image):
    """Timeout → {'timeout': True, 'timeout_sec': N}."""
    model = qm.QwenModel()
    model._healthy = True

    from openai import APITimeoutError

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = APITimeoutError.__new__(APITimeoutError)

    with patch("openai.OpenAI", return_value=fake_client):
        result = model.extract_frame_facts(small_image)

    assert result.get("timeout") is True
    assert "timeout_sec" in result


def test_extract_frame_facts_service_error_returns_unavailable(gemma_settings, small_image):
    """Generic exception → {'service_unavailable': True}."""
    model = qm.QwenModel()
    model._healthy = True

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = ConnectionError("refused")

    with patch("openai.OpenAI", return_value=fake_client):
        result = model.extract_frame_facts(small_image)

    assert result.get("service_unavailable") is True

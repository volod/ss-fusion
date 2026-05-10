"""Unit tests for pipeline/ocr_model.py — no GPU or model loading required."""

import base64
import io
import time
from unittest.mock import MagicMock, patch

from PIL import Image


def _make_image(w=64, h=64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


# ── _resolve_model_id ─────────────────────────────────────────────────────────


def test_resolve_explicit_model(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model

    monkeypatch.setattr(ocr_model.settings, "OCR_MODEL", "microsoft/trocr-base-printed")
    assert ocr_model._resolve_model_id() == "microsoft/trocr-base-printed"


def test_resolve_auto_delegates_to_registry(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model

    monkeypatch.setattr(ocr_model.settings, "OCR_MODEL", "auto")
    with (
        patch.object(ocr_model, "auto_select", return_value="ucaslcl/GOT-OCR2_0") as mock_sel,
        patch.object(ocr_model, "detect_resources", return_value={"vram_gb": 8.0, "ram_gb": 32.0}),
    ):
        result = ocr_model._resolve_model_id()
    mock_sel.assert_called_once_with("ocr", {"vram_gb": 8.0, "ram_gb": 32.0})
    assert result == "ucaslcl/GOT-OCR2_0"


def test_resolve_auto_fallback_when_none(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model

    monkeypatch.setattr(ocr_model.settings, "OCR_MODEL", "auto")
    with (
        patch.object(ocr_model, "auto_select", return_value=None),
        patch.object(ocr_model, "detect_resources", return_value={"vram_gb": 0.0, "ram_gb": 8.0}),
    ):
        result = ocr_model._resolve_model_id()
    assert result == "microsoft/trocr-base-printed"


# ── OCRModel.is_enabled ───────────────────────────────────────────────────────


def test_is_enabled_true(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", True)
    assert OCRModel().is_enabled() is True


def test_is_enabled_false(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", False)
    assert OCRModel().is_enabled() is False


# ── OCRModel.extract_text — disabled ─────────────────────────────────────────


def test_extract_text_disabled_returns_none_text(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", False)
    result = OCRModel().extract_text(_make_image())
    assert result["ocr_text"] is None
    assert result.get("ocr_disabled") is True


def test_extract_text_batch_disabled_all_disabled(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", False)
    imgs = [_make_image() for _ in range(3)]
    results = OCRModel().extract_text_batch(imgs)
    assert len(results) == 3
    assert all(r.get("ocr_disabled") for r in results)


# ── OCRModel backend selection ────────────────────────────────────────────────


def test_backend_is_vllm_when_api_url_set(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "http://localhost:8010/v1")
    monkeypatch.setattr(ocr_model.settings, "OCR_MODEL", "auto")
    model = OCRModel()
    model._model_id = "some-model"
    assert model._get_backend() == "vllm"


def test_backend_is_got_for_got_model_id(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "ucaslcl/GOT-OCR2_0"
    assert model._get_backend() == "got"


def test_backend_is_florence_for_florence_model(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/Florence-2-base"
    assert model._get_backend() == "florence"


def test_backend_is_trocr_for_trocr_model(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    assert model._get_backend() == "trocr"


def test_backend_is_vlm_for_deepseek_ocr(monkeypatch):
    """DeepSeek-OCR-2 uses the vlm backend (AutoProcessor + AutoModelForCausalLM)."""
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    monkeypatch.setattr(ocr_model.settings, "QWEN_API_URL", "")
    model = OCRModel()
    model._model_id = "deepseek-ai/DeepSeek-OCR-2"
    assert model._get_backend() == "vlm"


def test_backend_cached_on_second_call(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    b1 = model._get_backend()
    b2 = model._get_backend()
    assert b1 == b2 == "trocr"


# ── OCRModel._extract_one — success and error paths ──────────────────────────


def test_extract_one_returns_ocr_text_on_success(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    model._backend = "trocr"
    model._model = MagicMock()
    model._processor = MagicMock()

    with patch.object(model, "_extract_trocr", return_value="  hello world  "):
        result = model._extract_one(_make_image())

    assert result["ocr_text"] == "hello world"
    assert result["ocr_model"] == "microsoft/trocr-base-printed"
    assert "ocr_error" not in result


def test_extract_one_returns_error_dict_on_exception(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    model._backend = "trocr"

    with patch.object(model, "_extract_trocr", side_effect=RuntimeError("OOM")):
        result = model._extract_one(_make_image())

    assert result.get("ocr_error") is True
    assert result["ocr_text"] == ""


def test_extract_one_strips_whitespace(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    model._backend = "trocr"

    with patch.object(model, "_extract_trocr", return_value="\n  armored vehicle \t"):
        result = model._extract_one(_make_image())
    assert result["ocr_text"] == "armored vehicle"


# ── _encode_b64 ───────────────────────────────────────────────────────────────


def test_encode_b64_produces_valid_jpeg():
    from selfsuvis.pipeline.vision.ocr import _encode_b64

    img = _make_image(32, 32)
    b64str = _encode_b64(img)
    raw = base64.b64decode(b64str)
    # JPEG magic bytes
    assert raw[:2] == b"\xff\xd8"


def test_encode_b64_is_ascii_string():
    from selfsuvis.pipeline.vision.ocr import _encode_b64

    result = _encode_b64(_make_image())
    assert isinstance(result, str)
    result.encode("ascii")  # should not raise


def test_encode_b64_resizes_large_image(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import _encode_b64

    monkeypatch.setattr(ocr_model.settings, "OCR_IMAGE_MAX_SIDE", 40)
    b64str = _encode_b64(_make_image(160, 80))
    decoded = Image.open(io.BytesIO(base64.b64decode(b64str)))
    assert max(decoded.size) <= 40


# ── extract_text_batch — batch size matches input ─────────────────────────────


def test_extract_text_batch_length_matches_input(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "")
    model = OCRModel()
    model._model_id = "microsoft/trocr-base-printed"
    model._backend = "trocr"

    with patch.object(model, "_extract_trocr", return_value="text"):
        results = model.extract_text_batch([_make_image() for _ in range(5)])
    assert len(results) == 5


def test_extract_text_batch_vllm_parallel_preserves_order(monkeypatch):
    import selfsuvis.pipeline.vision.ocr as ocr_model
    from selfsuvis.pipeline.vision.ocr import OCRModel

    monkeypatch.setattr(ocr_model.settings, "OCR_ENABLED", True)
    monkeypatch.setattr(ocr_model.settings, "OCR_API_URL", "http://localhost:8010/v1")
    monkeypatch.setattr(ocr_model.settings, "OCR_SIDECAR_CONCURRENCY", 3)

    model = OCRModel()
    model._model_id = "some-sidecar-model"
    model._backend = "vllm"

    call_idx = {"value": 0}

    def fake_extract(_img):
        idx = call_idx["value"]
        call_idx["value"] += 1
        time.sleep(0.03 * (3 - idx))
        return {"ocr_text": f"text-{idx}", "ocr_model": model.model_id}

    with patch.object(model, "_extract_one", side_effect=fake_extract):
        results = model.extract_text_batch([_make_image() for _ in range(3)])

    assert [r["ocr_text"] for r in results] == ["text-0", "text-1", "text-2"]

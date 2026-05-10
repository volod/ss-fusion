"""Unit tests for pipeline/asr_model.py — no GPU or Whisper model required."""

from unittest.mock import MagicMock, patch

# ── _resolve_model_id ─────────────────────────────────────────────────────────


def test_resolve_explicit_model_id(monkeypatch):
    """When ASR_MODEL is a specific model ID, use it directly."""
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "openai/whisper-small")
    result = asr_model._resolve_model_id()
    assert result == "openai/whisper-small"


def test_resolve_auto_delegates_to_registry(monkeypatch):
    """When ASR_MODEL='auto', should call auto_select and return its result."""
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "auto")
    with (
        patch.object(asr_model, "auto_select", return_value="openai/whisper-medium") as mock_sel,
        patch.object(asr_model, "detect_resources", return_value={"vram_gb": 8.0, "ram_gb": 32.0}),
    ):
        result = asr_model._resolve_model_id()
    mock_sel.assert_called_once_with("asr", {"vram_gb": 8.0, "ram_gb": 32.0})
    assert result == "openai/whisper-medium"


def test_resolve_auto_fallback_when_registry_returns_none(monkeypatch):
    """If auto_select returns None, fall back to whisper-large-v3-turbo."""
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "auto")
    with (
        patch.object(asr_model, "auto_select", return_value=None),
        patch.object(asr_model, "detect_resources", return_value={"vram_gb": 0.0, "ram_gb": 8.0}),
    ):
        result = asr_model._resolve_model_id()
    assert result == "openai/whisper-large-v3-turbo"


def test_resolve_empty_string_treated_as_auto(monkeypatch):
    """Empty ASR_MODEL string should delegate to auto_select (not empty string)."""
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "")
    with (
        patch.object(asr_model, "auto_select", return_value="openai/whisper-tiny") as mock_sel,
        patch.object(asr_model, "detect_resources", return_value={"vram_gb": 2.0, "ram_gb": 16.0}),
    ):
        result = asr_model._resolve_model_id()
    mock_sel.assert_called_once()
    assert result == "openai/whisper-tiny"


# ── ASRModel.is_enabled ───────────────────────────────────────────────────────


def test_is_enabled_when_setting_true(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", True)
    assert ASRModel().is_enabled() is True


def test_is_enabled_when_setting_false(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", False)
    assert ASRModel().is_enabled() is False


# ── ASRModel.transcribe — disabled path ───────────────────────────────────────


def test_transcribe_returns_empty_when_disabled(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", False)
    model = ASRModel()
    result = model.transcribe("/fake/path.wav")
    assert result == []


# ── ASRModel.transcribe — enabled, mocked pipeline ───────────────────────────


def test_transcribe_returns_chunks_from_pipe(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", True)
    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "openai/whisper-tiny")
    monkeypatch.setattr(asr_model.settings, "ASR_LANGUAGE", "")
    monkeypatch.setattr(asr_model.settings, "ASR_BATCH_SIZE", 8)
    monkeypatch.setattr(asr_model.settings, "ASR_CHUNK_LENGTH_SEC", 30)
    monkeypatch.setattr(asr_model.settings, "USE_FP16", False)
    monkeypatch.setattr(asr_model.settings, "DEVICE", "cpu")

    fake_chunks = [
        {"text": "hello world", "timestamp": (0.0, 2.5)},
        {"text": "convoy spotted", "timestamp": (3.0, 5.0)},
    ]
    fake_pipe = MagicMock(return_value={"chunks": fake_chunks})

    model = ASRModel()
    model._pipe = fake_pipe
    model._model_id = "openai/whisper-tiny"

    result = model.transcribe("/fake/audio.wav")
    assert result == fake_chunks
    fake_pipe.assert_called_once()


def test_transcribe_returns_empty_on_pipe_exception(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", True)

    failing_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    model = ASRModel()
    model._pipe = failing_pipe
    model._model_id = "openai/whisper-large-v3"

    result = model.transcribe("/fake/audio.wav")
    assert result == []


def test_transcribe_passes_language_to_generate_kwargs(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", True)
    monkeypatch.setattr(asr_model.settings, "ASR_LANGUAGE", "uk")
    monkeypatch.setattr(asr_model.settings, "ASR_BATCH_SIZE", 4)
    monkeypatch.setattr(asr_model.settings, "ASR_CHUNK_LENGTH_SEC", 30)

    fake_pipe = MagicMock(return_value={"chunks": []})
    model = ASRModel()
    model._pipe = fake_pipe
    model._model_id = "openai/whisper-large-v3"

    model.transcribe("/fake/audio.wav")
    call_kwargs = fake_pipe.call_args.kwargs
    assert call_kwargs.get("generate_kwargs", {}).get("language") == "uk"


def test_transcribe_no_language_when_empty(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_ENABLED", True)
    monkeypatch.setattr(asr_model.settings, "ASR_LANGUAGE", "")
    monkeypatch.setattr(asr_model.settings, "ASR_BATCH_SIZE", 4)
    monkeypatch.setattr(asr_model.settings, "ASR_CHUNK_LENGTH_SEC", 30)

    fake_pipe = MagicMock(return_value={"chunks": []})
    model = ASRModel()
    model._pipe = fake_pipe
    model._model_id = "openai/whisper-tiny"

    model.transcribe("/fake/audio.wav")
    call_kwargs = fake_pipe.call_args.kwargs
    assert call_kwargs.get("generate_kwargs") == {"task": "transcribe"}


# ── ASRModel.model_id property ────────────────────────────────────────────────


def test_model_id_lazy_resolution(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "openai/whisper-base")
    model = ASRModel()
    assert model._model_id is None  # not yet resolved
    _ = model.model_id
    assert model._model_id == "openai/whisper-base"


def test_model_id_resolved_only_once(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model
    from selfsuvis.pipeline.vision.asr import ASRModel

    monkeypatch.setattr(asr_model.settings, "ASR_MODEL", "openai/whisper-base")
    model = ASRModel()
    id1 = model.model_id
    id2 = model.model_id
    assert id1 == id2 == "openai/whisper-base"


# ── _resolve_device ───────────────────────────────────────────────────────────


def test_resolve_device_cpu_when_no_torch(monkeypatch):
    """When torch is missing (ImportError), should always return 'cpu'."""
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "DEVICE", "auto")

    with patch("builtins.__import__", side_effect=ImportError):
        result = asr_model._resolve_device()
    assert result == "cpu"


def test_resolve_device_cpu_on_explicit_cpu_setting(monkeypatch):
    import selfsuvis.pipeline.vision.asr as asr_model

    monkeypatch.setattr(asr_model.settings, "DEVICE", "cpu")
    result = asr_model._resolve_device()
    assert result == "cpu"

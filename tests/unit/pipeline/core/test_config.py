"""Unit tests for pipeline.config."""

import pytest

from selfsuvis.pipeline.core import config


def test_validate_settings_success(monkeypatch):
    """validate_settings passes with valid config."""
    config.validate_settings()


def test_validate_settings_invalid_port(monkeypatch):
    """validate_settings raises when QDRANT_PORT is invalid."""
    monkeypatch.setattr(config.settings, "QDRANT_PORT", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "QDRANT_PORT" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "QDRANT_PORT", 6333)


def test_validate_settings_invalid_model_name(monkeypatch):
    """validate_settings raises when MODEL_NAME is invalid."""
    orig = config.settings.MODEL_NAME
    monkeypatch.setattr(config.settings, "MODEL_NAME", "invalid")
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "MODEL_NAME" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "MODEL_NAME", orig)


def test_validate_settings_negative_upload_bytes(monkeypatch):
    """validate_settings raises when MAX_UPLOAD_BYTES is negative."""
    orig = config.settings.MAX_UPLOAD_BYTES
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_BYTES", -1)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "MAX_UPLOAD_BYTES" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_BYTES", orig)


def test_validate_settings_invalid_ffmpeg_timeout(monkeypatch):
    """validate_settings raises when FFMPEG_TIMEOUT_SEC < 1."""
    orig = config.settings.FFMPEG_TIMEOUT_SEC
    monkeypatch.setattr(config.settings, "FFMPEG_TIMEOUT_SEC", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "FFMPEG_TIMEOUT_SEC" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "FFMPEG_TIMEOUT_SEC", orig)


def test_parse_allowed_paths_empty():
    """_parse_allowed_paths returns [] for None or empty string."""
    assert config._parse_allowed_paths(None) == []
    assert config._parse_allowed_paths("") == []
    assert config._parse_allowed_paths("   ") == []


def test_parse_allowed_paths_single():
    """_parse_allowed_paths returns single path."""
    assert config._parse_allowed_paths("/tmp") == ["/tmp"]


def test_parse_allowed_paths_multiple():
    """_parse_allowed_paths returns comma-separated paths."""
    result = config._parse_allowed_paths("/tmp, /var, /home")
    assert result == ["/tmp", "/var", "/home"]


def test_validate_settings_invalid_tile_size(monkeypatch):
    """validate_settings raises when TILE_SIZE < 1."""
    orig = config.settings.TILE_SIZE
    monkeypatch.setattr(config.settings, "TILE_SIZE", 0)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "TILE_SIZE" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "TILE_SIZE", orig)


def test_validate_settings_invalid_motion_range(monkeypatch):
    """validate_settings raises when MOTION_LOW > MOTION_HIGH."""
    low, high = config.settings.MOTION_LOW, config.settings.MOTION_HIGH
    monkeypatch.setattr(config.settings, "MOTION_LOW", 0.1)
    monkeypatch.setattr(config.settings, "MOTION_HIGH", 0.05)
    with pytest.raises(ValueError) as exc_info:
        config.validate_settings()
    assert "MOTION" in str(exc_info.value)
    monkeypatch.setattr(config.settings, "MOTION_LOW", low)
    monkeypatch.setattr(config.settings, "MOTION_HIGH", high)


def test_validate_settings_no_api_key_logs_warning(monkeypatch, caplog):
    """validate_settings logs a warning when API_KEY is not set."""
    import logging
    monkeypatch.setattr(config.settings, "API_KEY", "")
    with caplog.at_level(logging.WARNING, logger="selfsuvis.pipeline.config"):
        config.validate_settings()
    assert any("API_KEY" in r.message for r in caplog.records)


def test_validate_settings_no_allowed_paths_logs_warning(monkeypatch, caplog):
    """validate_settings logs a warning when ALLOWED_INDEX_PATHS is empty."""
    import logging
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [])
    with caplog.at_level(logging.WARNING, logger="selfsuvis.pipeline.config"):
        config.validate_settings()
    assert any("ALLOWED_INDEX_PATHS" in r.message for r in caplog.records)


def test_env_json_dict_valid(monkeypatch):
    """_env_json_dict returns parsed dict for valid JSON object values."""
    monkeypatch.setenv("CVAT_LABEL_MAPPINGS", '{"automobile":"car"}')
    parsed = config._env_json_dict("CVAT_LABEL_MAPPINGS", {})
    assert parsed == {"automobile": "car"}


def test_env_json_dict_invalid_json_fallback(monkeypatch):
    """_env_json_dict falls back to default for invalid JSON values."""
    monkeypatch.setenv("CVAT_LABEL_MAPPINGS", "{invalid")
    parsed = config._env_json_dict("CVAT_LABEL_MAPPINGS", {"default": "value"})
    assert parsed == {"default": "value"}


def test_env_json_dict_non_object_fallback(monkeypatch):
    """_env_json_dict falls back to default for non-object JSON values."""
    monkeypatch.setenv("CVAT_LABEL_MAPPINGS", '["not","an","object"]')
    parsed = config._env_json_dict("CVAT_LABEL_MAPPINGS", {"default": "value"})
    assert parsed == {"default": "value"}


def test_get_dino_model_name():
    """get_dino_model_name maps model families to concrete backbones."""
    assert config.get_dino_model_name("dinov2") == "dinov2_vitb14"
    assert config.get_dino_model_name("dinov3") == "dinov3_vitb14"
    assert config.get_dino_model_name("openclip") is None

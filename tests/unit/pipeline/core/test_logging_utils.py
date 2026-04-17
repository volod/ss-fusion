"""Unit tests for pipeline.logging_utils."""

import logging

import pytest

import selfsuvis.pipeline.core.logging as logging_utils_mod
from selfsuvis.pipeline.core.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_configured():
    """Reset the _CONFIGURED flag before each test."""
    original = logging_utils_mod._CONFIGURED
    logging_utils_mod._CONFIGURED = False
    yield
    logging_utils_mod._CONFIGURED = original


def test_configure_logging_sets_configured_flag():
    """configure_logging sets _CONFIGURED to True."""
    assert not logging_utils_mod._CONFIGURED
    configure_logging()
    assert logging_utils_mod._CONFIGURED


def test_configure_logging_is_idempotent():
    """Calling configure_logging multiple times does not raise and only configures once."""
    configure_logging()
    configure_logging()
    configure_logging()
    assert logging_utils_mod._CONFIGURED  # still True, no exception


def test_get_logger_returns_logger_instance():
    """get_logger returns a logging.Logger with the given name."""
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_get_logger_triggers_configure():
    """get_logger calls configure_logging if not yet configured."""
    assert not logging_utils_mod._CONFIGURED
    get_logger("trigger")
    assert logging_utils_mod._CONFIGURED


def test_get_logger_same_name_returns_same_instance():
    """Two calls with the same name return the same Logger object."""
    a = get_logger("shared.name")
    b = get_logger("shared.name")
    assert a is b


def test_configure_logging_respects_log_level_env(monkeypatch):
    """configure_logging passes the LOG_LEVEL env var to basicConfig."""
    from unittest.mock import patch

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    with patch("logging.basicConfig") as mock_basic:
        configure_logging()
    mock_basic.assert_called_once()
    _, kwargs = mock_basic.call_args
    assert kwargs.get("level") == "DEBUG"

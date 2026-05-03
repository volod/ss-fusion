"""Unit tests for GemmaSSLFinetuner in pipeline/ssl_finetune.py."""

from unittest.mock import MagicMock, patch

import pytest


def test_cpu_only_raises_skip_step():
    """GemmaSSLFinetuner raises SkipStep when CUDA is unavailable."""
    from selfsuvis.pipeline.training.ssl import GemmaSSLFinetuner, SkipStep

    fake_embedder = MagicMock()
    fake_embedder.encode_images.return_value = __import__("numpy").zeros((1, 1152), dtype="float32")

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(SkipStep, match="GemmaSSL requires CUDA"):
            GemmaSSLFinetuner(gemma_embedder=fake_embedder, device="auto")


def test_skip_step_is_runtime_error():
    """SkipStep is a subclass of RuntimeError so callers can catch it broadly."""
    from selfsuvis.pipeline.training.ssl import SkipStep

    exc = SkipStep("reason")
    assert isinstance(exc, RuntimeError)
    assert "reason" in str(exc)

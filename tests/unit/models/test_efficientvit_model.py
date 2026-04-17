"""Unit tests for models/efficientvit_model.py.

All tests mock torch/timm to avoid model downloads.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


def _make_small_image(w: int = 8, h: int = 8) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# ── EfficientViTEmbedder — disabled/import guard ──────────────────────────────

def test_efficientvit_embedder_missing_timm_raises():
    """ImportError raised with helpful message when timm is not installed."""
    import builtins
    real_import = builtins.__import__

    def _no_timm(name, *args, **kwargs):
        if name == "timm":
            raise ImportError("no module named timm")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_no_timm):
        from selfsuvis.models.efficientvit_model import EfficientViTEmbedder
        with pytest.raises(ImportError, match="timm is required"):
            EfficientViTEmbedder(device="cpu")


def test_efficientvit_embedder_cuda_oom_on_load_raises_helpful_message():
    """OOM during model load raises RuntimeError with VRAM hint."""
    import sys
    timm_mock = MagicMock()
    timm_mock.create_model.side_effect = RuntimeError("CUDA out of memory")

    with patch.dict("sys.modules", {"timm": timm_mock}):
        # Re-import so the mock is used at init time
        import importlib
        import selfsuvis.models.efficientvit_model as mod
        importlib.reload(mod)

        with pytest.raises(RuntimeError, match="EfficientViT Stage 1→2 requires"):
            mod.EfficientViTEmbedder(device="cpu")


def test_efficientvit_embedder_oom_on_encode_raises_helpful_message():
    """OOM during encode_images raises RuntimeError with VRAM hint."""
    import torch
    import sys

    # Build a mock timm model
    fake_model = MagicMock()
    # Dummy forward for dim inference (B=1 → (1, 384))
    dummy_out = torch.zeros(1, 384)
    fake_model.return_value = dummy_out
    fake_model.half.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    timm_mock = MagicMock()
    timm_mock.create_model.return_value = fake_model

    with patch.dict("sys.modules", {"timm": timm_mock}):
        import importlib
        import selfsuvis.models.efficientvit_model as mod
        importlib.reload(mod)

        embedder = mod.EfficientViTEmbedder(device="cpu", use_fp16=False)

        # Now make forward raise OOM
        fake_model.side_effect = RuntimeError("CUDA out of memory during encode")

        with pytest.raises(RuntimeError, match="EfficientViT Stage 1→2 requires"):
            embedder.encode_images([_make_small_image()])


def test_efficientvit_embedder_encode_empty():
    """encode_images([]) returns shape (0, dim) without errors."""
    import torch
    import sys

    fake_model = MagicMock()
    fake_model.return_value = torch.zeros(1, 384)
    fake_model.half.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    timm_mock = MagicMock()
    timm_mock.create_model.return_value = fake_model

    with patch.dict("sys.modules", {"timm": timm_mock}):
        import importlib
        import selfsuvis.models.efficientvit_model as mod
        importlib.reload(mod)

        embedder = mod.EfficientViTEmbedder(device="cpu", use_fp16=False)
        result = embedder.encode_images([])
        assert result.shape == (0, 384)


def test_efficientvit_embedder_encode_normalises():
    """encode_images returns L2-normalised vectors."""
    import torch
    import sys

    def _fake_forward(batch):
        # Return unnormalised embeddings (non-unit norms)
        return torch.ones(batch.shape[0], 384) * 5.0

    fake_model = MagicMock()
    fake_model.side_effect = _fake_forward
    fake_model.return_value = torch.zeros(1, 384)
    fake_model.half.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    timm_mock = MagicMock()
    timm_mock.create_model.return_value = fake_model

    with patch.dict("sys.modules", {"timm": timm_mock}):
        import importlib
        import selfsuvis.models.efficientvit_model as mod
        importlib.reload(mod)

        embedder = mod.EfficientViTEmbedder(device="cpu", use_fp16=False)
        # Override dim inference (happened with zeros in dummy pass)
        embedder._dim = 384

        images = [_make_small_image(), _make_small_image()]
        result = embedder.encode_images(images)

        assert result.dtype == np.float32
        assert result.shape == (2, 384)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_efficientvit_embedder_image_dim():
    """image_dim() returns the inferred embedding dimension."""
    import torch
    import sys

    fake_model = MagicMock()
    fake_model.return_value = torch.zeros(1, 256)
    fake_model.half.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    timm_mock = MagicMock()
    timm_mock.create_model.return_value = fake_model

    with patch.dict("sys.modules", {"timm": timm_mock}):
        import importlib
        import selfsuvis.models.efficientvit_model as mod
        importlib.reload(mod)

        embedder = mod.EfficientViTEmbedder(device="cpu", use_fp16=False)
        assert embedder.image_dim() == 256

"""Unit tests for pipeline.florence_model.

GPU tests (@pytest.mark.gpu) require a CUDA device and the Florence-2-large weights.
They are skipped automatically when torch.cuda.is_available() is False.

Non-GPU tests cover the confidence computation logic with mocked tensors so they
can run anywhere without model weights.
"""
from __future__ import annotations

import pytest
import torch

# ── helpers ───────────────────────────────────────────────────────────────────

_HAS_CUDA = torch.cuda.is_available()

# ── non-GPU tests: _compute_confidences ───────────────────────────────────────


def test_compute_confidences_basic():
    """Mean token probability is computed correctly for a 1-sequence, 2-step case."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    # batch=1, vocab=10, 2 steps
    step0 = torch.zeros(1, 10)
    step0[0, 3] = 10.0  # token 3 will have softmax probability ≈ 1.0
    step1 = torch.zeros(1, 10)
    step1[0, 5] = 10.0  # token 5 will have softmax probability ≈ 1.0

    generated_ids = torch.tensor([[3, 5]])  # batch=1, seq_len=2

    confs = _compute_confidences((step0, step1), generated_ids)

    assert len(confs) == 1
    assert isinstance(confs[0], float)
    assert abs(confs[0] - 1.0) < 0.01, f"Expected ~1.0 confidence, got {confs[0]}"


def test_compute_confidences_empty_scores():
    """Falls back to 0.5 when scores tuple is empty."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    generated_ids = torch.tensor([[3, 5]])
    confs = _compute_confidences((), generated_ids)
    assert confs == [0.5]


def test_compute_confidences_none_scores():
    """Falls back to 0.5 when scores is None."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    generated_ids = torch.tensor([[3, 5]])
    confs = _compute_confidences(None, generated_ids)
    assert confs == [0.5]


def test_compute_confidences_empty_sequence():
    """Falls back to 0.5 when generated_ids has seq_len=0."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    step0 = torch.zeros(1, 10)
    step0[0, 3] = 10.0
    generated_ids = torch.zeros(1, 0, dtype=torch.long)

    confs = _compute_confidences((step0,), generated_ids)
    assert confs == [0.5]


def test_compute_confidences_clamped():
    """Result is always in [0.0, 1.0] even with unusual inputs."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    # Uniform logits → softmax probability = 1/vocab ≈ 0.1 for vocab=10
    step0 = torch.zeros(1, 10)
    generated_ids = torch.tensor([[3]])

    confs = _compute_confidences((step0,), generated_ids)
    assert 0.0 <= confs[0] <= 1.0


def test_compute_confidences_batch():
    """Handles batch_size > 1 correctly."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    # batch=2, vocab=10, 1 step
    step0 = torch.zeros(2, 10)
    step0[0, 2] = 10.0   # sequence 0 → token 2 with prob ≈ 1.0
    step0[1, 7] = 10.0   # sequence 1 → token 7 with prob ≈ 1.0

    generated_ids = torch.tensor([[2], [7]])

    confs = _compute_confidences((step0,), generated_ids)
    assert len(confs) == 2
    assert abs(confs[0] - 1.0) < 0.01
    assert abs(confs[1] - 1.0) < 0.01


def test_compute_confidences_padding_skipped():
    """Token id=1 (padding/EOS) is skipped; count stays 0 → falls back to 0.5."""
    from selfsuvis.pipeline.vision.florence import _compute_confidences

    step0 = torch.zeros(1, 10)
    step0[0, 1] = 10.0   # token 1 = padding — should be skipped
    generated_ids = torch.tensor([[1]])

    confs = _compute_confidences((step0,), generated_ids)
    # token 1 is skipped, count[0]=0 → fallback 0.5
    assert confs == [0.5]


# ── GPU tests ─────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU required for Florence-2 inference")
def test_caption_batch_returns_correct_length():
    """caption_batch returns exactly one result per input image."""
    from PIL import Image
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = FlorenceModel()
    images = [Image.new("RGB", (224, 224))] * 3
    results = model.caption_batch(images)

    assert len(results) == 3


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU required for Florence-2 inference")
def test_caption_batch_confidence_is_float_in_range():
    """confidence is a float in [0.0, 1.0] — validates output_scores wiring."""
    from PIL import Image
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = FlorenceModel()
    images = [Image.new("RGB", (32, 32))]
    results = model.caption_batch(images)

    assert len(results) == 1
    caption, confidence = results[0]
    assert isinstance(caption, str)
    assert isinstance(confidence, float), f"confidence should be float, got {type(confidence)}"
    assert 0.0 <= confidence <= 1.0, f"confidence={confidence} is out of [0, 1]"


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU required for Florence-2 inference")
def test_caption_batch_empty_input():
    """Empty input list returns empty output."""
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = FlorenceModel()
    results = model.caption_batch([])
    assert results == []


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU required for Florence-2 inference")
def test_caption_batch_caption_non_null():
    """Non-empty image produces a non-NULL caption (may be empty string but not None)."""
    from PIL import Image
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = FlorenceModel()
    # Use a random noise image — more interesting than solid black
    import numpy as np
    arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    results = model.caption_batch([img])

    caption, confidence = results[0]
    assert caption is not None, "caption must not be None"
    assert isinstance(caption, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA GPU required for Florence-2 inference")
def test_model_tag():
    """model_tag has the structured provenance format model:version:precision."""
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = FlorenceModel()
    tag = model.model_tag
    parts = tag.split(":")
    assert len(parts) == 3, f"Expected 3 colon-separated parts, got: {tag!r}"
    assert parts[0] == "florence-2-large"
    assert parts[1].startswith("v"), f"Prompt version should start with 'v', got: {parts[1]!r}"
    assert parts[2] in {"fp16", "fp32"}, f"Precision must be fp16 or fp32, got: {parts[2]!r}"

"""Unit tests for pipeline.florence_model.

GPU tests (@pytest.mark.gpu) require a CUDA device and the Florence-2-large weights.
They are skipped automatically when torch.cuda.is_available() is False.

Non-GPU tests cover the confidence computation logic with mocked tensors so they
can run anywhere without model weights.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

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


def test_sanitize_model_inputs_drops_none_and_casts_float_tensor():
    from selfsuvis.pipeline.vision.florence import _sanitize_model_inputs

    inputs = {
        "pixel_values": torch.ones(1, 3, 4, 4, dtype=torch.float32),
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "token_type_ids": None,
    }

    cleaned = _sanitize_model_inputs(inputs, device="cpu", dtype=torch.float16)

    assert "token_type_ids" not in cleaned
    assert cleaned["pixel_values"].dtype == torch.float16
    assert cleaned["input_ids"].dtype == torch.long


def test_extract_generated_token_ids_uses_scores_length_not_prompt_length():
    from selfsuvis.pipeline.vision.florence import _extract_generated_token_ids

    sequences = torch.tensor([[11, 12, 13, 14]])
    scores = (torch.zeros(1, 10), torch.zeros(1, 10))

    generated_ids = _extract_generated_token_ids(sequences, scores)

    assert generated_ids.tolist() == [[13, 14]]


def test_scores_are_usable_rejects_none_entries():
    from selfsuvis.pipeline.vision.florence import _scores_are_usable

    assert _scores_are_usable((torch.zeros(1, 10),)) is True
    assert _scores_are_usable((None,)) is False
    assert _scores_are_usable(None) is False


def test_run_inference_accepts_missing_scores(monkeypatch):
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    class _FakeProcessor:
        def __call__(self, **kwargs):
            return {
                "pixel_values": torch.ones(1, 3, 4, 4),
                "input_ids": torch.ones(1, 2, dtype=torch.long),
            }

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["road scene"]

        def post_process_generation(self, raw, task, image_size):
            return {task: raw}

    model = object.__new__(FlorenceModel)
    model.device = "cpu"
    model._generation_mode = "scored"
    model._model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    model._processor = _FakeProcessor()

    monkeypatch.setattr(
        model,
        "_generate_with_fallback",
        lambda inputs: SimpleNamespace(sequences=torch.tensor([[11, 12, 13]]), scores=None),
    )

    results = model._run_inference([Image.new("RGB", (8, 8))])

    assert results == [("road scene", 0.5)]


def test_run_inference_accepts_scores_tuple_with_none_entries(monkeypatch):
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    class _FakeProcessor:
        def __call__(self, **kwargs):
            return {
                "pixel_values": torch.ones(1, 3, 4, 4),
                "input_ids": torch.ones(1, 2, dtype=torch.long),
            }

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["bridge"]

        def post_process_generation(self, raw, task, image_size):
            return {task: raw}

    model = object.__new__(FlorenceModel)
    model.device = "cpu"
    model._generation_mode = "scored"
    model._model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    model._processor = _FakeProcessor()

    monkeypatch.setattr(
        model,
        "_generate_with_fallback",
        lambda inputs: SimpleNamespace(sequences=torch.tensor([[11, 12]]), scores=(None, None)),
    )

    results = model._run_inference([Image.new("RGB", (8, 8))])

    assert results == [("bridge", 0.5)]


def test_generate_with_fallback_uses_caption_only_mode_for_eager_runtime():
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    calls = []

    class _FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(sequences=torch.tensor([[11, 12, 13]]), scores=None)

    model = object.__new__(FlorenceModel)
    model.device = "cpu"
    model._generation_mode = "eager"
    model._model = _FakeModel()

    generated = model._generate_with_fallback(
        {
            "pixel_values": torch.ones(1, 3, 4, 4),
            "input_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )

    assert generated.sequences.tolist() == [[11, 12, 13]]
    assert len(calls) == 1
    assert calls[0]["output_scores"] is False
    assert calls[0]["num_beams"] == 1
    assert calls[0]["use_cache"] is False
    assert model.runtime_mode == "eager-noscores"


def test_generate_with_fallback_retries_caption_only_after_scored_failure():
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    calls = []

    class _FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            if kwargs["output_scores"] is True:
                raise AttributeError("'NoneType' object has no attribute 'shape'")
            return SimpleNamespace(sequences=torch.tensor([[21, 22]]), scores=None)

    model = object.__new__(FlorenceModel)
    model.device = "cpu"
    model._generation_mode = "scored"
    model._model = _FakeModel()

    generated = model._generate_with_fallback(
        {
            "pixel_values": torch.ones(1, 3, 4, 4),
            "input_ids": torch.ones(1, 2, dtype=torch.long),
        }
    )

    assert generated.sequences.tolist() == [[21, 22]]
    assert [call["output_scores"] for call in calls] == [True, False]
    assert all(call["num_beams"] == 1 for call in calls)
    assert all(call["use_cache"] is False for call in calls)
    assert model.runtime_mode == "caption-only"


def test_caption_batch_chunk_falls_back_to_single_image_on_square_feature_assertion(monkeypatch):
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    model = object.__new__(FlorenceModel)
    images = [Image.new("RGB", (32, 24)), Image.new("RGB", (24, 32))]
    calls = {"single": 0}

    def _raise_batch_failure(_images):
        raise AssertionError("only support square feature maps for now")

    def _single_result(_image):
        calls["single"] += 1
        return (f"caption-{calls['single']}", 0.75)

    monkeypatch.setattr(model, "_run_inference", _raise_batch_failure)
    monkeypatch.setattr(model, "_caption_single", _single_result)

    results = model._caption_batch_chunk(images, batch_size=2)

    assert results == [("caption-1", 0.75), ("caption-2", 0.75)]
    assert calls["single"] == 2


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

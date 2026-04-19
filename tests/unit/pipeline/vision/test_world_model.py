"""Focused tests for world-model input preparation."""

import torch


def test_prepare_model_inputs_aligns_float_dtype_to_model():
    from selfsuvis.pipeline.vision.world import _prepare_model_inputs

    inputs = {
        "pixel_values": torch.ones(1, 8, 3, 16, 16, dtype=torch.float32),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }

    prepared = _prepare_model_inputs(inputs, device="cpu", dtype=torch.float16)

    assert prepared["pixel_values"].dtype == torch.float16
    assert prepared["attention_mask"].dtype == torch.long


def test_prepare_model_inputs_drops_none_values():
    from selfsuvis.pipeline.vision.world import _prepare_model_inputs

    prepared = _prepare_model_inputs(
        {"pixel_values": torch.ones(1, 1), "optional": None},
        device="cpu",
        dtype=torch.float32,
    )

    assert "optional" not in prepared

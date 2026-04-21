"""Focused tests for world-model input preparation."""

from types import SimpleNamespace

import torch
from PIL import Image


def test_is_videomae_pretraining_checkpoint_matches_pretraining_config():
    from selfsuvis.pipeline.vision.world import _is_videomae_pretraining_checkpoint

    config = SimpleNamespace(
        model_type="videomae",
        architectures=["VideoMAEForPreTraining"],
    )

    assert _is_videomae_pretraining_checkpoint("MCG-NJU/videomae-large", config) is True


def test_is_videomae_pretraining_checkpoint_rejects_non_pretraining_models():
    from selfsuvis.pipeline.vision.world import _is_videomae_pretraining_checkpoint

    config = SimpleNamespace(
        model_type="videomae",
        architectures=["VideoMAEModel"],
    )

    assert _is_videomae_pretraining_checkpoint("some-org/custom-videomae", config) is False


def test_remap_videomae_state_dict_converts_legacy_attention_biases():
    from selfsuvis.pipeline.vision.world import _remap_videomae_state_dict_for_modern_transformers

    state_dict = {
        "encoder.layer.0.attention.attention.query.weight": torch.ones(4, 4),
        "encoder.layer.0.attention.attention.key.weight": torch.ones(4, 4) * 2,
        "encoder.layer.0.attention.attention.value.weight": torch.ones(4, 4) * 3,
        "encoder.layer.0.attention.attention.q_bias": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "encoder.layer.0.attention.attention.v_bias": torch.tensor([5.0, 6.0, 7.0, 8.0]),
    }

    remapped = _remap_videomae_state_dict_for_modern_transformers(state_dict)

    assert "encoder.layer.0.attention.attention.q_bias" not in remapped
    assert "encoder.layer.0.attention.attention.v_bias" not in remapped
    assert torch.equal(
        remapped["encoder.layer.0.attention.attention.query.bias"],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert torch.equal(
        remapped["encoder.layer.0.attention.attention.value.bias"],
        torch.tensor([5.0, 6.0, 7.0, 8.0]),
    )
    assert torch.equal(
        remapped["encoder.layer.0.attention.attention.key.bias"],
        torch.zeros(4),
    )


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


def test_world_model_process_clip_uses_local_videomae_checkpoint_path(monkeypatch, tmp_path):
    from selfsuvis.pipeline.vision import world as wm_mod

    class _DummyFeatureExtractor:
        def __call__(self, images, return_tensors: str = "pt", **_kwargs):
            return {"pixel_values": torch.ones(1, len(images), 3, 16, 16, dtype=torch.float32)}

    class _DummyModel:
        def __init__(self):
            self.config = SimpleNamespace(num_frames=4)

        def parameters(self):
            yield torch.zeros(1, dtype=torch.float32)

        def __call__(self, **_inputs):
            hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(wm_mod.settings, "WORLD_MODEL_ENABLED", True)
    monkeypatch.setattr(wm_mod.settings, "WORLD_MODEL", "MCG-NJU/videomae-large")
    monkeypatch.setattr(wm_mod.settings, "WORLD_MODEL_CLIP_FRAMES", 8)
    monkeypatch.setattr(wm_mod.settings, "WORLD_MODEL_STORE_EMBED", False)
    monkeypatch.setattr(wm_mod.settings, "USE_FP16", False)
    monkeypatch.setattr(wm_mod, "_get_device", lambda: "cpu")
    cache_dir = tmp_path / "videomae-cache"
    cache_dir.mkdir()
    (cache_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(wm_mod, "_resolve_local_world_model_path", lambda _model_id: cache_dir)
    monkeypatch.setattr(
        wm_mod,
        "_load_world_preprocessor",
        lambda *_args, **_kwargs: _DummyFeatureExtractor(),
    )

    def _fake_loader(source, *, device: str, dtype):
        assert source == cache_dir
        assert device == "cpu"
        assert dtype == torch.float32
        return _DummyModel()

    monkeypatch.setattr(wm_mod, "_load_videomae_encoder_from_local_checkpoint", _fake_loader)

    class _DummyConfig:
        model_type = "videomae"
        architectures = ["VideoMAEForPreTraining"]

    class _DummyAutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _DummyConfig()

    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=_DummyAutoConfig,
            AutoImageProcessor=object(),
            AutoModel=object(),
            AutoProcessor=object(),
            VideoMAEImageProcessor=object(),
        ),
    )

    model = wm_mod.WorldModel()
    images = [Image.new("RGB", (8, 8), color="white") for _ in range(5)]
    result = model.process_clip(images)

    assert result["world_model"]["embedding_dim"] == 4
    assert result["world_model"]["temporal_window_frames"] == 4
    assert result["world_model"]["model"] == "MCG-NJU/videomae-large"
    assert model._load_note == "legacy VideoMAE checkpoint remapped successfully (local cache: videomae-cache)"

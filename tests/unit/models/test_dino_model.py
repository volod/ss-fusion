"""Tests for DINO model loading logic: alias resolution, HF wrapper, embed-dim lookup."""

import os
from unittest.mock import patch

import pytest


def test_resolve_dino_hub_aliases_dinov3_to_dinov2_reg():
    from selfsuvis.models.dino_model import _resolve_dino_hub

    with (
        patch("torch.hub.get_dir", return_value="/fake/hub"),
        patch("os.path.isdir", return_value=False),
    ):
        source, _repo, actual_name = _resolve_dino_hub("dinov3_vitb14")

    assert actual_name == "dinov2_vitb14_reg"
    assert source == "github"


def test_resolve_dino_hub_all_dinov3_aliases_map_to_reg():
    from selfsuvis.models.dino_model import _DINO_MODEL_ALIAS, _resolve_dino_hub

    for alias, expected in _DINO_MODEL_ALIAS.items():
        assert expected.endswith("_reg"), f"{alias} should map to a _reg variant"
        with (
            patch("torch.hub.get_dir", return_value="/fake/hub"),
            patch("os.path.isdir", return_value=False),
        ):
            _, _, actual_name = _resolve_dino_hub(alias)
        assert actual_name == expected


def test_resolve_dino_hub_returns_local_when_cache_dir_exists():
    from selfsuvis.models.dino_model import _resolve_dino_hub

    with (
        patch("torch.hub.get_dir", return_value="/fake/hub"),
        patch("os.path.isdir", return_value=True),
    ):
        source, repo_or_dir, actual_name = _resolve_dino_hub("dinov2_vitb14")

    assert source == "local"
    assert "facebookresearch_dinov2_main" in repo_or_dir
    assert actual_name == "dinov2_vitb14"


def test_resolve_dino_hub_returns_github_when_no_cache():
    from selfsuvis.models.dino_model import DINO_HUB_REPO, _resolve_dino_hub

    with (
        patch("torch.hub.get_dir", return_value="/fake/hub"),
        patch("os.path.isdir", return_value=False),
    ):
        source, repo_or_dir, _ = _resolve_dino_hub("dinov2_vitb14")

    assert source == "github"
    assert repo_or_dir == DINO_HUB_REPO


def test_hf_dino_wrapper_returns_cls_token():
    import torch
    from types import SimpleNamespace

    from selfsuvis.models.dino_model import _HFDINOWrapper

    B, N, D = 2, 197, 768
    fake_hidden = torch.randn(B, N, D)

    class _FakeHFModel(torch.nn.Module):
        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=fake_hidden)

    wrapper = _HFDINOWrapper(_FakeHFModel())
    x = torch.zeros(B, 3, 224, 224)
    out = wrapper(x)

    assert out.shape == (B, D)
    # CLS token is the first position along the sequence dimension.
    assert torch.allclose(out, fake_hidden[:, 0])


def test_load_dino_from_hf_raises_for_reg_variants_not_on_hf():
    # dinov3_vitb14 → dinov2_vitb14_reg, which has no HF repo entry.
    # This catches the silent failure mode where _reg models cannot be served by HF.
    from selfsuvis.models.dino_model import _load_dino_from_hf

    with pytest.raises(KeyError, match="No Hugging Face repo"):
        _load_dino_from_hf("dinov3_vitb14")


def test_dino_embed_dim_covers_all_vitb_variants():
    from selfsuvis.models.dino_model import _DINO_EMBED_DIM

    assert _DINO_EMBED_DIM["dinov2_vits14"] == 384
    assert _DINO_EMBED_DIM["dinov2_vitb14"] == 768
    assert _DINO_EMBED_DIM["dinov2_vitl14"] == 1024
    assert _DINO_EMBED_DIM["dinov2_vitg14"] == 1536


def test_dino_embed_dim_does_not_include_reg_variants():
    # _reg variants must resolve through _DINO_MODEL_ALIAS, not get their own entry.
    # If they got their own entry, future dim changes would require updating two places.
    from selfsuvis.models.dino_model import _DINO_EMBED_DIM

    for key in _DINO_EMBED_DIM:
        assert not key.endswith("_reg"), f"_reg variant {key} should not have its own embed-dim entry"

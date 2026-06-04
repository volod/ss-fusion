"""Cache-presence checkers for all model types and verify helper."""

import os
from pathlib import Path

from selfsuvis.pipeline.core.logging import get_logger

from ._utils import _CACHE_DIR

log = get_logger("prepare_models")


def _is_hf_cached(model_id: str) -> bool:
    """Return True if at least the config for *model_id* is in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for fname in (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ):
            result = try_to_load_from_cache(repo_id=model_id, filename=fname)
            if result is not None:
                return True
    except Exception:
        pass
    return False


def _is_openclip_cached(model: str, pretrained: str) -> bool:
    """Return True if open_clip weights are in the local cache.

    open_clip downloads weights to $CLIP_CACHE (.data/.cache/clip by default),
    naming each file after the URL basename (e.g. ViT-B-16.pt).
    HuggingFace-hosted pretrained models land in the HF hub cache instead.
    """
    try:
        clip_cache = Path(os.getenv("CLIP_CACHE", str(_CACHE_DIR / "clip")))
        if clip_cache.exists():
            try:
                import open_clip as _oc

                cfg = _oc.get_pretrained_cfg(model, pretrained)
                url = (cfg or {}).get("url", "")
                if url:
                    fname = Path(url).name
                    if (clip_cache / fname).exists():
                        return True
            except Exception:
                pass
            model_stem = model.replace("/", "-")
            for f in clip_cache.iterdir():
                if f.name.startswith(model_stem) or f.stem == model_stem:
                    return True

        hf_hub_id = None
        try:
            import open_clip as _oc

            cfg = _oc.get_pretrained_cfg(model, pretrained)
            hf_hub_id = (cfg or {}).get("hf_hub", "").rstrip("/")
        except Exception:
            pass
        if hf_hub_id:
            from huggingface_hub import try_to_load_from_cache as _try_cache

            for fname in (
                "open_clip_model.safetensors",
                "model.safetensors",
                "pytorch_model.bin",
                "config.json",
            ):
                try:
                    if _try_cache(repo_id=hf_hub_id, filename=fname) is not None:
                        return True
                except Exception:
                    pass
    except Exception:
        pass
    return False


def _is_dino_hub_cached(_model_name: str) -> bool:
    """Return True if the DINOv2 torch.hub archive is present."""
    try:
        import torch.hub as _hub

        hub_dir = Path(_hub.get_dir())
        repo_dir = hub_dir / "facebookresearch_dinov2_main"
        return repo_dir.exists()
    except Exception:
        pass
    return False


def _is_florence2_complete(model_id: str) -> bool:
    """Return True only if the Florence-2 snapshot includes modeling_florence2.py.

    A partial first download may cache only config files and satisfy _is_hf_cached(),
    but the runtime loader requires modeling_florence2.py (trust_remote_code=True).
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(repo_id=model_id, filename="modeling_florence2.py") is not None
    except Exception:
        return False


def _verify_models(model_specs: list) -> tuple:
    """Check cache status for a list of (label, cache_check_fn) tuples.

    Returns (ok_list, missing_list).
    """
    ok, missing = [], []
    for label, check_fn in model_specs:
        cached = False
        try:
            cached = check_fn()
        except Exception:
            pass
        if cached:
            ok.append(label)
        else:
            missing.append(label)
    return ok, missing

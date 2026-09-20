"""Model-weight cache probes used by startup preflight and prepare_models."""

import os
import shutil
import subprocess
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.getenv("CACHE_DIR", "./.data/.cache"))


def is_hf_cached(model_id: str) -> bool:
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


def is_openclip_cached(model: str, pretrained: str) -> bool:
    """Return True if open_clip weights are in the local cache."""
    try:
        clip_cache = Path(os.getenv("CLIP_CACHE", str(cache_dir() / "clip")))
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
            for item in clip_cache.iterdir():
                if item.name.startswith(model_stem) or item.stem == model_stem:
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


def is_dino_hub_cached(_model_name: str) -> bool:
    """Return True if the DINOv2 torch.hub archive is present."""
    try:
        import torch.hub as _hub

        hub_dir = Path(_hub.get_dir())
        repo_dir = hub_dir / "facebookresearch_dinov2_main"
        return repo_dir.exists()
    except Exception:
        pass
    return False


def is_yolo_cached(model_id: str) -> bool:
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    return (cache_dir() / "ultralytics" / model_file).exists()


_SCENETOK_CHECKPOINT_URLS = {
    "va-videodc_re10k": True,
    "va-videodc_dl3dv": True,
    "va-wan_dl3dv": True,
}


def normalize_scenetok_checkpoint_name(checkpoint_name: str) -> str:
    raw = (checkpoint_name or "").strip()
    if not raw:
        raise ValueError("SceneTok checkpoint name is empty.")
    ckpt_key = raw[:-5] if raw.endswith(".ckpt") else raw
    if ckpt_key not in _SCENETOK_CHECKPOINT_URLS:
        known = ", ".join(_SCENETOK_CHECKPOINT_URLS)
        raise ValueError(
            f"Unknown SceneTok checkpoint {checkpoint_name!r}. Known checkpoints: {known}"
        )
    return f"{ckpt_key}.ckpt"


def is_scenetok_cached(checkpoint_name: str) -> bool:
    ckpt_file = normalize_scenetok_checkpoint_name(checkpoint_name)
    return (cache_dir() / "selfsuvis" / "scenetok" / ckpt_file).exists()


def is_gemma_cached(model_id: str) -> bool:
    return is_hf_cached(model_id)


def is_ollama_model_cached(model: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0

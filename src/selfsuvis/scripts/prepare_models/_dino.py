"""DINOv2/v3 download, HF-fallback load, and dummy forward-pass verification."""

import os
import time
from pathlib import Path

from selfsuvis.pipeline.core.logging import get_logger

from ._utils import _CACHE_DIR, _label, _warmed

log = get_logger("prepare_models")


def _hf_load_with_license_check(model_name: str, hf_repo: str):
    from selfsuvis.models.dino_model import _load_dino_from_hf

    _GatedRepoError = None
    try:
        from huggingface_hub.errors import GatedRepoError as _GatedRepoError
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError as _GatedRepoError
        except ImportError:
            pass

    try:
        return _load_dino_from_hf(model_name)
    except Exception as exc:
        if _GatedRepoError and isinstance(exc, _GatedRepoError):
            raise RuntimeError(
                f"\nModel '{hf_repo}' requires license acceptance on Hugging Face.\n"
                f"  1. Open  https://huggingface.co/{hf_repo}\n"
                f"  2. Click 'Agree and access repository'\n"
                f"  3. Log in via CLI:  huggingface-cli login\n"
                f"  4. Re-run:  python scripts/prepare_models --dino --source hf"
            ) from exc
        raise


def _run_dummy(model, device: str) -> None:
    import torch as _torch

    dummy = _torch.zeros(1, 3, 224, 224, device=device)
    with _torch.no_grad():
        try:
            model(dummy)
        except RuntimeError as exc:
            msg = str(exc)
            if "memory_efficient_attention" in msg or "No operator found" in msg:
                log.warning(
                    "  Warmup forward pass skipped on %s — xformers does not yet support "
                    "this GPU (sm_12x+). Cached weights are usable at runtime once xformers "
                    "adds sm_120 support.",
                    device,
                )
            else:
                raise


def _download_dino(model_name: str, device: str, source: str = "auto") -> None:
    """Download (or verify) DINO weights."""
    import torch
    import torch.hub as _hub

    from selfsuvis.models.dino_model import (
        _DINO_HF_REPO,
        _DINO_MODEL_ALIAS,
        DINO_HUB_REPO,
        _resolve_dino_hub,
        hub_load_dino,
    )

    actual_name = _DINO_MODEL_ALIAS.get(model_name, model_name)
    label = _label(model_name, actual_name)

    if source == "hf":
        hf_repo = _DINO_HF_REPO.get(actual_name, "?")
        log.info("DINO (HF) — %s  hf_repo=%s", label, hf_repo)
        t0 = time.monotonic()
        model = _hf_load_with_license_check(model_name, hf_repo)
        model.to(device)
        _run_dummy(model, device)
        _warmed.add(model_name)
        log.info("  [ok] DINO cached from HF  %s  (%.1fs)", label, time.monotonic() - t0)
        return

    hub_source, repo_or_dir, resolved = _resolve_dino_hub(model_name)
    log.info("DINO — %s  source=%s", label, hub_source)

    if model_name in _warmed:
        log.info("  [ok] DINO already warmed in this session  %s", label)
        return

    if hub_source == "local":
        log.info("  Hub archive cached at %s", repo_or_dir)
        hub_checkpoints = (
            Path(os.getenv("TORCH_HOME", str(_CACHE_DIR / "torch"))) / "hub" / "checkpoints"
        )
        if any(hub_checkpoints.glob(f"{resolved}*pretrain*.pth")):
            _warmed.add(model_name)
            log.info("  [ok] DINO weights cached  %s  — skipping load", label)
            return
    else:
        log.info("  Downloading hub archive from %s …", DINO_HUB_REPO)
        log.info("  Cache dir: %s", _hub.get_dir())

    _orig = _hub.download_url_to_file

    def _prog(url: str, dst: str, *a, **kw):
        log.info("  ↓ %s", url)
        try:
            import urllib.request as _req

            from tqdm import tqdm as _tqdm

            with _req.urlopen(url) as r:
                total = int(r.headers.get("Content-Length", 0))
            bar = _tqdm(
                total=total, unit="B", unit_scale=True, desc=f"    {Path(dst).name}", leave=True
            )

            def _hook(_blk, blk_sz, tot):
                if tot > 0:
                    bar.total = tot
                bar.update(blk_sz)

            _req.urlretrieve(url, dst, reporthook=_hook)
            bar.close()
            log.info("  [ok] saved %s", dst)
        except Exception:
            _orig(url, dst, *a, **kw)

    _hub.download_url_to_file = _prog
    try:
        t0 = time.monotonic()
        if source == "hub":
            model = torch.hub.load(repo_or_dir, resolved, pretrained=True, source=hub_source)
        else:
            model = hub_load_dino(model_name, pretrained=True)
        model.to(device)
        _run_dummy(model, device)
        _warmed.add(model_name)
        elapsed = time.monotonic() - t0
        log.info("  [ok] DINO cached  %s  device=%s  (%.1fs)", label, device, elapsed)
    finally:
        _hub.download_url_to_file = _orig

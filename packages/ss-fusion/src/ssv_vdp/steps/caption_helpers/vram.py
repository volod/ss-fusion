"""VRAM management: offload/restore CLIP+DINO, guard, telemetry."""

import time
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from .ollama import _unload_known_sidecars

_log = get_logger("pipeline.local.caption")

_RUNTIME_TELEMETRY: dict[str, float] = {
    "vram_wait_time_sec": 0.0,
    "restore_failures": 0.0,
}


def reset_runtime_telemetry() -> None:
    _RUNTIME_TELEMETRY["vram_wait_time_sec"] = 0.0
    _RUNTIME_TELEMETRY["restore_failures"] = 0.0


def get_runtime_telemetry() -> dict[str, float]:
    return {
        "vram_wait_time_sec": float(_RUNTIME_TELEMETRY.get("vram_wait_time_sec", 0.0) or 0.0),
        "restore_failures": float(_RUNTIME_TELEMETRY.get("restore_failures", 0.0) or 0.0),
    }


def _log_vram_snapshot(label: str) -> None:
    """Best-effort VRAM snapshot. Uses torch.cuda.mem_get_info for per-process accuracy."""
    try:
        import torch

        from selfsuvis.pipeline.vision.registry import (  # noqa: PLC0415
            detect_ram_gb,
            detect_vram_gb,
        )

        total = detect_vram_gb()
        ram = detect_ram_gb()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free = free_bytes / (1024**3)
        else:
            from selfsuvis.pipeline.vision.registry import detect_free_vram_gb  # noqa: PLC0415

            free = detect_free_vram_gb()
        used = max(0.0, total - free)
        _log.info(
            "  [VRAM] %s | total=%.1f GiB free=%.1f GiB used~=%.1f GiB ram=%.1f GiB",
            label,
            total,
            free,
            used,
            ram,
        )
        return
    except Exception as exc:
        _log.debug("  [VRAM] %s | resource snapshot failed: %s", label, exc)


def _detect_free_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            return free_bytes / (1024**3)
        from selfsuvis.pipeline.vision.registry import detect_free_vram_gb  # noqa: PLC0415

        return detect_free_vram_gb()
    except Exception:
        return 0.0


def _flush_cuda_allocator() -> None:
    import gc

    gc.collect()
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            try:
                _torch.cuda.synchronize()
            except Exception:
                pass
            _torch.cuda.empty_cache()
    except Exception:
        pass


def _backbone_device(backbone: Any) -> str | None:
    try:
        return str(next(backbone.parameters()).device)
    except Exception:
        return None


def _device_matches(actual: Any, expected: Any) -> bool:
    """Treat 'cuda' and 'cuda:0' as equivalent for single-GPU residency checks."""
    import torch as _torch

    actual_dev = _torch.device(actual)
    expected_dev = _torch.device(expected)
    if actual_dev.type != expected_dev.type:
        return False
    if actual_dev.type != "cuda":
        return actual_dev == expected_dev
    if expected_dev.index is None:
        return True
    return actual_dev.index == expected_dev.index


def _wait_for_sidecar_vram_release(
    *,
    baseline_free_gb: float,
    unload_count: int,
    label: str,
    timeout_sec: float = 20.0,
    min_expected_gain_gb: float = 1.0,
    sufficient_free_gb: float | None = None,
) -> None:
    """Wait briefly for an unload request to turn into free VRAM."""
    if unload_count <= 0:
        return
    if sufficient_free_gb is not None and baseline_free_gb >= sufficient_free_gb:
        _log.info(
            "  VRAM already sufficient for %s: %.1f GiB free (target %.1f GiB)",
            label,
            baseline_free_gb,
            sufficient_free_gb,
        )
        return

    t_wait = time.time()
    deadline = time.time() + timeout_sec
    best_free_gb = baseline_free_gb
    target_free_gb = baseline_free_gb + min_expected_gain_gb
    while time.time() < deadline:
        _flush_cuda_allocator()
        free_gb = _detect_free_vram_gb()
        best_free_gb = max(best_free_gb, free_gb)
        if sufficient_free_gb is not None and free_gb >= sufficient_free_gb:
            _log.info(
                "  VRAM sufficient for %s after unload: %.1f GiB free",
                label,
                free_gb,
            )
            _RUNTIME_TELEMETRY["vram_wait_time_sec"] += max(0.0, time.time() - t_wait)
            return
        if free_gb >= target_free_gb:
            _log.info(
                "  VRAM recovered after %s: free %.1f → %.1f GiB",
                label,
                baseline_free_gb,
                free_gb,
            )
            _RUNTIME_TELEMETRY["vram_wait_time_sec"] += max(0.0, time.time() - t_wait)
            return
        time.sleep(0.5)

    _RUNTIME_TELEMETRY["vram_wait_time_sec"] += max(0.0, time.time() - t_wait)
    if best_free_gb > baseline_free_gb:
        _log.info(
            "  VRAM partially recovered after %s: free %.1f → %.1f GiB",
            label,
            baseline_free_gb,
            best_free_gb,
        )
    else:
        _log.warning(
            "  VRAM did not recover after %s within %.0fs; a sidecar may still be resident",
            label,
            timeout_sec,
        )


def _guard_min_free_vram(stage: str, min_free_gb: float | None = None) -> float:
    """Fail fast when a CUDA stage starts without enough free VRAM."""
    try:
        from selfsuvis.pipeline.vision.registry import detect_resources  # noqa: PLC0415

        resources = detect_resources()
    except Exception as exc:
        raise RuntimeError(f"{stage}: could not read VRAM availability: {exc}") from exc

    total_gb = float(resources.get("vram_gb", 0.0) or 0.0)
    free_gb = float(resources.get("free_vram_gb", 0.0) or 0.0)
    if total_gb <= 0.0:
        return free_gb

    required_gb = (
        min_free_gb
        if min_free_gb is not None
        else float(getattr(settings, "LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB", 6.0) or 6.0)
    )
    required_gb = min(total_gb, max(required_gb, total_gb * 0.35))
    if free_gb < required_gb:
        raise RuntimeError(
            f"{stage}: refusing to start CUDA stage with only {free_gb:.1f} GiB free "
            f"(required >= {required_gb:.1f} GiB, total {total_gb:.1f} GiB). "
            "A sidecar model may still be resident in VRAM; unload Ollama/vLLM models and retry."
        )
    return free_gb


def _offload_models_to_cpu(models: dict[str, Any]) -> None:
    """Move CLIP and DINO backbones to CPU and flush the CUDA allocator cache.

    Called before loading a large model (Florence-2, ASR) when VRAM is tight.
    The embedders keep their ``self.device`` attribute unchanged so they work
    correctly once the backbone is moved back by :func:`_restore_models_to_gpu`.
    """
    _log_vram_snapshot("before offload CLIP+DINO to CPU")
    moved = 0
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                current = _backbone_device(backbone)
                if current is not None and current.startswith("cpu"):
                    continue
                backbone.cpu()
                moved += 1
            except Exception:
                pass
    try:
        from selfsuvis.models.dino_model import _set_dino_xformers_enabled

        _set_dino_xformers_enabled(False)
    except Exception:
        pass
    _flush_cuda_allocator()
    try:
        import torch as _torch

        free_mb = _torch.cuda.mem_get_info(0)[0] / 1024**2 if _torch.cuda.is_available() else 0
    except Exception:
        free_mb = 0
    if moved > 0:
        _log.info("  CLIP+DINO offloaded to CPU — %.0f MiB free on GPU", free_mb)
    else:
        _log.debug("  CLIP+DINO already on CPU — %.0f MiB free on GPU", free_mb)
    _log_vram_snapshot("after offload CLIP+DINO to CPU")


def _prep_vram_for_step(
    models: dict[str, Any],
    device: str,
    ollama_url: str = "",
    ollama_model: str = "",
    extra_sidecars: list[tuple[str, str]] | None = None,
    label: str = "next step",
    required_free_gb: float | None = None,
) -> None:
    """Offload CLIP+DINOv3, evict any Ollama resident, and flush the CUDA allocator.

    Call this before loading any local inference model (OCR, depth, detection,
    world model) to maximise available VRAM and avoid OOM on 16 GiB class GPUs.

    Ollama HTTP eviction calls are skipped when VRAM is already above the target
    threshold — saves ~200 ms of HTTP round-trips per step on uncongested GPUs.
    """
    if device != "cuda":
        return
    _log_vram_snapshot("before prep VRAM for next step")
    required = required_free_gb or float(
        getattr(settings, "LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB", 6.0) or 6.0
    )

    # Offload CLIP+DINO to CPU first — fast and always worth doing.
    _offload_models_to_cpu(models)
    baseline_free_gb = _detect_free_vram_gb()

    if baseline_free_gb >= required:
        # VRAM already sufficient — skip the Ollama HTTP eviction calls.
        _flush_cuda_allocator()
        try:
            import torch as _torch

            free_mb = _torch.cuda.mem_get_info(0)[0] / 1024**2 if _torch.cuda.is_available() else 0
        except Exception:
            free_mb = 0
        _log.info("  VRAM cleared for next step — %.0f MiB free", free_mb)
        _log_vram_snapshot("after prep VRAM for next step")
        return

    # VRAM below threshold — evict Ollama sidecars and wait for release.
    unload_count = _unload_known_sidecars(
        [
            (ollama_url, ollama_model),
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
            (getattr(settings, "REASONING_API_URL", ""), getattr(settings, "REASONING_MODEL", "")),
            *(extra_sidecars or []),
        ]
    )
    _wait_for_sidecar_vram_release(
        baseline_free_gb=baseline_free_gb,
        unload_count=unload_count,
        label=label,
        sufficient_free_gb=required,
    )
    _flush_cuda_allocator()
    try:
        import torch as _torch

        free_mb = _torch.cuda.mem_get_info(0)[0] / 1024**2 if _torch.cuda.is_available() else 0
    except Exception:
        free_mb = 0
    _log.info("  VRAM cleared for next step — %.0f MiB free", free_mb)
    _log_vram_snapshot("after prep VRAM for next step")


def _restore_models_to_gpu(models: dict[str, Any], device: str) -> bool:
    """Move CLIP and DINO backbones back to *device* after a large model releases."""
    _log_vram_snapshot(f"before restore models to {device}")
    import os as _os

    import torch as _torch

    if str(device).startswith("cuda"):
        free_gb = _detect_free_vram_gb()
        if free_gb > 0.0 and free_gb < 2.5:
            _log.warning(
                "  Skipping CLIP+DINO restore to %s: only %.1f GiB free VRAM",
                device,
                free_gb,
            )
            return False
    # Expandable segments let the allocator grow existing blocks rather than
    # searching for a new contiguous region — eliminates most fragmentation OOMs
    # when moving models on/off GPU between pipeline steps.
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Free any GPU memory held by objects that were just released before trying to
    # restore the backbones — prevents partial moves caused by transient OOM.
    _flush_cuda_allocator()
    expected = _torch.device(device)
    restored_all = True
    moved = 0
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                current = _backbone_device(backbone)
                if current == str(expected):
                    continue
                backbone.to(device)
                moved += 1
            except RuntimeError as exc:
                # OOM mid-.to() leaves the model in a mixed-device state.
                # Roll back to CPU first (releases all partially-moved params),
                # flush the allocator, then retry once — transient fragmentation
                # clears after the rollback frees the contiguous blocks it needs.
                try:
                    backbone.cpu()
                except Exception:
                    pass
                _flush_cuda_allocator()
                try:
                    backbone.to(device)
                    moved += 1
                    _log.debug("  %s backbone moved to %s (retry succeeded)", key, device)
                except RuntimeError:
                    _log.warning(
                        "  Could not move %s backbone to %s (%s) — staying on CPU",
                        key,
                        device,
                        exc,
                    )
            try:
                actual = next(backbone.parameters()).device
            except StopIteration:
                actual = expected
            if not _device_matches(actual, expected):
                _log.warning(
                    "  %s backbone residency mismatch after restore: actual=%s expected=%s",
                    key,
                    actual,
                    expected,
                )
                restored_all = False
    try:
        from selfsuvis.models.dino_model import _set_dino_xformers_enabled

        _set_dino_xformers_enabled(str(device).startswith("cuda") and restored_all)
    except Exception:
        pass
    if restored_all and moved > 0:
        _log.info("  CLIP+DINO restored to %s", device)
    elif restored_all:
        _log.debug("  CLIP+DINO already resident on %s", device)
    else:
        _log.warning(
            "  CLIP+DINO not fully restored to %s; continuing with CPU fallback where needed",
            device,
        )
        _RUNTIME_TELEMETRY["restore_failures"] += 1.0
    _log_vram_snapshot(f"after restore models to {device}")
    return restored_all


def _models_on_device(models: dict[str, Any], device: str) -> bool:
    import torch as _torch

    expected = _torch.device(device)
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is None:
            continue
        try:
            actual = next(backbone.parameters()).device
        except StopIteration:
            continue
        if not _device_matches(actual, expected):
            return False
    return True

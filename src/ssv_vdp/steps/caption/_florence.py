"""Steps 04 and 04b — Florence-2 scene captioning and Gemma segment-boundary diffs."""

import time
from pathlib import Path
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..caption_helpers.frame_selection import (
    _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT,
    _select_segment_boundary_pairs,
)
from ..caption_helpers.gemma_api import _gemma_diff_two_frames_via_api
from ..caption_helpers.ollama import _unload_ollama_model
from ..caption_helpers.vlm_api import caption_via_florence_api, caption_via_qwen_api
from ..caption_helpers.vram import _log_vram_snapshot, _offload_models_to_cpu

_log = get_logger("pipeline.local.caption")


def step_scene_captioning(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    models: dict[str, Any] | None = None,
    qwen_api_url: str = "",
    qwen_model: str = "",
    florence_api_url: str = "",
    florence_model: str = "",
    domain_hint: str = "",
) -> dict[str, Any]:
    """Step 04: Florence-2 scene captioning with memory management and API support.

    Memory strategy (CUDA only):
      1. If ``florence_api_url`` is set: call Florence-2 via vLLM API — no local
         weights loaded, zero VRAM consumed.  Use this when another process
         (e.g. Ollama) already occupies most of VRAM.
      2. Otherwise load Florence-2 locally:
         a. Offload CLIP+DINO to CPU to free ~1.7 GiB.
         b. If ``qwen_api_url`` looks like Ollama (port 11434): send keep_alive=0
            to evict the VLM (~11-12 GiB freed), giving Florence plenty of room.
            Ollama auto-reloads on the next request (step 12).
         c. If Florence still OOMs and ``qwen_api_url`` + ``qwen_model`` are set:
            fall back to Qwen API captioning.
    """
    from ..report import write_scene_captions_md

    effective_florence_api_url = florence_api_url or settings.FLORENCE_API_URL
    effective_florence_model = florence_model or settings.FLORENCE_MODEL
    if effective_florence_api_url:
        _log.info("  Florence-2 via vLLM API at %s", effective_florence_api_url)
        _log_vram_snapshot("before Florence API captioning")
        if models and device == "cuda":
            _offload_models_to_cpu(models)
        result = caption_via_florence_api(
            frame_list,
            video_name,
            video_dir,
            effective_florence_api_url,
            effective_florence_model,
            domain_hint=domain_hint,
        )
        _log_vram_snapshot("after Florence API captioning")
        return result

    out_md = video_dir / "scene_captions.md"
    try:
        from selfsuvis.pipeline.vision.florence import FlorenceModel
    except ImportError as exc:
        _log.warning("  Florence-2 unavailable (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    if models and device == "cuda":
        _offload_models_to_cpu(models)

    if device == "cuda":
        if qwen_api_url and qwen_model:
            _unload_ollama_model(qwen_api_url, qwen_model)
        _gemma_url_cap = settings.GEMMA_API_URL
        _gemma_model_cap = settings.GEMMA_API_MODEL
        if _gemma_url_cap and _gemma_model_cap and _gemma_model_cap != qwen_model:
            _unload_ollama_model(_gemma_url_cap, _gemma_model_cap)

    _log.info("Loading Florence-2-large on %s …", device)
    _log_vram_snapshot("before local Florence load")
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        if qwen_api_url and qwen_model:
            _log.warning("  Florence-2 load failed (%s) — using Qwen API fallback", exc)
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _log.info("  CUDA cache cleared before Qwen fallback")
            except Exception:
                pass
            return caption_via_qwen_api(
                frame_list, video_name, video_dir, qwen_api_url, qwen_model, domain_hint=domain_hint
            )
        _log.warning(
            "  Florence-2 load failed (%s) — skipping captioning "
            "(pass --qwen-api-url + --qwen to enable Qwen API fallback)",
            exc,
        )
        return {"skipped": True, "reason": str(exc), "captions": []}

    _log.info("  [ok] Florence-2-large loaded in %.1fs", time.time() - t0)
    _log.info("  Captioning %d frames …", len(frame_list))
    caption_results: list[dict[str, Any]] = []
    florence_runtime_mode = florence.runtime_mode
    florence_model_tag = florence.model_tag
    batch_size = settings.FLORENCE_BATCH_SIZE
    _florence_oom = False
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        if _florence_oom:
            captions_and_confs: list[tuple[str, float]] = [("", 0.5)] * len(batch)
        else:
            pil_images = []
            for fp, _t in batch:
                try:
                    pil_images.append(Image.open(fp).convert("RGB"))
                except Exception:
                    pil_images.append(Image.new("RGB", (224, 224)))
            try:
                captions_and_confs = florence.caption_batch(pil_images)
                florence_runtime_mode = florence.runtime_mode
            except Exception as exc:
                from selfsuvis.pipeline.core.gpu_utils import is_cuda_oom, log_oom_banner

                if is_cuda_oom(exc):
                    remaining = len(frame_list) - batch_start
                    log_oom_banner(
                        _log,
                        "Florence-2 caption_batch",
                        f"batch_start={batch_start}, releasing model, "
                        f"{remaining} frames will get empty captions",
                    )
                    try:
                        import torch as _t

                        _t.cuda.empty_cache()
                        florence.release()
                    except Exception:
                        pass
                    _florence_oom = True
                else:
                    _log.warning("  Florence batch %d failed: %s", batch_start, exc, exc_info=True)
                captions_and_confs = [("", 0.5)] * len(batch)
        for (fp, t_sec), (cap, conf) in zip(batch, captions_and_confs):
            caption_results.append(
                {"frame_path": fp, "t_sec": t_sec, "caption": cap, "caption_confidence": conf}
            )

    elapsed = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  [ok] %d/%d frames captioned in %.1fs", captioned, len(frame_list), elapsed)
    _log_vram_snapshot("after local Florence captioning")
    write_scene_captions_md(
        out_md,
        video_name,
        caption_results,
        elapsed,
        model_tag=florence_model_tag,
        runtime_mode=florence_runtime_mode,
    )
    florence.release()

    return {
        "skipped": False,
        "captions": caption_results,
        "captioned_count": captioned,
        "elapsed_sec": elapsed,
        "florence_runtime_mode": florence_runtime_mode,
        "florence_model_tag": florence_model_tag,
    }


def step_gemma_segment_captions(
    frame_list: list[tuple[str, float]],
    caption_results: list[dict[str, Any]],
    video_name: str,
    video_dir: Path,
    gemma_api_url: str = "",
    gemma_api_model: str = "",
    gemma_api_backend: str = "",
) -> dict[str, Any]:
    """Step 4b: Gemma 4 multi-frame segment-boundary diff analysis.

    Uses _analyze_caption_sequence to find scene boundaries from Florence captions,
    then for each boundary pair calls the Gemma sidecar with both frames and a diff
    prompt ("What changed between these two frames?").

    Writes ``gemma_segment_captions.md`` to *video_dir*.
    Skips gracefully when no sidecar is configured or no captions are available.
    """
    from ..common import _analyze_caption_sequence
    from ..report import write_gemma_segment_captions_md

    result: dict[str, Any] = {"skipped": True, "reason": "", "boundary_diffs": []}

    effective_api_url = gemma_api_url or settings.GEMMA_API_URL
    effective_api_model = gemma_api_model or settings.GEMMA_API_MODEL
    settings_backend = getattr(settings, "GEMMA_API_BACKEND", "")
    effective_backend = gemma_api_backend or (
        settings_backend if isinstance(settings_backend, str) else ""
    )
    effective_timeout = float(settings.GEMMA_API_TIMEOUT_SEC)

    if not effective_api_url:
        result["reason"] = "GEMMA_API_URL not configured"
        _log.info("  Gemma segment captions skipped: %s", result["reason"])
        return result

    if not caption_results:
        result["reason"] = "no caption results available"
        _log.info("  Gemma segment captions skipped: %s", result["reason"])
        return result

    ts_to_fp: dict[float, str] = {t: fp for fp, t in frame_list}
    enriched = _analyze_caption_sequence(caption_results)

    max_boundaries = int(
        getattr(settings, "GEMMA_SEGMENT_DIFF_MAX_BOUNDARIES", _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT)
        or 0
    )
    boundary_pairs = _select_segment_boundary_pairs(enriched, max_boundaries=max_boundaries)

    if not boundary_pairs:
        result["reason"] = "no segment boundaries found"
        _log.info("  Gemma segment captions: no segment boundaries (all frames same segment)")
        return result

    total_boundaries = sum(1 for row in enriched if row.get("is_new_segment")) - 1
    if max_boundaries > 0 and total_boundaries > len(boundary_pairs):
        _log.info(
            "  Gemma segment boundary ranking: %d candidates → top %d strongest diffs",
            total_boundaries,
            len(boundary_pairs),
        )

    _log.info(
        "Gemma segment-boundary diff: %d boundaries  model=%s  backend=%s  url=%s ...",
        len(boundary_pairs),
        effective_api_model,
        effective_backend or "auto",
        effective_api_url,
    )
    t0 = time.time()

    boundary_diffs: list[dict[str, Any]] = []
    for idx, (prev_row, next_row) in enumerate(boundary_pairs):
        fp_before = ts_to_fp.get(prev_row.get("t_sec", -1.0), "") or prev_row.get("frame_path", "")
        fp_after = ts_to_fp.get(next_row.get("t_sec", -1.0), "") or next_row.get("frame_path", "")
        if not fp_before or not fp_after:
            _log.debug("  Gemma diff: missing frame paths at boundary %d — skipping", idx)
            continue

        desc = _gemma_diff_two_frames_via_api(
            fp_before,
            fp_after,
            effective_api_url,
            effective_api_model,
            effective_timeout,
            backend=effective_backend,
        )
        entry = {
            "boundary_idx": idx,
            "prev_t_sec": prev_row.get("t_sec", 0.0),
            "next_t_sec": next_row.get("t_sec", 0.0),
            "prev_segment_id": prev_row.get("segment_id", 0),
            "next_segment_id": next_row.get("segment_id", 0),
            "fp_before": fp_before,
            "fp_after": fp_after,
            "diff_description": desc,
        }
        boundary_diffs.append(entry)

    elapsed = time.time() - t0
    described = sum(1 for b in boundary_diffs if b.get("diff_description"))
    _log.info(
        "  [ok] Gemma segment diffs: %d/%d boundaries described in %.1fs",
        described,
        len(boundary_pairs),
        elapsed,
    )

    write_gemma_segment_captions_md(
        video_dir / "gemma_segment_captions.md", video_name, effective_api_model, boundary_diffs
    )

    result.update(
        {
            "skipped": False,
            "boundary_count": len(boundary_pairs),
            "described_count": described,
            "elapsed_sec": elapsed,
            "model": effective_api_model,
            "backend": effective_backend,
            "boundary_diffs": boundary_diffs,
        }
    )
    return result

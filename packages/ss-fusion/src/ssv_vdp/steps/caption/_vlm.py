"""Steps 12-13 — Qwen VLM detailed captioning and UniDriveVLA expert analysis."""

import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..caption_helpers.frame_selection import _adaptive_sparse_budget, _select_qwen_frames
from ..caption_helpers.gemma_api import step_qwen_captioning_gemma_fallback
from ..caption_helpers.vram import _log_vram_snapshot
from ..common import VideoKnowledge, _open_frame_image, _run_batched_frame_inference

_log = get_logger("pipeline.local.caption")


def step_qwen_captioning(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: dict[float, str],
    ocr_results: list[dict[str, Any]],
    clip_prescreen_fn=None,
    knowledge: Optional["VideoKnowledge"] = None,
) -> dict[str, Any]:
    """Step 12: Qwen VLM detailed scene captioning with full agentic context.

    When *knowledge* is provided, each frame's prompt is enriched with all
    prior observations: Florence caption, depth profile, detected objects,
    scene segment, ASR, OCR, and the previous frame's Qwen structured output.
    This lets Qwen reason about *what changed* rather than describing each
    frame in isolation.
    """
    from ..report import write_detailed_captions_md

    out_md = video_dir / "detailed_captions.md"
    result: dict[str, Any] = {"skipped": True, "results": []}
    try:
        from selfsuvis.pipeline.vision.qwen import QwenModel
    except ImportError as exc:
        _log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result
    qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)
    _log_vram_snapshot("before Qwen sidecar use")
    if not qwen.is_enabled():
        _log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        _log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        gemma_url = settings.GEMMA_API_URL
        gemma_model = settings.GEMMA_API_MODEL
        if gemma_url:
            return step_qwen_captioning_gemma_fallback(
                frame_list,
                video_name,
                video_dir,
                gemma_url,
                gemma_model,
            )
        return result
    ocr_map: dict[float, str] = {
        r["t_sec"]: r["ocr_text"]
        for r in ocr_results
        if r.get("t_sec") is not None and r.get("ocr_text")
    }

    domain = knowledge.domain_hint() if knowledge else ""
    if domain:
        _log.info("  Qwen domain hint: %s", domain)
    qwen_budget = _adaptive_sparse_budget(
        frame_list,
        configured_max=max(1, int(settings.QWEN_MAX_FRAMES)),
        seconds_per_sample=0.9,
        floor=8,
    )
    sampled_frame_list = _select_qwen_frames(
        frame_list,
        max_frames=qwen_budget,
        knowledge=knowledge,
        ocr_map=ocr_map,
    )
    if len(sampled_frame_list) < len(frame_list):
        _log.info(
            "  Qwen frame selection: %d/%d frames chosen for detailed captioning",
            len(sampled_frame_list),
            len(frame_list),
        )
    t0 = time.time()

    _use_agentic = knowledge is not None
    if _use_agentic and sampled_frame_list:
        _probe_fp, _probe_t = sampled_frame_list[0]
        _probe_img = _open_frame_image(_probe_fp)
        if _probe_img is not None:
            _probe_res = qwen.extract_batch(
                [_probe_img],
                subtitle_texts=[subtitle_map.get(_probe_t) or None],
                ocr_texts=[ocr_map.get(_probe_t) or None],
                extra_contexts=[knowledge.context_for_frame(_probe_t)],
                domain_hint=domain or None,
            )
            if _probe_res and _probe_res[0].get("parse_error"):
                _log.warning(
                    "  Qwen agentic probe: parse error on first frame -- "
                    "falling back to non-agentic mode. "
                    "Model '%s' appears too small for structured JSON output; "
                    "use qwen2.5vl:32b or larger to keep agentic mode.",
                    settings.QWEN_MODEL,
                )
                _use_agentic = False

    _log.info(
        "Running Qwen detailed captioning on %d sampled frames (from %d total, model=%s  agentic=%s) ...",
        len(sampled_frame_list),
        len(frame_list),
        settings.QWEN_MODEL,
        "yes" if _use_agentic else "no",
    )

    caption_results: list[dict[str, Any]] = []

    def _batch_fn(batch: list[tuple[str, float]], imgs: list) -> list[dict[str, Any]]:
        extra_contexts = None
        if _use_agentic and knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        results = qwen.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )
        if _use_agentic and knowledge:
            for r in results:
                knowledge.update_qwen_state(r)
        return results

    batch_results = _run_batched_frame_inference(
        sampled_frame_list,
        batch_size=4,
        batch_fn=_batch_fn,
        warning_label="Qwen",
        error_result={"service_unavailable": True},
    )
    for r in batch_results:
        t_sec = r.get("t_sec", 0.0)
        caption_results.append({**r, "subtitle_text": subtitle_map.get(t_sec) or ""})
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in caption_results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    )
    parse_errors = sum(1 for r in caption_results if r.get("parse_error"))
    subtitle_used = sum(1 for r in caption_results if r.get("subtitle_text"))
    _log.info(
        "  [ok] Qwen: %d/%d sampled frames captioned in %.1fs (%d with ASR  parse_errors=%d  agentic=%s)",
        ok,
        len(sampled_frame_list),
        elapsed,
        subtitle_used,
        parse_errors,
        "yes" if _use_agentic else "no",
    )
    _log_vram_snapshot("after Qwen sidecar use")
    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, settings.QWEN_MODEL)
    result.update(
        {
            "skipped": False,
            "results": caption_results,
            "ok_count": ok,
            "subtitle_used": subtitle_used,
            "elapsed_sec": elapsed,
            "sampled_count": len(sampled_frame_list),
            "total_frames": len(frame_list),
            "parse_error_count": parse_errors,
        }
    )
    return result


def step_unidrive_analysis(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: dict[float, str],
    ocr_results: list[dict[str, Any]],
    knowledge: Optional["VideoKnowledge"] = None,
) -> dict[str, Any]:
    """Step 13: UniDriveVLA expert analysis on a sparse frame sample."""
    from ..report import write_unidrive_analysis_md

    out_md = video_dir / "unidrive_analysis.md"
    result: dict[str, Any] = {"skipped": True, "results": []}
    try:
        from selfsuvis.pipeline.vision.unidrive import UniDriveVLAModel
    except ImportError as exc:
        _log.warning("  UniDriveVLA client unavailable (%s) — skipping", exc)
        return result

    client = UniDriveVLAModel()
    _log_vram_snapshot("before UniDrive sidecar use")
    if not client.is_enabled():
        _log.info("  UniDriveVLA disabled (no sidecar URL and no usable local HF model) — skipping")
        _log.info("  To enable sidecar mode: --unidrive-api-url http://localhost:8030/v1")
        _log.info(
            "  To enable local mode: cache HF weights with scripts/prepare_models.py --unidrive --unidrive-backend vllm"
        )
        return result

    max_frames = _adaptive_sparse_budget(
        frame_list,
        configured_max=max(1, int(getattr(settings, "UNIDRIVE_MAX_FRAMES", 24) or 24)),
        seconds_per_sample=1.4,
        floor=6,
    )
    sample_step = max(1, len(frame_list) // max_frames)
    sampled_frames = frame_list[::sample_step][:max_frames]
    ocr_map: dict[float, str] = {
        r["t_sec"]: r["ocr_text"]
        for r in ocr_results
        if r.get("t_sec") is not None and r.get("ocr_text")
    }
    domain = knowledge.domain_hint() if knowledge else ""
    _log.info(
        "Running UniDriveVLA expert analysis on %d sampled frames (model=%s backend=%s) …",
        len(sampled_frames),
        settings.UNIDRIVE_MODEL,
        getattr(settings, "UNIDRIVE_BACKEND", "vllm"),
    )
    t0 = time.time()

    def _batch_fn(batch: list[tuple[str, float]], imgs: list[Image.Image]) -> list[dict[str, Any]]:
        extra_contexts = None
        if knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        return client.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )

    batch_results = _run_batched_frame_inference(
        sampled_frames,
        batch_size=2,
        batch_fn=_batch_fn,
        warning_label="UniDriveVLA",
        error_result={"service_unavailable": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1 for r in batch_results if not r.get("service_unavailable") and not r.get("parse_error")
    )
    _log.info(
        "  [ok] UniDriveVLA: %d/%d sampled frames analysed in %.1fs",
        ok,
        len(batch_results),
        elapsed,
    )
    if ok == 0 and batch_results:
        first_reason = batch_results[0].get("reason", "unknown")
        _log.warning(
            "  UniDriveVLA: all %d frames failed (reason: %s). "
            "Set --unidrive-api-url to point at an Ollama/vLLM endpoint.",
            len(batch_results),
            first_reason,
        )
    _log_vram_snapshot("after UniDrive sidecar use")
    write_unidrive_analysis_md(out_md, video_name, batch_results, elapsed, settings.UNIDRIVE_MODEL)
    client.release()
    result.update(
        {
            "skipped": False,
            "results": batch_results,
            "ok_count": ok,
            "elapsed_sec": elapsed,
            "sampled_frames": len(batch_results),
        }
    )
    return result

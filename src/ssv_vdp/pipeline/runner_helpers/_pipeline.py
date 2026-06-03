"""Per-video pipeline orchestrator (monolithic path).

Contains run_video_pipeline() and its safe wrapper.
Activate the LangGraph path instead via SELFSUVIS_USE_GRAPH=1.

The orchestrator delegates each phase to a dedicated module:
  _pipeline_phase1  — Steps 01-02: frame extraction and vector-store indexing
  _pipeline_phase2  — Steps 03-20: multimodal analysis, 3D map, state fusion, threat
  _pipeline_phase3  — Steps 21-26: SSL adaptation, distillation, ONNX export, search
  _pipeline_phase4  — Steps 27-34: synthesis, agentic audit, drone/audio, final stats
"""

import os
from pathlib import Path
from typing import Any

# Re-exported so runner_helpers/__init__.py can still import them by name.
_TOTAL_STEPS = 35
_SSL_GATE_MAX_LOSS = 10.0

from selfsuvis.pipeline.core import resolve_device, settings
from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import (
    VideoKnowledge,
    _banner,
)
from ._pipeline_phase1 import run_phase1
from ._pipeline_phase2 import run_phase2
from ._pipeline_phase3 import run_phase3
from ._pipeline_phase4 import run_phase4

_log = get_logger(__name__)


def run_video_pipeline(
    args: Any,
    video_path: Path,
    output_dir: Path,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    _out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all pipeline steps for a single video. Returns per-video stats dict.

    When ``SELFSUVIS_USE_GRAPH=1`` is set this function delegates to the
    LangGraph-based orchestrator in ``runner_graph.py`` and returns its result
    directly.  All existing callers remain unaffected.

    *_out* is an optional external dict that is used as the stats container.
    When provided, callers can inspect it for partial results if an exception
    escapes — the timings and frame counts recorded up to the failure point
    are preserved.
    """
    if os.getenv("SELFSUVIS_USE_GRAPH", "").lower() in ("1", "true", "yes"):
        from ..graph import run_graph_pipeline

        return run_graph_pipeline(args, video_path, output_dir, models, store, is_qdrant, device)

    from ...steps.caption import reset_runtime_telemetry

    video_name = video_path.stem
    video_id = video_name.replace(" ", "_").lower()
    video_dir = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    reset_runtime_telemetry()

    _banner(f"Processing video: {video_path.name}")
    _log.info("Output directory: %s", video_dir)

    # Use the shared container when provided so partial state is visible outside.
    if _out is None:
        _out = {}
    _out.update({"name": video_name, "video_path": str(video_path), "timings": {}})
    stats: dict[str, Any] = _out
    T = stats["timings"]

    # Accumulated context passed through the pipeline; enriches synthesis at step 22.
    video_context: dict[str, Any] = {"video_name": video_name}
    agentic_trace: list[dict[str, Any]] = []
    video_context["agentic_trace"] = agentic_trace

    knowledge = VideoKnowledge(
        video_name=video_name,
        duration_sec=0.0,
        frame_count=0,
    )

    # ── Phase 1: ingestion ───────────────────────────────────────────────────
    frame_list, clip_dino_on_gpu = run_phase1(
        args=args,
        video_path=video_path,
        video_dir=video_dir,
        video_name=video_name,
        video_id=video_id,
        models=models,
        store=store,
        is_qdrant=is_qdrant,
        device=device,
        stats=stats,
        T=T,
        video_context=video_context,
        agentic_trace=agentic_trace,
        knowledge=knowledge,
    )

    if not frame_list:
        _log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # Re-initialize knowledge with correct frame count / duration after phase 1.
    knowledge = VideoKnowledge(
        video_name=video_name,
        duration_sec=stats["duration_sec"],
        frame_count=stats["frames"],
    )

    # ── Phase 2: multimodal analysis ─────────────────────────────────────────
    p2 = run_phase2(
        args=args,
        video_path=video_path,
        video_dir=video_dir,
        video_name=video_name,
        video_id=video_id,
        models=models,
        store=store,
        is_qdrant=is_qdrant,
        device=device,
        frame_list=frame_list,
        clip_dino_on_gpu=clip_dino_on_gpu,
        stats=stats,
        T=T,
        video_context=video_context,
        agentic_trace=agentic_trace,
        knowledge=knowledge,
    )

    # ── Phase 3: SSL-gated adaptation ────────────────────────────────────────
    p3 = run_phase3(
        args=args,
        video_path=video_path,
        video_dir=video_dir,
        video_name=video_name,
        video_id=video_id,
        models=models,
        store=store,
        is_qdrant=is_qdrant,
        device=device,
        frame_list=frame_list,
        clip_dino_on_gpu=p2["clip_dino_on_gpu"],
        caption_results=p2["caption_results"],
        base_results=p2["base_results"],
        query_frame=p2["query_frame"],
        query_t_sec=p2["query_t_sec"],
        gemma_tracking_result=p2["gemma_tracking_result"],
        depth_result=p2["depth_result"],
        platform_fusion_result=p2["platform_fusion_result"],
        full_fusion_result=p2["full_fusion_result"],
        physical_state_result=p2["physical_state_result"],
        stats=stats,
        T=T,
        video_context=video_context,
        agentic_trace=agentic_trace,
        knowledge=knowledge,
    )

    # ── Phase 4: finalization ────────────────────────────────────────────────
    run_phase4(
        args=args,
        video_path=video_path,
        video_dir=video_dir,
        video_name=video_name,
        output_dir=output_dir,
        device=device,
        frame_list=frame_list,
        clip_dino_on_gpu=p3["clip_dino_on_gpu"],
        j=p2["j"],
        qwen_result=p2["qwen_result"],
        unidrive_result=p2["unidrive_result"],
        threat_primitives_result=p2["threat_primitives_result"],
        physical_state_result=p2["physical_state_result"],
        base_results=p2["base_results"],
        ssl_gate_passed=p3["ssl_gate_passed"],
        ft_results=p3["ft_results"],
        stats=stats,
        T=T,
        video_context=video_context,
        agentic_trace=agentic_trace,
    )

    _banner(f"[ok] Video complete: {video_path.name}")
    _log.info("  Output dir: %s", video_dir)
    return stats


def _run_video_pipeline_safe(
    args: Any,
    video_path: "Path",
    output_dir: "Path",
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> dict[str, Any]:
    """Wrapper around :func:`run_video_pipeline` that always returns a stats dict.

    On exception, returns the partial stats dict with timings recorded up to
    the failure point so step times and frame counts are not lost.
    """
    _out: dict[str, Any] = {}
    try:
        result = run_video_pipeline(
            args, video_path, output_dir, models, store, is_qdrant, device, _out=_out
        )
        # The graph-pipeline path returns a new dict without mutating _out; merge it back.
        if result and result is not _out:
            _out.update(result)
    except Exception as exc:
        _log.error("Pipeline failed for %s: %s", video_path.name, exc, exc_info=True)
        _out.setdefault("name", video_path.stem)
        _out.setdefault("video_dir", str(output_dir / video_path.stem))
        _out["error"] = str(exc)
        _out.setdefault("timings", {})
        _out.setdefault("frames", 0)
        _out.setdefault("duration_sec", 0.0)
        timings = _out.get("timings", {})
        _out.setdefault("pipeline_sec", sum(timings.values()))
    _out.setdefault("name", video_path.stem)
    return _out


# -- Main entry point ----------------------------------------------------------

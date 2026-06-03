"""Phase 2 map nodes: submit 3D map to background thread, join result (step 16)."""

import concurrent.futures as _cf
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..state import PipelineState
from ..runner import _append_agentic_step

_log = get_logger(__name__)

# Module-level registry — keyed by video_id — holds the (executor, future) pair.
# Future objects are not JSON-serialisable so they cannot live in graph state.
_MAP_FUTURES: dict[str, tuple] = {}  # video_id → (_cf.ThreadPoolExecutor, _cf.Future)


def node_p2_map_3d_submit(state: PipelineState) -> dict[str, Any]:
    """Submit step_create_3d_map to a background thread and register the future."""
    from ...steps.perception.map import step_create_3d_map

    args = state["args"]
    video_id = state["video_id"]
    stats = dict(state.get("stats", {}))

    _sfm_min_dur = float(settings.SFM_MIN_DURATION_SEC)
    _clip_dur = float(stats.get("duration_sec", 0.0))
    run_sfm = not args.no_sfm
    if run_sfm and _sfm_min_dur > 0 and _clip_dur < _sfm_min_dur:
        _log.info(
            "  SfM skipped: clip %.1fs < SFM_MIN_DURATION_SEC=%.0fs — using pseudo-3D fallback",
            _clip_dur,
            _sfm_min_dur,
        )
        run_sfm = False

    _log.info("  -> Submitting 3D-map step 16 to background thread …")
    executor = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sfm-bg")
    future = executor.submit(
        step_create_3d_map,
        Path(state["video_path"]),
        video_id,
        Path(state["video_dir"]),
        state["frame_list"],
        state["models"],
        run_sfm_flag=run_sfm,
        run_gsplat_flag=not getattr(args, "no_gsplat", False),
        device=state["device"],
        depth_results=state.get("depth_result", {}).get("depth_results", []),
        yolo_detection_results=state.get("yolo_sam_result", {}).get("detection_results", []),
        tracking_results=state.get("gemma_tracking_result", {}),
    )
    _MAP_FUTURES[video_id] = (executor, future)
    return {}  # no state change — side-effect only


def node_p2_map_3d_join(state: PipelineState) -> dict[str, Any]:
    """Join the 3D-map background thread, then run advisor and semantic graph."""
    from ...steps.perception.map import step_advise_3d_map_quality
    from ...steps.perception.semantic_graph import step_build_semantic_environment_graph

    video_id = state["video_id"]
    t0 = time.monotonic()

    executor, future = _MAP_FUTURES.pop(video_id, (None, None))
    if future is not None:
        try:
            h = future.result(timeout=600)
        except Exception as exc:
            _log.warning("  3D-map background thread raised: %s", exc, exc_info=True)
            h = {
                "sfm_poses": 0,
                "method": "failed",
                "points": None,
                "gsplat_method": "failed",
                "splat_ply": None,
                "viewer_html": "",
            }
        finally:
            if executor:
                executor.shutdown(wait=False)
    else:
        h = {
            "sfm_poses": 0,
            "method": "skipped",
            "points": None,
            "gsplat_method": "skipped",
            "splat_ply": None,
            "viewer_html": "",
        }

    args = state["args"]
    semantic_graph_result: dict[str, Any] = {"skipped": True}
    if not getattr(args, "no_yolo", False) and settings.YOLO_SSG_ENABLED:
        semantic_graph_result = step_build_semantic_environment_graph(
            video_id=video_id,
            video_name=state["video_name"],
            video_dir=Path(state["video_dir"]),
            yolo_sam_result=state.get("yolo_sam_result", {}),
            map_result=h,
        )

    map_quality_advisor = step_advise_3d_map_quality(
        video_path=Path(state["video_path"]),
        video_dir=Path(state["video_dir"]),
        frame_list=state["frame_list"],
        map_result=h,
        caption_results=state.get("caption_results", []),
        tracking_results=state.get("gemma_tracking_result", {}),
    )

    stats = dict(state.get("stats", {}))
    stats["sfm_poses"] = h["sfm_poses"]
    stats["map_method"] = h["method"]
    stats["map_points"] = int(h["points"].shape[0]) if h.get("points") is not None else 0
    stats["gsplat_method"] = h.get("gsplat_method", "skipped")
    stats["map_degraded"] = bool(
        h.get("quality_degraded", stats["map_points"] < 50 or h["sfm_poses"] < 20)
    )
    stats["splat_ply"] = h.get("splat_ply")
    stats["semantic_graph_nodes"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("node_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    stats["semantic_graph_edges"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("edge_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    stats["map_quality_advisor"] = map_quality_advisor
    stats.setdefault("timings", {})["I_3dmap"] = float(
        h.get("elapsed_sec", time.monotonic() - t0) or 0.0
    )

    if stats["map_degraded"]:
        _log.warning(
            "3D map quality is degraded: %d points, %d SfM poses",
            stats["map_points"],
            stats["sfm_poses"],
        )

    video_context = dict(state.get("video_context", {}))
    video_context["map"] = {
        "method": h["method"],
        "points": stats["map_points"],
        "sfm_poses": h["sfm_poses"],
        "gsplat_method": stats["gsplat_method"],
        "splat_ply": stats["splat_ply"],
        "semantic_graph": semantic_graph_result.get("graph", {}).get("summary", {}),
    }

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="16",
        title="3D map creation",
        description="Recover scene geometry and export sparse-map or splat artifacts (ran concurrently with steps 11–14).",
        status="ok" if h["method"] not in ("failed", "skipped") else h["method"],
        context_inputs=["video frames", "camera-motion consistency"],
        context_outputs=[
            f"{stats['map_points']} map points",
            f"{stats['sfm_poses']} SfM poses",
            f"map method {stats['map_method']}",
        ],
        risks=[
            "geometry failure can create confident but wrong spatial context",
            "SfM fallback outputs may look valid while lacking metric truth",
        ],
        artifacts=["3d_map/sparse_map.npz", "3d_map/map_stats.json"],
    )

    return {
        "map_result": h,
        "semantic_graph_result": semantic_graph_result,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }

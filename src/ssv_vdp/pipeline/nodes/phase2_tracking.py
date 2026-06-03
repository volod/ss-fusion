"""Phase 2 tracking nodes: yolo_sam (step 9), gemma_tracking (step 10, agentic)."""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..state import PipelineState
from ..runner import _append_agentic_step
from .helpers import DEFAULT_TRACKING_TARGETS

_log = get_logger(__name__)


def node_p2_yolo_sam(state: PipelineState) -> dict[str, Any]:
    from ...steps.caption import _prep_vram_for_step
    from ...steps.perception.yolo_sam import step_yolo_sam_detection

    args = state["args"]
    yolo_sam_result: dict[str, Any] = {"skipped": True, "detection_results": []}
    knowledge = state.get("knowledge")
    t0 = time.monotonic()

    if not getattr(args, "no_yolo", False):
        _prep_vram_for_step(state["models"], state["device"])
        yolo_sam_result = step_yolo_sam_detection(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            state["device"],
            det_result=state.get("det_result", {}),
        )
        if not yolo_sam_result.get("skipped") and knowledge:
            knowledge.add_detections(yolo_sam_result.get("detection_results", []))

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="09",
        title="YOLO11 + SAM2/3 detection and segmentation",
        description=(
            "Run YOLO11 for fast instance detection with priority-ordered output, "
            "optionally refined with SAM2/3 segmentation masks."
        ),
        status="skipped" if yolo_sam_result.get("skipped") else "ok",
        context_inputs=["frames", "HF detection results from step 08"],
        context_outputs=[
            f"{yolo_sam_result.get('total_objects', 0)} YOLO detections",
            f"human={yolo_sam_result.get('human_count', 0)} vehicle={yolo_sam_result.get('vehicle_count', 0)}",
        ]
        if not yolo_sam_result.get("skipped")
        else ["no YOLO context"],
        risks=[
            "YOLO class confusion can misidentify humans as objects",
            "SAM masks may bleed across object boundaries in cluttered frames",
        ],
        artifacts=[
            "yolo_sam_results.json",
            "yolo_sam/frame_*_annotated.jpg",
            "detection_comparison.md",
        ]
        if not yolo_sam_result.get("skipped")
        else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["P2_yolo_sam"] = time.monotonic() - t0
    return {
        "yolo_sam_result": yolo_sam_result,
        "knowledge": knowledge,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


def node_p2_gemma_tracking(state: PipelineState) -> dict[str, Any]:
    """Step 10: Gemma directed tracking with JSON-guard fallback for empty target_labels."""
    from ...steps.caption import _prep_vram_for_step
    from ...steps.perception.gemma_tracking import step_gemma_directed_tracking

    args = state["args"]
    gemma_api_url = getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL
    gemma_api_model = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
    gemma_tracking_result: dict[str, Any] = {"skipped": True}
    t0 = time.monotonic()

    if not getattr(args, "no_rfdetr", False) and gemma_api_url:
        _prep_vram_for_step(state["models"], state["device"])
        gemma_tracking_result = step_gemma_directed_tracking(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            state["device"],
            models=state["models"],
            gemma_api_url=gemma_api_url,
            gemma_api_model=gemma_api_model,
            precomputed_scene=state.get("video_context", {}).get("gemma_structured_scene"),
        )

        # Agentic improvement: if JSON parse failure left target_labels empty,
        # substitute safe defaults so RF-DETR still tracks something useful.
        if not gemma_tracking_result.get("skipped"):
            tracking_degraded = gemma_tracking_result.get(
                "n_tracked_objects", 0
            ) == 0 and not gemma_tracking_result.get("target_labels")
            if tracking_degraded:
                _log.warning(
                    "  Step 10: Gemma tracking returned zero tracks with empty target_labels "
                    "— applying default targets %s",
                    DEFAULT_TRACKING_TARGETS,
                )
                gemma_tracking_result = {
                    **gemma_tracking_result,
                    "target_labels": DEFAULT_TRACKING_TARGETS,
                    "tracking_degraded": True,
                }

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="10",
        title="Gemma 4 directed tracking",
        description=(
            "Gemma 4 watches sampled frames, produces structured JSON with scene type and "
            "dominant object categories, then SAM segments and RF-DETR tracks those objects."
        ),
        status="skipped" if gemma_tracking_result.get("skipped") else "ok",
        context_inputs=["sampled frames", "Gemma sidecar API output", "CLIP embeddings"],
        context_outputs=[
            f"scene_type={gemma_tracking_result.get('scene_type', 'n/a')}",
            f"{gemma_tracking_result.get('n_tracked_objects', 0)} unique track IDs",
            f"tracking_degraded={gemma_tracking_result.get('tracking_degraded', False)}",
        ]
        if not gemma_tracking_result.get("skipped")
        else ["no Gemma tracking context"],
        risks=[
            "Gemma JSON parse failure now falls back to default targets (person/vehicle/sign)",
            "rough_bbox from Gemma may not align precisely — SAM mask may bleed",
            "RF-DETR tracking IDs reset per video; no cross-video identity",
        ],
        artifacts=[
            "gemma_tracking_results.json",
            "gemma_tracking/frame_*_tracked.jpg",
        ]
        if not gemma_tracking_result.get("skipped")
        else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["P3_gemma_tracking"] = time.monotonic() - t0
    return {
        "gemma_tracking_result": gemma_tracking_result,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }

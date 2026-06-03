"""Phase 1 graph nodes: init_state, extract_frames, index_vectors."""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import VideoKnowledge
from ..state import PipelineState
from ..runner import _append_agentic_step

_log = get_logger(__name__)


def node_init_state(state: PipelineState) -> dict[str, Any]:
    """Materialise per-video paths, reset stats/trace/context."""
    video_path = Path(state["video_path"])
    video_name = video_path.stem
    video_id = video_name.replace(" ", "_").lower()
    video_dir = Path(state["output_dir"]) / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    return {
        "video_name": video_name,
        "video_id": video_id,
        "video_dir": str(video_dir),
        "stats": {"name": video_name, "video_path": str(video_path), "timings": {}},
        "agentic_trace": [],
        "video_context": {"video_name": video_name},
        "completed_phases": [],
        "error": None,
        "clip_dino_on_gpu": False,
    }


def node_p1_extract_frames(state: PipelineState) -> dict[str, Any]:
    from ...steps.perception.embed import step_extract_frames

    video_path = Path(state["video_path"])
    video_id = state["video_id"]
    video_dir = Path(state["video_dir"])
    args = state["args"]

    t0 = time.monotonic()
    a = step_extract_frames(video_path, video_id, video_dir, fps=args.fps)
    elapsed = time.monotonic() - t0

    frame_list = a["frame_list"]
    stats = dict(state.get("stats", {}))
    stats["frames"] = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    stats.setdefault("timings", {})["A_extract"] = elapsed

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="01",
        title="Frame extraction",
        description="Decode the source video into a timestamped frame sequence that every later step reuses.",
        status="ok" if frame_list else "empty",
        context_inputs=["raw video bytes"],
        context_outputs=[
            f"{len(frame_list)} timestamped frames",
            f"duration {stats['duration_sec']:.1f}s",
        ],
        risks=[
            "sampling can miss short-lived objects or events",
            "timestamp drift can misalign later ASR/OCR/detection context",
        ],
        artifacts=["frames_metadata.json"],
    )

    video_context = dict(state.get("video_context", {}))
    video_context["meta"] = {
        "frame_count": stats["frames"],
        "duration_sec": stats["duration_sec"],
    }

    knowledge = VideoKnowledge(
        video_name=state["video_name"],
        duration_sec=stats["duration_sec"],
        frame_count=stats["frames"],
    )

    return {
        "frame_list": frame_list,
        "frames_meta": a.get("meta", {}),
        "stats": stats,
        "agentic_trace": agentic_trace,
        "video_context": video_context,
        "knowledge": knowledge,
    }


def node_p1_index_vectors(state: PipelineState) -> dict[str, Any]:
    from ...steps.caption import _models_on_device, _restore_models_to_gpu
    from ...steps.perception.embed import step_index_to_store

    models = state["models"]
    device = state["device"]
    clip_dino_on_gpu = state.get("clip_dino_on_gpu", False)

    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)

    t0 = time.monotonic()
    b = step_index_to_store(
        Path(state["video_path"]),
        state["video_id"],
        state["store"],
        state["is_qdrant"],
        models,
        state["frame_list"],
    )
    elapsed = time.monotonic() - t0

    if device == "cuda":
        clip_dino_on_gpu = _models_on_device(models, device)

    stats = dict(state.get("stats", {}))
    stats["index_sec"] = b["elapsed_sec"]
    stats.setdefault("timings", {})["B_index"] = elapsed

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="02",
        title="Vector store indexing",
        description="Embed frames for retrieval and establish the baseline semantic memory.",
        status="ok",
        context_inputs=["timestamped frames", "base CLIP/DINO embeddings"],
        context_outputs=[
            "retrieval index populated",
            f"index latency {b['elapsed_sec']:.1f}s",
        ],
        risks=["embedding collisions can mix semantically different frames"],
        artifacts=[],
    )

    return {
        "stats": stats,
        "agentic_trace": agentic_trace,
        "clip_dino_on_gpu": clip_dino_on_gpu,
    }

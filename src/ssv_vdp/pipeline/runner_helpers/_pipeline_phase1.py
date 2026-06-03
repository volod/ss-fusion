"""Phase 1 — Foundational ingestion: frame extraction and vector-store indexing (Steps 01-02)."""

from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import _Timer, _banner, _step
from ._agentic import _append_agentic_step

_log = get_logger(__name__)

_TOTAL_STEPS = 35


def run_phase1(
    *,
    args: Any,
    video_path: Path,
    video_dir: Path,
    video_name: str,
    video_id: str,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    # shared mutable state
    stats: dict[str, Any],
    T: dict[str, Any],
    video_context: dict[str, Any],
    agentic_trace: list[dict[str, Any]],
    knowledge: Any,
) -> tuple[list[tuple[str, float]], bool]:
    """Run Steps 01-02. Returns (frame_list, clip_dino_on_gpu)."""
    from ...steps.caption import _models_on_device, _restore_models_to_gpu
    from ...steps.perception.embed import step_extract_frames, step_index_to_store

    _banner("Phase 1 — Foundational ingestion")

    # Step 01: Extract frames
    _step(1, _TOTAL_STEPS, "Frame extraction")
    with _Timer(T, "A_extract"):
        a = step_extract_frames(video_path, video_id, video_dir, fps=args.fps)
    frame_list: list[tuple[str, float]] = a["frame_list"]
    stats["frames"] = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    video_context["meta"] = {
        "frame_count": stats["frames"],
        "duration_sec": stats["duration_sec"],
    }
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
            "frame timeline for all downstream alignment",
        ],
        risks=[
            "sampling can miss short-lived objects or events",
            "timestamp drift can misalign later ASR/OCR/detection context",
            "wrong extraction rate biases all downstream context",
        ],
        artifacts=["frames_metadata.json"],
    )

    clip_dino_on_gpu = device == "cuda" and _models_on_device(models, "cuda")

    # Step 02: Index — needs CLIP+DINO on GPU
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    with _Timer(T, "B_index"):
        b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    if device == "cuda":
        clip_dino_on_gpu = _models_on_device(models, device)
    stats["index_sec"] = b["elapsed_sec"]
    _append_agentic_step(
        agentic_trace,
        step_id="02",
        title="Vector store indexing",
        description="Embed frames for retrieval and establish the baseline semantic memory used by search steps.",
        status="ok",
        context_inputs=["timestamped frames", "base CLIP/DINO embeddings"],
        context_outputs=[
            "retrieval index populated",
            f"index latency {b['elapsed_sec']:.1f}s",
            "baseline visual neighborhoods",
        ],
        risks=[
            "embedding collisions can mix semantically different frames",
            "duplicate-heavy footage can distort nearest-neighbor context",
            "wrong baseline neighborhoods affect later search comparisons",
        ],
        artifacts=[],
    )

    return frame_list, clip_dino_on_gpu

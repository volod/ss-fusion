"""LangGraph-based orchestrator for the 24-step local learning pipeline.

Entry point: ``run_graph_pipeline()`` — drop-in replacement for the monolith
``run_video_pipeline()`` in ``runner.py``, activated by env var::

    SELFSUVIS_USE_GRAPH=1

Graph topology (see plan for full ASCII diagram):
  Phase 1: init → extract_frames → index_vectors
  Phase 2: gemma_analysis → [parallel: florence/asr/ocr/depth/detection]
           → merge_parallel → platform_fusion → yolo_sam → gemma_tracking
           → map_3d_submit → world_model → qwen_caption → unidrive
           → scenetok → base_search → map_3d_join → full_fusion
  Phase 3: ssl_finetune → [ssl_gate] → distill → onnx_export → ft_search → compare
  Phase 4: multi_model_compare → synthesis → audit → emit_analytics → END

Checkpointing
-------------
* In-process (default): MemorySaver — state survives exceptions within a run.
* Persistent: set SELFSUVIS_CHECKPOINT_PATH=/path/to/checkpoints.db to use
  SqliteSaver; set SELFSUVIS_RESUME_THREAD_ID=<id> to resume a prior run.

LangSmith tracing
-----------------
Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY=<key> to emit traces.
"""

import os
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger(__name__)


# -- Graph builder -------------------------------------------------------------


def build_graph(use_checkpoints: bool = True):
    """Assemble and compile the 24-step pipeline graph.

    Returns a ``CompiledStateGraph`` ready for ``.invoke()``.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        phase1,
        phase2_map,
        phase2_parallel,
        phase2_serial,
        phase2_tracking,
        phase3_ssl,
        phase4,
    )
    from .state import PipelineState

    sg = StateGraph(PipelineState)

    # -- Phase 1 --------------------------------------------------------------
    sg.add_node("init_state", phase1.node_init_state)
    sg.add_node("p1_extract_frames", phase1.node_p1_extract_frames)
    sg.add_node("p1_index_vectors", phase1.node_p1_index_vectors)

    # -- Phase 2 serial prefix ------------------------------------------------
    sg.add_node("p2_gemma_analysis", phase2_serial.node_p2_gemma_analysis)

    # -- Phase 2 parallel fan-out (steps 4–8) ---------------------------------
    sg.add_node("p2_florence_caption", phase2_parallel.node_p2_florence_caption)
    sg.add_node("p2_asr", phase2_parallel.node_p2_asr)
    sg.add_node("p2_ocr", phase2_parallel.node_p2_ocr)
    sg.add_node("p2_depth", phase2_parallel.node_p2_depth)
    sg.add_node("p2_detection", phase2_parallel.node_p2_detection)
    sg.add_node("p2_merge_parallel", phase2_serial.node_p2_merge_parallel)

    # -- Phase 2 serial suffix ------------------------------------------------
    sg.add_node("p2_platform_fusion", phase2_serial.node_p2_platform_fusion)
    sg.add_node("p2_yolo_sam", phase2_tracking.node_p2_yolo_sam)
    sg.add_node("p2_gemma_tracking", phase2_tracking.node_p2_gemma_tracking)
    sg.add_node("p2_map_3d_submit", phase2_map.node_p2_map_3d_submit)
    sg.add_node("p2_world_model", phase2_serial.node_p2_world_model)
    sg.add_node("p2_qwen_caption", phase2_serial.node_p2_qwen_caption)
    sg.add_node("p2_unidrive", phase2_serial.node_p2_unidrive)
    sg.add_node("p2_scenetok", phase2_serial.node_p2_scenetok)
    sg.add_node("p2_base_search", phase2_serial.node_p2_base_search)
    sg.add_node("p2_map_3d_join", phase2_map.node_p2_map_3d_join)
    sg.add_node("p2_full_fusion", phase2_serial.node_p2_full_fusion)

    # -- Phase 3 --------------------------------------------------------------
    sg.add_node("p3_ssl_finetune", phase3_ssl.node_p3_ssl_finetune)
    sg.add_node("p3_distill", phase3_ssl.node_p3_distill)
    sg.add_node("p3_onnx_export", phase3_ssl.node_p3_onnx_export)
    sg.add_node("p3_ft_search", phase3_ssl.node_p3_ft_search)
    sg.add_node("p3_compare", phase3_ssl.node_p3_compare)

    # -- Phase 4 --------------------------------------------------------------
    sg.add_node("p4_multi_model_compare", phase4.node_p4_multi_model_compare)
    sg.add_node("p4_synthesis", phase4.node_p4_synthesis)
    sg.add_node("p4_audit", phase4.node_p4_audit)
    sg.add_node("p4_emit_analytics", phase4.node_p4_emit_analytics)

    # -- Edges: phase 1 -------------------------------------------------------
    sg.add_edge(START, "init_state")
    sg.add_edge("init_state", "p1_extract_frames")
    sg.add_edge("p1_extract_frames", "p1_index_vectors")
    sg.add_edge("p1_index_vectors", "p2_gemma_analysis")

    # -- Edges: phase 2 parallel fan-out --------------------------------------
    # LangGraph runs all successors of a node concurrently when there are
    # multiple outgoing edges; they converge at p2_merge_parallel.
    for parallel_node in [
        "p2_florence_caption",
        "p2_asr",
        "p2_ocr",
        "p2_depth",
        "p2_detection",
    ]:
        sg.add_edge("p2_gemma_analysis", parallel_node)
        sg.add_edge(parallel_node, "p2_merge_parallel")

    # -- Edges: phase 2 serial suffix -----------------------------------------
    sg.add_edge("p2_merge_parallel", "p2_platform_fusion")
    sg.add_edge("p2_platform_fusion", "p2_yolo_sam")
    sg.add_edge("p2_yolo_sam", "p2_gemma_tracking")
    sg.add_edge("p2_gemma_tracking", "p2_map_3d_submit")
    sg.add_edge("p2_map_3d_submit", "p2_world_model")
    sg.add_edge("p2_world_model", "p2_qwen_caption")
    sg.add_edge("p2_qwen_caption", "p2_unidrive")
    sg.add_edge("p2_unidrive", "p2_scenetok")
    sg.add_edge("p2_scenetok", "p2_base_search")
    sg.add_edge("p2_base_search", "p2_map_3d_join")
    sg.add_edge("p2_map_3d_join", "p2_full_fusion")

    # -- Edges: phase 3 -------------------------------------------------------
    sg.add_edge("p2_full_fusion", "p3_ssl_finetune")
    sg.add_conditional_edges(
        "p3_ssl_finetune",
        phase3_ssl.ssl_gate_router,
        {
            "p3_distill": "p3_distill",
            "p4_multi_model_compare": "p4_multi_model_compare",
        },
    )
    sg.add_edge("p3_distill", "p3_onnx_export")
    sg.add_edge("p3_onnx_export", "p3_ft_search")
    sg.add_edge("p3_ft_search", "p3_compare")
    sg.add_edge("p3_compare", "p4_multi_model_compare")

    # -- Edges: phase 4 -------------------------------------------------------
    sg.add_edge("p4_multi_model_compare", "p4_synthesis")
    sg.add_edge("p4_synthesis", "p4_audit")
    sg.add_edge("p4_audit", "p4_emit_analytics")
    sg.add_edge("p4_emit_analytics", END)

    # -- Checkpointer selection ------------------------------------------------
    checkpointer = None
    if use_checkpoints:
        checkpoint_path = os.getenv("SELFSUVIS_CHECKPOINT_PATH")
        if checkpoint_path:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
                _log.info("Graph checkpointer: SqliteSaver at %s", checkpoint_path)
            except Exception as exc:
                _log.warning("SqliteSaver unavailable (%s) — falling back to MemorySaver", exc)
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
        else:
            checkpointer = MemorySaver()

    return sg.compile(checkpointer=checkpointer)


# -- Entry point ---------------------------------------------------------------


def run_graph_pipeline(
    args: Any,
    video_path: Path,
    output_dir: Path,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> dict[str, Any]:
    """Drop-in replacement for ``run_video_pipeline()``.

    Called from runner.py when SELFSUVIS_USE_GRAPH=1 is set.
    Returns the per-video stats dict (same contract as the monolith).
    """
    from .state import PipelineState, SerializableNamespace
    from ..steps.caption import reset_runtime_telemetry

    reset_runtime_telemetry()

    # Disable in-process MemorySaver when no persistent checkpoint path is set.
    # MemorySaver serialises all state via msgpack on every node — torch objects
    # (OpenCLIPEmbedder, DINOEmbedder) in the `models` field are not serialisable.
    # SqliteSaver is unaffected: it is only activated via SELFSUVIS_CHECKPOINT_PATH.
    use_ckpt = bool(os.getenv("SELFSUVIS_CHECKPOINT_PATH"))
    graph = build_graph(use_checkpoints=use_ckpt)
    video_name = video_path.stem
    thread_id = f"{video_name}_{int(time.time())}"

    # Resume support: if SELFSUVIS_RESUME_THREAD_ID is set, use it.
    resume_thread = os.getenv("SELFSUVIS_RESUME_THREAD_ID")
    if resume_thread:
        _log.info("Resuming graph run from thread_id=%s", resume_thread)
        thread_id = resume_thread

    initial_state: PipelineState = {
        "args": SerializableNamespace(vars(args)),
        "video_path": str(video_path),
        "video_name": video_name,
        "video_id": video_name.replace(" ", "_").lower(),
        "video_dir": str(output_dir / video_name),
        "output_dir": str(output_dir),
        "device": device,
        "models": models,
        "store": store,
        "is_qdrant": is_qdrant,
        "stats": {"name": video_name, "video_path": str(video_path), "timings": {}},
        "agentic_trace": [],
        "video_context": {"video_name": video_name},
        "completed_phases": [],
        "error": None,
        "clip_dino_on_gpu": False,
    }

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true"):
        config["run_name"] = f"selfsuvis/{video_name}"
        _log.info("LangSmith tracing enabled for run: selfsuvis/%s", video_name)

    _log.info("Starting graph pipeline for %s (thread_id=%s)", video_name, thread_id)
    t0 = time.monotonic()

    try:
        # On resume, pass None as the initial state to reload from checkpoint.
        invoke_state = None if resume_thread else initial_state
        final_state = graph.invoke(invoke_state, config)
    except Exception as exc:
        _log.error("Graph pipeline failed for %s: %s", video_name, exc, exc_info=True)
        return {
            "name": video_name,
            "video_path": str(video_path),
            "error": str(exc),
            "timings": {},
            "frames": 0,
            "duration_sec": 0.0,
            "pipeline_sec": time.monotonic() - t0,
        }

    stats = final_state.get("stats", {})
    _log.info(
        "Graph pipeline complete for %s in %.1fs (thread_id=%s)",
        video_name,
        time.monotonic() - t0,
        thread_id,
    )
    return stats

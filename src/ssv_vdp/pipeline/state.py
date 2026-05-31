"""LangGraph state schema for the 27-step local learning pipeline.

``PipelineState`` is a TypedDict consumed by every graph node.  Each node
returns a *partial* dict containing only the keys it writes; LangGraph merges
them with last-writer-wins (the default reducer for TypedDict fields).

Design notes
------------
* ``knowledge``  — VideoKnowledge instance stored as a Python object reference.
  MemorySaver keeps it alive in-process.  SqliteSaver reconstruction uses
  ``_reconstruct_knowledge`` (at resume time only, not per-step).
* ``models``     — Large PyTorch objects; never serialised.  On resume they are
  re-injected from the caller's namespace via ``run_graph_pipeline``.
* Parallel fan-out nodes (steps 4–8) each write *distinct* keys so there is
  no reducer conflict.  ``node_p2_merge_parallel`` reads all five and commits
  to ``knowledge`` and ``video_context``.
"""

from typing import Annotated, Any

from typing_extensions import TypedDict


class SerializableNamespace(dict):
    """Dict subclass with attribute-style access, safe for LangGraph msgpack checkpointing.

    Stores CLI args as a plain dict so the checkpointer can serialize them, while
    keeping the ``args.field`` access pattern used across all graph nodes unchanged.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


def _merge_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two stats dicts — deep-merges the 'timings' sub-dict."""
    result = dict(a)
    timings_a = dict(result.get("timings") or {})
    timings_b = dict((b or {}).get("timings") or {})
    result.update(b or {})
    result["timings"] = {**timings_a, **timings_b}
    return result


class PipelineState(TypedDict, total=False):
    # -- Runtime config (injected once at graph entry, never mutated) ---------
    args: Any  # parsed CLI args namespace
    video_path: str  # absolute path
    video_name: str
    video_id: str
    video_dir: str  # absolute path of per-video output dir
    output_dir: str
    device: str
    models: dict[str, Any]  # {clip, dino, uses_api_embedder, …}
    store: Any  # Qdrant client or InMemoryStore
    is_qdrant: bool

    # -- Phase 1 outputs -------------------------------------------------------
    frame_list: list[tuple[str, float]]  # [(frame_path, t_sec), …]
    frames_meta: dict[str, Any]
    knowledge: Any  # VideoKnowledge instance
    clip_dino_on_gpu: bool

    # -- Phase 2 outputs — one key per step -----------------------------------
    gemma_result: dict[str, Any]
    caption_results: list[dict[str, Any]]
    asr_result: dict[str, Any]
    platform_fusion_result: dict[str, Any]
    ocr_result: dict[str, Any]
    depth_result: dict[str, Any]
    det_result: dict[str, Any]
    yolo_sam_result: dict[str, Any]
    gemma_tracking_result: dict[str, Any]
    world_result: dict[str, Any]
    qwen_result: dict[str, Any]
    unidrive_result: dict[str, Any]
    scenetok_result: dict[str, Any]
    base_results: list[dict[str, Any]]
    query_frame: str
    query_t_sec: float
    map_result: dict[str, Any]
    semantic_graph_result: dict[str, Any]
    full_fusion_result: dict[str, Any]
    physical_state_result: dict[str, Any]
    field_state_result: dict[str, Any]
    threat_primitives_result: dict[str, Any]
    local_threat_result: dict[str, Any]
    policy_result: dict[str, Any]

    # -- Phase 3 SSL outputs ---------------------------------------------------
    ssl_result: dict[str, Any]
    ssl_gate_passed: bool
    checkpoint_path: str
    student_backbone: Any
    student_dim: int
    distill_result: dict[str, Any]
    export_result: dict[str, Any]
    ft_results: list[dict[str, Any]]
    compare_result: dict[str, Any]
    dae_result: dict[str, Any]  # DAE reconstruction pretraining output

    # -- Phase 4 outputs -------------------------------------------------------
    multi_model_result: dict[str, Any]
    synthesis_result: dict[str, Any]
    audit_result: dict[str, Any]

    # -- Cross-cutting accumulation --------------------------------------------
    stats: Annotated[dict[str, Any], _merge_stats]  # timing dict T + numeric summaries
    video_context: dict[str, Any]  # rich context fed to LLM synthesis/audit
    agentic_trace: list[dict[str, Any]]

    # -- Resume support -------------------------------------------------------
    completed_phases: list[str]
    error: str | None

"""Phase 4 nodes: multi_model_compare, synthesis (agentic), audit (agentic),
emit_analytics.
"""

import json
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..state import PipelineState
from ..runner import (
    _append_agentic_step,
    step_agentic_flow_artifact,
    step_multi_model_compare,
    step_video_synthesis,
)
from .helpers import build_evidence_summary, critique_pass, llm_call_with_retry

_log = get_logger(__name__)


# -- Step 22: Multi-model comparison ------------------------------------------


def node_p4_multi_model_compare(state: PipelineState) -> dict[str, Any]:
    qwen_result = state.get("qwen_result", {"skipped": True})
    unidrive_result = state.get("unidrive_result", {"skipped": True})
    gemma_result = state.get("gemma_result", {"skipped": True})
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))
    video_context = dict(state.get("video_context", {}))

    mm: dict[str, Any] = {}
    t0 = time.monotonic()
    if not qwen_result.get("skipped") and not unidrive_result.get("skipped"):
        mm = step_multi_model_compare(
            state["video_name"],
            Path(state["video_dir"]),
            gemma_result,
            qwen_result,
            unidrive_result,
        )
        video_context["multi_model_comparison"] = mm
    stats.setdefault("timings", {})["T_multimodel"] = time.monotonic() - t0

    _append_agentic_step(
        agentic_trace,
        step_id="22",
        title="Multi-model comparison",
        description="Compare Gemma, Qwen, and UniDriveVLA outputs and expose MoE agreement signals.",
        status="ok" if mm else "skipped",
        context_inputs=["Gemma summary", "Qwen structured scene facts", "UniDrive expert outputs"],
        context_outputs=[
            f"{mm.get('matched_frames', 0)} matched frames",
            f"mean_moe_agreement={mm.get('mean_moe_agreement', 0.0):.3f}",
        ]
        if mm
        else ["no cross-model comparison artifact"],
        risks=["timestamp-nearest matching can compare slightly different moments"],
        artifacts=["multi_model_comparison.md"] if mm else [],
    )

    return {
        "multi_model_result": mm,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


# -- Step 23: Video synthesis (agentic: draft → critique → conditional regen) -


def node_p4_synthesis(state: PipelineState) -> dict[str, Any]:
    """Video synthesis with a critique pass.

    Flow:
    1. Call step_video_synthesis() — existing logic writes video_synthesis.md.
    2. Run a lightweight critique prompt against factual evidence from state.
    3. If verdict is MAJOR_CONTRADICTION, regenerate with the critique embedded.
    """
    args = state["args"]
    _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    video_context = dict(state.get("video_context", {}))
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    from ..steps_caption import _offload_models_to_cpu

    device = state["device"]
    clip_dino_on_gpu = state.get("clip_dino_on_gpu", False)
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(state["models"])

    t0 = time.monotonic()
    step_video_synthesis(
        state["video_name"],
        Path(state["video_dir"]),
        video_context,
        api_url=_qwen_url,
        model=_qwen_model,
    )

    synthesis_result: dict[str, Any] = {
        "skipped": not _qwen_url,
        "api_url": _qwen_url,
    }

    # Critique pass — only if synthesis actually ran.
    if _qwen_url:
        synthesis_result = _synthesis_critique(
            synthesis_result,
            state,
            _qwen_url,
            _qwen_model,
            video_context,
        )

    stats.setdefault("timings", {})["Z_synthesis"] = time.monotonic() - t0

    _append_agentic_step(
        agentic_trace,
        step_id="23",
        title="Video synthesis",
        description="Use accumulated multimodal context to generate structured ontology and narrative.",
        status="ok" if _qwen_url else "skipped",
        context_inputs=["Gemma summary", "captions, ASR, OCR, detections, Qwen frame reasoning"],
        context_outputs=[
            "video ontology",
            f"critique_verdict={synthesis_result.get('critique_verdict', 'n/a')}",
            f"regenerated={synthesis_result.get('regenerated', False)}",
        ]
        if _qwen_url
        else ["no synthesis output"],
        risks=[
            "final narrative can collapse uncertain evidence into a single confident story",
            "MAJOR_CONTRADICTION trigger may regenerate unnecessarily on edge cases",
        ],
        artifacts=["video_synthesis.md", "video_ontology.json"] if _qwen_url else [],
    )

    return {
        "synthesis_result": synthesis_result,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
        "clip_dino_on_gpu": False,
    }


def _synthesis_critique(
    synthesis_result: dict[str, Any],
    state: PipelineState,
    api_url: str,
    model: str,
    video_context: dict[str, Any],
) -> dict[str, Any]:
    """Run critique pass; regenerate synthesis if MAJOR_CONTRADICTION detected."""
    video_dir = Path(state["video_dir"])
    synthesis_md = video_dir / "video_synthesis.md"
    if not synthesis_md.exists():
        return synthesis_result

    generation = synthesis_md.read_text(encoding="utf-8", errors="ignore")
    if len(generation) < 100:
        return synthesis_result

    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    evidence = build_evidence_summary(dict(state))
    verdict = critique_pass(endpoint, model, generation, evidence, timeout_sec=60.0)
    synthesis_result = {**synthesis_result, "critique_verdict": verdict}

    if verdict == "MAJOR_CONTRADICTION":
        _log.info("  Synthesis critique: MAJOR_CONTRADICTION — regenerating with correction note")
        corrected_ctx = {
            **video_context,
            "critique_note": (
                f"A prior synthesis was flagged as contradicting the frame evidence. "
                f"Please produce a corrected ontology and narrative that is consistent with: {evidence[:400]}"
            ),
        }
        try:
            step_video_synthesis(
                state["video_name"],
                video_dir,
                corrected_ctx,
                api_url=api_url,
                model=model,
            )
            synthesis_result = {**synthesis_result, "regenerated": True}
            _log.info("  Synthesis regenerated successfully.")
        except Exception as exc:
            _log.warning("  Synthesis regeneration failed: %s", exc)

    return synthesis_result


# -- Agentic flow audit (with reflection loop) — runner step 30 ---------------


def node_p4_audit(state: PipelineState) -> dict[str, Any]:
    """Agentic flow audit with a reflection sub-loop.

    Flow:
    1. Call step_agentic_flow_artifact() — existing 3-attempt + deterministic fallback.
    2. Reflection pass: check whether all step IDs in agentic_trace are covered.
    3. If HAS_GAPS: append gap section to agentic_flow.md.
    """
    args = state["args"]
    video_context = dict(state.get("video_context", {}))
    agentic_trace = list(state.get("agentic_trace", []))
    stats = dict(state.get("stats", {}))

    _agentic_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
        or getattr(args, "gemma_api_url", "")
        or settings.GEMMA_API_URL
        or getattr(args, "qwen_api_url", "")
        or settings.QWEN_API_URL
    )
    _agentic_model = (
        getattr(args, "reasoning_model", "")
        or getattr(settings, "REASONING_MODEL", "")
        or getattr(args, "gemma_api_model", "")
        or settings.GEMMA_API_MODEL
        or getattr(args, "qwen_model", "")
        or settings.QWEN_MODEL
    )

    t0 = time.monotonic()
    step_agentic_flow_artifact(
        state["video_name"],
        Path(state["video_dir"]),
        video_context,
        api_url=_agentic_url,
        model=_agentic_model,
    )
    audit_result: dict[str, Any] = {
        "api_url": _agentic_url,
        "llm_used": bool(_agentic_url),
    }

    # Reflection pass.
    if _agentic_url:
        audit_result = _audit_reflection(
            audit_result, state, agentic_trace, _agentic_url, _agentic_model
        )

    stats.setdefault("timings", {})["AA_agentic"] = time.monotonic() - t0

    # Evict all sidecars from VRAM now that the pipeline is complete.
    if state["device"] == "cuda":
        from ..steps_caption import _unload_known_sidecars

        _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
        _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
        _unload_known_sidecars(
            [
                (_agentic_url, _agentic_model),
                (_qwen_url, _qwen_model),
                (
                    getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                    getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
                ),
            ]
        )

    _append_agentic_step(
        agentic_trace,
        step_id="24",
        title="Agentic flow audit",
        description="Audit the full context chain, explain step-to-step reasoning, and register risks.",
        status="ok",
        context_inputs=["complete pipeline trace", "all accumulated artifacts"],
        context_outputs=[
            "agentic_flow.md audit report",
            f"reflection_verdict={audit_result.get('reflection_verdict', 'n/a')}",
        ],
        risks=[
            "reasoning model can restate upstream errors coherently",
            "fallback deterministic summary is less nuanced than LLM audit",
        ],
        artifacts=["agentic_flow.md"],
    )

    return {
        "audit_result": audit_result,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


def _audit_reflection(
    audit_result: dict[str, Any],
    state: PipelineState,
    agentic_trace: list,
    api_url: str,
    model: str,
) -> dict[str, Any]:
    """Check whether the audit covered all pipeline step IDs; append gaps if not."""
    video_dir = Path(state["video_dir"])
    audit_md = video_dir / "agentic_flow.md"
    if not audit_md.exists():
        return audit_result

    llm_text = audit_md.read_text(encoding="utf-8", errors="ignore")
    if len(llm_text) < 200:
        return audit_result

    trace_step_ids = {item["step_id"] for item in agentic_trace}
    endpoint = f"{api_url.rstrip('/')}/chat/completions"

    reflection_prompt = (
        f"You produced this pipeline audit:\n\n{llm_text[:1200]}\n\n"
        f"The pipeline had {len(trace_step_ids)} steps: "
        f"{', '.join(sorted(trace_step_ids))}.\n"
        "Check: (1) Did you cover all steps? "
        "(2) Did you mention cross-step context propagation risk? "
        "Reply COMPLETE if yes to both, or list specific gaps in 3 bullets."
    )
    try:
        reflection_raw, _ = llm_call_with_retry(
            endpoint,
            {
                "model": model,
                "messages": [{"role": "user", "content": reflection_prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            max_attempts=2,
            timeout_sec=90.0,
        )
        verdict = "COMPLETE" if "COMPLETE" in reflection_raw.upper() else "HAS_GAPS"
        audit_result = {
            **audit_result,
            "reflection_verdict": verdict,
            "reflection_text": reflection_raw[:400],
        }
        if verdict == "HAS_GAPS":
            _log.info("  Audit reflection: HAS_GAPS — appending gap section to agentic_flow.md")
            with audit_md.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n## Reflection Gaps\n\n{reflection_raw}\n")
    except Exception as exc:
        _log.debug("Audit reflection pass failed: %s", exc)
        audit_result = {**audit_result, "reflection_verdict": "REFLECTION_FAILED"}

    return audit_result


# -- Emit analytics ------------------------------------------------------------


def node_p4_emit_analytics(state: PipelineState) -> dict[str, Any]:
    from ..runner import _emit_local_run_analytics
    from ..steps_caption import get_runtime_telemetry

    video_dir = Path(state["video_dir"])
    stats = dict(state.get("stats", {}))

    runtime_metrics = get_runtime_telemetry()
    stats["vram_wait_time_sec"] = runtime_metrics.get("vram_wait_time_sec", 0.0)
    stats["restore_failures"] = int(runtime_metrics.get("restore_failures", 0.0))
    (video_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2), encoding="utf-8"
    )

    timings = stats.get("timings", {})
    stats["pipeline_sec"] = sum(timings.values())

    analysis_summary = _emit_local_run_analytics(video_dir) or {}
    stats["analysis_summary"] = analysis_summary

    return {"stats": stats}

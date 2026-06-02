"""Agentic-trace helpers: step recorder, context/flow prompt builders, validation."""

import re
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger(__name__)


def _build_context_prompt(video_name: str, video_context: dict[str, Any]) -> str:
    """Build a text prompt summarising accumulated observations for the LLM."""
    parts = [f"Video: {video_name}"]

    meta = video_context.get("meta", {})
    if meta:
        parts.append(
            f"Duration: {meta.get('duration_sec', 0):.1f}s | Frames: {meta.get('frame_count', 0)}"
        )

    gem_ctx = video_context.get("gemma_analysis", {})
    if gem_ctx:
        parts.append(
            f"\nGemma analysis ({gem_ctx.get('n_frames', 0)} frames, "
            f"{gem_ctx.get('n_tasks', 0)} analyses):"
        )
        task_res = gem_ctx.get("task_results", {})
        clf = task_res.get("scene_classification", {})
        if clf.get("category_distribution"):
            top_cat = next(iter(clf["category_distribution"]))
            parts.append(f"  - dominant scene type: {top_cat}")
        fv = task_res.get("fact_verification", {})
        if fv.get("claims"):
            top_claim = max(fv["claims"], key=lambda r: r.get("mean_score", 0.0))
            parts.append(
                f"  - strongest visual claim: {top_claim['claim']} "
                f"(score={top_claim['mean_score']:.3f})"
            )
        sc = task_res.get("scene_change_detection", {})
        if sc.get("n_changes") is not None:
            parts.append(f"  - scene transitions detected: {sc['n_changes']}")
        cl = task_res.get("scene_clustering", {})
        if cl.get("n_clusters"):
            parts.append(f"  - semantic clusters: {cl['n_clusters']}")
        mnn = gem_ctx.get("mnn_rate")
        if mnn is None:
            mnn = gem_ctx.get("mnn_rate_dino")
        if mnn is not None:
            parts.append(f"  - Gemma/DINOv3 MNN agreement: {mnn:.1%}")

    top_descs = video_context.get("top_descriptions", [])
    if top_descs:
        parts.append("\nTop scene descriptions (CLIP similarity):")
        for desc, score in top_descs[:5]:
            parts.append(f"  - {desc} (score={score:.3f})")

    captions = video_context.get("captions", [])
    if captions:
        step = max(1, len(captions) // 20)
        sampled = captions[::step][:20]
        parts.append(f"\nPer-frame captions ({len(sampled)} sampled from {len(captions)}):")
        for r in sampled:
            cap = r.get("caption", "")
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {cap}")

    asr_segs = video_context.get("asr_segments", [])
    if asr_segs:
        parts.append(f"\nAudio transcript ({len(asr_segs)} segments):")
        for seg in asr_segs[:10]:
            ts = seg.get("timestamp") or (0.0, 0.0)
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"  [{ts[0]:.1f}s–{ts[1]:.1f}s] {text}")

    ocr_list = video_context.get("ocr", [])
    if ocr_list:
        ocr_with_text = [r for r in ocr_list if r.get("ocr_text")][:10]
        if ocr_with_text:
            parts.append(f"\nVisible text (OCR, {len(ocr_with_text)} frames with text):")
            for r in ocr_with_text[:5]:
                parts.append(f"  [{r['t_sec']:.1f}s] {r['ocr_text'][:100]}")

    obj_counts = video_context.get("detections", {})
    if obj_counts:
        parts.append("\nDetected objects (label: count):")
        for label, count in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"  - {label}: {count}")

    qwen_caps = video_context.get("qwen_captions", [])
    if qwen_caps:
        step = max(1, len(qwen_caps) // 10)
        sampled = qwen_caps[::step][:10]
        parts.append(f"\nDetailed scene analysis ({len(sampled)} sampled from {len(qwen_caps)}):")
        for r in sampled:
            cap = r.get("caption") or r.get("scene_description") or ""
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {str(cap)[:200]}")

    unidrive_rows = video_context.get("unidrive_analysis", [])
    if unidrive_rows:
        parts.append(
            f"\nUniDriveVLA expert analysis ({min(len(unidrive_rows), 8)} sampled frames):"
        )
        for r in unidrive_rows[:8]:
            understanding = (r.get("understanding") or {}).get("scene_summary", "")
            planning = (r.get("planning") or {}).get("recommended_action", "")
            moe = (r.get("mixture_of_experts") or {}).get("consensus_summary", "")
            if understanding or planning or moe:
                parts.append(
                    f"  [{r.get('t_sec', 0.0):.1f}s] understand={understanding[:90]} "
                    f"| plan={planning[:70]} | moe={moe[:90]}"
                )

    mm = video_context.get("multi_model_comparison", {})
    if mm:
        parts.append("\nCross-model comparison:")
        parts.append(
            f"  - matched frames: {mm.get('matched_frames', 0)} | "
            f"Qwen/UniDrive agreement: {mm.get('mean_qwen_unidrive_agreement', 0.0):.3f} | "
            f"high-risk UniDrive frames: {mm.get('high_risk_frames', 0)}"
        )

    local_threat = video_context.get("local_threat", {})
    policy_decision = video_context.get("policy_decision", {})
    if local_threat and not local_threat.get("skipped"):
        parts.append("\nLocal threat assessment:")
        parts.append(
            f"  - score: {float(local_threat.get('local_threat_score', 0.0)):.3f} | "
            f"recommended action: {policy_decision.get('recommended_action', local_threat.get('recommended_action', 'continue'))}"
        )
        for threat in (local_threat.get("top_threats") or [])[:3]:
            evidence = threat.get("evidence") or {}
            parts.append(
                f"  - {threat.get('type', 'unknown')}: score={float(threat.get('score', 0.0)):.3f} "
                f"| sources={', '.join(evidence.get('evidence_sources', []) or []) or 'none'} "
                f"| frames={int(evidence.get('support_frames', 0) or 0)} "
                f"| persistence={int(evidence.get('temporal_persistence', 0) or 0)}"
            )

    return "\n".join(parts)


def _append_agentic_step(
    trace: list[dict[str, Any]],
    *,
    step_id: str,
    title: str,
    description: str,
    status: str,
    context_inputs: list[str] | None = None,
    context_outputs: list[str] | None = None,
    risks: list[str] | None = None,
    artifacts: list[str] | None = None,
) -> None:
    trace.append(
        {
            "step_id": step_id,
            "title": title,
            "description": description,
            "status": status,
            "context_inputs": context_inputs or [],
            "context_outputs": context_outputs or [],
            "risks": risks or [],
            "artifacts": artifacts or [],
        }
    )


def _build_agentic_flow_prompt(video_name: str, video_context: dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "You are auditing a local multimodal video-analysis pipeline.",
        "Analyze how context is accumulated step by step, how later steps depend on earlier outputs, and where wrong context can propagate.",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.extend(
            [
                f"- Step {item.get('step_id')} {item.get('title')}",
                f"  Description: {item.get('description', '')}",
                f"  Status: {item.get('status', 'unknown')}",
                f"  Context received: {', '.join(item.get('context_inputs', [])) or 'none'}",
                f"  Context produced: {', '.join(item.get('context_outputs', [])) or 'none'}",
                f"  Risks: {', '.join(item.get('risks', [])) or 'none'}",
                f"  Artifacts: {', '.join(item.get('artifacts', [])) or 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "Write markdown with these sections exactly:",
            "## Flow Summary",
            "Short explanation of how context evolves through the pipeline.",
            "## Step-by-Step Agentic Context",
            "Use one bullet per step. For each step explain what context entered, what new context was created, and what later steps rely on it.",
            "## Risk Register",
            "Use one bullet per step. Explicitly call out misidentification risk, wrong-context risk, and propagation risk.",
            "## Highest-Risk Context Failures",
            "List the most important compounded failure modes across the pipeline.",
            "## Mitigations",
            "Recommend concrete checks or gates to reduce context corruption.",
            "",
            "Be specific. Focus on agentic context flow, not generic ML commentary.",
        ]
    )
    return "\n".join(lines)


def _build_agentic_flow_prompt_compact(video_name: str, video_context: dict[str, Any]) -> str:
    """Compact audit prompt tuned for slow reasoning models on Ollama."""
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "Audit the agentic pipeline context flow.",
        "Return markdown with these exact sections:",
        "## Flow Summary",
        "## Step-by-Step Agentic Context",
        "## Risk Register",
        "## Highest-Risk Context Failures",
        "## Mitigations",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.append(
            f"- {item.get('step_id')} {item.get('title')} | "
            f"status={item.get('status', 'unknown')} | "
            f"in={'; '.join(item.get('context_inputs', [])[:3]) or 'none'} | "
            f"out={'; '.join(item.get('context_outputs', [])[:3]) or 'none'} | "
            f"risks={'; '.join(item.get('risks', [])[:3]) or 'none'}"
        )
    lines += [
        "",
        "Be concise and specific.",
        "Keep the whole answer under 900 words.",
        "Focus on context propagation, stale context, misidentification, and mitigation.",
    ]
    return "\n".join(lines)


def _build_agentic_flow_prompt_simple(video_name: str, video_context: dict[str, Any]) -> str:
    """Minimal audit prompt for short, low-branching local runs."""
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "Audit context propagation for this local video-analysis run.",
        "Return markdown with these exact sections:",
        "## Flow Summary",
        "## Highest-Risk Steps",
        "## Failure Propagation",
        "## Mitigations",
        "",
        "Use short bullets. Keep the answer under 600 words.",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.append(
            f"- {item.get('step_id')} {item.get('title')} | "
            f"in={'; '.join(item.get('context_inputs', [])[:2]) or 'none'} | "
            f"out={'; '.join(item.get('context_outputs', [])[:2]) or 'none'} | "
            f"risk={'; '.join(item.get('risks', [])[:2]) or 'none'}"
        )
    lines += [
        "",
        "Focus on stale context, compounded misidentification, and where a wrong early cue can affect later steps.",
    ]
    return "\n".join(lines)


def _is_simple_agentic_audit(video_context: dict[str, Any]) -> bool:
    """Heuristic: use a smaller reasoning budget for low-branching videos."""
    caption_segments = int(video_context.get("caption_segments", 0) or 0)
    qwen_frames = len(video_context.get("qwen_captions", []) or [])
    ocr_with_text = sum(1 for r in (video_context.get("ocr", []) or []) if r.get("ocr_text"))
    has_unidrive = bool(video_context.get("unidrive_analysis"))
    has_multimodel = bool(video_context.get("multi_model_comparison"))
    map_points = int((video_context.get("map", {}) or {}).get("points", 0) or 0)
    world_clips = int(video_context.get("world_model_clips", 0) or 0)
    return (
        caption_segments <= 3
        and qwen_frames <= 20
        and ocr_with_text <= 10
        and not has_unidrive
        and not has_multimodel
        and map_points <= 20
        and world_clips <= 8
    )


def _agentic_flow_required_sections(simple: bool) -> list[str]:
    if simple:
        return [
            "## Flow Summary",
            "## Highest-Risk Steps",
            "## Failure Propagation",
            "## Mitigations",
        ]
    return [
        "## Flow Summary",
        "## Step-by-Step Agentic Context",
        "## Risk Register",
        "## Highest-Risk Context Failures",
        "## Mitigations",
    ]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking_tokens(text: str) -> str:
    """Remove <think>…</think> blocks emitted by qwen3 / deepseek-r1 thinking models."""
    return _THINK_RE.sub("", text or "").strip()


def _is_valid_agentic_flow_analysis(text: str, *, simple: bool) -> bool:
    """Return True when the reasoning output is usable without a retry."""
    body = _strip_thinking_tokens(text)
    if not body:
        return False
    if len(body) < 120:
        return False
    required = _agentic_flow_required_sections(simple)
    return all(section in body for section in required)


def _reasoning_timeout_for_model(
    model: str,
    api_url: str = "",
    resources: dict[str, Any] | None = None,
) -> float:
    from ...steps.caption import _compute_sidecar_timeout

    # REASONING_TIMEOUT_SEC env/settings hard-override takes priority
    hard = float(getattr(settings, "REASONING_TIMEOUT_SEC", 0))
    if hard > 0:
        return hard
    return _compute_sidecar_timeout(model, api_url or "", resources)


def _fallback_agentic_flow_analysis(video_context: dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        "## Flow Summary",
        "The local full-analysis pipeline accumulates context progressively: frame sampling establishes the timeline, multimodal steps add semantic and geometric evidence, and later reasoning steps consume that evidence to produce higher-level conclusions. The main agentic risk is not a single wrong model output but error carry-over from early observations into later narrative and structured reasoning.",
        "",
        "## Step-by-Step Agentic Context",
    ]
    for item in trace:
        received = ", ".join(item.get("context_inputs", [])) or "no prior context"
        produced = ", ".join(item.get("context_outputs", [])) or "no durable context"
        lines.append(
            f"- **{item.get('step_id')} {item.get('title')}** receives {received}; "
            f"produces {produced}; downstream consumers inherit both its evidence and its errors."
        )
    lines += ["", "## Risk Register"]
    for item in trace:
        risk_text = "; ".join(item.get("risks", [])) or "low direct risk"
        lines.append(f"- **{item.get('step_id')} {item.get('title')}**: {risk_text}.")
    lines += [
        "",
        "## Highest-Risk Context Failures",
        "- Qwen detailed captioning is the most exposed step because it consumes accumulated Florence, ASR, OCR, depth, detection, and prior-Qwen state. One bad upstream cue can shift the frame narrative.",
        "- Final synthesis can convert uncertain intermediate evidence into confident-looking ontology or narrative text if confidence and disagreement are not surfaced explicitly.",
        "- Distillation and fine-tuning can preserve or amplify weak teacher assumptions if retrieval gains are accepted without semantic validation.",
        "",
        "## Mitigations",
        "- Gate downstream prompts with confidence and disagreement summaries rather than only positive evidence.",
        "- Keep per-step provenance in artifacts so a reviewer can trace each claim to its source step.",
        "- Add contradiction checks between captions, OCR, ASR, detections, and final narratives before exporting final conclusions.",
    ]
    return "\n".join(lines)


# -- Step 23: agentic flow artifact -------------------------------------------

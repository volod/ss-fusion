"""Video synthesis and agentic flow report writers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common import (
    _RUNNER_LABEL,
    _log,
    write_markdown_artifact,
)
from ..threat.threat_contradictions import (
    contradiction_signals_for_threat,
    sensor_sources_from_evidence,
    summarize_contradictions,
    support_frame_names,
)


def _normalise_threat_rows(
    local_threat: dict[str, Any] | None,
    policy_decision: dict[str, Any] | None,
    threat_primitives_result: dict[str, Any] | None,
    unidrive_rows: list[dict[str, Any]] | None,
    physical_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    local_threat = local_threat or {}
    policy_decision = policy_decision or {}
    threat_primitives_result = threat_primitives_result or {}
    unidrive_rows = unidrive_rows or []
    physical_state = physical_state or {}

    primitive_by_type = {
        str(p.get("type", "")): p
        for p in (threat_primitives_result.get("primitives") or [])
        if p.get("type")
    }
    rows: list[dict[str, Any]] = []
    for threat in local_threat.get("top_threats") or []:
        threat_type = str(threat.get("type", "unknown"))
        primitive = primitive_by_type.get(threat_type, {})
        evidence = dict(threat.get("evidence") or {})
        evidence_sources = list(
            evidence.get("evidence_sources") or primitive.get("evidence_sources") or []
        )
        contradiction_signals = contradiction_signals_for_threat(
            threat_type, primitive, unidrive_rows, physical_state
        )
        sensor_sources = sensor_sources_from_evidence(evidence_sources)
        disagreeing_sources = [
            str(signal.get("description", "") or "")
            for signal in contradiction_signals
            if signal.get("description")
        ]
        uncertainty = float(
            evidence.get("uncertainty", primitive.get("uncertainty", 0.0)) or 0.0
        )
        rows.append({
            "threat_type": threat_type,
            "score": float(threat.get("score", 0.0) or 0.0),
            "uncertainty": uncertainty,
            "sensor_sources": sensor_sources,
            "disagreeing_sources": disagreeing_sources,
            "contradiction_signals": contradiction_signals,
            "recommended_action": str(
                policy_decision.get(
                    "recommended_action", local_threat.get("recommended_action", "continue")
                )
            ),
            "support_frames": support_frame_names(primitive.get("spatial_support") or []),
            "confidence": max(0.0, 1.0 - uncertainty),
            "evidence_sources": evidence_sources,
        })
    return rows


def _threat_evidence_lines(
    threat_rows: list[dict[str, Any]],
    local_threat: dict[str, Any],
    section_title: str = "Threat Evidence",
) -> list[str]:
    contradiction_summary = summarize_contradictions(threat_rows)
    conflict_patterns = [
        f"{row['pattern']} ({row['count']})"
        for row in contradiction_summary.get("source_pair_conflicts", [])[:4]
    ]
    lines: list[str] = [
        "",
        f"## {section_title}",
        "",
        "| threat_type | score | uncertainty | sensor_sources | disagreeing_sources | recommended_action |",
        "|-------------|-------|-------------|----------------|---------------------|--------------------|",
    ]
    if threat_rows:
        for row in threat_rows:
            lines.append(
                f"| {row['threat_type']} | {row['score']:.3f} | {row['uncertainty']:.3f} | "
                f"{'; '.join(row['sensor_sources']) or 'none'} | "
                f"{'; '.join(row['disagreeing_sources']) or 'none'} | "
                f"{row['recommended_action']} |"
            )
    else:
        lines.append("| — | 0.000 | 0.000 | none | none | continue |")
    lines += [
        "",
        "## Contradiction Metrics",
        "",
        f"- disagreement_count: {int(contradiction_summary.get('disagreement_count', 0))}",
        f"- disagreement_rate: {float(contradiction_summary.get('disagreement_rate', 0.0)):.3f}",
        f"- trust_penalty: {float(local_threat.get('trust_penalty', contradiction_summary.get('trust_penalty', 0.0))):.3f}",
        f"- source_pair_conflicts: {', '.join(conflict_patterns) if conflict_patterns else 'none'}",
        "",
    ]
    return lines


def write_video_synthesis_md(
    output_path: Path,
    video_name: str,
    ontology: dict[str, Any],
    narrative: str,
    elapsed_sec: float,
    model_id: str,
    local_threat: dict[str, Any] | None = None,
    policy_decision: dict[str, Any] | None = None,
    threat_primitives_result: dict[str, Any] | None = None,
    unidrive_rows: list[dict[str, Any]] | None = None,
    physical_state: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = [
        f"# Video Synthesis — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Elapsed: {elapsed_sec:.1f}s",
        "",
    ]
    if ontology:
        lines += ["## Video Ontology", "", "| Field | Value |", "|-------|-------|"]
        for k, v in ontology.items():
            val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            lines.append(f"| {k} | {val.replace('|', '&#124;')} |")
        lines.append("")
    if narrative:
        lines += ["## Video Narrative", "", narrative, ""]
    if local_threat and not local_threat.get("skipped"):
        threat_rows = _normalise_threat_rows(
            local_threat, policy_decision, threat_primitives_result, unidrive_rows, physical_state
        )
        action = (policy_decision or {}).get(
            "recommended_action", local_threat.get("recommended_action", "continue")
        )
        lines += [
            "## Local Threat Assessment",
            "",
            f"- Local threat score: {float(local_threat.get('local_threat_score', 0.0)):.3f}",
            f"- Recommended action: `{action}`",
            f"- Automation confidence: {float(local_threat.get('automation_confidence', 1.0)):.3f}",
            f"- Trust penalty: {float(local_threat.get('trust_penalty', 0.0)):.3f}",
        ]
        lines += _threat_evidence_lines(threat_rows, local_threat)
    lines += [
        "---",
        f"*Produced by {_RUNNER_LABEL} · synthesis step 28 · context from steps 01-27*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_agentic_flow_md(
    output_path: Path,
    video_name: str,
    trace: list[dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
    llm_analysis: str,
    video_context: dict[str, Any] | None = None,
) -> None:
    video_context = video_context or {}
    lines: list[str] = [
        f"# Agentic Flow Trace — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Reasoning model: {model_id}  |  Elapsed: {elapsed_sec:.1f}s",
        "",
        "## Step Trace",
        "",
        "| Step | Status | Context Received | Context Produced | Key Risks |",
        "|------|--------|------------------|------------------|-----------|",
    ]
    for item in trace:
        inputs = "; ".join(item.get("context_inputs", [])[:4]) or "—"
        outputs = "; ".join(item.get("context_outputs", [])[:4]) or "—"
        risks = "; ".join(item.get("risks", [])[:3]) or "—"
        lines.append(
            f"| {item.get('step_id', '?')} {item.get('title', '')} | "
            f"{item.get('status', 'unknown')} | "
            f"{inputs.replace('|', '&#124;')[:180]} | "
            f"{outputs.replace('|', '&#124;')[:180]} | "
            f"{risks.replace('|', '&#124;')[:180]} |"
        )

    threat_rows = _normalise_threat_rows(
        video_context.get("local_threat", {}),
        video_context.get("policy_decision", {}),
        video_context.get("threat_primitives", {}),
        video_context.get("unidrive_analysis", []),
        video_context.get("physical_state", {}),
    )
    if threat_rows:
        local_threat_ctx = video_context.get("local_threat", {})
        lines += _threat_evidence_lines(threat_rows, local_threat_ctx, section_title="Threat Evidence")
        contradiction_summary = summarize_contradictions(threat_rows)
        lines += ["## Threat Provenance", ""]
        for row in threat_rows:
            frames = ", ".join(row["support_frames"]) or "none"
            evidence = ", ".join(row["evidence_sources"]) or "none"
            disagreements = "; ".join(row["disagreeing_sources"]) or "none"
            lines.append(
                f"- **{row['threat_type']}**: frames={frames}; "
                f"confidence={row['confidence']:.3f}; evidence_sources={evidence}; "
                f"sensor_sources={', '.join(row['sensor_sources']) or 'none'}; "
                f"disagreeing_sources={disagreements}; "
                f"recommended_action={row['recommended_action']}."
            )

    lines += [
        "",
        "## Agentic Analysis",
        "",
        llm_analysis.strip() if llm_analysis.strip() else "Reasoning analysis unavailable.",
        "",
        "---",
        f"*Produced by {_RUNNER_LABEL} · final agentic audit step*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)

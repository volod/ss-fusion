"""Multimodal feature and VLM analysis report writers."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..common import (
    _RUNNER_LABEL,
    _analyze_caption_sequence,
    _log,
    write_markdown_artifact,
)


def _diff_structured_caption(prev: dict[str, Any], curr: dict[str, Any]) -> str:
    """Return a short string describing what changed between two Qwen structured dicts."""
    changes: list[str] = []

    prev_surface = prev.get("road_surface", "unknown")
    curr_surface = curr.get("road_surface", "unknown")
    if prev_surface != curr_surface:
        changes.append(f"road: {prev_surface}→{curr_surface}")

    prev_cond = prev.get("road_condition", "unknown")
    curr_cond = curr.get("road_condition", "unknown")
    if prev_cond != curr_cond:
        changes.append(f"condition: {prev_cond}→{curr_cond}")

    def _vehicle_signature(groups: list) -> dict[str, int]:
        sig: dict[str, int] = {}
        for g in groups or []:
            vtype = g.get("type", "other")
            sig[vtype] = sig.get(vtype, 0) + int(g.get("count") or 1)
        return sig

    prev_sig = _vehicle_signature(prev.get("vehicle_groups", []))
    curr_sig = _vehicle_signature(curr.get("vehicle_groups", []))
    if prev_sig != curr_sig:
        if not prev_sig and curr_sig:
            changes.append("vehicles appeared")
        elif prev_sig and not curr_sig:
            changes.append("vehicles left")
        else:
            for vt in sorted(set(prev_sig) | set(curr_sig)):
                p, c = prev_sig.get(vt, 0), curr_sig.get(vt, 0)
                if p != c:
                    changes.append(f"{vt}: {p}→{c}")

    return "; ".join(changes) if changes else ""


def write_multimodal_md(
    output_path: Path,
    video_name: str,
    asr_result: dict[str, Any],
    ocr_result: dict[str, Any],
    depth_result: dict[str, Any],
    det_result: dict[str, Any],
    world_result: dict[str, Any],
    state_fusion_result: dict[str, Any],
    qwen_result: dict[str, Any],
    unidrive_result: dict[str, Any],
) -> None:
    ok = lambda r: "[ok]" if not r.get("skipped") else "—"  # noqa: E731
    lines = [
        f"# Multimodal Features — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Step | Status | Detail |",
        "|------|--------|--------|",
        f"| ASR (Whisper) | {ok(asr_result)} | {asr_result.get('covered_frames', 0)} frames with subtitles |",
        f"| OCR | {ok(ocr_result)} | {ocr_result.get('non_empty', 0)} frames with text |",
        f"| Depth | {ok(depth_result)} | {depth_result.get('ok_count', 0)} frames estimated |",
        f"| Detection | {ok(det_result)} | {det_result.get('total_objects', 0)} objects detected |",
        f"| World Model | {ok(world_result)} | {world_result.get('ok_count', 0)} clips processed |",
        f"| Platform-state fusion | {ok(state_fusion_result)} | {state_fusion_result.get('summary', {}).get('frame_count', 0)} posterior samples |",
        f"| Qwen VLM captioning | {ok(qwen_result)} | {qwen_result.get('ok_count', 0)} frames captioned |",
        f"| UniDriveVLA expert analysis | {ok(unidrive_result)} | {unidrive_result.get('ok_count', 0)} frames analysed |",
        "",
    ]
    if not ocr_result.get("skipped"):
        lines += ["## OCR — Sample Text Extractions", ""]
        ocr_rows = [r for r in ocr_result.get("ocr_results", []) if r.get("ocr_text")][:10]
        if ocr_rows:
            lines += ["| t (s) | Extracted Text |", "|-------|----------------|"]
            for r in ocr_rows:
                txt = (r.get("ocr_text") or "").replace("|", "\\|")[:120]
                lines.append(f"| {r['t_sec']:.1f} | {txt} |")
        lines.append("")
    if not det_result.get("skipped"):
        lines += ["## Detection — Objects Found", ""]
        det_rows = [r for r in det_result.get("detection_results", []) if r.get("detections")][:10]
        if det_rows:
            lines += ["| t (s) | Detections |", "|-------|------------|"]
            for r in det_rows:
                objs = ", ".join(
                    f"{d['label']} ({d['confidence']:.2f})" for d in r["detections"][:5]
                )
                lines.append(f"| {r['t_sec']:.1f} | {objs} |")
        lines.append("")
    if not depth_result.get("skipped"):
        lines += ["## Depth — Percentile Summary (sample)", ""]
        depth_rows = [r for r in depth_result.get("depth_results", []) if r.get("depth")][:5]
        if depth_rows:
            lines += ["| t (s) | p10 | p25 | p50 | p75 | p90 |", "|-------|-----|-----|-----|-----|-----|"]
            for r in depth_rows:
                p = r["depth"].get("percentiles", [0] * 5)
                lines.append(
                    f"| {r['t_sec']:.1f} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {p[4]:.3f} |"
                )
        lines.append("")
    if not state_fusion_result.get("skipped"):
        lines += ["## Platform-State Fusion — Posterior Summary", ""]
        summary = state_fusion_result.get("summary", {})
        final_state = summary.get("final_state") or {}
        pos = final_state.get("position_enu_m") or {}
        vel = final_state.get("velocity_enu_mps") or {}
        lines += [
            f"- Telemetry sources: {', '.join(summary.get('telemetry_sources', [])) or 'none'}",
            f"- Mean covariance trace: {summary.get('mean_covariance_trace')!s}",
            f"- Final covariance trace: {summary.get('final_covariance_trace')!s}",
            f"- Final ENU position: ({pos.get('x', 0.0):.2f}, {pos.get('y', 0.0):.2f}, {pos.get('z', 0.0):.2f}) m",
            f"- Final ENU velocity: ({vel.get('x', 0.0):.2f}, {vel.get('y', 0.0):.2f}, {vel.get('z', 0.0):.2f}) m/s",
            "",
        ]
    lines += ["---", f"*Produced by {_RUNNER_LABEL} · multimodal steps M–S*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_state_fusion_md(output_path: Path, video_name: str, fusion_result: Any) -> None:
    summary = fusion_result.summary()
    lines = [
        f"# Platform-State Fusion — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Reason: {summary.get('reason', '') or 'n/a'}",
        f"- Source: {summary.get('source', 'n/a')}",
        f"- Telemetry sources: {', '.join(summary.get('telemetry_sources', [])) or 'none'}",
        f"- Posterior samples: {summary.get('frame_count', 0)}",
        f"- Mean covariance trace: {summary.get('mean_covariance_trace')!s}",
        f"- Final covariance trace: {summary.get('final_covariance_trace')!s}",
        "",
        "## Measurement Counts",
        "",
        "| Kind | Count |",
        "|------|-------|",
    ]
    for kind, count in sorted((summary.get("measurement_counts") or {}).items()):
        lines.append(f"| {kind} | {count} |")
    if not (summary.get("measurement_counts") or {}):
        lines.append("| — | 0 |")
    samples = fusion_result.posterior_samples[:12]
    lines += [
        "",
        "## Posterior Samples (first 12)",
        "",
        "| t (s) | x | y | z | vx | vy | vz | Cov Trace | Quality |",
        "|-------|---|---|---|----|----|----|-----------|---------|",
    ]
    for s in samples:
        pos = s.position_enu_m
        vel = s.velocity_enu_mps
        lines.append(
            f"| {s.t_sec:.1f} | {pos['x']:.2f} | {pos['y']:.2f} | {pos['z']:.2f} | "
            f"{vel['x']:.2f} | {vel['y']:.2f} | {vel['z']:.2f} | "
            f"{s.covariance_trace:.3f} | {s.quality} |"
        )
    if not samples:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · probabilistic platform-state fusion MVP*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_detailed_captions_md(
    output_path: Path,
    video_name: str,
    results: list[dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
) -> None:
    ok = sum(
        1 for r in results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    )
    parse_errors = sum(1 for r in results if r.get("parse_error"))
    unavailable = sum(1 for r in results if r.get("service_unavailable"))

    text_results = [
        {**r, "caption": r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""}
        for r in results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    ]
    enriched_valid = _analyze_caption_sequence(text_results) if text_results else []
    enriched_index = {
        (str(r.get("frame_path", "")), float(r.get("t_sec", 0.0))): r for r in enriched_valid
    }

    segments: list[dict[str, Any]] = []
    for r in enriched_valid:
        if r["is_new_segment"]:
            segments.append({
                "segment_id": r["segment_id"],
                "start_t": r["t_sec"],
                "end_t": r["t_sec"],
                "frame_count": 1,
                "scene_summary": r.get("scene_summary") or r.get("caption") or "",
                "road_surface": r.get("road_surface", ""),
                "road_condition": r.get("road_condition", ""),
                "vehicle_groups": r.get("vehicle_groups", []),
            })
        elif segments:
            segments[-1]["end_t"] = r["t_sec"]
            segments[-1]["frame_count"] += 1

    n_unchanged = sum(1 for r in enriched_valid if not r["is_new_segment"])

    lines = [
        f"# Detailed Scene Captions — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Frames processed: {ok}/{len(results)}"
        f"  |  Unique scenes: {len(segments)}  |  Repeated: {n_unchanged}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"Structured parse errors: {parse_errors}/{len(results)}  |  Service unavailable: {unavailable}",
        "",
        "## Scene Timeline",
        "",
        "| # | Start (s) | End (s) | Frames | Road | Condition | Vehicles | Summary |",
        "|---|-----------|---------|--------|------|-----------|----------|---------|",
    ]
    if not segments:
        lines += [
            "",
            "_No valid structured Qwen outputs were parsed for this run. Per-frame rows below may contain parse-error markers only._",
            "",
        ]
    for seg in segments:
        vg = seg.get("vehicle_groups") or []
        v_str = "; ".join(f"{g.get('count', 1)}×{g.get('type', '?')}" for g in vg) if vg else "none"
        summary = (seg.get("scene_summary") or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {seg['segment_id'] + 1} | {seg['start_t']:.1f} | {seg['end_t']:.1f}"
            f" | {seg['frame_count']} | {seg.get('road_surface') or '—'}"
            f" | {seg.get('road_condition') or '—'} | {v_str} | {summary} |"
        )

    lines += [
        "",
        "## Per-Frame Analysis",
        "",
        "The **Δ Changes** column shows structured fields that differ from the previous frame.",
        "Frames with no changes are marked *unchanged*.",
        "",
        "| Frame | t (s) | Seg | Δ Changes | Caption / Scene Facts | Audio Context |",
        "|-------|-------|-----|-----------|----------------------|---------------|",
    ]

    prev_structured: dict[str, Any] = {}
    for r in results:
        fp = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t = r.get("t_sec", 0.0)
        subtitle = (r.get("subtitle_text") or "").replace("|", "\\|")[:60]
        enriched_row = enriched_index.get((str(fp), float(t)))
        seg = str(enriched_row["segment_id"] + 1) if enriched_row else "—"

        if r.get("service_unavailable"):
            caption, delta = "*sidecar unavailable*", "—"
        elif r.get("parse_error"):
            raw = str(r.get("raw", "") or "").replace("|", "\\|").strip()
            caption = f"*parse error*{(' ' + raw[:160]) if raw else ''}"
            delta = "—"
        elif r.get("skipped"):
            caption, delta = "*skipped*", "—"
        else:
            delta = _diff_structured_caption(prev_structured, r) if prev_structured else ""
            delta = delta.replace("|", "\\|") if delta else ("—" if prev_structured else "first")
            facts = r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""
            if not facts:
                parts = [
                    f"{k}: {v}" for k, v in r.items()
                    if k not in (
                        "frame_path", "t_sec", "subtitle_text", "ocr_text", "segment_id",
                        "is_new_segment", "similarity", "segment_start_t", "caption",
                    ) and v
                ]
                facts = "; ".join(parts[:4])
            caption = str(facts).replace("|", "\\|")[:200]
            if enriched_row and not enriched_row["is_new_segment"]:
                caption = f"*unchanged* {caption}"
            if not r.get("parse_error"):
                prev_structured = r

        lines.append(f"| `{name}` | {t:.1f} | {seg} | {delta} | {caption} | {subtitle} |")

    lines += [
        "",
        "---",
        f"*Produced by {_RUNNER_LABEL} · Qwen VLM step 12 · ASR subtitle context injected where available*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_unidrive_analysis_md(
    output_path: Path,
    video_name: str,
    results: list[dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
) -> None:
    ok = sum(1 for r in results if not r.get("service_unavailable") and not r.get("parse_error"))
    lines = [
        f"# UniDriveVLA Expert Analysis — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Frames processed: {ok}/{len(results)}",
        f"Elapsed: {elapsed_sec:.1f}s",
        "",
        "| t (s) | Risk | Drivable | Expert Agreement | Understanding | Planning |",
        "|-------|------|----------|------------------|---------------|----------|",
    ]
    for r in results:
        t = r.get("t_sec", 0.0)
        if r.get("service_unavailable"):
            lines.append(f"| {t:.1f} | — | — | — | *service unavailable* | — |")
            continue
        if r.get("parse_error"):
            lines.append(f"| {t:.1f} | — | — | — | *parse error* | — |")
            continue
        u = r.get("understanding", {}) or {}
        p = r.get("perception", {}) or {}
        plan = r.get("planning", {}) or {}
        moe = r.get("mixture_of_experts", {}) or {}
        understanding = (u.get("scene_summary", "") or "").replace("|", "\\|")[:70]
        planning = (plan.get("recommended_action", "") or "").replace("|", "\\|")[:70]
        lines.append(
            f"| {t:.1f} | {u.get('risk_level', 'unknown')} | "
            f"{p.get('drivable_area', 'unknown')} | {moe.get('expert_agreement', 'unknown')} | "
            f"{understanding} | {planning} |"
        )
    lines += ["", "## Mixture-of-Experts Consensus", ""]
    for r in results[:12]:
        moe = r.get("mixture_of_experts", {}) or {}
        consensus = (moe.get("consensus_summary", "") or "").strip()
        if not consensus:
            continue
        disagreements = moe.get("disagreement_points", []) or []
        dis_str = "; ".join(disagreements[:3]) if disagreements else "none"
        lines.append(
            f"- t={r.get('t_sec', 0.0):.1f}s: {consensus} "
            f"(agreement={moe.get('expert_agreement', 'unknown')}; disagreements: {dis_str})"
        )
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · UniDriveVLA step 13*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_multi_model_comparison_md(
    output_path: Path,
    video_name: str,
    gemma_result: dict[str, Any],
    qwen_result: dict[str, Any],
    unidrive_result: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from ..common import _jaccard

    qwen_rows = [
        r for r in qwen_result.get("results", [])
        if not r.get("service_unavailable") and not r.get("parse_error")
    ]
    uni_rows = [
        r for r in unidrive_result.get("results", [])
        if not r.get("service_unavailable") and not r.get("parse_error")
    ]

    def _nearest(rows: list[dict[str, Any]], t_sec: float) -> dict[str, Any] | None:
        if not rows:
            return None
        return min(rows, key=lambda r: abs(float(r.get("t_sec", 0.0)) - t_sec))

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for u in uni_rows:
        q = _nearest(qwen_rows, float(u.get("t_sec", 0.0)))
        if q is not None and abs(float(u.get("t_sec", 0.0)) - float(q.get("t_sec", 0.0))) <= 2.0:
            pairs.append((q, u))

    agreement_scores: list[float] = []
    example_rows: list[tuple[float, str, str, str, str]] = []
    for q, u in pairs[:10]:
        q_summary = str(q.get("scene_summary") or q.get("caption") or "")
        u_under = u.get("understanding", {}) or {}
        u_moe = u.get("mixture_of_experts", {}) or {}
        u_summary = str(u_under.get("scene_summary", "") or "")
        moe_summary = str(u_moe.get("consensus_summary", "") or "")
        agreement_scores.append(_jaccard(q_summary, u_summary or moe_summary))
        example_rows.append((
            float(u.get("t_sec", 0.0)), q_summary, u_summary,
            moe_summary, str(u_moe.get("expert_agreement", "unknown") or "unknown"),
        ))

    mean_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0
    clf = (gemma_result.get("task_results", {}) or {}).get("scene_classification", {}) or {}
    gemma_scene = next(iter((clf.get("category_distribution") or {}).keys()), "")

    risk_levels = [((r.get("understanding") or {}).get("risk_level", "unknown")) for r in uni_rows]
    agreement_levels = [((r.get("mixture_of_experts") or {}).get("expert_agreement", "unknown")) for r in uni_rows]

    lines = [
        f"# Multi-Model Comparison — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Coverage",
        "",
        "| Model family | Frames analysed | Primary output |",
        "|-------------|-----------------|----------------|",
        f"| Gemma | {gemma_result.get('n_frames', 0)} | scene classification, clustering, cross-model probes |",
        f"| Qwen | {qwen_result.get('ok_count', 0)} | structured per-frame scene facts |",
        f"| UniDriveVLA | {len(uni_rows)} | understanding/perception/planning + MoE consensus |",
        "",
        "## Cross-Model Signals",
        "",
        f"- Gemma dominant scene category: `{gemma_scene or 'unknown'}`",
        f"- Qwen ↔ UniDrive scene-summary token agreement: {mean_agreement:.3f} across {len(agreement_scores)} matched frames",
        f"- UniDrive risk profile: low={sum(1 for v in risk_levels if v == 'low')}, medium={sum(1 for v in risk_levels if v == 'medium')}, high={sum(1 for v in risk_levels if v == 'high')}",
        f"- UniDrive expert agreement: high={sum(1 for v in agreement_levels if v == 'high')}, medium={sum(1 for v in agreement_levels if v == 'medium')}, low={sum(1 for v in agreement_levels if v == 'low')}",
        "",
        "## Matched Examples",
        "",
        "| t (s) | Qwen summary | UniDrive understanding | UniDrive MoE consensus | Expert agreement |",
        "|-------|--------------|------------------------|------------------------|------------------|",
    ]
    for t_sec, q_sum, u_sum, moe_sum, expert_agreement in example_rows:
        lines.append(
            f"| {t_sec:.1f} | {q_sum.replace('|', chr(92)+'|')[:60]} | "
            f"{u_sum.replace('|', chr(92)+'|')[:60]} | "
            f"{moe_sum.replace('|', chr(92)+'|')[:60]} | {expert_agreement} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "- Qwen is the structured scene-facts baseline."
            if qwen_rows
            else "- Qwen produced no valid structured rows in this run; treat UniDrive as the only usable structured VLM output."
        ),
        "- UniDrive adds explicit understanding, perception, and planning experts.",
        "- The UniDrive MoE consensus field is the best single input for downstream synthesis because it preserves both consensus and disagreement.",
        "",
        "---",
        f"*Produced by {_RUNNER_LABEL} · multi-model comparison step 21*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)
    return {
        "matched_frames": len(agreement_scores),
        "mean_qwen_unidrive_agreement": mean_agreement,
        "high_risk_frames": sum(1 for v in risk_levels if v == "high"),
    }

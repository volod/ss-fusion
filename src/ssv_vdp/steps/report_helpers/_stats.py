"""Run statistics: step timing labels, formatters, final stats MD, print_run_stats."""

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common import (
    _log,
    write_markdown_artifact,
)

# (timing_key, step_label, computation_type) — ordered by execution sequence.
_STEP_LABELS: list[tuple[str, str, str]] = [
    ("A_extract",         "01 Ingest: Frame extraction",           "I/O"),
    ("B_index",           "02 Ingest: Vector indexing",            "GPU embed"),
    ("J_gemma",           "03 Analyze: Gemma multimodal",          "LLM API"),
    ("L_caption",         "04 Analyze: Florence captions",         "GPU vision"),
    ("L_seg_caps",        "04b Analyze: Gemma segment diffs",      "LLM API"),
    ("M_asr",             "05 Analyze: ASR transcription",         "GPU speech"),
    ("N_ocr",             "06 Analyze: OCR text extraction",       "LLM API"),
    ("O_depth",           "07 Analyze: Depth estimation",          "GPU vision"),
    ("P_detection",       "08 Analyze: Object detection",          "GPU vision"),
    ("P2_yolo_sam",       "09 Analyze: YOLO+SAM detection",        "GPU vision"),
    ("P3_gemma_tracking", "10 Analyze: Gemma directed tracking",   "LLM API+GPU"),
    ("Q_world",           "11 Analyze: World model embeddings",    "GPU vision"),
    ("R_qwen",            "12 Analyze: Qwen detailed captions",    "LLM API"),
    ("S_unidrive",        "13 Analyze: UniDriveVLA expert",        "LLM API"),
    ("S_scenetok",        "14 Analyze: SceneTok encoder+seg",      "GPU vision"),
    ("C_base_search",     "15 Eval: Base search test",             "GPU embed"),
    ("I_3dmap",           "16 Map: SfM + Gaussian Splat",          "GPU 3D"),
    ("PS_physical_state", "17 Analyze: Physical scene state",      "CPU fusion"),
    ("PS_field_state",    "18 Analyze: Environmental field state", "CPU fusion"),
    ("PS_threat_primitives", "19 Analyze: Threat primitives",      "CPU fusion"),
    ("D_finetune",        "21 Adapt: SSL DINOv3 fine-tune",        "GPU train"),
    ("E_distill",         "22 Adapt: Knowledge distillation",      "GPU train"),
    ("E_distill_stage2",  "23 Adapt: Stage 2 distillation",        "GPU train"),
    ("F_export",          "24 Export: ONNX + gallery",             "CPU"),
    ("G_ft_search",       "25 Eval: Fine-tuned search test",       "GPU embed"),
    ("H_compare",         "26 Eval: Model comparison",             "GPU embed"),
    ("T_multimodel",      "27 Audit: Multi-model comparison",      "GPU vision"),
    ("PS_local_threat",   "28 Analyze: Local threat inference",    "CPU fusion"),
    ("PS_policy",         "29 Decide: Action policy",              "CPU policy"),
    ("Z_synthesis",       "30 Synthesize: Ontology+narrative",     "LLM API"),
    ("AA_agentic",        "31 Audit: Agentic flow",                "LLM API"),
    ("AC_drone_detection","32 Train: Drone detection",             "GPU train"),
    ("AC_drone_audio",    "33 Train: Drone audio",                 "GPU train"),
    ("AC_drau_eval",      "34 Eval: drau range",                   "CPU analysis"),
    ("AB_model_advisor",  "35 Optimize: Model/run advisor",        "CPU analysis"),
]


def _fmt_sec(sec: float) -> str:
    if math.isnan(sec) or sec < 0:
        return "—"
    if sec >= 3600:
        h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if sec >= 60:
        m = int(sec // 60)
        return f"{m}m {sec % 60:04.1f}s"
    return f"{sec:.1f}s"


def write_final_stats_md(
    output_path: Path,
    per_video: list[dict[str, Any]],
    total_elapsed: float,
) -> None:
    step_sum = sum(sum(v.get("timings", {}).values()) for v in per_video)
    concurrent_overlap = max(0.0, step_sum - total_elapsed)
    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]

    header = "| Step | Type | " + " | ".join(names) + " | Total |"
    sep = "|------|------|" + "|".join(["-------"] * len(names)) + "|-------|"
    lines = [
        "# Local Full-Analysis Pipeline — Final Statistics",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total elapsed: {total_elapsed:.1f}s",
        f"Step-time sum: {step_sum:.1f}s",
        f"Concurrent overlap: {concurrent_overlap:.1f}s",
        f"Videos processed: {len(per_video)}",
        "",
        "## Step Timing",
        "",
        "Step totals are per-step durations. They may exceed elapsed time because the 3D map step can run in the background.",
        "",
        header, sep,
    ]
    for key, label, comp_type in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        if total_step == 0 and key not in ("A_extract", "B_index"):
            continue
        dur_cells = " | ".join(_fmt_sec(s) for s in vals)
        lines.append(f"| {label} | {comp_type} | {dur_cells} | **{_fmt_sec(total_step)}** |")

    lines += [
        "",
        "## Per-Video Summary",
        "",
        "| Video | Frames | Index (s) | Finetune loss | Distill loss | SfM poses | Ckpt (MB) |",
        "|-------|--------|-----------|---------------|--------------|-----------|-----------|",
    ]
    for i, v in enumerate(per_video):
        distill_loss = v.get("distill_loss", float("nan"))
        distill_str = f"{distill_loss:.4f}" if not math.isnan(distill_loss) else "skipped"
        lines.append(
            f"| {v.get('name', f'video{i}')} | {v.get('frames', 0)} | "
            f"{v.get('index_sec', 0):.1f} | "
            f"{v.get('best_loss', float('nan')):.4f} | "
            f"{distill_str} | "
            f"{v.get('sfm_poses', 0)} | "
            f"{v.get('ckpt_mb', 0):.1f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"Each video produced these outputs under `{output_path.parent}/{{video_name}}/`:",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `frames_metadata.json` | Extracted frame paths, timestamps, fps |",
        "| `base_search.md` | Nearest-neighbour results with base DINOv3 |",
        "| `scene_captions.md` | Per-frame Florence-2 captions (confidence scores) |",
        "| `finetune_stats.md` | SSL fine-tuning loss curve + config |",
        "| `finetuned_search.md` | Nearest-neighbour results with fine-tuned DINOv3 |",
        "| `comparison.md` | Base vs fine-tuned stats + video description |",
        "| `checkpoints/dino_ssl_best.pt` | Fine-tuned teacher backbone (PyTorch) |",
        "| `checkpoints/student_best.pt` | Distilled student backbone (PyTorch, ~22M params) |",
        "| `distill_stats.md` | Distillation loss curve + architecture notes |",
        "| `edge_models/dino_local.onnx` | ONNX export (student when distilled, teacher otherwise) |",
        "| `edge_models/gallery.npz` | Embedding gallery for 1-NN classification |",
        "| `asr_subtitles.md` | Whisper ASR segments + per-frame subtitle coverage (step 05) |",
        "| `state_fusion.md` | Probabilistic platform-state posterior summary and covariance samples |",
        "| `state_fusion.json` | Raw local probabilistic platform-state posterior payload |",
        "| `physical_state_summary.json` | Clip-level physical state summary: pose confidence, occupancy, object velocity, free-space estimate |",
        "| `field_state_summary.json` | Coarse local environmental field summary for visibility, RF interference, and thermal anomaly evidence |",
        "| `threat_primitives.json` | Structured evidence-gated threat primitives with score, uncertainty, support, and persistence |",
        "| `local_threat_assessment.json` | Clip-level threat estimate, top threats, automation confidence, and contradiction metrics (step 26) |",
        "| `policy_decision.json` | Separated action-policy output with recommended action, rationale, and sensor-health context (step 27) |",
        "| `multimodal_features.md` | OCR text, depth percentiles, detections, world model (steps 06-11) |",
        "| `detailed_captions.md` | Qwen VLM detailed per-frame scene captions with ASR context (step 12) |",
        "| `unidrive_analysis.md` | UniDriveVLA understanding, perception, planning, and MoE consensus (step 13) |",
        "| `multi_model_comparison.md` | Gemma vs Qwen vs UniDriveVLA comparison and MoE agreement summary (step 24) |",
        "| `video_synthesis.md` | LLM video ontology + fine-grained narrative (step 28) |",
        "| `agentic_flow.md` | Step-by-step agentic context trace, risk analysis, and context-propagation audit (step 29) |",
        "| `video_ontology.json` | Structured ontology JSON (domain, environment, activities, objects) |",
        "| `3d_map/sparse_map.npz` | 3D point cloud (from SfM or PCA fallback) |",
        "| `3d_map/map_stats.json` | Point count, SfM pose count, scene count |",
        "| `3d_map/map_quality_advisor.json` | Measured mapping-quality diagnostics and readiness score |",
        "| `3d_map/map_quality_advisor.md` | Capture guidance and flight-plan recommendations for higher-quality maps |",
        "",
        f"Run-level artifacts are written under `{output_path.parent}/`:",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `model_run_advisor.json` | Post-run model, environment, and rerun recommendations for the current hardware |",
        "| `model_run_advisor.md` | Human-readable model/run optimization plan based on warnings and analytics |",
        "",
        "---",
        "*Run `python main.py --mode local --help` for all options.*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("[ok] Final stats written to %s", output_path)


# ---------------------------------------------------------------------------
# Analytics summary formatters (used by print_run_stats)
# ---------------------------------------------------------------------------


def _fmt_analytics_coverage(summary: dict[str, Any]) -> str:
    rh = summary.get("run_health", {}) or {}
    text = (
        f"{100.0 * float(rh.get('florence_caption_coverage', 0.0)):.0f}/"
        f"{100.0 * float(rh.get('qwen_caption_coverage', 0.0)):.0f}/"
        f"{100.0 * float(rh.get('asr_coverage', 0.0)):.0f}/"
        f"{100.0 * float(rh.get('ocr_coverage', 0.0)):.0f}%"
    )
    parse_errors = int(rh.get("qwen_parse_error_count", 0) or 0)
    if parse_errors > 0:
        text += f" (Qwen parse={parse_errors})"
    return text


def _fmt_analytics_detections(summary: dict[str, Any]) -> str:
    ds = summary.get("detection_stats", {}) or {}
    total = ds.get("total_objects")
    mean_per_frame = ds.get("mean_per_frame")
    if total in (None, ""):
        return "—"
    if mean_per_frame in (None, ""):
        return str(total)
    return f"{int(total)} ({float(mean_per_frame):.1f}/fr)"


def _fmt_analytics_temporal(summary: dict[str, Any]) -> str:
    ts = summary.get("temporal_stats", {}) or {}
    mean_surprise = ts.get("mean_surprise")
    peak_frames = ts.get("peak_frames", []) or []
    if mean_surprise in (None, ""):
        return "—"
    return f"{float(mean_surprise):.3f} / {len(peak_frames)} peaks"


def _fmt_analytics_world_tracking(summary: dict[str, Any]) -> str:
    rh = summary.get("run_health", {}) or {}
    tr = summary.get("tracking_stats", {}) or {}
    world = "ok" if rh.get("world_model_ok") else "degraded"
    tracks = tr.get("unique_track_ids")
    if tracks in (None, ""):
        return world
    return f"{world} / {int(tracks)} tracks"


def _fmt_analytics_map(summary: dict[str, Any]) -> str:
    ms = summary.get("map_stats", {}) or {}
    if not ms:
        return "—"
    quality = "degraded" if ms.get("degraded") else "ok"
    points = int(ms.get("points", 0) or 0)
    poses = int(ms.get("poses", 0) or 0)
    sfm_poses = int(ms.get("sfm_poses", poses) or poses)
    anchors = int(ms.get("frame_anchor_count", poses) or poses)
    if anchors != sfm_poses:
        return f"{quality} ({points}p/{sfm_poses} SfM, {anchors} anchors)"
    return f"{quality} ({points}p/{poses} poses)"


def _fmt_analytics_warnings(summary: dict[str, Any]) -> str:
    warnings = (summary.get("run_health", {}) or {}).get("warnings", []) or []
    if not warnings:
        return "—"
    text = ", ".join(str(item) for item in warnings[:2])
    if len(warnings) > 2:
        text += f" +{len(warnings) - 2}"
    return text


def print_run_stats(
    per_video: list[dict[str, Any]],
    total_elapsed: float,
    init_elapsed: float,
    device: str,
) -> None:
    from ..common import _banner

    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    LABEL_W = max(34, max((len(label) for _, label, _ in _STEP_LABELS), default=34) + 1)
    TYPE_W = max(14, max((len(t) for _, _, t in _STEP_LABELS), default=14) + 1)
    DUR_W = max(9, max((len(n) for n in names), default=9) + 1, len(_fmt_sec(total_elapsed)) + 1)
    n_vids = len(per_video)
    W = LABEL_W + TYPE_W + DUR_W * (n_vids + 1) + 4
    SEP = "-" * W

    def _fit(value: str, width: int) -> str:
        return value[:width] if len(value) <= width else value[:max(0, width - 3)] + "..."

    def _row(label: str, comp_type: str, *dur_cols: str) -> str:
        row = f"  {label:<{LABEL_W}} {comp_type:<{TYPE_W}}"
        for c in dur_cols:
            row += f"{_fit(c, DUR_W):>{DUR_W}}"
        return row

    _banner("RUN STATISTICS")
    _log.info("  Device       : %s", device.upper())
    _log.info("  Videos       : %d", len(per_video))
    total_frames = sum(v.get("frames", 0) for v in per_video)
    total_duration = sum(v.get("duration_sec", 0.0) for v in per_video)
    _log.info("  Total frames : %d  (%.1f min of video)", total_frames, total_duration / 60)
    _log.info("  Total runtime: %s", _fmt_sec(total_elapsed))
    _log.info("")
    _log.info("  STEP TIMING  (wall-clock per step)")
    _log.info("  " + SEP)
    _log.info(_row("Step", "Type", *(names + ["TOTAL"])))
    _log.info("  " + SEP)

    by_type: dict[str, float] = {}
    col_totals = [0.0] * n_vids
    grand_total = 0.0
    _always_show = {"A_extract", "B_index", "S_scenetok", "S_unidrive"}
    for key, label, comp_type in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        if total_step > 0 or key in _always_show:
            if total_step == 0 and key in _always_show:
                _log.info(_row(label + " (skipped)", comp_type, *["—"] * n_vids, "—"))
            else:
                _log.info(_row(label, comp_type, *[_fmt_sec(s) for s in vals], _fmt_sec(total_step)))
            for i, s in enumerate(vals):
                col_totals[i] += s
            grand_total += total_step
        by_type[comp_type] = by_type.get(comp_type, 0.0) + total_step

    _log.info("  " + SEP)
    _log.info(_row("TOTAL", "", *[_fmt_sec(s) for s in col_totals], _fmt_sec(grand_total)))
    _log.info("  " + SEP)

    pipeline_per_video = [v.get("pipeline_sec", 0.0) for v in per_video]
    _log.info(_row("Pipeline (steps sum)", "", *[_fmt_sec(s) for s in pipeline_per_video], _fmt_sec(sum(pipeline_per_video))))
    pipeline_sum = sum(pipeline_per_video)
    overlap_adjustment = max(0.0, pipeline_sum + init_elapsed - total_elapsed)
    overhead = total_elapsed - pipeline_sum - init_elapsed + overlap_adjustment
    _log.info(_row("Model initialisation", "", _fmt_sec(init_elapsed), *([""] * (n_vids - 1)), ""))
    if overlap_adjustment > 0:
        _log.info(_row("Concurrent overlap adjustment", "", *([""] * n_vids), f"-{_fmt_sec(overlap_adjustment)}"))
    _log.info(_row("Overhead (I/O, viewer, etc.)", "", *([""] * n_vids), _fmt_sec(max(0.0, overhead))))
    _log.info(_row("WALL CLOCK TOTAL", "", *([""] * n_vids), _fmt_sec(total_elapsed)))

    _log.info("")
    _log.info("  COMPUTATION TYPE BREAKDOWN  (pipeline steps only)")
    _log.info("  " + "-" * (TYPE_W + DUR_W + LABEL_W + 2))
    for ct in ["I/O", "GPU embed", "GPU vision", "GPU speech", "GPU 3D", "GPU train", "CPU", "LLM API", "LLM API+GPU"]:
        t = by_type.get(ct, 0.0)
        if t > 0:
            pct = 100.0 * t / max(sum(by_type.values()), 1e-9)
            _log.info("  %-14s  %s  (%4.1f%%)", ct, _fmt_sec(t), pct)

    _log.info("")
    _log.info("  THROUGHPUT")
    _log.info("  " + SEP[: W - 2])
    for v in per_video:
        t_extract = v.get("timings", {}).get("A_extract", 0.0) or 1e-9
        t_index = v.get("timings", {}).get("B_index", 0.0) or 1e-9
        frames = v.get("frames", 0)
        _log.info("  %-26s  extract: %5.1f fr/s   index: %5.1f fr/s", v.get("name", "?"), frames / t_extract, frames / t_index)

    _log.info("")
    _log.info("  MODEL METRICS")
    _log.info("  " + SEP[: W - 2])
    _log.info(_row("Metric", *names))
    _log.info("  " + SEP[: W - 2])
    _log.info(_row("SSL finetune loss", *[f"{v.get('best_loss', float('nan')):.4f}" for v in per_video]))
    _log.info(_row(
        "Distill loss",
        *[f"{v.get('distill_loss', float('nan')):.4f}" if not math.isnan(v.get("distill_loss", float("nan"))) else "skipped" for v in per_video],
    ))
    _log.info(_row("Teacher ckpt (MB)", *[f"{v.get('ckpt_mb', 0.0):.1f}" for v in per_video]))
    _log.info(_row("Student ckpt (MB)", *[f"{v.get('student_ckpt_mb', 0.0):.1f}" if v.get("student_ckpt_mb") else "—" for v in per_video]))
    _log.info(_row("ONNX size (MB)", *[f"{v.get('onnx_mb', 0.0):.1f}" if v.get("onnx_exported") else "—" for v in per_video]))
    _log.info(_row(
        "Compression ratio",
        *[f"{v['distill_compression_ratio']:.1f}×" if v.get("distill_compression_ratio")
          else (f"{v['teacher_dim'] / v['student_dim']:.1f}×" if v.get("student_dim") and v.get("teacher_dim") else "—")
          for v in per_video],
    ))
    _log.info(_row("Base infer (ms/fr)", *[f"{v.get('base_infer_ms', 0.0):.1f}" for v in per_video]))
    _log.info(_row("Fine-tuned infer (ms/fr)", *[f"{v.get('ft_infer_ms', 0.0):.1f}" for v in per_video]))

    _log.info("")
    _log.info("  SEARCH QUALITY  (top-1 cosine score, self/near-temporal matches excluded)")
    _log.info("  " + SEP[: W - 2])
    _log.info(_row("Base model (pretrained)", *[f"{v.get('base_top_score', 0.0):.4f}" for v in per_video]))
    _log.info(_row("Fine-tuned model", *[f"{v.get('ft_top_score', 0.0):.4f}" for v in per_video]))

    _log.info("")
    _log.info("  3D MAP")
    _log.info("  " + SEP[: W - 2])
    _log.info(_row("Method", *[v.get("map_method", "—") for v in per_video]))
    _log.info(_row("Points", *[str(v.get("map_points", 0)) for v in per_video]))
    _log.info(_row("SfM poses", *[str(v.get("sfm_poses", 0)) for v in per_video]))

    _log.info("")
    _log.info("  ANALYTICS SUMMARY")
    _log.info("  " + SEP[: W - 2])
    _as = lambda v, k: (v.get("analysis_summary", {}) or {}).get(k) or "—"  # noqa: E731
    _log.info(_row("Domain", *[_as(v, "domain") for v in per_video]))
    _log.info(_row("Top category", *[_as(v, "top_category") for v in per_video]))
    _log.info(_row("Artifacts", *[str(_as(v, "artifact_count")) for v in per_video]))
    _log.info(_row("Coverage F/Q/A/O", *[_fmt_analytics_coverage(v.get("analysis_summary", {}) or {}) for v in per_video]))
    _log.info(_row("Detections", *[_fmt_analytics_detections(v.get("analysis_summary", {}) or {}) for v in per_video]))
    _log.info(_row("Temporal", *[_fmt_analytics_temporal(v.get("analysis_summary", {}) or {}) for v in per_video]))
    _log.info(_row("World/Tracking", *[_fmt_analytics_world_tracking(v.get("analysis_summary", {}) or {}) for v in per_video]))
    _log.info(_row("Map quality", *[_fmt_analytics_map(v.get("analysis_summary", {}) or {}) for v in per_video]))
    _log.info(_row("Warnings", *[_fmt_analytics_warnings(v.get("analysis_summary", {}) or {}) for v in per_video]))

    _log.info("")
    _log.info("  TOP VIDEO DESCRIPTION  (CLIP text similarity)")
    _log.info("  " + SEP[: W - 2])
    for v in per_video:
        _log.info("  %-20s  %s", v.get("name", "?"), v.get("top_description", "—") or "—")
    _log.info("")
    _log.info("  " + "=" * (W - 2))

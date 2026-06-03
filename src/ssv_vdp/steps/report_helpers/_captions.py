"""Caption report writers: scene captions, Gemma frame descriptions, segment diffs."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common import (
    _RUNNER_LABEL,
    _analyze_caption_sequence,
    _log,
    write_markdown_artifact,
)


def write_scene_captions_md(
    output_path: Path,
    video_name: str,
    caption_results: list[dict[str, Any]],
    elapsed_sec: float,
    *,
    model_tag: str = "florence-2-large",
    runtime_mode: str = "scored",
) -> None:
    enriched = _analyze_caption_sequence(caption_results)

    segments: list[dict[str, Any]] = []
    for r in enriched:
        if r["is_new_segment"]:
            segments.append({
                "segment_id": r["segment_id"],
                "start_t": r["t_sec"],
                "end_t": r["t_sec"],
                "caption": r.get("caption") or "",
                "frame_count": 1,
            })
        elif segments:
            segments[-1]["end_t"] = r["t_sec"]
            segments[-1]["frame_count"] += 1

    n_segments = len(segments)
    n_unchanged = sum(1 for r in enriched if not r["is_new_segment"])

    lines = [
        f"# Scene Captions — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_tag}",
        f"Runtime mode: {runtime_mode}",
        f"Frames captioned: {len(caption_results)}  |  Unique scenes: {n_segments}"
        f"  |  Repeated frames: {n_unchanged}",
        f"Elapsed: {elapsed_sec:.1f}s",
        "",
        "## Scene Timeline",
        "",
        "| # | Start (s) | End (s) | Frames | Caption |",
        "|---|-----------|---------|--------|---------|",
    ]
    for seg in segments:
        cap = seg["caption"].replace("|", "\\|")[:200]
        lines.append(
            f"| {seg['segment_id'] + 1} | {seg['start_t']:.1f}"
            f" | {seg['end_t']:.1f} | {seg['frame_count']} | {cap} |"
        )

    lines += [
        "",
        "## Per-Frame Captions",
        "",
        "Frames with similarity ≥ 0.45 to the previous caption are marked *same scene*.",
        "",
        "| Frame | t (s) | Seg | Sim | Confidence | Caption |",
        "|-------|-------|-----|-----|------------|---------|",
    ]
    for r in enriched:
        fp = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t = r.get("t_sec", 0.0)
        conf = r.get("caption_confidence", 0.0) or 0.0
        cap = (r.get("caption") or "").replace("|", "\\|")
        seg = r["segment_id"] + 1
        sim = r["similarity"]
        sim_str = f"{sim:.2f}" if sim is not None else "—"
        if not r["is_new_segment"]:
            cap = f"*same scene* {cap}"
        lines.append(f"| `{name}` | {t:.1f} | {seg} | {sim_str} | {conf:.3f} | {cap} |")

    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · Florence-2-large · phase1 captioning*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def _write_gemma_captions_md(
    output_path: Path,
    video_name: str,
    model_id: str,
    captions: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Gemma Frame Descriptions -- {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{model_id}`  |  Frames: {len(captions)}",
        "",
        "| # | t (s) | Frame | Description |",
        "|---|-------|-------|-------------|",
    ]
    for i, c in enumerate(captions, 1):
        fp = Path(c.get("frame_path", "")).name
        t = c.get("t_sec", 0.0)
        desc = c.get("description", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {t:.1f} | `{fp}` | {desc} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL}*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  Written %s", output_path)


def write_gemma_segment_captions_md(
    output_path: Path,
    video_name: str,
    model_id: str,
    boundary_diffs: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Gemma Segment Boundary Diffs — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{model_id}`  |  Boundaries: {len(boundary_diffs)}",
        "",
        (
            "Each row shows the transition between two scene segments: the last frame of "
            "segment N and the first frame of segment N+1, with a Gemma-generated diff description."
        ),
        "",
        "| # | Seg N→N+1 | Before (t) | After (t) | What changed |",
        "|---|-----------|-----------|-----------|--------------|",
    ]
    for b in boundary_diffs:
        seg_label = f"{b.get('prev_segment_id', '?')}→{b.get('next_segment_id', '?')}"
        t_before = f"{b.get('prev_t_sec', 0.0):.1f}s"
        t_after = f"{b.get('next_t_sec', 0.0):.1f}s"
        desc = (
            (b.get("diff_description") or "*(no description)*")
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(
            f"| {b.get('boundary_idx', 0) + 1} | {seg_label} | {t_before} | {t_after} | {desc} |"
        )
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL}*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  Written %s", output_path)

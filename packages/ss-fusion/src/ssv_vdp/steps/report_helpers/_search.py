"""Search and comparison report writers."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..common import (
    _RUNNER_LABEL,
    _log,
    write_markdown_artifact,
)


def write_search_md(
    output_path: Path,
    video_name: str,
    model_label: str,
    query_frame: str,
    results: list[dict[str, Any]],
    query_t_sec: float,
) -> None:
    def _md_image(rel_path: str, alt: str = "frame") -> str:
        return f"![{alt}]({rel_path})"

    lines = [
        f"# {model_label} Transformation Test — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Model: {model_label}",
        "",
        "## Query Frame",
        "",
        f"**Timestamp:** {query_t_sec:.2f}s",
        "",
        _md_image(os.path.relpath(query_frame, output_path.parent), "Query frame"),
        "",
        f"## Top {len(results)} Similar Frames",
        "",
        "Query frame self-match and near-temporal neighbours (±1.0s) are excluded from the ranking.",
        "",
        "| Rank | Score | Timestamp | Frame |",
        "|------|-------|-----------|-------|",
    ]
    for i, r in enumerate(results, 1):
        payload = r.get("payload", r)
        fp = payload.get("frame_path", "")
        t = payload.get("t_sec", 0.0)
        score = r.get("score", 0.0)
        rel = os.path.relpath(fp, output_path.parent) if fp else ""
        lines.append(f"| {i} | {score:.4f} | {t:.2f}s | {_md_image(rel, f'match {i}')} |")
    lines += ["", "---", f"*Artifact produced by {_RUNNER_LABEL}.*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_comparison_md(
    output_path: Path,
    video_name: str,
    base_results: list[dict],
    ft_results: list[dict],
    base_infer_ms: float,
    ft_infer_ms: float,
    ckpt_mb: float,
    onnx_mb: float,
    text_descriptions: list[tuple[str, float]],
) -> None:
    base_paths = {r.get("payload", r).get("frame_path", "") for r in base_results}
    ft_paths = {r.get("payload", r).get("frame_path", "") for r in ft_results}
    overlap = len(base_paths & ft_paths)
    base_scores = [r.get("score", 0) for r in base_results]
    ft_scores = [r.get("score", 0) for r in ft_results]
    avg_base = float(np.mean(base_scores)) if base_scores else 0.0
    avg_ft = float(np.mean(ft_scores)) if ft_scores else 0.0

    lines = [
        f"# Model Comparison — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Video-to-Text Description",
        "",
        "Top content descriptions (via CLIP text similarity):",
        "",
    ]
    for desc, score in text_descriptions[:3]:
        lines.append(f"- **{desc}** (similarity: {score:.3f})")
    lines += [
        "",
        "## Search Quality Comparison",
        "",
        "| Metric | Base Model | Fine-tuned Model |",
        "|--------|-----------|-----------------|",
        f"| Avg top-5 score | {avg_base:.4f} | {avg_ft:.4f} |",
        f"| Δ score | — | {avg_ft - avg_base:+.4f} |",
        f"| Result overlap | {overlap}/{len(base_results)} frames in common | |",
        "",
        "## Model Statistics",
        "",
        "| Metric | Base Model | Fine-tuned (PyTorch) | Fine-tuned (ONNX) |",
        "|--------|-----------|---------------------|------------------|",
        f"| Checkpoint size | ~330 MB (hub) | {ckpt_mb:.1f} MB | {onnx_mb:.1f} MB |",
        f"| Inference time (GPU/CPU) | {base_infer_ms:.1f} ms/frame | {ft_infer_ms:.1f} ms/frame | — |",
        "",
        "## How to Use Artifacts",
        "",
        "- **`base_search.md`** — nearest-neighbour results with the pretrained DINOv3 backbone",
        "- **`finetuned_search.md`** — same query with the mission-adapted backbone",
        "- **`edge_models/dino_local.onnx`** — ONNX model for on-device inference (Jetson, Hailo-8)",
        "- **`edge_models/gallery.npz`** — embedding gallery for 1-NN classification",
        "- **`3d_map/`** — sparse 3D point cloud from Structure-from-Motion",
        "",
        "```python",
        "from pipeline.training.edge_inference import EdgeClassifier",
        "clf = EdgeClassifier('edge_models/dino_local.onnx', 'edge_models/gallery.npz')",
        "labels = clf.classify(frame_pil)   # [(label, score), ...]",
        "```",
        "",
        "---",
        f"*Artifact produced by {_RUNNER_LABEL}.*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_description_md(
    output_path: Path,
    video_name: str,
    frame_list: list[tuple[str, float]],
    text_descriptions: list[tuple[str, float]],
    all_scored: list[tuple[str, float]],
) -> None:
    lines = [
        f"# Image-to-Text Description — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Top Video Descriptions",
        "",
        "Ranked by cosine similarity between the average CLIP frame embedding and each text prompt:",
        "",
        "| Rank | Description | Similarity |",
        "|------|-------------|-----------|",
    ]
    for rank, (desc, score) in enumerate(text_descriptions, 1):
        lines.append(f"| {rank} | {desc} | {score:.4f} |")
    lines += [
        "",
        "## All Prompts Scored",
        "",
        "| Description | Similarity |",
        "|-------------|-----------|",
    ]
    for desc, score in all_scored:
        lines.append(f"| {desc} | {score:.4f} |")
    lines += [
        "",
        "## Sample Frames",
        "",
        "Frames used for description (evenly spaced, up to 32):",
        "",
    ]
    step = max(1, len(frame_list) // 8)
    for fp, t_sec in frame_list[::step][:8]:
        lines.append(f"- `{Path(fp).name}` (t={t_sec:.1f}s)")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · model: OpenCLIP ViT-B/16 (openai)*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)

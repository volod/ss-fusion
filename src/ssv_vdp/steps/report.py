"""All write_*_md functions, print_run_stats, and markdown helpers."""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    _RUNNER_LABEL,
    _SCENE_CHANGE_THRESH,
    _analyze_caption_sequence,
    _jaccard,
    _log,
    write_markdown_artifact,
)
from .threat_contradictions import (
    contradiction_signals_for_threat,
    sensor_sources_from_evidence,
    summarize_contradictions,
    support_frame_names,
)

# -- Markdown helpers ----------------------------------------------------------


def _md_image(rel_path: str, alt: str = "frame") -> str:
    return f"![{alt}]({rel_path})"


def write_search_md(
    output_path: Path,
    video_name: str,
    model_label: str,
    query_frame: str,
    results: list[dict[str, Any]],
    query_t_sec: float,
) -> None:
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
            all_types = set(prev_sig) | set(curr_sig)
            for vt in sorted(all_types):
                p = prev_sig.get(vt, 0)
                c = curr_sig.get(vt, 0)
                if p != c:
                    changes.append(f"{vt}: {p}→{c}")

    return "; ".join(changes) if changes else ""


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

    # Build segment-level summary
    segments: list[dict[str, Any]] = []
    for r in enriched:
        if r["is_new_segment"]:
            segments.append(
                {
                    "segment_id": r["segment_id"],
                    "start_t": r["t_sec"],
                    "end_t": r["t_sec"],
                    "caption": r.get("caption") or "",
                    "frame_count": 1,
                }
            )
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
    """Write per-frame Gemma generative descriptions to a markdown file."""
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
    """Write Gemma segment-boundary diff descriptions to a markdown file."""
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


def write_gemma_analysis_md(
    output_path: Path,
    video_name: str,
    model_id: str,
    sample_n: int,
    analysis: dict[str, Any],
    dino_comparison: dict[str, Any],
    text_query_results: list[dict[str, Any]],
    elapsed_sec: float,
    clip_comparison: dict[str, Any] | None = None,
) -> None:
    """Write Gemma multimodal analysis report to *output_path*."""
    lines = [
        f"# Gemma Open-Weight Analysis — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{model_id}`  |  Frames sampled: {sample_n}  |  Elapsed: {elapsed_sec:.1f}s",
        "",
        "## Analyses Performed",
        "",
        "| Analysis | Status |",
        "|----------|--------|",
    ]
    for key, res in analysis.items():
        label = key.replace("_", " ").title()
        if res.get("error"):
            status = f"✗ {res['error'][:60]}"
        else:
            status = "[ok]"
        lines.append(f"| {label} | {status} |")
    lines += [""]

    # DINOv3 comparison
    if dino_comparison.get("available"):
        mnn = dino_comparison.get("mnn_rate", 0.0)
        k = dino_comparison.get("k", 5)
        cg = dino_comparison.get("mean_cossim_gemma", 0.0)
        cd = dino_comparison.get("mean_cossim_dino", 0.0)
        lines += [
            "## Gemma vs DINOv3 Embedding Comparison",
            "",
            f"Both models embedded the same {dino_comparison.get('n_frames', sample_n)} frames.",
            f"Gemma model: `{model_id}`.  DINOv3 model: `dinov3_vitb14`.",
            "",
            "| Metric | Gemma | DINOv3 |",
            "|--------|-------|--------|",
            f"| Mean pairwise cosine similarity | {cg:.4f} | {cd:.4f} |",
            f"| Mutual nearest-neighbor overlap (k={k}) | {mnn:.3f} | — |",
            "",
            "**Mean pairwise cosine similarity**: lower = more discriminative embedding space.",
            "",
            f"**MNN@{k}** ({mnn:.1%}): fraction of frames whose top-{k} visual neighbours agree",
            "between Gemma and DINOv3.",
            "",
        ]
    else:
        lines += [
            "## Gemma vs DINOv3 Embedding Comparison",
            "",
            f"Skipped: {dino_comparison.get('reason', 'DINOv3 not available')}",
            "",
        ]

    # CLIP comparison
    cc = clip_comparison or {}
    if cc.get("available"):
        mnn_c = cc.get("mnn_rate", 0.0)
        k_c = cc.get("k", 5)
        cg_c = cc.get("mean_cossim_gemma", 0.0)
        cl_c = cc.get("mean_cossim_clip", 0.0)
        lines += [
            "## Gemma vs CLIP Embedding Comparison",
            "",
            f"Both models embedded the same {cc.get('n_frames', sample_n)} frames.",
            f"Gemma model: `{model_id}`.  CLIP model: `ViT-B-16/openai`.",
            "",
            "| Metric | Gemma | CLIP |",
            "|--------|-------|------|",
            f"| Mean pairwise cosine similarity | {cg_c:.4f} | {cl_c:.4f} |",
            f"| Mutual nearest-neighbor overlap (k={k_c}) | {mnn_c:.3f} | — |",
            "",
            f"**MNN@{k_c}** ({mnn_c:.1%}): fraction of frames whose top-{k_c} visual neighbours agree",
            "between Gemma and CLIP.",
            "",
        ]
    elif cc:
        lines += [
            "## Gemma vs CLIP Embedding Comparison",
            "",
            f"Skipped: {cc.get('reason', 'CLIP not available')}",
            "",
        ]

    # Scene change detection
    sc = analysis.get("scene_change_detection", {})
    if not sc.get("error") and sc.get("changes") is not None:
        changes = sc.get("changes", [])
        lines += [
            "## Scene Change Detection",
            "",
            f"Cosine distance > {_SCENE_CHANGE_THRESH} between consecutive sampled frames.",
            f"Detected {sc.get('n_changes', 0)} transition(s).",
            "",
        ]
        if changes:
            lines += ["| # | t (s) | Cosine Distance |", "|---|-------|-----------------|"]
            for i, ch in enumerate(changes[:15], 1):
                lines.append(f"| {i} | {ch['t_sec']:.1f} | {ch['distance']:.4f} |")
            lines += [""]

    # Zero-shot classification
    clf = analysis.get("scene_classification", {})
    if not clf.get("error") and clf.get("category_distribution"):
        lines += [
            "## Zero-Shot Scene Classification",
            "",
            f"Top predicted scene categories across {sample_n} frames:",
            "",
            "| Category | Frame Count |",
            "|----------|-------------|",
        ]
        for cat, cnt in clf["category_distribution"].items():
            lines.append(f"| {cat} | {cnt} |")
        lines += [""]

    # Cross-modal text queries
    if text_query_results:
        lines += [
            "## Cross-Modal Text → Frame Retrieval",
            "",
            "Text probes (mean-pooled text embeddings) vs frame embeddings (cosine similarity):",
            "",
            "| Query | Best Frame (t) | Score |",
            "|-------|---------------|-------|",
        ]
        for qr in text_query_results:
            q = qr.get("query", "—")
            top = qr.get("top_results", [])
            if top:
                fp = Path(top[0].get("frame_path", "")).name
                t_s = top[0].get("t_sec", 0.0)
                score = top[0].get("score", 0.0)
                lines.append(f"| {q} | `{fp}` ({t_s:.1f}s) | {score:.4f} |")
            else:
                lines.append(f"| {q} | — | — |")
        lines += [""]

    # Temporal video embedding
    te = analysis.get("temporal_embedding", {})
    if not te.get("error"):
        lines += [
            "## Temporal Video Embedding",
            "",
            f"Mean-pool of all {sample_n} frame embeddings → single video-level vector",
            f"(dim={te.get('dim', 0)}).  Can be used for video-level retrieval or comparison.",
            "",
        ]

    # Clustering
    cl = analysis.get("scene_clustering", {})
    if not cl.get("error") and cl.get("n_clusters"):
        lines += [
            "## Scene Clustering",
            "",
            f"{cl['n_clusters']} semantic clusters from {sample_n} frames",
            f"(mean cluster size: {cl.get('mean_cluster_size', 0):.1f} frames).",
            "",
        ]

    # -- Analysis interpretation -----------------------------------------------
    lines += ["## Findings & Interpretation", ""]

    # Embedding discrimination
    dino_avail = dino_comparison.get("available", False)
    cc = clip_comparison or {}
    clip_avail = cc.get("available", False)

    if dino_avail:
        cg = dino_comparison.get("mean_cossim_gemma", 0.0)
        cd = dino_comparison.get("mean_cossim_dino", 0.0)
        mnn = dino_comparison.get("mnn_rate", 0.0)
        if cg < cd:
            lines.append(
                f"- **Gemma is more discriminative than DINOv3** for this video "
                f"(mean cosine {cg:.4f} < {cd:.4f}). Gemma's language-grounded embeddings "
                f"spread frames further apart in embedding space — useful for precise retrieval."
            )
        elif abs(cg - cd) < 0.05:
            lines.append(
                f"- **Gemma and DINOv3 have similar discrimination** (cosine {cg:.4f} vs {cd:.4f}). "
                f"Both models capture similar visual structure for this mission content."
            )
        else:
            lines.append(
                f"- **DINOv3 is more discriminative than Gemma** for this video "
                f"(cosine {cd:.4f} < {cg:.4f}). DINOv3's self-supervised visual features "
                f"give finer-grained distinctions. Gemma remains valuable for language-grounded queries."
            )
        if mnn >= 0.8:
            lines.append(
                f"- **High DINOv3↔Gemma agreement (MNN={mnn:.1%})**: both models agree on which "
                f"frames are visually similar. Gemma embeddings can safely substitute DINOv3 for "
                f"retrieval with additional benefit of text-query compatibility."
            )
        elif mnn >= 0.5:
            lines.append(
                f"- **Moderate DINOv3↔Gemma agreement (MNN={mnn:.1%})**: the models partially "
                f"disagree on visual neighbourhoods. Gemma captures semantic similarity; DINOv3 "
                f"captures low-level visual similarity. Both are complementary — use Gemma for "
                f"text queries, DINOv3 for image-to-image search."
            )
        else:
            lines.append(
                f"- **Low DINOv3↔Gemma agreement (MNN={mnn:.1%})**: the models assign very "
                f"different neighbourhoods. Likely cause: 30 fps near-duplicate frames collapse "
                f"to the same DINOv3 cluster while Gemma's language bias separates them differently. "
                f"This is expected and not a failure — the two spaces serve different query types."
            )
        lines.append("")

    if clip_avail:
        mnn_c = cc.get("mnn_rate", 0.0)
        if mnn_c >= 0.8:
            lines.append(
                f"- **High CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma embeddings are "
                f"strongly aligned with CLIP's image-text space. Gemma can replace CLIP for "
                f"cross-modal retrieval while also supporting image-to-image search."
            )
        elif mnn_c >= 0.5:
            lines.append(
                f"- **Moderate CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma and CLIP agree "
                f"on roughly half of visual neighbourhoods. Use CLIP for image-text matching "
                f"and Gemma for richer structured reasoning."
            )
        else:
            lines.append(
                f"- **Low CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma organises this "
                f"visual content differently from CLIP. Gemma may be using scene-level semantics "
                f"while CLIP relies on global appearance statistics."
            )
        lines.append("")

    # Scene change detection
    sc = analysis.get("scene_change_detection", {})
    n_changes = sc.get("n_changes", 0)
    if not sc.get("error") and sc.get("changes") is not None:
        if n_changes == 0:
            lines.append(
                f"- **No scene transitions detected**: all {sample_n} sampled frames are "
                f"visually continuous. This is typical of 30 fps missions where scenes evolve slowly. "
                f"Use the Scene Timeline in `scene_captions.md` for segment-level analysis."
            )
        elif n_changes <= 3:
            lines.append(
                f"- **{n_changes} scene transition(s)**: the video has a small number of "
                f"distinct visual states. Gemma embedding distances reliably flag these transitions "
                f"as higher-priority frames for annotation (`al_tag=needs_annotation`)."
            )
        else:
            lines.append(
                f"- **{n_changes} scene transitions**: high visual variability in this mission. "
                f"Frames at transition boundaries carry the most novel information and should be "
                f"prioritised for SSL training data."
            )
        lines.append("")

    # Clustering
    cl = analysis.get("scene_clustering", {})
    n_clusters = cl.get("n_clusters", 0)
    mean_sz = cl.get("mean_cluster_size", 0)
    if not cl.get("error") and n_clusters:
        if mean_sz > sample_n * 0.3:
            lines.append(
                f"- **Few, large clusters ({n_clusters} clusters, ~{mean_sz:.0f} frames each)**: "
                f"the mission covers a small set of visually distinct scenes. "
                f"SSL temporal pairs will be highly informative — nearby frames share the same cluster."
            )
        else:
            lines.append(
                f"- **Many small clusters ({n_clusters} clusters, ~{mean_sz:.0f} frames each)**: "
                f"high scene diversity. More SSL epochs may be needed to cover all visual states."
            )
        lines.append("")

    # Distillation recommendation
    if dino_avail:
        mnn_d = dino_comparison.get("mnn_rate", 0.0)
        if mnn_d >= 0.7:
            lines.append(
                "- **Distillation recommendation**: Gemma embeddings are a strong teacher signal. "
                "Set `gemma_embedder` in `step_distill` (done automatically when `MODEL_NAME=gemma`) "
                "for maximum-hydration distillation — the student inherits both visual and language-grounded structure."
            )
        else:
            lines.append(
                "- **Distillation recommendation**: Gemma and DINOv3 neighbourhoods diverge for "
                "this content. Run both distillation chains and compare Recall@1: "
                "DINOv3-teacher for image retrieval, Gemma-teacher for text-query tasks."
            )
        lines.append("")

    lines += ["---", f"*Produced by {_RUNNER_LABEL} — Gemma open-weight multimodal analysis.*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: Any,
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: list[float],
) -> None:
    from .ssl import _analyze_loss_curve, _interpret_finetune_results, _loss_sparkline

    ckpt_mb = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0
    best_epoch = int(np.argmin(loss_history)) + 1 if loss_history else 0
    stats = _analyze_loss_curve(loss_history)
    sparkline = _loss_sparkline(loss_history)
    deltas = stats.get("deltas", [])
    bullets = _interpret_finetune_results(cfg, stats, elapsed_sec)

    lines = [
        f"# SSL Fine-Tuning Statistics — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## What We Do",
        "",
        "**Self-Supervised Learning (SSL)** adapts a pre-trained vision backbone to the "
        "specific visual domain of this mission without any labelled annotations.",
        "",
        "### Method: NT-Xent Contrastive Loss",
        "",
        "We use **NT-Xent** (Normalised Temperature-scaled Cross Entropy, a.k.a. InfoNCE) "
        "contrastive learning:",
        "",
        "1. Each training step produces a batch of *positive pairs* (two views of the same scene).",
        f"2. The model encodes both views through the DINOv3 backbone + a small projection head "
        f"   (embed_dim={cfg.embed_dim} → proj_dim={cfg.proj_out_dim}).",
        "3. The loss pushes the two views of the same scene together in embedding space "
        "   and pushes all other pairs in the batch apart.",
        f"4. Temperature τ={cfg.temperature} controls the sharpness of the distribution "
        f"   (lower = harder negatives, more informative but less stable).",
        "",
        f"The backbone is **partially frozen**: the first {cfg.freeze_blocks} transformer blocks "
        f"are kept fixed (preserving generic low-level features), and only the top "
        f"{12 - cfg.freeze_blocks} blocks + projection head are trained. "
        f"This prevents catastrophic forgetting on a small video dataset.",
        "",
        f"### Pair Construction Strategy: `{cfg.approach}`",
        "",
    ]

    if cfg.approach == "track_cycle":
        lines += [
            "**Track cycle-consistency triplets** — RF-DETR track IDs provide triplets "
            "(A, B, C) of the same tracked object at times t, t+k, t+2k.",
            "Loss: NTXent(A,B) + NTXent(B,C) + 0.3·NTXent(A,C). "
            "The cycle term enforces that object identity is stable across the widest "
            "temporal gap in the triplet, preventing embedding drift along long tracks.",
        ]
    elif cfg.approach == "track":
        lines += [
            "**Track pairs** — RF-DETR track IDs provide pairs (A, B) of the same "
            "tracked object at two different times (gap 2–5 appearances). "
            "Crops are taken around the tracked bbox with 15 % padding.",
            "Rationale: same-object pairs encode identity consistency that full-frame "
            "temporal pairs cannot — the model must match the object across viewpoint "
            "and appearance changes, not just spatial proximity.",
        ]
    elif cfg.approach == "temporal":
        lines += [
            f"**Temporal pairs** — consecutive frames within ±{cfg.max_gap} positions "
            f"in the frame sequence form positive pairs.",
            "Rationale: adjacent frames in a 30 fps outdoor video show nearly the same scene, "
            "so pulling their embeddings together teaches the model scene-level consistency "
            "while naturally using real mission content (no synthetic augmentation needed).",
        ]
    else:
        lines += [
            "**Augmentation pairs** — each frame is augmented twice with random crops, "
            "horizontal flips, colour jitter, and Gaussian blur.",
            f"Rationale: fewer than {cfg.batch_size * 2} frames are available, so temporal "
            f"pairing would produce too few unique positive pairs. "
            f"Augmentation-based SSL is used as a fallback.",
        ]

    lines += [
        "",
        "### Optimiser",
        "",
        "| Component | Setting |",
        "|-----------|---------|",
        "| Optimiser | AdamW |",
        f"| Learning rate | {cfg.lr} |",
        f"| Weight decay | {cfg.weight_decay} |",
        f"| LR schedule | Cosine annealing over {cfg.epochs} epochs |",
        f"| Batch size | {cfg.batch_size} pairs |",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | `{cfg.model_name}` |",
        f"| Approach | `{cfg.approach}` |",
        f"| Epochs | {cfg.epochs} |",
        f"| Batch size | {cfg.batch_size} |",
        f"| Learning rate | {cfg.lr} |",
        f"| Temperature | {cfg.temperature} |",
        f"| Frozen blocks | {cfg.freeze_blocks} / 12 |",
        f"| Embed dim | {cfg.embed_dim} → proj {cfg.proj_out_dim} |",
        f"| Device | `{cfg.device}` |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Best loss | {best_loss:.4f} |",
        f"| Best epoch | {best_epoch}/{cfg.epochs} |",
        f"| First loss | {stats.get('first_loss', float('nan')):.4f} |",
        f"| Last loss | {stats.get('last_loss', float('nan')):.4f} |",
        f"| Total drop | {stats.get('drop_pct', 0):.1f} % |",
        f"| Convergence epoch | {stats.get('convergence_epoch', '—')} |",
        f"| Training time | {elapsed_sec:.1f}s |",
        f"| Checkpoint size | {ckpt_mb:.1f} MB |",
        f"| Checkpoint path | `{checkpoint_path}` |",
        "",
        "## Result Analysis",
        "",
    ]
    for b in bullets:
        lines.append(f"- {b}")
        lines.append("")

    lines += [
        "## Loss Curve",
        "",
        "```",
        f"high │{sparkline}│",
        f" low │{'-' * len(sparkline)}│",
        f"      epoch 1{'':>{max(0, len(sparkline) - 9)}}epoch {len(loss_history)}",
        "```",
        "",
        f"*Each character represents {'one epoch' if len(loss_history) <= 40 else 'a range of epochs'}. "
        f"Higher bar = higher loss.*",
        "",
        "| Epoch | Loss | Δ vs prev | Trend |",
        "|-------|------|-----------|-------|",
    ]
    for ep, loss in enumerate(loss_history, 1):
        if ep == 1:
            delta_str = "—"
            trend = "—"
        else:
            d = deltas[ep - 2]
            delta_str = f"{d:+.4f}"
            if d < -0.01:
                trend = "↓ improving"
            elif d > 0.01:
                trend = "↑ worsening"
            else:
                trend = "→ stable"
        marker = " ← best" if ep == best_epoch else ""
        lines.append(f"| {ep} | {loss:.4f} | {delta_str} | {trend}{marker} |")

    lines += [
        "",
        "## How to Use This Checkpoint",
        "",
        "```bash",
        f"export DINO_CHECKPOINT={checkpoint_path}",
        "python main.py --mode local --videos-dir .data/videos",
        "```",
        "",
        "---",
        f"*Artifact produced by {_RUNNER_LABEL}. See `edge_models/` for ONNX export.*",
    ]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


def write_distill_stats_md(
    output_path: Path,
    video_name: str,
    stats: dict[str, Any],
) -> None:
    loss_history = stats.get("loss_history", [])
    recall_history = stats.get("recall_history", [])
    loss_components = stats.get("loss_components", {})
    compression = stats.get("compression_ratio", 0.0)
    t_params = stats.get("teacher_params", 0)
    s_params = stats.get("student_params", 0)
    best_recall = stats.get("best_recall", float("nan"))

    lines = [
        f"# Knowledge Distillation — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Teacher | DINOv3 ViT-B/14 (fine-tuned SSL) — dim={stats.get('teacher_dim', 768)}, {t_params // 1_000_000}M params |",
        f"| Student | {stats.get('student_model', 'dinov2_vits14')} — dim={stats.get('student_dim', 384)}, {s_params // 1_000_000}M params |",
        "| Method | RKD-DA (distance + angle) + KoLeo spread regulariser + cosine anchor |",
        "| Loss weights | λ_D=25  λ_A=50  λ_kd=1.0  λ_koleo=0.1 |",
        f"| Epochs | {len(loss_history)} |",
        f"| Elapsed | {stats.get('elapsed', 0):.1f}s |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Best total loss | {stats.get('best_loss', float('nan')):.4f} |",
        f"| Best Recall@1 (student vs teacher) | {best_recall:.3f} |",
        f"| Compression ratio | {compression:.1f}× ({t_params // 1_000_000}M → {s_params // 1_000_000}M params) |",
        f"| Student dim | {stats.get('student_dim', 384)} (vs teacher {stats.get('teacher_dim', 768)}) |",
        f"| Best checkpoint | `{Path(stats.get('best_path', '')).name}` |",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Epoch | Total | RKD-D | RKD-A | Cosine | KoLeo | Recall@1 |",
        "|-------|-------|-------|-------|--------|-------|----------|",
    ]
    n = len(loss_history)
    for i in range(n):
        r1 = recall_history[i] if i < len(recall_history) else float("nan")
        rd = (
            loss_components.get("rkd_d", [])[i]
            if i < len(loss_components.get("rkd_d", []))
            else float("nan")
        )
        ra = (
            loss_components.get("rkd_a", [])[i]
            if i < len(loss_components.get("rkd_a", []))
            else float("nan")
        )
        cos = (
            loss_components.get("cosine", [])[i]
            if i < len(loss_components.get("cosine", []))
            else float("nan")
        )
        kol = (
            loss_components.get("koleo", [])[i]
            if i < len(loss_components.get("koleo", []))
            else float("nan")
        )
        lines.append(
            f"| {i + 1} | {loss_history[i]:.4f} | {rd:.4f} | {ra:.4f} | {cos:.4f} | {kol:.4f} | {r1:.3f} |"
        )

    lines += [
        "",
        "## Architecture",
        "",
        "```",
        "Teacher (frozen):  DINOv3 ViT-B/14  →  768-dim embedding",
        "                         ↓ RKD-DA (distance + angle) + cosine anchor",
        "Proj head (temp):  Linear(384 → 768, orthogonal init)  [discarded after training]",
        "                         ↑",
        "Student (trained): DINOv2 ViT-S/14  →  384-dim embedding",
        "                         ↑",
        "                    KoLeo spread regulariser (prevents collapse)",
        "```",
        "",
        "**RKD-DA** (Relational Knowledge Distillation) preserves pairwise neighbourhood",
        "topology in the student embedding space, directly optimising retrieval Recall@K.",
        f"The student is {compression:.1f}× smaller and ~2× faster at inference.",
        "The projection head is used only during training to align embedding spaces.",
        "The saved checkpoint contains **only the student backbone weights**.",
        "",
        "---",
        f"*Artifact produced by {_RUNNER_LABEL}. Student exported to `edge_models/dino_local.onnx`.*",
    ]
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


def write_final_stats_md(
    output_path: Path,
    per_video: list[dict[str, Any]],
    total_elapsed: float,
) -> None:
    step_sum = sum(sum(v.get("timings", {}).values()) for v in per_video)
    concurrent_overlap = max(0.0, step_sum - total_elapsed)
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
    ]
    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    header = "| Step | Type | " + " | ".join(names) + " | Total |"
    sep = "|------|------|" + "|".join(["-------"] * len(names)) + "|-------|"
    lines += [header, sep]
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
    lines = [
        f"# Multimodal Features — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Step | Status | Detail |",
        "|------|--------|--------|",
        f"| ASR (Whisper) | {'[ok]' if not asr_result.get('skipped') else '—'} | "
        f"{asr_result.get('covered_frames', 0)} frames with subtitles |",
        f"| OCR | {'[ok]' if not ocr_result.get('skipped') else '—'} | "
        f"{ocr_result.get('non_empty', 0)} frames with text |",
        f"| Depth | {'[ok]' if not depth_result.get('skipped') else '—'} | "
        f"{depth_result.get('ok_count', 0)} frames estimated |",
        f"| Detection | {'[ok]' if not det_result.get('skipped') else '—'} | "
        f"{det_result.get('total_objects', 0)} objects detected |",
        f"| World Model | {'[ok]' if not world_result.get('skipped') else '—'} | "
        f"{world_result.get('ok_count', 0)} clips processed |",
        f"| Platform-state fusion | {'[ok]' if not state_fusion_result.get('skipped') else '—'} | "
        f"{state_fusion_result.get('summary', {}).get('frame_count', 0)} posterior samples |",
        f"| Qwen VLM captioning | {'[ok]' if not qwen_result.get('skipped') else '—'} | "
        f"{qwen_result.get('ok_count', 0)} frames captioned |",
        f"| UniDriveVLA expert analysis | {'[ok]' if not unidrive_result.get('skipped') else '—'} | "
        f"{unidrive_result.get('ok_count', 0)} frames analysed |",
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
            lines += [
                "| t (s) | p10 | p25 | p50 | p75 | p90 |",
                "|-------|-----|-----|-----|-----|-----|",
            ]
            for r in depth_rows:
                p = r["depth"].get("percentiles", [0] * 5)
                lines.append(
                    f"| {r['t_sec']:.1f} | "
                    f"{p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {p[4]:.3f} |"
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
    for sample in samples:
        pos = sample.position_enu_m
        vel = sample.velocity_enu_mps
        lines.append(
            f"| {sample.t_sec:.1f} | {pos['x']:.2f} | {pos['y']:.2f} | {pos['z']:.2f} | "
            f"{vel['x']:.2f} | {vel['y']:.2f} | {vel['z']:.2f} | "
            f"{sample.covariance_trace:.3f} | {sample.quality} |"
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
        1
        for r in results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    )
    parse_errors = sum(1 for r in results if r.get("parse_error"))
    unavailable = sum(1 for r in results if r.get("service_unavailable"))

    # Build text captions for scene-segment detection from only valid structured rows.
    text_results: list[dict[str, Any]] = []
    for r in results:
        if r.get("service_unavailable") or r.get("skipped") or r.get("parse_error"):
            continue
        summary = r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""
        text_results.append({**r, "caption": summary})
    enriched_valid = _analyze_caption_sequence(text_results) if text_results else []
    enriched_index = {
        (str(r.get("frame_path", "")), float(r.get("t_sec", 0.0))): r for r in enriched_valid
    }

    # Segment-level summary
    segments: list[dict[str, Any]] = []
    for r in enriched_valid:
        if r["is_new_segment"]:
            segments.append(
                {
                    "segment_id": r["segment_id"],
                    "start_t": r["t_sec"],
                    "end_t": r["t_sec"],
                    "frame_count": 1,
                    "scene_summary": r.get("scene_summary") or r.get("caption") or "",
                    "road_surface": r.get("road_surface", ""),
                    "road_condition": r.get("road_condition", ""),
                    "vehicle_groups": r.get("vehicle_groups", []),
                }
            )
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
            caption = "*sidecar unavailable*"
            delta = "—"
        elif r.get("parse_error"):
            caption = "*parse error*"
            raw = str(r.get("raw", "") or "").replace("|", "\\|").strip()
            if raw:
                caption += f" {raw[:160]}"
            delta = "—"
        elif r.get("skipped"):
            caption = "*skipped*"
            delta = "—"
        else:
            # Structured diff against previous frame
            delta = _diff_structured_caption(prev_structured, r) if prev_structured else ""
            delta = delta.replace("|", "\\|") if delta else ("—" if prev_structured else "first")

            # Caption text: prefer scene_summary, then fallback keys
            facts = r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""
            if not facts:
                parts = []
                for k, v in r.items():
                    if (
                        k
                        not in (
                            "frame_path",
                            "t_sec",
                            "subtitle_text",
                            "ocr_text",
                            "segment_id",
                            "is_new_segment",
                            "similarity",
                            "segment_start_t",
                            "caption",
                        )
                        and v
                    ):
                        parts.append(f"{k}: {v}")
                facts = "; ".join(parts[:4])
            caption = str(facts).replace("|", "\\|")[:200]
            if enriched_row and not enriched_row["is_new_segment"]:
                caption = f"*unchanged* {caption}"

            # Update structured state for next diff
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
        if r.get("service_unavailable"):
            lines.append(f"| {r.get('t_sec', 0.0):.1f} | — | — | — | *service unavailable* | — |")
            continue
        if r.get("parse_error"):
            lines.append(f"| {r.get('t_sec', 0.0):.1f} | — | — | — | *parse error* | — |")
            continue
        u = r.get("understanding", {}) or {}
        p = r.get("perception", {}) or {}
        plan = r.get("planning", {}) or {}
        moe = r.get("mixture_of_experts", {}) or {}
        understanding = (u.get("scene_summary", "") or "").replace("|", "\\|")[:70]
        planning = (plan.get("recommended_action", "") or "").replace("|", "\\|")[:70]
        lines.append(
            f"| {r.get('t_sec', 0.0):.1f} | {u.get('risk_level', 'unknown')} | "
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
    qwen_rows = [
        r
        for r in qwen_result.get("results", [])
        if not r.get("service_unavailable") and not r.get("parse_error")
    ]
    uni_rows = [
        r
        for r in unidrive_result.get("results", [])
        if not r.get("service_unavailable") and not r.get("parse_error")
    ]

    def _nearest(rows: list[dict[str, Any]], t_sec: float) -> dict[str, Any] | None:
        if not rows:
            return None
        return min(rows, key=lambda r: abs(float(r.get("t_sec", 0.0)) - t_sec))

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for u in uni_rows:
        q = _nearest(qwen_rows, float(u.get("t_sec", 0.0)))
        if q is None:
            continue
        if abs(float(u.get("t_sec", 0.0)) - float(q.get("t_sec", 0.0))) <= 2.0:
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
        example_rows.append(
            (
                float(u.get("t_sec", 0.0)),
                q_summary,
                u_summary,
                moe_summary,
                str(u_moe.get("expert_agreement", "unknown") or "unknown"),
            )
        )

    mean_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0
    gemma_scene = ""
    task_results = gemma_result.get("task_results", {}) or {}
    clf = task_results.get("scene_classification", {}) or {}
    cat_dist = clf.get("category_distribution", {}) or {}
    if cat_dist:
        gemma_scene = next(iter(cat_dist))

    risk_levels = [((r.get("understanding") or {}).get("risk_level", "unknown")) for r in uni_rows]
    agreement_levels = [
        ((r.get("mixture_of_experts") or {}).get("expert_agreement", "unknown")) for r in uni_rows
    ]
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
    for t_sec, q_summary, u_summary, moe_summary, expert_agreement in example_rows:
        q_summary_md = q_summary.replace("|", "\\|")[:60]
        u_summary_md = u_summary.replace("|", "\\|")[:60]
        moe_summary_md = moe_summary.replace("|", "\\|")[:60]
        lines.append(
            f"| {t_sec:.1f} | {q_summary_md} | "
            f"{u_summary_md} | "
            f"{moe_summary_md} | {expert_agreement} |"
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
            threat_type,
            primitive,
            unidrive_rows,
            physical_state,
        )
        sensor_sources = sensor_sources_from_evidence(evidence_sources)
        disagreeing_sources = [
            str(signal.get("description", "") or "")
            for signal in contradiction_signals
            if signal.get("description")
        ]
        rows.append(
            {
                "threat_type": threat_type,
                "score": float(threat.get("score", 0.0) or 0.0),
                "uncertainty": float(
                    evidence.get("uncertainty", primitive.get("uncertainty", 0.0)) or 0.0
                ),
                "sensor_sources": sensor_sources,
                "disagreeing_sources": disagreeing_sources,
                "contradiction_signals": contradiction_signals,
                "recommended_action": str(
                    policy_decision.get(
                        "recommended_action", local_threat.get("recommended_action", "continue")
                    )
                ),
                "support_frames": support_frame_names(primitive.get("spatial_support") or []),
                "confidence": max(
                    0.0,
                    1.0
                    - float(evidence.get("uncertainty", primitive.get("uncertainty", 0.0)) or 0.0),
                ),
                "evidence_sources": evidence_sources,
            }
        )
    return rows


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
    lines = [
        f"# Video Synthesis — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Elapsed: {elapsed_sec:.1f}s",
        "",
    ]
    if ontology:
        lines += [
            "## Video Ontology",
            "",
            "| Field | Value |",
            "|-------|-------|",
        ]
        for k, v in ontology.items():
            val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            lines.append(f"| {k} | {val.replace('|', '&#124;')} |")
        lines.append("")
    if narrative:
        lines += [
            "## Video Narrative",
            "",
            narrative,
            "",
        ]
    if local_threat and not local_threat.get("skipped"):
        threat_rows = _normalise_threat_rows(
            local_threat,
            policy_decision,
            threat_primitives_result,
            unidrive_rows,
            physical_state,
        )
        lines += [
            "## Local Threat Assessment",
            "",
            f"- Local threat score: {float(local_threat.get('local_threat_score', 0.0)):.3f}",
            f"- Recommended action: `{(policy_decision or {}).get('recommended_action', local_threat.get('recommended_action', 'continue'))}`",
            f"- Automation confidence: {float(local_threat.get('automation_confidence', 1.0)):.3f}",
            f"- Trust penalty: {float(local_threat.get('trust_penalty', 0.0)):.3f}",
            "",
            "## Threat Evidence",
            "",
            "| threat_type | score | uncertainty | sensor_sources | disagreeing_sources | recommended_action |",
            "|-------------|-------|-------------|----------------|---------------------|--------------------|",
        ]
        if threat_rows:
            for row in threat_rows:
                lines.append(
                    f"| {row['threat_type']} | "
                    f"{row['score']:.3f} | "
                    f"{row['uncertainty']:.3f} | "
                    f"{'; '.join(row['sensor_sources']) or 'none'} | "
                    f"{'; '.join(row['disagreeing_sources']) or 'none'} | "
                    f"{row['recommended_action']} |"
                )
        else:
            lines.append("| — | 0.000 | 0.000 | none | none | continue |")
        contradiction_summary = summarize_contradictions(threat_rows)
        conflict_patterns = [
            f"{row['pattern']} ({row['count']})"
            for row in contradiction_summary.get("source_pair_conflicts", [])[:4]
        ]
        lines += [
            "",
            "## Contradiction Metrics",
            "",
            f"- disagreement_count: {int(contradiction_summary.get('disagreement_count', 0))}",
            f"- disagreement_rate: {float(contradiction_summary.get('disagreement_rate', 0.0)):.3f}",
            f"- trust_penalty: {float(local_threat.get('trust_penalty', contradiction_summary.get('trust_penalty', 0.0))):.3f}",
            f"- source_pair_conflicts: {', '.join(conflict_patterns) if conflict_patterns else 'none'}",
        ]
        lines.append("")
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
    lines = [
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
        contradiction_summary = summarize_contradictions(threat_rows)
        lines += ["", "## Threat Evidence", ""]
        lines += [
            "| threat_type | score | uncertainty | sensor_sources | disagreeing_sources | recommended_action |",
            "|-------------|-------|-------------|----------------|---------------------|--------------------|",
        ]
        for row in threat_rows:
            lines.append(
                f"| {row['threat_type']} | {row['score']:.3f} | {row['uncertainty']:.3f} | "
                f"{'; '.join(row['sensor_sources']) or 'none'} | "
                f"{'; '.join(row['disagreeing_sources']) or 'none'} | "
                f"{row['recommended_action']} |"
            )
        lines += [
            "",
            "## Contradiction Metrics",
            "",
            f"- disagreement_count: {int(contradiction_summary.get('disagreement_count', 0))}",
            f"- disagreement_rate: {float(contradiction_summary.get('disagreement_rate', 0.0)):.3f}",
            f"- trust_penalty: {float(video_context.get('local_threat', {}).get('trust_penalty', contradiction_summary.get('trust_penalty', 0.0))):.3f}",
        ]
        lines += ["", "## Threat Provenance", ""]
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

    lines += ["", "## Agentic Analysis", ""]
    if llm_analysis.strip():
        lines.append(llm_analysis.strip())
    else:
        lines.append("Reasoning analysis unavailable.")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · final agentic audit step*"]
    write_markdown_artifact(output_path, lines)
    _log.info("  [ok] Written %s", output_path)


# -- Run statistics printer ----------------------------------------------------

# (timing_key, step_label, computation_type)
# Ordered by typical execution sequence.
_STEP_LABELS: list[tuple[str, str, str]] = [
    ("A_extract", "01 Ingest: Frame extraction", "I/O"),
    ("B_index", "02 Ingest: Vector indexing", "GPU embed"),
    ("J_gemma", "03 Analyze: Gemma multimodal", "LLM API"),
    ("L_caption", "04 Analyze: Florence captions", "GPU vision"),
    ("L_seg_caps", "04b Analyze: Gemma segment diffs", "LLM API"),
    ("M_asr", "05 Analyze: ASR transcription", "GPU speech"),
    ("N_ocr", "06 Analyze: OCR text extraction", "LLM API"),
    ("O_depth", "07 Analyze: Depth estimation", "GPU vision"),
    ("P_detection", "08 Analyze: Object detection", "GPU vision"),
    ("P2_yolo_sam", "09 Analyze: YOLO+SAM detection", "GPU vision"),
    ("P3_gemma_tracking", "10 Analyze: Gemma directed tracking", "LLM API+GPU"),
    ("Q_world", "11 Analyze: World model embeddings", "GPU vision"),
    ("R_qwen", "12 Analyze: Qwen detailed captions", "LLM API"),
    ("S_unidrive", "13 Analyze: UniDriveVLA expert", "LLM API"),
    ("S_scenetok", "14 Analyze: SceneTok encoder+seg", "GPU vision"),
    ("C_base_search", "15 Eval: Base search test", "GPU embed"),
    ("I_3dmap", "16 Map: SfM + Gaussian Splat", "GPU 3D"),
    ("PS_physical_state", "17 Analyze: Physical scene state", "CPU fusion"),
    ("PS_field_state", "18 Analyze: Environmental field state", "CPU fusion"),
    ("PS_threat_primitives", "19 Analyze: Threat primitives", "CPU fusion"),
    ("D_finetune", "20 Adapt: SSL DINOv3 fine-tune", "GPU train"),
    ("E_distill", "21 Adapt: Knowledge distillation", "GPU train"),
    ("E_distill_stage2", "21b Adapt: Stage 2 distillation", "GPU train"),
    ("F_export", "22 Export: ONNX + gallery", "CPU"),
    ("G_ft_search", "23 Eval: Fine-tuned search test", "GPU embed"),
    ("H_compare", "24 Eval: Model comparison", "GPU embed"),
    ("T_multimodel", "25 Audit: Multi-model comparison", "GPU vision"),
    ("PS_local_threat", "26 Analyze: Local threat inference", "CPU fusion"),
    ("PS_policy", "27 Decide: Action policy", "CPU policy"),
    ("Z_synthesis", "28 Synthesize: Ontology+narrative", "LLM API"),
    ("AA_agentic", "29 Audit: Agentic flow", "LLM API"),
    ("AC_drone_detection", "31 Train: Drone detection", "GPU train"),
    ("AB_model_advisor", "32 Optimize: Model/run advisor", "CPU analysis"),
]


def _fmt_sec(sec: float) -> str:
    if math.isnan(sec) or sec < 0:
        return "—"
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if sec >= 60:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}m {s:04.1f}s"
    return f"{sec:.1f}s"


def print_run_stats(
    per_video: list[dict[str, Any]],
    total_elapsed: float,
    init_elapsed: float,
    device: str,
) -> None:
    from .common import _banner

    # Column widths: step label, computation type, per-video durations, total
    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    label_candidates = [label for _, label, _ in _STEP_LABELS]
    type_candidates = [comp_type for _, _, comp_type in _STEP_LABELS]
    LABEL_W = max(34, max((len(label) for label in label_candidates), default=34) + 1)
    TYPE_W = max(14, max((len(comp) for comp in type_candidates), default=14) + 1)
    DUR_W = max(
        9,
        max((len(name) for name in names), default=9) + 1,
        len(_fmt_sec(total_elapsed)) + 1,
    )
    n_vids = len(per_video)
    W = LABEL_W + TYPE_W + DUR_W * (n_vids + 1) + 4
    SEP = "-" * W

    def _fit_cell(value: str, width: int) -> str:
        value = str(value)
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[: width - 3] + "..."

    def _row(label: str, comp_type: str, *dur_cols: str) -> str:
        row = f"  {label:<{LABEL_W}} {comp_type:<{TYPE_W}}"
        for c in dur_cols:
            row += f"{_fit_cell(c, DUR_W):>{DUR_W}}"
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

    # Group by computation type for the subtotals
    by_type: dict[str, float] = {}
    col_totals = [0.0] * n_vids
    grand_total = 0.0
    for key, label, comp_type in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        # Always show these keys even when 0s — they are explicitly configured
        # steps whose absence signals a skipped/unavailable sidecar, not that
        # the step is irrelevant.  All other 0s rows are omitted to reduce noise.
        _always_show = {"A_extract", "B_index", "S_scenetok", "S_unidrive"}
        if total_step > 0 or key in _always_show:
            if total_step == 0 and key in _always_show:
                _log.info(_row(label + " (skipped)", comp_type, *["—"] * n_vids, "—"))
            else:
                _log.info(
                    _row(label, comp_type, *[_fmt_sec(s) for s in vals], _fmt_sec(total_step))
                )
            for i, s in enumerate(vals):
                col_totals[i] += s
            grand_total += total_step
        by_type[comp_type] = by_type.get(comp_type, 0.0) + total_step
    _log.info("  " + SEP)
    _log.info(_row("TOTAL", "", *[_fmt_sec(s) for s in col_totals], _fmt_sec(grand_total)))

    _log.info("  " + SEP)
    pipeline_per_video = [v.get("pipeline_sec", 0.0) for v in per_video]
    _log.info(
        _row(
            "Pipeline (steps sum)",
            "",
            *[_fmt_sec(s) for s in pipeline_per_video],
            _fmt_sec(sum(pipeline_per_video)),
        )
    )
    pipeline_sum = sum(pipeline_per_video)
    overlap_adjustment = max(0.0, pipeline_sum + init_elapsed - total_elapsed)
    overhead = total_elapsed - pipeline_sum - init_elapsed + overlap_adjustment
    _log.info(_row("Model initialisation", "", _fmt_sec(init_elapsed), *([""] * (n_vids - 1)), ""))
    if overlap_adjustment > 0:
        _log.info(
            _row(
                "Concurrent overlap adjustment",
                "",
                *([""] * n_vids),
                f"-{_fmt_sec(overlap_adjustment)}",
            )
        )
    _log.info(
        _row("Overhead (I/O, viewer, etc.)", "", *([""] * n_vids), _fmt_sec(max(0.0, overhead)))
    )
    _log.info(_row("WALL CLOCK TOTAL", "", *([""] * n_vids), _fmt_sec(total_elapsed)))

    # -- Computation-type subtotals ---------------------------------------------
    _log.info("")
    _log.info("  COMPUTATION TYPE BREAKDOWN  (pipeline steps only)")
    _log.info("  " + "-" * (TYPE_W + DUR_W + LABEL_W + 2))
    TYPE_ORDER = [
        "I/O",
        "GPU embed",
        "GPU vision",
        "GPU speech",
        "GPU 3D",
        "GPU train",
        "CPU",
        "LLM API",
        "LLM API+GPU",
    ]
    for ct in TYPE_ORDER:
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
        _log.info(
            "  %-26s  extract: %5.1f fr/s   index: %5.1f fr/s",
            v.get("name", "?"),
            frames / t_extract,
            frames / t_index,
        )
    _log.info("")
    _log.info("  MODEL METRICS")
    _log.info("  " + SEP[: W - 2])
    _log.info(_row("Metric", *names))
    _log.info("  " + SEP[: W - 2])
    _log.info(
        _row("SSL finetune loss", *[f"{v.get('best_loss', float('nan')):.4f}" for v in per_video])
    )
    _log.info(
        _row(
            "Distill loss",
            *[
                f"{v.get('distill_loss', float('nan')):.4f}"
                if not math.isnan(v.get("distill_loss", float("nan")))
                else "skipped"
                for v in per_video
            ],
        )
    )
    _log.info(_row("Teacher ckpt (MB)", *[f"{v.get('ckpt_mb', 0.0):.1f}" for v in per_video]))
    _log.info(
        _row(
            "Student ckpt (MB)",
            *[
                f"{v.get('student_ckpt_mb', 0.0):.1f}" if v.get("student_ckpt_mb") else "—"
                for v in per_video
            ],
        )
    )
    _log.info(
        _row(
            "ONNX size (MB)",
            *[f"{v.get('onnx_mb', 0.0):.1f}" if v.get("onnx_exported") else "—" for v in per_video],
        )
    )
    _log.info(
        _row(
            "Compression ratio",
            *[
                f"{v['distill_compression_ratio']:.1f}×"
                if v.get("distill_compression_ratio")
                else (
                    f"{v['teacher_dim'] / v['student_dim']:.1f}×"
                    if v.get("student_dim") and v.get("teacher_dim")
                    else "—"
                )
                for v in per_video
            ],
        )
    )
    _log.info(
        _row("Base infer (ms/fr)", *[f"{v.get('base_infer_ms', 0.0):.1f}" for v in per_video])
    )
    _log.info(
        _row("Fine-tuned infer (ms/fr)", *[f"{v.get('ft_infer_ms', 0.0):.1f}" for v in per_video])
    )
    _log.info("")
    _log.info("  SEARCH QUALITY  (top-1 cosine score, self/near-temporal matches excluded)")
    _log.info("  " + SEP[: W - 2])
    _log.info(
        _row("Base model (pretrained)", *[f"{v.get('base_top_score', 0.0):.4f}" for v in per_video])
    )
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
    _log.info(
        _row(
            "Domain",
            *[(v.get("analysis_summary", {}) or {}).get("domain") or "—" for v in per_video],
        )
    )
    _log.info(
        _row(
            "Top category",
            *[(v.get("analysis_summary", {}) or {}).get("top_category") or "—" for v in per_video],
        )
    )
    _log.info(
        _row(
            "Artifacts",
            *[
                str((v.get("analysis_summary", {}) or {}).get("artifact_count", "—"))
                for v in per_video
            ],
        )
    )
    _log.info(
        _row(
            "Coverage F/Q/A/O",
            *[_fmt_analytics_coverage(v.get("analysis_summary", {}) or {}) for v in per_video],
        )
    )
    _log.info(
        _row(
            "Detections",
            *[_fmt_analytics_detections(v.get("analysis_summary", {}) or {}) for v in per_video],
        )
    )
    _log.info(
        _row(
            "Temporal",
            *[_fmt_analytics_temporal(v.get("analysis_summary", {}) or {}) for v in per_video],
        )
    )
    _log.info(
        _row(
            "World/Tracking",
            *[
                _fmt_analytics_world_tracking(v.get("analysis_summary", {}) or {})
                for v in per_video
            ],
        )
    )
    _log.info(
        _row(
            "Map quality",
            *[_fmt_analytics_map(v.get("analysis_summary", {}) or {}) for v in per_video],
        )
    )
    _log.info(
        _row(
            "Warnings",
            *[_fmt_analytics_warnings(v.get("analysis_summary", {}) or {}) for v in per_video],
        )
    )
    _log.info("")
    _log.info("  TOP VIDEO DESCRIPTION  (CLIP text similarity)")
    _log.info("  " + SEP[: W - 2])
    for v in per_video:
        _log.info("  %-20s  %s", v.get("name", "?"), v.get("top_description", "—") or "—")
    _log.info("")
    _log.info("  " + "=" * (W - 2))


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

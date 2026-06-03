"""Adaptation report writers: SSL fine-tuning and distillation stats — rendered via Jinja2."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment

from ..common import (
    _RUNNER_LABEL,
    _log,
    write_markdown_artifact,
)

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
_env.filters["rjust"] = lambda s, w: str(s).rjust(w)

_FINETUNE_STATS_TEMPLATE = """\
# SSL Fine-Tuning Statistics — {{ video_name }}

Generated: {{ generated }}

## What We Do

**Self-Supervised Learning (SSL)** adapts a pre-trained vision backbone to the
specific visual domain of this mission without any labelled annotations.

### Method: NT-Xent Contrastive Loss

We use **NT-Xent** (Normalised Temperature-scaled Cross Entropy, a.k.a. InfoNCE)
contrastive learning:

1. Each training step produces a batch of *positive pairs* (two views of the same scene).
2. The model encodes both views through the DINOv3 backbone + a small projection head
   (embed_dim={{ cfg.embed_dim }} → proj_dim={{ cfg.proj_out_dim }}).
3. The loss pushes the two views of the same scene together in embedding space
   and pushes all other pairs in the batch apart.
4. Temperature τ={{ cfg.temperature }} controls the sharpness of the distribution
   (lower = harder negatives, more informative but less stable).

The backbone is **partially frozen**: the first {{ cfg.freeze_blocks }} transformer blocks
are kept fixed (preserving generic low-level features), and only the top
{{ 12 - cfg.freeze_blocks }} blocks + projection head are trained.
This prevents catastrophic forgetting on a small video dataset.

### Pair Construction Strategy: `{{ cfg.approach }}`

{% if cfg.approach == "track_cycle" %}
**Track cycle-consistency triplets** — RF-DETR track IDs provide triplets
(A, B, C) of the same tracked object at times t, t+k, t+2k.
Loss: NTXent(A,B) + NTXent(B,C) + 0.3·NTXent(A,C).
The cycle term enforces that object identity is stable across the widest
temporal gap in the triplet, preventing embedding drift along long tracks.
{% elif cfg.approach == "track" %}
**Track pairs** — RF-DETR track IDs provide pairs (A, B) of the same
tracked object at two different times (gap 2–5 appearances).
Crops are taken around the tracked bbox with 15 % padding.
Rationale: same-object pairs encode identity consistency that full-frame
temporal pairs cannot — the model must match the object across viewpoint
and appearance changes, not just spatial proximity.
{% elif cfg.approach == "temporal" %}
**Temporal pairs** — consecutive frames within ±{{ cfg.max_gap }} positions
in the frame sequence form positive pairs.
Rationale: adjacent frames in a 30 fps outdoor video show nearly the same scene,
so pulling their embeddings together teaches the model scene-level consistency
while naturally using real mission content (no synthetic augmentation needed).
{% else %}
**Augmentation pairs** — each frame is augmented twice with random crops,
horizontal flips, colour jitter, and Gaussian blur.
Rationale: fewer than {{ cfg.batch_size * 2 }} frames are available, so temporal
pairing would produce too few unique positive pairs.
Augmentation-based SSL is used as a fallback.
{% endif %}

### Optimiser

| Component | Setting |
|-----------|---------|
| Optimiser | AdamW |
| Learning rate | {{ cfg.lr }} |
| Weight decay | {{ cfg.weight_decay }} |
| LR schedule | Cosine annealing over {{ cfg.epochs }} epochs |
| Batch size | {{ cfg.batch_size }} pairs |

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `{{ cfg.model_name }}` |
| Approach | `{{ cfg.approach }}` |
| Epochs | {{ cfg.epochs }} |
| Batch size | {{ cfg.batch_size }} |
| Learning rate | {{ cfg.lr }} |
| Temperature | {{ cfg.temperature }} |
| Frozen blocks | {{ cfg.freeze_blocks }} / 12 |
| Embed dim | {{ cfg.embed_dim }} → proj {{ cfg.proj_out_dim }} |
| Device | `{{ cfg.device }}` |

## Results

| Metric | Value |
|--------|-------|
| Best loss | {{ "%.4f"|format(best_loss) }} |
| Best epoch | {{ best_epoch }}/{{ cfg.epochs }} |
| First loss | {{ "%.4f"|format(stats.first_loss) }} |
| Last loss | {{ "%.4f"|format(stats.last_loss) }} |
| Total drop | {{ "%.1f"|format(stats.drop_pct) }} % |
| Convergence epoch | {{ stats.convergence_epoch or '—' }} |
| Training time | {{ elapsed_sec }}s |
| Checkpoint size | {{ "%.1f"|format(ckpt_mb) }} MB |
| Checkpoint path | `{{ checkpoint_path }}` |

## Result Analysis

{% for b in bullets %}
- {{ b }}

{% endfor %}
## Loss Curve

```
high │{{ sparkline }}│
 low │{{ '-' * sparkline | length }}│
      epoch 1{{ '' | rjust([0, sparkline | length - 9] | max) }}epoch {{ loss_history | length }}
```

*Each character represents {{ 'one epoch' if loss_history | length <= 40 else 'a range of epochs' }}.
Higher bar = higher loss.*

| Epoch | Loss | Δ vs prev | Trend |
|-------|------|-----------|-------|
{% for ep in range(1, loss_history | length + 1) %}
{% set loss = loss_history[ep - 1] %}
{% if ep == 1 %}
| {{ ep }} | {{ "%.4f"|format(loss) }} | — | — {{ ' ← best' if ep == best_epoch else '' }} |
{% else %}
{% set d = deltas[ep - 2] %}
| {{ ep }} | {{ "%.4f"|format(loss) }} | {{ "%+.4f"|format(d) }} | {{ '↓ improving' if d < -0.01 else ('↑ worsening' if d > 0.01 else '→ stable') }}{{ ' ← best' if ep == best_epoch else '' }} |
{% endif %}
{% endfor %}

## How to Use This Checkpoint

```bash
export DINO_CHECKPOINT={{ checkpoint_path }}
python main.py --mode local --videos-dir .data/videos
```

---
*Artifact produced by {{ runner_label }}. See `edge_models/` for ONNX export.*
"""


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: Any,
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: list[float],
) -> None:
    from ..adaptation.ssl import _analyze_loss_curve, _interpret_finetune_results, _loss_sparkline

    ckpt_mb = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0
    best_epoch = int(np.argmin(loss_history)) + 1 if loss_history else 0
    stats = _analyze_loss_curve(loss_history)
    sparkline = _loss_sparkline(loss_history)
    bullets = _interpret_finetune_results(cfg, stats, elapsed_sec)

    tmpl = _env.from_string(_FINETUNE_STATS_TEMPLATE)

    content = tmpl.render(
        video_name=video_name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cfg=cfg,
        best_loss=best_loss,
        best_epoch=best_epoch,
        stats=stats,
        sparkline=sparkline,
        deltas=stats.get("deltas", []),
        bullets=bullets,
        loss_history=loss_history,
        ckpt_mb=ckpt_mb,
        checkpoint_path=checkpoint_path,
        elapsed_sec=f"{elapsed_sec:.1f}",
        runner_label=_RUNNER_LABEL,
    )
    write_markdown_artifact(output_path, content.splitlines())
    _log.info("  [ok] Written %s", output_path)


def write_distill_stats_md(
    output_path: Path,
    video_name: str,
    stats: dict[str, Any],
) -> None:
    loss_history = stats.get("loss_history", [])
    recall_history = stats.get("recall_history", [])
    lc = stats.get("loss_components", {})
    compression = stats.get("compression_ratio", 0.0)
    t_params = stats.get("teacher_params", 0)
    s_params = stats.get("student_params", 0)
    best_recall = stats.get("best_recall", float("nan"))

    def _component(key: str, i: int) -> str:
        vals = lc.get(key, [])
        return f"{vals[i]:.4f}" if i < len(vals) else "nan"

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
    for i in range(len(loss_history)):
        r1 = recall_history[i] if i < len(recall_history) else float("nan")
        lines.append(
            f"| {i + 1} | {loss_history[i]:.4f} | {_component('rkd_d', i)} | "
            f"{_component('rkd_a', i)} | {_component('cosine', i)} | "
            f"{_component('koleo', i)} | {r1:.3f} |"
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

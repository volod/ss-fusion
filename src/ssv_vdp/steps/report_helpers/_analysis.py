"""Gemma multimodal analysis report — rendered via Jinja2."""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment

from ..common import (
    _RUNNER_LABEL,
    _SCENE_CHANGE_THRESH,
    _log,
    write_markdown_artifact,
)

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
# Filters used inside _GEMMA_ANALYSIS_TEMPLATE
_env.filters["basename"] = lambda p: Path(str(p)).name
# Global functions used inside templates (enumerate isn't a Jinja2 builtin)
_env.globals["enumerate"] = enumerate

_GEMMA_ANALYSIS_TEMPLATE = """\
# Gemma Open-Weight Analysis — {{ video_name }}

Generated: {{ generated }}
Model: `{{ model_id }}`  |  Frames sampled: {{ sample_n }}  |  Elapsed: {{ elapsed_sec }}s

## Analyses Performed

| Analysis | Status |
|----------|--------|
{% for key, res in analysis.items() %}
| {{ key.replace('_', ' ').title() }} | {{ '[ok]' if not res.get('error') else '✗ ' + (res.get('error') or '')[:60] }} |
{% endfor %}

{% if dino.available %}
## Gemma vs DINOv3 Embedding Comparison

Both models embedded the same {{ dino.n_frames }} frames.
Gemma model: `{{ model_id }}`.  DINOv3 model: `dinov3_vitb14`.

| Metric | Gemma | DINOv3 |
|--------|-------|--------|
| Mean pairwise cosine similarity | {{ "%.4f"|format(dino.mean_cossim_gemma) }} | {{ "%.4f"|format(dino.mean_cossim_dino) }} |
| Mutual nearest-neighbor overlap (k={{ dino.k }}) | {{ "%.3f"|format(dino.mnn_rate) }} | — |

**Mean pairwise cosine similarity**: lower = more discriminative embedding space.

**MNN@{{ dino.k }}** ({{ "%.1f"|format(dino.mnn_rate * 100) }}%): fraction of frames whose top-{{ dino.k }} visual neighbours agree
between Gemma and DINOv3.

{% else %}
## Gemma vs DINOv3 Embedding Comparison

Skipped: {{ dino.reason | default('DINOv3 not available') }}

{% endif %}
{% if clip.available %}
## Gemma vs CLIP Embedding Comparison

Both models embedded the same {{ clip.n_frames }} frames.
Gemma model: `{{ model_id }}`.  CLIP model: `ViT-B-16/openai`.

| Metric | Gemma | CLIP |
|--------|-------|------|
| Mean pairwise cosine similarity | {{ "%.4f"|format(clip.mean_cossim_gemma) }} | {{ "%.4f"|format(clip.mean_cossim_clip) }} |
| Mutual nearest-neighbor overlap (k={{ clip.k }}) | {{ "%.3f"|format(clip.mnn_rate) }} | — |

**MNN@{{ clip.k }}** ({{ "%.1f"|format(clip.mnn_rate * 100) }}%): fraction of frames whose top-{{ clip.k }} visual neighbours agree
between Gemma and CLIP.

{% elif clip %}
## Gemma vs CLIP Embedding Comparison

Skipped: {{ clip.reason | default('CLIP not available') }}

{% endif %}
{% if sc_data and sc_data.changes is not none %}
## Scene Change Detection

Cosine distance > {{ scene_change_thresh }} between consecutive sampled frames.
Detected {{ sc_data.n_changes }} transition(s).

{% if sc_data.changes %}
| # | t (s) | Cosine Distance |
|---|-------|-----------------|
{% for i, ch in sc_data.changes[:15] | enumerate %}
| {{ i + 1 }} | {{ "%.1f"|format(ch.t_sec) }} | {{ "%.4f"|format(ch.distance) }} |
{% endfor %}
{% endif %}
{% endif %}
{% if clf_data and clf_data.category_distribution %}
## Zero-Shot Scene Classification

Top predicted scene categories across {{ sample_n }} frames:

| Category | Frame Count |
|----------|-------------|
{% for cat, cnt in clf_data.category_distribution.items() %}
| {{ cat }} | {{ cnt }} |
{% endfor %}

{% endif %}
{% if text_query_results %}
## Cross-Modal Text → Frame Retrieval

Text probes (mean-pooled text embeddings) vs frame embeddings (cosine similarity):

| Query | Best Frame (t) | Score |
|-------|---------------|-------|
{% for qr in text_query_results %}
{% set top = qr.top_results %}
| {{ qr.query }} | {% if top %}`{{ top[0].frame_path | basename }}` ({{ "%.1f"|format(top[0].t_sec) }}s) | {{ "%.4f"|format(top[0].score) }}{% else %}— | —{% endif %} |
{% endfor %}

{% endif %}
{% if te_data and not te_data.error %}
## Temporal Video Embedding

Mean-pool of all {{ sample_n }} frame embeddings → single video-level vector
(dim={{ te_data.dim }}).  Can be used for video-level retrieval or comparison.

{% endif %}
{% if cl_data and cl_data.n_clusters %}
## Scene Clustering

{{ cl_data.n_clusters }} semantic clusters from {{ sample_n }} frames
(mean cluster size: {{ "%.1f"|format(cl_data.mean_cluster_size) }} frames).

{% endif %}
## Findings & Interpretation

{% if dino.available %}
{% if dino.mean_cossim_gemma < dino.mean_cossim_dino %}
- **Gemma is more discriminative than DINOv3** for this video \
(mean cosine {{ "%.4f"|format(dino.mean_cossim_gemma) }} < {{ "%.4f"|format(dino.mean_cossim_dino) }}). \
Gemma's language-grounded embeddings spread frames further apart in embedding space — useful for precise retrieval.
{% elif (dino.mean_cossim_gemma - dino.mean_cossim_dino) | abs < 0.05 %}
- **Gemma and DINOv3 have similar discrimination** \
(cosine {{ "%.4f"|format(dino.mean_cossim_gemma) }} vs {{ "%.4f"|format(dino.mean_cossim_dino) }}). \
Both models capture similar visual structure for this mission content.
{% else %}
- **DINOv3 is more discriminative than Gemma** for this video \
(cosine {{ "%.4f"|format(dino.mean_cossim_dino) }} < {{ "%.4f"|format(dino.mean_cossim_gemma) }}). \
DINOv3's self-supervised visual features give finer-grained distinctions. \
Gemma remains valuable for language-grounded queries.
{% endif %}
{% if dino.mnn_rate >= 0.8 %}
- **High DINOv3↔Gemma agreement (MNN={{ "%.1f"|format(dino.mnn_rate * 100) }}%)**: both models agree on which \
frames are visually similar. Gemma embeddings can safely substitute DINOv3 for \
retrieval with additional benefit of text-query compatibility.
{% elif dino.mnn_rate >= 0.5 %}
- **Moderate DINOv3↔Gemma agreement (MNN={{ "%.1f"|format(dino.mnn_rate * 100) }}%)**: the models partially \
disagree on visual neighbourhoods. Gemma captures semantic similarity; DINOv3 \
captures low-level visual similarity. Both are complementary — use Gemma for \
text queries, DINOv3 for image-to-image search.
{% else %}
- **Low DINOv3↔Gemma agreement (MNN={{ "%.1f"|format(dino.mnn_rate * 100) }}%)**: the models assign very \
different neighbourhoods. Likely cause: 30 fps near-duplicate frames collapse \
to the same DINOv3 cluster while Gemma's language bias separates them differently. \
This is expected and not a failure — the two spaces serve different query types.
{% endif %}

{% endif %}
{% if clip.available %}
{% if clip.mnn_rate >= 0.8 %}
- **High CLIP↔Gemma agreement (MNN={{ "%.1f"|format(clip.mnn_rate * 100) }}%)**: Gemma embeddings are \
strongly aligned with CLIP's image-text space. Gemma can replace CLIP for \
cross-modal retrieval while also supporting image-to-image search.
{% elif clip.mnn_rate >= 0.5 %}
- **Moderate CLIP↔Gemma agreement (MNN={{ "%.1f"|format(clip.mnn_rate * 100) }}%)**: Gemma and CLIP agree \
on roughly half of visual neighbourhoods. Use CLIP for image-text matching \
and Gemma for richer structured reasoning.
{% else %}
- **Low CLIP↔Gemma agreement (MNN={{ "%.1f"|format(clip.mnn_rate * 100) }}%)**: Gemma organises this \
visual content differently from CLIP. Gemma may be using scene-level semantics \
while CLIP relies on global appearance statistics.
{% endif %}

{% endif %}
{% if sc_data and sc_data.changes is not none %}
{% if sc_data.n_changes == 0 %}
- **No scene transitions detected**: all {{ sample_n }} sampled frames are \
visually continuous. This is typical of 30 fps missions where scenes evolve slowly. \
Use the Scene Timeline in `scene_captions.md` for segment-level analysis.
{% elif sc_data.n_changes <= 3 %}
- **{{ sc_data.n_changes }} scene transition(s)**: the video has a small number of \
distinct visual states. Gemma embedding distances reliably flag these transitions \
as higher-priority frames for annotation (`al_tag=needs_annotation`).
{% else %}
- **{{ sc_data.n_changes }} scene transitions**: high visual variability in this mission. \
Frames at transition boundaries carry the most novel information and should be \
prioritised for SSL training data.
{% endif %}

{% endif %}
{% if cl_data and cl_data.n_clusters %}
{% if cl_data.mean_cluster_size > sample_n * 0.3 %}
- **Few, large clusters ({{ cl_data.n_clusters }} clusters, ~{{ "%.0f"|format(cl_data.mean_cluster_size) }} frames each)**: \
the mission covers a small set of visually distinct scenes. \
SSL temporal pairs will be highly informative — nearby frames share the same cluster.
{% else %}
- **Many small clusters ({{ cl_data.n_clusters }} clusters, ~{{ "%.0f"|format(cl_data.mean_cluster_size) }} frames each)**: \
high scene diversity. More SSL epochs may be needed to cover all visual states.
{% endif %}

{% endif %}
{% if dino.available %}
{% if dino.mnn_rate >= 0.7 %}
- **Distillation recommendation**: Gemma embeddings are a strong teacher signal. \
Set `gemma_embedder` in `step_distill` (done automatically when `MODEL_NAME=gemma`) \
for maximum-hydration distillation — the student inherits both visual and language-grounded structure.
{% else %}
- **Distillation recommendation**: Gemma and DINOv3 neighbourhoods diverge for \
this content. Run both distillation chains and compare Recall@1: \
DINOv3-teacher for image retrieval, Gemma-teacher for text-query tasks.
{% endif %}

{% endif %}
---
*Produced by {{ runner_label }} — Gemma open-weight multimodal analysis.*
"""


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
    cc = clip_comparison or {}
    sc_data = analysis.get("scene_change_detection", {})
    clf_data = analysis.get("scene_classification", {})
    te_data = analysis.get("temporal_embedding", {})
    cl_data = analysis.get("scene_clustering", {})

    tmpl = _env.from_string(_GEMMA_ANALYSIS_TEMPLATE)

    content = tmpl.render(
        video_name=video_name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_id=model_id,
        sample_n=sample_n,
        elapsed_sec=f"{elapsed_sec:.1f}",
        analysis=analysis,
        dino=dino_comparison,
        clip=cc,
        sc_data=sc_data if not sc_data.get("error") else None,
        clf_data=clf_data if not clf_data.get("error") else None,
        te_data=te_data if not te_data.get("error") else None,
        cl_data=cl_data if not cl_data.get("error") else None,
        text_query_results=text_query_results,
        scene_change_thresh=_SCENE_CHANGE_THRESH,
        runner_label=_RUNNER_LABEL,
    )
    write_markdown_artifact(output_path, content.splitlines())
    _log.info("  [ok] Written %s", output_path)

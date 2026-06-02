"""Per-video comparison, description and multi-model contrast steps."""

import time
from pathlib import Path
from typing import Any

from PIL import Image

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.core.logging import get_logger
from ...steps.common import _TEXT_PROMPTS

_log = get_logger(__name__)



def step_compare_and_describe(
    frame_list: list[tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    base_results: list[dict],
    ft_results: list[dict],
    models: dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    ckpt_mb: float,
    onnx_mb: float,
) -> dict[str, Any]:
    """Step 20: compare results, caption video, write comparison.md."""
    from ...steps.report import write_comparison_md, write_description_md

    out_md = video_dir / "comparison.md"
    sample_paths = [fp for fp, _ in frame_list[:10]]
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    # Single warm-up pass so CUDA clock-scaling and kernel-JIT don't inflate
    # the first timed batch (especially relevant after CPU-heavy steps).
    _warmup_img = [Image.open(sample_paths[0]).convert("RGB")]
    clip_model.encode_images(_warmup_img)
    if dino_model:
        dino_model.encode_images(_warmup_img)
    t0 = time.time()
    clip_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
    base_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    ft_infer_ms = base_infer_ms
    if dino_model:
        t0 = time.time()
        dino_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
        ft_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    _log.info("Computing video-to-text description …")
    try:
        step = max(1, len(frame_list) // 32)
        sampled_imgs = [Image.open(fp).convert("RGB") for fp, _ in frame_list[::step]]
        frame_embeds = clip_model.encode_images(sampled_imgs)
        avg_embed = frame_embeds.mean(axis=0)
        text_embeds = clip_model.encode_texts(_TEXT_PROMPTS)
        scores = text_embeds @ avg_embed
        ranked = sorted(zip(_TEXT_PROMPTS, scores.tolist()), key=lambda x: x[1], reverse=True)
        text_descriptions = ranked[:3]
        all_scored = ranked
        for desc, score in text_descriptions:
            _log.info('  Video description: "%s" (sim=%.3f)', desc, score)
    except Exception as exc:
        _log.warning("  Video-to-text failed (%s)", exc)
        text_descriptions = [("description unavailable", 0.0)]
        all_scored = text_descriptions
    write_comparison_md(
        out_md,
        video_name,
        base_results,
        ft_results,
        base_infer_ms,
        ft_infer_ms,
        ckpt_mb,
        onnx_mb,
        text_descriptions,
    )
    desc_md = video_dir / "description.md"
    write_description_md(desc_md, video_name, frame_list, text_descriptions, all_scored)
    return {
        "text_descriptions": text_descriptions,
        "base_infer_ms": base_infer_ms,
        "ft_infer_ms": ft_infer_ms,
        "top_description": text_descriptions[0][0] if text_descriptions else "",
    }


def step_multi_model_compare(
    video_name: str,
    video_dir: Path,
    gemma_result: dict[str, Any],
    qwen_result: dict[str, Any],
    unidrive_result: dict[str, Any],
) -> dict[str, Any]:
    """Write a Gemma vs Qwen vs UniDriveVLA comparison artifact."""
    from ...steps.report import write_multi_model_comparison_md

    out_md = video_dir / "multi_model_comparison.md"
    return write_multi_model_comparison_md(
        out_md,
        video_name,
        gemma_result,
        qwen_result,
        unidrive_result,
    )


# -- Agentic video synthesis helpers -------------------------------------------

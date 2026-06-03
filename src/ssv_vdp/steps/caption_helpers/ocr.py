"""OCR frame scoring and candidate selection."""

import math
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger("pipeline.local.caption")

_OCR_TEXT_HINT_TERMS = (
    "text",
    "sign",
    "label",
    "banner",
    "billboard",
    "license plate",
    "plate",
    "screen",
    "display",
    "storefront",
    "marking",
    "road sign",
    "street sign",
    "poster",
    "numbers",
    "letters",
)

_OCR_TEXTLESS_SCENE_TERMS = (
    "sky",
    "field",
    "vegetation",
    "forest",
    "water",
    "farmland",
    "open terrain",
)

_OCR_TEMPORAL_MIN_GAP_SEC = 0.75


def _caption_keyword_score(
    text: str,
    positive_terms: tuple[str, ...],
    negative_terms: tuple[str, ...] = (),
) -> float:
    lowered = " ".join((text or "").strip().lower().split())
    if not lowered:
        return 0.0
    score = 0.0
    for term in positive_terms:
        if term in lowered:
            score += 1.0
    for term in negative_terms:
        if term in lowered:
            score -= 0.4
    return score


def _estimate_ocr_frame_score(
    frame_path: str,
    t_sec: float,
    caption_text: str,
    caption_confidence: float | None,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    """Estimate how likely a frame is to contain useful visible text."""
    diagnostics = {
        "caption_uncertainty": 0.0,
        "caption_hint": 0.0,
        "contrast": 0.0,
        "edge_density": 0.0,
        "stripe_variation": 0.0,
        "timing_bias": 0.0,
    }
    score = 0.0

    conf = None if caption_confidence is None else float(caption_confidence)
    if conf is not None:
        uncertainty = max(0.0, threshold - conf) / max(threshold, 1e-6)
        diagnostics["caption_uncertainty"] = uncertainty
        score += 1.6 * uncertainty

    hint_score = _caption_keyword_score(
        caption_text, _OCR_TEXT_HINT_TERMS, _OCR_TEXTLESS_SCENE_TERMS
    )
    diagnostics["caption_hint"] = hint_score
    score += 0.8 * hint_score

    diagnostics["timing_bias"] = 0.08 * math.sin(min(max(t_sec, 0.0), 30.0) / 30.0 * math.pi)
    score += diagnostics["timing_bias"]

    try:
        img = Image.open(frame_path).convert("L")
        img = ImageOps.autocontrast(img)
        img.thumbnail((320, 320))
        stat = ImageStat.Stat(img)
        stddev = float(stat.stddev[0] if stat.stddev else 0.0)
        contrast = min(1.0, stddev / 64.0)
        diagnostics["contrast"] = contrast
        score += 0.7 * contrast

        edges = img.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_mean = float(edge_stat.mean[0] if edge_stat.mean else 0.0)
        edge_density = min(1.0, edge_mean / 72.0)
        diagnostics["edge_density"] = edge_density
        score += 0.9 * edge_density

        stripe_height = max(4, img.height // 10)
        stripe_means = []
        for y in range(0, img.height, stripe_height):
            box = (0, y, img.width, min(img.height, y + stripe_height))
            band_stat = ImageStat.Stat(img.crop(box))
            stripe_means.append(float(band_stat.mean[0] if band_stat.mean else 0.0))
        if len(stripe_means) > 1:
            stripe_span = max(stripe_means) - min(stripe_means)
            stripe_variation = min(1.0, stripe_span / 96.0)
            diagnostics["stripe_variation"] = stripe_variation
            score += 0.5 * stripe_variation
    except Exception:
        pass

    return score, diagnostics


def _fallback_ocr_frame_sample(
    frame_list: list[tuple[str, float]],
    max_samples: int = 8,
) -> list[tuple[str, float]]:
    """Select a small evenly spaced OCR subset when caption prescreen selects none."""
    if len(frame_list) <= max_samples:
        return list(frame_list)
    last = len(frame_list) - 1
    indices = sorted({round(i * last / max(max_samples - 1, 1)) for i in range(max_samples)})
    return [frame_list[i] for i in indices]


def _select_ocr_candidate_frames(
    frame_list: list[tuple[str, float]],
    caption_results: list[dict[str, Any]] | None,
    ocr_model_id: str,
    threshold: float,
    max_ocr: int,
) -> tuple[list[tuple[str, float]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    caption_conf_by_frame: dict[str, float] = {}
    caption_text_by_frame: dict[str, str] = {}
    if caption_results:
        caption_conf_by_frame = {
            str(r.get("frame_path")): float(r.get("caption_confidence", 0.0) or 0.0)
            for r in caption_results
            if r.get("frame_path")
        }
        caption_text_by_frame = {
            str(r.get("frame_path")): str(r.get("caption", "") or "")
            for r in caption_results
            if r.get("frame_path")
        }

    selected = list(frame_list)
    skipped: dict[str, dict[str, Any]] = {}
    ranking: list[dict[str, Any]] = []

    if threshold > 0.0 and caption_conf_by_frame:
        for fp, t_sec in frame_list:
            conf = caption_conf_by_frame.get(fp)
            caption_text = caption_text_by_frame.get(fp, "")
            score, diagnostics = _estimate_ocr_frame_score(fp, t_sec, caption_text, conf, threshold)
            ranking.append(
                {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "score": score,
                    "caption_confidence": conf if conf is not None else -1.0,
                    **diagnostics,
                }
            )
        ranking.sort(key=lambda item: (float(item["score"]), -float(item["t_sec"])), reverse=True)
        if max_ocr > 0 and len(ranking) > max_ocr:
            selected_ranked: list[dict[str, Any]] = []
            for item in ranking:
                if len(selected_ranked) >= max_ocr:
                    break
                if any(
                    abs(float(item["t_sec"]) - float(prev["t_sec"])) < _OCR_TEMPORAL_MIN_GAP_SEC
                    for prev in selected_ranked
                ):
                    continue
                selected_ranked.append(item)
            if len(selected_ranked) < max_ocr:
                used = {str(item["frame_path"]) for item in selected_ranked}
                for item in ranking:
                    if len(selected_ranked) >= max_ocr:
                        break
                    if str(item["frame_path"]) in used:
                        continue
                    selected_ranked.append(item)
            selected_paths = {str(item["frame_path"]) for item in selected_ranked}
            selected = [(fp, t_sec) for fp, t_sec in frame_list if fp in selected_paths]
            for fp, t_sec in frame_list:
                if fp not in selected_paths:
                    skipped[fp] = {
                        "frame_path": fp,
                        "t_sec": t_sec,
                        "ocr_text": "",
                        "ocr_model": ocr_model_id,
                        "ocr_skipped_by_rank": True,
                    }
            ranking = selected_ranked
        else:
            selected = list(frame_list)
    elif threshold > 0.0 and max_ocr > 0 and len(frame_list) > max_ocr:
        # Caption prescreen confidences are unavailable — OCR runs concurrently
        # with Florence captioning in the parallel phase, so caption_results is
        # usually empty when OCR starts. Without a ranking we would otherwise OCR
        # every frame; at ~20 s/frame on a VLM sidecar that is the pipeline's
        # single largest cost (51 frames ≈ 17 min). Cap to an evenly spaced
        # subset of OCR_MAX_FRAMES so OCR stays bounded regardless of ordering.
        selected = _fallback_ocr_frame_sample(frame_list, max_ocr)
        selected_paths = {fp for fp, _ in selected}
        for fp, t_sec in frame_list:
            if fp not in selected_paths:
                skipped[fp] = {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "ocr_text": "",
                    "ocr_model": ocr_model_id,
                    "ocr_skipped_by_rank": True,
                }
    return selected, skipped, ranking

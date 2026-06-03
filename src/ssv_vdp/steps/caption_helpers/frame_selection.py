"""Frame subset selection for LLM/VLM steps and segment boundary ranking."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..common import _open_frame_image

if TYPE_CHECKING:
    from ..common import VideoKnowledge

_log = get_logger("pipeline.local.caption")

_SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT = 16
_SEGMENT_DIFF_MIN_JACCARD_DELTA = 0.18


def _reduce_llm_sample_frames(
    frame_list: list[tuple[str, float]],
    *,
    max_frames: int,
) -> list[tuple[str, float]]:
    """Reduce near-duplicate sampled frames for LLM-heavy Gemma steps."""
    import numpy as np

    if len(frame_list) <= max_frames:
        return frame_list
    step = max(1, len(frame_list) // max_frames)
    sampled = frame_list[::step][:max_frames]
    kept: list[tuple[str, float]] = []
    prev_small = None
    for fp, t_sec in sampled:
        try:
            img = _open_frame_image(fp).convert("L").resize((64, 64))
            small = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            kept.append((fp, t_sec))
            continue
        if prev_small is None:
            kept.append((fp, t_sec))
            prev_small = small
            continue
        diff = float(np.mean(np.abs(small - prev_small)))
        if diff >= float(settings.GEMMA_STABLE_FRAME_DIFF_THRESHOLD):
            kept.append((fp, t_sec))
            prev_small = small
    min_keep = min(len(sampled), int(settings.GEMMA_MIN_SAMPLE_FRAMES))
    if len(kept) < min_keep:
        seen = {fp for fp, _ in kept}
        for fp, t_sec in sampled:
            if fp in seen:
                continue
            kept.append((fp, t_sec))
            seen.add(fp)
            if len(kept) >= min_keep:
                break
    return kept


def _adaptive_sparse_budget(
    frame_list: list[tuple[str, float]],
    *,
    configured_max: int,
    seconds_per_sample: float,
    floor: int,
) -> int:
    """Scale sparse expert-pass budgets down on short clips."""
    if not frame_list:
        return max(1, min(configured_max, floor))
    duration_sec = max(0.0, float(frame_list[-1][1]) - float(frame_list[0][1]))
    duration_budget = int(duration_sec / max(seconds_per_sample, 1e-6)) + 1
    return max(1, min(configured_max, max(floor, duration_budget)))


def _select_qwen_frames(
    frame_list: list[tuple[str, float]],
    *,
    max_frames: int,
    knowledge: Optional["VideoKnowledge"] = None,
    ocr_map: dict[float, str] | None = None,
) -> list[tuple[str, float]]:
    """Select a representative subset of frames for Qwen.

    Priority order:
    - first / middle / last frame
    - caption-derived scene segment boundaries
    - frames with OCR text
    - uniform temporal coverage to fill the budget
    """
    if len(frame_list) <= max_frames:
        return list(frame_list)

    must_keep: set[int] = set()
    scored: dict[int, int] = {}

    def _add(idx: int, weight: int) -> None:
        if 0 <= idx < len(frame_list):
            must_keep.add(idx)
            scored[idx] = max(weight, scored.get(idx, 0))

    _add(0, 1000)
    _add(len(frame_list) - 1, 1000)
    _add(len(frame_list) // 2, 900)

    if knowledge is not None:
        for seg in getattr(knowledge, "_segments", []):
            start_t = float(seg.get("start_t", 0.0) or 0.0)
            idx = min(range(len(frame_list)), key=lambda i: abs(frame_list[i][1] - start_t))
            _add(idx, 800)

    if ocr_map:
        for idx, (_fp, t_sec) in enumerate(frame_list):
            if ocr_map.get(t_sec):
                _add(idx, 500)

    selected = set(sorted(must_keep, key=lambda idx: (-scored.get(idx, 0), idx))[:max_frames])
    if len(selected) < max_frames:
        step = len(frame_list) / max_frames
        for n in range(max_frames):
            idx = min(len(frame_list) - 1, int(round(n * step)))
            selected.add(idx)
            if len(selected) >= max_frames:
                break
    if len(selected) < max_frames:
        for idx in range(len(frame_list)):
            selected.add(idx)
            if len(selected) >= max_frames:
                break

    ordered = [frame_list[idx] for idx in sorted(selected)]
    return ordered[:max_frames]


def _select_segment_boundary_pairs(
    enriched: list[dict[str, Any]],
    max_boundaries: int,
    min_delta: float = _SEGMENT_DIFF_MIN_JACCARD_DELTA,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    boundary_candidates: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    for i, row in enumerate(enriched):
        if i <= 0 or not row.get("is_new_segment"):
            continue
        similarity = row.get("similarity")
        sim = float(similarity) if similarity is not None else 0.0
        strength = 1.0 - sim
        if strength < min_delta:
            continue
        boundary_candidates.append((strength, i, enriched[i - 1], row))

    if not boundary_candidates:
        return []
    if max_boundaries <= 0 or len(boundary_candidates) <= max_boundaries:
        return [(prev_row, next_row) for _, _, prev_row, next_row in boundary_candidates]

    strongest = sorted(boundary_candidates, key=lambda item: (-item[0], item[1]))[:max_boundaries]
    strongest.sort(key=lambda item: item[1])
    return [(prev_row, next_row) for _, _, prev_row, next_row in strongest]

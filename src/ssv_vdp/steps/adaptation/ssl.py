"""SSL fine-tuning steps and loss analysis helpers."""

import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from selfsuvis.pipeline.core import settings

from ..common import _log, write_json_artifact

if TYPE_CHECKING:
    from selfsuvis.pipeline.training.ssl import FinetuneConfig


def _loss_sparkline(history: list[float], width: int = 40) -> str:
    """Return a fixed-width ASCII sparkline for a loss curve.

    Uses Unicode block elements ▁▂▃▄▅▆▇█ to represent relative height.
    Values are normalised to [0, 1] then mapped to 8 levels.
    """
    if not history:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(history), max(history)
    span = hi - lo if hi > lo else 1.0
    # Sample evenly if more epochs than width
    if len(history) > width:
        step = len(history) / width
        sampled = [history[int(i * step)] for i in range(width)]
    else:
        sampled = list(history)
    chars = [blocks[min(8, int(((v - lo) / span) * 8) + 1)] for v in sampled]
    return "".join(chars)


def _analyze_loss_curve(history: list[float]) -> dict[str, Any]:
    """Compute summary statistics for a training loss curve."""
    if not history:
        return {}
    n = len(history)
    first, last = history[0], history[-1]
    best = min(history)
    best_epoch = int(np.argmin(history)) + 1

    # Total relative drop
    drop_pct = (first - best) / first * 100 if first > 0 else 0.0

    # Convergence epoch: first epoch within 5 % of best loss
    threshold = best * 1.05
    convergence_epoch = next((i + 1 for i, v in enumerate(history) if v <= threshold), best_epoch)

    # Monotone check: count epochs where loss increased
    increases = sum(1 for a, b in zip(history, history[1:]) if b > a)

    # Plateau: last 20 % of epochs — std relative to mean
    tail = history[max(0, n - max(2, n // 5)) :]
    tail_mean = float(np.mean(tail)) if tail else float("nan")
    tail_std = float(np.std(tail)) if tail else float("nan")
    plateau_cv = (tail_std / tail_mean) if tail_mean > 0 else float("nan")

    # Epoch-over-epoch deltas
    deltas = [b - a for a, b in zip(history, history[1:])]
    avg_drop_per_epoch = float(np.mean(deltas)) if deltas else 0.0

    return {
        "n_epochs": n,
        "first_loss": first,
        "last_loss": last,
        "best_loss": best,
        "best_epoch": best_epoch,
        "drop_pct": drop_pct,
        "convergence_epoch": convergence_epoch,
        "n_increases": increases,
        "plateau_cv": plateau_cv,
        "avg_drop_per_epoch": avg_drop_per_epoch,
        "deltas": deltas,
    }


def _interpret_finetune_results(
    cfg: "FinetuneConfig",
    stats: dict[str, Any],
    elapsed_sec: float,
) -> list[str]:
    """Return a list of Markdown bullet-point strings interpreting the training run."""
    if not stats:
        return ["*No training data — stats unavailable.*"]

    bullets: list[str] = []
    drop = stats["drop_pct"]
    best = stats["best_loss"]
    best_e = stats["best_epoch"]
    n = stats["n_epochs"]
    cv = stats["plateau_cv"]
    incr = stats["n_increases"]
    conv_e = stats["convergence_epoch"]

    # --- Approach explanation ---
    if cfg.approach == "track_cycle":
        bullets.append(
            "**Approach (track cycle-consistency):** Positive triplets (A, B, C) are "
            "crops of the *same RF-DETR-tracked object* at times t, t+k, t+2k. "
            "The CycleConsistencyLoss enforces: embed(A)≈embed(B), embed(B)≈embed(C), "
            "and (with λ=0.3) embed(A)≈embed(C). "
            "The cycle term prevents embedding drift along long tracks and teaches "
            "the model object-identity consistency across the widest temporal gap."
        )
    elif cfg.approach == "multimodal":
        bullets.append(
            "**Approach (multimodal pairs):** Positive pairs combine track-consistent RGB "
            "pairs with auxiliary depth, motion, and geometry targets mined from the "
            "current video. The base contrastive loss remains active, while optional "
            "consistency terms encourage embedding similarity to reflect depth agreement, "
            "platform-dynamics continuity, and pose-near overlap."
        )
    elif cfg.approach == "track":
        bullets.append(
            "**Approach (track pairs):** Positive pairs are bbox-crops of the "
            "*same RF-DETR-tracked object* at two different times (gap 2–5 appearances). "
            "This is a stronger signal than full-frame temporal pairs: the model must "
            "learn appearance-invariant features for the specific object instance, not "
            "just spatial proximity between consecutive frames."
        )
    elif cfg.approach == "temporal":
        bullets.append(
            "**Approach (temporal pairs):** Consecutive frames from the mission video "
            "form *positive pairs* under the assumption that nearby frames show the same "
            "scene. The model learns to pull these embeddings together and push apart "
            "embeddings from different timesteps (negatives within the same batch). "
            "This is the preferred approach when enough frames are available (≥ 2 × batch size)."
        )
    else:
        bullets.append(
            "**Approach (augmentation pairs):** Each frame is augmented twice with random "
            "crops, flips, colour jitter, and Gaussian blur to produce a positive pair. "
            "The model learns viewpoint- and appearance-invariant representations. "
            "This approach is used when the frame count is too low for temporal pairing."
        )

    # --- Loss magnitude ---
    if best < 0.5:
        loss_comment = "excellent — the model has learned tight, well-separated embeddings"
    elif best < 1.5:
        loss_comment = "good — further epochs or a lower LR could improve it"
    elif best < 3.0:
        loss_comment = "moderate — consider more epochs, a lower temperature, or more frames"
    else:
        loss_comment = (
            "high — the model may not have converged; try more epochs or check data quality"
        )
    bullets.append(f"**Best loss ({best:.4f}):** {loss_comment}.")

    # --- Drop ---
    if drop > 40:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Large improvement over training — "
            "the backbone adapted meaningfully to this mission's visual domain."
        )
    elif drop > 15:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Moderate improvement — "
            "the model captured some domain-specific structure."
        )
    else:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Small improvement — "
            "the pre-trained weights already generalise well, or training was too short."
        )

    # --- Convergence ---
    if conv_e <= n // 3:
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss converged early. "
            "Remaining epochs did not help much — future runs can use fewer epochs."
        )
    elif conv_e >= int(n * 0.85):
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss was still improving near the end. "
            "Training more epochs would likely yield a lower loss."
        )
    else:
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss stabilised in the middle of training — "
            "epoch budget looks appropriate."
        )

    # --- Best epoch position ---
    if best_e < n:
        bullets.append(
            f"**Best checkpoint (epoch {best_e}/{n}):** Loss increased in later epochs, "
            "suggesting slight overfitting or LR too high at the end. "
            "The saved checkpoint is from epoch {best_e}."
        )
    else:
        bullets.append(
            f"**Best checkpoint (epoch {best_e}/{n}):** Best loss was at the final epoch — "
            "the run had not overfit."
        )

    # --- Plateau ---
    if not math.isnan(cv):
        if cv < 0.01:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Loss is flat in the final epochs — "
                "the model has converged and additional epochs are unlikely to help."
            )
        elif cv < 0.05:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Minor oscillation in the final epochs — "
                "training is mostly converged."
            )
        else:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Noisy loss in the final epochs — "
                "try reducing LR or increasing batch size for more stable convergence."
            )

    # --- Non-monotone ---
    if incr > n // 4:
        bullets.append(
            f"**Instability ({incr} loss increases out of {n - 1} steps):** "
            "Loss oscillated frequently — consider a lower learning rate or larger batch size."
        )

    # --- Speed ---
    secs_per_epoch = elapsed_sec / n if n else 0
    bullets.append(
        f"**Training speed:** {elapsed_sec:.1f}s total, "
        f"~{secs_per_epoch:.1f}s/epoch on `{cfg.device}`."
    )

    return bullets


def _extract_track_map(
    tracking_results: list[dict[str, Any]],
) -> dict[int, list[tuple[str, list[float], float]]]:
    """Build {track_id: [(frame_path, bbox_norm, t_sec), ...]} from tracking results.

    Skips unassigned detections (track_id ≤ 0) and degenerate bboxes.
    Returns only tracks with ≥ 2 appearances, each list sorted by t_sec.
    """
    raw: dict[int, list[tuple[str, list[float], float]]] = {}
    for frame_res in tracking_results:
        fp = frame_res.get("frame_path", "")
        t_sec = float(frame_res.get("t_sec", 0.0))
        for det in frame_res.get("detections", []):
            tid = int(det.get("track_id", 0) or 0)
            if tid <= 0:
                continue
            bbox = det.get("bbox_norm") or []
            if len(bbox) != 4:
                continue
            raw.setdefault(tid, []).append((fp, list(bbox), t_sec))
    return {
        tid: sorted(appearances, key=lambda x: x[2])
        for tid, appearances in raw.items()
        if len(appearances) >= 2
    }


def _count_potential_pairs(track_map: dict[int, list], min_gap: int = 2, max_gap: int = 5) -> int:
    """Count how many (i, j) pairs can be formed across all tracks."""
    count = 0
    for appearances in track_map.values():
        n = len(appearances)
        for i in range(n - min_gap):
            hi = min(n - 1, i + max_gap)
            lo = i + min_gap
            if lo <= hi:
                count += hi - lo + 1
    return count


def _count_potential_triplets(
    track_map: dict[int, list], min_gap: int = 2, max_gap: int = 5
) -> int:
    """Count how many (i, i+k, i+2k) triplets can be formed across all tracks."""
    count = 0
    for appearances in track_map.values():
        n = len(appearances)
        for i in range(n - 2 * min_gap):
            max_k = min(max_gap, (n - 1 - i) // 2)
            if max_k >= min_gap:
                count += max_k - min_gap + 1
    return count


def _frame_lookup(frame_list: list[tuple[str, float]]) -> tuple[dict[str, float], dict[str, int]]:
    by_path: dict[str, float] = {}
    idx_by_path: dict[str, int] = {}
    for idx, (frame_path, t_sec) in enumerate(frame_list):
        key = str(frame_path)
        by_path[key] = float(t_sec)
        idx_by_path[key] = idx
    return by_path, idx_by_path


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _vector_norm_3d(payload: dict[str, Any], key: str) -> float | None:
    vec = payload.get(key) or {}
    if not isinstance(vec, dict):
        return None
    x = _safe_float(vec.get("x"), 0.0)
    y = _safe_float(vec.get("y"), 0.0)
    z = _safe_float(vec.get("z"), 0.0)
    return math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2)


def _extract_track_pairs_as_records(
    track_map: dict[int, list[tuple[str, list[float], float]]],
    *,
    min_gap: int,
    max_gap: int,
    occupancy_summary: dict[str, Any] | None = None,
) -> list[Any]:
    from selfsuvis.pipeline.training.ssl import TemporalVisualPair

    pairs: list[Any] = []
    occ = dict(occupancy_summary or {})
    for track_id, appearances in sorted(track_map.items()):
        n = len(appearances)
        for i in range(n - min_gap):
            lo = i + min_gap
            hi = min(n - 1, i + max_gap)
            if lo > hi:
                continue
            for j in range(lo, hi + 1):
                fp_a, _bbox_a, t_a = appearances[i]
                fp_b, _bbox_b, t_b = appearances[j]
                pairs.append(
                    TemporalVisualPair(
                        anchor_frame_path=str(fp_a),
                        positive_frame_path=str(fp_b),
                        time_delta_sec=max(0.0, float(t_b) - float(t_a)),
                        track_id=int(track_id),
                        sample_weight=1.0,
                        modality_payload={
                            "occupancy_summary": occ,
                            "track_temporal_gap_sec": max(0.0, float(t_b) - float(t_a)),
                        },
                        pair_source="track_rgb",
                    )
                )
    return pairs


def _extract_depth_alignment_pairs(
    frame_list: list[tuple[str, float]],
    depth_result: dict[str, Any],
    physical_state_result: dict[str, Any] | None,
    *,
    max_gap: int = 3,
    min_target: float = 0.6,
) -> list[Any]:
    from selfsuvis.pipeline.training.ssl import CrossModalPair

    depth_rows = depth_result.get("depth_results") or []
    by_path = {
        str(r.get("frame_path", "")): r
        for r in depth_rows
        if r.get("frame_path")
        and not (r.get("depth_error") or r.get("depth_unavailable") or r.get("depth_disabled"))
    }
    occ = {
        "near_field_occupancy_density": float(
            (physical_state_result or {}).get("near_field_occupancy_density", 0.0) or 0.0
        ),
        "free_space_estimate": float(
            (physical_state_result or {}).get("free_space_estimate", 0.0) or 0.0
        ),
    }
    pairs: list[Any] = []
    ordered = [(str(fp), float(t_sec)) for fp, t_sec in frame_list if str(fp) in by_path]
    for i, (anchor_fp, anchor_t) in enumerate(ordered[:-1]):
        anchor_row = by_path[anchor_fp]
        anchor_nr = _safe_float(
            anchor_row.get("near_ratio", anchor_row.get("near_frac")),
            None,
        )
        if anchor_nr is None:
            continue
        for j in range(i + 1, min(len(ordered), i + max_gap + 1)):
            positive_fp, positive_t = ordered[j]
            positive_row = by_path[positive_fp]
            positive_nr = _safe_float(
                positive_row.get("near_ratio", positive_row.get("near_frac")),
                None,
            )
            if positive_nr is None:
                continue
            target = max(0.0, 1.0 - abs(anchor_nr - positive_nr) / 0.35)
            if target < min_target:
                continue
            pairs.append(
                CrossModalPair(
                    anchor_frame_path=anchor_fp,
                    positive_frame_path=positive_fp,
                    time_delta_sec=max(0.0, positive_t - anchor_t),
                    sample_weight=0.9 + 0.2 * target,
                    modality_payload={
                        "depth_similarity_target": round(target, 4),
                        "anchor_near_ratio": round(anchor_nr, 4),
                        "positive_near_ratio": round(positive_nr, 4),
                        "anchor_depth_confidence": _safe_float(
                            anchor_row.get("depth_confidence"), None
                        ),
                        "positive_depth_confidence": _safe_float(
                            positive_row.get("depth_confidence"), None
                        ),
                        "occupancy_summary": occ,
                    },
                    pair_source="depth_alignment",
                )
            )
    return pairs


def _extract_motion_alignment_pairs(
    frame_list: list[tuple[str, float]],
    platform_state_fusion: dict[str, Any],
    physical_state_result: dict[str, Any] | None,
    *,
    max_gap: int = 4,
    min_target: float = 0.55,
) -> list[Any]:
    from selfsuvis.pipeline.training.ssl import CrossModalPair

    samples = sorted(
        platform_state_fusion.get("posterior_samples") or [],
        key=lambda row: float(row.get("t_sec", 0.0)),
    )
    if len(samples) < 2:
        return []

    frame_paths, _idx = _frame_lookup(frame_list)
    ordered: list[tuple[str, float, dict[str, Any]]] = []
    sample_i = 0
    for frame_path, t_sec in frame_list:
        while sample_i + 1 < len(samples) and abs(
            float(samples[sample_i + 1].get("t_sec", 0.0)) - t_sec
        ) <= abs(float(samples[sample_i].get("t_sec", 0.0)) - t_sec):
            sample_i += 1
        ordered.append((str(frame_path), float(t_sec), samples[sample_i]))

    mean_occ = float((physical_state_result or {}).get("near_field_occupancy_density", 0.0) or 0.0)
    pairs: list[Any] = []
    for i in range(len(ordered) - 1):
        fp_a, t_a, sample_a = ordered[i]
        speed_a = _vector_norm_3d(sample_a, "velocity_enu_mps")
        cov_a = _safe_float(sample_a.get("covariance_trace"), 0.0) or 0.0
        if speed_a is None:
            continue
        for j in range(i + 1, min(len(ordered), i + max_gap + 1)):
            fp_b, t_b, sample_b = ordered[j]
            speed_b = _vector_norm_3d(sample_b, "velocity_enu_mps")
            cov_b = _safe_float(sample_b.get("covariance_trace"), 0.0) or 0.0
            if speed_b is None:
                continue
            dt = max(1e-6, float(t_b) - float(t_a))
            speed_delta = abs(speed_a - speed_b)
            cov_penalty = min(0.4, (cov_a + cov_b) / 200.0)
            target = max(0.0, min(1.0, 1.0 - speed_delta / 3.0 - cov_penalty))
            if target < min_target:
                continue
            pairs.append(
                CrossModalPair(
                    anchor_frame_path=fp_a,
                    positive_frame_path=fp_b,
                    time_delta_sec=dt,
                    sample_weight=0.8 + 0.3 * target,
                    modality_payload={
                        "motion_similarity_target": round(target, 4),
                        "anchor_speed_mps": round(speed_a, 4),
                        "positive_speed_mps": round(speed_b, 4),
                        "speed_delta_mps": round(speed_delta, 4),
                        "anchor_covariance_trace": round(cov_a, 4),
                        "positive_covariance_trace": round(cov_b, 4),
                        "occupancy_summary": {"near_field_occupancy_density": mean_occ},
                    },
                    pair_source="platform_motion",
                )
            )
    return pairs


def _extract_pose_overlap_pairs(
    frame_list: list[tuple[str, float]],
    full_fusion_result: dict[str, Any],
    physical_state_result: dict[str, Any] | None,
    *,
    min_gap: int = 2,
    max_gap: int = 8,
    max_distance_m: float = 3.0,
    min_overlap: float = 0.55,
) -> list[Any]:
    from selfsuvis.pipeline.training.ssl import GeometryPair

    smoothed = full_fusion_result.get("smoothed_trajectory") or []
    if len(smoothed) < len(frame_list):
        return []

    pairs: list[Any] = []
    pose_conf = float((physical_state_result or {}).get("platform_pose_confidence", 0.0) or 0.0)
    for i in range(max(0, len(frame_list) - 1)):
        fp_a, t_a = frame_list[i]
        pos_a = (smoothed[i] or {}).get("position_enu_m") or {}
        xa = _safe_float(pos_a.get("x"), None)
        ya = _safe_float(pos_a.get("y"), None)
        za = _safe_float(pos_a.get("z"), None)
        if xa is None or ya is None or za is None:
            continue
        for j in range(i + min_gap, min(len(frame_list), i + max_gap + 1)):
            fp_b, t_b = frame_list[j]
            pos_b = (smoothed[j] or {}).get("position_enu_m") or {}
            xb = _safe_float(pos_b.get("x"), None)
            yb = _safe_float(pos_b.get("y"), None)
            zb = _safe_float(pos_b.get("z"), None)
            if xb is None or yb is None or zb is None:
                continue
            dist = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
            overlap = max(0.0, 1.0 - dist / max_distance_m)
            overlap *= max(0.25, pose_conf)
            if overlap < min_overlap:
                continue
            pairs.append(
                GeometryPair(
                    anchor_frame_path=str(fp_a),
                    positive_frame_path=str(fp_b),
                    time_delta_sec=max(0.0, float(t_b) - float(t_a)),
                    pose_overlap_score=round(overlap, 4),
                    sample_weight=0.9 + 0.2 * overlap,
                    modality_payload={
                        "geometry_similarity_target": round(overlap, 4),
                        "pose_distance_m": round(dist, 4),
                        "platform_pose_confidence": round(pose_conf, 4),
                        "anchor_position_enu_m": {
                            "x": round(xa, 4),
                            "y": round(ya, 4),
                            "z": round(za, 4),
                        },
                        "positive_position_enu_m": {
                            "x": round(xb, 4),
                            "y": round(yb, 4),
                            "z": round(zb, 4),
                        },
                    },
                    pair_source="sfm_pose_overlap",
                )
            )
    return pairs


def _build_multimodal_pair_mining(
    frame_list: list[tuple[str, float]],
    track_map: dict[int, list[tuple[str, list[float], float]]],
    depth_result: dict[str, Any] | None,
    platform_state_fusion: dict[str, Any] | None,
    full_fusion_result: dict[str, Any] | None,
    physical_state_result: dict[str, Any] | None,
    *,
    min_gap: int,
    max_gap: int,
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    track_pairs = _extract_track_pairs_as_records(
        track_map,
        min_gap=min_gap,
        max_gap=max_gap,
        occupancy_summary={
            "near_field_occupancy_density": float(
                (physical_state_result or {}).get("near_field_occupancy_density", 0.0) or 0.0
            ),
            "free_space_estimate": float(
                (physical_state_result or {}).get("free_space_estimate", 0.0) or 0.0
            ),
        },
    )
    depth_pairs = _extract_depth_alignment_pairs(
        frame_list,
        depth_result or {"depth_results": []},
        physical_state_result,
        max_gap=max_gap,
    )
    motion_pairs = _extract_motion_alignment_pairs(
        frame_list,
        platform_state_fusion or {"posterior_samples": []},
        physical_state_result,
        max_gap=max_gap,
    )
    pose_pairs = _extract_pose_overlap_pairs(
        frame_list,
        full_fusion_result or {"smoothed_trajectory": []},
        physical_state_result,
        min_gap=min_gap,
        max_gap=max_gap + 3,
    )

    all_pairs = track_pairs + depth_pairs + motion_pairs + pose_pairs
    pair_stats = {
        "track_pairs": len(track_pairs),
        "depth_pairs": len(depth_pairs),
        "motion_pairs": len(motion_pairs),
        "pose_overlap_pairs": len(pose_pairs),
        "total_pairs": len(all_pairs),
        "has_auxiliary_pairs": bool(depth_pairs or motion_pairs or pose_pairs),
    }
    sfm_overlap_rows = [pair.to_dict() for pair in pose_pairs]
    return all_pairs, pair_stats, sfm_overlap_rows


def step_ssl_finetune(
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: list,
    device: str,
    epochs: int,
    batch_size: int,
    tracking_results: list[dict[str, Any]] | None = None,
    depth_result: dict[str, Any] | None = None,
    platform_state_fusion: dict[str, Any] | None = None,
    full_fusion_result: dict[str, Any] | None = None,
    physical_state_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 16: SSL DINOv3 fine-tuning, write finetune_stats.md."""
    from selfsuvis.pipeline.training.ssl import FinetuneConfig

    from ..report import write_finetune_stats_md

    out_md = video_dir / "finetune_stats.md"
    ckpt_dir = video_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(frame_list)

    # Build track map from RF-DETR results when available.
    track_map: dict[int, list[tuple[str, list[float], float]]] = {}
    if tracking_results:
        track_map = _extract_track_map(tracking_results)

    multimodal_pairs: list[Any] = []
    pair_mining_stats: dict[str, Any] = {
        "track_pairs": 0,
        "depth_pairs": 0,
        "motion_pairs": 0,
        "pose_overlap_pairs": 0,
        "total_pairs": 0,
        "has_auxiliary_pairs": False,
    }
    sfm_overlap_rows: list[dict[str, Any]] = []

    _MIN_GAP, _MAX_GAP = 2, 5
    multimodal_pairs, pair_mining_stats, sfm_overlap_rows = _build_multimodal_pair_mining(
        frame_list,
        track_map,
        depth_result,
        platform_state_fusion,
        full_fusion_result,
        physical_state_result,
        min_gap=_MIN_GAP,
        max_gap=_MAX_GAP,
    )
    pair_mining_path = video_dir / "ssl_pair_mining.json"
    write_json_artifact(pair_mining_path, pair_mining_stats)
    sfm_overlap_path = video_dir / "sfm_overlap_pairs.json"
    write_json_artifact(sfm_overlap_path, sfm_overlap_rows)

    # Select pairing approach in priority order:
    # multimodal > track_cycle > track > temporal > augment.
    n_triplets = _count_potential_triplets(track_map, _MIN_GAP, _MAX_GAP)
    n_pairs = _count_potential_pairs(track_map, _MIN_GAP, _MAX_GAP)
    frames_dir = str(Path(frame_list[0][0]).parent) if frame_list else settings.FRAMES_DIR

    if pair_mining_stats["has_auxiliary_pairs"] and pair_mining_stats["total_pairs"] >= batch_size:
        approach = "multimodal"
        _log.info(
            "  Multimodal SSL: %d total pairs (track=%d depth=%d motion=%d pose=%d)",
            pair_mining_stats["total_pairs"],
            pair_mining_stats["track_pairs"],
            pair_mining_stats["depth_pairs"],
            pair_mining_stats["motion_pairs"],
            pair_mining_stats["pose_overlap_pairs"],
        )
    elif n_triplets >= batch_size:
        approach = "track_cycle"
        _log.info(
            "  Track-cycle SSL: %d triplets from %d tracks",
            n_triplets,
            len(track_map),
        )
    elif n_pairs >= batch_size:
        approach = "track"
        _log.info(
            "  Track-pair SSL: %d pairs from %d tracks",
            n_pairs,
            len(track_map),
        )
    elif n_frames >= batch_size * 2:
        approach = "temporal"
    else:
        approach = "augment"
        _log.info("  Only %d frames — using augment approach", n_frames)
    cfg = FinetuneConfig(
        frames_dir=frames_dir,
        output_dir=str(ckpt_dir),
        model_name="dinov3_vitb14",
        approach=approach,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-5,
        weight_decay=0.04,
        temperature=0.07,
        freeze_blocks=10,
        embed_dim=768,
        proj_out_dim=128,
        num_workers=0,
        save_every=1,
        max_gap=3,
        device=device,
        seed=42,
        depth_consistency_weight=0.15,
        motion_consistency_weight=0.10,
        geometry_consistency_weight=0.15,
    )
    _log.info(
        "Starting SSL fine-tuning: %d epochs, approach=%s, device=%s", epochs, approach, device
    )
    t0 = time.time()
    loss_history: list[float] = []
    component_history: dict[str, list[float]] = {
        "contrastive_loss": [],
        "depth_consistency_loss": [],
        "motion_consistency_loss": [],
        "geometry_consistency_loss": [],
    }

    def _run_capturing(c: Any) -> str:
        import random

        import torch

        random.seed(c.seed)
        torch.manual_seed(c.seed)
        os.makedirs(c.output_dir, exist_ok=True)
        from torch.utils.data import DataLoader

        from selfsuvis.pipeline.training.ssl import (
            AugmentPairDataset,
            CycleConsistencyLoss,
            DINOFineTuner,
            MultimodalConsistencyLoss,
            MultimodalPairDataset,
            NTXentLoss,
            TemporalPairDataset,
            TrackPairDataset,
            TrackTripletDataset,
            build_augment_transform,
            multimodal_batch_collate,
        )

        transform = build_augment_transform()
        ntxent = NTXentLoss(temperature=c.temperature)
        component_keys = list(component_history.keys())
        if c.approach == "multimodal":
            dataset = MultimodalPairDataset(multimodal_pairs, transform=transform)
            loss_fn = MultimodalConsistencyLoss(
                ntxent,
                depth_weight=c.depth_consistency_weight,
                motion_weight=c.motion_consistency_weight,
                geometry_weight=c.geometry_consistency_weight,
            )
            collate_fn = multimodal_batch_collate
        elif c.approach == "track_cycle":
            dataset = TrackTripletDataset(
                track_map, transform=transform, min_gap=_MIN_GAP, max_gap=_MAX_GAP
            )
            loss_fn = CycleConsistencyLoss(ntxent)
            collate_fn = None
        elif c.approach == "track":
            dataset = TrackPairDataset(
                track_map, transform=transform, min_gap=_MIN_GAP, max_gap=_MAX_GAP
            )
            loss_fn = ntxent
            collate_fn = None
        elif c.approach == "temporal":
            dataset = TemporalPairDataset(c.frames_dir, transform=transform, max_gap=c.max_gap)
            loss_fn = ntxent
            collate_fn = None
        else:
            dataset = AugmentPairDataset(c.frames_dir, transform=transform)
            loss_fn = ntxent
            collate_fn = None
        loader = DataLoader(
            dataset,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=c.num_workers,
            pin_memory=(c.device != "cpu"),
            drop_last=True,
            collate_fn=collate_fn,
        )
        tuner = DINOFineTuner(
            model_name=c.model_name,
            freeze_blocks=c.freeze_blocks,
            device=c.device,
            embed_dim=c.embed_dim,
            proj_out_dim=c.proj_out_dim,
        )
        optimizer = torch.optim.AdamW(
            tuner.trainable_params(), lr=c.lr, weight_decay=c.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
        best_loss = float("inf")
        best_path = os.path.join(c.output_dir, "dino_ssl_best.pt")
        is_triplet = c.approach == "track_cycle"
        is_multimodal = c.approach == "multimodal"
        patience = 3
        no_improve = 0
        for epoch in range(1, c.epochs + 1):
            tuner.train()
            epoch_losses = []
            epoch_component_values: dict[str, list[float]] = {k: [] for k in component_keys}
            for batch in loader:
                if is_triplet:
                    v1 = batch[0].to(c.device)
                    v2 = batch[1].to(c.device)
                    v3 = batch[2].to(c.device)
                    loss = loss_fn(tuner.forward(v1), tuner.forward(v2), tuner.forward(v3))
                elif is_multimodal:
                    v1 = batch[0].to(c.device)
                    v2 = batch[1].to(c.device)
                    meta = batch[2]
                    z1 = tuner.forward(v1)
                    z2 = tuner.forward(v2)
                    loss, components = loss_fn(z1, z2, meta)
                    for key in component_keys:
                        epoch_component_values[key].append(float(components.get(key, 0.0)))
                else:
                    v1 = batch[0].to(c.device)
                    v2 = batch[1].to(c.device)
                    loss = loss_fn(tuner.forward(v1), tuner.forward(v2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()
            avg = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            loss_history.append(avg)
            component_msg = ""
            if is_multimodal:
                for key in component_keys:
                    mean_component = (
                        float(np.mean(epoch_component_values[key]))
                        if epoch_component_values[key]
                        else 0.0
                    )
                    component_history[key].append(mean_component)
                component_msg = (
                    " | contrast={:.4f} depth={:.4f} motion={:.4f} geometry={:.4f}".format(
                        component_history["contrastive_loss"][-1],
                        component_history["depth_consistency_loss"][-1],
                        component_history["motion_consistency_loss"][-1],
                        component_history["geometry_consistency_loss"][-1],
                    )
                )
            _log.info("    Epoch %d/%d  loss=%.4f%s", epoch, c.epochs, avg, component_msg)
            if avg < best_loss:
                best_loss = avg
                tuner.save_checkpoint(best_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _log.info(
                        "    Early stop at epoch %d/%d (no improvement for %d epochs)",
                        epoch,
                        c.epochs,
                        patience,
                    )
                    break
        return best_path

    best_path = _run_capturing(cfg)
    elapsed = time.time() - t0
    best_loss = min(loss_history) if loss_history else float("nan")
    metrics_path = video_dir / "ssl_training_metrics.json"
    write_json_artifact(
        metrics_path,
        {
            "approach": cfg.approach,
            "loss_history": loss_history,
            "component_history": component_history,
            "pair_mining": pair_mining_stats,
        },
    )
    _log.info(
        "  [ok] Fine-tuning complete in %.1fs | best loss=%.4f | checkpoint: %s",
        elapsed,
        best_loss,
        best_path,
    )
    _log.info("  To use: export DINO_CHECKPOINT=%s", best_path)
    write_finetune_stats_md(out_md, video_name, cfg, best_loss, best_path, elapsed, loss_history)
    ckpt_mb = os.path.getsize(best_path) / 1e6 if os.path.exists(best_path) else 0
    return {
        "checkpoint": best_path,
        "best_loss": best_loss,
        "elapsed_sec": elapsed,
        "ckpt_mb": ckpt_mb,
        "cfg": cfg,
        "loss_history": loss_history,
        "component_history": component_history,
        "pair_mining_path": str(pair_mining_path),
        "sfm_overlap_path": str(sfm_overlap_path),
        "metrics_path": str(metrics_path),
        "pair_mining": pair_mining_stats,
    }


def step_dae_finetune(
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: list,
    device: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Train a Denoising Autoencoder on mission frames (no labels required).

    Implements the reconstruction-based self-supervised pretext task derived
    from the denoising autoencoder approach (Vincent et al., 2008).  The trained
    model is used by DAEAnomalyScorer to assign per-frame reconstruction MSE
    scores, which are incorporated into the active-learning scoring formula.

    Skips gracefully when fewer frames are available than two batches, or when
    torch is unavailable.

    Returns a result dict with:
        checkpoint      -- path to the best DAE checkpoint (or "" if skipped)
        best_mse        -- best epoch reconstruction MSE
        elapsed_sec     -- wall-clock training time
        ckpt_mb         -- checkpoint file size in MB
        n_frames        -- number of frames used
        skipped         -- True when training was not possible
    """
    from selfsuvis.pipeline.training.dae import DAEFinetuneConfig, run_dae_finetune

    n_frames = len(frame_list)
    ckpt_dir = video_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    min_frames = batch_size * 2
    if n_frames < min_frames:
        _log.info(
            "  DAE skip: only %d frames (need >= %d for reconstruction training)",
            n_frames,
            min_frames,
        )
        return {"checkpoint": "", "best_mse": float("nan"), "elapsed_sec": 0.0,
                "ckpt_mb": 0.0, "n_frames": n_frames, "skipped": True}

    frames_dir = str(Path(frame_list[0][0]).parent) if frame_list else ""
    if not frames_dir:
        return {"checkpoint": "", "best_mse": float("nan"), "elapsed_sec": 0.0,
                "ckpt_mb": 0.0, "n_frames": n_frames, "skipped": True}

    cfg = DAEFinetuneConfig(
        frames_dir=frames_dir,
        output_dir=str(ckpt_dir),
        image_size=224,
        latent_ch=256,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        corruption_mode="both",
        noise_std=0.2,
        patch_size=16,
        mask_frac=0.15,
        num_workers=0,
        save_every=max(1, epochs // 3),
        device=device,
        seed=42,
    )
    _log.info(
        "  DAE fine-tuning: %d frames | epochs=%d | corruption=%s | device=%s",
        n_frames, epochs, cfg.corruption_mode, device,
    )
    t0 = time.time()
    try:
        best_path = run_dae_finetune(cfg)
    except Exception as exc:
        _log.warning("  DAE training failed (%s) -- skipping", exc)
        return {"checkpoint": "", "best_mse": float("nan"), "elapsed_sec": 0.0,
                "ckpt_mb": 0.0, "n_frames": n_frames, "skipped": True}

    elapsed = time.time() - t0
    ckpt_mb = os.path.getsize(best_path) / 1e6 if os.path.exists(best_path) else 0.0

    metrics_path = video_dir / "dae_training_metrics.json"
    write_json_artifact(metrics_path, {
        "video_id": video_id,
        "n_frames": n_frames,
        "epochs": epochs,
        "corruption_mode": cfg.corruption_mode,
        "device": device,
        "elapsed_sec": round(elapsed, 2),
        "ckpt_mb": round(ckpt_mb, 2),
    })

    _log.info(
        "  [ok] DAE training complete in %.1fs | checkpoint: %s (%.1f MB)",
        elapsed, best_path, ckpt_mb,
    )
    return {
        "checkpoint": best_path,
        "best_mse": float("nan"),  # populated by run_dae_finetune logs; not re-read here
        "elapsed_sec": elapsed,
        "ckpt_mb": ckpt_mb,
        "n_frames": n_frames,
        "skipped": False,
        "metrics_path": str(metrics_path),
    }

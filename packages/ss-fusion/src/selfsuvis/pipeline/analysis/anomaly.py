"""Reconstruction-error anomaly scoring for mission frame sequences.

Wraps DAEAnomalyScorer for batch analysis.  Produces per-frame anomaly scores
and tags that are compatible with the active-learning tagging system.

The anomaly threshold is computed as a percentile of reconstruction MSE scores
within the current batch rather than an absolute value, so "anomaly" means
"in the top X% for this mission."  This keeps the signal useful across missions
with differing lighting, terrain type, and camera quality.
"""

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

if TYPE_CHECKING:
    from selfsuvis.pipeline.training.dae import DAEAnomalyScorer

logger = get_logger(__name__)

_DEFAULT_ANOMALY_PERCENTILE = 90.0
_DEFAULT_HIGH_ANOMALY_PERCENTILE = 97.0


def load_dae_scorer(
    checkpoint_path: str,
    device: str = "cpu",
    image_size: int = 224,
    latent_ch: int = 256,
) -> "DAEAnomalyScorer | None":
    """Load a DAEAnomalyScorer from a checkpoint.

    Returns None if the checkpoint does not exist so callers can treat anomaly
    scoring as an optional enrichment rather than a hard dependency.
    """
    from selfsuvis.pipeline.training.dae import DAEAnomalyScorer

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.debug("DAE checkpoint not found at %r -- anomaly scoring disabled", checkpoint_path)
        return None
    return DAEAnomalyScorer(
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=image_size,
        latent_ch=latent_ch,
    )


def score_frames_anomaly(
    frame_paths: list[str],
    scorer: "DAEAnomalyScorer | None",
    batch_size: int = 32,
) -> list[float]:
    """Compute per-frame reconstruction MSE scores using a trained DAE.

    Args:
        frame_paths:  Absolute paths to frame images.
        scorer:       DAEAnomalyScorer instance, or None to disable.
        batch_size:   Forward-pass batch size.

    Returns:
        List of float MSE values, same length as frame_paths.
        Returns zeros when scorer is None or a frame cannot be read.
    """
    if scorer is None:
        return [0.0] * len(frame_paths)

    from PIL import Image

    scores: list[float] = []
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images: list[Any] = []
        valid_mask: list[bool] = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_mask.append(True)
            except Exception:
                logger.warning("anomaly: could not read frame %s", p)
                valid_mask.append(False)

        valid_images = [img for img, ok in zip(images, valid_mask) if ok]
        batch_raw = scorer.score_batch(valid_images) if valid_images else []

        it = iter(batch_raw)
        for ok in valid_mask:
            scores.append(next(it) if ok else 0.0)

    return scores


def tag_anomalous_frames(
    reconstruction_scores: list[float],
    anomaly_percentile: float = _DEFAULT_ANOMALY_PERCENTILE,
    high_anomaly_percentile: float = _DEFAULT_HIGH_ANOMALY_PERCENTILE,
) -> tuple[list[float], list[str]]:
    """Normalise reconstruction scores and assign per-frame anomaly tags.

    Scores are min-max normalised to [0, 1] within the current batch.

    Tags (mutually exclusive, highest takes precedence):
        "high_anomaly"  -- MSE above high_anomaly_percentile threshold
        "anomaly"       -- MSE above anomaly_percentile threshold
        "normal"        -- all other frames

    Args:
        reconstruction_scores:  Raw per-frame MSE values from score_frames_anomaly.
        anomaly_percentile:     Percentile threshold for "anomaly" tag (default 90).
        high_anomaly_percentile: Percentile for "high_anomaly" tag (default 97).

    Returns:
        (normalised_scores, tags) -- same length as reconstruction_scores.
    """
    if not reconstruction_scores:
        return [], []

    arr = np.array(reconstruction_scores, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 1e-9)
    normalised = ((arr - lo) / span).tolist()

    thresh_anomaly = float(np.percentile(arr, anomaly_percentile))
    thresh_high = float(np.percentile(arr, high_anomaly_percentile))

    tags: list[str] = []
    for s in reconstruction_scores:
        if s >= thresh_high:
            tags.append("high_anomaly")
        elif s >= thresh_anomaly:
            tags.append("anomaly")
        else:
            tags.append("normal")

    return normalised, tags

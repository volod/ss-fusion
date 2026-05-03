"""Grounded semantic priors derived from VLM / LLM scene analysis outputs.

Reads structured scene understanding from Gemma/Qwen analysis results and
translates domain knowledge into Kalman-filter noise parameters:

  - process_noise_scale  : Q multiplier (>1 = more uncertainty, e.g. stop-and-go)
  - gps_noise_scale      : R multiplier for GPS (>1 = urban multipath inflation)
  - object_speed_priors  : per-label maximum realistic speed (m/s in ENU frame)
  - temporal_noise_scale : Q multiplier from RSSM surprise (scene changes)

Usage::

    prior = build_semantic_prior(gemma_analysis, rssm_surprise_mean=0.8)
    # Then pass prior.process_noise_scale / prior.gps_noise_scale into the filter.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Scene-type → process noise scale ─────────────────────────────────────────
# Higher value = more dynamic scene (starts/stops) → inflate Q.
_SCENE_PROCESS_NOISE: dict[str, float] = {
    "highway":        0.60,   # mostly constant velocity, low noise
    "motorway":       0.60,
    "rural_road":     0.80,
    "rural_terrain":  1.00,
    "aerial":         0.70,   # drone: smooth flight
    "waterway":       0.70,
    "urban_street":   1.60,   # stops at lights, lane changes
    "urban_road":     1.60,
    "intersection":   2.00,   # highest dynamic uncertainty
    "parking":        1.80,
    "construction":   1.40,
    "industrial":     1.20,
    "indoor":         1.50,
    "other":          1.00,
    "unknown":        1.00,
}

# ── Object category → maximum plausible speed (m/s) ──────────────────────────
# Used as a hard-clamp on the velocity state post-update.
_OBJECT_SPEED_PRIORS: dict[str, float] = {
    "person":       3.5,
    "pedestrian":   3.5,
    "child":        3.0,
    "worker":       2.5,
    "bicycle":      10.0,
    "motorbike":    30.0,
    "motorcycle":   30.0,
    "car":          40.0,
    "van":          35.0,
    "truck":        30.0,
    "bus":          25.0,
    "train":        60.0,
    "boat":         20.0,
    "airplane":     250.0,
    "vehicle":      40.0,   # generic fallback
    "animal":       15.0,
}

# ── GPS multipath inflation factors ──────────────────────────────────────────
# When objects suggesting "urban canyon" are prominent, GPS is less reliable.
_URBAN_CANYON_OBJECTS = frozenset({"building", "skyscraper", "wall", "overpass", "tunnel"})
_GPS_MULTIPATH_SCALE = 2.5   # inflate GPS noise by this factor in urban canyons


@dataclass
class SemanticPrior:
    """Noise-adaptive parameters derived from VLM scene analysis."""
    scene_type: str = "unknown"

    # Multiplicative scales for Kalman noise matrices
    process_noise_scale: float = 1.0
    gps_noise_scale: float = 1.0
    temporal_noise_scale: float = 1.0  # from RSSM surprise

    # Per-label speed caps (m/s) for object velocity clamping
    object_speed_priors: dict[str, float] = field(default_factory=dict)

    # Diagnostics
    dominant_labels: list[str] = field(default_factory=list)
    urban_canyon_detected: bool = False
    raw_scene_type: str = "unknown"


def build_semantic_prior(
    gemma_analysis: dict[str, Any] | None = None,
    qwen_captions: Sequence[dict[str, Any]] | None = None,
    rssm_surprise_mean: float | None = None,
) -> SemanticPrior:
    """Derive SemanticPrior from structured VLM outputs.

    Args:
        gemma_analysis:   Dict from step_gemma_analysis (keys: scene_type,
                          dominant_objects, top_category, etc.)
        qwen_captions:    Per-frame Qwen structured outputs (keys: scene_summary,
                          vehicle_groups, road_surface, road_condition).
        rssm_surprise_mean: Mean RSSM temporal-surprise score [0, 1] for the video.
    """
    scene_type = "unknown"
    dominant_labels: list[str] = []
    urban_canyon_detected = False

    # ── Parse Gemma analysis ──────────────────────────────────────────────────
    if gemma_analysis:
        raw_scene = (
            gemma_analysis.get("scene_type")
            or gemma_analysis.get("top_category")
            or ""
        )
        scene_type = _normalise_scene_type(str(raw_scene))

        # Dominant objects from Gemma structured output
        dom_objects = gemma_analysis.get("dominant_objects") or []
        if isinstance(dom_objects, list):
            for obj in dom_objects:
                if isinstance(obj, dict):
                    cat = str(obj.get("category") or obj.get("label") or "")
                    if cat:
                        dominant_labels.append(cat.lower())
                elif isinstance(obj, str):
                    dominant_labels.append(obj.lower())

        # Detection of urban canyon objects
        all_labels_lower = {lbl.lower() for lbl in dominant_labels}
        if all_labels_lower & _URBAN_CANYON_OBJECTS:
            urban_canyon_detected = True

    # ── Parse Qwen structured captions for scene type refinement ─────────────
    if qwen_captions and scene_type == "unknown":
        road_surfaces = []
        for cap in qwen_captions:
            if not isinstance(cap, dict):
                continue
            rs = cap.get("road_surface") or cap.get("scene_type") or ""
            if rs:
                road_surfaces.append(str(rs).lower())
        if road_surfaces:
            # Most frequent surface type
            from collections import Counter
            most_common = Counter(road_surfaces).most_common(1)[0][0]
            scene_type = _normalise_scene_type(most_common)

    # ── Process noise scale ───────────────────────────────────────────────────
    proc_scale = _SCENE_PROCESS_NOISE.get(scene_type, 1.0)

    # ── GPS noise scale ───────────────────────────────────────────────────────
    gps_scale = _GPS_MULTIPATH_SCALE if urban_canyon_detected else 1.0

    # ── Temporal noise from RSSM surprise ────────────────────────────────────
    # High temporal surprise → scene is changing → inflate process noise.
    # surprise ∈ [0, 1], linear scale: 0 → 1.0×, 0.5 → 1.5×, 1.0 → 3.0×
    temporal_scale = 1.0
    if rssm_surprise_mean is not None:
        s = float(max(0.0, min(1.0, rssm_surprise_mean)))
        temporal_scale = 1.0 + 2.0 * s  # [1.0, 3.0]

    # Combined process noise scale
    combined_proc = proc_scale * temporal_scale

    # ── Object speed priors ───────────────────────────────────────────────────
    obj_priors: dict[str, float] = dict(_OBJECT_SPEED_PRIORS)
    # Label-specific priors from Gemma dominant objects
    for lbl in dominant_labels:
        if lbl not in obj_priors:
            obj_priors[lbl] = _OBJECT_SPEED_PRIORS.get("vehicle", 40.0)

    logger.info(
        "Semantic prior: scene=%s proc_scale=%.2f gps_scale=%.2f "
        "temporal_scale=%.2f urban_canyon=%s",
        scene_type, combined_proc, gps_scale, temporal_scale, urban_canyon_detected,
    )

    return SemanticPrior(
        scene_type=scene_type,
        process_noise_scale=combined_proc,
        gps_noise_scale=gps_scale,
        temporal_noise_scale=temporal_scale,
        object_speed_priors=obj_priors,
        dominant_labels=dominant_labels,
        urban_canyon_detected=urban_canyon_detected,
        raw_scene_type=str(
            (gemma_analysis or {}).get("scene_type")
            or (gemma_analysis or {}).get("top_category")
            or "unknown"
        ),
    )


def _normalise_scene_type(raw: str) -> str:
    """Map free-form scene descriptions to canonical keys."""
    r = raw.lower().replace("-", "_").replace(" ", "_")
    for key in _SCENE_PROCESS_NOISE:
        if key in r:
            return key
    # Keyword matching
    if any(k in r for k in ("highway", "motorway", "freeway")):
        return "highway"
    if any(k in r for k in ("intersection", "crossroad", "junction")):
        return "intersection"
    if any(k in r for k in ("urban", "city", "downtown", "street", "road")):
        return "urban_street"
    if any(k in r for k in ("rural", "country", "farm")):
        return "rural_terrain"
    if any(k in r for k in ("aerial", "drone", "overhead", "bird")):
        return "aerial"
    if any(k in r for k in ("parking", "carpark", "lot")):
        return "parking"
    return "unknown"

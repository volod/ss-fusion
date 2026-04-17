"""Change detection across GPS-overlapping missions.

For each frame in a new mission, query Qdrant for frames from earlier missions
within a GPS bounding box. If the closest matching frame's cosine distance
exceeds the threshold, a change event is recorded.

Phase 4 extension: when both frames have `frame_facts_json` populated (by the
Gemma/Qwen Phase 2 pass), a structured `semantic_diff_json` is computed and an
optional natural-language `change_explanation` is generated via the Gemma API.

The query_fn abstraction lets unit tests inject a mock without a live Qdrant.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# Approximate metres per degree of latitude at the equator
_M_PER_DEG_LAT = 111_320.0

QueryFn = Callable[[np.ndarray, Tuple[float, float, float, float]], List[Dict[str, Any]]]


def latlon_bbox(
    lat: float, lon: float, radius_m: float
) -> Tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) bounding box for a GPS circle.

    Uses a flat-earth approximation valid for radius_m << Earth radius.
    """
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * (abs(np.cos(np.radians(lat))) + 1e-9))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 − cosine_similarity) between two vectors.

    Returns 1.0 if either vector is a zero vector.
    """
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def threshold_for_model() -> float:
    """Return the change-detection threshold appropriate for the configured model."""
    if settings.MODEL_NAME in {"dinov2", "dinov3"}:
        return settings.CHANGE_DETECTION_THRESHOLD_DINO
    return settings.CHANGE_DETECTION_THRESHOLD_CLIP


def detect_changes(
    new_frames: List[Dict[str, Any]],
    query_fn: QueryFn,
    threshold: Optional[float] = None,
    radius_m: float = 50.0,
) -> List[Dict[str, Any]]:
    """Detect visual changes between new_frames and historically indexed frames.

    Args:
        new_frames: Frames from the current mission. Each dict must have:
            frame_id (str), mission_id (str), embedding (list[float]),
            gps (dict with lat/lon keys, or None).
        query_fn: Callable(embedding, bbox) → list of candidate dicts.
            Each candidate must have: frame_id, mission_id, embedding.
        threshold: Cosine-distance threshold; defaults to model-appropriate value.
        radius_m: GPS search radius in metres (default 50 m).

    Returns:
        List of change-event dicts:
            frame_id, mission_id, ref_frame_id, ref_mission_id,
            change_score (cosine dist), threshold.
    """
    if threshold is None:
        threshold = threshold_for_model()

    changes: List[Dict[str, Any]] = []
    for frame in new_frames:
        gps = frame.get("gps")
        if not gps or gps.get("lat") is None or gps.get("lon") is None:
            continue

        bbox = latlon_bbox(gps["lat"], gps["lon"], radius_m)
        embedding = np.asarray(frame["embedding"], dtype=np.float32)
        candidates = query_fn(embedding, bbox)

        best_dist: Optional[float] = None
        best_ref: Optional[Dict[str, Any]] = None
        for cand in candidates:
            if cand.get("mission_id") == frame["mission_id"]:
                continue  # skip same-mission frames
            ref_emb = np.asarray(cand["embedding"], dtype=np.float32)
            dist = cosine_distance(embedding, ref_emb)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ref = cand

        if best_ref is not None and best_dist is not None and best_dist >= threshold:
            changes.append(
                {
                    "frame_id": frame["frame_id"],
                    "mission_id": frame["mission_id"],
                    "ref_frame_id": best_ref["frame_id"],
                    "ref_mission_id": best_ref["mission_id"],
                    "change_score": best_dist,
                    "threshold": threshold,
                }
            )
            logger.debug(
                "Change detected: frame=%s ref=%s dist=%.3f",
                frame["frame_id"],
                best_ref["frame_id"],
                best_dist,
            )

            # Phase 4: compute semantic diff if both frames have facts data
            event = changes[-1]
            new_facts = frame.get("frame_facts_json")
            ref_facts = best_ref.get("frame_facts_json")
            if new_facts and ref_facts:
                event["semantic_diff_json"] = compute_semantic_diff(ref_facts, new_facts)
            else:
                event["semantic_diff_json"] = None

    return changes


# ── Phase 4: Semantic diff ────────────────────────────────────────────────────

def compute_semantic_diff(
    ref_facts: Dict[str, Any],
    new_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute a structured diff between two frame_facts_json dicts.

    Compares vehicle count, road_condition, and road_surface between the
    reference (earlier mission) and new (current mission) frame.

    Args:
        ref_facts: frame_facts_json from the earlier (reference) frame.
        new_facts: frame_facts_json from the current mission frame.

    Returns:
        Dict with keys for each changed field; e.g.::

            {
                "vehicle_count": {"before": 2, "after": 5, "delta": 3},
                "road_condition": {"before": "clear", "after": "wet"},
            }

        Empty dict if no structured difference is detected.
    """

    def _vehicle_count(facts: Dict[str, Any]) -> Optional[int]:
        groups = facts.get("vehicle_groups")
        if not isinstance(groups, list):
            return None
        total = 0
        for g in groups:
            if isinstance(g, dict):
                total += int(g.get("count", 0))
        return total

    diff: Dict[str, Any] = {}

    ref_count = _vehicle_count(ref_facts)
    new_count = _vehicle_count(new_facts)
    if ref_count is not None and new_count is not None and ref_count != new_count:
        diff["vehicle_count"] = {
            "before": ref_count,
            "after": new_count,
            "delta": new_count - ref_count,
        }

    for key in ("road_condition", "road_surface"):
        ref_val = ref_facts.get(key)
        new_val = new_facts.get(key)
        if ref_val and new_val and ref_val != new_val:
            diff[key] = {"before": ref_val, "after": new_val}

    return diff


def generate_change_explanation(
    semantic_diff: Dict[str, Any],
    change_score: float,
) -> Optional[str]:
    """Generate a natural language change explanation via the Gemma API.

    Only called when ``GEMMA_API_URL`` is configured.  Returns ``None`` if
    the API is unavailable or the diff is empty.

    Args:
        semantic_diff: Output of :func:`compute_semantic_diff`.
        change_score:  Cosine distance (0–1) between the two frames.

    Returns:
        One-sentence explanation string, or ``None`` on failure.
    """
    if not settings.GEMMA_API_URL or not semantic_diff:
        return None

    # Build a concise prompt from the diff
    parts: list[str] = [f"Embedding distance: {change_score:.2f}."]
    if "vehicle_count" in semantic_diff:
        vc = semantic_diff["vehicle_count"]
        delta = vc.get("delta", 0)
        direction = "increased" if delta > 0 else "decreased"
        parts.append(
            f"Vehicle count {direction} from {vc['before']} to {vc['after']}."
        )
    if "road_condition" in semantic_diff:
        rc = semantic_diff["road_condition"]
        parts.append(f"Road condition changed from {rc['before']} to {rc['after']}.")
    if "road_surface" in semantic_diff:
        rs = semantic_diff["road_surface"]
        parts.append(f"Road surface changed from {rs['before']} to {rs['after']}.")

    observation = " ".join(parts)
    prompt = (
        f"Observations: {observation}\n"
        "Write a single concise sentence explaining what changed at this location "
        "between the two mission passes. Be specific and factual."
    )

    try:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(
            api_key="EMPTY",
            base_url=settings.GEMMA_API_URL,
            timeout=30,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=settings.GEMMA_API_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        # Enforce single sentence: truncate at first period if multiple.
        if ". " in text:
            text = text[: text.index(". ") + 1]
        return text or None
    except Exception as exc:
        logger.debug("Gemma change explanation failed: %s", exc)
        return None

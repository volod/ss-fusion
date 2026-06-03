"""Step 10: Gemma 4 directed tracking — SAM segmentation guided by Gemma scene
understanding, followed by RF-DETR multi-frame tracking.

Flow:
  1. Gemma 4 analyses sampled frames → structured JSON with scene type, dominant
     objects (category + rough_bbox), and a priority-ordered tracking list.
  2. SAM segments Gemma-identified objects using:
       Path A — Gemma rough_bbox fed directly as box prompts to SAMPredictor.
       Path B — SAM auto-mask generation + CLIP zero-shot filtering (fallback when
                Gemma could not localise objects precisely).
  3. RF-DETR tracks Gemma-priority classes across the full frame sequence with
     persistent track IDs (IoU-based greedy matching, threshold 0.45).

Artifacts produced under ``<video_dir>/``:
  gemma_tracking/
    frame_{t:.3f}_tracked.jpg      annotated frame (tracking boxes + SAM masks)
  gemma_tracking_results.json      per-frame detections with track IDs + SAM metadata
  gemma_tracking_summary.md        Gemma scene intel, tracking & segmentation summary
"""

import base64
import hashlib
import io
import json
import logging as _logging_mod
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.vision.rfdetr import RFDETRTracker

from .. import common as _local_common

_log = _logging_mod.getLogger("pipeline.local.tracking")
_open_frame_image = _local_common._open_frame_image
gemma_cache_file = getattr(
    _local_common,
    "gemma_cache_file",
    lambda video_dir: video_dir / "runtime_cache" / "gemma_responses.json",
)
gemma_frame_cache_key = getattr(
    _local_common,
    "gemma_frame_cache_key",
    lambda frame_path, *, model, prompt_tag: (
        f"{prompt_tag}:{model}:{hashlib.sha256(Path(frame_path).read_bytes()).hexdigest()}"
    ),
)
load_gemma_cache = getattr(
    _local_common,
    "load_gemma_cache",
    lambda video_dir, *, enabled: (
        {}
        if (not enabled or not gemma_cache_file(video_dir).exists())
        else json.loads(gemma_cache_file(video_dir).read_text(encoding="utf-8"))
    ),
)
save_gemma_cache = getattr(
    _local_common,
    "save_gemma_cache",
    lambda video_dir, cache, *, enabled: (
        None
        if not enabled
        else (
            gemma_cache_file(video_dir).parent.mkdir(parents=True, exist_ok=True),
            gemma_cache_file(video_dir).write_text(
                json.dumps(cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            ),
        )
    ),
)
write_json_artifact = getattr(
    _local_common,
    "write_json_artifact",
    lambda path, payload, ensure_ascii=True: path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    ),
)
write_markdown_artifact = getattr(
    _local_common,
    "write_markdown_artifact",
    lambda path, lines: path.write_text("\n".join(lines) + "\n", encoding="utf-8"),
)

# -- Constants ------------------------------------------------------------------

# Frames sent to Gemma for structured scene analysis (sparse sample)
_GEMMA_STRUCTURED_SAMPLE_N = 12
# Max frames processed by RF-DETR tracking
_MAX_TRACKING_FRAMES = 90
# CLIP similarity threshold for mask filtering (Path B)
_CLIP_MASK_THRESHOLD = 0.18
# Fallback bbox Gemma uses when it cannot localise an object
_FALLBACK_BBOX = [0.1, 0.1, 0.9, 0.9]
# Gemma produces bboxes; consider "non-fallback" when coverage < this fraction
_FALLBACK_AREA_THRESHOLD = 0.72  # (0.9-0.1)^2 ≈ 0.64; use 0.72 to catch near-fallbacks
# Render line width relative to image width
_BOX_WIDTH_RATIO = 0.003
# Maximum frames on which Path B (SAM AMG) is allowed to run.
# Path B is expensive (~5-60s per frame depending on hardware); cap it so a
# scene with all-fallback Gemma bboxes doesn't freeze the pipeline for hours.
_MAX_PATH_B_FRAMES = 3
# Priority colours (RGB) — reuse the same scheme as YOLO+SAM step
_PRIORITY_COLORS = {
    1: (229, 57, 53),  # human → red
    2: (30, 136, 229),  # vehicle → blue
    3: (67, 160, 71),  # artificial → green
    4: (158, 158, 158),  # other → grey
}

_TRACKING_TARGET_CANONICAL = {
    "person": {"person", "pedestrian", "people", "human", "worker", "rider", "child"},
    "vehicle": {
        "vehicle",
        "car",
        "truck",
        "bus",
        "van",
        "pickup",
        "motorcycle",
        "motorbike",
        "bike",
        "bicycle",
        "train",
        "boat",
        "airplane",
    },
    "sign": {"sign", "stop sign", "traffic light", "traffic sign"},
}

_VALID_SCENE_TYPES = frozenset(
    {
        "urban_street",
        "rural_terrain",
        "indoor",
        "aerial",
        "waterway",
        "construction",
        "industrial",
        "other",
    }
)


# -- Gemma structured scene analysis -------------------------------------------


def _gemma_structured_scene_analysis(
    frame_list: list[tuple[str, float]],
    api_url: str,
    model: str,
    timeout: float,
    clip_model: Any,
    *,
    video_dir: Path | None = None,
) -> dict[str, Any]:
    """Send sampled frames to Gemma asking for structured JSON scene understanding.

    The JSON schema asks for:
      - scene_type: coarse environment category
      - dominant_objects: list of {category, count_estimate, spatial_hint, rough_bbox}
      - areas_of_interest: notable regions
      - tracking_priority: ordered list of labels to focus RF-DETR on

    Aggregates per-frame responses into a single scene summary by:
    - Taking the most frequent scene_type across frames
    - Merging dominant_objects, deduplicating by category
    - Ranking tracking_priority by cross-frame frequency

    Returns the aggregated dict, or a minimal empty fallback on total failure.
    """
    try:
        import httpx
    except ImportError:
        return _empty_scene()

    n_avail = len(frame_list)
    n_sample = min(
        int(settings.GEMMA_TRACKING_MAX_SAMPLE_FRAMES), _GEMMA_STRUCTURED_SAMPLE_N, n_avail
    )
    step = max(1, n_avail // max(1, n_sample))
    sampled = frame_list[::step][:n_sample]
    try:
        from ssv_vdp.steps_caption import (
            _reduce_llm_sample_frames,  # noqa: PLC0415
        )

        sampled = _reduce_llm_sample_frames(sampled, max_frames=n_sample)
    except Exception:
        pass

    structured_prompt = (
        "Analyse this frame from a robotics/drone mission video.\n"
        "Return ONLY a valid JSON object with this exact schema — no other text:\n"
        "{\n"
        '  "scene_type": "<one of: urban_street|rural_terrain|indoor|aerial|'
        'waterway|construction|industrial|other>",\n'
        '  "dominant_objects": [\n'
        "    {\n"
        '      "category": "<short label, e.g. person, vehicle, building, drone>",\n'
        '      "count_estimate": 1,\n'
        '      "spatial_hint": "<rough location: center|left|right|background|foreground>",\n'
        '      "rough_bbox": [x1_frac, y1_frac, x2_frac, y2_frac]\n'
        "    }\n"
        "  ],\n"
        '  "areas_of_interest": ["<brief description of up to 3 notable regions>"],\n'
        '  "motion_present": true,\n'
        '  "tracking_priority": ["<up to 5 category labels to prioritise for tracking>"]\n'
        "}\n\n"
        "Fractional coordinates are 0.0-1.0 (x: left=0 right=1; y: top=0 bottom=1).\n"
        "If you cannot estimate rough_bbox precisely, use [0.1, 0.1, 0.9, 0.9].\n"
        "Only list objects you can clearly identify. Be concise and factual."
    )

    # Two endpoint strategies:
    #   1. OpenAI-compatible /v1/chat/completions with image_url content
    #   2. Ollama native /api/chat with images[] array
    # Thinking models (e.g. gemma4:e4b) return empty content via the OpenAI path
    # because they emit all output as reasoning tokens.  The native Ollama path
    # bypasses the thinking token mechanism and returns proper content.
    base = api_url.rstrip("/")
    openai_endpoint = f"{base}/chat/completions"
    # Derive native Ollama base: strip trailing /v1 if present
    ollama_base = base[:-3] if base.endswith("/v1") else base
    ollama_endpoint = f"{ollama_base}/api/chat"

    per_frame_responses: list[dict[str, Any]] = []
    n_failed = 0
    cache: dict[str, Any] = {}
    cache_enabled = video_dir is not None and settings.GEMMA_CACHE_RESPONSES
    if video_dir is not None:
        try:
            cache = load_gemma_cache(video_dir, enabled=cache_enabled)
        except Exception:
            cache = {}

    for fp, _t_sec in sampled:
        try:
            cache_key = ""
            if cache_enabled:
                cache_key = gemma_frame_cache_key(
                    fp,
                    model=model,
                    prompt_tag="gemma_tracking_structured_v1",
                )
                cached = cache.get(cache_key)
                if isinstance(cached, dict) and cached.get("parsed_json"):
                    per_frame_responses.append(cached["parsed_json"])
                    continue
            img = Image.open(fp).convert("RGB")
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            b64 = base64.b64encode(buf.getvalue()).decode()

            t_req = time.time()
            content = _gemma_vision_request(
                httpx,
                model,
                b64,
                structured_prompt,
                openai_endpoint,
                ollama_endpoint,
                timeout,
            )

            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )

            # Strip markdown fences before JSON parse
            raw = content.strip()
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            per_frame_responses.append(parsed)
            elapsed = time.time() - t_req
            if elapsed >= float(settings.GEMMA_SLOW_CALL_SEC):
                _log.info(
                    "  [Step10/Gemma] slow structured frame call: %.1fs for %s",
                    elapsed,
                    Path(fp).name,
                )
            if cache_key:
                cache[cache_key] = {"parsed_json": parsed, "elapsed_sec": round(elapsed, 3)}
        except Exception as exc:
            n_failed += 1
            _log.debug("  [Step10/Gemma] frame analysis failed: %s", exc)
    if video_dir is not None:
        try:
            save_gemma_cache(video_dir, cache, enabled=cache_enabled)
        except Exception:
            pass

    if not per_frame_responses:
        _log.warning(
            "  [Step10/Gemma] No structured responses received (%d/%d frames failed) — "
            "using empty scene.  Check that the model supports vision inputs.",
            n_failed,
            len(sampled),
        )
        return _empty_scene()

    return _aggregate_scene_responses(per_frame_responses)


def _gemma_vision_request(
    httpx: Any,
    model: str,
    b64: str,
    prompt: str,
    openai_endpoint: str,
    ollama_endpoint: str,
    timeout: float,
) -> str:
    """Send one image + prompt to the model; return the text content string.

    Strategy:
      1. Try the OpenAI-compatible endpoint (works for most providers).
      2. If content is empty (thinking model — all output in *reasoning* tokens),
         fall back to the Ollama native /api/chat endpoint which bypasses thinking
         tokens and writes the answer directly to content.
    """
    import re as _re

    # -- Strategy 1: OpenAI-compatible ------------------------------------------
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 800,
        "temperature": 0.1,
    }
    resp = httpx.post(openai_endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""

    # Flatten list-style content (some providers return [{type,text}, ...])
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

    if content.strip():
        return content

    # -- Strategy 2: Ollama native /api/chat ------------------------------------
    # gemma4 and other thinking models route all output to reasoning tokens,
    # leaving content empty on the OpenAI path.  The native Ollama endpoint
    # uses the images[] array format which bypasses reasoning-only mode.
    native_payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
    }
    try:
        native_resp = httpx.post(ollama_endpoint, json=native_payload, timeout=timeout)
        native_resp.raise_for_status()
        native_content = native_resp.json().get("message", {}).get("content", "")
        if native_content.strip():
            return native_content
    except Exception:
        pass  # Native endpoint not available (non-Ollama provider) — fall through

    # -- Last resort: extract JSON from reasoning tokens -------------------------
    reasoning = msg.get("reasoning") or msg.get("thinking") or ""
    if reasoning:
        # Find the outermost {...} block in the reasoning text
        match = _re.search(r"\{[\s\S]*\}", reasoning)
        if match:
            return match.group()

    return ""


def _empty_scene() -> dict[str, Any]:
    return {
        "scene_type": "other",
        "dominant_objects": [],
        "areas_of_interest": [],
        "motion_present": False,
        "tracking_priority": [],
    }


def _valid_scene_type(scene_type: Any) -> bool:
    value = str(scene_type or "").strip().lower()
    if not value or value not in _VALID_SCENE_TYPES:
        return False
    # Reject schema placeholders copied verbatim from prompts.
    return "|" not in value and "<" not in value and "one of" not in value


def _valid_gemma_object(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    category = str(obj.get("category") or "").strip().lower()
    if not category or any(token in category for token in ("<", ">", "|", "e.g.")):
        return False
    bbox = obj.get("rough_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def _scene_is_actionable(scene: dict[str, Any] | None) -> bool:
    """Return True when a precomputed scene contains usable tracking targets.

    Text-only step-03 summaries often know *what* to track before they can infer
    precise boxes. In that case the precomputed scene is still valuable for
    RF-DETR label filtering even if SAM-directed masks will be sparse.
    """
    if not scene:
        return False
    if not _valid_scene_type(scene.get("scene_type")):
        return False
    valid_objects = [obj for obj in scene.get("dominant_objects", []) if _valid_gemma_object(obj)]
    tracking_targets = _normalise_tracking_targets(
        scene.get("tracking_priority", []),
        valid_objects,
    )
    has_non_fallback_geometry = any(
        "fallback" not in str(obj.get("spatial_hint") or "").lower() for obj in valid_objects
    )
    if valid_objects and tracking_targets and has_non_fallback_geometry:
        return True
    return bool(
        tracking_targets
        and (scene.get("areas_of_interest") or scene.get("motion_present"))
        and not valid_objects
    )


def _aggregate_scene_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-frame structured Gemma responses into one scene summary."""
    from collections import Counter

    scene_types: list[str] = []
    objects_by_cat: dict[str, dict[str, Any]] = {}
    priorities: list[str] = []
    areas: list[str] = []

    for r in responses:
        st = str(r.get("scene_type", "other") or "").strip().lower()
        if _valid_scene_type(st):
            scene_types.append(st)

        for obj in r.get("dominant_objects", []):
            if not _valid_gemma_object(obj):
                continue
            cat = (obj.get("category") or "").strip().lower()
            if not cat:
                continue
            if cat not in objects_by_cat:
                objects_by_cat[cat] = {
                    "category": cat,
                    "count_estimate": obj.get("count_estimate", 1),
                    "spatial_hint": obj.get("spatial_hint", ""),
                    "rough_bbox": obj.get("rough_bbox", _FALLBACK_BBOX),
                    "_seen": 1,
                }
            else:
                objects_by_cat[cat]["_seen"] += 1
                # Use the most recently seen bbox (likely the most specific)
                bbox = obj.get("rough_bbox", _FALLBACK_BBOX)
                if _bbox_area(bbox) < _FALLBACK_AREA_THRESHOLD:
                    objects_by_cat[cat]["rough_bbox"] = bbox

        for p in r.get("tracking_priority", []):
            lbl = (p or "").strip().lower()
            if lbl:
                priorities.append(lbl)

        for a in r.get("areas_of_interest", []):
            if a and a not in areas:
                areas.append(a)

    dominant_scene = Counter(scene_types).most_common(1)[0][0] if scene_types else "other"
    # If any frame classified the scene as aerial, that overrides urban/rural
    # misclassifications that occur when Gemma sees road structure from above.
    aerial_votes = scene_types.count("aerial")
    if aerial_votes > 0 and dominant_scene in ("urban_street", "rural_terrain", "other"):
        dominant_scene = "aerial"
    # Sort objects by observation frequency (most-seen first)
    sorted_objects = sorted(objects_by_cat.values(), key=lambda o: -o.get("_seen", 0))
    for o in sorted_objects:
        o.pop("_seen", None)

    # Rank tracking_priority labels by cross-frame frequency
    priority_counts = Counter(priorities)
    ranked_priority = [lbl for lbl, _ in priority_counts.most_common(5)]

    return {
        "scene_type": dominant_scene,
        "dominant_objects": sorted_objects,
        "areas_of_interest": areas[:3],
        "motion_present": any(r.get("motion_present", False) for r in responses),
        "tracking_priority": ranked_priority,
    }


def _bbox_area(bbox: list[float]) -> float:
    """Return fractional area of a normalised [x1, y1, x2, y2] bbox."""
    try:
        w = max(0.0, bbox[2] - bbox[0])
        h = max(0.0, bbox[3] - bbox[1])
        return w * h
    except Exception:
        return 1.0


def _normalise_tracking_targets(
    tracking_priority: list[str],
    gemma_objects: list[dict[str, Any]],
) -> list[str]:
    """Reduce Gemma scene nouns to detector-aligned target classes."""
    candidates = [*(tracking_priority or [])]
    candidates.extend(
        (obj.get("category") or "").strip().lower()
        for obj in gemma_objects
        if (obj.get("category") or "").strip()
    )
    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        norm = " ".join(raw.lower().replace("-", " ").replace("_", " ").split())
        if not norm:
            continue
        canonical = None
        for family, aliases in _TRACKING_TARGET_CANONICAL.items():
            if norm == family or norm in aliases or any(token in aliases for token in norm.split()):
                canonical = family
                break
        if canonical is None:
            continue
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def _track_length_stats(tracking_results: list[dict[str, Any]]) -> tuple[float, float]:
    lengths: dict[int, int] = {}
    for frame_res in tracking_results:
        seen_ids: set[int] = set()
        for det in frame_res.get("detections", []):
            tid = int(det.get("track_id", 0) or 0)
            if tid <= 0 or tid in seen_ids:
                continue
            seen_ids.add(tid)
            lengths[tid] = lengths.get(tid, 0) + 1
    if not lengths:
        return 0.0, 0.0
    values = sorted(lengths.values())
    mean_len = float(sum(values) / len(values))
    mid = len(values) // 2
    if len(values) % 2 == 1:
        median_len = float(values[mid])
    else:
        median_len = float(values[mid - 1] + values[mid]) / 2.0
    return mean_len, median_len


# -- SAM directed by Gemma ------------------------------------------------------


def _get_sam_auto_masks(
    image: Image.Image,
    sam_predictor: Any,
) -> list[dict[str, Any]]:
    """Generate candidate masks from SAM in automatic mode.

    Uses the cached ``SAMPredictor.get_auto_mask_generator()`` (points_per_side=8,
    64 prompts) so the generator is not re-instantiated on every frame call.
    Falls back to a 4×4 grid of box prompts when the AMG is unavailable.

    Returns list of dicts with keys: ``mask`` (bool ndarray), ``bbox``
    ([x, y, w, h] pixel coords), ``area`` (int), ``score`` (float).
    """
    img_np = np.array(image.convert("RGB"))

    # Use the SAMPredictor's cached AMG (avoids recreating it every frame).
    try:
        import torch

        amg = sam_predictor.get_auto_mask_generator(points_per_side=8)
        if amg is not None:
            with torch.inference_mode():
                masks = amg.generate(img_np)
            return [
                {
                    "mask": m["segmentation"].astype(bool),
                    "bbox": m["bbox"],  # [x, y, w, h] pixels
                    "area": int(m["area"]),
                    "score": float(m.get("predicted_iou", 0.0)),
                }
                for m in masks
            ]
    except Exception as exc:
        _log.debug("  [Step10/SAM] Auto-mask generator unavailable (%s); using grid fallback", exc)

    # Grid fallback: 4×4 evenly-spaced boxes (5% of image per side)
    grid_bboxes: list[tuple[float, float, float, float]] = []
    pad = 0.05
    for row in range(4):
        for col in range(4):
            cx = (col + 0.5) / 4
            cy = (row + 0.5) / 4
            grid_bboxes.append(
                (
                    max(0.0, cx - pad),
                    max(0.0, cy - pad),
                    min(1.0, cx + pad),
                    min(1.0, cy + pad),
                )
            )
    raw_masks = sam_predictor.predict_boxes(image, grid_bboxes)
    results = []
    for mask_info in raw_masks:
        if mask_info is None or mask_info.get("mask") is None:
            continue
        mask = mask_info["mask"].astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        results.append(
            {
                "mask": mask,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int(mask.sum()),
                "score": float(mask_info.get("score", 0.0)),
            }
        )
    return results


def _clip_filter_sam_masks(
    auto_masks: list[dict[str, Any]],
    target_categories: list[str],
    clip_model: Any,
    image: Image.Image,
    threshold: float = _CLIP_MASK_THRESHOLD,
) -> list[dict[str, Any]]:
    """Filter SAM auto-masks to those semantically matching *target_categories*.

    Encodes all valid mask crops in a **single batched** ``encode_images`` call
    instead of one call per mask — this was the root cause of ~25 min/frame
    when SAM AMG produced ~300 masks and CLIP was on CPU.

    Attaches ``matched_category`` and ``clip_score`` to passing masks.
    Returns empty list when clip_model is None or no categories given.
    """
    if clip_model is None or not target_categories or not auto_masks:
        return []

    try:
        text_embeds = clip_model.encode_texts(target_categories)  # (C, dim)
    except Exception as exc:
        _log.debug("  [Step10/CLIP] text encoding failed: %s", exc)
        return []

    w_img, h_img = image.size

    # -- Pass 1: extract all valid crops and remember their mask indices --------
    valid_indices: list[int] = []
    crops: list[Any] = []  # PIL images

    for i, mask_info in enumerate(auto_masks):
        bbox = mask_info.get("bbox")
        if bbox is None:
            continue
        try:
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + bw)
            y2 = min(h_img, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            if crop.width < 8 or crop.height < 8:
                continue
            crops.append(crop)
            valid_indices.append(i)
        except Exception as exc:
            _log.debug("  [Step10/CLIP] mask crop extraction failed: %s", exc)

    if not crops:
        return []

    # -- Pass 2: batch-encode all crops in one forward pass --------------------
    try:
        img_embeds = clip_model.encode_images(crops)  # (N, dim)
    except Exception as exc:
        _log.debug("  [Step10/CLIP] batch image encoding failed: %s", exc)
        return []

    # -- Pass 3: score and filter -----------------------------------------------
    # scores_matrix[c, n] = cosine similarity of text[c] vs crop[n]
    scores_matrix = np.dot(text_embeds, img_embeds.T)  # (C, N)
    max_scores = scores_matrix.max(axis=0)  # (N,)
    best_cats = scores_matrix.argmax(axis=0)  # (N,)

    passing: list[dict[str, Any]] = []
    for j, orig_idx in enumerate(valid_indices):
        if max_scores[j] >= threshold:
            m = dict(auto_masks[orig_idx])
            m["matched_category"] = target_categories[int(best_cats[j])]
            m["clip_score"] = round(float(max_scores[j]), 4)
            m["source"] = "clip_filtered_automask"
            passing.append(m)

    return passing


def _sam_directed_by_gemma(
    image: Image.Image,
    gemma_objects: list[dict[str, Any]],
    sam_predictor: Any,
    clip_model: Any,
    use_auto_mask: bool = True,
    path_b_allowed: bool = True,
) -> list[dict[str, Any]]:
    """Segment objects identified by Gemma using SAM.

    Path A (preferred): use Gemma's rough_bbox as direct box prompts to SAM.
    Path B (fallback):  run SAM auto-mask + CLIP zero-shot filtering.
                        Only runs when Path A found nothing AND path_b_allowed.
                        Never used as a supplement to Path A — it is a pure fallback.

    Returns list of mask dicts, each with:
        mask (bool ndarray or None), area_norm (float), category (str),
        score (float), source (str)
    """
    if not gemma_objects:
        return []

    categories = [o.get("category", "") for o in gemma_objects if o.get("category")]
    results: list[dict[str, Any]] = []

    # -- Path A: Gemma bbox prompts --------------------------------------------
    path_a_bboxes: list[tuple[float, float, float, float]] = []
    path_a_categories: list[str] = []
    for obj in gemma_objects:
        bbox = obj.get("rough_bbox", _FALLBACK_BBOX)
        cat = obj.get("category", "unknown")
        # Only use as a prompt when Gemma gave a real estimate
        if _bbox_area(bbox) < _FALLBACK_AREA_THRESHOLD:
            path_a_bboxes.append(tuple(bbox[:4]))  # type: ignore[arg-type]
            path_a_categories.append(cat)

    # When all Gemma bboxes were too large (e.g. aerial full-frame estimates) AND
    # the AMG path is also blocked (cap reached or disabled), use the standard
    # fallback bbox so SAM still runs rather than returning 0 masks.
    if not path_a_bboxes and not path_b_allowed:
        path_a_bboxes = [tuple(_FALLBACK_BBOX)]  # type: ignore[arg-type]
        path_a_categories = [categories[0] if categories else "unknown"]

    if path_a_bboxes:
        try:
            w_img, h_img = image.size
            raw_masks = sam_predictor.predict_boxes(image, path_a_bboxes)
            for mask_info, cat in zip(raw_masks, path_a_categories):
                if mask_info is None:
                    continue
                mask = mask_info.get("mask")
                area_norm = float(mask.sum()) / (w_img * h_img) if mask is not None else 0.0
                results.append(
                    {
                        "mask": mask,
                        "area_norm": round(area_norm, 6),
                        "category": cat,
                        "score": float(mask_info.get("score", 0.0)),
                        "source": "gemma_bbox",
                        "clip_score": None,
                        "matched_category": cat,
                    }
                )
        except Exception as exc:
            _log.debug("  [Step10/SAM] Path A box-prompt failed: %s", exc)

    # -- Path B: auto-mask + CLIP filtering ------------------------------------
    # Pure fallback: only runs when Path A produced no masks at all.
    # NOT used as a supplement — previously, running AMG on every frame (because
    # abstract Gemma categories like "traffic_flow" always have fallback bboxes,
    # making len(path_a_bboxes) < len(gemma_objects)) caused 35+ min freezes.
    path_a_found = len(results) > 0
    path_b_needed = (not path_a_found) and use_auto_mask and path_b_allowed

    if path_b_needed and categories and clip_model is not None:
        try:
            auto_masks = _get_sam_auto_masks(image, sam_predictor)
            filtered = _clip_filter_sam_masks(auto_masks, categories, clip_model, image)
            # Add Path B results; skip near-duplicates of Path A (IoU > 0.7)
            for fm in filtered:
                if not _is_duplicate(fm, results):
                    mask = fm.get("mask")
                    w_img2, h_img2 = image.size
                    area_norm = float(mask.sum()) / (w_img2 * h_img2) if mask is not None else 0.0
                    results.append(
                        {
                            "mask": mask,
                            "area_norm": round(area_norm, 6),
                            "category": fm.get("matched_category", "unknown"),
                            "score": float(fm.get("score", 0.0)),
                            "source": fm.get("source", "clip_filtered_automask"),
                            "clip_score": fm.get("clip_score"),
                            "matched_category": fm.get("matched_category", "unknown"),
                        }
                    )
        except Exception as exc:
            _log.debug("  [Step10/SAM] Path B auto-mask failed: %s", exc)

    return results


def _is_duplicate(
    candidate: dict[str, Any],
    existing: list[dict[str, Any]],
    iou_threshold: float = 0.7,
) -> bool:
    """Return True when candidate mask overlaps any existing mask above iou_threshold."""
    cand_mask = candidate.get("mask")
    if cand_mask is None:
        return False
    for ex in existing:
        ex_mask = ex.get("mask")
        if ex_mask is None:
            continue
        try:
            if cand_mask.shape != ex_mask.shape:
                continue
            inter = float((cand_mask & ex_mask).sum())
            union = float((cand_mask | ex_mask).sum())
            if union > 0 and (inter / union) > iou_threshold:
                return True
        except Exception:
            pass
    return False


# -- Annotation rendering -------------------------------------------------------


def _draw_tracking_frame(
    image: Image.Image,
    detections: list[dict[str, Any]],
    sam_masks: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """Draw tracking boxes (with IDs) and optional SAM masks on *image*."""
    w, h = image.size
    result = image.copy().convert("RGBA")

    # SAM mask overlays (semi-transparent)
    if sam_masks:
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for mask_info in sam_masks:
            mask = mask_info.get("mask")
            if mask is None:
                continue
            # Pick colour by category priority
            from selfsuvis.pipeline.vision.rfdetr import _classify_priority

            cat = mask_info.get("category", "")
            prio = _classify_priority(cat)
            color = _PRIORITY_COLORS.get(prio, (158, 158, 158))
            mask_img = Image.fromarray((mask * 70).astype(np.uint8), mode="L")
            if mask_img.size != (w, h):
                mask_img = mask_img.resize((w, h), Image.NEAREST)
            fill = Image.new("RGBA", (w, h), color + (0,))
            fill.putalpha(mask_img)
            overlay = Image.alpha_composite(overlay, fill)
        result = Image.alpha_composite(result, overlay)

    draw = ImageDraw.Draw(result)
    lw = max(2, int(w * _BOX_WIDTH_RATIO))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        priority = det.get("priority", 4)
        color = _PRIORITY_COLORS.get(priority, (158, 158, 158))
        x1n, y1n, x2n, y2n = det.get("bbox_norm", [0, 0, 1, 1])
        x1, y1 = int(x1n * w), int(y1n * h)
        x2, y2 = int(x2n * w), int(y2n * h)

        for off in range(lw):
            draw.rectangle([x1 + off, y1 + off, x2 - off, y2 - off], outline=color)

        label = det.get("label", "?")
        tid = det.get("track_id", 0)
        conf = det.get("confidence", 0.0)
        badge = f"[{tid}] {label} {conf:.2f}"

        if font is not None:
            try:
                bbox_text = draw.textbbox((x1, max(0, y1 - 16)), badge, font=font)
                draw.rectangle(bbox_text, fill=color + (210,))
                draw.text((x1, max(0, y1 - 16)), badge, fill=(255, 255, 255), font=font)
            except Exception:
                pass

    return result.convert("RGB")


# -- Main step function ---------------------------------------------------------


def step_gemma_directed_tracking(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    models: dict[str, Any],
    gemma_api_url: str,
    gemma_api_model: str,
    precomputed_scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 10: Gemma 4 directed tracking (SAM segmentation + RF-DETR tracking).

    Args:
        frame_list:       List of (frame_path, t_sec) from the extraction step.
        video_name:       Human-readable video identifier.
        video_dir:        Per-video output directory.
        device:           Torch device string ("cpu" or "cuda").
        models:           Shared model dict; uses models["clip"] for CLIP filtering.
        gemma_api_url:    Gemma Ollama/vLLM endpoint URL.
        gemma_api_model:  Gemma model name (e.g. "gemma4:e4b").

    Returns:
        Dict with keys: skipped (bool), scene_type (str), tracking_priority (list),
            n_tracked_objects (int), n_frames (int), sam_masks_total (int),
            elapsed_sec (float), results_json_path (str), summary_md_path (str).
    """
    result: dict[str, Any] = {"skipped": True, "detection_results": []}

    if not settings.RFDETR_ENABLED:
        result["reason"] = "RFDETR_ENABLED=false"
        return result

    if not gemma_api_url:
        result["reason"] = "gemma_api_url not configured"
        return result

    effective_timeout = float(settings.GEMMA_API_TIMEOUT_SEC)
    clip_model = models.get("clip")

    t0 = time.time()

    # -- Sub-step 1: Gemma structured scene analysis ---------------------------
    _log.info(
        "Gemma structured scene analysis (up to %d sampled frames, model=%s) ...",
        _GEMMA_STRUCTURED_SAMPLE_N,
        gemma_api_model,
    )
    if _scene_is_actionable(precomputed_scene):
        gemma_scene = precomputed_scene
        _log.info("Using precomputed Gemma structured scene from step 03")
    else:
        if precomputed_scene:
            _log.info(
                "Precomputed Gemma scene was too weak for tracking; re-running structured vision analysis"
            )
        gemma_scene = _gemma_structured_scene_analysis(
            frame_list,
            api_url=gemma_api_url,
            model=gemma_api_model,
            timeout=effective_timeout,
            clip_model=clip_model,
            video_dir=video_dir,
        )
    scene_type = gemma_scene.get("scene_type", "other")
    tracking_priority = gemma_scene.get("tracking_priority", [])
    gemma_objects = gemma_scene.get("dominant_objects", [])
    areas_of_interest = gemma_scene.get("areas_of_interest", [])
    tracking_targets = _normalise_tracking_targets(tracking_priority, gemma_objects)

    _log.info(
        "Scene: %s | priority: %s | objects: %d",
        scene_type,
        tracking_priority,
        len(gemma_objects),
    )

    # Unload the Gemma Ollama sidecar now — it holds ~9 GiB VRAM and SAM/RF-DETR
    # need that headroom.  _unload_known_sidecars is a no-op if not on CUDA.
    if device == "cuda" and gemma_api_url:
        try:
            from ..caption import _flush_cuda_allocator, _unload_known_sidecars

            _unload_known_sidecars([(gemma_api_url, gemma_api_model)])
            _flush_cuda_allocator()
        except Exception:
            pass

    # -- Sub-step 2: SAM directed segmentation (sampled frames) ---------------
    sam_available = False
    sam_predictor = None
    if settings.SAM_ENABLED and gemma_objects:
        try:
            from selfsuvis.pipeline.vision.sam import SAMPredictor

            sam_predictor = SAMPredictor()
            sam_available = sam_predictor.is_available()
            if not sam_available:
                _log.info("SAM not available — segmentation skipped")
        except Exception as exc:
            _log.debug("SAM import failed: %s", exc)

    # Sample frames for SAM segmentation (same as Gemma analysis sample)
    n_avail = len(frame_list)
    sam_step = max(1, n_avail // _GEMMA_STRUCTURED_SAMPLE_N)
    sam_frames = frame_list[::sam_step][:_GEMMA_STRUCTURED_SAMPLE_N]

    frame_sam_masks: dict[str, list[dict[str, Any]]] = {}  # frame_path → masks
    sam_masks_total = 0

    if sam_available and sam_predictor is not None and gemma_objects:
        _log.info(
            "SAM directed segmentation on %d frames (%d target objects) ...",
            len(sam_frames),
            len(gemma_objects),
        )
        path_b_frames_used = 0  # track how many frames triggered expensive AMG path
        for frame_idx, (fp, _t) in enumerate(sam_frames):
            t_frame = time.time()
            path_b_ok = path_b_frames_used < _MAX_PATH_B_FRAMES
            try:
                img = _open_frame_image(fp)
                masks = _sam_directed_by_gemma(
                    img,
                    gemma_objects,
                    sam_predictor,
                    clip_model,
                    path_b_allowed=path_b_ok,
                )
                # Count how many results came from Path B (AMG) to track the cap
                for m in masks:
                    if m.get("source") in ("clip_filtered_automask",):
                        path_b_frames_used += 1
                        break
                # Drop raw numpy masks from JSON-able summary (keep metadata only)
                mask_summary = [
                    {
                        "category": m.get("category", "unknown"),
                        "area_norm": m.get("area_norm", 0.0),
                        "source": m.get("source", "unknown"),
                        "score": round(float(m.get("score", 0.0)), 4),
                        "clip_score": m.get("clip_score"),
                    }
                    for m in masks
                ]
                frame_sam_masks[fp] = mask_summary
                sam_masks_total += len(masks)
                _log.info(
                    "  SAM frame %d/%d: %d masks (%.1fs)%s",
                    frame_idx + 1,
                    len(sam_frames),
                    len(masks),
                    time.time() - t_frame,
                    " [amg]"
                    if any(m.get("source") == "clip_filtered_automask" for m in masks)
                    else "",
                )
            except Exception as exc:
                _log.warning("  SAM frame %d/%d failed: %s", frame_idx + 1, len(sam_frames), exc)
        _log.info(
            "SAM segmentation done: %d masks across %d frames (%d used AMG fallback)",
            sam_masks_total,
            len(sam_frames),
            path_b_frames_used,
        )

    # -- Sub-step 3: RF-DETR tracking ------------------------------------------
    # Release SAM before loading RF-DETR — SAM holds ~10 GiB which leaves no
    # room for RF-DETR on 12 GiB cards.  SAM is finished at this point.
    if sam_predictor is not None:
        sam_predictor.release()
        sam_predictor = None
        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    n_avail = len(frame_list)
    track_step = max(1, n_avail // _MAX_TRACKING_FRAMES)
    track_frames = frame_list[::track_step][:_MAX_TRACKING_FRAMES]
    n_track = len(track_frames)

    tracker = RFDETRTracker()
    tracking_results: list[dict[str, Any]] = []
    n_unique_track_ids = 0
    by_category: dict[str, int] = {}
    total_objects = 0

    if tracker.is_enabled():
        _log.info(
            "RF-DETR tracking (%s) on %d/%d frames, target_labels=%s",
            tracker.model_id,
            n_track,
            n_avail,
            tracking_targets or "(all)",
        )
        effective_tracking_targets: list[str] | None = (
            list(tracking_targets) if tracking_targets else None
        )
        tracking_results = tracker.track_sequence(
            track_frames,
            target_labels=effective_tracking_targets,
        )
        filter_retry_mode = "none"
        if tracking_targets:
            first_pass_total = sum(
                len(frame_res.get("detections", [])) for frame_res in tracking_results
            )
            if first_pass_total == 0:
                reduced_targets = [
                    label for label in tracking_targets if label in {"vehicle", "person"}
                ]
                if reduced_targets and reduced_targets != tracking_targets:
                    _log.warning(
                        "RF-DETR returned zero detections for Gemma targets %s; retrying with reduced filter %s",
                        tracking_targets,
                        reduced_targets,
                    )
                    tracking_results = tracker.track_sequence(
                        track_frames, target_labels=reduced_targets
                    )
                    effective_tracking_targets = list(reduced_targets)
                    filter_retry_mode = "reduced"
                second_pass_total = sum(
                    len(frame_res.get("detections", [])) for frame_res in tracking_results
                )
                if second_pass_total == 0:
                    _log.warning(
                        "RF-DETR returned zero detections for Gemma targets %s; retrying without label filter",
                        tracking_targets,
                    )
                    tracking_results = tracker.track_sequence(track_frames, target_labels=None)
                    effective_tracking_targets = None
                    filter_retry_mode = "unfiltered"
        # Collect stats
        all_track_ids: set = set()
        for frame_res in tracking_results:
            for det in frame_res.get("detections", []):
                total_objects += 1
                tid = det.get("track_id", 0)
                if tid:
                    all_track_ids.add(tid)
                lbl = det.get("label", "unknown")
                by_category[lbl] = by_category.get(lbl, 0) + 1
        n_unique_track_ids = len(all_track_ids)
        _log.info(
            "RF-DETR done: %d objects, %d unique tracks in %d frames",
            total_objects,
            n_unique_track_ids,
            n_track,
        )
        mean_track_len, median_track_len = _track_length_stats(tracking_results)
    else:
        _log.info("RF-DETR disabled — tracking skipped")
        tracking_results = [
            {"frame_path": fp, "t_sec": t, "detections": []} for fp, t in track_frames
        ]
        filter_retry_mode = "none"
        effective_tracking_targets = None
        mean_track_len = 0.0
        median_track_len = 0.0

    # -- Annotate frames -------------------------------------------------------
    out_dir = video_dir / "gemma_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_paths: list[str] = []

    for frame_res in tracking_results:
        fp = frame_res["frame_path"]
        t_sec = frame_res["t_sec"]
        dets = frame_res.get("detections", [])

        # Build renderable mask list (source field only, no numpy — use color blobs)
        # For annotation, rebuild minimal mask from bbox if raw mask not available
        try:
            img = _open_frame_image(fp)
            annotated = _draw_tracking_frame(img, dets, sam_masks=None)
            ann_path = out_dir / f"frame_{t_sec:.3f}_tracked.jpg"
            annotated.save(ann_path, quality=88)
            annotated_paths.append(str(ann_path))
        except Exception as exc:
            _log.debug("annotation failed for %s: %s", fp, exc)

    # -- Save JSON results -----------------------------------------------------
    elapsed = time.time() - t0

    results_json = {
        "model": tracker.model_id,
        "gemma_model": gemma_api_model,
        "gemma_scene_type": scene_type,
        "tracking_priority": tracking_priority,
        "tracking_targets_effective": effective_tracking_targets or [],
        "tracking_filter_retry_mode": filter_retry_mode,
        "dominant_objects": [
            {k: v for k, v in o.items() if k != "rough_bbox"}
            | {"rough_bbox": o.get("rough_bbox", _FALLBACK_BBOX)}
            for o in gemma_objects
        ],
        "areas_of_interest": areas_of_interest,
        "sam_enabled": sam_available,
        "sam_masks_total": sam_masks_total,
        "n_frames": n_track,
        "n_unique_track_ids": n_unique_track_ids,
        "total_detections": total_objects,
        "mean_track_length_frames": round(mean_track_len, 2),
        "median_track_length_frames": round(median_track_len, 2),
        "by_category": by_category,
        "elapsed_sec": round(elapsed, 2),
        "frames": [
            {
                "frame_path": r["frame_path"],
                "t_sec": r["t_sec"],
                "n_detections": len(r.get("detections", [])),
                "detections": r.get("detections", []),
                "sam_masks": frame_sam_masks.get(r["frame_path"], []),
            }
            for r in tracking_results
        ],
    }
    results_path = video_dir / "gemma_tracking_results.json"
    write_json_artifact(results_path, results_json, ensure_ascii=False)
    _log.info("  [ok] Gemma tracking results → %s", results_path)

    # -- Write summary markdown ------------------------------------------------
    summary_path = _write_gemma_tracking_summary_md(
        video_dir,
        video_name,
        gemma_scene,
        results_json,
        annotated_paths=annotated_paths[:8],
        elapsed_sec=elapsed,
    )

    # Release models (sam_predictor already released before RF-DETR load)
    tracker.release()
    if sam_predictor is not None:
        sam_predictor.release()
        sam_predictor = None

    result.update(
        {
            "skipped": False,
            "scene_type": scene_type,
            "tracking_priority": tracking_priority,
            "n_tracked_objects": n_unique_track_ids,
            "n_frames": n_track,
            "sam_masks_total": sam_masks_total,
            "total_objects": total_objects,
            "elapsed_sec": elapsed,
            "results_json_path": str(results_path),
            "summary_md_path": str(summary_path),
            "sam_enabled": sam_available,
            "annotated_count": len(annotated_paths),
            "tracking_results": tracking_results,
            "tracking_targets_effective": effective_tracking_targets or [],
            "tracking_filter_retry_mode": filter_retry_mode,
        }
    )
    return result


# -- Summary markdown writer ----------------------------------------------------


def _write_gemma_tracking_summary_md(
    video_dir: Path,
    video_name: str,
    gemma_scene: dict[str, Any],
    results_json: dict[str, Any],
    annotated_paths: list[str],
    elapsed_sec: float,
) -> str:
    """Write ``gemma_tracking_summary.md`` and return its path."""
    scene_type = gemma_scene.get("scene_type", "other")
    tracking_priority = gemma_scene.get("tracking_priority", [])
    dominant_objects = gemma_scene.get("dominant_objects", [])
    areas = gemma_scene.get("areas_of_interest", [])
    motion = gemma_scene.get("motion_present", False)

    n_tracks = results_json.get("n_unique_track_ids", 0)
    n_frames = results_json.get("n_frames", 0)
    total_dets = results_json.get("total_detections", 0)
    by_cat = results_json.get("by_category", {})
    sam_total = results_json.get("sam_masks_total", 0)
    sam_enabled = results_json.get("sam_enabled", False)
    rfdetr_model = results_json.get("model", "rfdetr_base")
    gemma_model = results_json.get("gemma_model", "?")

    lines = [
        f"# Gemma Directed Tracking — {video_name}",
        "",
        f"Generated by `steps_gemma_tracking.py` | {n_frames} frames | {elapsed_sec:.1f}s",
        "",
        "## Gemma Scene Interpretation",
        "",
        f"- **Scene type**: `{scene_type}`",
        f"- **Motion detected**: {motion}",
        f"- **Gemma model**: `{gemma_model}`",
        "",
    ]
    if areas:
        lines += ["**Areas of interest**:", ""]
        for a in areas:
            lines.append(f"- {a}")
        lines.append("")

    if dominant_objects:
        lines += [
            "**Dominant objects detected by Gemma**:",
            "",
            "| Category | Count est. | Spatial hint | Bbox source |",
            "|----------|-----------|-------------|------------|",
        ]
        for o in dominant_objects:
            cat = o.get("category", "?")
            cnt = o.get("count_estimate", 1)
            hint = o.get("spatial_hint", "")
            bbox = o.get("rough_bbox", _FALLBACK_BBOX)
            is_fallback = _bbox_area(bbox) >= _FALLBACK_AREA_THRESHOLD
            bbox_note = "fallback (whole frame)" if is_fallback else "Gemma estimate"
            lines.append(f"| {cat} | {cnt} | {hint} | {bbox_note} |")
        lines.append("")

    if tracking_priority:
        lines += [
            "**Tracking priority order** (passed to RF-DETR):",
            "",
        ]
        for i, lbl in enumerate(tracking_priority, 1):
            lines.append(f"{i}. `{lbl}`")
        lines.append("")

    lines += [
        "## RF-DETR Tracking Results",
        "",
        f"- **Model**: `{rfdetr_model}`",
        f"- **Frames analysed**: {n_frames}",
        f"- **Total detections**: {total_dets}",
        f"- **Unique track IDs**: {n_tracks}",
        "",
    ]
    if by_cat:
        lines += [
            "**Detections by class**:",
            "",
            "| Class | Count |",
            "|-------|-------|",
        ]
        for cls, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
            lines.append(f"| {cls} | {cnt} |")
        lines.append("")

    lines += [
        "## SAM Segmentation",
        "",
    ]
    if sam_enabled:
        lines += [
            f"SAM directed segmentation produced **{sam_total} masks**.",
            "",
            "- **Path A** (Gemma bbox → SAM box-prompt): used when Gemma "
            "provided a non-fallback rough_bbox.",
            "- **Path B** (SAM auto-mask + CLIP filter): used only as a fallback "
            "when Path A yields no masks for a frame.",
        ]
    else:
        lines += [
            "SAM segmentation was **not available** (SAM_ENABLED=false or no backend installed).",
            "",
            "To enable: set `SAM_ENABLED=true`, install project extras with `make venv`, or add `sam3` manually.",
        ]
    lines.append("")

    if annotated_paths:
        lines += ["## Sample Annotated Frames", ""]
        for p in annotated_paths[:6]:
            try:
                rel = Path(p).relative_to(video_dir)
            except ValueError:
                rel = Path(p).name
            lines.append(f"- `{rel}`")
        lines.append("")

    lines += [
        "## Artifacts",
        "",
        "- `gemma_tracking_results.json` — full per-frame detection + SAM metadata",
        f"- `gemma_tracking/frame_*.jpg` — annotated frames ({len(annotated_paths)} written)",
        "",
        f"*Elapsed: {elapsed_sec:.1f}s for {n_frames} frames*",
    ]

    out_path = video_dir / "gemma_tracking_summary.md"
    write_markdown_artifact(out_path, lines)
    _log.info("  [ok] Gemma tracking summary → %s", out_path)
    return str(out_path)

"""Gemma sidecar API helpers: frame analysis, structured extraction, segment diff, Qwen fallback."""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..common import gemma_frame_cache_key, load_gemma_cache, save_gemma_cache
from .frame_selection import _adaptive_sparse_budget

if TYPE_CHECKING:
    pass

_log = get_logger("pipeline.local.caption")

_STRUCTURED_SCENE_TYPES = frozenset(
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

_SEGMENT_DIFF_PROMPT = (
    "You are comparing two consecutive frames from a video mission. "
    "The LEFT image is the last frame of scene segment N; "
    "the RIGHT image is the first frame of scene segment N+1. "
    "In 2-3 sentences describe: what changed between the two frames? "
    "Focus on movement, new objects, environment changes, viewpoint shift. "
    "Be concise and factual."
)

_GEMMA_QWEN_FALLBACK_PROMPT = (
    "Analyse this image and return ONLY a JSON object with these keys:\n"
    "{\n"
    '  "vehicle_groups": [\n'
    '    {"type": "truck|car|bus|motorcycle|emergency|military|van|other",\n'
    '     "count": <integer>,\n'
    '     "color": "<dominant color or unknown>",\n'
    '     "position": "<front|centre|rear|left|right|scattered>"}\n'
    "  ],\n"
    '  "road_surface": "asphalt|concrete|gravel|dirt|unknown",\n'
    '  "road_condition": "clear|wet|snow|ice|debris|unknown",\n'
    '  "scene_summary": "<one sentence describing the scene>"\n'
    "}\n"
    "If no vehicles are visible, return an empty vehicle_groups list. "
    "Return only the JSON object, no other text."
)


def _fallback_tracking_bbox(scene_type: str) -> list[float]:
    """Return a moderate default bbox when text-only Gemma summaries omit one."""
    if scene_type == "aerial":
        return [0.22, 0.28, 0.78, 0.72]
    if scene_type in {"urban_street", "construction", "industrial"}:
        return [0.18, 0.24, 0.82, 0.78]
    return [0.2, 0.2, 0.8, 0.8]


def _gemma_analyse_frame_via_api(
    fp: str,
    api_url: str,
    model: str,
    timeout: float,
    *,
    video_dir: Path | None = None,
) -> str:
    """Send a single frame to a Gemma Ollama/vLLM sidecar and return its description."""
    import base64
    import io

    try:
        import httpx
    except ImportError:
        return ""

    cache: dict[str, Any] = {}
    cache_key = ""
    if video_dir is not None and settings.GEMMA_CACHE_RESPONSES:
        try:
            cache = load_gemma_cache(video_dir, enabled=settings.GEMMA_CACHE_RESPONSES)
            cache_key = gemma_frame_cache_key(fp, model=model, prompt_tag="gemma_analysis_v1")
            if cache_key in cache:
                return str(cache[cache_key].get("content", "") or "")
        except Exception:
            cache = {}
            cache_key = ""

    try:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((768, 768))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyse this frame from aerial/robotics mission video. "
                                "Describe in 2-3 sentences: scene type, visible objects, "
                                "terrain, any notable features or anomalies. "
                                "Be concise and factual."
                            ),
                        },
                    ],
                }
            ],
            # 600 tokens: thinking models (gemma4:e4b) consume ~300-400 on reasoning
            # before writing the final answer into content.
            "max_tokens": 600,
            "temperature": 0.1,
        }
        endpoint = f"{api_url.rstrip('/')}/chat/completions"
        t_req = time.time()
        resp = httpx.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content")
        # Thinking models (gemma4:e4b) place the answer in content and
        # the chain-of-thought in reasoning.  If content is still empty
        # (budget exhausted on reasoning), use the last sentence of reasoning.
        if not content:
            reasoning = msg.get("reasoning") or msg.get("thinking") or ""
            if reasoning:
                # Take last non-empty sentence as a best-effort summary
                sentences = [
                    s.strip() for s in reasoning.replace("\n", " ").split(".") if s.strip()
                ]
                content = sentences[-1] if sentences else reasoning[-200:]
        # content may be a list of parts (some backends)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        elapsed = time.time() - t_req
        if elapsed >= float(settings.GEMMA_SLOW_CALL_SEC):
            _log.info("  [Gemma API] slow frame analysis: %.1fs for %s", elapsed, Path(fp).name)
        content = (content or "").strip()
        if cache_key:
            cache[cache_key] = {"content": content, "elapsed_sec": round(elapsed, 3)}
            save_gemma_cache(video_dir, cache, enabled=settings.GEMMA_CACHE_RESPONSES)  # type: ignore[arg-type]
        return content
    except Exception as exc:
        _log.debug("  [Gemma API] frame analysis failed for %s: %s", Path(fp).name, exc)
        return ""


def _summarise_gemma_captions_to_structured_scene(
    gemma_captions: list[dict[str, Any]],
    api_url: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Use one text-only call to derive a structured scene summary from step 03 descriptions."""

    def _empty_structured_scene() -> dict[str, Any]:
        return {
            "scene_type": "other",
            "dominant_objects": [],
            "areas_of_interest": [],
            "motion_present": False,
            "tracking_priority": [],
        }

    def _clean_structured_scene(parsed: dict[str, Any]) -> dict[str, Any]:
        scene_type = str(parsed.get("scene_type") or "").strip().lower()
        if scene_type not in _STRUCTURED_SCENE_TYPES or "|" in scene_type or "<" in scene_type:
            return _empty_structured_scene()

        clean_objects: list[dict[str, Any]] = []
        for obj in parsed.get("dominant_objects", []):
            if not isinstance(obj, dict):
                continue
            category = str(obj.get("category") or "").strip().lower()
            if not category or any(token in category for token in ("<", ">", "|", "e.g.")):
                continue
            bbox = obj.get("rough_bbox")
            spatial_hint = str(obj.get("spatial_hint") or "").strip()
            bbox_fallback = False
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    x1, y1, x2, y2 = _fallback_tracking_bbox(scene_type)
                    bbox_fallback = True
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    x1, y1, x2, y2 = _fallback_tracking_bbox(scene_type)
                    bbox_fallback = True
            else:
                x1, y1, x2, y2 = _fallback_tracking_bbox(scene_type)
                bbox_fallback = True
            if bbox_fallback:
                spatial_hint = f"{spatial_hint} fallback-bbox".strip()
            try:
                count_estimate = int(float(obj.get("count_estimate") or 1))
            except Exception:
                count_estimate = 1
            clean_objects.append(
                {
                    "category": category,
                    "count_estimate": count_estimate,
                    "spatial_hint": spatial_hint,
                    "rough_bbox": [x1, y1, x2, y2],
                }
            )

        priorities = []
        for item in parsed.get("tracking_priority", []):
            label = str(item or "").strip().lower()
            if label and not any(token in label for token in ("<", ">", "|")):
                priorities.append(label)

        if not clean_objects and priorities:
            fallback_bbox = _fallback_tracking_bbox(scene_type)
            for label in priorities[:2]:
                clean_objects.append(
                    {
                        "category": label,
                        "count_estimate": 1,
                        "spatial_hint": "scene-context fallback",
                        "rough_bbox": list(fallback_bbox),
                    }
                )

        areas = [
            str(item).strip()
            for item in parsed.get("areas_of_interest", [])
            if str(item or "").strip()
        ][:3]
        return {
            "scene_type": scene_type,
            "dominant_objects": clean_objects,
            "areas_of_interest": areas,
            "motion_present": bool(parsed.get("motion_present", False)),
            "tracking_priority": priorities[:5],
        }

    try:
        import httpx
    except ImportError:
        return _empty_structured_scene()

    description_lines = [
        f"- t={float(item.get('t_sec', 0.0)):.1f}s: {str(item.get('description', '') or '').strip()}"
        for item in gemma_captions
        if str(item.get("description", "") or "").strip()
    ]
    if not description_lines:
        return _empty_structured_scene()
    prompt = (
        "You are converting frame descriptions into structured scene JSON for object tracking.\n"
        "Return ONLY valid JSON. For scene_type, choose exactly one value from "
        "urban_street, rural_terrain, indoor, aerial, waterway, construction, industrial, other. "
        "Do not copy the list as a pipe-separated string.\n"
        "Use this schema shape:\n"
        "{"
        '"scene_type":"aerial",'
        '"dominant_objects":[{"category":"vehicle","count_estimate":1,"spatial_hint":"center","rough_bbox":[0.1,0.1,0.9,0.9]}],'
        '"areas_of_interest":["..."],'
        '"motion_present":true,'
        '"tracking_priority":["vehicle","person"]'
        "}\n"
        "Use only detector-aligned classes where possible, especially vehicle/person/sign/building.\n"
        "Descriptions:\n" + "\n".join(description_lines[:20])
    )
    base = api_url.rstrip("/")
    endpoint = f"{base}/chat/completions"
    # Ollama native endpoint — bypasses thinking-token routing for models like gemma4
    ollama_base = base[:-3] if base.endswith("/v1") else base
    ollama_endpoint = f"{ollama_base}/api/chat"
    t_req = time.time()
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        # Thinking models (e.g. gemma4:e4b) emit output as reasoning tokens, leaving
        # content empty on the OpenAI path.  Fall back to the Ollama native endpoint
        # which bypasses the thinking mechanism and writes directly to content.
        if not content.strip():
            try:
                native_resp = httpx.post(
                    ollama_endpoint,
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=timeout,
                )
                native_resp.raise_for_status()
                native_content = native_resp.json().get("message", {}).get("content", "")
                if native_content.strip():
                    content = native_content
            except Exception:
                pass
        if not content:
            content = msg.get("reasoning") or msg.get("thinking") or ""
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.lower().startswith("json"):
                content = content[4:]
        elapsed = time.time() - t_req
        if elapsed >= float(settings.GEMMA_SLOW_CALL_SEC):
            _log.info("  [Gemma API] slow structured-summary synthesis: %.1fs", elapsed)
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict):
                return _clean_structured_scene(parsed)
        except Exception:
            pass
    except Exception as exc:
        _log.warning(
            "  [Gemma API] structured-scene synthesis failed (%s) — using empty scene", exc
        )
    return _empty_structured_scene()


def _gemma_diff_two_frames_via_api(
    fp_before: str,
    fp_after: str,
    api_url: str,
    model: str,
    timeout: float,
    backend: str = "",
) -> str:
    """Send two frames to a Gemma sidecar and return a diff description.

    Tries OpenAI-compatible /v1/chat/completions with two image_url entries,
    then falls back to Ollama native /api/chat with images:[b64_a, b64_b].
    When the backend is explicitly Ollama, use the native endpoint first to
    avoid paying a full OpenAI-compat timeout for every boundary pair.
    Returns "" on any failure.
    """
    import base64
    import io as _io

    try:
        import httpx
    except ImportError:
        return ""

    def _encode(fp: str) -> str:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((768, 768))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    try:
        b64_before = _encode(fp_before)
        b64_after = _encode(fp_after)
    except Exception as exc:
        _log.debug("  [Gemma diff] image load failed: %s", exc)
        return ""

    base = api_url.rstrip("/")
    openai_endpoint = f"{base}/chat/completions"
    ollama_base = base[:-3] if base.endswith("/v1") else base
    ollama_endpoint = f"{ollama_base}/api/chat"
    use_ollama_first = backend.lower() == "ollama" or (not backend and ":11434" in api_url)

    openai_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_before}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_after}"},
                    },
                    {"type": "text", "text": _SEGMENT_DIFF_PROMPT},
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    native_payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": _SEGMENT_DIFF_PROMPT,
                "images": [b64_before, b64_after],
            }
        ],
    }

    def _request_openai_compat() -> str:
        try:
            resp = httpx.post(openai_endpoint, json=openai_payload, timeout=timeout)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            if content.strip():
                return content.strip()
        except Exception as exc:
            _log.debug("  [Gemma diff] OpenAI-compat request failed: %s", exc)
        return ""

    def _request_ollama_native() -> str:
        try:
            resp = httpx.post(ollama_endpoint, json=native_payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            if content.strip():
                return content.strip()
        except Exception as exc:
            _log.debug("  [Gemma diff] Ollama native request failed: %s", exc)
        return ""

    if use_ollama_first:
        return _request_ollama_native() or _request_openai_compat()

    return _request_openai_compat() or _request_ollama_native()


def _gemma_extract_frame_structured(
    fp: str,
    api_url: str,
    model: str,
    timeout: float,
    t_sec: float,
) -> dict[str, Any]:
    """Call Gemma sidecar for per-frame structured JSON matching the Qwen schema.

    Returns a dict with the same keys as QwenModel.extract_frame_facts output:
    vehicle_groups, road_surface, road_condition, scene_summary, t_sec, frame_path.
    Returns a partial result with parse_error=True on JSON parse failure.
    """
    import base64
    import io as _io
    import json as _json
    import re as _re

    base_result: dict[str, Any] = {
        "t_sec": t_sec,
        "frame_path": fp,
        "vehicle_groups": [],
        "road_surface": "unknown",
        "road_condition": "unknown",
        "scene_summary": "",
    }

    try:
        import httpx
    except ImportError:
        return {**base_result, "skipped": True, "reason": "httpx not installed"}

    try:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((512, 512))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        _log.debug("  [Gemma fallback] image load failed %s: %s", Path(fp).name, exc)
        return {**base_result, "skipped": True, "reason": str(exc)}

    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    ollama_base = api_url.rstrip("/")
    if ollama_base.endswith("/v1"):
        ollama_base = ollama_base[:-3]
    ollama_endpoint = f"{ollama_base}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _GEMMA_QWEN_FALLBACK_PROMPT},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    raw_content = ""
    try:
        resp = httpx.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        raw_content = msg.get("content") or ""
        if isinstance(raw_content, list):
            raw_content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content
            )
    except Exception as exc:
        _log.debug("  [Gemma fallback] API call failed %s: %s", Path(fp).name, exc)

    # Ollama native fallback when content is empty (thinking model)
    if not raw_content.strip():
        try:
            native_payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": _GEMMA_QWEN_FALLBACK_PROMPT,
                        "images": [b64],
                    }
                ],
            }
            resp = httpx.post(ollama_endpoint, json=native_payload, timeout=timeout)
            resp.raise_for_status()
            raw_content = resp.json().get("message", {}).get("content", "")
        except Exception:
            pass

    if not raw_content.strip():
        return {**base_result, "service_unavailable": True}

    text = raw_content.strip()
    match = _re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {**base_result, "parse_error": True, "raw_content": text[:200]}
    try:
        parsed = _json.loads(match.group())
    except _json.JSONDecodeError:
        return {**base_result, "parse_error": True, "raw_content": text[:200]}

    result = {**base_result}
    result["vehicle_groups"] = parsed.get("vehicle_groups") or []
    if not isinstance(result["vehicle_groups"], list):
        result["vehicle_groups"] = []
    result["road_surface"] = str(parsed.get("road_surface") or "unknown").lower()
    result["road_condition"] = str(parsed.get("road_condition") or "unknown").lower()
    result["scene_summary"] = str(parsed.get("scene_summary") or "").strip()
    return result


def step_qwen_captioning_gemma_fallback(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    gemma_url: str,
    gemma_model: str,
) -> dict[str, Any]:
    """Gemma fallback for step_qwen_captioning when QWEN_API_URL is unset.

    Produces per-frame structured JSON with the same schema as QwenModel
    (vehicle_groups, road_surface, road_condition, scene_summary) via the
    Gemma sidecar. Frame selection mirrors the Qwen budget logic.
    """
    from ..report import write_detailed_captions_md

    out_md = video_dir / "detailed_captions.md"
    result: dict[str, Any] = {"skipped": True, "results": []}
    effective_timeout = float(settings.GEMMA_API_TIMEOUT_SEC)

    budget = _adaptive_sparse_budget(
        frame_list,
        configured_max=max(1, int(settings.QWEN_MAX_FRAMES)),
        seconds_per_sample=0.9,
        floor=8,
    )
    sampled = frame_list[:: max(1, len(frame_list) // max(1, budget))][:budget]

    _log.info(
        "Gemma structured extraction (Qwen fallback): %d/%d frames  model=%s ...",
        len(sampled),
        len(frame_list),
        gemma_model,
    )
    t0 = time.time()
    caption_results: list[dict[str, Any]] = []
    for fp, t_sec in sampled:
        r = _gemma_extract_frame_structured(fp, gemma_url, gemma_model, effective_timeout, t_sec)
        caption_results.append(r)

    elapsed = time.time() - t0
    ok = sum(
        1
        for r in caption_results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    )
    parse_errors = sum(1 for r in caption_results if r.get("parse_error"))
    _log.info(
        "  [ok] Gemma fallback: %d/%d frames extracted in %.1fs (parse_errors=%d)",
        ok,
        len(sampled),
        elapsed,
        parse_errors,
    )
    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, gemma_model)
    result.update(
        {
            "skipped": False,
            "results": caption_results,
            "ok_count": ok,
            "elapsed_sec": elapsed,
            "sampled_count": len(sampled),
            "total_frames": len(frame_list),
            "parse_error_count": parse_errors,
            "backend": "gemma_fallback",
        }
    )
    return result

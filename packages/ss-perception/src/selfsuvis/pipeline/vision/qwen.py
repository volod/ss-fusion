"""Thin HTTP client for Gemma-4 structured scene extraction via vLLM/ollama sidecar.

Phase 2 of the selfsuvis scene captioning system. This module provides
`QwenModel`, which calls an OpenAI-compatible vision endpoint (now backed by
Gemma-4 via ``GEMMA_API_URL``) and returns structured ``frame_facts_json``
dicts for vehicle/road scene understanding.

The class retains the name ``QwenModel`` and exposes an identical public
interface to preserve backward compatibility with callers in ``indexer.py``,
``steps_caption.py``, and test mocks.

The contract: ``extract_frame_facts`` never returns None.
"""

import base64
import concurrent.futures
import io
import json
from collections.abc import Callable
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)
# Use Gemma API timeout if configured; fall back to legacy QWEN_TIMEOUT_SEC.
# Minimum 90s to accommodate multimodal inference on large frames.
_EFFECTIVE_QWEN_TIMEOUT_SEC = max(
    settings.GEMMA_API_TIMEOUT_SEC if settings.GEMMA_API_URL else settings.QWEN_TIMEOUT_SEC,
    90,
)

# -- Module-level constants ----------------------------------------------------

_VEHICLE_LABELS: frozenset = frozenset(
    {
        "vehicle",
        "truck",
        "car",
        "bus",
        "convoy",
        "emergency vehicle",
        "armoured vehicle",
        "tank",
        "motorcycle",
        "van",
    }
)

_QWEN_SYSTEM_PROMPT = (
    "You are a precise outdoor-scene analyst specialised in military and "
    "logistics convoy imagery. Extract structured facts from the image. "
    "Respond ONLY with valid JSON — no markdown, no extra text."
)

_QWEN_USER_PROMPT = (
    "Analyse the image and return ONLY a JSON object with these keys:\n"
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
    "If no vehicles are visible, return an empty vehicle_groups list."
)


def _looks_like_ollama(api_url: str) -> bool:
    return ":11434" in (api_url or "")


def _request_kwargs_for_backend(api_url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": 512,
        "temperature": 0.0,
    }
    if _looks_like_ollama(api_url):
        kwargs["extra_body"] = {"format": "json"}
    return kwargs


# -- Helpers -------------------------------------------------------------------


def _build_user_content(
    image: Image.Image,
    subtitle_text: str | None = None,
    ocr_text: str | None = None,
    extra_context: str | None = None,
) -> list:
    """Build the user-message content list enriched with all available prior knowledge.

    Injection order (each block optional):
      1. Image
      2. Task prompt
      3. Prior knowledge block (Florence caption, depth, detections, scene segment,
         previous Qwen state) — from VideoKnowledge.context_for_frame()
      4. ASR audio context
      5. OCR visible text
    """
    b64 = _encode_image_base64(image)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {"type": "text", "text": _QWEN_USER_PROMPT},
    ]
    if extra_context and extra_context.strip():
        content.append(
            {
                "type": "text",
                "text": f"\n[Prior observations about this scene]:\n{extra_context.strip()}",
            }
        )
    if subtitle_text and subtitle_text.strip():
        content.append(
            {
                "type": "text",
                "text": f"\n[Audio context at this moment]: {subtitle_text.strip()}",
            }
        )
    if ocr_text and ocr_text.strip():
        content.append(
            {
                "type": "text",
                "text": f"\n[Text visible in frame]: {ocr_text.strip()}",
            }
        )
    return content


def _encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string (data URI body only)."""
    max_side = max(0, int(getattr(settings, "QWEN_IMAGE_MAX_SIDE", 0) or 0))
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _health_check_vllm(base_url: str, timeout: int) -> bool:
    """Return True if the vLLM health endpoint responds with HTTP 200."""
    try:
        import httpx

        # Strip trailing /v1 if present to get the server root.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        resp = httpx.get(f"{root}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _health_check_ollama(base_url: str, timeout: int) -> bool:
    """Return True if the ollama /api/tags endpoint responds with HTTP 200."""
    try:
        import httpx

        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        resp = httpx.get(f"{root}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _parse_qwen_response(raw_text: str) -> dict[str, Any]:
    """Parse raw Qwen response text into a structured dict.

    Strips markdown code fences, parses JSON, validates top-level structure,
    and returns a normalised dict. On any error returns a parse_error dict.
    """
    import re as _re

    text = raw_text.strip()

    # Strip <think>...</think> blocks produced by reasoning models.
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    # Strip markdown code fences: ```json...``` or ```...```
    if text.startswith("```"):
        lines = text.splitlines()
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block from surrounding text — small
        # models sometimes prepend a sentence before the JSON object.
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        return {"parse_error": True, "raw": raw_text[:500]}

    # Unwrap single-key envelope: {"analysis": {...}} → use inner dict.
    if isinstance(data, dict) and len(data) == 1:
        only_val = next(iter(data.values()))
        if isinstance(only_val, dict):
            data = only_val

    # Accept first element of a JSON array.
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        return {"parse_error": True, "raw": raw_text[:500]}

    # Normalise: ensure expected keys are present with safe defaults
    normalised: dict[str, Any] = {
        "vehicle_groups": data.get("vehicle_groups", []),
        "road_surface": data.get("road_surface", "unknown"),
        "road_condition": data.get("road_condition", "unknown"),
        "scene_summary": data.get("scene_summary", ""),
    }

    # Coerce vehicle_groups to a list
    if not isinstance(normalised["vehicle_groups"], list):
        normalised["vehicle_groups"] = []

    return normalised


# -- Main class ----------------------------------------------------------------


class QwenModel:
    """HTTP client for Gemma-4 structured scene extraction (OpenAI-compatible API).

    Previously backed by Qwen2.5-VL; now uses ``GEMMA_API_URL`` /
    ``GEMMA_API_MODEL`` when set, falling back to ``QWEN_API_URL`` /
    ``QWEN_MODEL`` for legacy deployments that have not yet migrated.

    Parameters
    ----------
    clip_prescreen_fn:
        Optional callable that accepts a PIL.Image and returns True if the
        image likely contains a vehicle (above threshold). When provided,
        this avoids loading a second CLIP model inside QwenModel — the
        caller (VideoIndexer) passes a closure that reuses the already-loaded
        OpenCLIPEmbedder.
    """

    def __init__(self, clip_prescreen_fn: Callable[[Image.Image], bool] | None = None):
        self._clip_prescreen_fn = clip_prescreen_fn
        self._tagger = None  # lazily initialised OpenCLIPTagger
        self._healthy: bool | None = None  # cached health state
        self._client = None

    # -- Public interface ------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True when GEMMA_API_URL or QWEN_API_URL is configured (non-empty)."""
        return bool(settings.GEMMA_API_URL or settings.QWEN_API_URL)

    def is_healthy(self) -> bool:
        """Return True when the configured sidecar is reachable.

        Result is cached after the first call to avoid repeated network round
        trips during batch processing.
        """
        if not self.is_enabled():
            return False
        if self._healthy is None:
            self._check_health()
        return bool(self._healthy)

    def extract_frame_facts(
        self,
        image: Image.Image,
        subtitle_text: str | None = None,
        ocr_text: str | None = None,
    ) -> dict[str, Any]:
        """Extract structured scene facts from a single frame image.

        Parameters
        ----------
        image:
            The frame to analyse.
        subtitle_text:
            Optional ASR transcription near this frame's timestamp (from the
            audio track).  When provided, injected into the Qwen prompt as
            ``[Audio context at this moment]`` to help disambiguate scene content.
        ocr_text:
            Optional OCR-extracted text visible in the frame.  When provided,
            injected as ``[Text visible in frame]`` in the Qwen prompt.

        Returns a dict — never None — with one of the following shapes:

        - Disabled:             ``{"disabled": True}``
        - Service unavailable:  ``{"service_unavailable": True}``
        - CLIP filtered:        ``{"clip_filtered": True, "reason": "below_vehicle_threshold"}``
        - Success:              ``{"vehicle_groups": [...], "road_surface": ..., ...}``
        - Timeout:              ``{"timeout": True, "timeout_sec": N}``
        - Parse error:          ``{"parse_error": True, "raw": "..."}``
        """
        if not self.is_enabled():
            return {"disabled": True}

        if not self.is_healthy():
            return {"service_unavailable": True}

        # CLIP pre-screen: skip frames that are unlikely to contain vehicles.
        if self._clip_prescreen_fn is not None:
            try:
                if not self._clip_prescreen_fn(image):
                    return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
            except Exception:
                logger.debug(
                    "CLIP prescreen raised an exception; proceeding without filter", exc_info=True
                )
        elif settings.QWEN_CLIP_THRESHOLD > 0:
            tagger = self._lazy_tagger()
            if tagger is not None:
                try:
                    result = tagger.describe_image(image, top_k=1)
                    # describe_image returns {"labels": [{"label": ..., "score": ...}]}
                    labels = result.get("labels", []) if isinstance(result, dict) else []
                    if labels:
                        top_label = labels[0]["label"]
                        top_score = labels[0]["score"]
                        if (
                            top_label.lower() not in _VEHICLE_LABELS
                            or top_score < settings.QWEN_CLIP_THRESHOLD
                        ):
                            return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                    else:
                        return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                except Exception:
                    logger.debug(
                        "Tagger prescreen failed; proceeding without filter", exc_info=True
                    )

        # Build the user message content, including optional audio/OCR context.
        user_content = _build_user_content(image, subtitle_text, ocr_text)

        # Call the Gemma sidecar via OpenAI-compatible API (falls back to Qwen settings
        # for deployments that have not yet migrated to GEMMA_API_URL).
        _api_url = settings.GEMMA_API_URL or settings.QWEN_API_URL
        _api_model = settings.GEMMA_API_MODEL if settings.GEMMA_API_URL else settings.QWEN_MODEL
        try:
            response = self._client_for(_api_url).chat.completions.create(
                model=_api_model,
                messages=[
                    {"role": "system", "content": _QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                **_request_kwargs_for_backend(_api_url),
            )

            raw_text = response.choices[0].message.content or ""
            parsed = _parse_qwen_response(raw_text)
            if not parsed.get("parse_error"):
                return parsed

            retry_response = self._client_for(_api_url).chat.completions.create(
                model=_api_model,
                messages=[
                    {
                        "role": "system",
                        "content": _QWEN_SYSTEM_PROMPT + " Return exactly one JSON object.",
                    },
                    {"role": "user", "content": user_content},
                ],
                **_request_kwargs_for_backend(_api_url),
            )
            retry_raw = retry_response.choices[0].message.content or ""
            retry_parsed = _parse_qwen_response(retry_raw)
            if retry_parsed.get("parse_error"):
                retry_parsed["raw"] = retry_raw[:500]
            return retry_parsed

        except Exception as exc:
            # Check for timeout specifically (openai >= 1.0 raises APITimeoutError)
            exc_type = type(exc).__name__
            if exc_type == "APITimeoutError" or "timeout" in exc_type.lower():
                logger.warning("Gemma extraction timeout after %ds", _EFFECTIVE_QWEN_TIMEOUT_SEC)
                return {"timeout": True, "timeout_sec": _EFFECTIVE_QWEN_TIMEOUT_SEC}

            # Try to import and check the proper class if available
            try:
                from openai import APITimeoutError as _APITimeoutError

                if isinstance(exc, _APITimeoutError):
                    logger.warning(
                        "Gemma extraction timeout after %ds", _EFFECTIVE_QWEN_TIMEOUT_SEC
                    )
                    return {"timeout": True, "timeout_sec": _EFFECTIVE_QWEN_TIMEOUT_SEC}
            except ImportError:
                pass

            logger.warning("Gemma extraction failed: %s", exc, exc_info=True)
            return {"service_unavailable": True}

    def extract_batch(
        self,
        images: list[Image.Image],
        subtitle_texts: list[str | None] | None = None,
        ocr_texts: list[str | None] | None = None,
        extra_contexts: list[str | None] | None = None,
        domain_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract frame facts for a list of images with optional per-image context.

        ``subtitle_texts``, ``ocr_texts``, and ``extra_contexts`` must be the
        same length as ``images`` when provided.  Calls ``extract_frame_facts``
        sequentially (the sidecar is typically single-GPU and does not benefit
        from concurrent requests).

        Parameters
        ----------
        extra_contexts:
            Per-frame prior knowledge strings (from VideoKnowledge.context_for_frame).
            When provided, injected into each Qwen prompt as
            ``[Prior observations about this scene]``.
        domain_hint:
            Optional scene domain string (e.g. "military convoy") prepended to
            the system prompt to steer structured extraction.
        """
        n = len(images)
        sub = subtitle_texts if subtitle_texts and len(subtitle_texts) == n else [None] * n
        ocr = ocr_texts if ocr_texts and len(ocr_texts) == n else [None] * n
        ctx = extra_contexts if extra_contexts and len(extra_contexts) == n else [None] * n
        jobs = list(zip(images, sub, ocr, ctx))
        if not jobs:
            return []

        concurrency = max(1, int(getattr(settings, "QWEN_SIDECAR_CONCURRENCY", 1) or 1))
        if concurrency == 1 or len(jobs) == 1:
            return [
                self._extract_frame_facts_with_context(img, s, o, c, domain_hint)
                for img, s, o, c in jobs
            ]

        max_workers = min(concurrency, len(jobs))
        results: list[dict[str, Any] | None] = [None] * len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._extract_frame_facts_with_context, img, s, o, c, domain_hint): idx
                for idx, (img, s, o, c) in enumerate(jobs)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.warning("Gemma batch extraction failed for item %d", idx, exc_info=True)
                    results[idx] = {"service_unavailable": True}
        return [r if r is not None else {"service_unavailable": True} for r in results]

    def _extract_frame_facts_with_context(
        self,
        image: Image.Image,
        subtitle_text: str | None = None,
        ocr_text: str | None = None,
        extra_context: str | None = None,
        domain_hint: str | None = None,
    ) -> dict[str, Any]:
        """Internal: extract_frame_facts extended with extra_context and domain_hint."""
        if not self.is_enabled():
            return {"disabled": True}

        if not self.is_healthy():
            return {"service_unavailable": True}

        # CLIP pre-screen: skip frames that are unlikely to contain vehicles.
        if self._clip_prescreen_fn is not None:
            try:
                if not self._clip_prescreen_fn(image):
                    return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
            except Exception:
                logger.debug(
                    "CLIP prescreen raised an exception; proceeding without filter", exc_info=True
                )
        elif settings.QWEN_CLIP_THRESHOLD > 0:
            tagger = self._lazy_tagger()
            if tagger is not None:
                try:
                    result = tagger.describe_image(image, top_k=1)
                    labels = result.get("labels", []) if isinstance(result, dict) else []
                    if labels:
                        top_label = labels[0]["label"]
                        top_score = labels[0]["score"]
                        if (
                            top_label.lower() not in _VEHICLE_LABELS
                            or top_score < settings.QWEN_CLIP_THRESHOLD
                        ):
                            return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                    else:
                        return {"clip_filtered": True, "reason": "below_vehicle_threshold"}
                except Exception:
                    logger.debug(
                        "Tagger prescreen failed; proceeding without filter", exc_info=True
                    )

        user_content = _build_user_content(image, subtitle_text, ocr_text, extra_context)

        # Build system prompt — optionally prefixed with domain hint.
        system_prompt = _QWEN_SYSTEM_PROMPT
        if domain_hint and domain_hint.strip():
            system_prompt = f"[Scene domain: {domain_hint.strip()}]\n" + system_prompt

        _api_url = settings.GEMMA_API_URL or settings.QWEN_API_URL
        _api_model = settings.GEMMA_API_MODEL if settings.GEMMA_API_URL else settings.QWEN_MODEL
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key="EMPTY",
                base_url=_api_url,
                timeout=_EFFECTIVE_QWEN_TIMEOUT_SEC,
                max_retries=0,
            )

            response = client.chat.completions.create(
                model=_api_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                **_request_kwargs_for_backend(_api_url),
            )

            raw_text = response.choices[0].message.content or ""
            parsed = _parse_qwen_response(raw_text)
            if not parsed.get("parse_error"):
                return parsed

            retry_response = client.chat.completions.create(
                model=_api_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt + " Return exactly one JSON object.",
                    },
                    {"role": "user", "content": user_content},
                ],
                **_request_kwargs_for_backend(_api_url),
            )
            retry_raw = retry_response.choices[0].message.content or ""
            retry_parsed = _parse_qwen_response(retry_raw)
            if retry_parsed.get("parse_error"):
                retry_parsed["raw"] = retry_raw[:500]
            return retry_parsed

        except Exception as exc:
            exc_type = type(exc).__name__
            if exc_type == "APITimeoutError" or "timeout" in exc_type.lower():
                logger.warning("Gemma extraction timeout after %ds", _EFFECTIVE_QWEN_TIMEOUT_SEC)
                return {"timeout": True, "timeout_sec": _EFFECTIVE_QWEN_TIMEOUT_SEC}

            try:
                from openai import APITimeoutError as _APITimeoutError

                if isinstance(exc, _APITimeoutError):
                    logger.warning(
                        "Gemma extraction timeout after %ds", _EFFECTIVE_QWEN_TIMEOUT_SEC
                    )
                    return {"timeout": True, "timeout_sec": _EFFECTIVE_QWEN_TIMEOUT_SEC}
            except ImportError:
                pass

            logger.warning("Gemma extraction failed: %s", exc, exc_info=True)
            return {"service_unavailable": True}

    # -- Private helpers -------------------------------------------------------

    def _check_health(self) -> None:
        """Check sidecar health and cache the result in self._healthy.

        Prefers GEMMA_API_URL / GEMMA_API_BACKEND; falls back to legacy
        QWEN_API_URL / QWEN_BACKEND for deployments that have not migrated.
        """
        _api_url = settings.GEMMA_API_URL or settings.QWEN_API_URL
        backend = (
            settings.GEMMA_API_BACKEND if settings.GEMMA_API_URL else settings.QWEN_BACKEND
        ).lower()
        timeout = min(
            settings.GEMMA_API_TIMEOUT_SEC if settings.GEMMA_API_URL else settings.QWEN_TIMEOUT_SEC,
            10,
        )

        # Auto-detect ollama from the default port (11434) when backend is not
        # explicitly set to "ollama" — the vllm health endpoint (/health) returns
        # 404 on ollama servers, which only expose /api/tags.
        if backend != "ollama" and ":11434" in _api_url:
            backend = "ollama"
        if backend == "ollama":
            self._healthy = _health_check_ollama(_api_url, timeout)
        else:
            # default: vllm
            self._healthy = _health_check_vllm(_api_url, timeout)
            # Fallback: if vllm check fails, try ollama (covers non-standard ports)
            if not self._healthy:
                self._healthy = _health_check_ollama(_api_url, timeout)
                if self._healthy:
                    backend = "ollama"

        sidecar = "Gemma" if settings.GEMMA_API_URL else "Qwen"
        if self._healthy:
            logger.info("%s sidecar healthy (backend=%s url=%s)", sidecar, backend, _api_url)
        else:
            logger.warning(
                "%s sidecar unreachable (backend=%s url=%s); Phase 2 will be skipped",
                sidecar,
                backend,
                _api_url,
            )

    def _client_for(self, api_url: str):
        """Reuse one OpenAI client to keep HTTP connections warm across frames."""
        if self._client is not None:
            return self._client
        from openai import OpenAI

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=api_url,
            timeout=_EFFECTIVE_QWEN_TIMEOUT_SEC,
            max_retries=0,
        )
        return self._client

    def _lazy_tagger(self):
        """Lazily create and cache an OpenCLIPTagger for vehicle pre-screening."""
        if self._tagger is not None:
            return self._tagger
        try:
            from selfsuvis.pipeline.vision.factory import OpenCLIPTagger

            self._tagger = OpenCLIPTagger()
            return self._tagger
        except Exception:
            logger.debug("Could not load OpenCLIPTagger for Qwen pre-screen", exc_info=True)
            return None

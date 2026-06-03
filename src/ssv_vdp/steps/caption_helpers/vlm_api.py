"""Florence-2 and Qwen VLM API captioning backends."""

import time
from pathlib import Path
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger("pipeline.local.caption")


def caption_via_florence_api(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> dict[str, Any]:
    """Caption frames via a vLLM endpoint serving Florence-2-large.

    vLLM serves Florence-2 with ``--task generate --trust-remote-code``.
    The ``<MORE_DETAILED_CAPTION>`` task token is passed as a text message
    alongside the base64-encoded image; the response is the plain caption string.

    This path consumes zero local VRAM — all inference runs inside the vLLM
    process, which can be on a separate GPU or port from Ollama.
    """
    import base64
    import io

    from ..report import write_scene_captions_md

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Florence API")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 via vLLM API (url=%s  model=%s  frames=%d)",
        api_url,
        model,
        len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: list[dict[str, Any]] = []
    t0 = time.time()

    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
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
                                    f"[Context: {domain_hint}] <MORE_DETAILED_CAPTION>"
                                    if domain_hint
                                    else "<MORE_DETAILED_CAPTION>"
                                ),
                            },
                        ],
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.0,
            }
            resp = httpx.post(endpoint, json=payload, timeout=60.0)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Florence-2 sometimes echoes the task token; strip it
            if raw.startswith("<MORE_DETAILED_CAPTION>"):
                raw = raw[len("<MORE_DETAILED_CAPTION>"):].strip()
            caption = raw
        except Exception as exc:
            _log.debug("  Florence API error for %s: %s", Path(fp).name, exc)

        caption_results.append(
            {
                "frame_path": fp,
                "t_sec": t_sec,
                "caption": caption,
                "caption_confidence": 0.75 if caption else 0.0,
            }
        )
        if (idx + 1) % 20 == 0:
            _log.info("    ... %d/%d frames captioned via Florence API", idx + 1, len(frame_list))

    elapsed = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info(
        "  [ok] Florence API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed
    )
    write_scene_captions_md(video_dir / "scene_captions.md", video_name, caption_results, elapsed)
    return {
        "skipped": False,
        "captions": caption_results,
        "captioned_count": captioned,
        "elapsed_sec": elapsed,
        "backend": "florence_api",
    }


def caption_via_qwen_api(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> dict[str, Any]:
    """Caption frames via an OpenAI-compatible VLM endpoint (Ollama / vLLM).

    Used as a fallback when Florence-2 cannot load due to OOM.  Sends one
    ``/chat/completions`` request per frame with the image embedded as a base64
    data-URI.  Images are downscaled to 512 px on the longest side before
    encoding to keep latency reasonable.
    """
    import base64
    import io

    from ..report import write_scene_captions_md

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Qwen API for captioning")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 unavailable locally — falling back to Qwen API captioning "
        "(url=%s  model=%s  frames=%d)",
        api_url,
        model,
        len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: list[dict[str, Any]] = []
    t0 = time.time()
    _MAX_CONSECUTIVE_FAILURES = 3

    # Pre-flight: verify the endpoint is responsive before iterating all frames.
    try:
        probe = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15.0,
        )
        if probe.status_code >= 500:
            _log.warning(
                "  Qwen API pre-flight failed (HTTP %d) — skipping captioning",
                probe.status_code,
            )
            return {
                "skipped": True,
                "reason": f"Qwen API returned {probe.status_code}",
                "captions": [],
            }
    except Exception as exc:
        _log.warning("  Qwen API pre-flight error (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    consecutive_failures = 0
    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
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
                                    (f"[Context: {domain_hint}]\n" if domain_hint else "")
                                    + "Describe this image in one or two sentences. "
                                    "Focus on the scene type, visible objects, and environment."
                                ),
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.1,
            }
            resp = httpx.post(endpoint, json=payload, timeout=30.0)
            resp.raise_for_status()
            caption = resp.json()["choices"][0]["message"]["content"].strip()
            consecutive_failures = 0
        except Exception as exc:
            _log.debug("  Qwen caption error for %s: %s", Path(fp).name, exc)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                _log.warning(
                    "  Qwen API: %d consecutive failures — aborting captioning early "
                    "(%d/%d frames done)",
                    consecutive_failures,
                    idx + 1,
                    len(frame_list),
                )
                for fp2, t2 in frame_list[idx + 1:]:
                    caption_results.append(
                        {"frame_path": fp2, "t_sec": t2, "caption": "", "caption_confidence": 0.0}
                    )
                break

        caption_results.append(
            {
                "frame_path": fp,
                "t_sec": t_sec,
                "caption": caption,
                "caption_confidence": 0.7 if caption else 0.0,
            }
        )
        if (idx + 1) % 50 == 0:
            _log.info("    ... %d/%d frames captioned via Qwen API", idx + 1, len(frame_list))

    elapsed = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info(
        "  [ok] Qwen API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed
    )
    write_scene_captions_md(video_dir / "scene_captions.md", video_name, caption_results, elapsed)
    return {
        "skipped": False,
        "captions": caption_results,
        "captioned_count": captioned,
        "elapsed_sec": elapsed,
        "backend": "qwen_api",
    }

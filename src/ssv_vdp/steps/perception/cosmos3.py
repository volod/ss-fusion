"""Step 15: Cosmos3 omnimodal world-model inference.

Runs nvidia/Cosmos3-Nano (or a vLLM-Omni sidecar) on sampled video frames to
produce a temporal scene understanding narrative — physical state, entity
dynamics, and environment classification — that downstream synthesis steps
can consume.

Hardware selection (local mode):
    free_vram < 18 GB   → skip  (model weights alone need ~32 GB BF16)
    18 GB <= vram < 40   → Cosmos3-Nano with layerwise CPU offloading
    vram >= 40 GB        → Cosmos3-Nano without offloading (faster)
    COSMOS3_MODEL=auto   → auto (above rules); set to a full model ID to override

Sidecar mode (preferred on single-GPU setups):
    Set COSMOS3_API_URL to a running ``vllm serve nvidia/Cosmos3-Nano --omni``
    endpoint.  The step sends frames as base64 image_url messages via
    OpenAI-compatible /v1/chat/completions.  COSMOS3_MODEL selects the model
    served by the sidecar (default nvidia/Cosmos3-Nano).
"""

import base64
import json
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..common import _open_frame_image

_log = get_logger("pipeline.local.cosmos3")

_COSMOS3_NANO_ID = "nvidia/Cosmos3-Nano"
_COSMOS3_SUPER_ID = "nvidia/Cosmos3-Super"

# Minimum free VRAM (GiB) thresholds for local inference
_VRAM_OFFLOAD_MIN_GB = 18.0
_VRAM_FULL_MIN_GB = 40.0

# Sampling caps
_DEFAULT_MAX_FRAMES_LOCAL = 8
_DEFAULT_MAX_FRAMES_SIDECAR = 16

_ANALYSIS_PROMPT = (
    "You are a physical-AI world model analyzing a video sequence. "
    "For the provided frames describe concisely:\n"
    "1. Scene environment and type (indoor/outdoor, terrain, lighting)\n"
    "2. Key entities, their positions and physical states\n"
    "3. Observable motion and temporal dynamics\n"
    "4. Safety-relevant observations or anomalies\n"
    "Return a structured JSON with keys: "
    "scene_type, entities, dynamics, safety_notes, confidence (0-1)."
)


# ---------------------------------------------------------------------------
# Hardware-aware model selection
# ---------------------------------------------------------------------------

def _select_local_model(
    free_vram_gb: float,
    model_override: str,
) -> tuple[str, bool] | None:
    """Return (model_id, use_layerwise_offload) or None if hardware is insufficient."""
    if model_override and model_override != "auto":
        use_offload = free_vram_gb < _VRAM_FULL_MIN_GB
        return model_override, use_offload

    if free_vram_gb < _VRAM_OFFLOAD_MIN_GB:
        return None  # not enough VRAM even with offloading
    use_offload = free_vram_gb < _VRAM_FULL_MIN_GB
    return _COSMOS3_NANO_ID, use_offload


def _sample_frames(
    frame_list: list[tuple[str, float]], max_frames: int
) -> list[tuple[str, float]]:
    if len(frame_list) <= max_frames:
        return frame_list
    step = len(frame_list) / max_frames
    return [frame_list[int(i * step)] for i in range(max_frames)]


# ---------------------------------------------------------------------------
# Sidecar path  (vLLM-Omni / OpenAI-compatible)
# ---------------------------------------------------------------------------

def _run_sidecar(
    api_url: str,
    model_id: str,
    sampled_frames: list[tuple[str, float]],
    timeout: int = 120,
) -> dict[str, Any]:
    """Call vLLM-Omni endpoint with base64-encoded frames."""
    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return {"skipped": True, "reason": "urllib unavailable"}

    images_content: list[dict[str, Any]] = []
    for fp, t_sec in sampled_frames:
        img = _open_frame_image(fp)
        if img is None:
            continue
        from io import BytesIO

        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        images_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    if not images_content:
        return {"skipped": True, "reason": "no readable frames"}

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": _ANALYSIS_PROMPT}] + images_content,
        }
    ]
    payload = json.dumps(
        {"model": model_id, "messages": messages, "max_tokens": 1024, "temperature": 0.1}
    ).encode()

    endpoint = api_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        return {"skipped": True, "reason": f"sidecar unavailable: {exc}"}
    except Exception as exc:
        return {"skipped": True, "reason": f"sidecar error: {exc}"}

    raw_text = (
        body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    ).strip()
    return {"raw_text": raw_text, "model_id": model_id, "via": "sidecar"}


# ---------------------------------------------------------------------------
# Local diffusers path
# ---------------------------------------------------------------------------

def _run_local(
    model_id: str,
    use_offload: bool,
    sampled_frames: list[tuple[str, float]],
    device: str,
) -> dict[str, Any]:
    """Load Cosmos3 via diffusers and run inference on sampled frames."""
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        return {"skipped": True, "reason": f"diffusers/torch unavailable: {exc}"}

    try:
        _log.info(
            "  Loading %s (offload=%s) …", model_id, use_offload
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if use_offload:
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device if device == "cuda" else "cpu"

        pipe = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)

        if use_offload:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
        elif device == "cuda":
            pipe = pipe.to(device)
    except Exception as exc:
        return {"skipped": True, "reason": f"model load failed: {exc}"}

    images_pil = []
    for fp, _ in sampled_frames:
        img = _open_frame_image(fp)
        if img is not None:
            images_pil.append(img.convert("RGB"))

    if not images_pil:
        return {"skipped": True, "reason": "no readable frames"}

    try:
        out = pipe(
            prompt=_ANALYSIS_PROMPT,
            images=images_pil if len(images_pil) > 1 else images_pil[0],
            max_new_tokens=1024,
        )
        raw_text = out.get("text", "") if isinstance(out, dict) else str(out)
    except Exception as exc:
        _log.warning("  Cosmos3 inference error: %s", exc)
        raw_text = ""
    finally:
        try:
            del pipe
            if device == "cuda":
                import torch as _t
                _t.cuda.empty_cache()
        except Exception:
            pass

    return {"raw_text": raw_text, "model_id": model_id, "via": "local"}


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Best-effort extraction of the JSON object from a model response."""
    if not raw_text:
        return {}
    for start, end in [(raw_text.find("{"), raw_text.rfind("}")),]:
        if start != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw": raw_text}


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------

def step_cosmos3_inference(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Step 15: Cosmos3 omnimodal world-model inference on sampled video frames.

    Writes to video_dir:
        cosmos3_inference.json   — per-clip structured scene understanding
    """
    result: dict[str, Any] = {"skipped": True, "clips": [], "n_clips": 0}

    cosmos3_enabled = str(getattr(settings, "COSMOS3_ENABLED", "false")).lower()
    if cosmos3_enabled not in {"true", "1", "yes", "on"}:
        _log.info("  Cosmos3 disabled (COSMOS3_ENABLED=false) — skipping")
        return result

    api_url: str = str(getattr(settings, "COSMOS3_API_URL", "") or "")
    model_override: str = str(getattr(settings, "COSMOS3_MODEL", "auto") or "auto")

    # --- determine mode and capability check --------------------------------
    if api_url:
        mode = "sidecar"
        model_id = model_override if model_override != "auto" else _COSMOS3_NANO_ID
        max_frames = int(getattr(settings, "COSMOS3_MAX_FRAMES", _DEFAULT_MAX_FRAMES_SIDECAR) or _DEFAULT_MAX_FRAMES_SIDECAR)
        _log.info(
            "  Cosmos3 sidecar mode: endpoint=%s model=%s", api_url, model_id
        )
    else:
        mode = "local"
        try:
            from selfsuvis.pipeline.vision.registry import detect_resources
            resources = detect_resources()
            free_vram_gb = resources.get("free_vram_gb", resources.get("vram_gb", 0.0))
        except Exception:
            free_vram_gb = 0.0

        selection = _select_local_model(free_vram_gb, model_override)
        if selection is None:
            _log.info(
                "  Cosmos3 skipped — free VRAM %.1f GB < %.0f GB minimum "
                "(set COSMOS3_API_URL to use a vLLM-Omni sidecar instead)",
                free_vram_gb,
                _VRAM_OFFLOAD_MIN_GB,
            )
            return result
        model_id, use_offload = selection
        max_frames = int(getattr(settings, "COSMOS3_MAX_FRAMES", _DEFAULT_MAX_FRAMES_LOCAL) or _DEFAULT_MAX_FRAMES_LOCAL)
        _log.info(
            "  Cosmos3 local mode: model=%s offload=%s free_vram=%.1fGB",
            model_id,
            use_offload,
            free_vram_gb,
        )

    # --- sample frames and split into clips ---------------------------------
    max_clips = int(getattr(settings, "COSMOS3_MAX_CLIPS", 4) or 4)
    clip_size = max(1, len(frame_list) // max_clips)
    clips: list[list[tuple[str, float]]] = []
    for start in range(0, len(frame_list), clip_size):
        clip = frame_list[start : start + clip_size]
        clips.append(_sample_frames(clip, max_frames))
    clips = clips[:max_clips]

    _log.info(
        "  Running Cosmos3 on %d clip(s) × up to %d frames …",
        len(clips),
        max_frames,
    )
    t0 = time.time()
    clip_results: list[dict[str, Any]] = []

    for clip_idx, clip_frames in enumerate(clips):
        mid_fp, mid_t = clip_frames[len(clip_frames) // 2]
        if mode == "sidecar":
            raw = _run_sidecar(api_url, model_id, clip_frames)
        else:
            raw = _run_local(model_id, use_offload, clip_frames, device)  # type: ignore[arg-type]

        if raw.get("skipped"):
            _log.warning(
                "  Cosmos3 clip %d skipped: %s", clip_idx, raw.get("reason", "unknown")
            )
            clip_results.append(
                {"clip_idx": clip_idx, "t_sec": mid_t, "skipped": True, "reason": raw.get("reason")}
            )
            continue

        parsed = _extract_json_payload(raw.get("raw_text", ""))
        clip_results.append(
            {
                "clip_idx": clip_idx,
                "t_sec": mid_t,
                "frame_path": mid_fp,
                "n_frames": len(clip_frames),
                "model_id": raw.get("model_id", model_id),
                "via": raw.get("via", mode),
                "scene_type": parsed.get("scene_type", ""),
                "entities": parsed.get("entities", []),
                "dynamics": parsed.get("dynamics", ""),
                "safety_notes": parsed.get("safety_notes", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "raw_text": raw.get("raw_text", ""),
            }
        )

    elapsed = time.time() - t0
    ok_clips = [c for c in clip_results if not c.get("skipped")]

    if not ok_clips:
        _log.warning("  Cosmos3: all clips skipped — check hardware / sidecar")
        return result

    # --- persist artifact ---------------------------------------------------
    artifact_path = video_dir / "cosmos3_inference.json"
    artifact_path.write_text(
        json.dumps(
            {
                "video_name": video_name,
                "model_id": model_id,
                "mode": mode,
                "n_clips": len(ok_clips),
                "elapsed_sec": elapsed,
                "clips": clip_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _log.info(
        "  [ok] Cosmos3: %d/%d clips processed in %.1fs → %s",
        len(ok_clips),
        len(clips),
        elapsed,
        artifact_path.name,
    )

    # Derive a short scene summary from the most-confident clip
    best = max(ok_clips, key=lambda c: c.get("confidence", 0.0))
    result.update(
        {
            "skipped": False,
            "clips": clip_results,
            "n_clips": len(ok_clips),
            "elapsed_sec": elapsed,
            "model_id": model_id,
            "mode": mode,
            "scene_type": best.get("scene_type", ""),
            "top_entities": best.get("entities", []),
            "top_dynamics": best.get("dynamics", ""),
            "safety_notes": best.get("safety_notes", ""),
        }
    )
    return result

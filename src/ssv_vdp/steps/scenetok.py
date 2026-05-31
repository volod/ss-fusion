"""Step 14: SceneTok streaming scene encoder + segmentation decoder."""

import base64
import time
from pathlib import Path
from typing import Any

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from .common import _open_frame_image

_log = get_logger("pipeline.local.scenetok")

_DEFAULT_MAX_FRAMES = 32


def step_scenetok(
    frame_list: list[tuple[str, float]],
    video_dir: Path,
    *,
    checkpoint: str = "",
    mode: str = "",
) -> dict[str, Any]:
    """Step 14: encode frames into SceneTok scene tokens, decode to masks or views.

    Writes to video_dir:
      scenetok_tokens.npz   — compressed permutation-invariant latent tokens
      scenetok_masks/       — per-frame PNG segmentation masks  (mode=masks)
      scenetok_views/       — per-frame JPEG novel-view renders (mode=rgb)
    """
    result: dict[str, Any] = {"skipped": True, "n_tokens": 0, "n_frames": 0}

    try:
        from selfsuvis.pipeline.vision.scenetok import SceneTokModel
    except ImportError as exc:
        _log.warning("SceneTok client unavailable (%s) — skipping", exc)
        return result

    client = SceneTokModel()
    if not client.is_enabled():
        _log.info(
            "  SceneTok disabled — no sidecar URL and local model not available. "
            "Set SCENETOK_API_URL or install the scenetok package with a ~24 GB GPU."
        )
        return result

    effective_mode = mode or str(getattr(settings, "SCENETOK_MODE", "masks") or "masks")
    effective_checkpoint = checkpoint or str(
        getattr(settings, "SCENETOK_CHECKPOINT", "va-videodc_re10k") or "va-videodc_re10k"
    )
    max_frames = int(
        getattr(settings, "SCENETOK_MAX_FRAMES", _DEFAULT_MAX_FRAMES) or _DEFAULT_MAX_FRAMES
    )
    sample_step = max(1, len(frame_list) // max_frames)
    sampled = frame_list[::sample_step][:max_frames]

    _log.info(
        "Running SceneTok encoder+decoder on %d sampled frames (checkpoint=%s mode=%s) …",
        len(sampled),
        effective_checkpoint,
        effective_mode,
    )
    t0 = time.time()

    images: list[Image.Image] = []
    valid_frames: list[tuple[str, float]] = []
    for fp, t_sec in sampled:
        img = _open_frame_image(fp)
        if img is not None:
            images.append(img)
            valid_frames.append((fp, t_sec))

    if not images:
        _log.warning("  SceneTok: no readable frames in sample — skipping")
        return result

    out = client.encode_decode(valid_frames, images, mode=effective_mode)
    elapsed = time.time() - t0

    if out.get("service_unavailable"):
        _log.warning("  SceneTok unavailable: %s", out.get("reason", "unknown"))
        client.release()
        return result

    # -- save tokens ----------------------------------------------------------
    tokens_b64 = out.get("tokens_b64_npz", "")
    if tokens_b64:
        tokens_path = video_dir / "scenetok_tokens.npz"
        tokens_path.write_bytes(base64.b64decode(tokens_b64))
        _log.info("  Saved scene tokens → %s", tokens_path.name)

    # -- save per-frame outputs ------------------------------------------------
    frame_results: list[dict[str, Any]] = out.get("results", [])
    if effective_mode == "masks":
        out_dir = video_dir / "scenetok_masks"
    else:
        out_dir = video_dir / "scenetok_views"
    out_dir.mkdir(exist_ok=True)

    saved = 0
    for item in frame_results:
        t_sec = item.get("t_sec", 0.0)
        b64_png = item.get("b64_png", "")
        if not b64_png:
            continue
        fname = f"{t_sec:.3f}.png"
        (out_dir / fname).write_bytes(base64.b64decode(b64_png))
        saved += 1

    n_tokens = int(out.get("n_tokens", 0))
    _log.info(
        "  [ok] SceneTok: %d frames → %d tokens → %d %s in %.1fs",
        len(images),
        n_tokens,
        saved,
        "masks" if effective_mode == "masks" else "views",
        elapsed,
    )
    client.release()

    result.update(
        {
            "skipped": False,
            "n_tokens": n_tokens,
            "n_frames": saved,
            "elapsed_sec": elapsed,
            "mode": effective_mode,
            "checkpoint": effective_checkpoint,
            "tokens_path": str(video_dir / "scenetok_tokens.npz") if tokens_b64 else "",
            "output_dir": str(out_dir),
        }
    )
    return result

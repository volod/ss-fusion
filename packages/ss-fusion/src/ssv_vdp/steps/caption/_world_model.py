"""Step 11 — World model video embeddings and RSSM temporal surprise scoring."""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..caption_helpers.vram import _log_vram_snapshot
from ..common import _open_frame_batch, write_json_artifact

_log = get_logger("pipeline.local.caption")


def step_world_model_pass(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 11: world model video embeddings + RSSM temporal surprise scoring.

    Two sub-steps run sequentially:

    Q-A  VideoMAE/VideoWorld clip embeddings (requires WORLD_MODEL_ENABLED=true
         and the VideoMAE model to be loaded).

    Q-B  RSSM temporal surprise (requires DREAMER_ENABLED=true and CLIP model
         in *models*).  Encodes the per-frame CLIP sequence with a lightweight
         GRU-based RSSM (DreamerV3-inspired) and writes per-frame surprise
         scores to ``rssm_temporal.json`` in *video_dir*.  These scores are
         later consumed by the active-learning tagging step.
    """
    result: dict[str, Any] = {"skipped": True, "world_results": []}

    # -- Q-A: VideoMAE world model clip embeddings -----------------------------
    try:
        from selfsuvis.pipeline.vision.world import WorldModel
    except ImportError as exc:
        _log.warning("  World model unavailable (%s) — skipping Q-A", exc)
    else:
        wm = WorldModel()
        _log_vram_snapshot("before world model use")
        if not wm.is_enabled():
            _log.info("  World model disabled (WORLD_MODEL_ENABLED=false) — skipping Q-A")
        else:
            clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
            _log.info(
                "Running world model on %d frames in clips of %d (model=%s) …",
                len(frame_list),
                clip_frames,
                wm.model_id,
            )
            t0 = time.time()
            world_results: list[dict[str, Any]] = []
            for clip_start in range(0, len(frame_list), clip_frames):
                clip = frame_list[clip_start : clip_start + clip_frames]
                imgs = _open_frame_batch(clip)
                try:
                    clip_out = wm.process_clip(imgs)
                except Exception as exc:
                    _log.warning("  World model clip %d failed: %s", clip_start, exc)
                    clip_out = {"world_model_error": True}
                mid = clip_start + len(clip) // 2
                fp, t_sec = frame_list[mid]
                world_results.append({"frame_path": fp, "t_sec": t_sec, **clip_out})
            elapsed = time.time() - t0
            ok = sum(
                1
                for r in world_results
                if not r.get("world_model_error")
                and not r.get("world_model_unavailable")
                and not r.get("world_model_disabled")
            )
            _log.info("  [ok] World model: %d clips processed in %.1fs", ok, elapsed)
            result.update(
                {
                    "skipped": False,
                    "world_results": world_results,
                    "ok_count": ok,
                    "elapsed_sec": elapsed,
                }
            )
            wm.release()
            _log_vram_snapshot("after world model use")

    # -- Q-B: RSSM temporal surprise -------------------------------------------
    if models is not None and getattr(settings, "DREAMER_ENABLED", False):
        clip_model = models.get("clip")
        if clip_model is not None:
            try:
                import numpy as np
                from PIL import Image as _PILImage

                from selfsuvis.models.rssm_model import RSSMEmbedder  # type: ignore[import]

                _log.info("  RSSM: embedding %d frames for temporal surprise …", len(frame_list))
                t_rssm = time.time()

                clip_embeds: list = []
                for fp, _t in frame_list:
                    try:
                        img = _PILImage.open(fp).convert("RGB")
                        emb = clip_model.encode_images([img])[0]
                        clip_embeds.append(emb.astype(np.float32))
                    except Exception as _exc:
                        _log.debug("  RSSM: skipping frame %s (%s)", fp, _exc)

                if len(clip_embeds) >= 2:
                    rssm = RSSMEmbedder(
                        hidden_dim=getattr(settings, "DREAMER_HIDDEN_DIM", 256),
                        latent_dim=getattr(settings, "DREAMER_LATENT_DIM", 32),
                        train_steps=getattr(settings, "DREAMER_TRAIN_STEPS", 20),
                    )
                    all_embeds = np.stack(clip_embeds)
                    rssm_out = rssm.encode_sequence(all_embeds)
                    surprise_scores = rssm_out["surprise_scores"].tolist()
                    method = rssm_out.get("method", "rssm")

                    n_frames = len(frame_list)
                    n_valid = len(clip_embeds)
                    dense: list[float] = [0.5] * n_frames
                    valid_idx = 0
                    for i, (fp, _t) in enumerate(frame_list):
                        if valid_idx < n_valid:
                            dense[i] = float(
                                surprise_scores[min(valid_idx, len(surprise_scores) - 1)]
                            )
                            valid_idx += 1

                    rssm_json: dict[str, Any] = {
                        "method": method,
                        "hidden_dim": rssm_out.get("hidden_dim", 256),
                        "latent_dim": rssm_out.get("latent_dim", 32),
                        "n_frames": n_frames,
                        "n_embedded": n_valid,
                        "surprise_scores": dense,
                        "frames": [
                            {"frame_path": fp, "t_sec": t, "surprise": dense[i]}
                            for i, (fp, t) in enumerate(frame_list)
                        ],
                    }
                    rssm_path = video_dir / "rssm_temporal.json"
                    write_json_artifact(rssm_path, rssm_json)
                    elapsed_rssm = time.time() - t_rssm
                    _log.info(
                        "  [ok] RSSM: method=%s  mean_surprise=%.3f  elapsed=%.1fs → %s",
                        method,
                        float(np.mean(dense)),
                        elapsed_rssm,
                        rssm_path.name,
                    )
                    result.update(
                        {"rssm_scores": dense, "rssm_method": method, "rssm_path": str(rssm_path)}
                    )
                    result["skipped"] = False
                else:
                    _log.info(
                        "  RSSM: too few embedded frames (%d) — skipping Q-B", len(clip_embeds)
                    )
            except Exception as exc:
                _log.warning("  RSSM temporal surprise failed (%s) — skipping Q-B", exc)
        else:
            _log.info("  RSSM: no CLIP model in models dict — skipping Q-B")

    return result

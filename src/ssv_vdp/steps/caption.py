"""Captioning steps: Gemma, Florence, Qwen, ASR, OCR, depth, detection, world model."""

import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from .caption_helpers.frame_selection import (
    _adaptive_sparse_budget,
    _reduce_llm_sample_frames,
    _select_qwen_frames,
    _select_segment_boundary_pairs,
)
from .caption_helpers.gemma_api import (
    _fallback_tracking_bbox,
    _gemma_analyse_frame_via_api,
    _gemma_diff_two_frames_via_api,
    _summarise_gemma_captions_to_structured_scene,
    step_qwen_captioning_gemma_fallback,
)
from .caption_helpers.ocr import (
    _fallback_ocr_frame_sample,
    _select_ocr_candidate_frames,
)
from .caption_helpers.frame_selection import _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT  # noqa: F401
from .caption_helpers.ollama import (
    _compute_sidecar_timeout,  # noqa: F401 — re-exported for pipeline imports
    _list_ollama_models,  # noqa: F401
    _recommend_gemma_sidecar_models,  # noqa: F401
    _resolve_ollama_gemma_model,  # noqa: F401
    _resolve_ollama_reasoning_model,  # noqa: F401
    _unload_known_sidecars,  # noqa: F401 — re-exported for pipeline imports
    _unload_ollama_model,  # noqa: F401 — re-exported for pipeline imports
)
from .caption_helpers.vram import (
    _flush_cuda_allocator,  # noqa: F401 — re-exported for pipeline imports
    _guard_min_free_vram,  # noqa: F401 — re-exported for pipeline imports
    _log_vram_snapshot,  # noqa: F401 — re-exported for pipeline imports
    _models_on_device,  # noqa: F401 — re-exported for pipeline imports
    _offload_models_to_cpu,  # noqa: F401 — re-exported for pipeline imports
    _prep_vram_for_step,  # noqa: F401 — re-exported for pipeline imports
    _restore_models_to_gpu,  # noqa: F401 — re-exported for pipeline imports
    get_runtime_telemetry,  # noqa: F401 — re-exported for pipeline imports
    reset_runtime_telemetry,  # noqa: F401 — re-exported for pipeline imports
)
from .caption_helpers.vlm_api import caption_via_florence_api, caption_via_qwen_api
from .common import (
    _GEMMA_ANALYSIS_SAMPLE_N,
    _GEMMA_TEXT_PROBES,
    _SCENE_CHANGE_THRESH,
    VideoKnowledge,
    _open_frame_batch,
    _open_frame_image,
    _run_batched_frame_inference,
    write_json_artifact,
    write_markdown_artifact,
)

_log = get_logger("pipeline.local.caption")

try:
    from selfsuvis.models.dino_model import DINOEmbedder

    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

try:
    from selfsuvis.models.gemma_model import GemmaEmbedder

    _HAS_GEMMA = True
except Exception:
    _HAS_GEMMA = False


# ---------------------------------------------------------------------------
# Step 03 — Gemma multimodal video analysis
# ---------------------------------------------------------------------------


def step_gemma_analysis(
    video_path: Path,
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: list[tuple[str, float]],
    models: dict[str, Any],
    gemma_api_url: str = "",
    gemma_api_model: str = "",
) -> dict[str, Any]:
    """Step 03: Gemma open-weight multimodal video analysis.

    Uses the local GemmaEmbedder for embedding-based analysis and, when a
    Gemma Ollama/vLLM sidecar is configured (GEMMA_API_URL), also runs
    generative per-frame scene description.

    Analyses performed:
    1. Generative frame descriptions via Gemma sidecar (if GEMMA_API_URL set)
    2. Scene change detection — consecutive-frame cosine distance
    3. Semantic scene clustering — greedy cosine-based grouping
    4. Zero-shot scene classification — text probe vs frame embedding scores
    5. Cross-modal text -> frame retrieval — nearest-neighbour in Gemma space
    6. Temporal video embedding — mean-pool of all frame embeddings
    7. Gemma vs CLIP embedding comparison — MNN@k and mean pairwise cosine sim
    8. Gemma vs DINOv3 embedding comparison — MNN@k and mean pairwise cosine sim

    Skipped gracefully when GemmaEmbedder is unavailable and no sidecar is set.
    Writes ``gemma_analysis.md`` to *video_dir*.
    """
    import numpy as np

    from .report import _write_gemma_captions_md, write_gemma_analysis_md

    result: dict[str, Any] = {"skipped": True, "reason": ""}

    effective_api_url = gemma_api_url or settings.GEMMA_API_URL
    effective_api_model = gemma_api_model or settings.GEMMA_API_MODEL
    effective_timeout = float(settings.GEMMA_API_TIMEOUT_SEC)

    # Use local GemmaEmbedder when available; otherwise fall back to whatever
    # embedder is loaded (OpenCLIP) — all embedding analyses still run, just
    # powered by a different backbone.
    _clip_model = models.get("clip")
    has_local = _clip_model is not None
    _embedder_name = (
        "GemmaEmbedder"
        if (_HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder))
        else type(_clip_model).__name__
        if _clip_model is not None
        else "none"
    )
    has_sidecar = bool(effective_api_url)

    if not has_local and not has_sidecar:
        result["reason"] = "No embedder available and GEMMA_API_URL not set"
        _log.info("  Gemma analysis skipped: %s", result["reason"])
        return result

    t0 = time.time()

    # Sample frames evenly, then drop near-duplicates for stable scenes.
    n_avail = len(frame_list)
    n_sample = min(
        int(settings.GEMMA_ANALYSIS_MAX_SAMPLE_FRAMES), _GEMMA_ANALYSIS_SAMPLE_N, n_avail
    )
    step = max(1, n_avail // max(1, n_sample))
    sample_frames = frame_list[::step][:n_sample]
    sample_frames = _reduce_llm_sample_frames(sample_frames, max_frames=n_sample)
    sample_images = [_open_frame_image(fp) for fp, _ in sample_frames]
    sample_paths = [fp for fp, _ in sample_frames]
    sample_ts = [t for _, t in sample_frames]
    n = len(sample_images)
    _log.info("  Gemma analysis: %d sampled frames (from %d total)", n, n_avail)

    task_results: dict[str, Any] = {}

    # 1. Generative per-frame analysis via Ollama/vLLM sidecar
    gemma_captions: list[dict[str, Any]] = []
    if has_sidecar:
        _log.info(
            "Generative scene analysis via sidecar (url=%s  model=%s  frames=%d) ...",
            effective_api_url,
            effective_api_model,
            n,
        )
        for idx, (fp, t_sec) in enumerate(sample_frames):
            desc = _gemma_analyse_frame_via_api(
                fp,
                effective_api_url,
                effective_api_model,
                effective_timeout,
                video_dir=video_dir,
            )
            gemma_captions.append({"frame_path": fp, "t_sec": t_sec, "description": desc})
            if (idx + 1) % 10 == 0:
                _log.info("    ... %d/%d frames analysed via Gemma sidecar", idx + 1, n)
        described = sum(1 for c in gemma_captions if c.get("description"))
        _log.info("Generative descriptions: %d/%d frames", described, n)
        task_results["generative_descriptions"] = {
            "description": "Per-frame scene description generated by Gemma sidecar",
            "n_frames": n,
            "described_count": described,
            "model": effective_api_model,
            "captions": gemma_captions,
        }
        _write_gemma_captions_md(
            video_dir / "gemma_captions.md",
            video_name,
            effective_api_model,
            gemma_captions,
        )
        structured_scene = _summarise_gemma_captions_to_structured_scene(
            gemma_captions,
            effective_api_url,
            effective_api_model,
            effective_timeout,
        )
        task_results["structured_scene_summary"] = structured_scene
    else:
        task_results["generative_descriptions"] = {
            "description": "Skipped — GEMMA_API_URL not configured",
            "skipped": True,
        }
        structured_scene = {}

    # For the remaining embedding-based analyses we need a local embedder.
    if not has_local:
        _log.info("Skipping embedding analyses (no embedder loaded)")
        text_query_results: list[dict[str, Any]] = []
        dino_comparison: dict[str, Any] = {"available": False, "reason": "no embedder loaded"}
        clip_comparison: dict[str, Any] = {"available": False, "reason": "GemmaEmbedder not loaded"}
        elapsed = time.time() - t0
        write_gemma_analysis_md(
            video_dir / "gemma_analysis.md",
            video_name,
            effective_api_model or settings.GEMMA_MODEL_ID,
            n,
            task_results,
            dino_comparison,
            text_query_results,
            elapsed,
            clip_comparison=clip_comparison,
        )
        result.update(
            {
                "skipped": False,
                "n_frames": n,
                "task_results": task_results,
                "dino_comparison": dino_comparison,
                "clip_comparison": clip_comparison,
                "elapsed_sec": elapsed,
                "structured_scene": structured_scene,
            }
        )
        return result

    # Use whichever embedder is loaded (GemmaEmbedder preferred, CLIP fallback).
    gemma: GemmaEmbedder = models["clip"]  # type: ignore[assignment]
    _log.info("Embedding analyses using %s", _embedder_name)

    # 2. Scene change detection via consecutive-frame cosine distance
    gemma_embeds: np.ndarray | None = None
    try:
        _log.info("Scene change detection ...")
        gemma_embeds = gemma.encode_images(sample_images)
        changes = []
        for i in range(1, n):
            cos_sim = float(np.dot(gemma_embeds[i - 1], gemma_embeds[i]))
            distance = 1.0 - cos_sim
            if distance >= _SCENE_CHANGE_THRESH:
                changes.append({"frame_idx": i, "t_sec": sample_ts[i], "distance": distance})
        task_results["scene_change_detection"] = {
            "description": "Consecutive-frame cosine distance > threshold",
            "n_changes": len(changes),
            "threshold": _SCENE_CHANGE_THRESH,
            "changes": changes,
        }
        _log.info("Scene changes detected: %d", len(changes))
    except Exception as exc:
        task_results["scene_change_detection"] = {"error": str(exc)}
        _log.warning("Scene change detection failed: %s", exc)

    # 3. Greedy cosine-based scene clustering
    try:
        _log.info("Semantic scene clustering ...")
        cl_embeds = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        sim_mat = np.dot(cl_embeds, cl_embeds.T)
        labels = [-1] * n
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] == -1 and sim_mat[i, j] > (1.0 - _SCENE_CHANGE_THRESH):
                    labels[j] = cluster_id
            cluster_id += 1
        task_results["scene_clustering"] = {
            "description": "Greedy scene grouping by cosine similarity",
            "n_clusters": cluster_id,
            "n_frames": n,
            "mean_cluster_size": round(n / max(1, cluster_id), 2),
        }
        _log.info("Scene clusters: %d from %d frames", cluster_id, n)
    except Exception as exc:
        task_results["scene_clustering"] = {"error": str(exc)}
        _log.warning("Scene clustering failed: %s", exc)

    # 4. Zero-shot scene classification via text probe matching
    try:
        _log.info("Zero-shot scene classification (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        clf_frame = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        clf_text = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        clf_scores = np.dot(clf_frame, clf_text.T)  # (n_frames, n_categories)
        from collections import Counter

        top_cats: list[str] = [_GEMMA_TEXT_PROBES[int(np.argmax(clf_scores[i]))] for i in range(n)]
        cat_dist = dict(Counter(top_cats).most_common(5))
        task_results["scene_classification"] = {
            "description": f"Zero-shot classification against {len(_GEMMA_TEXT_PROBES)} scene categories",
            "n_frames": n,
            "category_distribution": cat_dist,
        }
        _log.info("Top category: %s", next(iter(cat_dist)) if cat_dist else "---")
    except Exception as exc:
        task_results["scene_classification"] = {"error": str(exc)}
        _log.warning("Zero-shot classification failed: %s", exc)

    # 5. Cross-modal text -> frame retrieval
    text_query_results = []
    try:
        _log.info("Cross-modal text->frame retrieval (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        doc_embeds = (
            gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        )
        query_embeds = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        tq_scores = np.dot(query_embeds, doc_embeds.T)  # (n_queries, n_frames)
        for q_idx, query in enumerate(_GEMMA_TEXT_PROBES):
            top_idxs = list(np.argsort(-tq_scores[q_idx])[:3])
            text_query_results.append(
                {
                    "query": query,
                    "top_results": [
                        {
                            "frame_path": sample_paths[i],
                            "t_sec": sample_ts[i],
                            "score": float(tq_scores[q_idx, i]),
                        }
                        for i in top_idxs
                    ],
                }
            )
        task_results["cross_modal_retrieval"] = {
            "description": "Text probes matched against Gemma frame embeddings",
            "n_queries": len(_GEMMA_TEXT_PROBES),
        }
    except Exception as exc:
        task_results["cross_modal_retrieval"] = {"error": str(exc)}
        _log.warning("Cross-modal retrieval failed: %s", exc)

    # 6. Temporal video embedding (mean-pool all frames)
    try:
        _log.info("Temporal video embedding ...")
        if hasattr(gemma, "encode_images_temporal"):
            vid_embed = gemma.encode_images_temporal(sample_images)
        else:
            # OpenCLIP fallback: mean-pool per-frame image embeddings and L2-normalise
            import torch as _torch

            _feats = gemma.encode_images(sample_images)  # (N, dim) numpy
            _t = _torch.from_numpy(_feats).mean(dim=0, keepdim=True)
            vid_embed = _torch.nn.functional.normalize(_t, dim=-1)
        task_results["temporal_embedding"] = {
            "description": f"Mean-pool of {n} frame embeddings -> single video-level vector",
            "dim": int(vid_embed.shape[1]),
            "n_frames": n,
        }
        _log.info("Temporal embedding dim=%d", vid_embed.shape[1])
    except Exception as exc:
        task_results["temporal_embedding"] = {"error": str(exc)}
        _log.warning("Temporal embedding failed: %s", exc)

    # 7. Gemma vs CLIP comparison — skip when the main embedder IS CLIP (trivial)
    clip_comparison: dict[str, Any] = {"available": False}
    from selfsuvis.models.openclip_model import OpenCLIPEmbedder as _CLIPModel

    _main_is_clip = isinstance(gemma, _CLIPModel)
    if _main_is_clip:
        clip_comparison = {
            "available": False,
            "reason": "main embedder is OpenCLIP — comparison skipped (self vs self)",
        }
        _log.info("  [Gemma vs CLIP] Skipped — main embedder is already OpenCLIP")
    else:
        try:
            _log.info("  [Gemma vs CLIP] Loading temporary OpenCLIP ViT-B-16 ...")
            temp_clip = _CLIPModel()
            clip_frame = temp_clip.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_c = np.dot(g_e, g_e.T)
            c_sim = np.dot(clip_frame, clip_frame.T)
            mask_c = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_c = float(np.mean(g_sim_c[mask_c]))
            mean_cossim_clip = float(np.mean(c_sim[mask_c]))
            k_c = min(5, n - 1)
            mnn_c = 0
            for i in range(n):
                gr = g_sim_c[i].copy()
                gr[i] = -2.0
                cr = c_sim[i].copy()
                cr[i] = -2.0
                mnn_c += len(
                    set(np.argsort(-gr)[:k_c].tolist()) & set(np.argsort(-cr)[:k_c].tolist())
                )
            mnn_rate_c = mnn_c / (n * k_c)
            clip_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_c,
                "mnn_rate": mnn_rate_c,
                "mean_cossim_gemma": mean_cossim_gemma_c,
                "mean_cossim_clip": mean_cossim_clip,
            }
            _log.info(
                "  [Gemma vs CLIP] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  CLIP=%.4f",
                k_c,
                mnn_rate_c,
                mean_cossim_gemma_c,
                mean_cossim_clip,
            )
            try:
                import torch as _t

                _bb = getattr(temp_clip, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_clip, clip_frame
        except Exception as exc:
            clip_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs CLIP] comparison failed: %s", exc)

    # 8. Gemma vs DINOv3 comparison
    dino_comparison: dict[str, Any] = {"available": False}
    if _HAS_DINO and n > 1:
        try:
            _log.info("  [Gemma vs DINOv3] Loading temporary DINOv3 ViT-B/14 ...")
            temp_dino = DINOEmbedder("dinov3_vitb14")
            dino_embeds = temp_dino.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_d = np.dot(g_e, g_e.T)
            d_sim = np.dot(dino_embeds, dino_embeds.T)
            mask_d = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_d = float(np.mean(g_sim_d[mask_d]))
            mean_cossim_dino = float(np.mean(d_sim[mask_d]))
            k_d = min(5, n - 1)
            mnn_d = 0
            for i in range(n):
                gr = g_sim_d[i].copy()
                gr[i] = -2.0
                dr = d_sim[i].copy()
                dr[i] = -2.0
                mnn_d += len(
                    set(np.argsort(-gr)[:k_d].tolist()) & set(np.argsort(-dr)[:k_d].tolist())
                )
            mnn_rate_d = mnn_d / (n * k_d)
            dino_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_d,
                "mnn_rate": mnn_rate_d,
                "mean_cossim_gemma": mean_cossim_gemma_d,
                "mean_cossim_dino": mean_cossim_dino,
            }
            _log.info(
                "  [Gemma vs DINOv3] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  DINO=%.4f",
                k_d,
                mnn_rate_d,
                mean_cossim_gemma_d,
                mean_cossim_dino,
            )
            try:
                import torch as _t

                _bb = getattr(temp_dino, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_dino, dino_embeds
        except Exception as exc:
            dino_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs DINOv3] comparison failed: %s", exc)
    else:
        dino_comparison["reason"] = (
            "DINOv3 not available" if not _HAS_DINO else "too few frames for comparison"
        )

    elapsed = time.time() - t0

    write_gemma_analysis_md(
        video_dir / "gemma_analysis.md",
        video_name,
        effective_api_model or settings.GEMMA_MODEL_ID,
        n,
        task_results,
        dino_comparison,
        text_query_results,
        elapsed,
        clip_comparison=clip_comparison,
    )

    result.update(
        {
            "skipped": False,
            "n_frames": n,
            "task_results": task_results,
            "dino_comparison": dino_comparison,
            "clip_comparison": clip_comparison,
            "elapsed_sec": elapsed,
            "structured_scene": structured_scene,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Step 04 — Florence-2 scene captioning
# ---------------------------------------------------------------------------


def step_scene_captioning(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    models: dict[str, Any] | None = None,
    qwen_api_url: str = "",
    qwen_model: str = "",
    florence_api_url: str = "",
    florence_model: str = "",
    domain_hint: str = "",
) -> dict[str, Any]:
    """Step 04: Florence-2 scene captioning with memory management and API support.

    Memory strategy (CUDA only):
      1. If ``florence_api_url`` is set: call Florence-2 via vLLM API — no local
         weights loaded, zero VRAM consumed.  Use this when another process
         (e.g. Ollama) already occupies most of VRAM.
      2. Otherwise load Florence-2 locally:
         a. Offload CLIP+DINO to CPU to free ~1.7 GiB.
         b. If ``qwen_api_url`` looks like Ollama (port 11434): send keep_alive=0
            to evict the VLM (~11-12 GiB freed), giving Florence plenty of room.
            Ollama auto-reloads on the next request (step 12).
         c. If Florence still OOMs and ``qwen_api_url`` + ``qwen_model`` are set:
            fall back to Qwen API captioning.
    """
    from .report import write_scene_captions_md

    # -- API route: vLLM serving Florence-2 ------------------------------------
    effective_florence_api_url = florence_api_url or settings.FLORENCE_API_URL
    effective_florence_model = florence_model or settings.FLORENCE_MODEL
    if effective_florence_api_url:
        _log.info("  Florence-2 via vLLM API at %s", effective_florence_api_url)
        _log_vram_snapshot("before Florence API captioning")
        if models and device == "cuda":
            _offload_models_to_cpu(models)
        result = caption_via_florence_api(
            frame_list,
            video_name,
            video_dir,
            effective_florence_api_url,
            effective_florence_model,
            domain_hint=domain_hint,
        )
        _log_vram_snapshot("after Florence API captioning")
        return result

    # -- Local route: load Florence-2 weights into this process ----------------
    out_md = video_dir / "scene_captions.md"
    try:
        from selfsuvis.pipeline.vision.florence import FlorenceModel
    except ImportError as exc:
        _log.warning("  Florence-2 unavailable (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    # Step 1: offload CLIP+DINO to free ~1.7 GiB
    if models and device == "cuda":
        _offload_models_to_cpu(models)

    # Step 2: unload all Ollama models to free VRAM for Florence
    if device == "cuda":
        if qwen_api_url and qwen_model:
            _unload_ollama_model(qwen_api_url, qwen_model)
        # Also unload Gemma sidecar if configured (may still be resident from step 03)
        _gemma_url_cap = settings.GEMMA_API_URL
        _gemma_model_cap = settings.GEMMA_API_MODEL
        if _gemma_url_cap and _gemma_model_cap and _gemma_model_cap != qwen_model:
            _unload_ollama_model(_gemma_url_cap, _gemma_model_cap)

    _log.info("Loading Florence-2-large on %s …", device)
    _log_vram_snapshot("before local Florence load")
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        if qwen_api_url and qwen_model:
            _log.warning("  Florence-2 load failed (%s) — using Qwen API fallback", exc)
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _log.info("  CUDA cache cleared before Qwen fallback")
            except Exception:
                pass
            return caption_via_qwen_api(
                frame_list, video_name, video_dir, qwen_api_url, qwen_model, domain_hint=domain_hint
            )
        _log.warning(
            "  Florence-2 load failed (%s) — skipping captioning "
            "(pass --qwen-api-url + --qwen to enable Qwen API fallback)",
            exc,
        )
        return {"skipped": True, "reason": str(exc), "captions": []}

    _log.info("  [ok] Florence-2-large loaded in %.1fs", time.time() - t0)
    _log.info("  Captioning %d frames …", len(frame_list))
    caption_results: list[dict[str, Any]] = []
    florence_runtime_mode = florence.runtime_mode
    florence_model_tag = florence.model_tag
    batch_size = settings.FLORENCE_BATCH_SIZE
    _florence_oom = False
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        if _florence_oom:
            captions_and_confs: list[tuple[str, float]] = [("", 0.5)] * len(batch)
        else:
            pil_images = []
            for fp, _t in batch:
                try:
                    pil_images.append(Image.open(fp).convert("RGB"))
                except Exception:
                    pil_images.append(Image.new("RGB", (224, 224)))
            try:
                captions_and_confs = florence.caption_batch(pil_images)
                florence_runtime_mode = florence.runtime_mode
            except Exception as exc:
                from selfsuvis.pipeline.core.gpu_utils import is_cuda_oom, log_oom_banner

                if is_cuda_oom(exc):
                    remaining = len(frame_list) - batch_start
                    log_oom_banner(
                        _log,
                        "Florence-2 caption_batch",
                        f"batch_start={batch_start}, releasing model, "
                        f"{remaining} frames will get empty captions",
                    )
                    try:
                        import torch as _t

                        _t.cuda.empty_cache()
                        florence.release()
                    except Exception:
                        pass
                    _florence_oom = True
                else:
                    _log.warning("  Florence batch %d failed: %s", batch_start, exc, exc_info=True)
                captions_and_confs = [("", 0.5)] * len(batch)
        for (fp, t_sec), (cap, conf) in zip(batch, captions_and_confs):
            caption_results.append(
                {"frame_path": fp, "t_sec": t_sec, "caption": cap, "caption_confidence": conf}
            )

    elapsed = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  [ok] %d/%d frames captioned in %.1fs", captioned, len(frame_list), elapsed)
    _log_vram_snapshot("after local Florence captioning")
    write_scene_captions_md(
        out_md,
        video_name,
        caption_results,
        elapsed,
        model_tag=florence_model_tag,
        runtime_mode=florence_runtime_mode,
    )
    florence.release()
    # VRAM freed — caller (_run_video_pipeline) decides when to restore CLIP+DINO

    return {
        "skipped": False,
        "captions": caption_results,
        "captioned_count": captioned,
        "elapsed_sec": elapsed,
        "florence_runtime_mode": florence_runtime_mode,
        "florence_model_tag": florence_model_tag,
    }


# ---------------------------------------------------------------------------
# Step 04b — Gemma segment-boundary diff
# ---------------------------------------------------------------------------


def step_gemma_segment_captions(
    frame_list: list[tuple[str, float]],
    caption_results: list[dict[str, Any]],
    video_name: str,
    video_dir: Path,
    gemma_api_url: str = "",
    gemma_api_model: str = "",
) -> dict[str, Any]:
    """Step 4b: Gemma 4 multi-frame segment-boundary diff analysis.

    Uses _analyze_caption_sequence to find scene boundaries from Florence captions,
    then for each boundary pair calls the Gemma sidecar with both frames and a diff
    prompt ("What changed between these two frames?").

    Writes ``gemma_segment_captions.md`` to *video_dir*.
    Skips gracefully when no sidecar is configured or no captions are available.
    """
    from .common import _analyze_caption_sequence
    from .report import write_gemma_segment_captions_md

    result: dict[str, Any] = {"skipped": True, "reason": "", "boundary_diffs": []}

    effective_api_url = gemma_api_url or settings.GEMMA_API_URL
    effective_api_model = gemma_api_model or settings.GEMMA_API_MODEL
    effective_timeout = float(settings.GEMMA_API_TIMEOUT_SEC)

    if not effective_api_url:
        result["reason"] = "GEMMA_API_URL not configured"
        _log.info("  Gemma segment captions skipped: %s", result["reason"])
        return result

    if not caption_results:
        result["reason"] = "no caption results available"
        _log.info("  Gemma segment captions skipped: %s", result["reason"])
        return result

    ts_to_fp: dict[float, str] = {t: fp for fp, t in frame_list}
    enriched = _analyze_caption_sequence(caption_results)

    from .caption_helpers.frame_selection import _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT

    max_boundaries = int(
        getattr(settings, "GEMMA_SEGMENT_DIFF_MAX_BOUNDARIES", _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT)
        or 0
    )
    boundary_pairs = _select_segment_boundary_pairs(enriched, max_boundaries=max_boundaries)

    if not boundary_pairs:
        result["reason"] = "no segment boundaries found"
        _log.info("  Gemma segment captions: no segment boundaries (all frames same segment)")
        return result

    total_boundaries = sum(1 for row in enriched if row.get("is_new_segment")) - 1
    if max_boundaries > 0 and total_boundaries > len(boundary_pairs):
        _log.info(
            "  Gemma segment boundary ranking: %d candidates → top %d strongest diffs",
            total_boundaries,
            len(boundary_pairs),
        )

    _log.info(
        "Gemma segment-boundary diff: %d boundaries  model=%s  url=%s ...",
        len(boundary_pairs),
        effective_api_model,
        effective_api_url,
    )
    t0 = time.time()

    boundary_diffs: list[dict[str, Any]] = []
    for idx, (prev_row, next_row) in enumerate(boundary_pairs):
        fp_before = ts_to_fp.get(prev_row.get("t_sec", -1.0), "") or prev_row.get("frame_path", "")
        fp_after = ts_to_fp.get(next_row.get("t_sec", -1.0), "") or next_row.get("frame_path", "")
        if not fp_before or not fp_after:
            _log.debug("  Gemma diff: missing frame paths at boundary %d — skipping", idx)
            continue

        desc = _gemma_diff_two_frames_via_api(
            fp_before, fp_after, effective_api_url, effective_api_model, effective_timeout
        )
        entry = {
            "boundary_idx": idx,
            "prev_t_sec": prev_row.get("t_sec", 0.0),
            "next_t_sec": next_row.get("t_sec", 0.0),
            "prev_segment_id": prev_row.get("segment_id", 0),
            "next_segment_id": next_row.get("segment_id", 0),
            "fp_before": fp_before,
            "fp_after": fp_after,
            "diff_description": desc,
        }
        boundary_diffs.append(entry)

    elapsed = time.time() - t0
    described = sum(1 for b in boundary_diffs if b.get("diff_description"))
    _log.info(
        "  [ok] Gemma segment diffs: %d/%d boundaries described in %.1fs",
        described,
        len(boundary_pairs),
        elapsed,
    )

    write_gemma_segment_captions_md(
        video_dir / "gemma_segment_captions.md", video_name, effective_api_model, boundary_diffs
    )

    result.update(
        {
            "skipped": False,
            "boundary_count": len(boundary_pairs),
            "described_count": described,
            "elapsed_sec": elapsed,
            "model": effective_api_model,
            "boundary_diffs": boundary_diffs,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Step 05 — ASR transcription
# ---------------------------------------------------------------------------


def step_asr_transcription(
    video_path: Path,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 05: extract audio, run Whisper ASR."""
    from datetime import datetime

    from .common import _RUNNER_LABEL

    out_md = video_dir / "asr_subtitles.md"
    result: dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    try:
        from selfsuvis.pipeline.media.audio import extract_audio, map_subtitles_to_frames
        from selfsuvis.pipeline.vision.asr import ASRModel
    except ImportError as exc:
        _log.warning("  ASR unavailable (%s) — skipping", exc)
        return result
    asr = ASRModel()
    _log_vram_snapshot("before ASR model use")
    if not asr.is_enabled():
        _log.info("  ASR disabled (ASR_ENABLED=false) — skipping")
        return result
    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Extracting audio from %s …", video_path.name)
    wav_path = extract_audio(str(video_path), str(audio_dir))
    if not wav_path:
        _log.warning("  No audio stream found in %s — ASR skipped", video_path.name)
        return result
    _log.info("Transcribing audio with %s …", asr.model_id)
    t0 = time.time()
    segments = asr.transcribe(wav_path)
    elapsed = time.time() - t0
    if not segments:
        _log.warning("  ASR returned no segments for %s", video_path.name)
        return result
    frame_timestamps = [t for _, t in frame_list]
    subtitle_map = map_subtitles_to_frames(
        segments, frame_timestamps, window_sec=settings.ASR_SUBTITLE_WINDOW_SEC
    )
    covered = sum(1 for t in frame_timestamps if t in subtitle_map)
    _log.info(
        "  [ok] ASR: %d segments → %d/%d frames have subtitles (%.1fs, model=%s)",
        len(segments),
        covered,
        len(frame_list),
        elapsed,
        asr.model_id,
    )
    _log_vram_snapshot("after ASR model use")
    lines = [
        f"# ASR Subtitles — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{asr.model_id}`",
        f"Segments: {len(segments)}  |  Frames with subtitles: {covered}/{len(frame_list)}",
        f"Elapsed: {elapsed:.1f}s",
        "",
        "## Subtitle Segments",
        "",
        "| Start (s) | End (s) | Text |",
        "|-----------|---------|------|",
    ]
    for seg in segments:
        ts = seg.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        start = float(ts[0]) if len(ts) > 0 and ts[0] is not None else 0.0
        end = float(ts[1]) if len(ts) > 1 and ts[1] is not None else start
        text = seg.get("text", "").strip().replace("|", "\\|")
        lines.append(f"| {start:.2f} | {end:.2f} | {text} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · ASR step 05*"]
    write_markdown_artifact(out_md, lines)
    result.update(
        {
            "skipped": False,
            "subtitle_map": subtitle_map,
            "segments": segments,
            "elapsed_sec": elapsed,
            "covered_frames": covered,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Step 06 — OCR text extraction
# ---------------------------------------------------------------------------


def step_ocr_extraction(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    caption_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Step 06: visible text extraction per frame."""
    result: dict[str, Any] = {"skipped": True, "ocr_results": []}
    try:
        from selfsuvis.pipeline.vision.ocr import OCRModel
    except ImportError as exc:
        _log.warning("  OCR unavailable (%s) — skipping", exc)
        return result
    ocr = OCRModel()
    _log_vram_snapshot("before OCR model use")
    if not ocr.is_enabled():
        _log.info("  OCR disabled (OCR_ENABLED=false) — skipping")
        return result
    _log.info("Running OCR on %d frames (model=%s) …", len(frame_list), ocr.model_id)
    t0 = time.time()
    threshold = settings.OCR_MIN_CAPTION_CONFIDENCE
    max_ocr = int(settings.OCR_MAX_FRAMES)
    selected_frame_list, skipped_by_caption, ranking = _select_ocr_candidate_frames(
        frame_list=frame_list,
        caption_results=caption_results,
        ocr_model_id=ocr.model_id,
        threshold=threshold,
        max_ocr=max_ocr,
    )
    if ranking:
        top_score = max(float(item.get("score", 0.0) or 0.0) for item in ranking)
        _log.info(
            "  OCR ranked selection: %d/%d frames kept (top score %.2f, OCR_MAX_FRAMES=%d)",
            len(selected_frame_list),
            len(frame_list),
            top_score,
            max_ocr,
        )
    elif threshold > 0.0:
        if len(selected_frame_list) < len(frame_list):
            _log.info(
                "  OCR caption prescreen unavailable (ran concurrently) — "
                "capped to %d/%d evenly spaced frames (OCR_MAX_FRAMES=%d)",
                len(selected_frame_list),
                len(frame_list),
                max_ocr,
            )
        else:
            _log.info(
                "  OCR ranked selection unavailable — using all %d frames",
                len(selected_frame_list),
            )

    if not selected_frame_list and frame_list:
        selected_frame_list = _fallback_ocr_frame_sample(frame_list)
        selected_paths = {fp for fp, _ in selected_frame_list}
        for fp, meta in skipped_by_caption.items():
            if fp in selected_paths:
                meta.pop("ocr_skipped_by_caption", None)
                meta["ocr_prescreen_fallback"] = True
        _log.info(
            "  OCR prescreen fallback: selected %d evenly spaced frames because caption prescreen skipped everything",
            len(selected_frame_list),
        )

    processed_results = _run_batched_frame_inference(
        selected_frame_list,
        batch_size=settings.OCR_BATCH_SIZE,
        batch_fn=lambda _batch, imgs: ocr.extract_text_batch(imgs),
        warning_label="OCR",
        error_result={"ocr_text": "", "ocr_error": True},
    )
    processed_by_frame = {str(r["frame_path"]): r for r in processed_results}
    ocr_results: list[dict[str, Any]] = []
    for fp, t_sec in frame_list:
        if fp in processed_by_frame:
            ocr_results.append(processed_by_frame[fp])
        else:
            ocr_results.append(
                skipped_by_caption.get(
                    fp,
                    {"frame_path": fp, "t_sec": t_sec, "ocr_text": "", "ocr_error": True},
                )
            )
    elapsed = time.time() - t0
    non_empty = sum(1 for r in ocr_results if r.get("ocr_text"))
    _log.info("  [ok] OCR: %d/%d frames have text in %.1fs", non_empty, len(frame_list), elapsed)
    result.update(
        {
            "skipped": False,
            "ocr_results": ocr_results,
            "non_empty": non_empty,
            "elapsed_sec": elapsed,
        }
    )
    ocr.release()
    _log_vram_snapshot("after OCR model use")
    return result


# ---------------------------------------------------------------------------
# Step 07 — Depth estimation
# ---------------------------------------------------------------------------


def step_depth_estimation(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 07: depth estimation per frame."""
    result: dict[str, Any] = {"skipped": True, "depth_results": []}
    try:
        from selfsuvis.pipeline.vision.depth import DepthModel
    except ImportError as exc:
        _log.warning("  Depth model unavailable (%s) — skipping", exc)
        return result
    depth_model = DepthModel()
    _log_vram_snapshot("before depth model use")
    if not depth_model.is_enabled():
        _log.info("  Depth disabled (DEPTH_ENABLED=false) — skipping")
        return result
    _log.info(
        "Running depth estimation on %d frames (model=%s) …", len(frame_list), depth_model.model_id
    )
    t0 = time.time()
    depth_results = _run_batched_frame_inference(
        frame_list,
        batch_size=max(1, int(getattr(settings, "DEPTH_BATCH_SIZE", 8) or 8)),
        batch_fn=lambda _batch, imgs: depth_model.estimate_batch(imgs),
        warning_label="Depth",
        error_result={"depth_error": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in depth_results
        if not r.get("depth_error")
        and not r.get("depth_unavailable")
        and not r.get("depth_disabled")
    )
    _log.info("  [ok] Depth: %d/%d frames estimated in %.1fs", ok, len(frame_list), elapsed)
    result.update(
        {"skipped": False, "depth_results": depth_results, "ok_count": ok, "elapsed_sec": elapsed}
    )
    depth_model.release()
    _log_vram_snapshot("after depth model use")
    return result


# ---------------------------------------------------------------------------
# Step 08 — Object detection
# ---------------------------------------------------------------------------


def step_object_detection(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 08: object detection per frame."""
    result: dict[str, Any] = {"skipped": True, "detection_results": []}
    try:
        from selfsuvis.pipeline.vision.detection import DetectionModel
    except ImportError as exc:
        _log.warning("  Detection model unavailable (%s) — skipping", exc)
        return result
    det_model = DetectionModel()
    _log_vram_snapshot("before detection model use")
    if not det_model.is_enabled():
        _log.info("  Detection disabled (DETECTION_ENABLED=false) — skipping")
        return result
    _log.info(
        "Running object detection on %d frames (model=%s) …", len(frame_list), det_model.model_id
    )
    t0 = time.time()
    det_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: det_model.detect_batch(imgs),
        warning_label="Detection",
        error_result={"detection_error": True},
    )
    elapsed = time.time() - t0
    total_objs = sum(len(r.get("detections", [])) for r in det_results)
    ok = sum(
        1
        for r in det_results
        if not r.get("detection_error")
        and not r.get("detection_unavailable")
        and not r.get("detection_disabled")
    )
    _log.info(
        "  [ok] Detection: %d objects across %d/%d frames in %.1fs",
        total_objs,
        ok,
        len(frame_list),
        elapsed,
    )
    result.update(
        {
            "skipped": False,
            "detection_results": det_results,
            "total_objects": total_objs,
            "ok_count": ok,
            "elapsed_sec": elapsed,
        }
    )
    det_model.release()
    _log_vram_snapshot("after detection model use")
    return result


# ---------------------------------------------------------------------------
# Step 11 — World model pass
# ---------------------------------------------------------------------------


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

                # Embed all frames with CLIP (cheap; model already loaded)
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

                    # Align scores back to frame_list (some frames may have been skipped)
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


# ---------------------------------------------------------------------------
# Step 12 — Qwen VLM detailed captioning
# ---------------------------------------------------------------------------


def step_qwen_captioning(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: dict[float, str],
    ocr_results: list[dict[str, Any]],
    clip_prescreen_fn=None,
    knowledge: Optional["VideoKnowledge"] = None,
) -> dict[str, Any]:
    """Step 12: Qwen VLM detailed scene captioning with full agentic context.

    When *knowledge* is provided, each frame's prompt is enriched with all
    prior observations: Florence caption, depth profile, detected objects,
    scene segment, ASR, OCR, and the previous frame's Qwen structured output.
    This lets Qwen reason about *what changed* rather than describing each
    frame in isolation.
    """
    from .report import write_detailed_captions_md

    out_md = video_dir / "detailed_captions.md"
    result: dict[str, Any] = {"skipped": True, "results": []}
    try:
        from selfsuvis.pipeline.vision.qwen import QwenModel
    except ImportError as exc:
        _log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result
    qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)
    _log_vram_snapshot("before Qwen sidecar use")
    if not qwen.is_enabled():
        _log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        _log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        # Gemma fallback: when GEMMA_API_URL is set, use Gemma structured extraction
        # to produce the same JSON schema as Qwen (vehicle_groups, road_surface, etc.)
        gemma_url = settings.GEMMA_API_URL
        gemma_model = settings.GEMMA_API_MODEL
        if gemma_url:
            return step_qwen_captioning_gemma_fallback(
                frame_list,
                video_name,
                video_dir,
                gemma_url,
                gemma_model,
            )
        return result
    ocr_map: dict[float, str] = {
        r["t_sec"]: r["ocr_text"]
        for r in ocr_results
        if r.get("t_sec") is not None and r.get("ocr_text")
    }

    domain = knowledge.domain_hint() if knowledge else ""
    if domain:
        _log.info("  Qwen domain hint: %s", domain)
    qwen_budget = _adaptive_sparse_budget(
        frame_list,
        configured_max=max(1, int(settings.QWEN_MAX_FRAMES)),
        seconds_per_sample=0.9,
        floor=8,
    )
    sampled_frame_list = _select_qwen_frames(
        frame_list,
        max_frames=qwen_budget,
        knowledge=knowledge,
        ocr_map=ocr_map,
    )
    if len(sampled_frame_list) < len(frame_list):
        _log.info(
            "  Qwen frame selection: %d/%d frames chosen for detailed captioning",
            len(sampled_frame_list),
            len(frame_list),
        )
    t0 = time.time()

    # Probe agentic mode with one frame before committing to the full run.
    # Models < 7B parameters frequently fail to produce the structured JSON
    # required by agentic prompts; detecting this on frame 1 avoids wasting
    # time running the entire frame list only to get 100% parse errors.
    _use_agentic = knowledge is not None
    if _use_agentic and sampled_frame_list:
        _probe_fp, _probe_t = sampled_frame_list[0]
        _probe_img = _open_frame_image(_probe_fp)
        if _probe_img is not None:
            _probe_res = qwen.extract_batch(
                [_probe_img],
                subtitle_texts=[subtitle_map.get(_probe_t) or None],
                ocr_texts=[ocr_map.get(_probe_t) or None],
                extra_contexts=[knowledge.context_for_frame(_probe_t)],
                domain_hint=domain or None,
            )
            if _probe_res and _probe_res[0].get("parse_error"):
                _log.warning(
                    "  Qwen agentic probe: parse error on first frame -- "
                    "falling back to non-agentic mode. "
                    "Model '%s' appears too small for structured JSON output; "
                    "use qwen2.5vl:32b or larger to keep agentic mode.",
                    settings.QWEN_MODEL,
                )
                _use_agentic = False

    _log.info(
        "Running Qwen detailed captioning on %d sampled frames (from %d total, model=%s  agentic=%s) ...",
        len(sampled_frame_list),
        len(frame_list),
        settings.QWEN_MODEL,
        "yes" if _use_agentic else "no",
    )

    caption_results: list[dict[str, Any]] = []

    def _batch_fn(batch: list[tuple[str, float]], imgs: list) -> list[dict[str, Any]]:
        extra_contexts = None
        if _use_agentic and knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        results = qwen.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )
        # Feed each successful result back into knowledge as prior state
        if _use_agentic and knowledge:
            for r in results:
                knowledge.update_qwen_state(r)
        return results

    batch_results = _run_batched_frame_inference(
        sampled_frame_list,
        batch_size=4,
        batch_fn=_batch_fn,
        warning_label="Qwen",
        error_result={"service_unavailable": True},
    )
    for r in batch_results:
        t_sec = r.get("t_sec", 0.0)
        caption_results.append({**r, "subtitle_text": subtitle_map.get(t_sec) or ""})
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in caption_results
        if not r.get("service_unavailable") and not r.get("skipped") and not r.get("parse_error")
    )
    parse_errors = sum(1 for r in caption_results if r.get("parse_error"))
    subtitle_used = sum(1 for r in caption_results if r.get("subtitle_text"))
    _log.info(
        "  [ok] Qwen: %d/%d sampled frames captioned in %.1fs (%d with ASR  parse_errors=%d  agentic=%s)",
        ok,
        len(sampled_frame_list),
        elapsed,
        subtitle_used,
        parse_errors,
        "yes" if _use_agentic else "no",
    )
    _log_vram_snapshot("after Qwen sidecar use")
    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, settings.QWEN_MODEL)
    result.update(
        {
            "skipped": False,
            "results": caption_results,
            "ok_count": ok,
            "subtitle_used": subtitle_used,
            "elapsed_sec": elapsed,
            "sampled_count": len(sampled_frame_list),
            "total_frames": len(frame_list),
            "parse_error_count": parse_errors,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Step 13 — UniDriveVLA expert analysis
# ---------------------------------------------------------------------------


def step_unidrive_analysis(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: dict[float, str],
    ocr_results: list[dict[str, Any]],
    knowledge: Optional["VideoKnowledge"] = None,
) -> dict[str, Any]:
    """Step 13: UniDriveVLA expert analysis on a sparse frame sample."""
    from .report import write_unidrive_analysis_md

    out_md = video_dir / "unidrive_analysis.md"
    result: dict[str, Any] = {"skipped": True, "results": []}
    try:
        from selfsuvis.pipeline.vision.unidrive import UniDriveVLAModel
    except ImportError as exc:
        _log.warning("  UniDriveVLA client unavailable (%s) — skipping", exc)
        return result

    client = UniDriveVLAModel()
    _log_vram_snapshot("before UniDrive sidecar use")
    if not client.is_enabled():
        _log.info("  UniDriveVLA disabled (no sidecar URL and no usable local HF model) — skipping")
        _log.info("  To enable sidecar mode: --unidrive-api-url http://localhost:8030/v1")
        _log.info(
            "  To enable local mode: cache HF weights with scripts/prepare_models.py --unidrive --unidrive-backend vllm"
        )
        return result

    max_frames = _adaptive_sparse_budget(
        frame_list,
        configured_max=max(1, int(getattr(settings, "UNIDRIVE_MAX_FRAMES", 24) or 24)),
        seconds_per_sample=1.4,
        floor=6,
    )
    sample_step = max(1, len(frame_list) // max_frames)
    sampled_frames = frame_list[::sample_step][:max_frames]
    ocr_map: dict[float, str] = {
        r["t_sec"]: r["ocr_text"]
        for r in ocr_results
        if r.get("t_sec") is not None and r.get("ocr_text")
    }
    domain = knowledge.domain_hint() if knowledge else ""
    _log.info(
        "Running UniDriveVLA expert analysis on %d sampled frames (model=%s backend=%s) …",
        len(sampled_frames),
        settings.UNIDRIVE_MODEL,
        getattr(settings, "UNIDRIVE_BACKEND", "vllm"),
    )
    t0 = time.time()

    def _batch_fn(batch: list[tuple[str, float]], imgs: list[Image.Image]) -> list[dict[str, Any]]:
        extra_contexts = None
        if knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        return client.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )

    batch_results = _run_batched_frame_inference(
        sampled_frames,
        batch_size=2,
        batch_fn=_batch_fn,
        warning_label="UniDriveVLA",
        error_result={"service_unavailable": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1 for r in batch_results if not r.get("service_unavailable") and not r.get("parse_error")
    )
    _log.info(
        "  [ok] UniDriveVLA: %d/%d sampled frames analysed in %.1fs",
        ok,
        len(batch_results),
        elapsed,
    )
    if ok == 0 and batch_results:
        first_reason = batch_results[0].get("reason", "unknown")
        _log.warning(
            "  UniDriveVLA: all %d frames failed (reason: %s). "
            "Set --unidrive-api-url to point at an Ollama/vLLM endpoint.",
            len(batch_results),
            first_reason,
        )
    _log_vram_snapshot("after UniDrive sidecar use")
    write_unidrive_analysis_md(out_md, video_name, batch_results, elapsed, settings.UNIDRIVE_MODEL)
    client.release()
    result.update(
        {
            "skipped": False,
            "results": batch_results,
            "ok_count": ok,
            "elapsed_sec": elapsed,
            "sampled_frames": len(batch_results),
        }
    )
    return result

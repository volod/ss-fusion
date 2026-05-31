"""Phase 3 SSL-gated nodes: ssl_finetune, ssl_gate_router, distill, onnx_export,
ft_search, compare.
"""

import os
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..state import PipelineState
from ..runner import _append_agentic_step

_log = get_logger(__name__)

_SSL_GATE_MAX_LOSS = 10.0


def node_p3_ssl_finetune(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import _guard_min_free_vram, _prep_vram_for_step
    from ..steps_ssl import step_ssl_finetune

    args = state["args"]
    models = state["models"]
    device = state["device"]
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    if models.get("uses_api_embedder"):
        stats.setdefault("timings", {})["D_finetune"] = 0.0
        for step_id, title in [("17", "SSL fine-tuning"), ("18", "Knowledge distillation")]:
            _append_agentic_step(
                agentic_trace,
                step_id=step_id,
                title=title,
                description="Adapt the local DINO backbone to mission-specific footage.",
                status="skipped",
                context_inputs=["API embedder mode"],
                context_outputs=["no local fine-tuning checkpoint"],
                risks=["no task-specific adaptation in API-embedder mode"],
                artifacts=[],
            )
        return {
            "ssl_gate_passed": False,
            "checkpoint_path": "",
            "student_backbone": None,
            "student_dim": 768,
            "ssl_result": {"skipped": True},
            "stats": stats,
            "agentic_trace": agentic_trace,
        }

    if device == "cuda":
        _prep_vram_for_step(
            models,
            device,
            extra_sidecars=[
                (
                    getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL,
                    getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                ),
            ],
            label="SSL fine-tuning",
        )
        _guard_min_free_vram("SSL fine-tuning")

    frame_list = state["frame_list"]
    _n_batches = max(1, len(frame_list) // max(1, args.batch_size))
    _ssl_epochs = max(args.epochs, min(20, (200 + _n_batches - 1) // _n_batches))
    if _ssl_epochs != args.epochs:
        _log.info(
            "  SSL adaptive epochs: %d (CLI default=%d) — %d frames, %d batches/epoch",
            _ssl_epochs,
            args.epochs,
            len(frame_list),
            _n_batches,
        )

    t0 = time.monotonic()
    d = step_ssl_finetune(
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        frame_list,
        device,
        epochs=_ssl_epochs,
        batch_size=args.batch_size,
        tracking_results=((state.get("gemma_tracking_result") or {}).get("tracking_results") or []),
        depth_result=state.get("depth_result"),
        platform_state_fusion=state.get("platform_fusion_result"),
        full_fusion_result=state.get("full_fusion_result"),
        physical_state_result=state.get("physical_state_result"),
    )
    elapsed = time.monotonic() - t0

    stats["best_loss"] = d["best_loss"]
    stats["ckpt_mb"] = d["ckpt_mb"]
    stats.setdefault("timings", {})["D_finetune"] = elapsed

    checkpoint_path = d["checkpoint"]
    ssl_gate_passed = (
        bool(checkpoint_path)
        and os.path.exists(checkpoint_path)
        and d["best_loss"] < _SSL_GATE_MAX_LOSS
    )
    if ssl_gate_passed:
        _log.info(
            "  [ok] SSL gate passed (best_loss=%.4f < %.1f)", d["best_loss"], _SSL_GATE_MAX_LOSS
        )
    else:
        _log.warning(
            "  ✗ SSL gate did not pass (checkpoint=%r, best_loss=%.4f, threshold=%.1f)",
            checkpoint_path,
            d["best_loss"],
            _SSL_GATE_MAX_LOSS,
        )

    _append_agentic_step(
        agentic_trace,
        step_id="17",
        title="SSL fine-tuning",
        description="Adapt the local DINO backbone to mission-specific footage.",
        status="ok",
        context_inputs=["frame sequence", "base DINO initialization"],
        context_outputs=[f"best loss {d['best_loss']:.4f}", "mission-adapted backbone checkpoint"],
        risks=["small-video adaptation can overfit to accidental patterns"],
        artifacts=["finetune_stats.md", "checkpoints/dino_ssl_best.pt"],
    )

    return {
        "ssl_result": d,
        "ssl_gate_passed": ssl_gate_passed,
        "checkpoint_path": checkpoint_path,
        "student_backbone": None,
        "student_dim": 768,
        "stats": stats,
        "agentic_trace": agentic_trace,
        "clip_dino_on_gpu": False,
    }


def ssl_gate_router(state: PipelineState) -> str:
    """Conditional edge: route to distill if SSL gate passed, else skip to phase 4."""
    if state.get("ssl_gate_passed"):
        return "p3_distill"
    return "p4_multi_model_compare"


def node_p3_distill(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import _restore_models_to_gpu
    from ..steps_distill import step_distill

    args = state["args"]
    models = state["models"]
    device = state["device"]
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    if args.no_distill:
        stats.setdefault("timings", {})["E_distill"] = 0.0
        _append_agentic_step(
            agentic_trace,
            step_id="18",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["--no-distill flag"],
            context_outputs=["no distilled student"],
            risks=[],
            artifacts=[],
        )
        return {
            "student_backbone": None,
            "student_dim": 768,
            "distill_result": {"skipped": True},
            "stats": stats,
            "agentic_trace": agentic_trace,
        }

    # Restore CLIP+DINO for caption anchor embedding if needed.
    clip_dino_on_gpu = state.get("clip_dino_on_gpu", False)
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)

    # Build caption anchor embeddings from Florence captions via CLIP text encoder.
    caption_results = state.get("caption_results", [])
    _cap_anchor_embs = None
    if caption_results and models.get("clip"):
        try:
            _cap_texts = [
                r.get("caption", "") for r in caption_results if r.get("caption", "").strip()
            ]
            clip_model = models["clip"]
            if _cap_texts and hasattr(clip_model, "encode_texts"):
                _cap_anchor_embs = clip_model.encode_texts(_cap_texts)
                _log.info(
                    "  Distillation caption anchors: %d CLIP text embeddings", len(_cap_anchor_embs)
                )
        except Exception as exc:
            _log.debug("  Caption anchor prep failed (%s) — distilling without anchor", exc)

    t0 = time.monotonic()
    e_distill = step_distill(
        state["checkpoint_path"],
        state["frame_list"],
        state["video_name"],
        Path(state["video_dir"]),
        device,
        distill_epochs=args.distill_epochs,
        batch_size=args.batch_size,
        caption_embeddings=_cap_anchor_embs,
        gemma_embedder=None,
    )
    elapsed = time.monotonic() - t0

    student_backbone = None
    student_dim = 768
    if not e_distill.get("skipped"):
        student_backbone = e_distill["student_backbone"]
        student_dim = e_distill["student_dim"]
        stats["distill_loss"] = e_distill["best_loss"]
        stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
        stats["student_dim"] = student_dim
    stats.setdefault("timings", {})["E_distill"] = elapsed

    _append_agentic_step(
        agentic_trace,
        step_id="18",
        title="Knowledge distillation",
        description="Compress teacher geometry into a smaller ViT-S/14 student.",
        status="skipped" if e_distill.get("skipped") else "ok",
        context_inputs=["fine-tuned teacher checkpoint", "optional Florence caption anchors"],
        context_outputs=[f"student dim {student_dim}"]
        if not e_distill.get("skipped")
        else ["no distilled student"],
        risks=["teacher mistakes transfer into the student representation"],
        artifacts=["distill_stats.md", "checkpoints/student_best.pt"]
        if not e_distill.get("skipped")
        else [],
    )

    return {
        "student_backbone": student_backbone,
        "student_dim": student_dim,
        "distill_result": e_distill,
        "stats": stats,
        "agentic_trace": agentic_trace,
    }


def node_p3_onnx_export(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import _models_on_device, _restore_models_to_gpu
    from ..steps_distill import step_export_model

    args = state["args"]
    device = state["device"]
    models = state["models"]
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    clip_dino_on_gpu = state.get("clip_dino_on_gpu", False)
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)

    t0 = time.monotonic()
    e = step_export_model(
        state["checkpoint_path"],
        state["frame_list"],
        Path(state["video_dir"]),
        device,
        models,
        no_onnx=args.no_onnx,
        student_backbone=state.get("student_backbone"),
        student_dim=state.get("student_dim", 768),
    )
    elapsed = time.monotonic() - t0

    stats["onnx_mb"] = e.get("onnx_mb", 0.0)
    stats["onnx_exported"] = e.get("exported", False)
    stats.setdefault("timings", {})["F_export"] = elapsed

    _append_agentic_step(
        agentic_trace,
        step_id="19",
        title="ONNX export",
        description="Package the best available backbone and gallery into deployment artifacts.",
        status="ok",
        context_inputs=["teacher or student backbone", "retrieval gallery frames"],
        context_outputs=[
            f"onnx exported={e.get('exported', False)}",
            "gallery.npz for edge classification",
        ],
        risks=["export mismatches can change runtime behavior versus training"],
        artifacts=["edge_models/dino_local.onnx", "edge_models/gallery.npz"],
    )

    return {
        "export_result": e,
        "stats": stats,
        "agentic_trace": agentic_trace,
        "clip_dino_on_gpu": _models_on_device(models, device),
    }


def node_p3_ft_search(state: PipelineState) -> dict[str, Any]:
    from ..steps_embed import step_finetuned_model_search_test

    args = state["args"]
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    t0 = time.monotonic()
    f = step_finetuned_model_search_test(
        state["frame_list"],
        state["store"],
        state["is_qdrant"],
        state["models"],
        state["query_frame"],
        state["query_t_sec"],
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        top_k=args.top_k,
    )
    ft_results = f["results"]
    stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0
    stats.setdefault("timings", {})["G_ft_search"] = time.monotonic() - t0

    _append_agentic_step(
        agentic_trace,
        step_id="20",
        title="Fine-tuned search test",
        description="Re-run retrieval after adaptation to quantify search-space changes.",
        status="ok",
        context_inputs=["fine-tuned or distilled backbone", "same query frame as baseline"],
        context_outputs=[
            f"top-{len(ft_results)} adapted matches",
            f"top score {stats['ft_top_score']:.4f}",
        ],
        risks=["score improvements can hide semantic regressions"],
        artifacts=["finetuned_search.md"],
    )

    return {"ft_results": ft_results, "stats": stats, "agentic_trace": agentic_trace}


def node_p3_dae_finetune(state: PipelineState) -> dict[str, Any]:
    """Phase 3 DAE node: train a Denoising Autoencoder on mission frames.

    Runs after the DINO SSL step as an orthogonal self-supervised pretext task.
    The resulting checkpoint is stored in dae_result and used downstream by
    the active-learning scorer to add reconstruction-error novelty signals.

    Gracefully skips when fewer frames than two batches are available, or when
    the SSL gate did not pass (no point training a second model if the first
    did not converge).
    """
    from ..steps_ssl import step_dae_finetune

    args = state["args"]
    device = state["device"]
    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))

    frame_list = state.get("frame_list", [])
    if not frame_list:
        dae_result: dict[str, Any] = {"skipped": True, "reason": "no frames"}
        stats.setdefault("timings", {})["D2_dae"] = 0.0
        _append_agentic_step(
            agentic_trace,
            step_id="17b",
            title="DAE self-supervised pretraining",
            description="Train denoising autoencoder for reconstruction-error anomaly scoring.",
            status="skipped",
            context_inputs=["no frames available"],
            context_outputs=["no DAE checkpoint"],
            risks=[],
            artifacts=[],
        )
        return {"dae_result": dae_result, "stats": stats, "agentic_trace": agentic_trace}

    import time

    t0 = time.monotonic()
    dae_result = step_dae_finetune(
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        frame_list,
        device,
        epochs=getattr(args, "dae_epochs", max(5, getattr(args, "epochs", 10))),
        batch_size=getattr(args, "batch_size", 16),
    )
    elapsed = time.monotonic() - t0
    stats.setdefault("timings", {})["D2_dae"] = elapsed
    if not dae_result.get("skipped"):
        stats["dae_ckpt_mb"] = dae_result.get("ckpt_mb", 0.0)

    _append_agentic_step(
        agentic_trace,
        step_id="17b",
        title="DAE self-supervised pretraining",
        description="Train denoising autoencoder for reconstruction-error anomaly scoring.",
        status="skipped" if dae_result.get("skipped") else "ok",
        context_inputs=["mission frame sequence (unlabeled)"],
        context_outputs=[
            f"DAE checkpoint {dae_result.get('ckpt_mb', 0):.1f} MB",
            "per-frame reconstruction MSE available for anomaly scoring",
        ]
        if not dae_result.get("skipped")
        else ["not enough frames or training failed"],
        risks=["small dataset may cause DAE to memorise rather than generalise"],
        artifacts=["checkpoints/dae_best.pt", "dae_training_metrics.json"]
        if not dae_result.get("skipped")
        else [],
    )

    return {"dae_result": dae_result, "stats": stats, "agentic_trace": agentic_trace}


def node_p3_compare(state: PipelineState) -> dict[str, Any]:
    from ..runner import step_compare_and_describe

    stats = dict(state.get("stats", {}))
    agentic_trace = list(state.get("agentic_trace", []))
    video_context = dict(state.get("video_context", {}))

    t0 = time.monotonic()
    g = step_compare_and_describe(
        state["frame_list"],
        state["store"],
        state["is_qdrant"],
        state.get("base_results", []),
        state.get("ft_results", []),
        state["models"],
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        stats.get("ckpt_mb", 0.0),
        stats.get("onnx_mb", 0.0),
    )
    elapsed = time.monotonic() - t0

    if g:
        stats["base_infer_ms"] = g.get("base_infer_ms", 0.0)
        stats["ft_infer_ms"] = g.get("ft_infer_ms", 0.0)
        stats["top_description"] = g.get("top_description", "")
        video_context["top_descriptions"] = g.get("text_descriptions", [])
    stats.setdefault("timings", {})["H_compare"] = elapsed

    _append_agentic_step(
        agentic_trace,
        step_id="21",
        title="Comparison and description",
        description="Summarize retrieval changes and derive CLIP-based video description.",
        status="ok",
        context_inputs=["baseline and adapted retrieval outputs"],
        context_outputs=[f"top description: {stats.get('top_description', 'unknown')}"],
        risks=["top text prompt may sound plausible but be too coarse"],
        artifacts=["comparison.md", "description.md"],
    )

    return {
        "compare_result": g or {},
        "video_context": video_context,
        "stats": stats,
        "agentic_trace": agentic_trace,
    }

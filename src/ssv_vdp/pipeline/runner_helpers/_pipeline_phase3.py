"""Phase 3 — SSL-gated adaptation: Steps 21-26.

SSL fine-tuning, knowledge distillation (two stages), ONNX export, fine-tuned search,
and model comparison/description. All steps are guarded by the SSL gate.
"""

from pathlib import Path
from typing import Any

import numpy as np

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import _Timer, _banner, _step
from ._agentic import _append_agentic_step

_log = get_logger(__name__)

_TOTAL_STEPS = 35
_SSL_GATE_MAX_LOSS = 10.0


def run_phase3(
    *,
    args: Any,
    video_path: Path,
    video_dir: Path,
    video_name: str,
    video_id: str,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    frame_list: list[tuple[str, float]],
    clip_dino_on_gpu: bool,
    # results from phase 2
    caption_results: list[dict[str, Any]],
    base_results: list[dict[str, Any]],
    query_frame: str,
    query_t_sec: float,
    gemma_tracking_result: dict[str, Any],
    depth_result: dict[str, Any],
    platform_fusion_result: dict[str, Any],
    full_fusion_result: dict[str, Any],
    physical_state_result: dict[str, Any],
    # shared mutable state
    stats: dict[str, Any],
    T: dict[str, Any],
    video_context: dict[str, Any],
    agentic_trace: list[dict[str, Any]],
    knowledge: Any,
) -> dict[str, Any]:
    """Run Steps 21-26. Returns dict with ssl_gate_passed, ft_results, clip_dino_on_gpu."""
    from ...steps.caption import (
        _guard_min_free_vram,
        _models_on_device,
        _prep_vram_for_step,
        _restore_models_to_gpu,
    )
    from ...steps.adaptation.distill import step_distill, step_distill_stage2, step_export_model
    from ...steps.perception.embed import step_finetuned_model_search_test
    from ._compare import step_compare_and_describe

    try:
        from selfsuvis.models.gemma_model import GemmaEmbedder
        _HAS_GEMMA = True
    except Exception:
        _HAS_GEMMA = False

    _banner("Phase 3 — SSL-gated adaptation")

    # Steps 21-22 are skipped entirely when using an API-based embedder.
    if models.get("uses_api_embedder"):
        T["D_finetune"] = 0.0
        T["E_distill"] = 0.0
        _step(21, _TOTAL_STEPS, "SSL DINOv3 fine-tuning (skipped — API embedder)")
        _step(22, _TOTAL_STEPS, "Knowledge distillation (skipped — API embedder)")
        _append_agentic_step(
            agentic_trace,
            step_id="18",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no local fine-tuning checkpoint"],
            risks=["no task-specific adaptation is learned in API-embedder mode"],
            artifacts=[],
        )
        _append_agentic_step(
            agentic_trace,
            step_id="19",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no distillation artifacts"],
            risks=["no student compression path is available in API-embedder mode"],
            artifacts=[],
        )
        return {
            "ssl_gate_passed": False,
            "ft_results": [],
            "clip_dino_on_gpu": clip_dino_on_gpu,
        }

    # Step 21: SSL fine-tuning
    if device == "cuda":
        _prep_vram_for_step(
            models,
            device,
            extra_sidecars=[
                (getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL, getattr(args, "qwen_model", "") or settings.QWEN_MODEL),
                (getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL, getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL),
            ],
            label="SSL fine-tuning",
        )
        _guard_min_free_vram("SSL fine-tuning")
        clip_dino_on_gpu = False
    _step(21, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
    # Adaptive epoch count: scale up for short clips so the training sees ~200 gradient steps.
    _n_batches_per_epoch = max(1, len(frame_list) // max(1, args.batch_size))
    _ssl_epochs = max(args.epochs, min(20, (200 + _n_batches_per_epoch - 1) // _n_batches_per_epoch))
    if _ssl_epochs != args.epochs:
        _log.info(
            "  SSL adaptive epochs: %d (CLI default=%d) — %d frames, %d batches/epoch",
            _ssl_epochs, args.epochs, len(frame_list), _n_batches_per_epoch,
        )

    from ...steps.adaptation.ssl import step_ssl_finetune
    with _Timer(T, "D_finetune"):
        d = step_ssl_finetune(
            video_id,
            video_name,
            video_dir,
            frame_list,
            device,
            epochs=_ssl_epochs,
            batch_size=args.batch_size,
            tracking_results=(
                gemma_tracking_result.get("tracking_results") or []
                if not gemma_tracking_result.get("skipped")
                else []
            ),
            depth_result=depth_result,
            platform_state_fusion=platform_fusion_result,
            full_fusion_result=full_fusion_result,
            physical_state_result=physical_state_result,
        )
    stats["best_loss"] = d["best_loss"]
    stats["ckpt_mb"] = d["ckpt_mb"]
    checkpoint_path = d["checkpoint"]
    _append_agentic_step(
        agentic_trace,
        step_id="18",
        title="SSL fine-tuning",
        description="Adapt the local DINO backbone to mission-specific footage so retrieval neighborhoods reflect this video domain more closely.",
        status="ok",
        context_inputs=["frame sequence", "base DINO initialization"],
        context_outputs=[f"best loss {d['best_loss']:.4f}", "mission-adapted backbone checkpoint"],
        risks=[
            "small-video adaptation can overfit to accidental patterns",
            "temporal positives can encode wrong sameness assumptions",
            "adapted features can improve scores while harming semantics",
        ],
        artifacts=["finetune_stats.md", "checkpoints/dino_ssl_best.pt"],
    )

    # SSL gate check
    import os as _os
    _ssl_best_loss = stats.get("best_loss", float("inf"))
    ssl_gate_passed = (
        bool(checkpoint_path)
        and _os.path.exists(checkpoint_path)
        and _ssl_best_loss < _SSL_GATE_MAX_LOSS
    )
    if ssl_gate_passed:
        _log.info(
            "  [ok] SSL gate passed (best_loss=%.4f < %.1f) — proceeding to distillation, "
            "ONNX export, and search comparison",
            _ssl_best_loss, _SSL_GATE_MAX_LOSS,
        )
    else:
        _log.warning(
            "  ✗ SSL gate did not pass (checkpoint=%r, best_loss=%.4f, threshold=%.1f) — "
            "skipping steps E/F/G/H (distillation, ONNX export, search comparison)",
            checkpoint_path, _ssl_best_loss, _SSL_GATE_MAX_LOSS,
        )

    # Step 22: Knowledge distillation (Stage 1)
    student_backbone = None
    student_dim = 768
    e_distill: dict[str, Any] = {"skipped": True}
    if ssl_gate_passed and not args.no_distill:
        _cap_anchor_embs: np.ndarray | None = None
        if caption_results and models.get("clip"):
            try:
                _cap_texts = [r.get("caption") or "" for r in caption_results]
                _cap_texts = [t for t in _cap_texts if t.strip()]
                if _cap_texts:
                    _clip_model = models["clip"]
                    if hasattr(_clip_model, "encode_texts") and not (
                        _HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder)
                    ):
                        _cap_anchor_embs = _clip_model.encode_texts(_cap_texts)
                        _log.info(
                            "  Distillation caption anchors: %d CLIP text embeddings from Florence captions",
                            len(_cap_anchor_embs),
                        )
            except Exception as _exc:
                _log.debug("  Caption anchor prep failed (%s) — distilling without anchor", _exc)

        _gemma_teacher = None
        if _HAS_GEMMA and isinstance(models.get("clip"), GemmaEmbedder):
            _gemma_teacher = models["clip"]
            _log.info("  Using GemmaVisionTeacher for distillation (max hydration)")

        _step(22, _TOTAL_STEPS, "Knowledge distillation (max hydration) → ViT-S/14 student")
        with _Timer(T, "E_distill"):
            e_distill = step_distill(
                checkpoint_path,
                frame_list,
                video_name,
                video_dir,
                device,
                distill_epochs=args.distill_epochs,
                batch_size=args.batch_size,
                caption_embeddings=_cap_anchor_embs,
                gemma_embedder=_gemma_teacher,
            )
        if not e_distill["skipped"]:
            student_backbone = e_distill["student_backbone"]
            student_dim = e_distill["student_dim"]
            stats["distill_loss"] = e_distill["best_loss"]
            stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
            stats["student_dim"] = student_dim
            stats["teacher_dim"] = e_distill["teacher_dim"]
            stats["distill_compression_ratio"] = e_distill.get("compression_ratio", 0.0)
        _append_agentic_step(
            agentic_trace,
            step_id="19",
            title="Knowledge distillation",
            description="Compress teacher geometry and optional language anchors into a smaller student suitable for deployment.",
            status="skipped" if e_distill.get("skipped") else "ok",
            context_inputs=[
                "fine-tuned teacher checkpoint",
                "optional Gemma teacher alignment",
                "optional Florence caption anchors",
            ],
            context_outputs=[
                f"student dim {student_dim}",
                f"best distill loss {e_distill.get('best_loss', float('nan')):.4f}",
                "student deployment checkpoint",
            ]
            if not e_distill.get("skipped")
            else ["no distilled student"],
            risks=[
                "teacher mistakes transfer into the student representation",
                "caption anchors can inject wrong semantics into retrieval space",
                "compression can erase rare but important distinctions",
            ],
            artifacts=["distill_stats.md", "checkpoints/student_best.pt"]
            if not e_distill.get("skipped")
            else [],
        )
    else:
        T["E_distill"] = 0.0
        _gate_reason = "SSL gate did not pass" if not ssl_gate_passed else "--no-distill"
        _step(22, _TOTAL_STEPS, f"Knowledge distillation (skipped — {_gate_reason})")
        _append_agentic_step(
            agentic_trace,
            step_id="19",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["fine-tuned teacher checkpoint"],
            context_outputs=["no distilled student"],
            risks=[
                "without this step, deployment relies on the larger teacher or ONNX export only",
                "no compression audit is produced",
            ],
            artifacts=[],
        )

    # Step 23: Stage 2 distillation — ViT-S/14 → EfficientViT-B1
    e_distill_stage2: dict[str, Any] = {"skipped": True, "onnx_exported": False}
    if ssl_gate_passed and not args.no_distill and not e_distill.get("skipped"):
        _step(23, _TOTAL_STEPS, "Stage 2 distillation: ViT-S/14 → EfficientViT-B1 student")
        with _Timer(T, "E_distill_stage2"):
            e_distill_stage2 = step_distill_stage2(
                student_backbone,
                frame_list,
                video_name,
                video_dir,
                device,
                distill_epochs=args.distill_epochs,
                batch_size=args.batch_size,
            )
        if not e_distill_stage2["skipped"]:
            stats["distill_stage2_loss"] = e_distill_stage2.get("best_loss", float("nan"))
            stats["efficientvit_ckpt_mb"] = e_distill_stage2.get("ckpt_mb", 0.0)
            stats["efficientvit_onnx_mb"] = e_distill_stage2.get("onnx_mb", 0.0)
        _append_agentic_step(
            agentic_trace,
            step_id="19b",
            title="Stage 2 distillation (EfficientViT-B1)",
            description="Compress the Stage 1 ViT-S/14 student into a 384-dim EfficientViT-B1 student using RKD-D + KoLeo. Requires only ~2 GB VRAM.",
            status="skipped" if e_distill_stage2.get("skipped") else "ok",
            context_inputs=["Stage 1 ViT-S/14 student backbone", "frame paths"],
            context_outputs=[
                "EfficientViT-B1 dim=384",
                f"best_loss={e_distill_stage2.get('best_loss', float('nan')):.4f}",
                f"onnx_exported={e_distill_stage2.get('onnx_exported', False)}",
            ]
            if not e_distill_stage2.get("skipped")
            else ["no Stage 2 student"],
            risks=[
                "Stage 2 adds a second compression hop — errors from Stage 1 compound",
                "RKD-D-only may underfit when teacher/student topologies diverge",
            ],
            artifacts=["distill_stage2_stats.md", "checkpoints_stage2/student_best.pt", "edge_models/efficientvit_local.onnx"]
            if not e_distill_stage2.get("skipped")
            else [],
        )
    else:
        T["E_distill_stage2"] = 0.0
        _gate_reason_s2 = (
            "SSL gate did not pass" if not ssl_gate_passed
            else "--no-distill" if args.no_distill
            else "no Stage 1 student"
        )
        _step(23, _TOTAL_STEPS, f"Stage 2 distillation (skipped — {_gate_reason_s2})")
        _append_agentic_step(
            agentic_trace,
            step_id="19b",
            title="Stage 2 distillation (EfficientViT-B1)",
            description="Compress Stage 1 student into EfficientViT-B1 using RKD-D + KoLeo.",
            status="skipped",
            context_inputs=["Stage 1 student backbone"],
            context_outputs=["no EfficientViT student"],
            risks=["no ultra-lightweight deployment artifact produced"],
            artifacts=[],
        )

    # Step 24: ONNX export + gallery
    if ssl_gate_passed:
        if device == "cuda" and not clip_dino_on_gpu:
            _restore_models_to_gpu(models, device)
            clip_dino_on_gpu = _models_on_device(models, device)
        _step(24, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
        with _Timer(T, "F_export"):
            e = step_export_model(
                checkpoint_path,
                frame_list,
                video_dir,
                device,
                models,
                no_onnx=args.no_onnx,
                student_backbone=student_backbone,
                student_dim=student_dim,
            )
        stats["onnx_mb"] = e.get("onnx_mb", 0.0)
        stats["onnx_exported"] = e.get("exported", False)
        _append_agentic_step(
            agentic_trace,
            step_id="20",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="ok",
            context_inputs=["teacher or student backbone", "retrieval gallery frames"],
            context_outputs=[f"onnx exported={e.get('exported', False)}", "gallery.npz for edge classification"],
            risks=[
                "export mismatches can change runtime behavior versus training",
                "gallery coverage can be too narrow for field use",
                "deployment artifacts can hide upstream semantic errors behind good latency",
            ],
            artifacts=["edge_models/dino_local.onnx", "edge_models/gallery.npz"],
        )
    else:
        T["F_export"] = 0.0
        stats.setdefault("onnx_mb", 0.0)
        stats.setdefault("onnx_exported", False)
        _step(24, _TOTAL_STEPS, "ONNX export (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="20",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no deployment artifacts"],
            risks=["no edge deployment artifacts produced"],
            artifacts=[],
        )

    # Step 25: Fine-tuned search
    ft_results: list[dict] = []
    if ssl_gate_passed:
        _step(25, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
        with _Timer(T, "G_ft_search"):
            f = step_finetuned_model_search_test(
                frame_list, store, is_qdrant, models, query_frame, query_t_sec,
                video_id, video_name, video_dir, top_k=args.top_k,
            )
        ft_results = f["results"]
        stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0
        _append_agentic_step(
            agentic_trace,
            step_id="21",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="ok",
            context_inputs=["fine-tuned or distilled backbone", "same query frame as baseline"],
            context_outputs=[
                f"top-{len(ft_results)} adapted matches",
                f"top score {stats['ft_top_score']:.4f}",
            ],
            risks=[
                "score improvements can hide semantic regressions",
                "query reuse can overstate adaptation gains",
                "retrieval differences may reflect memorization rather than better context",
            ],
            artifacts=["finetuned_search.md"],
        )
    else:
        T["G_ft_search"] = 0.0
        stats.setdefault("ft_top_score", 0.0)
        _step(25, _TOTAL_STEPS, "Fine-tuned search (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="21",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no fine-tuned retrieval results"],
            risks=["no before/after retrieval comparison available"],
            artifacts=[],
        )

    # Step 26: Model comparison + description
    if ssl_gate_passed:
        _step(25, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
        with _Timer(T, "H_compare"):
            g = step_compare_and_describe(
                frame_list, store, is_qdrant, base_results, ft_results, models,
                video_id, video_name, video_dir,
                stats.get("ckpt_mb", 0.0), stats.get("onnx_mb", 0.0),
            )
        if g:
            stats["base_infer_ms"] = g.get("base_infer_ms", 0.0)
            stats["ft_infer_ms"] = g.get("ft_infer_ms", 0.0)
            stats["top_description"] = g.get("top_description", "")
            video_context["top_descriptions"] = g.get("text_descriptions", [])
        _append_agentic_step(
            agentic_trace,
            step_id="22",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="ok",
            context_inputs=["baseline and adapted retrieval outputs", "sampled frame embeddings"],
            context_outputs=[
                f"top description: {stats.get('top_description', 'unknown')}",
                "comparison summary across model variants",
            ],
            risks=[
                "top text prompt may sound plausible but be too coarse or wrong",
                "comparison metrics can privilege ranking stability over semantics",
                "narrative labels can bias the final synthesis context",
            ],
            artifacts=["comparison.md", "description.md"],
        )
    else:
        T["H_compare"] = 0.0
        _step(26, _TOTAL_STEPS, "Model comparison (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="22",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no comparison or description artifacts"],
            risks=["no adaptation quality signal produced"],
            artifacts=[],
        )

    return {
        "ssl_gate_passed": ssl_gate_passed,
        "ft_results": ft_results,
        "clip_dino_on_gpu": clip_dino_on_gpu,
    }

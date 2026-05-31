"""Distillation and ONNX export steps."""

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from selfsuvis.pipeline.training import DistillConfig, run_distillation

from .common import _log

try:
    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

try:
    _HAS_GEMMA = True
except Exception:
    _HAS_GEMMA = False


def step_distill(
    teacher_checkpoint: str,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    distill_epochs: int,
    batch_size: int,
    caption_embeddings: np.ndarray | None = None,
    gemma_embedder: Any | None = None,
) -> dict[str, Any]:
    """Step 17: distil fine-tuned teacher → student with maximum hydration.

    Maximum-hydration distillation chain:
      1. Teacher backbone: fine-tuned DINOv3 ViT-B/14 (SSL checkpoint).
         When *gemma_embedder* is provided, uses GemmaVisionTeacher instead —
         richer multimodal embeddings as the distillation target.
      2. Caption anchor loss: when *caption_embeddings* are provided (CLIP text
         embeddings of per-frame captions), adds a λ=0.5 cosine term that pulls
         the student toward language-grounded targets, transferring Gemma's
         semantic understanding into the small student.
      3. Student: DINOv2 ViT-S/14 (22M params, 384-dim).
    """
    from .report import write_distill_stats_md

    out_md = video_dir / "distill_stats.md"
    result: dict[str, Any] = {
        "student_backbone": None,
        "best_path": "",
        "best_loss": float("nan"),
        "best_recall": float("nan"),
        "compression_ratio": 0.0,
        "student_dim": 384,
        "teacher_dim": 768,
        "student_model": "dinov2_vits14",
        "ckpt_mb": 0.0,
        "skipped": False,
    }
    if not _HAS_DINO:
        _log.warning("  DINO not available — skipping distillation")
        result["skipped"] = True
        return result

    # -- Choose teacher --------------------------------------------------------
    teacher_bb = None
    teacher_label = "DINOv3 ViT-B/14 (SSL)"

    if gemma_embedder is not None and _HAS_GEMMA:
        try:
            from selfsuvis.pipeline.training.distill import GemmaVisionTeacher

            teacher_bb = GemmaVisionTeacher(gemma_embedder)
            teacher_label = f"Gemma 4 vision encoder (dim={gemma_embedder.image_dim()})"
            result["teacher_dim"] = gemma_embedder.image_dim()
            _log.info("  Distillation teacher: %s", teacher_label)
        except Exception as exc:
            _log.warning("  GemmaVisionTeacher failed (%s) — falling back to DINOv3", exc)
            teacher_bb = None

    if teacher_bb is None:
        try:
            import torch

            from selfsuvis.models.dino_model import hub_load_dino

            teacher_bb = hub_load_dino("dinov3_vitb14", pretrained=True).to(device)
            state = torch.load(teacher_checkpoint, map_location=device)
            teacher_bb.load_state_dict(state)
            teacher_bb.eval()
            _log.info("  Teacher loaded from checkpoint: %s", teacher_checkpoint)
        except Exception as exc:
            _log.warning("  Could not load teacher checkpoint (%s) — skipping distillation", exc)
            result["skipped"] = True
            return result

    # -- Caption anchor --------------------------------------------------------
    lambda_cap = 0.0
    cap_embs = None
    if caption_embeddings is not None and len(caption_embeddings) > 0:
        lambda_cap = 0.5
        cap_embs = caption_embeddings
        _log.info(
            "  Caption anchor loss enabled: λ=%.1f  anchors=%d  dim=%d",
            lambda_cap,
            len(cap_embs),
            cap_embs.shape[1],
        )

    cfg = DistillConfig(
        student_model="dinov2_vits14",
        epochs=distill_epochs,
        batch_size=batch_size,
        device=device,
        lambda_caption_anchor=lambda_cap,
        caption_embeddings=cap_embs,
    )
    frame_paths = [fp for fp, _ in frame_list]
    _log.info(
        "Starting distillation: %s → ViT-S/14  epochs=%d  frames=%d  caption_anchor=%s",
        teacher_label,
        cfg.epochs,
        len(frame_paths),
        f"λ={lambda_cap}" if lambda_cap > 0 else "off",
    )
    try:
        stats = run_distillation(teacher_bb, frame_paths, video_dir / "checkpoints", cfg)
    except Exception as exc:
        _log.warning("  Distillation failed (%s) — skipping", exc)
        result["skipped"] = True
        return result
    distiller = stats.pop("distiller")
    best_path = stats.get("best_path", "")
    if (
        not best_path
        or not os.path.exists(best_path)
        or not math.isfinite(stats.get("best_loss", float("nan")))
    ):
        _log.warning("  Distillation produced no valid student checkpoint — skipping")
        result["skipped"] = True
        return result
    result.update(stats)
    result["student_backbone"] = distiller.student_backbone()
    result["ckpt_mb"] = os.path.getsize(best_path) / 1e6
    result["teacher_label"] = teacher_label
    result["caption_anchor_used"] = lambda_cap > 0
    _log.info(
        "  [ok] Distillation complete in %.1fs | best_loss=%.4f | best_R@1=%.3f | "
        "compression=%.1f× | student=%s (dim=%d)",
        stats["elapsed"],
        stats["best_loss"],
        stats.get("best_recall", float("nan")),
        stats.get("compression_ratio", 0.0),
        stats["student_model"],
        stats["student_dim"],
    )
    write_distill_stats_md(out_md, video_name, stats)
    return result


def step_distill_stage2(
    stage1_student_backbone: Any,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    distill_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Step 22: Stage 1→2 distillation — ViT-S/14 teacher → EfficientViT-B1 student.

    Uses RKD-D + KoLeo only (no RKD-A); both teacher and student are 384-dim so
    angle-triplet loss gives no benefit.  Requires only ~2 GB VRAM.
    """
    from selfsuvis.pipeline.training.distill import run_distillation_efficientvit
    from selfsuvis.pipeline.training.edge_inference import export_efficientvit_onnx

    from .report import write_distill_stats_md

    edge_dir = video_dir / "edge_models"
    edge_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(edge_dir / "efficientvit_local.onnx")
    result: dict[str, Any] = {
        "student_backbone": None,
        "best_path": "",
        "best_loss": float("nan"),
        "best_recall": float("nan"),
        "compression_ratio": 0.0,
        "student_dim": 384,
        "teacher_dim": 384,
        "student_model": "efficientvit_b1",
        "ckpt_mb": 0.0,
        "onnx_path": "",
        "onnx_mb": 0.0,
        "onnx_exported": False,
        "skipped": False,
    }

    if stage1_student_backbone is None:
        _log.warning("  Stage 2 distillation skipped — no Stage 1 student backbone")
        result["skipped"] = True
        return result

    cfg = DistillConfig(
        student_model="efficientvit_b1",
        epochs=distill_epochs,
        batch_size=batch_size,
        device=device,
        stage=2,
        lambda_rkd_d=25.0,
        lambda_rkd_a=0.0,
        lambda_kd=1.0,
        lambda_koleo=0.1,
    )
    frame_paths = [fp for fp, _ in frame_list]
    _log.info(
        "Starting Stage 2 distillation: ViT-S/14 → EfficientViT-B1  epochs=%d  frames=%d",
        cfg.epochs,
        len(frame_paths),
    )
    try:
        stats = run_distillation_efficientvit(
            stage1_student_backbone, frame_paths, video_dir / "checkpoints_stage2", cfg
        )
    except Exception as exc:
        _log.warning("  Stage 2 distillation failed (%s) — skipping", exc)
        result["skipped"] = True
        return result

    distiller = stats.pop("distiller")
    best_path = stats.get("best_path", "")
    if (
        not best_path
        or not os.path.exists(best_path)
        or not math.isfinite(stats.get("best_loss", float("nan")))
    ):
        _log.warning("  Stage 2 distillation produced no valid checkpoint — skipping")
        result["skipped"] = True
        return result

    result.update(stats)
    result["student_backbone"] = distiller.student_backbone()
    result["ckpt_mb"] = os.path.getsize(best_path) / 1e6
    _log.info(
        "  [ok] Stage 2 complete in %.1fs | best_loss=%.4f | best_R@1=%.3f | compression=%.1f×",
        stats["elapsed"],
        stats["best_loss"],
        stats.get("best_recall", float("nan")),
        stats.get("compression_ratio", 0.0),
    )
    write_distill_stats_md(video_dir / "distill_stage2_stats.md", video_name, stats)

    try:
        export_efficientvit_onnx(result["student_backbone"], onnx_path)
        result["onnx_path"] = onnx_path
        result["onnx_mb"] = os.path.getsize(onnx_path) / 1e6
        result["onnx_exported"] = True
        _log.info("  [ok] EfficientViT ONNX: %.1f MB → %s", result["onnx_mb"], onnx_path)
    except Exception as exc:
        _log.warning("  EfficientViT ONNX export failed (%s)", exc)

    return result


def step_export_model(
    checkpoint_path: str,
    frame_list: list[tuple[str, float]],
    video_dir: Path,
    device: str,
    models: dict[str, Any],
    no_onnx: bool,
    student_backbone: Any | None = None,
    student_dim: int = 768,
) -> dict[str, Any]:
    """Step 18: export model to ONNX + build gallery.npz."""
    from selfsuvis.models.openclip_model import OpenCLIPEmbedder
    from selfsuvis.pipeline.training.edge_inference import build_gallery

    edge_dir = video_dir / "edge_models"
    edge_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(edge_dir / "dino_local.onnx")
    gallery_path = str(edge_dir / "gallery.npz")
    result: dict[str, Any] = {
        "onnx_path": onnx_path,
        "gallery_path": gallery_path,
        "onnx_mb": 0.0,
        "exported": False,
        "gallery_saved": False,
    }
    backbone_to_export = None
    if student_backbone is not None:
        backbone_to_export = student_backbone
        model_label = f"distilled student (ViT-S/14, dim={student_dim})"
    elif _HAS_DINO:
        dino = models.get("dino")
        if dino is None:
            _log.warning("  DINO not available — will use CLIP for gallery only")
        else:
            try:
                _log.info("Loading fine-tuned checkpoint: %s", checkpoint_path)
                dino.load_backbone_checkpoint(checkpoint_path)
                backbone_to_export = dino.model.eval()
                model_label = "fine-tuned teacher (ViT-B/14)"
            except Exception as exc:
                _log.warning("  Could not load checkpoint (%s) — using base DINO", exc)
                backbone_to_export = dino.model.eval()
                model_label = "base DINOv3 teacher (ViT-B/14)"
    else:
        _log.warning("  DINO not available — skipping ONNX export; will use CLIP for gallery")

    if backbone_to_export is not None and not no_onnx:
        try:
            import torch

            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = False
            # Cast to float32 — SSL fine-tuning may leave projection heads in
            # float16/bfloat16, which causes a dtype mismatch on float32 dummy input.
            backbone_cpu = backbone_to_export.cpu().float().eval()

            # Wrap in a single-input module so ONNX never captures 'masks'
            # as a required input (DINOv2 forward(x, masks=None) leaks the
            # masks node into the graph under torch.onnx tracing).
            class _SingleInputWrapper(torch.nn.Module):
                def __init__(self, bb):
                    super().__init__()
                    self.bb = bb

                def forward(self, x):
                    return self.bb(x)

            export_model = _SingleInputWrapper(backbone_cpu).eval()
            if hasattr(export_model.bb, "interpolate_antialias"):
                export_model.bb.interpolate_antialias = False
            if hasattr(export_model.bb, "interpolate_offset"):
                export_model.bb.interpolate_offset = 0.0
            # 224 matches EdgeClassifier._preprocess_image default (224×224).
            # DINOv2 accepts any multiple of patch_size=14; 224=14×16 is valid.
            dummy = torch.zeros(1, 3, 224, 224)
            _log.info("Exporting ONNX (%s) to %s …", model_label, onnx_path)
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                torch.onnx.export(
                    export_model,
                    dummy,
                    onnx_path,
                    opset_version=18,
                    input_names=["pixel_values"],
                    output_names=["embedding"],
                    do_constant_folding=True,
                    dynamo=False,
                )
            if os.path.exists(onnx_path):
                onnx_mb = os.path.getsize(onnx_path) / 1e6
                result["onnx_mb"] = onnx_mb
                result["exported"] = True
                _log.info("  [ok] ONNX export complete: %.1f MB → %s", onnx_mb, onnx_path)
            else:
                _log.warning("  ONNX export ran but file not found at %s", onnx_path)
            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = True
            backbone_to_export = backbone_to_export.to(device).eval()
        except Exception as exc:
            _log.warning("  ONNX export failed (%s) — skipping", exc)
            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = True
            if backbone_to_export is not None:
                try:
                    backbone_to_export = backbone_to_export.to(device).eval()
                except Exception:
                    pass
    elif backbone_to_export is not None:
        _log.info("  ONNX export skipped (--no-onnx)")

    _log.info("Building embedding gallery from %d frames …", len(frame_list))
    try:
        step = max(1, len(frame_list) // 200)
        sampled = [fp for fp, _ in frame_list[::step] if os.path.isfile(fp)]
        if not sampled:
            raise ValueError("No valid frame paths for gallery build")
        labels_map = {"scene": sampled}
        if result["exported"] and os.path.exists(onnx_path):
            build_gallery(labels_map=labels_map, output_path=gallery_path, onnx_path=onnx_path)
            _log.info("  Gallery built using ONNX model")
        elif backbone_to_export is not None:
            build_gallery(
                labels_map=labels_map, output_path=gallery_path, backbone=backbone_to_export
            )
            _log.info("  Gallery built using PyTorch backbone")
        else:
            clip_model: OpenCLIPEmbedder = models["clip"]
            all_embeds = []
            for fp in sampled:
                img = Image.open(fp).convert("RGB")
                emb = clip_model.encode_images([img])[0]
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                all_embeds.append(emb.astype(np.float32))
            np.savez(
                gallery_path,
                embeddings=np.stack(all_embeds, axis=0),
                labels=np.array(["scene"] * len(all_embeds), dtype=object),
                label_names=np.array(["scene"], dtype=object),
            )
            _log.info("  Gallery built using CLIP fallback")
        if os.path.exists(gallery_path):
            result["gallery_saved"] = True
            _log.info(
                "  [ok] Gallery saved: %d embeddings → %s (%.1f MB)",
                len(sampled),
                gallery_path,
                os.path.getsize(gallery_path) / 1e6,
            )
        else:
            _log.warning("  Gallery file not found after build: %s", gallery_path)
    except Exception as exc:
        _log.warning("  Gallery build failed (%s)", exc, exc_info=True)
    return result

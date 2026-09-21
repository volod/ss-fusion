"""Default step manifests wrapping the local monolith `step_*` functions."""

from ss_kernel.contracts import StepManifest

LEGACY = "python:ss_kernel.adapters.local:legacy_step"

# (manifest_id, name, skip_flags, latency_ms, vram_mb, timing_key)
_LEGACY_STEPS: tuple[tuple[str, str, tuple[str, ...], int, int, str], ...] = (
    ("extract-frames", "Frame extraction", (), 8000, 0, "A_extract"),
    ("index-vectors", "Vector store indexing", (), 4000, 512, "B_index"),
    ("gemma-analysis", "Gemma multimodal analysis", (), 15000, 0, "J_gemma"),
    ("florence-caption", "Florence-2 scene captions", (), 20000, 2048, "L_caption"),
    ("asr", "ASR transcription", ("no-asr",), 30000, 1024, "M_asr"),
    ("ocr", "OCR extraction", ("no-ocr",), 20000, 1024, "N_ocr"),
    ("depth", "Depth estimation", ("no-depth",), 25000, 2048, "O_depth"),
    ("detection", "Object detection", ("no-detection",), 20000, 2048, "P_detection"),
    ("yolo-sam", "YOLO11 + SAM", ("no-yolo", "no-sam"), 40000, 4096, "P2_yolo_sam"),
    ("gemma-tracking", "Gemma directed tracking", ("no-rfdetr",), 30000, 2048, "P3_gemma_tracking"),
    ("world-model", "World model embeddings", ("no-world-model",), 40000, 4096, "Q_world"),
    ("qwen-caption", "Qwen detailed captions", ("no-qwen",), 60000, 0, "R_qwen"),
    ("unidrive", "UniDriveVLA", ("no-unidrive",), 60000, 0, "S_unidrive"),
    ("scenetok", "SceneTok", ("no-scenetok",), 30000, 2048, "S_scenetok"),
    ("cosmos3", "Cosmos3 world model", ("no-cosmos3",), 120000, 8192, ""),
    ("base-search", "Base model search test", (), 5000, 512, "C_base_search"),
    ("map-3d", "SfM + Gaussian splat", ("no-sfm", "no-gsplat"), 90000, 4096, "I_3dmap"),
    ("physical-state", "Physical scene state", (), 2000, 0, "PS_physical_state"),
    ("field-state", "Environmental field state", (), 2000, 0, "PS_field_state"),
    ("threat-primitives", "Threat primitives", (), 2000, 0, "PS_threat_primitives"),
    ("ssl-finetune", "SSL DINOv3 fine-tune", (), 120000, 8192, "D_finetune"),
    ("distill", "Knowledge distillation", ("no-distill",), 90000, 4096, "E_distill"),
    ("distill-stage2", "Stage 2 distillation", ("no-distill",), 90000, 4096, "E_distill_stage2"),
    ("onnx-export", "ONNX export + gallery", ("no-onnx",), 20000, 0, "F_export"),
    ("ft-search", "Fine-tuned search test", (), 5000, 512, "G_ft_search"),
    ("compare", "Model comparison", (), 8000, 512, "H_compare"),
    ("multi-model-compare", "Multi-model comparison", (), 15000, 1024, "T_multimodel"),
    ("local-threat", "Local threat inference", (), 3000, 0, "PS_local_threat"),
    ("policy", "Action policy", (), 2000, 0, "PS_policy"),
    ("synthesis", "Video synthesis", (), 20000, 0, "Z_synthesis"),
    ("agentic-audit", "Agentic flow audit", (), 20000, 0, "AA_agentic"),
    (
        "drone-detection",
        "Drone detection training",
        ("no-drone-detection",),
        60000,
        4096,
        "AC_drone_detection",
    ),
    ("drone-audio", "Drone audio training", ("no-drone-audio",), 60000, 2048, "AC_drone_audio"),
    ("drau-eval", "drau range eval", ("no-drau-eval",), 15000, 0, "AC_drau_eval"),
    ("model-advisor", "Model/run advisor", (), 3000, 0, "AB_model_advisor"),
)

TIMING_KEYS: dict[str, str] = {
    manifest_id: timing for manifest_id, _, _, _, _, timing in _LEGACY_STEPS if timing
}


def default_manifests() -> dict[str, StepManifest]:
    """Manifests for the local-research steps (legacy wrap)."""
    out: dict[str, StepManifest] = {}
    for manifest_id, name, skip_flags, latency_ms, vram_mb, _timing in _LEGACY_STEPS:
        item = StepManifest.model_validate(
            {
                "manifest_id": manifest_id,
                "version": "1.0.0",
                "name": name,
                "ports_in": [{"name": "video", "type": "path"}],
                "ports_out": [{"name": "artifacts", "type": "dir"}],
                "tiers": [
                    {
                        "name": "cuda",
                        "device_class": "cuda",
                        "latency_ms": latency_ms,
                        "vram_mb": vram_mb,
                        "quality": 1.0,
                    },
                    {
                        "name": "cpu",
                        "device_class": "cpu",
                        "latency_ms": latency_ms * 3,
                        "vram_mb": 0,
                        "quality": 0.8,
                    },
                ],
                "binding": LEGACY,
                "skip_flags": list(skip_flags),
                "wrap": "legacy",
            }
        )
        out[manifest_id] = item
    return out

"""Multimodal model registry with GPU/RAM auto-detection and top-10 model catalogs.

Each task has a ranked list of models ordered by size (small → large). The
``auto_select()`` function picks the largest model that fits in available VRAM
(plus a safety margin). Users can override any selection via env vars or CLI.

Usage::

    from selfsuvis.pipeline.vision.registry import auto_select, CATALOGS, detect_resources

    resources = detect_resources()   # {"vram_gb": 15.8, "ram_gb": 128.0}
    model_id = auto_select("asr", resources)
    # → "openai/whisper-large-v3-turbo"
"""

import os
import subprocess
from dataclasses import dataclass, field

from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)


# -- Resource detection --------------------------------------------------------


def _env_float_override(key: str) -> float | None:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float override %s=%r ignored", key, raw)
        return None


def detect_vram_gb() -> float:
    """Return total GPU VRAM in GiB. Returns 0.0 if no GPU found."""
    override = _env_float_override("GPU_TOTAL_GB_HINT")
    if override is not None:
        return override
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mib = int(result.stdout.strip().splitlines()[0].strip())
            return mib / 1024.0
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


def detect_free_vram_gb() -> float:
    """Return currently free GPU VRAM in GiB. Returns 0.0 if no GPU found."""
    override = _env_float_override("GPU_FREE_GB_HINT")
    if override is not None:
        return override
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mib = int(result.stdout.strip().splitlines()[0].strip())
            return mib / 1024.0
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.cuda, "mem_get_info"):
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            return free_bytes / (1024**3)
    except Exception:
        pass
    return 0.0


def detect_ram_gb() -> float:
    """Return total system RAM in GiB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass
    return 8.0  # conservative fallback


def detect_resources() -> dict[str, float]:
    """Return dict with GPU total/free VRAM and system RAM."""
    vram = detect_vram_gb()
    free_vram = detect_free_vram_gb()
    ram = detect_ram_gb()
    logger.debug(
        "Detected resources: VRAM(total)=%.1f GB  VRAM(free)=%.1f GB  RAM=%.1f GB",
        vram,
        free_vram,
        ram,
    )
    return {"vram_gb": vram, "free_vram_gb": free_vram, "ram_gb": ram}


# -- Model entry ---------------------------------------------------------------


@dataclass
class ModelEntry:
    """A single model in the registry catalog."""

    model_id: str  # HuggingFace model ID
    params_b: float  # Parameter count in billions
    vram_fp16_gb: float  # Approx VRAM required at FP16 (GiB)
    description: str  # One-line capability description
    supports_video: bool = False
    # Extra per-entry kwargs forwarded to the model loader (e.g. quantization hints)
    extra: dict = field(default_factory=dict)


# -- Top-10 catalogs per task --------------------------------------------------
# Each list is ordered small → large. ``auto_select`` picks the largest that fits.

CATALOGS: dict[str, list[ModelEntry]] = {
    # -- Automatic Speech Recognition -----------------------------------------
    # Note: current ASR models top out around 1.55B params. No 3B+ ASR models
    # exist as of 2026-Q1; Whisper family dominates.
    "asr": [
        ModelEntry(
            "openai/whisper-tiny",
            0.039,
            0.1,
            "Whisper tiny — fastest, English-focused, ~32× real-time",
        ),
        ModelEntry(
            "openai/whisper-base",
            0.074,
            0.2,
            "Whisper base — good quality/speed balance, 99 languages",
        ),
        ModelEntry(
            "openai/whisper-small", 0.244, 0.5, "Whisper small — strong multilingual, ~6× real-time"
        ),
        ModelEntry(
            "openai/whisper-medium",
            0.769,
            1.5,
            "Whisper medium — near large-v2 quality, lower VRAM",
        ),
        ModelEntry(
            "distil-whisper/distil-large-v3",
            0.756,
            1.5,
            "Distil-Whisper large-v3 — 6× faster than large-v3, same WER",
        ),
        ModelEntry(
            "openai/whisper-large-v3-turbo",
            0.809,
            1.6,
            "Whisper large-v3-turbo — pruned decoder, 8× speedup vs large-v3",
        ),
        ModelEntry(
            "openai/whisper-large-v2",
            1.55,
            3.0,
            "Whisper large-v2 — best pre-v3 accuracy, 99 languages",
        ),
        ModelEntry(
            "openai/whisper-large-v3",
            1.55,
            3.0,
            "Whisper large-v3 — best accuracy, handles accented speech well",
        ),
        ModelEntry(
            "nvidia/canary-1b",
            1.0,
            2.0,
            "NVIDIA Canary-1B — CTC+AED, punctuation/capitalisation output",
        ),
        ModelEntry(
            "facebook/seamless-m4t-v2-large",
            2.3,
            4.6,
            "SeamlessM4T-v2-large — speech-to-speech/text, 100+ languages",
        ),
    ],
    # -- OCR / Document Understanding -----------------------------------------
    "ocr": [
        ModelEntry(
            "microsoft/trocr-base-printed",
            0.334,
            0.7,
            "TrOCR base — printed document OCR, fast CPU-runnable",
        ),
        ModelEntry(
            "microsoft/trocr-large-printed",
            0.558,
            1.2,
            "TrOCR large — better accuracy on complex printed text",
        ),
        ModelEntry(
            "ucaslcl/GOT-OCR2_0",
            0.580,
            1.2,
            "GOT-OCR2 — scene text, tables, formulas, multi-page docs",
        ),
        ModelEntry(
            "microsoft/Florence-2-base",
            0.230,
            0.5,
            "Florence-2 base — already in pipeline, handles OCR tasks",
        ),
        ModelEntry(
            "microsoft/Florence-2-large",
            0.770,
            1.5,
            "Florence-2 large — already in pipeline, caption+OCR",
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            3.0,
            6.0,
            "Qwen2.5-VL-3B — strong OCR with spatial understanding",
        ),
        ModelEntry(
            "deepseek-ai/DeepSeek-OCR-2",
            3.0,
            6.8,
            "DeepSeek-OCR-2 — DeepEncoder V2, human-like reading order; "
            "tables, mixed layouts, complex documents",
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            7.0,
            14.0,
            "Qwen2.5-VL-7B — already in pipeline; top OCR quality",
        ),
        ModelEntry(
            "microsoft/Phi-3.5-vision-instruct",
            4.2,
            8.5,
            "Phi-3.5-Vision — 128K context, strong doc understanding",
        ),
        ModelEntry(
            "llava-hf/llava-1.5-13b-hf",
            13.0,
            26.0,
            "LLaVA-1.5-13B — strong VLM with OCR capabilities",
        ),
    ],
    # -- Depth Estimation -----------------------------------------------------
    "depth": [
        ModelEntry(
            "depth-anything/Depth-Anything-V2-Small-hf",
            0.025,
            0.05,
            "DepthAnything-V2-Small — 24M params, very fast, good quality",
        ),
        ModelEntry(
            "depth-anything/Depth-Anything-V2-Base-hf",
            0.097,
            0.2,
            "DepthAnything-V2-Base — 97M params, strong indoor+outdoor",
        ),
        ModelEntry(
            "Intel/dpt-large",
            0.307,
            0.6,
            "DPT-Large — dense prediction transformer, MiDaS backbone",
        ),
        ModelEntry(
            "vinvino02/glpn-kitti",
            0.085,
            0.2,
            "GLPN-KITTI — global-local path network, outdoor scenes",
        ),
        ModelEntry(
            "depth-anything/Depth-Anything-V2-Large-hf",
            0.335,
            0.7,
            "DepthAnything-V2-Large — best quality, 335M params",
        ),
        ModelEntry(
            "LiheYoung/depth-anything-large-hf",
            0.335,
            0.7,
            "DepthAnything V1-Large — predecessor, still strong",
        ),
        ModelEntry(
            "prs-eth/marigold-lcm-v1-0",
            0.859,
            1.7,
            "Marigold-LCM — diffusion-based, photorealistic depth",
        ),
        ModelEntry(
            "tencent/DepthCrafter",
            2.0,
            8.0,
            "DepthCrafter — temporally-consistent depth for open-world video; "
            "CVPR 2025 Highlight; only video-native depth model",
            supports_video=True,
        ),
        ModelEntry(
            "apple/DepthPro-hf",
            1.1,
            2.2,
            "Apple DepthPro — metric depth + focal estimation, sharp edges",
        ),
    ],
    # -- Object Detection -----------------------------------------------------
    "detection": [
        ModelEntry(
            "facebook/detr-resnet-50",
            0.041,
            0.1,
            "DETR-ResNet50 — classic transformer detection, COCO 42 mAP",
        ),
        ModelEntry(
            "PekingU/rtdetr_r50vd",
            0.042,
            0.1,
            "RT-DETR-R50 — real-time DETR, 53.1 mAP @ 108 FPS on T4",
        ),
        ModelEntry("PekingU/rtdetr_r101vd", 0.076, 0.2, "RT-DETR-R101 — larger backbone, 54.3 mAP"),
        ModelEntry(
            "omlab/omdet-turbo-swin-tiny-hf",
            0.108,
            0.3,
            "OmDet-Turbo-Tiny — open-vocabulary zero-shot detection",
        ),
        ModelEntry(
            "omlab/omdet-turbo-swin-large-hf",
            0.218,
            0.5,
            "OmDet-Turbo-Large — open-vocabulary, best speed/accuracy",
        ),
        ModelEntry(
            "IDEA-Research/grounding-dino-tiny",
            0.173,
            0.4,
            "Grounding DINO Tiny — open-vocab, text-guided detection",
        ),
        ModelEntry(
            "IDEA-Research/grounding-dino-base",
            0.341,
            0.7,
            "Grounding DINO Base — open-vocab, stronger accuracy",
        ),
        ModelEntry(
            "microsoft/conditional-detr-resnet-101",
            0.062,
            0.2,
            "Conditional DETR-R101 — faster convergence than DETR",
        ),
        ModelEntry(
            "jozhang97/deta-swin-large",
            0.218,
            0.5,
            "DETA-Swin-Large — anchor-based DETR, 63.5 COCO AP",
        ),
        ModelEntry(
            "SenseTime/deformable-detr",
            0.040,
            0.1,
            "Deformable-DETR — sparse attention, fast convergence",
        ),
    ],
    # -- Image Segmentation ---------------------------------------------------
    "segmentation": [
        ModelEntry(
            "facebook/sam2-hiera-tiny",
            0.038,
            0.1,
            "SAM2-Tiny — fastest SAM2, video + image, interactive",
            supports_video=True,
        ),
        ModelEntry(
            "facebook/sam2-hiera-small",
            0.046,
            0.1,
            "SAM2-Small — good quality, real-time capable",
            supports_video=True,
        ),
        ModelEntry(
            "facebook/sam-vit-base", 0.093, 0.2, "SAM-Base — Segment Anything, prompt-based"
        ),
        ModelEntry("facebook/sam-vit-large", 0.308, 0.6, "SAM-Large — strong edge quality"),
        ModelEntry(
            "CIDAS/clipseg-rd64-refined", 0.071, 0.2, "CLIPSeg — text + click guided segmentation"
        ),
        ModelEntry(
            "nvidia/segformer-b5-finetuned-ade-512-512",
            0.085,
            0.2,
            "SegFormer-B5 — semantic segmentation, ADE20k 84.0 mIoU",
        ),
        ModelEntry("facebook/sam-vit-huge", 0.641, 1.3, "SAM-Huge — best SAM1 quality"),
        ModelEntry(
            "facebook/sam2-hiera-large",
            0.224,
            0.5,
            "SAM2-Large — best SAM2 quality, video tracking",
            supports_video=True,
        ),
        ModelEntry(
            "shi-labs/oneformer_coco_swin_large",
            0.219,
            0.5,
            "OneFormer — panoptic + semantic + instance, single model",
        ),
        ModelEntry(
            "openmmlab/mask2former-swin-large-coco-panoptic",
            0.216,
            0.5,
            "Mask2Former-Large — state-of-art panoptic segmentation",
        ),
    ],
    # -- Visual Question Answering / Vision-Language Models -------------------
    "vqa": [
        ModelEntry(
            "microsoft/Florence-2-base",
            0.230,
            0.5,
            "Florence-2-Base — already in pipeline, fast VQA",
        ),
        ModelEntry(
            "allenai/MolmoE-1B-0924",
            1.0,
            2.0,
            "MolmoE-1B — mixture-of-experts VLM, strong pointing",
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-3B-Instruct", 3.0, 6.0, "Qwen2.5-VL-3B — compact, strong grounding+OCR"
        ),
        ModelEntry(
            "llava-hf/llava-1.5-7b-hf", 7.0, 14.0, "LLaVA-1.5-7B — instruction-following VLM"
        ),
        ModelEntry(
            "allenai/Molmo-7B-D-0924", 7.0, 14.0, "Molmo-7B-D — strong spatial reasoning + pointing"
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            7.0,
            14.0,
            "Qwen2.5-VL-7B — already in pipeline as Phase 2 engine",
        ),
        ModelEntry(
            "microsoft/Phi-3.5-vision-instruct",
            4.2,
            8.5,
            "Phi-3.5-Vision — 128K context, strong reasoning",
        ),
        ModelEntry(
            "llava-hf/llava-1.5-13b-hf", 13.0, 26.0, "LLaVA-1.5-13B — best open LLaVA quality"
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            32.0,
            64.0,
            "Qwen2.5-VL-32B — near GPT-4V quality, needs A100",
        ),
        ModelEntry(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            72.0,
            144.0,
            "Qwen2.5-VL-72B — top open-source VLM, needs 2×A100",
        ),
    ],
    # -- Zero-shot Classification (CLIP / SigLIP) -----------------------------
    "zero_shot_classification": [
        ModelEntry(
            "openai/clip-vit-base-patch32",
            0.151,
            0.3,
            "CLIP ViT-B/32 — fastest OpenAI CLIP, already in pipeline",
        ),
        ModelEntry(
            "openai/clip-vit-base-patch16",
            0.151,
            0.3,
            "CLIP ViT-B/16 — finer patches, better spatial understanding",
        ),
        ModelEntry(
            "google/siglip-base-patch16-224",
            0.200,
            0.4,
            "SigLIP-Base — sigmoid loss, better than CLIP on zero-shot",
        ),
        ModelEntry(
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            0.151,
            0.3,
            "CLIP ViT-B/32 on LAION-2B — improved open-world coverage",
        ),
        ModelEntry(
            "openai/clip-vit-large-patch14",
            0.428,
            0.9,
            "CLIP ViT-L/14 — strong baseline, 23M+ downloads",
        ),
        ModelEntry(
            "openai/clip-vit-large-patch14-336",
            0.428,
            0.9,
            "CLIP ViT-L/14 @336px — higher resolution input",
        ),
        ModelEntry(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            0.986,
            2.0,
            "CLIP ViT-H/14 on LAION-2B — largest standard CLIP",
        ),
        ModelEntry(
            "google/siglip-so400m-patch14-384",
            0.878,
            1.8,
            "SigLIP-SO400M — 400M patch encoder, top zero-shot accuracy",
        ),
        ModelEntry(
            "google/siglip2-so400m-patch14-384",
            0.878,
            1.8,
            "SigLIP2-SO400M — improved version with better calibration",
        ),
        ModelEntry(
            "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
            1.366,
            2.7,
            "CLIP ViT-g/14 on LAION-2B — largest available CLIP variant",
        ),
    ],
    # -- Video Understanding / World Models -----------------------------------
    # Self-supervised and supervised video representation models. These produce
    # temporal embeddings suitable for scene understanding, anomaly detection,
    # and change detection in mission video.
    #
    # LeWorldModel (arxiv 2603.19312, March 2026) — Joint-Embedding Predictive
    # Architecture, ~15M params, trains end-to-end from pixels. No HuggingFace
    # ID as of 2026-Q1; set WORLD_MODEL=<hf_id> once released publicly.
    # Its latent space encodes physical quantities and detects physically
    # implausible events — useful as a lightweight anomaly scorer.
    "world_model": [
        ModelEntry(
            "google/vivit-b-16x2-kinetics400",
            0.086,
            0.2,
            "ViViT-B — pure-transformer video classification, Kinetics-400",
            supports_video=True,
        ),
        ModelEntry(
            "facebook/timesformer-base-finetuned-k400",
            0.122,
            0.3,
            "TimeSformer-Base — divided space-time attention, Kinetics-400",
            supports_video=True,
        ),
        ModelEntry(
            "MCG-NJU/videomae-base",
            0.122,
            0.3,
            "VideoMAE-Base — masked autoencoder pretraining on video",
            supports_video=True,
        ),
        ModelEntry(
            "google/videoprism-base-f16r288",
            0.300,
            0.7,
            "VideoPrism-Base — Google dual-encoder video-text model",
            supports_video=True,
        ),
        ModelEntry(
            "MCG-NJU/videomae-large",
            0.307,
            0.6,
            "VideoMAE-Large — larger backbone, stronger video features",
            supports_video=True,
        ),
        ModelEntry(
            "facebook/vjepa2-vitl-fpc64-256",
            0.307,
            0.7,
            "V-JEPA2-ViT-L — Facebook self-supervised video JEPA, 64 frames",
            supports_video=True,
        ),
        ModelEntry(
            "OpenGVLab/VideoMAEv2-Huge",
            0.600,
            1.3,
            "VideoMAE-v2-Huge — stronger v2 pretraining, 600M params",
            supports_video=True,
        ),
        ModelEntry(
            "facebook/vjepa2-vitg-fpc64-256",
            1.0,
            2.0,
            "V-JEPA2-ViT-G — largest Facebook JEPA, best video representations",
            supports_video=True,
        ),
        ModelEntry(
            "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
            1.0,
            2.0,
            "InternVideo2-1B — video-language model, 224px/4fps",
            supports_video=True,
        ),
        ModelEntry(
            "nvidia/Cosmos-1.0-Autoregressive-4B",
            4.0,
            8.0,
            "Cosmos-1.0-4B — NVIDIA physical world model, autoregressive; "
            "gated repo — accept license at huggingface.co/nvidia/Cosmos-1.0-Autoregressive-4B",
            supports_video=True,
        ),
    ],
}

# VRAM safety margin: never use the last N GB to leave headroom for other workers.
_VRAM_SAFETY_MARGIN_GB = 2.0
_TASK_RUNTIME_FALLBACKS: dict[str, dict[str, str]] = {
    "world_model": {
        "nvidia/Cosmos-1.0-": "MCG-NJU/videomae-base",
        "facebook/vjepa2-": "MCG-NJU/videomae-base",
        # ViViT feature extractor is unrecognized in transformers >= 4.50;
        # fall back to VideoMAE-Base which uses AutoImageProcessor correctly.
        "google/vivit-": "MCG-NJU/videomae-base",
        # InternVideo2 and VideoMAEv2 use a custom model_type not in transformers'
        # model registry; AutoModel.from_pretrained raises ValueError at load time.
        "OpenGVLab/InternVideo2-": "MCG-NJU/videomae-base",
        "OpenGVLab/VideoMAEv2-": "MCG-NJU/videomae-base",
    }
}


# -- Auto-selection ------------------------------------------------------------


def normalize_model_id(task: str, model_id: str) -> str:
    """Return a runtime-supported model ID for a task."""
    task_rules = _TASK_RUNTIME_FALLBACKS.get(task, {})
    for prefix, fallback in task_rules.items():
        if model_id.startswith(prefix):
            return fallback
    return model_id


def auto_select(
    task: str,
    resources: dict[str, float] | None = None,
    *,
    prefer_video: bool = False,
) -> str | None:
    """Return the largest model for ``task`` that fits in available VRAM.

    Falls back to the smallest model if nothing fits.
    Returns None if the task has no catalog.
    """
    catalog = CATALOGS.get(task)
    if not catalog:
        logger.warning("No model catalog for task %r", task)
        return None

    if resources is None:
        resources = detect_resources()

    total_vram = resources.get("vram_gb", 0.0)
    free_vram = resources.get("free_vram_gb", total_vram)
    available_vram = max(0.0, free_vram - _VRAM_SAFETY_MARGIN_GB)
    # If no GPU at all, models must run on CPU — only allow very small models.
    cpu_only = total_vram < 0.5
    cpu_vram_limit = 0.5  # only <0.5 GB "VRAM" = model loaded in CPU RAM

    candidates = [
        entry for entry in catalog if normalize_model_id(task, entry.model_id) == entry.model_id
    ] or catalog
    if prefer_video:
        video_candidates = [m for m in catalog if m.supports_video]
        if video_candidates:
            filtered_video_candidates = [
                entry
                for entry in video_candidates
                if normalize_model_id(task, entry.model_id) == entry.model_id
            ]
            candidates = filtered_video_candidates or video_candidates

    selected = candidates[0]  # default: smallest
    for entry in candidates:
        required = entry.vram_fp16_gb
        if cpu_only:
            if required <= cpu_vram_limit:
                selected = entry
        else:
            if required <= available_vram:
                selected = entry

    logger.info(
        "auto_select task=%s → %s (%.1fB params, %.1f GB VRAM; free=%.1f GB, usable=%.1f GB)",
        task,
        selected.model_id,
        selected.params_b,
        selected.vram_fp16_gb,
        free_vram,
        available_vram,
    )
    return selected.model_id


def get_entry(task: str, model_id: str) -> ModelEntry | None:
    """Look up a ModelEntry by task + exact model_id."""
    for entry in CATALOGS.get(task, []):
        if entry.model_id == model_id:
            return entry
    return None


def list_models(task: str) -> list[ModelEntry]:
    """Return the catalog for a task, or empty list."""
    return CATALOGS.get(task, [])


def resolve_model_id(setting_value: str, task: str, fallback: str) -> str:
    """Resolve a model ID from a settings value, using GPU-aware auto-selection.

    This is the standard helper used by every vision model wrapper
    (``asr``, ``ocr``, ``depth``, ``detection``, ``world_model``) to avoid
    duplicating the same four-line pattern:

    - If *setting_value* is non-empty and not ``"auto"``, return it unchanged.
    - Otherwise call :func:`auto_select` with the current hardware resources.
    - If ``auto_select`` returns ``None`` (no catalog entry fits), return *fallback*.

    Args:
        setting_value: Raw value from ``settings.<TASK>_MODEL`` (may be ``"auto"``).
        task: Registry task key (e.g. ``"asr"``, ``"depth"``).
        fallback: HuggingFace model ID to use when auto-selection yields nothing.
    """
    cfg = setting_value.strip()
    if cfg and cfg.lower() != "auto":
        return cfg
    return auto_select(task, detect_resources()) or fallback

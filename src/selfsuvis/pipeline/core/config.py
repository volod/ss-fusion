import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

def _load_settings_env() -> None:
    """Load packaged env defaults first, then let project-root .env override them."""
    env_name = os.getenv("APP_ENV", "dev")
    package_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[4]
    env_file = package_root / "env" / f"{env_name}.env"
    root_env = repo_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
    if root_env.exists():
        load_dotenv(root_env, override=True)


_load_settings_env()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def mask_secret(value: str, visible_suffix: int = 4) -> str:
    """Return *value* with all but the last *visible_suffix* chars replaced by '*'.

    Safe to pass to any logger. Examples::

        mask_secret("hf_notarealtoken")  → "****...VXER"
        mask_secret("")                                          → "<not set>"
        mask_secret("short")                                     → "****t"
    """
    if not value:
        return "<not set>"
    if len(value) <= visible_suffix:
        return "*" * (len(value) - 1) + value[-1]
    return "*" * (len(value) - visible_suffix) + value[-visible_suffix:]


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _parse_allowed_paths(val: Optional[str]) -> List[str]:
    """Parse ALLOWED_INDEX_PATHS as comma-separated list. Empty means no restriction."""
    if val is None or not val.strip():
        return []
    return [p.strip() for p in val.split(",") if p.strip()]


def _env_json_dict(key: str, default: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Parse a JSON object from env, returning a safe default on invalid values."""
    fallback = default or {}
    raw = os.getenv(key, "")
    if not raw:
        return dict(fallback)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s contains invalid JSON; using default value", key)
        return dict(fallback)
    if not isinstance(parsed, dict):
        logger.warning("%s must be a JSON object; using default value", key)
        return dict(fallback)
    return parsed


def get_dino_model_name(model_name: str) -> Optional[str]:
    """Resolve configured model family to a concrete DINO backbone name."""
    if model_name == "dinov2":
        return "dinov2_vitb14"
    if model_name == "dinov3":
        return "dinov3_vitb14"
    return None


class Settings:
    DATA_DIR = _env("DATA_DIR", "./data")
    FRAMES_DIR = _env("FRAMES_DIR", os.path.join(DATA_DIR, "frames"))
    TILES_DIR = _env("TILES_DIR", os.path.join(DATA_DIR, "tiles"))
    VIDEOS_DIR = _env("VIDEOS_DIR", os.path.join(DATA_DIR, "videos"))

    # HuggingFace authentication — required for gated models (Gemma, Llama, …).
    # Set HF_TOKEN=hf_... in .env or export it before running.
    # Alternatively run: huggingface-cli login
    HF_TOKEN = _env("HF_TOKEN", _env("HUGGING_FACE_HUB_TOKEN", ""))

    MODEL_NAME = _env("MODEL_NAME", "openclip")
    OPENCLIP_MODEL = _env("OPENCLIP_MODEL", "ViT-B-16")
    OPENCLIP_PRETRAINED = _env("OPENCLIP_PRETRAINED", "openai")

    # Gemma open-weight local embedder (MODEL_NAME=gemma)
    # gemma-3-4b-it: ~8 GiB BF16, multimodal (vision + text) — recommended default.
    # Smaller text-only option: gemma-3-1b-it (~2 GiB, no vision → image captions fall back to text prompt).
    # Larger option: gemma-3-12b-it (~24 GiB, requires offloading other models).
    # All require HF_TOKEN + accepting license at huggingface.co/google/gemma-3-4b-it
    GEMMA_MODEL_ID = _env("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    GEMMA_USE_BF16 = _env("GEMMA_USE_BF16", "true").lower() == "true"

    # Gemma sidecar API (Ollama or vLLM serving Gemma for generative video analysis)
    # Set GEMMA_API_URL to enable: http://localhost:11434/v1 (Ollama) or http://localhost:8000/v1 (vLLM)
    # Leave empty to use local GemmaEmbedder only (no generative analysis).
    GEMMA_API_URL = _env("GEMMA_API_URL", "")
    GEMMA_API_BACKEND = _env("GEMMA_API_BACKEND", "ollama")  # "ollama" or "vllm"
    # Default model: Ollama uses its own tag format; vLLM uses HF repo IDs.
    # For maximum size on mixed GPU+CPU hardware, use the largest model your VRAM+RAM allows.
    # Ollama offloads layers to RAM automatically when VRAM is insufficient.
    # gemma4:e4b is the edge-optimised 4B (~3 GiB Q4) — fits on 16 GiB GPUs alongside
    # CLIP+pipeline models.  Use gemma4:26b / gemma4:31b only if VRAM allows.
    _gemma_api_model_default = (
        "gemma4:e4b"
        if _env("GEMMA_API_BACKEND", "ollama").lower() == "ollama"
        or "11434" in _env("GEMMA_API_URL", "")
        else "google/gemma-4-4b-it"
    )
    GEMMA_API_MODEL = _env("GEMMA_API_MODEL", _gemma_api_model_default)
    GEMMA_API_TIMEOUT_SEC = _env_int("GEMMA_API_TIMEOUT_SEC", 60)
    GEMMA_ANALYSIS_MAX_SAMPLE_FRAMES = _env_int("GEMMA_ANALYSIS_MAX_SAMPLE_FRAMES", 30)
    GEMMA_TRACKING_MAX_SAMPLE_FRAMES = _env_int("GEMMA_TRACKING_MAX_SAMPLE_FRAMES", 12)
    GEMMA_MIN_SAMPLE_FRAMES = _env_int("GEMMA_MIN_SAMPLE_FRAMES", 6)
    GEMMA_STABLE_FRAME_DIFF_THRESHOLD = _env_float("GEMMA_STABLE_FRAME_DIFF_THRESHOLD", 0.025)
    GEMMA_CACHE_RESPONSES = _env("GEMMA_CACHE_RESPONSES", "true").lower() == "true"
    GEMMA_SLOW_CALL_SEC = _env_float("GEMMA_SLOW_CALL_SEC", 8.0)
    # Max frames captioned per mission via Gemma; ranked by histogram-diff quality score.
    # Remaining frames fall back to Florence.  0 = caption all frames via Gemma.
    GEMMA_MAX_CAPTION_FRAMES = _env_int("GEMMA_MAX_CAPTION_FRAMES", 200)
    # Chunk size and retry for async Gemma captioning.
    GEMMA_CAPTION_CHUNK_SIZE = _env_int("GEMMA_CAPTION_CHUNK_SIZE", 3)
    GEMMA_CAPTION_RETRIES = _env_int("GEMMA_CAPTION_RETRIES", 1)
    # Final reasoning/audit sidecar. Defaults to the same endpoint as Gemma
    # analysis, but can use a larger long-thinking model for the last step.
    REASONING_API_URL = _env("REASONING_API_URL", GEMMA_API_URL)
    REASONING_BACKEND = _env("REASONING_BACKEND", GEMMA_API_BACKEND)
    REASONING_MODEL = _env("REASONING_MODEL", "")
    REASONING_TIMEOUT_SEC = _env_int("REASONING_TIMEOUT_SEC", 240)

    SAM_MODEL_TYPE = _env("SAM_MODEL_TYPE", "vit_h")
    SAM_CHECKPOINT = _env("SAM_CHECKPOINT", "")
    # ── YOLO11 detection (ultralytics) ────────────────────────────────────────
    # Enabled separately from the HF DetectionModel (DETECTION_ENABLED).
    # Default model: yolo11l.pt (~48 MB, 25.3 M params, 53.4 COCO mAP50-95).
    # Available tiers: yolo11n (6 MB) · yolo11s (18 MB) · yolo11m (38 MB)
    #                  yolo11l (48 MB) · yolo11x (109 MB)
    YOLO_ENABLED = _env("YOLO_ENABLED", "true").lower() == "true"
    YOLO_MODEL = _env("YOLO_MODEL", "yolo11l")
    YOLO_CONFIDENCE = _env_float("YOLO_CONFIDENCE", 0.25)
    # ── YOLO Semantic Scene Graph (SSG) ──────────────────────────────────────
    # Builds an observation-centric 3D semantic environment graph from YOLO
    # detections anchored to frame poses (ENU in production, SfM/PCA in local mode).
    YOLO_SSG_ENABLED = _env("YOLO_SSG_ENABLED", "true").lower() == "true"
    YOLO_SSG_MIN_OBSERVATIONS = _env_int("YOLO_SSG_MIN_OBSERVATIONS", 1)
    YOLO_SSG_CLUSTER_RADIUS_METERS = _env_float("YOLO_SSG_CLUSTER_RADIUS_METERS", 12.0)
    YOLO_SSG_NEAR_EDGE_RADIUS_METERS = _env_float("YOLO_SSG_NEAR_EDGE_RADIUS_METERS", 20.0)
    YOLO_SSG_CLUSTER_RADIUS_PCA = _env_float("YOLO_SSG_CLUSTER_RADIUS_PCA", 0.85)
    YOLO_SSG_NEAR_EDGE_RADIUS_PCA = _env_float("YOLO_SSG_NEAR_EDGE_RADIUS_PCA", 1.5)
    # ── SAM2 / SAM3 segmentation ──────────────────────────────────────────────
    # When enabled, each YOLO bounding box is refined with a SAM mask.
    # SAM_MODEL: "auto" (tries sam3 → sam2 → segment-anything) | "sam2" | "sam3" | "sam1"
    SAM_ENABLED = _env("SAM_ENABLED", "true").lower() == "true"
    SAM_MODEL = _env("SAM_MODEL", "auto")
    # ── RF-DETR directed tracking (step P3) ──────────────────────────────────
    # Gemma 4 directed tracking: Gemma understands the scene → directs SAM to
    # segment named objects → RF-DETR tracks them across frames.
    # Requires GEMMA_API_URL to be set; silently skipped otherwise.
    # RFDETR_MODEL: "base" (RFDETRBase, faster) | "large" (RFDETRLarge, higher accuracy)
    RFDETR_ENABLED = _env("RFDETR_ENABLED", "true").lower() == "true"
    RFDETR_MODEL = _env("RFDETR_MODEL", "base")
    RFDETR_CONFIDENCE = _env_float("RFDETR_CONFIDENCE", 0.35)
    LABELS_FILE = _env("LABELS_FILE", os.path.join(DATA_DIR, "labels", "openclip_rich.txt"))

    DEVICE = _env("DEVICE", "auto")
    USE_FP16 = _env("USE_FP16", "true").lower() == "true"
    LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB = _env_float("LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB", 6.0)

    SAMPLE_FPS_BASE = float(_env("SAMPLE_FPS_BASE", "2"))
    SAMPLE_FPS_MIN = float(_env("SAMPLE_FPS_MIN", "0.5"))
    SAMPLE_FPS_MAX = float(_env("SAMPLE_FPS_MAX", "5"))

    HIST_THRESH = float(_env("HIST_THRESH", "0.25"))
    EMBED_DRIFT_THRESH = float(_env("EMBED_DRIFT_THRESH", "0.15"))
    MAX_GAP_SEC = float(_env("MAX_GAP_SEC", "10"))

    MOTION_LOW = float(_env("MOTION_LOW", "0.02"))
    MOTION_HIGH = float(_env("MOTION_HIGH", "0.08"))

    STAB_ENABLE = _env("STAB_ENABLE", "true").lower() == "true"
    STAB_SIZE = int(_env("STAB_SIZE", "64"))
    PHASECORR_MIN_RESPONSE = float(_env("PHASECORR_MIN_RESPONSE", "0.15"))
    STAB_MAX_SHIFT = float(_env("STAB_MAX_SHIFT", "12"))

    TILE_SIZE = int(_env("TILE_SIZE", "384"))
    STRIDE = int(_env("STRIDE", "256"))

    BLUR_LAPL_VAR_MIN_FRAME = float(_env("BLUR_LAPL_VAR_MIN_FRAME", "80"))
    BLUR_LAPL_VAR_MIN_TILE = float(_env("BLUR_LAPL_VAR_MIN_TILE", "60"))
    MEAN_INTENSITY_MIN = float(_env("MEAN_INTENSITY_MIN", "20"))
    MEAN_INTENSITY_MAX = float(_env("MEAN_INTENSITY_MAX", "235"))

    SKY_BLUE_RATIO_MAX = float(_env("SKY_BLUE_RATIO_MAX", "0.35"))
    EDGE_DENSITY_MIN = float(_env("EDGE_DENSITY_MIN", "0.02"))

    TILE_STD_MIN = float(_env("TILE_STD_MIN", "12"))
    TILE_ENTROPY_MIN = float(_env("TILE_ENTROPY_MIN", "3.5"))

    CELL_SIZE = int(_env("CELL_SIZE", str(STRIDE)))
    CELL_WINDOW_SEC = float(_env("CELL_WINDOW_SEC", "5"))

    PHASH_LRU_SIZE = int(_env("PHASH_LRU_SIZE", "50000"))
    PHASH_HAMMING_MAX = int(_env("PHASH_HAMMING_MAX", "6"))

    DEDUP_RECENT_TILES = int(_env("DEDUP_RECENT_TILES", "200000"))
    DEDUP_TTL_SEC = float(_env("DEDUP_TTL_SEC", "120"))
    DEDUP_COS_SIM_THRESH = float(_env("DEDUP_COS_SIM_THRESH", "0.95"))

    TILE_INDEX_IF_EMBED_DRIFT_GT = float(_env("TILE_INDEX_IF_EMBED_DRIFT_GT", "0.10"))
    MAX_TILES_PER_SEGMENT = int(_env("MAX_TILES_PER_SEGMENT", "200"))

    K_RETRIEVE = int(_env("K_RETRIEVE", "100"))
    K_RETURN = int(_env("K_RETURN", "20"))

    QDRANT_HOST = _env("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(_env("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "video_semantic")

    # PostgreSQL
    DATABASE_URL = _env("DATABASE_URL", "")

    # Realtime ingest / autonomy scaffolding
    REALTIME_ENABLED = _env("REALTIME_ENABLED", "false").lower() == "true"
    REALTIME_BACKEND = _env("REALTIME_BACKEND", "stub")
    REALTIME_POSE_BACKEND = _env("REALTIME_POSE_BACKEND", "stub")
    REALTIME_PACKET_BATCH_SIZE = _env_int("REALTIME_PACKET_BATCH_SIZE", 128)
    REALTIME_MAX_SENSOR_LAG_MS = _env_int("REALTIME_MAX_SENSOR_LAG_MS", 120)
    REALTIME_SESSION_TIMEOUT_SEC = _env_int("REALTIME_SESSION_TIMEOUT_SEC", 30)

    # Reports and maps output directories
    REPORTS_DIR = _env("REPORTS_DIR", os.path.join(DATA_DIR, "reports"))
    MAPS_DIR = _env("MAPS_DIR", os.path.join(DATA_DIR, "maps"))

    # Pipeline — SfM and 3DGS
    SFM_FPS = _env_float("SFM_FPS", 2.0)
    PYCOLMAP_CAMERA_MODEL = _env("PYCOLMAP_CAMERA_MODEL", "SIMPLE_RADIAL")
    PYCOLMAP_SINGLE_CAMERA = _env("PYCOLMAP_SINGLE_CAMERA", "true").lower() == "true"
    PYCOLMAP_MAX_IMAGE_SIZE = _env_int("PYCOLMAP_MAX_IMAGE_SIZE", 1920)
    PYCOLMAP_NUM_THREADS = _env_int("PYCOLMAP_NUM_THREADS", 8)
    PYCOLMAP_MATCHING = _env("PYCOLMAP_MATCHING", "sequential")
    PYCOLMAP_SEQUENTIAL_OVERLAP = _env_int("PYCOLMAP_SEQUENTIAL_OVERLAP", 8)
    PYCOLMAP_MIN_LOG_LEVEL = _env_int("PYCOLMAP_MIN_LOG_LEVEL", 2)
    PYCOLMAP_INIT_MIN_NUM_INLIERS = _env_int("PYCOLMAP_INIT_MIN_NUM_INLIERS", 50)
    PYCOLMAP_INIT_MIN_TRI_ANGLE = _env_float("PYCOLMAP_INIT_MIN_TRI_ANGLE", 4.0)
    PYCOLMAP_INIT_MAX_FORWARD_MOTION = _env_float("PYCOLMAP_INIT_MAX_FORWARD_MOTION", 0.99)
    PYCOLMAP_ABS_POSE_MIN_INLIER_RATIO = _env_float("PYCOLMAP_ABS_POSE_MIN_INLIER_RATIO", 0.15)

    # Pipeline — Florence-2 captioning
    FLORENCE_BATCH_SIZE = _env_int("FLORENCE_BATCH_SIZE", 16)
    # Prompt version tag stored in caption_model alongside model name and precision.
    # Bump this (e.g. "v2") whenever the Florence task prompt or post-processing
    # changes, so existing captions can be distinguished from newly generated ones.
    FLORENCE_PROMPT_VERSION = _env("FLORENCE_PROMPT_VERSION", "v1")
    # Optional: serve Florence-2 via a vLLM endpoint instead of loading locally.
    # Set to e.g. "http://localhost:8020/v1" and start vLLM with:
    #   vllm serve microsoft/Florence-2-large --trust-remote-code --task generate --port 8020
    # When set, no local Florence weights are loaded and VRAM is not consumed.
    FLORENCE_API_URL = _env("FLORENCE_API_URL", "")
    FLORENCE_MODEL = _env("FLORENCE_MODEL", "microsoft/Florence-2-large")

    # Pipeline — GPS extraction
    GPS_SIDECAR_PATH = _env("GPS_SIDECAR_PATH", "")
    GPS_FILTER_2D = _env("GPS_FILTER_2D", "false").lower() == "true"

    # Active learning
    AL_TAG_K = _env_int("AL_TAG_K", 50)
    # Switch from KMeans to MiniBatchKMeans when total embedded frame count exceeds this.
    # Default 25_000 ≈ 50 missions × 500 frames.
    KMEANS_BATCH_THRESHOLD = _env_int("KMEANS_BATCH_THRESHOLD", 25_000)

    # Change detection cosine-distance thresholds (per model family)
    CHANGE_DETECTION_THRESHOLD_CLIP = _env_float("CHANGE_DETECTION_THRESHOLD_CLIP", 0.35)
    CHANGE_DETECTION_THRESHOLD_DINO = _env_float("CHANGE_DETECTION_THRESHOLD_DINO", 0.25)

    # Services
    STATIC_SERVER_URL = _env("STATIC_SERVER_URL", "http://localhost:8080")
    SUPERSPLAT_SERVER_URL = _env("SUPERSPLAT_SERVER_URL", "http://localhost:8090")
    NERFSTUDIO_API_URL = _env("NERFSTUDIO_API_URL", "http://nerfstudio:8000")
    # ICP fusion mapper service (docker-compose.override.yml, port 8100 on host / 8000 in container)
    MAPPER_API_URL = _env("MAPPER_API_URL", "http://mapper:8000")

    # Phase 2 — Qwen2.5-VL-7B structured scene extraction (HTTP sidecar)
    # Set QWEN_API_URL to enable: http://qwen:8000/v1 (vLLM) or http://qwen:11434/v1 (ollama)
    # Leave empty to disable Phase 2 Qwen extraction.
    QWEN_API_URL = _env("QWEN_API_URL", "")
    QWEN_BACKEND = _env("QWEN_BACKEND", "vllm")  # "vllm" or "ollama"
    # Default model name differs per backend: Ollama uses its own tag format.
    _qwen_model_default = (
        "qwen2.5vl:7b"
        if _env("QWEN_BACKEND", "vllm").lower() == "ollama"
        or "11434" in _env("QWEN_API_URL", "")
        else "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    QWEN_MODEL = _env("QWEN_MODEL", _qwen_model_default)
    QWEN_TIMEOUT_SEC = _env_int("QWEN_TIMEOUT_SEC", 30)
    QWEN_CLIP_THRESHOLD = _env_float("QWEN_CLIP_THRESHOLD", 0.25)

    # UniDriveVLA external analysis sidecar.
    # Designed for an OpenAI-compatible bridge that serves UniDrive-style
    # understanding/perception/planning JSON, without vendoring the full
    # upstream autonomous-driving stack into selfsuvis runtime processes.
    UNIDRIVE_ENABLED = _env("UNIDRIVE_ENABLED", "false").lower() == "true"
    UNIDRIVE_API_URL = _env("UNIDRIVE_API_URL", "")
    UNIDRIVE_BACKEND = _env("UNIDRIVE_BACKEND", "vllm")  # "vllm" or "ollama"
    _unidrive_model_default = (
        "qwen2.5vl:7b"
        if _env("UNIDRIVE_BACKEND", "vllm").lower() == "ollama"
        or "11434" in _env("UNIDRIVE_API_URL", "")
        else "owl10/UniDriveVLA_Nusc_Base_Stage3"
    )
    UNIDRIVE_MODEL = _env("UNIDRIVE_MODEL", _unidrive_model_default)
    UNIDRIVE_TIMEOUT_SEC = _env_int("UNIDRIVE_TIMEOUT_SEC", 60)
    UNIDRIVE_MAX_FRAMES = _env_int("UNIDRIVE_MAX_FRAMES", 24)

    # ── ASR (Whisper) — audio-to-subtitle transcription ──────────────────────
    # Extracted subtitles are stored in frames.subtitle_text and injected into
    # the Qwen2.5-VL prompt as audio context for richer scene description.
    #
    # ASR_MODEL: "auto" = GPU-aware auto-selection (see pipeline/model_registry.py)
    #   or any HuggingFace ASR model ID, e.g. "openai/whisper-large-v3-turbo"
    # ASR_LANGUAGE: ISO-639-1 code (e.g. "en", "fr") or "" for auto-detect.
    ASR_ENABLED = _env("ASR_ENABLED", "false").lower() == "true"
    ASR_MODEL = _env("ASR_MODEL", "auto")
    ASR_LANGUAGE = _env("ASR_LANGUAGE", "")
    ASR_BATCH_SIZE = _env_int("ASR_BATCH_SIZE", 8)
    ASR_CHUNK_LENGTH_SEC = _env_int("ASR_CHUNK_LENGTH_SEC", 30)
    # Subtitle window: ±seconds around a frame timestamp when matching segments.
    ASR_SUBTITLE_WINDOW_SEC = _env_float("ASR_SUBTITLE_WINDOW_SEC", 3.0)
    # Directory for extracted WAV files (temporary audio for Whisper input).
    # Defaults to a subdirectory of DATA_DIR to co-locate with video data.
    ASR_AUDIO_DIR = _env("ASR_AUDIO_DIR", os.path.join(DATA_DIR, "audio"))

    # ── OCR — visible text extraction from frame images ───────────────────────
    # OCR results are stored in frame_facts_json["ocr_text"] and the dedicated
    # frames.ocr_text column, and injected into the Qwen prompt as visual context.
    #
    # OCR_MODEL: "auto" or HuggingFace model ID, e.g. "deepseek-ai/DeepSeek-OCR-2"
    # OCR_API_URL: if non-empty, use this vLLM/ollama endpoint instead of local model.
    OCR_ENABLED = _env("OCR_ENABLED", "false").lower() == "true"
    OCR_MODEL = _env("OCR_MODEL", "auto")
    OCR_API_URL = _env("OCR_API_URL", "")
    OCR_BATCH_SIZE = _env_int("OCR_BATCH_SIZE", 4)
    OCR_TIMEOUT_SEC = _env_int("OCR_TIMEOUT_SEC", 30)
    # Minimum caption_confidence below which OCR is run (high-confidence frames
    # usually have been captioned well enough; set to 1.0 to OCR all frames).
    OCR_MIN_CAPTION_CONFIDENCE = _env_float("OCR_MIN_CAPTION_CONFIDENCE", 0.0)

    # ── Depth estimation ──────────────────────────────────────────────────────
    # Stores 5-bucket depth percentiles in frame_facts_json["depth"].
    # DEPTH_MODEL: "auto" or HuggingFace model ID.
    DEPTH_ENABLED = _env("DEPTH_ENABLED", "false").lower() == "true"
    DEPTH_MODEL = _env("DEPTH_MODEL", "auto")

    # ── Object detection ──────────────────────────────────────────────────────
    # Stores normalised bounding boxes in frame_facts_json["detections"].
    # DETECTION_MODEL: "auto" or HuggingFace model ID.
    # DETECTION_LABELS: comma-separated candidate labels for open-vocabulary
    #   models (Grounding DINO, OmDet-Turbo). Empty = use model's built-in COCO.
    DETECTION_ENABLED = _env("DETECTION_ENABLED", "false").lower() == "true"
    DETECTION_MODEL = _env("DETECTION_MODEL", "auto")
    DETECTION_CONFIDENCE = _env_float("DETECTION_CONFIDENCE", 0.5)
    DETECTION_LABELS = _env("DETECTION_LABELS", "")
    DETECTION_BATCH_SIZE = _env_int("DETECTION_BATCH_SIZE", 8)

    # ── World model ───────────────────────────────────────────────────────────
    # Produces video clip embeddings for temporal scene understanding.
    # Target: arxiv.org/abs/2603.19312v1 — set WORLD_MODEL to its HF ID when released.
    # WORLD_MODEL: "auto" or HuggingFace model ID.
    # WORLD_MODEL_CLIP_FRAMES: number of frames aggregated per clip embedding.
    # WORLD_MODEL_STORE_EMBED: store raw embedding vector in frame_facts_json
    #   (increases DB size significantly; default false — only stores metadata).
    WORLD_MODEL_ENABLED = _env("WORLD_MODEL_ENABLED", "false").lower() == "true"
    WORLD_MODEL = _env("WORLD_MODEL", "nvidia/Cosmos-1.0-Autoregressive-4B")
    WORLD_MODEL_CLIP_FRAMES = _env_int("WORLD_MODEL_CLIP_FRAMES", 8)
    WORLD_MODEL_STORE_EMBED = _env("WORLD_MODEL_STORE_EMBED", "false").lower() == "true"

    # ── DreamerV3 RSSM temporal surprise scoring ──────────────────────────────
    # Lightweight Recurrent State Space Model inspired by:
    #   Romero et al., "Dream to Fly", ICRA 2026 (rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf)
    # Operates on pre-computed CLIP embedding sequences — no GPU required.
    # Per-frame surprise score (how unexpected was this frame given prior context?)
    # is fed into the active learning formula:
    #   al_score = 0.35*dino + 0.25*(1-caption_conf) + 0.40*rssm_surprise
    # instead of the baseline:
    #   al_score = 0.60*dino + 0.40*(1-caption_conf)
    # Enables better SSL fine-tuning frame selection → better hydrated edge models.
    #
    # DREAMER_ENABLED:       master switch (default true; lightweight, CPU-only)
    # DREAMER_HIDDEN_DIM:    RSSM GRU hidden state dimension (default 256)
    # DREAMER_LATENT_DIM:    stochastic latent z_k dimension (default 32)
    # DREAMER_TRAIN_STEPS:   online gradient steps per mission (default 20)
    # DREAMER_STORE_TEMPORAL: store recurrent state h_k in frame_facts_json
    #   (default false — only stores surprise_score and method)
    DREAMER_ENABLED = _env("DREAMER_ENABLED", "true").lower() == "true"
    DREAMER_HIDDEN_DIM = _env_int("DREAMER_HIDDEN_DIM", 256)
    DREAMER_LATENT_DIM = _env_int("DREAMER_LATENT_DIM", 32)
    DREAMER_TRAIN_STEPS = _env_int("DREAMER_TRAIN_STEPS", 20)
    DREAMER_STORE_TEMPORAL = _env("DREAMER_STORE_TEMPORAL", "false").lower() == "true"

    # ── RF signal analysis (TorchSig) ─────────────────────────────────────────
    # Analyzes IQ recordings captured alongside mission video.
    # Stores per-frame signal metrics in frame_facts_json["rf_signal"].
    #
    # IQ data sources (auto-detected, in priority order):
    #   1. <video>.iq / <video>.bin — raw interleaved float32 I/Q
    #   2. <video>.sigmf-data       — SigMF binary (+ matching .sigmf-meta)
    #   3. Video audio track        — real-valued proxy (16 kHz, no SDR required)
    #
    # RF_SAMPLE_RATE:            ADC sample rate of the IQ capture (Hz).
    #                            Ignored for audio proxy (rate read from WAV).
    # RF_WINDOW_SEC:             IQ window per frame (seconds).
    # RF_NPERSEG:                FFT segment length for spectrogram (samples).
    # RF_CLASSIFIER_CHECKPOINT:  Path to TorchScript modulation classifier .pt.
    #                            Leave empty to skip modulation labelling.
    # RF_CLASSIFIER_CLASSES:     JSON list of class names matching classifier output.
    #                            Defaults to 24-class TorchSig NarrowBand vocab.
    RF_ENABLED = _env("RF_ENABLED", "true").lower() == "true"
    RF_SAMPLE_RATE = _env_int("RF_SAMPLE_RATE", 1_000_000)
    RF_WINDOW_SEC = _env_float("RF_WINDOW_SEC", 0.5)
    RF_NPERSEG = _env_int("RF_NPERSEG", 256)
    RF_CLASSIFIER_CHECKPOINT = _env("RF_CLASSIFIER_CHECKPOINT", "")
    RF_CLASSIFIER_CLASSES = _env("RF_CLASSIFIER_CLASSES", "")

    # ── Sensor fusion ─────────────────────────────────────────────────────────────
    # Enable the multi-modal sensor fusion step (fuses THERMAL, LIDAR, GAS,
    # ACOUSTIC readings with visual detections into a unified frame_facts_json entry).
    # SENSOR_FUSION_MAX_LAG_MS: max timestamp skew for cross-sensor alignment.
    SENSOR_FUSION_ENABLED = _env("SENSOR_FUSION_ENABLED", "true").lower() == "true"
    SENSOR_FUSION_MAX_LAG_MS = _env_int("SENSOR_FUSION_MAX_LAG_MS", 100)

    # ── Thermal (FLIR / LWIR cameras) ────────────────────────────────────────────
    # When enabled the pipeline will attempt to load a YOLO-nano fine-tuned on the
    # FLIR ADAS thermal dataset (auto-downloaded from HuggingFace on first run).
    # THERMAL_MODEL: HuggingFace model ID or local path to a YOLO .pt checkpoint.
    THERMAL_ENABLED = _env("THERMAL_ENABLED", "true").lower() == "true"
    THERMAL_MODEL = _env("THERMAL_MODEL", "")

    # ── LiDAR (3-D point-cloud fusion) ───────────────────────────────────────────
    # Fuses LiDAR point-cloud data (if available alongside the video) into depth
    # estimates per detected object.  Expects a matching .pcd / .bin sidecar file.
    LIDAR_ENABLED = _env("LIDAR_ENABLED", "true").lower() == "true"

    # ── Gas / chemical sensors ────────────────────────────────────────────────────
    # Reads gas sensor readings from a CSV sidecar file and annotates frames with
    # hazard levels (CO₂, CH₄, VOC, etc.).
    GAS_ENABLED = _env("GAS_ENABLED", "true").lower() == "true"

    # ── Acoustic sensors ──────────────────────────────────────────────────────────
    # Runs audio event detection on the video audio track (or a separate WAV
    # sidecar).  Annotates frames with detected events (gunshot, siren, alarm…).
    ACOUSTIC_ENABLED = _env("ACOUSTIC_ENABLED", "true").lower() == "true"

    # CVAT annotation service (http://localhost:8091 when running via make cvat-up)
    CVAT_URL = _env("CVAT_URL", "http://localhost:8091")
    # HMAC-SHA256 secret for verifying CVAT webhook payloads (X-Hook-Secret header).
    # If empty, signature verification is skipped (dev mode only).
    CVAT_WEBHOOK_SECRET = _env("CVAT_WEBHOOK_SECRET", "")
    # JSON dict mapping CVAT label names to canonical vocabulary.
    # Example: '{"automobile":"car","person":"pedestrian"}'
    # Labels absent from the dict are passed through unchanged.
    # Applied in AnnotatedFrameDataset.from_xml() and from_db() to normalize
    # label names across CVAT tasks before building the class vocabulary.
    CVAT_LABEL_MAPPINGS: dict = _env_json_dict("CVAT_LABEL_MAPPINGS", {})

    # Security and limits
    ALLOWED_INDEX_PATHS = _parse_allowed_paths(os.getenv("ALLOWED_INDEX_PATHS"))
    MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    MAX_DOWNLOAD_BYTES = _env_int("MAX_DOWNLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    PRECHECK_URL_TIMEOUT = _env_int("PRECHECK_URL_TIMEOUT", 20)
    MAX_REDIRECTS = _env_int("MAX_REDIRECTS", 5)
    ALLOW_PRIVATE_URLS = _env("ALLOW_PRIVATE_URLS", "false").lower() == "true"

    # Robot identity (used to tag Qdrant payloads for multi-robot filtering)
    ROBOT_ID = _env("ROBOT_ID", "robot_0")

    # Worker identity (used in gpu_jobs table for resource isolation)
    WORKER_ID = _env("WORKER_ID", __import__("socket").gethostname())
    # Maximum seconds a gpu_jobs entry may be held before it is considered stale.
    GPU_JOB_TIMEOUT_SEC = _env_int("GPU_JOB_TIMEOUT_SEC", 3600)

    # Model version provenance: ID stored in Qdrant frame payloads at embed/re-embed time.
    # Set automatically by the worker after a successful supervised fine-tune.
    # Override via env var to annotate frames embedded with a known checkpoint.
    MODEL_VERSION_ID = _env("MODEL_VERSION_ID", "base")

    # Self-supervised DINOv3 domain adaptation (scripts/finetune_dino.py)
    SSL_CHECKPOINT_DIR = _env("SSL_CHECKPOINT_DIR", os.path.join(DATA_DIR, "checkpoints"))
    # Supervised CVAT fine-tuning (scripts/supervised_finetune_dino.py)
    SUP_CHECKPOINT_DIR = _env("SUP_CHECKPOINT_DIR", os.path.join(DATA_DIR, "checkpoints", "supervised"))
    SSL_FINETUNE_EPOCHS = _env_int("SSL_FINETUNE_EPOCHS", 10)
    SSL_FINETUNE_LR = _env_float("SSL_FINETUNE_LR", 1e-5)
    SSL_FINETUNE_BATCH_SIZE = _env_int("SSL_FINETUNE_BATCH_SIZE", 32)
    SSL_FINETUNE_FREEZE_BLOCKS = _env_int("SSL_FINETUNE_FREEZE_BLOCKS", 10)
    SSL_FINETUNE_TEMPERATURE = _env_float("SSL_FINETUNE_TEMPERATURE", 0.07)
    # "temporal": consecutive frames from same video dir; "augment": two augmented views
    SSL_FINETUNE_APPROACH = _env("SSL_FINETUNE_APPROACH", "temporal")
    # Path to fine-tuned DINOv3 backbone weights. When set and the file exists,
    # DINOEmbedder loads this checkpoint instead of the pretrained hub weights.
    DINO_CHECKPOINT = _env("DINO_CHECKPOINT", "")

    SUP_FINETUNE_EPOCHS = _env_int("SUP_FINETUNE_EPOCHS", 10)
    SUP_FINETUNE_LR = _env_float("SUP_FINETUNE_LR", 1e-5)
    SUP_FINETUNE_BATCH_SIZE = _env_int("SUP_FINETUNE_BATCH_SIZE", 16)
    SUP_FINETUNE_FREEZE_BLOCKS = _env_int("SUP_FINETUNE_FREEZE_BLOCKS", 8)
    SUP_FINETUNE_TEMPERATURE = _env_float("SUP_FINETUNE_TEMPERATURE", 0.07)

    # Active learning loop closure
    # Whether the CVAT webhook auto-triggers supervised fine-tuning.
    SUP_AUTO_TRIGGER = _env("SUP_AUTO_TRIGGER", "true").lower() == "true"
    # Minimum annotated frames before any finetune job is enqueued.
    MIN_ANNOTATED_FRAMES = _env_int("MIN_ANNOTATED_FRAMES", 50)
    # Minimum new annotated frames since last retrain before re-triggering.
    MIN_NEW_ANNOTATED_SINCE_RETRAIN = _env_int("MIN_NEW_ANNOTATED_SINCE_RETRAIN", 100)
    # Batch size for re-embedding sweeps (frames per Qdrant upsert call).
    REEMBED_BATCH_SIZE = _env_int("REEMBED_BATCH_SIZE", 256)
    # CVAT API bearer token for fetching annotation labels via REST API.
    CVAT_API_TOKEN = _env("CVAT_API_TOKEN", "")
    # Fraction of annotated frames held out for eval gate (0.0 disables gate).
    SUP_EVAL_FRACTION = _env_float("SUP_EVAL_FRACTION", 0.1)
    # Minimum frames per class in the eval split (single-class → reject).
    SUP_MIN_PER_CLASS_EVAL = _env_int("SUP_MIN_PER_CLASS_EVAL", 2)
    # Minimum total frames required to run the eval gate; below this → reject.
    SUP_MIN_EVAL_GATE_FRAMES = _env_int("SUP_MIN_EVAL_GATE_FRAMES", 20)
    # Minimum 1-NN eval accuracy required to promote a checkpoint.
    SUP_EVAL_GATE_THRESHOLD = _env_float("SUP_EVAL_GATE_THRESHOLD", 0.6)
    # Intra-vs-inter cosine gap above this value triggers an overfitting warning (not a gate).
    SUP_OVERFITTING_SHIFT_THRESHOLD = _env_float("SUP_OVERFITTING_SHIFT_THRESHOLD", 0.9)

    # Edge model hydration (scripts/export_onnx.py, scripts/build_gallery.py, pipeline/edge_inference.py)
    EDGE_MODELS_DIR = _env("EDGE_MODELS_DIR", os.path.join(DATA_DIR, "models"))
    EDGE_GALLERY_DIR = _env("EDGE_GALLERY_DIR", os.path.join(DATA_DIR, "gallery"))
    EDGE_ONNX_PATH = _env("EDGE_ONNX_PATH", "")       # path to quantized ONNX for EdgeClassifier
    EDGE_GALLERY_PATH = _env("EDGE_GALLERY_PATH", "")  # path to gallery NPZ for EdgeClassifier
    EDGE_TOP_K = _env_int("EDGE_TOP_K", 3)

    # Worker and ffmpeg
    WORKER_POLL_INTERVAL = _env_float("WORKER_POLL_INTERVAL", 2.0)
    FFMPEG_TIMEOUT_SEC = _env_int("FFMPEG_TIMEOUT_SEC", 3600)  # 1 hour for long videos
    # Maximum RTSP recording duration in seconds (caps duration_sec in POST /index/rtsp)
    RTSP_MAX_DURATION_SEC = _env_int("RTSP_MAX_DURATION_SEC", 3600)
    # RTSP live captioner (Phase 6): frames-per-second to sample from the stream.
    # Default 0.5 = one caption every 2 seconds. Higher values increase VRAM pressure.
    RTSP_CAPTION_FPS = _env_float("RTSP_CAPTION_FPS", 0.5)

    # Video extensions (indexing)
    VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".avi"})

    # API auth and rate limiting
    API_KEY = _env("API_KEY", "")
    RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 120)
    RATE_LIMIT_BURST = _env_int("RATE_LIMIT_BURST", 60)
    TRUST_PROXY_HEADERS = _env("TRUST_PROXY_HEADERS", "false").lower() == "true"

    # Input limits
    MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 80_000_000)
    MAX_DIR_FILES = _env_int("MAX_DIR_FILES", 5000)
    MAX_DIR_BYTES = _env_int("MAX_DIR_BYTES", 50 * 1024 * 1024 * 1024)  # 50 GB
    MAX_DIR_DEPTH = _env_int("MAX_DIR_DEPTH", 10)


def validate_settings() -> None:
    """Validate critical settings at startup. Raises ValueError on invalid config."""
    if settings.QDRANT_PORT < 1 or settings.QDRANT_PORT > 65535:
        raise ValueError("QDRANT_PORT must be 1-65535")
    if settings.MODEL_NAME not in {"openclip", "dinov2", "dinov3", "gemma"}:
        raise ValueError("MODEL_NAME must be openclip, dinov2, dinov3, or gemma")
    if settings.MAX_UPLOAD_BYTES < 0 or settings.MAX_DOWNLOAD_BYTES < 0:
        raise ValueError("MAX_UPLOAD_BYTES and MAX_DOWNLOAD_BYTES must be non-negative")
    if settings.FFMPEG_TIMEOUT_SEC < 1:
        raise ValueError("FFMPEG_TIMEOUT_SEC must be >= 1")
    if settings.TILE_SIZE < 1 or settings.STRIDE < 1:
        raise ValueError("TILE_SIZE and STRIDE must be >= 1")
    if settings.MOTION_LOW < 0 or settings.MOTION_HIGH < 0 or settings.MOTION_LOW > settings.MOTION_HIGH:
        raise ValueError("MOTION_LOW and MOTION_HIGH must be non-negative and MOTION_LOW <= MOTION_HIGH")
    if settings.SAMPLE_FPS_MIN <= 0 or settings.SAMPLE_FPS_MAX < settings.SAMPLE_FPS_MIN:
        raise ValueError("SAMPLE_FPS_MIN must be > 0 and SAMPLE_FPS_MAX >= SAMPLE_FPS_MIN")
    if settings.MAX_REDIRECTS < 0:
        raise ValueError("MAX_REDIRECTS must be >= 0")
    if settings.RATE_LIMIT_PER_MIN < 0 or settings.RATE_LIMIT_BURST < 0:
        raise ValueError("RATE_LIMIT_PER_MIN and RATE_LIMIT_BURST must be >= 0")
    if settings.MAX_IMAGE_PIXELS < 0:
        raise ValueError("MAX_IMAGE_PIXELS must be >= 0")
    if settings.MAX_DIR_FILES < 0 or settings.MAX_DIR_BYTES < 0 or settings.MAX_DIR_DEPTH < 0:
        raise ValueError("MAX_DIR_FILES/MAX_DIR_BYTES/MAX_DIR_DEPTH must be >= 0")
    if settings.HF_TOKEN:
        logger.info("HF_TOKEN configured: %s", mask_secret(settings.HF_TOKEN))
    if not settings.API_KEY:
        logger.warning(
            "API_KEY is not set; the API is unauthenticated and open to any caller. "
            "Set the API_KEY environment variable for production use."
        )
    if not settings.ALLOWED_INDEX_PATHS:
        logger.warning(
            "ALLOWED_INDEX_PATHS is not set; path-based indexing endpoints "
            "(/index/video path=, /index/dir, /index/precheck path=, /index/precheck_dir) "
            "are disabled. Set ALLOWED_INDEX_PATHS to a comma-separated list of allowed base directories."
        )
    logger.info("Settings validated successfully")


settings = Settings()

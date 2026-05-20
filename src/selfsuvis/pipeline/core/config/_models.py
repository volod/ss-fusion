"""ML model settings mixin: Gemma, Qwen, Florence, YOLO, SAM, RF, fusion, etc."""

import os

from ._helpers import _env, _env_float, _env_int


class _ModelSettings:
    OPENCLIP_MODEL = _env("OPENCLIP_MODEL", "ViT-B-16")
    OPENCLIP_PRETRAINED = _env("OPENCLIP_PRETRAINED", "openai")

    # -- Gemma open-weight local embedder (MODEL_NAME=gemma) -------------------
    # gemma-3-4b-it: ~8 GiB BF16, multimodal (vision + text) -- recommended default.
    # Smaller text-only option: gemma-3-1b-it (~2 GiB, no vision -> captions fall back to text).
    # Larger option: gemma-3-12b-it (~24 GiB, requires offloading other models).
    # All require HF_TOKEN + accepting license at huggingface.co/google/gemma-3-4b-it
    GEMMA_MODEL_ID = _env("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    GEMMA_USE_BF16 = _env("GEMMA_USE_BF16", "true").lower() == "true"

    # -- Gemma sidecar API -----------------------------------------------------
    # Set GEMMA_API_URL to enable: http://localhost:11434/v1 (Ollama) or http://localhost:8000/v1 (vLLM)
    # Leave empty to use local GemmaEmbedder only (no generative analysis).
    GEMMA_API_URL = _env("GEMMA_API_URL", "")
    GEMMA_API_BACKEND = _env("GEMMA_API_BACKEND", "ollama")  # "ollama" or "vllm"
    # gemma4:e4b is the edge-optimised 4B (~3 GiB Q4) -- fits on 16 GiB GPUs alongside
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
    GEMMA_CAPTION_CHUNK_SIZE = _env_int("GEMMA_CAPTION_CHUNK_SIZE", 3)
    GEMMA_CAPTION_RETRIES = _env_int("GEMMA_CAPTION_RETRIES", 1)
    # Multi-frame diffs are expensive and become redundant on short clips with caption churn.
    GEMMA_SEGMENT_DIFF_MAX_BOUNDARIES = _env_int("GEMMA_SEGMENT_DIFF_MAX_BOUNDARIES", 16)

    # -- Final reasoning/audit sidecar -----------------------------------------
    # Defaults to the same endpoint as Gemma analysis, but can use a larger
    # long-thinking model for the last step.
    REASONING_API_URL = _env("REASONING_API_URL", GEMMA_API_URL)
    REASONING_BACKEND = _env("REASONING_BACKEND", GEMMA_API_BACKEND)
    REASONING_MODEL = _env("REASONING_MODEL", "")
    REASONING_TIMEOUT_SEC = _env_int("REASONING_TIMEOUT_SEC", 240)
    # qwen3:8b and similar chain-of-thought models emit <think> tokens before
    # the answer body.  The previous defaults (700/900/1300) were too small:
    # the model exhausted its budget mid-answer, producing incomplete output.
    # Raised to give ~800 thinking tokens + full answer headroom.
    REASONING_MAX_TOKENS_SIMPLE = _env_int("REASONING_MAX_TOKENS_SIMPLE", 1200)
    REASONING_MAX_TOKENS_COMPACT = _env_int("REASONING_MAX_TOKENS_COMPACT", 2000)
    REASONING_MAX_TOKENS_FULL = _env_int("REASONING_MAX_TOKENS_FULL", 3200)

    SAM_MODEL_TYPE = _env("SAM_MODEL_TYPE", "vit_h")
    SAM_CHECKPOINT = _env("SAM_CHECKPOINT", "")

    # -- YOLO11 detection (ultralytics) ----------------------------------------
    # Default model: yolo11l.pt (~48 MB, 25.3 M params, 53.4 COCO mAP50-95).
    # Available tiers: yolo11n (6 MB) / yolo11s (18 MB) / yolo11m (38 MB)
    #                  yolo11l (48 MB) / yolo11x (109 MB)
    YOLO_ENABLED = _env("YOLO_ENABLED", "true").lower() == "true"
    YOLO_MODEL = _env("YOLO_MODEL", "yolo11l")
    YOLO_CONFIDENCE = _env_float("YOLO_CONFIDENCE", 0.25)

    # -- YOLO Semantic Scene Graph (SSG) --------------------------------------
    # Builds an observation-centric 3D semantic environment graph from YOLO
    # detections anchored to frame poses (ENU in production, SfM/PCA in local mode).
    YOLO_SSG_ENABLED = _env("YOLO_SSG_ENABLED", "true").lower() == "true"
    YOLO_SSG_MIN_OBSERVATIONS = _env_int("YOLO_SSG_MIN_OBSERVATIONS", 1)
    YOLO_SSG_CLUSTER_RADIUS_METERS = _env_float("YOLO_SSG_CLUSTER_RADIUS_METERS", 12.0)
    YOLO_SSG_NEAR_EDGE_RADIUS_METERS = _env_float("YOLO_SSG_NEAR_EDGE_RADIUS_METERS", 20.0)
    YOLO_SSG_CLUSTER_RADIUS_PCA = _env_float("YOLO_SSG_CLUSTER_RADIUS_PCA", 0.85)
    YOLO_SSG_NEAR_EDGE_RADIUS_PCA = _env_float("YOLO_SSG_NEAR_EDGE_RADIUS_PCA", 1.5)

    # -- SAM2 / SAM3 segmentation ----------------------------------------------
    # SAM_MODEL: "auto" (tries sam3 -> sam2 -> segment-anything) | "sam2" | "sam3" | "sam1"
    SAM_ENABLED = _env("SAM_ENABLED", "true").lower() == "true"
    SAM_MODEL = _env("SAM_MODEL", "auto")

    # -- RF-DETR directed tracking (step P3) -----------------------------------
    # Requires GEMMA_API_URL to be set; silently skipped otherwise.
    # RFDETR_MODEL: "base" (RFDETRBase, faster) | "large" (RFDETRLarge, higher accuracy)
    RFDETR_ENABLED = _env("RFDETR_ENABLED", "true").lower() == "true"
    RFDETR_MODEL = _env("RFDETR_MODEL", "base")
    RFDETR_CONFIDENCE = _env_float("RFDETR_CONFIDENCE", 0.35)

    _data_dir = _env("DATA_DIR", "./.data")
    LABELS_FILE = _env("LABELS_FILE", os.path.join(_data_dir, "labels", "openclip_rich.txt"))

    # -- Florence-2 captioning -------------------------------------------------
    FLORENCE_BATCH_SIZE = _env_int("FLORENCE_BATCH_SIZE", 16)
    # Bump FLORENCE_PROMPT_VERSION (e.g. "v2") when the Florence task prompt or
    # post-processing changes so existing captions can be distinguished from new ones.
    FLORENCE_PROMPT_VERSION = _env("FLORENCE_PROMPT_VERSION", "v1")
    # Optional vLLM endpoint: vllm serve microsoft/Florence-2-large --trust-remote-code
    #   --task generate --port 8020  -> set FLORENCE_API_URL=http://localhost:8020/v1
    FLORENCE_API_URL = _env("FLORENCE_API_URL", "")
    FLORENCE_MODEL = _env("FLORENCE_MODEL", "microsoft/Florence-2-large")

    # -- GPS extraction --------------------------------------------------------
    GPS_SIDECAR_PATH = _env("GPS_SIDECAR_PATH", "")
    GPS_FILTER_2D = _env("GPS_FILTER_2D", "false").lower() == "true"

    # -- Qwen2.5-VL structured scene extraction (HTTP sidecar) ----------------
    # Set QWEN_API_URL to enable: http://qwen:8000/v1 (vLLM) or http://qwen:11434/v1 (ollama)
    QWEN_API_URL = _env("QWEN_API_URL", "")
    QWEN_BACKEND = _env("QWEN_BACKEND", "vllm")  # "vllm" or "ollama"
    _qwen_model_default = (
        "qwen2.5vl:7b"
        if _env("QWEN_BACKEND", "vllm").lower() == "ollama" or "11434" in _env("QWEN_API_URL", "")
        else "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    QWEN_MODEL = _env("QWEN_MODEL", _qwen_model_default)
    QWEN_TIMEOUT_SEC = _env_int("QWEN_TIMEOUT_SEC", 30)
    QWEN_CLIP_THRESHOLD = _env_float("QWEN_CLIP_THRESHOLD", 0.25)
    _qwen_sidecar_concurrency_default = (
        1
        if _env("QWEN_BACKEND", "vllm").lower() == "ollama" or "11434" in _env("QWEN_API_URL", "")
        else 2
    )
    _qwen_is_local_ollama = _env("QWEN_BACKEND", "vllm").lower() == "ollama" or "11434" in _env(
        "QWEN_API_URL", ""
    )
    QWEN_SIDECAR_CONCURRENCY = max(
        1, _env_int("QWEN_SIDECAR_CONCURRENCY", _qwen_sidecar_concurrency_default)
    )
    QWEN_IMAGE_MAX_SIDE = _env_int("QWEN_IMAGE_MAX_SIDE", 768 if _qwen_is_local_ollama else 960)
    QWEN_MAX_FRAMES = _env_int("QWEN_MAX_FRAMES", 20 if _qwen_is_local_ollama else 24)

    # -- UniDriveVLA external analysis sidecar ---------------------------------
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

    # -- SceneTok streaming encoder + segmentation decoder (Step 14) ----------
    # SceneTok (arxiv 2602.18882): permutation-invariant latent tokens -> novel views or masks.
    # SCENETOK_CHECKPOINT variants: va-videodc_re10k (default), va-videodc_dl3dv, va-wan_dl3dv.
    # SCENETOK_MODE: "masks" (segmentation decoder) or "rgb" (novel-view decoder).
    SCENETOK_ENABLED = _env("SCENETOK_ENABLED", "false").lower() == "true"
    SCENETOK_API_URL = _env("SCENETOK_API_URL", "")
    SCENETOK_CHECKPOINT = _env("SCENETOK_CHECKPOINT", "va-videodc_re10k")
    SCENETOK_MODE = _env("SCENETOK_MODE", "masks")
    SCENETOK_TIMEOUT_SEC = _env_int("SCENETOK_TIMEOUT_SEC", 300)
    SCENETOK_MAX_FRAMES = _env_int("SCENETOK_MAX_FRAMES", 32)

    # -- ASR (Whisper) -- audio-to-subtitle transcription ----------------------
    # ASR_MODEL: "auto" = GPU-aware auto-selection or any HuggingFace ASR model ID.
    # ASR_LANGUAGE: ISO-639-1 code (e.g. "en") or "" for auto-detect.
    ASR_ENABLED = _env("ASR_ENABLED", "false").lower() == "true"
    ASR_MODEL = _env("ASR_MODEL", "auto")
    ASR_LANGUAGE = _env("ASR_LANGUAGE", "")
    ASR_BATCH_SIZE = _env_int("ASR_BATCH_SIZE", 8)
    ASR_CHUNK_LENGTH_SEC = _env_int("ASR_CHUNK_LENGTH_SEC", 30)
    # Subtitle window: +-seconds around a frame timestamp when matching segments.
    ASR_SUBTITLE_WINDOW_SEC = _env_float("ASR_SUBTITLE_WINDOW_SEC", 3.0)
    ASR_AUDIO_DIR = _env("ASR_AUDIO_DIR", os.path.join(_data_dir, "audio"))

    # -- OCR -- visible text extraction from frame images ----------------------
    # OCR_MODEL: "auto" or HuggingFace model ID, e.g. "deepseek-ai/DeepSeek-OCR-2"
    # OCR_API_URL: if non-empty, use this vLLM/ollama endpoint instead of local model.
    OCR_ENABLED = _env("OCR_ENABLED", "false").lower() == "true"
    OCR_MODEL = _env("OCR_MODEL", "auto")
    OCR_API_URL = _env("OCR_API_URL", "")
    OCR_BATCH_SIZE = _env_int("OCR_BATCH_SIZE", 4)
    OCR_TIMEOUT_SEC = _env_int("OCR_TIMEOUT_SEC", 30)
    _ocr_sidecar_concurrency_default = (
        1
        if "11434" in _env("OCR_API_URL", "")
        or (
            not _env("OCR_API_URL", "")
            and (
                _env("QWEN_BACKEND", "vllm").lower() == "ollama"
                or "11434" in _env("QWEN_API_URL", "")
            )
        )
        else 2
    )
    OCR_SIDECAR_CONCURRENCY = max(
        1, _env_int("OCR_SIDECAR_CONCURRENCY", _ocr_sidecar_concurrency_default)
    )
    OCR_IMAGE_MAX_SIDE = _env_int("OCR_IMAGE_MAX_SIDE", 1280)
    # Minimum caption_confidence below which OCR is run.  Set to 1.0 to OCR all frames.
    OCR_MIN_CAPTION_CONFIDENCE = _env_float("OCR_MIN_CAPTION_CONFIDENCE", 0.55)
    # Hard cap on frames sent to OCR regardless of confidence prescreen.  0 = no cap.
    OCR_MAX_FRAMES = _env_int("OCR_MAX_FRAMES", 30)

    # -- Depth estimation ------------------------------------------------------
    # DEPTH_AUTO_PROFILE: "fast" prefers a lighter model with better throughput.
    DEPTH_ENABLED = _env("DEPTH_ENABLED", "false").lower() == "true"
    DEPTH_MODEL = _env("DEPTH_MODEL", "auto")
    DEPTH_OUTPUT_MODE = _env("DEPTH_OUTPUT_MODE", "summary").strip().lower()
    DEPTH_AUTO_PROFILE = _env("DEPTH_AUTO_PROFILE", "fast")  # "fast" | "quality"
    DEPTH_BATCH_SIZE = _env_int("DEPTH_BATCH_SIZE", 8)
    DEPTH_IMAGE_MAX_SIDE = _env_int("DEPTH_IMAGE_MAX_SIDE", 768)

    # -- Object detection (HF model, separate from YOLO) -----------------------
    # DETECTION_LABELS: comma-separated labels for open-vocabulary models.
    # Empty = use model's built-in COCO labels.
    DETECTION_ENABLED = _env("DETECTION_ENABLED", "false").lower() == "true"
    DETECTION_MODEL = _env("DETECTION_MODEL", "auto")
    DETECTION_CONFIDENCE = _env_float("DETECTION_CONFIDENCE", 0.5)
    DETECTION_LABELS = _env("DETECTION_LABELS", "")
    DETECTION_BATCH_SIZE = _env_int("DETECTION_BATCH_SIZE", 8)

    # -- Image segmentation ----------------------------------------------------
    # SEGMENTATION_MODEL: "auto" | "sam3" | "sam2" | "sam1" | HF model ID
    # Independent from the YOLO/Gemma box-refinement SAM settings.
    SEGMENTATION_ENABLED = _env("SEGMENTATION_ENABLED", "false").lower() == "true"
    SEGMENTATION_MODEL = _env("SEGMENTATION_MODEL", "facebook/sam2-hiera-small")
    SEGMENTATION_POINTS_PER_SIDE = _env_int("SEGMENTATION_POINTS_PER_SIDE", 8)
    SEGMENTATION_MAX_MASKS = _env_int("SEGMENTATION_MAX_MASKS", 16)
    SEGMENTATION_MIN_AREA_NORM = _env_float("SEGMENTATION_MIN_AREA_NORM", 0.002)

    # -- World model -----------------------------------------------------------
    # WORLD_MODEL_STORE_EMBED: store raw vector in frame_facts_json (increases DB size).
    WORLD_MODEL_ENABLED = _env("WORLD_MODEL_ENABLED", "false").lower() == "true"
    WORLD_MODEL = _env("WORLD_MODEL", "nvidia/Cosmos-1.0-Autoregressive-4B")
    WORLD_MODEL_CLIP_FRAMES = _env_int("WORLD_MODEL_CLIP_FRAMES", 8)
    WORLD_MODEL_STORE_EMBED = _env("WORLD_MODEL_STORE_EMBED", "false").lower() == "true"

    # -- DreamerV3 RSSM temporal surprise scoring ------------------------------
    # Lightweight RSSM inspired by Romero et al., "Dream to Fly", ICRA 2026.
    # Operates on pre-computed CLIP embedding sequences -- no GPU required.
    # Per-frame surprise score feeds the active learning formula:
    #   al_score = 0.35*dino + 0.25*(1-caption_conf) + 0.40*rssm_surprise
    # DREAMER_STORE_TEMPORAL: store h_k in frame_facts_json (default false).
    DREAMER_ENABLED = _env("DREAMER_ENABLED", "true").lower() == "true"
    DREAMER_HIDDEN_DIM = _env_int("DREAMER_HIDDEN_DIM", 256)
    DREAMER_LATENT_DIM = _env_int("DREAMER_LATENT_DIM", 32)
    DREAMER_TRAIN_STEPS = _env_int("DREAMER_TRAIN_STEPS", 20)
    DREAMER_STORE_TEMPORAL = _env("DREAMER_STORE_TEMPORAL", "false").lower() == "true"

    # -- RF signal analysis (TorchSig) -----------------------------------------
    # IQ data sources (auto-detected, in priority order):
    #   1. <video>.iq / <video>.bin -- raw interleaved float32 I/Q
    #   2. <video>.sigmf-data       -- SigMF binary (+ matching .sigmf-meta)
    #   3. Video audio track        -- real-valued proxy (16 kHz, no SDR required)
    # RF_CLASSIFIER_CHECKPOINT: TorchScript .pt; empty = skip modulation labelling.
    RF_ENABLED = _env("RF_ENABLED", "true").lower() == "true"
    RF_SAMPLE_RATE = _env_int("RF_SAMPLE_RATE", 1_000_000)
    RF_WINDOW_SEC = _env_float("RF_WINDOW_SEC", 0.5)
    RF_NPERSEG = _env_int("RF_NPERSEG", 256)
    RF_CLASSIFIER_CHECKPOINT = _env("RF_CLASSIFIER_CHECKPOINT", "")
    RF_CLASSIFIER_CLASSES = _env("RF_CLASSIFIER_CLASSES", "")

    # -- Sensor fusion ---------------------------------------------------------
    # Fuses THERMAL, LIDAR, GAS, ACOUSTIC readings with visual detections.
    # SENSOR_FUSION_MAX_LAG_MS: max timestamp skew for cross-sensor alignment.
    SENSOR_FUSION_ENABLED = _env("SENSOR_FUSION_ENABLED", "true").lower() == "true"
    SENSOR_FUSION_MAX_LAG_MS = _env_int("SENSOR_FUSION_MAX_LAG_MS", 100)

    # -- Probabilistic platform-state fusion ------------------------------------
    # Uses GPS extracted from video plus optional .imu.jsonl / .baro.jsonl sidecars.
    STATE_FUSION_ENABLED = _env("STATE_FUSION_ENABLED", "true").lower() == "true"
    STATE_FUSION_GPS_POS_STD_M = _env_float("STATE_FUSION_GPS_POS_STD_M", 5.0)
    STATE_FUSION_BARO_ALT_STD_M = _env_float("STATE_FUSION_BARO_ALT_STD_M", 2.5)
    STATE_FUSION_IMU_ACCEL_STD_MPS2 = _env_float("STATE_FUSION_IMU_ACCEL_STD_MPS2", 1.5)
    STATE_FUSION_PROCESS_POS_STD_M = _env_float("STATE_FUSION_PROCESS_POS_STD_M", 0.75)
    STATE_FUSION_PROCESS_VEL_STD_MPS = _env_float("STATE_FUSION_PROCESS_VEL_STD_MPS", 1.5)
    STATE_FUSION_INIT_VEL_STD_MPS = _env_float("STATE_FUSION_INIT_VEL_STD_MPS", 3.0)
    STATE_FUSION_CONTEXT_GAP_SEC = _env_float("STATE_FUSION_CONTEXT_GAP_SEC", 1.0)
    STATE_FUSION_SFM_POS_STD_M = _env_float("STATE_FUSION_SFM_POS_STD_M", 2.0)
    STATE_FUSION_SFM_MIN_FRAMES = _env_int("STATE_FUSION_SFM_MIN_FRAMES", 6)
    OBJECT_FUSION_ENABLED = _env("OBJECT_FUSION_ENABLED", "true").lower() == "true"
    OBJECT_FUSION_OBS_NOISE = _env_float("OBJECT_FUSION_OBS_NOISE", 0.005)
    OBJECT_FUSION_CONFIRM_HITS = _env_int("OBJECT_FUSION_CONFIRM_HITS", 3)
    OBJECT_FUSION_MAX_MISS = _env_int("OBJECT_FUSION_MAX_MISS", 5)
    MAP_FUSION_SMOOTH = _env("MAP_FUSION_SMOOTH", "true").lower() == "true"

    # -- Thermal (FLIR / LWIR cameras) -----------------------------------------
    # THERMAL_MODEL: HuggingFace model ID or local path to a YOLO .pt checkpoint.
    THERMAL_ENABLED = _env("THERMAL_ENABLED", "true").lower() == "true"
    THERMAL_MODEL = _env("THERMAL_MODEL", "")

    # -- LiDAR (3-D point-cloud fusion) ----------------------------------------
    # Expects a matching .pcd / .bin sidecar file alongside the video.
    LIDAR_ENABLED = _env("LIDAR_ENABLED", "true").lower() == "true"

    # -- Gas / chemical sensors ------------------------------------------------
    # Reads CSV sidecar and annotates frames with hazard levels (CO2, CH4, VOC, etc.).
    GAS_ENABLED = _env("GAS_ENABLED", "true").lower() == "true"

    # -- Acoustic sensors ------------------------------------------------------
    # Audio event detection on the video audio track (or a separate WAV sidecar).
    ACOUSTIC_ENABLED = _env("ACOUSTIC_ENABLED", "true").lower() == "true"

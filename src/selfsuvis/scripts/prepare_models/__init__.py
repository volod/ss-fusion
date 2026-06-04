"""Download and cache all model weights needed by selfsuvis.

Run this once (or in a Docker build step) to pre-populate the local cache so
that the API, worker, and local full-analysis pipeline can start without network access.

Usage
-----
    # Download everything (default):
    selfsuvis-models                # all configured models
    selfsuvis-models --all          # explicit form

    # Core models only:
    selfsuvis-models --clip --dino
    selfsuvis-models --clip         # OpenCLIP only
    selfsuvis-models --dino         # DINOv2/v3 hub archive + weights only

    # Gemma open-weight (downloads weights from HuggingFace — requires HF_TOKEN):
    selfsuvis-models --gemma        # Step 03: google/gemma-3-4b-it (default, multimodal)
    selfsuvis-models --gemma --gemma-model google/gemma-3-1b-it   # text-only, ~2 GiB
    selfsuvis-models --gemma --gemma-model google/gemma-3-12b-it  # 12B, ~24 GiB

    # Step-specific optional models:
    selfsuvis-models --flash-attn   # Install flash-attn (CUDA required)
    selfsuvis-models --whisper      # Step 05: Whisper ASR
    selfsuvis-models --florence     # Step 04: Florence-2 scene captioning
    selfsuvis-models --ocr          # Step 06: OCR (auto-selects by VRAM)
    selfsuvis-models --depth        # Step 07: Depth estimation
    selfsuvis-models --detection    # Step 08: Object detection
    selfsuvis-models --world-model  # Step 11: World model video embeddings
    selfsuvis-models --unidrive     # Step 13: UniDriveVLA expert model assets
    selfsuvis-models --yolo         # Step 09: YOLO11l detection (~48 MB)
    selfsuvis-models --sam          # Step 09: SAM3/SAM2 segmentation (tries sam3 first)

    # Override auto-selected model for any step:
    selfsuvis-models --ocr       --ocr-model       microsoft/trocr-base-printed
    selfsuvis-models --depth     --depth-model     depth-anything/Depth-Anything-V2-Large-hf
    selfsuvis-models --detection --detection-model IDEA-Research/grounding-dino-base
    selfsuvis-models --world-model --world-model-id MCG-NJU/videomae-base
    selfsuvis-models --unidrive --unidrive-model owl10/UniDriveVLA_Nusc_Base_Stage3
    selfsuvis-models --yolo        --yolo-model yolo11x
    selfsuvis-models --sam         --sam-model facebook/sam2-hiera-large

    # Step 14: SceneTok streaming scene encoder + segmentation decoder (~24 GB VRAM to run):
    # Checkpoints are downloaded from MPI Nextcloud (public, no login required).
    selfsuvis-models --scenetok                                    # default: va-videodc_re10k
    selfsuvis-models --scenetok --scenetok-checkpoint va-videodc_dl3dv
    selfsuvis-models --scenetok --scenetok-checkpoint va-wan_dl3dv

    # Check what is already cached (no network):
    selfsuvis-models --verify      # verify all configured models
    python scripts/prepare_models --verify --clip --dino

    # Force Hugging Face as the download source (useful when GitHub is unreachable):
    python scripts/prepare_models --dino --source hf

    # Force torch.hub / GitHub (skip HF fallback):
    python scripts/prepare_models --dino --source hub

Gated / access-restricted models
----------------------------------
Some models (e.g. nvidia/Cosmos-1.0-Autoregressive-4B) require accepting a
license on HuggingFace before downloading.  This script detects such errors and
switches to interactive mode: it prints step-by-step instructions and waits for
you to complete authentication before retrying automatically.

If you need to pre-authenticate non-interactively:
    huggingface-cli login           # stores token in HF_HOME (default .data/.cache/huggingface)
    export HF_TOKEN=<your_token>    # or set this env var

Environment
-----------
    DINO_MODEL           Comma-separated model names to warm up
                         (default: dinov2_vitb14,dinov3_vitb14)
    OPENCLIP_MODEL       OpenCLIP model name        (default from pipeline.core.config)
    OPENCLIP_PRETRAINED  OpenCLIP pretrained tag     (default from pipeline.core.config)
    GEMMA_MODEL_ID       Gemma model repo ID (default: google/gemma-3-4b-it)
    UNIDRIVE_MODEL       UniDriveVLA model repo ID (default: owl10/UniDriveVLA_Nusc_Base_Stage3)
    DEVICE               torch device for loading    (default: cpu)
    HF_TOKEN             HuggingFace token for gated / private model access (required for Gemma)
"""

import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy third-party warnings that are irrelevant to warmup.
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="xFormers is available", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*copying from a non-meta parameter", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*will be the default.*")
warnings.filterwarnings("ignore", message=".*VideoMAEFeatureExtractor is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")
warnings.filterwarnings("ignore", message=".*A new version of the following files was downloaded.*")
for _mod_name in list(sys.modules):
    _mod = sys.modules.get(_mod_name)
    reg = getattr(_mod, "__warningregistry__", None)
    if isinstance(reg, dict):
        reg.clear()

from selfsuvis.pipeline.core.env import load_script_env  # noqa: E402
from selfsuvis.pipeline.core.logging import get_logger  # noqa: E402

# anchor_file matches the original prepare_models.py path so project_roots() depth is unchanged.
load_script_env(anchor_file=str(Path(__file__).parent.parent / "prepare_models.py"))

os.environ.setdefault("DEVICE", "auto")
os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
os.environ.setdefault("API_KEY", "")

_DATA_DIR = Path(os.getenv("DATA_DIR", "./.data")).resolve()
_CACHE_DIR = Path(os.getenv("CACHE_DIR", str(_DATA_DIR / ".cache"))).resolve()
os.environ.setdefault("CACHE_DIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(_CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_DIR / "torch"))
os.environ.setdefault("CLIP_CACHE", str(_CACHE_DIR / "clip"))
os.environ.setdefault("OPENCLIP_CACHE_DIR", str(_CACHE_DIR / "open_clip"))
os.environ.setdefault("UV_CACHE_DIR", str(_CACHE_DIR / "uv"))
os.environ.setdefault("PIP_CACHE_DIR", str(_CACHE_DIR / "pip"))

log = get_logger("prepare_models")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Re-export everything that external callers access via `import prepare_models as pm; pm.X`
from ._cli import main, _default_all_if_no_selection, _resolve_hf_model  # noqa: E402
from ._ollama import (  # noqa: E402
    _resolve_unidrive_backend,
    _resolve_unidrive_prepare_model,
    _is_ollama_model_cached,
)
from ._cache import _is_hf_cached, _is_openclip_cached, _is_dino_hub_cached  # noqa: E402
from ._special import (  # noqa: E402
    _is_yolo_cached,
    _is_scenetok_cached,
    _normalize_scenetok_checkpoint_name,
)
from ._gemma import _is_gemma_cached  # noqa: E402

__all__ = [
    "main",
    "_default_all_if_no_selection",
    "_resolve_hf_model",
    "_resolve_unidrive_backend",
    "_resolve_unidrive_prepare_model",
    "_is_ollama_model_cached",
    "_is_hf_cached",
    "_is_openclip_cached",
    "_is_dino_hub_cached",
    "_is_yolo_cached",
    "_is_scenetok_cached",
    "_normalize_scenetok_checkpoint_name",
    "_is_gemma_cached",
]

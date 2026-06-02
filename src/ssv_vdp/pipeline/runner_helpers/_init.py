"""Model and vector-store initialisation; video discovery helpers."""

import sys
import time
from pathlib import Path
from typing import Any

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.storage import InMemoryStore

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

from ...steps.common import _banner

_log = get_logger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}



def init_models(device: str) -> dict[str, Any]:
    from ...steps.caption import _log_vram_snapshot, _unload_known_sidecars

    _banner("Initialising models")
    models: dict[str, Any] = {"device": device, "uses_api_embedder": False}

    # The pre-flight check above may have left Ollama sidecars resident in VRAM.
    # Evict them now so local model loads (GemmaEmbedder / OpenCLIP / DINO) have
    # enough headroom.  We'll re-load the sidecar models on-demand in each step.
    if device == "cuda":
        import gc as _gc

        import torch as _torch_init

        _unload_known_sidecars(
            [
                (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
                (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
                (
                    getattr(settings, "REASONING_API_URL", ""),
                    getattr(settings, "REASONING_MODEL", ""),
                ),
            ]
        )
        _gc.collect()
        _torch_init.cuda.empty_cache()

    if settings.MODEL_NAME == "gemma":
        if not _HAS_GEMMA:
            raise ImportError(
                "models.gemma_model is unavailable — install transformers and accelerate."
            )
        hf_token = settings.HF_TOKEN
        if not hf_token:
            _log.warning(
                "HF_TOKEN is not set. Gemma is a gated model — set HF_TOKEN=hf_... in .env "
                "or run: huggingface-cli login"
            )
        else:
            from selfsuvis.pipeline.core.config import mask_secret as _mask  # noqa: PLC0415

            _log.info("  HF_TOKEN: %s", _mask(hf_token))
        _log.info("Loading GemmaEmbedder (%s) …", settings.GEMMA_MODEL_ID)
        t0 = time.time()
        try:
            models["clip"] = GemmaEmbedder(
                model_id=settings.GEMMA_MODEL_ID,
                device=device,
                use_bf16=settings.GEMMA_USE_BF16,
                hf_token=hf_token,
            )
        except Exception as exc:
            if settings.GEMMA_API_URL:
                _log.warning(
                    "  GemmaEmbedder load failed (%s) — falling back to OpenCLIP for embeddings. "
                    "Sidecar (%s) will still handle generative analysis.",
                    exc,
                    settings.GEMMA_API_URL,
                )
            else:
                raise RuntimeError(
                    f"GemmaEmbedder failed to load: {exc}\n"
                    "Fix: set HF_TOKEN=hf_... in .env (accept license at "
                    "huggingface.co/google/gemma-4-it-2b) or run: huggingface-cli login"
                ) from exc
        else:
            models["dino"] = None
            models["uses_api_embedder"] = False
            _log.info(
                "  [ok] GemmaEmbedder ready in %.1fs  (dim=%d)",
                time.time() - t0,
                models["clip"].image_dim(),
            )
            _log.info(
                "  [info]  SSL fine-tuning and distillation steps are skipped for Gemma embedder."
            )
            return models
        # Fall through to load OpenCLIP when local Gemma failed but sidecar is set

    _log.info("Loading OpenCLIP ViT-B-16 …")
    t0 = time.time()
    _log_vram_snapshot("before OpenCLIP load")
    models["clip"] = OpenCLIPEmbedder()
    _log.info("  [ok] CLIP ready in %.1fs  (dim=%d)", time.time() - t0, models["clip"].image_dim())

    if _HAS_DINO:
        _log.info("Loading DINOv3 ViT-B/14 …  (first run downloads ~330 MB)")
        t0 = time.time()
        try:
            models["dino"] = DINOEmbedder("dinov3_vitb14")
            _log.info(
                "  [ok] DINO ready in %.1fs  (dim=%d)", time.time() - t0, models["dino"].image_dim()
            )
        except Exception as exc:
            _log.warning("  ✗ DINOv3 load failed (%s) — using CLIP only", exc)
            models["dino"] = None
    else:
        _log.warning("  ✗ models.dino_model unavailable — using CLIP only")
        models["dino"] = None

    return models


def init_store(models: dict[str, Any], use_qdrant: bool) -> tuple[Any, bool]:
    if not use_qdrant:
        _log.info("Qdrant disabled (--no-qdrant) — using in-memory cosine store")
        return InMemoryStore(), False
    try:
        from selfsuvis.pipeline.storage.qdrant import QdrantStore

        clip_dim = models["clip"].image_dim()
        dino_dim = models["dino"].image_dim() if models.get("dino") else None
        store = QdrantStore(clip_dim=clip_dim, dino_dim=dino_dim)
        store.client.get_collections()
        _log.info(
            "[ok] Qdrant connected at %s:%s  collection=%s",
            settings.QDRANT_HOST,
            settings.QDRANT_PORT,
            settings.QDRANT_COLLECTION,
        )
        return store, True
    except Exception as exc:
        _log.info("Qdrant unavailable (%s) — falling back to in-memory store", exc)
        _log.info("  To enable persistent vector search: docker run -p 6333:6333 qdrant/qdrant")
        return InMemoryStore(), False


# -- Video discovery ------------------------------------------------------------


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


def resolve_local_videos(args: Any) -> tuple[str, list[Path]]:
    """Resolve input videos for the local full-analysis workflow.

    Priority:
    1. ``--input`` single file
    2. ``--dir`` directory
    3. ``--videos-dir`` directory
    """
    input_path = getattr(args, "input", None)
    if input_path:
        video_path = Path(input_path).resolve()
        if not video_path.is_file():
            _log.error("Input video does not exist: %s", video_path)
            sys.exit(1)
        if video_path.suffix.lower() not in _VIDEO_EXTS:
            _log.error("Unsupported input extension for %s", video_path)
            _log.error("Supported formats: %s", " ".join(sorted(_VIDEO_EXTS)))
            sys.exit(1)
        return str(video_path.parent), [video_path]

    dir_path = getattr(args, "dir", None) or getattr(args, "videos_dir", None)
    videos_dir = Path(dir_path)
    if not videos_dir.is_dir():
        _log.error("Videos directory does not exist: %s", videos_dir)
        _log.error("Use the local data directory:  --videos-dir .data/videos")
        _log.error("Create it with:  mkdir -p .data/videos")
        sys.exit(1)

    videos = find_videos(videos_dir)
    if not videos:
        _log.error("No video files found in %s", videos_dir)
        _log.error("Supported formats: %s", " ".join(sorted(_VIDEO_EXTS)))
        sys.exit(1)
    return str(videos_dir), videos

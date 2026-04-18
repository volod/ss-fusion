"""World model wrapper for video scene understanding and future-state prediction.

Implements an embedding/feature interface for world models that understand
scene dynamics, physical plausibility, and temporal relationships in video.

Target: LeWorldModel (arxiv.org/abs/2603.19312v1, March 2026)
  "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
   from Pixels" — Maes, Le Lidec, Scieur, LeCun, Balestriero
  Architecture: JEPA. ~15M params, trains end-to-end from raw pixels using
  only two loss terms (next-embedding prediction + Gaussian regularizer).
  Plans 48× faster than foundation-model world models.  Latent space encodes
  physical quantities; detects physically implausible events.
  HuggingFace model ID: not yet released as of 2026-Q1.
  → Set WORLD_MODEL=<hf_id> once it appears on HuggingFace.

Current auto-selection hierarchy (``WORLD_MODEL=auto``):
  V-JEPA2-ViT-G → V-JEPA2-ViT-L → VideoMAEv2-Huge → VideoMAE-Large → ...
  (largest model fitting in available VRAM – 2 GB safety margin)

World model output stored in ``frame_facts_json["world_model"]``:
  {
    "embedding": null,           # full embedding (omitted if WORLD_MODEL_STORE_EMBED=false)
    "embedding_dim": 768,
    "model": "facebook/vjepa2-vitg-fpc64-256",
    "temporal_window_frames": 8  # how many frames were aggregated
  }

Disabled by default (``WORLD_MODEL_ENABLED=false``).

Top-10 video understanding models (small → large):

  1. google/vivit-b-16x2-kinetics400              86 M  ~0.2 GB  ViViT-B Kinetics-400
  2. facebook/timesformer-base-finetuned-k400    122 M  ~0.3 GB  divided space-time attn
  3. MCG-NJU/videomae-base                       122 M  ~0.3 GB  masked video autoencoding
  4. google/videoprism-base-f16r288              300 M  ~0.7 GB  Google dual-encoder
  5. MCG-NJU/videomae-large                      307 M  ~0.6 GB  stronger features
  6. facebook/vjepa2-vitl-fpc64-256              307 M  ~0.7 GB  V-JEPA2 ViT-L, 64 frames
  7. OpenGVLab/VideoMAEv2-Huge                   600 M  ~1.3 GB  VideoMAEv2 Huge
  8. facebook/vjepa2-vitg-fpc64-256              1.0 B  ~2.0 GB  V-JEPA2 ViT-G (strongest)
  9. OpenGVLab/InternVideo2-Stage2_1B-224p-f4    1.0 B  ~2.0 GB  video-language model
 10. nvidia/Cosmos-1.0-Autoregressive-4B          4.0 B  ~8.0 GB  physical world model

CLI override::

    WORLD_MODEL_ENABLED=true WORLD_MODEL=facebook/vjepa2-vitg-fpc64-256
    WORLD_MODEL_ENABLED=true WORLD_MODEL=MCG-NJU/videomae-large
    WORLD_MODEL_ENABLED=true WORLD_MODEL=nvidia/Cosmos-1.0-Autoregressive-4B
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings

from .registry import auto_select, detect_resources, normalize_model_id

logger = get_logger(__name__)

_VIDEOMAE_PREFIXES = ("MCG-NJU/videomae", "MCG-NJU/VideoMAE")
def _resolve_model_id() -> str:
    cfg = settings.WORLD_MODEL.strip()
    if cfg and cfg.lower() != "auto":
        return normalize_model_id("world_model", cfg)
    resources = detect_resources()
    return auto_select("world_model", resources) or "MCG-NJU/videomae-base"


class WorldModel:
    """World model interface for scene understanding from video frames.

    Operates in *aggregated clip* mode: collects a buffer of consecutive kept
    frames (up to ``WORLD_MODEL_CLIP_FRAMES``) then computes a single world
    embedding for the clip.  The result is assigned to the representative
    (middle) frame's ``frame_facts_json``.

    For the arxiv 2603.19312 model, once its HuggingFace ID is known, set
    ``WORLD_MODEL=<hf_id>`` and re-run — the interface is model-agnostic.
    """

    def __init__(self) -> None:
        self._feature_extractor = None
        self._model = None
        self._model_id: Optional[str] = None
        self._frame_buffer: List[Image.Image] = []
        self._clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
        self._load_failed: bool = False
        self._inference_failed: bool = False

    def is_enabled(self) -> bool:
        return settings.WORLD_MODEL_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    def process_clip(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Extract world-model features from a list of consecutive frames.

        Returns a dict suitable for merging into ``frame_facts_json``.
        """
        if not self.is_enabled():
            return {"world_model_disabled": True}
        if self._inference_failed:
            return {"world_model_unavailable": True}

        model, feat_extractor = self._load_model()
        if model is None:
            return {"world_model_unavailable": True}

        try:
            import torch

            # Sample up to clip_frames evenly spaced from the buffer
            n = len(images)
            if n == 0:
                return {"world_model_unavailable": True}
            target_frames = int(getattr(getattr(model, "config", None), "num_frames", self._clip_frames))
            sampled = _sample_exact_frames(images, max(1, target_frames))

            inputs = feat_extractor(sampled, return_tensors="pt")
            pixel_values = _normalise_video_pixel_values(
                inputs.get("pixel_values"),
                target_frames=target_frames,
            )
            if pixel_values is None:
                raise RuntimeError("World model preprocessor did not return pixel_values")
            inputs["pixel_values"] = pixel_values
            device = _get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract mean-pooled last hidden state as embedding
            hidden = outputs.last_hidden_state  # (1, T, D) or (1, D)
            embedding_np = hidden.mean(dim=1).squeeze(0).cpu().float().numpy()

            result: Dict[str, Any] = {
                "world_model": {
                    "embedding_dim": int(embedding_np.shape[0]),
                    "model": self.model_id,
                    "temporal_window_frames": len(sampled),
                }
            }
            if settings.WORLD_MODEL_STORE_EMBED:
                result["world_model"]["embedding"] = embedding_np.tolist()
            return result

        except Exception:
            self._inference_failed = True
            logger.warning("World model inference failed", exc_info=True)
            return {"world_model_error": True}

    def _load_model(self):
        if self._model is not None:
            return self._model, self._feature_extractor
        if self._load_failed:
            return None, None
        candidate_ids = _candidate_model_ids(self.model_id)
        last_exc: Optional[Exception] = None

        for candidate_id in candidate_ids:
            source = _resolve_local_world_model_path(candidate_id)
            source_label = str(source) if isinstance(source, Path) else candidate_id
            # Incomplete-cache check: look for any preprocessor/feature-extractor
            # config, not just preprocessor_config.json — VideoMAE-based models
            # (e.g. InternVideo2) may only have feature_extractor_type.json or
            # store it inside config.json.
            if isinstance(source, Path):
                _has_preprocessor = any(
                    (source / f).exists()
                    for f in ("preprocessor_config.json", "feature_extractor_config.json", "config.json")
                )
                if not _has_preprocessor:
                    logger.info(
                        "World model cache for %s is incomplete; retrying from repo id.",
                        candidate_id,
                    )
                    source = candidate_id
                    source_label = candidate_id
            logger.info("Loading world model: %s", source_label)
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel, AutoProcessor

                device = _get_device()
                load_kwargs = {
                    "local_files_only": isinstance(source, Path),
                    "trust_remote_code": True,
                }
                # VideoMAEImageProcessor is the correct preprocessor for InternVideo2
                # and other VideoMAE-family models. AutoProcessor fails on these
                # because their config.json has no auto_map / model_type entry that
                # transformers recognises.
                try:
                    from transformers import VideoMAEImageProcessor
                    _preprocessor_loaders: tuple = (VideoMAEImageProcessor, AutoImageProcessor, AutoProcessor)
                except ImportError:
                    _preprocessor_loaders = (AutoImageProcessor, AutoProcessor)
                self._feature_extractor = _load_world_preprocessor(
                    source_label,
                    load_kwargs,
                    *_preprocessor_loaders,
                )
                self._model = AutoModel.from_pretrained(
                    source_label,
                    torch_dtype=torch.float16 if settings.USE_FP16 and device != "cpu" else torch.float32,
                    **load_kwargs,
                ).to(device).eval()
                self._model_id = candidate_id
                logger.info("World model loaded: %s on %s", source_label, device)
                return self._model, self._feature_extractor
            except Exception as exc:
                last_exc = exc
                if candidate_id != candidate_ids[-1]:
                    logger.warning(
                        "World model %s is not compatible with the current embedding interface; falling back to %s.",
                        candidate_id,
                        candidate_ids[candidate_ids.index(candidate_id) + 1],
                    )
                    continue
                logger.warning(
                    "Failed to load world model %s — run: python scripts/prepare_models.py --world-model",
                    candidate_id, exc_info=True,
                )

        self._model = None
        self._feature_extractor = None
        self._load_failed = True
        if last_exc is not None:
            logger.debug("World model load failure detail: %r", last_exc)
        return None, None

    def release(self) -> None:
        """Delete the model and flush CUDA cache."""
        import gc
        self._model = None
        self._feature_extractor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _sample_indices(n: int, target: int) -> List[int]:
    """Return up to *target* evenly spaced indices from [0, n)."""
    if n <= target:
        return list(range(n))
    step = n / target
    return [int(i * step) for i in range(target)]


def _sample_exact_frames(images: List[Image.Image], target: int) -> List[Image.Image]:
    """Return exactly *target* frames by evenly sampling with duplication if needed."""
    if not images:
        return []
    if len(images) == target:
        return list(images)
    if len(images) > target:
        return [images[i] for i in _sample_indices(len(images), target)]
    if len(images) == 1:
        return [images[0]] * target
    last = len(images) - 1
    indices = [round(i * last / max(target - 1, 1)) for i in range(target)]
    return [images[i] for i in indices]


def _normalise_video_pixel_values(pixel_values, target_frames: int):
    """Normalise preprocessor output to VideoMAE's expected (B, T, C, H, W) layout."""
    if pixel_values is None:
        return None

    try:
        import torch
    except ImportError:
        return pixel_values

    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim != 5:
        raise RuntimeError(f"Unexpected world-model pixel_values shape: {tuple(pixel_values.shape)}")

    # Some processors emit (B, C, T, H, W). VideoMAE expects (B, T, C, H, W).
    if pixel_values.shape[1] == 3 and pixel_values.shape[2] != 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()

    if pixel_values.shape[2] != 3:
        raise RuntimeError(f"Unexpected world-model channel layout: {tuple(pixel_values.shape)}")

    if pixel_values.shape[1] != target_frames:
        pixel_values = _resample_frame_tensor(pixel_values, target_frames)

    return pixel_values


def _resample_frame_tensor(pixel_values, target_frames: int):
    """Resample a (B, T, C, H, W) tensor to exactly target_frames along time."""
    if pixel_values.shape[1] == target_frames:
        return pixel_values
    if pixel_values.shape[1] <= 0:
        raise RuntimeError("Cannot resample empty world-model frame tensor")

    try:
        import torch
    except ImportError:
        return pixel_values

    source_frames = pixel_values.shape[1]
    if source_frames == 1:
        return pixel_values.expand(pixel_values.shape[0], target_frames, *pixel_values.shape[2:]).contiguous()

    indices = torch.linspace(
        0,
        source_frames - 1,
        steps=target_frames,
        device=pixel_values.device,
    ).round().long()
    return pixel_values.index_select(1, indices)


def _resolve_local_world_model_path(model_id: str) -> str | Path:
    """Prefer an already-cached HF snapshot so runtime does not hit the network/auth path."""
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        path = Path(local_dir)
        if path.exists():
            logger.info("World model cache hit: %s → %s", model_id, path)
            return path
    except Exception:
        pass
    return model_id


def _candidate_model_ids(model_id: str) -> List[str]:
    supported = normalize_model_id("world_model", model_id)
    if supported == model_id:
        return [model_id]
    return [model_id, supported]


class _SimpleVideoPreprocessor:
    """Minimal fallback preprocessor for models without a transformers-compatible
    preprocessor_config.json (e.g. InternVideo2-Stage2).

    Applies standard ImageNet normalisation and returns pixel_values in
    (1, T, C, H, W) layout so _normalise_video_pixel_values can handle it.
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]
    _SIZE = (224, 224)

    def __call__(self, images, return_tensors: str = "pt", **_kwargs):
        import torch
        from torchvision import transforms  # type: ignore[import]

        tfm = transforms.Compose([
            transforms.Resize(self._SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])
        frames = [tfm(img.convert("RGB")) for img in images]  # each: (C, H, W)
        # Stack → (T, C, H, W) → unsqueeze batch → (1, T, C, H, W)
        pixel_values = torch.stack(frames, dim=0).unsqueeze(0)
        return {"pixel_values": pixel_values}


def _load_world_preprocessor(source_label: str, load_kwargs: Dict[str, Any], *loader_classes):
    last_exc: Optional[Exception] = None
    for loader_cls in loader_classes:
        try:
            return loader_cls.from_pretrained(source_label, **load_kwargs)
        except Exception as exc:
            last_exc = exc
    # All standard loaders failed (common for models like InternVideo2 that
    # ship no preprocessor_config.json and have no auto_map entry).
    # Fall back to simple ImageNet-normalised preprocessing which is compatible
    # with VideoMAE/InternVideo2 input expectations.
    logger.warning(
        "No transformers preprocessor found for %s (%s) — using _SimpleVideoPreprocessor fallback",
        source_label,
        last_exc,
    )
    return _SimpleVideoPreprocessor()


def _get_device() -> str:
    return resolve_device()

"""Edge model hydration — ONNX export and lightweight inference for on-device deployment.

Supports two distilled backbone variants for export and inference:

- **ViT-S/14** (Stage 1 distillation output, 384-dim) — exported by
  :func:`step_export_model` in ``steps_distill.py`` to ``edge_models/dino_local.onnx``.
- **EfficientViT-B1** (Stage 2 distillation output, 384-dim, ~9M params) — exported by
  :func:`export_efficientvit_onnx` to ``edge_models/efficientvit_local.onnx``.

Both ONNX files work with :class:`EdgeClassifier` for cosine-similarity nearest-neighbour
classification on edge hardware (Jetson Orin, Hailo-8, CPU-only ARM SBC) without PyTorch.

Key classes and functions:
  EdgeClassifier           — loads a quantized ONNX backbone + gallery NPZ; classifies PIL images.
  build_gallery            — embeds representative frames and saves an NPZ gallery file.
  export_efficientvit_onnx — exports an EfficientViT-B1 backbone to ONNX (opset 18).

No PyTorch or torchvision imports at module level — only inside from_torch and helper
functions that are only called with a PyTorch backbone. This keeps the edge deployment
free of PyTorch.

Usage (on robot):
    from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
    clf = EdgeClassifier("efficientvit_local.onnx", "mission_objects.npz")
    labels = clf.classify(frame_pil)   # [(label, score), ...]
"""

import os

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# ImageNet normalisation constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -- Provider selection ---------------------------------------------------------


def _select_providers(device: str) -> list[str]:
    """Select ONNX Runtime execution providers for *device*, logging gaps.

    Preference order for ``device="cuda"``:
      1. TensorrtExecutionProvider  — highest throughput on NVIDIA GPUs
         (shipped in ``onnxruntime-gpu``; requires TensorRT shared libs at runtime)
      2. CUDAExecutionProvider      — general CUDA acceleration
      3. CPUExecutionProvider       — fallback (always available)

    Note on QNN: ``QNNExecutionProvider`` targets Qualcomm Hexagon DSP / NPU
    hardware (Snapdragon, Windows-on-ARM).  It is not available on CUDA/x86
    machines and there is no CUDA simulator for it.  TensorRT is the correct
    NVIDIA analogue.

    Warns explicitly (rather than silently falling back) whenever a requested
    accelerated provider is absent so the operator knows inference degraded.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error(
            "onnxruntime not installed — cannot select providers. "
            "Install: pip install onnxruntime  (CPU)  or  pip install onnxruntime-gpu  (CUDA)."
        )
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    logger.debug("ONNX Runtime available providers on this host: %s", available)

    if device == "cpu":
        return ["CPUExecutionProvider"]

    if device == "cuda":
        selected: list[str] = []

        # -- TensorRT EP --------------------------------------------------------
        # Highest throughput on NVIDIA GPUs. Included in onnxruntime-gpu but
        # also requires libnvinfer.so.* (TensorRT) to be present at runtime.
        if "TensorrtExecutionProvider" in available:
            selected.append("TensorrtExecutionProvider")
            logger.info(
                "ONNX Runtime: TensorrtExecutionProvider available — "
                "using TensorRT for best NVIDIA GPU throughput."
            )
        else:
            logger.info(
                "ONNX Runtime: TensorrtExecutionProvider not available. "
                "To enable: install onnxruntime-gpu and TensorRT "
                "(apt install tensorrt  or  pip install tensorrt). "
                "Falling through to CUDAExecutionProvider."
            )

        # -- CUDA EP ------------------------------------------------------------
        if "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")
        else:
            logger.warning(
                "ONNX Runtime: CUDAExecutionProvider not available even though "
                "device='cuda' was requested. "
                "Install onnxruntime-gpu:  pip install onnxruntime-gpu  "
                "(current package may be CPU-only onnxruntime). "
                "Inference will fall back to CPU — expect slower performance."
            )

        selected.append("CPUExecutionProvider")
        return selected

    logger.warning(
        "EdgeClassifier: unrecognised device=%r — falling back to CPUExecutionProvider.", device
    )
    return ["CPUExecutionProvider"]


# -- Preprocessing -------------------------------------------------------------


def _preprocess_image(image_pil: Image.Image, image_size: int = 224) -> np.ndarray:
    """Preprocess a PIL image into a float32 NCHW numpy array.

    Steps: resize shortest side → centre crop → normalise (ImageNet stats).

    Args:
        image_pil:  Input PIL image (any mode, will be converted to RGB).
        image_size: Output spatial resolution (default 224).

    Returns:
        numpy array of shape (1, 3, image_size, image_size), dtype float32.
    """
    img = image_pil.convert("RGB")

    # Resize shortest side to image_size (bicubic)
    w, h = img.size
    if w < h:
        new_w = image_size
        new_h = int(h * image_size / w)
    else:
        new_h = image_size
        new_w = int(w * image_size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Centre crop to image_size × image_size
    w, h = img.size
    left = (w - image_size) // 2
    top = (h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))

    # Convert to float32 HWC in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # ImageNet normalisation
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # (H, W, 3)

    # HWC → CHW → NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1, 3, H, W)
    return arr.astype(np.float32)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D float32 array along the last axis."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


# -- EdgeClassifier ------------------------------------------------------------


class EdgeClassifier:
    """Cosine-similarity nearest-neighbour classifier backed by an ONNX backbone.

    Loads a (quantized) ONNX model produced by scripts/export_onnx.py and a gallery
    NPZ file produced by build_gallery or scripts/build_gallery.py.  No PyTorch
    dependency at inference time — only onnxruntime and numpy.

    Args:
        onnx_path:    Path to the ONNX model file.
        gallery_path: Path to the gallery NPZ (keys: embeddings, labels, label_names).
        top_k:        Maximum number of (label, score) pairs to return from classify().
        device:       Inference device string passed to onnxruntime providers.
                      "cpu" → CPUExecutionProvider; "cuda" → CUDAExecutionProvider.
    """

    def __init__(
        self,
        onnx_path: str,
        gallery_path: str,
        top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for EdgeClassifier. "
                "Install it with: pip install onnxruntime  (CPU) or  pip install onnxruntime-gpu  (CUDA)."
            ) from exc

        self.top_k = top_k
        self._device = device

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"EdgeClassifier: ONNX model not found: {onnx_path}. "
                "Export it first with:  python scripts/export_onnx.py"
            )

        providers = _select_providers(device)
        logger.info("EdgeClassifier: requested providers for device=%r: %s", device, providers)

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        active_providers = self._session.get_providers()
        logger.info(
            "EdgeClassifier: ONNX session active providers: %s  (model=%s)",
            active_providers,
            onnx_path,
        )

        # Warn when the session silently downgraded to CPU
        if device == "cuda" and "CUDAExecutionProvider" not in active_providers:
            logger.warning(
                "EdgeClassifier: CUDA was requested but CUDAExecutionProvider is NOT active. "
                "All inference will run on CPU. "
                "Verify onnxruntime-gpu is installed: pip install onnxruntime-gpu"
            )

        self._input_name: str = self._session.get_inputs()[0].name
        logger.info(
            "EdgeClassifier: loaded ONNX model from %s  (input=%r)",
            onnx_path,
            self._input_name,
        )

        # Load gallery
        # gallery_path="" is a supported "embed-only" mode used by build_gallery
        # (it only needs the ONNX session for embed(); classify() must not be called).
        if gallery_path:
            if not os.path.isfile(gallery_path):
                raise FileNotFoundError(
                    f"EdgeClassifier: gallery NPZ not found: {gallery_path}. "
                    "Build it first with:  python scripts/build_gallery.py"
                )
            data = np.load(gallery_path, allow_pickle=True)
            missing = [k for k in ("embeddings", "labels") if k not in data]
            if missing:
                raise KeyError(
                    f"EdgeClassifier: gallery NPZ {gallery_path!r} is missing keys: {missing}. "
                    "Rebuild with:  python scripts/build_gallery.py"
                )
            self._gallery_embeddings: np.ndarray = data["embeddings"].astype(np.float32)
            self._gallery_labels: np.ndarray = data["labels"]
            if self._gallery_embeddings.ndim != 2:
                raise ValueError(
                    f"EdgeClassifier: gallery embeddings have unexpected shape "
                    f"{self._gallery_embeddings.shape} (expected 2-D array). "
                    "Rebuild the gallery."
                )
            logger.info(
                "EdgeClassifier: gallery loaded from %s  (%d embeddings, dim=%d)",
                gallery_path,
                len(self._gallery_embeddings),
                self._gallery_embeddings.shape[1],
            )
        else:
            logger.debug(
                "EdgeClassifier: no gallery_path provided — running in embed-only mode. "
                "classify() must not be called."
            )
            self._gallery_embeddings = np.zeros((0, 1), dtype=np.float32)
            self._gallery_labels = np.array([], dtype=object)

    # -- Public API ------------------------------------------------------------

    def embed(self, image_pil: Image.Image) -> np.ndarray:
        """Embed a PIL image using the ONNX backbone.

        Args:
            image_pil: Input PIL image.

        Returns:
            L2-normalised embedding vector of shape (D,), float32.
        """
        x = _preprocess_image(image_pil)  # (1, 3, 224, 224)
        outputs = self._session.run(None, {self._input_name: x})
        vec = outputs[0].astype(np.float32)  # (1, D)
        vec = _l2_normalise(vec)
        return vec[0]  # (D,)

    def classify(self, image_pil: Image.Image) -> list[tuple[str, float]]:
        """Classify a PIL image against the gallery.

        Args:
            image_pil: Input PIL image.

        Returns:
            List of (label, cosine_score) pairs, sorted by score descending.
            At most top_k results. Scores are in [-1, 1].
        """
        query = self.embed(image_pil)  # (D,)
        # Cosine similarity: query (D,) · gallery (N, D)^T → (N,)
        scores: np.ndarray = self._gallery_embeddings @ query  # (N,)

        k = min(self.top_k, len(scores))
        # Get indices of top-k scores (partial sort)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(str(self._gallery_labels[i]), float(scores[i])) for i in top_idx]

    # -- Alternative constructor (for testing / dev without ONNX) -------------

    @classmethod
    def from_torch(
        cls,
        backbone,
        gallery_path: str,
        top_k: int = 3,
        device: str = "cpu",
    ) -> "EdgeClassifier":
        """Create an EdgeClassifier that wraps a PyTorch backbone instead of ONNX.

        Convenience constructor for testing and development when an ONNX file has not
        been exported yet. Requires PyTorch and torchvision; NOT suitable for edge deploy.

        Args:
            backbone:     Pretrained/fine-tuned PyTorch backbone (e.g. DINOv3 ViT).
            gallery_path: Path to gallery NPZ.
            top_k:        Maximum results to return.
            device:       PyTorch device string.

        Returns:
            EdgeClassifier-compatible object with the same embed / classify interface.
        """
        import torch
        import torch.nn.functional as F

        instance = cls.__new__(cls)
        instance.top_k = top_k
        instance._device = device
        instance._session = None  # no ONNX session

        # Load gallery
        data = np.load(gallery_path, allow_pickle=True)
        instance._gallery_embeddings = data["embeddings"].astype(np.float32)
        instance._gallery_labels = data["labels"]

        # Store PyTorch backbone
        instance._torch_backbone = backbone.to(device).eval()

        def _embed_torch(self_inner: "EdgeClassifier", image_pil: Image.Image) -> np.ndarray:
            """Torch-based embed used by from_torch instances."""
            import torchvision.transforms as T

            transform = T.Compose(
                [
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            img = image_pil.convert("RGB")
            x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
            with torch.no_grad():
                out = self_inner._torch_backbone(x)  # (1, D)
            out = F.normalize(out, dim=-1)
            return out[0].cpu().numpy().astype(np.float32)  # (D,)

        import types

        instance.embed = types.MethodType(_embed_torch, instance)

        logger.info(
            "EdgeClassifier.from_torch: gallery loaded from %s  (%d embeddings)",
            gallery_path,
            len(instance._gallery_embeddings),
        )
        return instance


# -- EfficientViT ONNX export -------------------------------------------------


def export_efficientvit_onnx(
    backbone,
    output_path: str,
    image_size: int = 224,
) -> str:
    """Export an EfficientViT backbone to ONNX for edge deployment.

    Args:
        backbone:    Trained EfficientViT-B1 PyTorch backbone.
        output_path: Destination path for the ONNX file.
        image_size:  Spatial resolution for the dummy input (default 224).

    Returns:
        output_path (unchanged), for caller convenience.

    Raises:
        ImportError: If torch is not installed.
        RuntimeError: If the ONNX export fails.
    """
    try:
        import warnings

        import torch
    except ImportError as exc:
        raise ImportError("torch is required for export_efficientvit_onnx") from exc

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    class _Wrapper(torch.nn.Module):
        def __init__(self, bb: torch.nn.Module) -> None:
            super().__init__()
            self.bb = bb

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.bb(x)

    backbone_cpu = backbone.cpu().eval()
    wrapper = _Wrapper(backbone_cpu).eval()
    dummy = torch.zeros(1, 3, image_size, image_size)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            opset_version=18,
            input_names=["pixel_values"],
            output_names=["embedding"],
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_mb = os.path.getsize(output_path) / 1e6
    logger.info("export_efficientvit_onnx: %.1f MB → %s", onnx_mb, output_path)
    return output_path


# -- Gallery builder -----------------------------------------------------------


def build_gallery(
    labels_map: dict[str, list[str]],
    output_path: str,
    onnx_path: str | None = None,
    backbone=None,
    image_size: int = 224,
) -> None:
    """Embed representative frames and save a gallery NPZ.

    Exactly one of onnx_path or backbone must be provided.

    Args:
        labels_map:   Mapping of label name → list of frame file paths.
                      e.g. {"vehicle": ["path1.jpg", "path2.jpg"], "barrier": [...]}
        output_path:  Destination path for the NPZ file.
        onnx_path:    Path to ONNX model (used if provided; no PyTorch required).
        backbone:     PyTorch backbone (used when onnx_path is None).
        image_size:   Spatial resolution for preprocessing (default 224).

    Raises:
        ValueError:        If labels_map is empty.
        FileNotFoundError: If any frame path in labels_map does not exist.
    """
    if not labels_map:
        raise ValueError("labels_map must not be empty.")

    # Validate all paths exist upfront
    for label, paths in labels_map.items():
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Frame file not found for label '{label}': {p}")

    # Build embedder
    if onnx_path is not None:
        # gallery_path="" activates embed-only mode (no gallery file required)
        clf = EdgeClassifier(onnx_path=onnx_path, gallery_path="", top_k=1)

        def _embed_fn(img: Image.Image) -> np.ndarray:
            return clf.embed(img)
    elif backbone is not None:
        import torch
        import torch.nn.functional as F

        try:
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError("torchvision is required when using a PyTorch backbone.") from exc

        _device = next(backbone.parameters()).device
        transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        backbone_eval = backbone.eval()

        def _embed_fn(img: Image.Image) -> np.ndarray:
            img_rgb = img.convert("RGB")
            x = transform(img_rgb).unsqueeze(0).to(_device)
            with torch.no_grad():
                out = backbone_eval(x)  # (1, D)
            out = F.normalize(out, dim=-1)
            return out[0].cpu().numpy().astype(np.float32)
    else:
        raise ValueError("Either onnx_path or backbone must be provided.")

    # Embed all frames
    all_embeddings: list[np.ndarray] = []
    all_labels: list[str] = []
    skipped: list[str] = []

    total_frames = sum(len(v) for v in labels_map.values())
    logger.info(
        "build_gallery: embedding %d frames across %d labels …",
        total_frames,
        len(labels_map),
    )

    for label in sorted(labels_map.keys()):
        paths = labels_map[label]
        logger.debug("build_gallery: label=%r — %d frames", label, len(paths))
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                emb = _embed_fn(img)
                emb = _l2_normalise(emb.reshape(1, -1))[0]
                all_embeddings.append(emb)
                all_labels.append(label)
                logger.debug("build_gallery: embedded %s → label=%s", p, label)
            except Exception:
                logger.warning(
                    "build_gallery: FAILED to embed %s (label=%r) — skipping frame.",
                    p,
                    label,
                    exc_info=True,
                )
                skipped.append(p)

    if skipped:
        logger.warning(
            "build_gallery: %d / %d frames were skipped due to errors: %s",
            len(skipped),
            total_frames,
            skipped,
        )

    if not all_embeddings:
        raise RuntimeError(
            "build_gallery: no frames were embedded successfully — gallery not saved. "
            "Check the errors above."
        )

    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)  # (N, D)
    labels_arr = np.array(all_labels, dtype=object)
    label_names = np.array(sorted(labels_map.keys()), dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        labels=labels_arr,
        label_names=label_names,
    )
    logger.info(
        "build_gallery: saved %d embeddings (%d labels) → %s",
        len(embeddings),
        len(label_names),
        output_path,
    )

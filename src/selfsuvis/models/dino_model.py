from typing import List, Tuple

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.gpu_utils import is_cuda_oom, resolve_device
from selfsuvis.pipeline.core.logging import get_logger

# ---------------------------------------------------------------------------
# Hub constants — shared by all DINO loaders in this codebase.
# facebookresearch/dinov3 does NOT exist on GitHub.  All DINO models
# (including those labelled "dinov3" in this project) are served from the
# facebookresearch/dinov2 repository.
# ---------------------------------------------------------------------------
DINO_HUB_REPO = "facebookresearch/dinov2"

# "dinov3" in this codebase means DINOv2 with register tokens (_reg variants).
# Register tokens are the architectural improvement Facebook released after the
# original DINOv2 paper — they reduce attention artifacts and produce better
# features.  The _reg checkpoints are distinct files from the plain _vitb14 ones.
_DINO_MODEL_ALIAS: dict = {
    "dinov3_vits14": "dinov2_vits14_reg",
    "dinov3_vitb14": "dinov2_vitb14_reg",
    "dinov3_vitl14": "dinov2_vitl14_reg",
    "dinov3_vitg14": "dinov2_vitg14_reg",
}

# dinov2 hub entry-point → Hugging Face repo id (facebook/dinov2-*)
# Used as a fallback when GitHub / torch.hub is unreachable.
# NOTE: Facebook has not published pure _reg backbone repos on HF — only the
# non-reg backbones are available.  _reg variants must be fetched via torch.hub.
_DINO_HF_REPO: dict = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}

_DINO_EMBED_DIM: dict = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# None = untested, True = works, False = failed (skip probe on next load)
_xformers_cuda_ok: "bool | None" = None


def _set_dino_xformers_enabled(enabled: bool) -> None:
    """Toggle dinov2 xFormers attention for already-imported hub modules."""
    import sys

    for mod_name, mod in sys.modules.items():
        if "dinov2" in mod_name and hasattr(mod, "XFORMERS_AVAILABLE"):
            mod.XFORMERS_AVAILABLE = enabled


class _HFDINOWrapper(torch.nn.Module):
    """Wrap a HF ``Dinov2Model`` to match the torch.hub forward interface.

    The torch.hub model returns a ``(B, D)`` tensor directly from ``forward()``.
    The HF model returns a ``BaseModelOutputWithPooling``; we extract the CLS
    token (``last_hidden_state[:, 0]``) which is the same representation.
    """

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self._hf = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) → (B,D)
        out = self._hf(pixel_values=x)
        return out.last_hidden_state[:, 0]


def _load_dino_from_hf(model_name: str) -> torch.nn.Module:
    """Load a DINOv2 backbone from Hugging Face as a fallback.

    Returns a ``_HFDINOWrapper`` whose ``forward()`` is compatible with the
    torch.hub model: accepts ``(B, 3, H, W)`` with ImageNet normalisation and
    returns ``(B, D)`` CLS-token embeddings.

    Raises ``ImportError`` if ``transformers`` is not installed.
    Raises ``KeyError`` if no HF repo is known for *model_name*.
    """
    try:
        from transformers import AutoModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the Hugging Face fallback. "
            "Install it with: pip install transformers"
        ) from exc

    actual_name = _DINO_MODEL_ALIAS.get(model_name, model_name)
    hf_repo = _DINO_HF_REPO.get(actual_name)
    if hf_repo is None:
        raise KeyError(
            f"No Hugging Face repo mapping for DINO model '{actual_name}'. "
            f"Known models: {list(_DINO_HF_REPO)}"
        )

    # Import gated-repo exception — location changed between huggingface_hub versions.
    _GatedRepoError = None
    try:
        from huggingface_hub.errors import GatedRepoError as _GatedRepoError  # ≥0.24
    except ImportError:
        try:
            from huggingface_hub.utils import GatedRepoError as _GatedRepoError  # <0.24
        except ImportError:
            pass

    try:
        hf_model = AutoModel.from_pretrained(hf_repo)
    except Exception as exc:
        if _GatedRepoError and isinstance(exc, _GatedRepoError):
            raise RuntimeError(
                f"\nModel '{hf_repo}' requires license acceptance on Hugging Face.\n"
                f"  1. Open  https://huggingface.co/{hf_repo}\n"
                f"  2. Click 'Agree and access repository'\n"
                f"  3. Log in:  huggingface-cli login\n"
                f"  4. Re-run:  python scripts/prepare_models.py --dino --source hf"
            ) from exc
        raise

    return _HFDINOWrapper(hf_model)


def _resolve_dino_hub(model_name: str) -> Tuple[str, str, str]:
    """Return *(source, repo_or_dir, actual_model_name)* for ``torch.hub.load``.

    Translates ``dinov3_*`` aliases to the real ``dinov2_*`` entry-point names
    and returns ``source='local'`` + the cached directory path when the hub
    archive has already been downloaded, avoiding any network access.
    """
    import torch.hub as _hub

    actual_name = _DINO_MODEL_ALIAS.get(model_name, model_name)
    local_path = os.path.join(_hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(local_path):
        return "local", local_path, actual_name
    return "github", DINO_HUB_REPO, actual_name


_logger = get_logger(__name__)


def hub_load_dino(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    """Load a DINO backbone, with a Hugging Face fallback.

    Resolution order:
    1. Local torch.hub cache (``~/.cache/torch/hub/facebookresearch_dinov2_main``)
    2. GitHub via ``torch.hub.load`` (``facebookresearch/dinov2``)
    3. Hugging Face ``transformers`` (``facebook/dinov2-{variant}``)

    Handles ``dinov3_*`` → ``dinov2_*`` aliasing transparently.
    """
    source, repo_or_dir, actual_name = _resolve_dino_hub(model_name)
    try:
        return torch.hub.load(repo_or_dir, actual_name,
                              pretrained=pretrained, source=source)
    except Exception as hub_exc:
        _logger.warning(
            "torch.hub load failed (%s: %s) — trying Hugging Face fallback …",
            type(hub_exc).__name__, hub_exc,
        )
        try:
            model = _load_dino_from_hf(model_name)
            _logger.info("DINO loaded from Hugging Face: %s", _DINO_HF_REPO.get(actual_name))
            return model
        except Exception as hf_exc:
            raise RuntimeError(
                f"All DINO load attempts failed.\n"
                f"  torch.hub: {hub_exc}\n"
                f"  Hugging Face: {hf_exc}\n"
                f"To pre-download offline: python scripts/prepare_models.py --dino"
            ) from hf_exc


class DINOEmbedder:
    def __init__(self, model_name: str = "dinov2_vitb14"):
        self.logger = get_logger(__name__)
        self.device = self._resolve_device()
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._embed_dim = self._infer_embed_dim(model_name)
        self.logger.info("DINO loaded: %s on %s (dim=%d)", model_name, self.device, self._embed_dim)

    def _resolve_device(self) -> str:
        return resolve_device()

    def _load_model(self, model_name: str):
        import torch.hub as _hub

        source, repo_or_dir, actual_name = _resolve_dino_hub(model_name)

        if source == "local":
            self.logger.info("DINO: loading from local hub cache: %s", repo_or_dir)
        else:
            self.logger.info("DINO: downloading from GitHub (first run) …")

        # Patch torch.hub's downloader to show a tqdm progress bar on first
        # download.  The patch is applied regardless of source so that weight
        # downloads (separate from the repo archive) are also shown.
        _orig_download = _hub.download_url_to_file

        def _download_with_progress(url, dst, *args, **kwargs):
            self.logger.info("  ↓ %s", url)
            try:
                from tqdm import tqdm as _tqdm
                import urllib.request as _req
                with _req.urlopen(url) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                bar = _tqdm(total=total, unit="B", unit_scale=True,
                            desc=f"  {os.path.basename(dst)}", leave=False)

                def _hook(count, block_size, total_size):
                    if total_size > 0:
                        bar.total = total_size
                    bar.update(block_size)

                _req.urlretrieve(url, dst, reporthook=_hook)
                bar.close()
                self.logger.info("  ✓ saved %s", dst)
            except Exception:
                _orig_download(url, dst, *args, **kwargs)

        _hub.download_url_to_file = _download_with_progress
        try:
            # hub_load_dino already handles: local cache → GitHub → HF fallback.
            model = hub_load_dino(model_name, pretrained=True)
        finally:
            _hub.download_url_to_file = _orig_download

        def _load_to_device(use_xformers: bool) -> torch.nn.Module:
            _set_dino_xformers_enabled(use_xformers)
            # fa2 (available on Blackwell) requires fp16/bf16 — cast before .to(device)
            # so xformers attention kernels are eligible from the first probe pass.
            if str(self.device).startswith("cuda") and settings.USE_FP16:
                model.half()
            m = model.to(self.device)
            if str(self.device).startswith("cuda") and use_xformers:
                # Probe a tiny forward pass to detect GPU kernel mismatches.
                # Use the model's actual dtype — fp16 when USE_FP16 is set —
                # otherwise a float32 dummy against fp16 weights raises a
                # dtype mismatch before xformers is even reached.
                import torch as _pt
                _probe_dtype = (
                    _pt.float16
                    if (settings.USE_FP16 and str(self.device).startswith("cuda"))
                    else _pt.float32
                )
                with _pt.no_grad():
                    dummy = _pt.zeros(1, 3, 224, 224, device=self.device, dtype=_probe_dtype)
                    m.eval()
                    m(dummy)
            return m

        global _xformers_cuda_ok
        # If xformers already failed in this process, skip the probe and go
        # straight to SDPA — no warning repeated on subsequent DINOv3 loads.
        if _xformers_cuda_ok is False:
            _prev = os.environ.get("XFORMERS_DISABLED")
            os.environ["XFORMERS_DISABLED"] = "1"
            try:
                model = _load_to_device(use_xformers=False)
            finally:
                if _prev is None:
                    os.environ.pop("XFORMERS_DISABLED", None)
                else:
                    os.environ["XFORMERS_DISABLED"] = _prev
        else:
            try:
                model = _load_to_device(use_xformers=True)
                _xformers_cuda_ok = True
            except (RuntimeError, NotImplementedError) as _exc:
                _exc_str = str(_exc)
                _no_kernel = (
                    "no kernel image" in _exc_str
                    or "cudaErrorNoKernelImageForDevice" in _exc_str
                    or "memory_efficient_attention_forward" in _exc_str
                    or "requires device with capability" in _exc_str
                    or "c10::Half" in _exc_str
                    or "bias type" in _exc_str
                )
                if not _no_kernel:
                    raise
                # Log once per process — subsequent loads skip the probe entirely.
                self.logger.warning(
                    "DINO: flash-attn/xformers not compatible with this GPU "
                    "(capability too new or unsupported dtype) — "
                    "using PyTorch SDPA instead (warning shown once per run)"
                )
                _xformers_cuda_ok = False
                # Move the already-loaded weights back to CPU to clear any partial
                # GPU allocation from the failed probe, then retry without xformers.
                try:
                    model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                _prev = os.environ.get("XFORMERS_DISABLED")
                os.environ["XFORMERS_DISABLED"] = "1"
                try:
                    model = _load_to_device(use_xformers=False)
                finally:
                    if _prev is None:
                        os.environ.pop("XFORMERS_DISABLED", None)
                    else:
                        os.environ["XFORMERS_DISABLED"] = _prev
        # Resolve checkpoint: DINO_CHECKPOINT env var takes priority,
        # then active_checkpoint.txt (written by POST /admin/reload-model).
        ckpt = settings.DINO_CHECKPOINT
        if not ckpt:
            from pathlib import Path
            active_txt = Path(settings.SUP_CHECKPOINT_DIR) / "active_checkpoint.txt"
            if active_txt.exists():
                ckpt = active_txt.read_text().strip()
                if ckpt:
                    self.logger.info("DINO: using active_checkpoint.txt → %s", ckpt)

        if ckpt and os.path.isfile(ckpt):
            import torch as _torch
            state = _torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state)
            self.logger.info("DINO: loaded fine-tuned checkpoint %s", ckpt)
        elif ckpt:
            self.logger.warning("DINO checkpoint set but file not found: %s", ckpt)
        return model

    def _infer_embed_dim(self, model_name: str) -> int:
        actual_name = _DINO_MODEL_ALIAS.get(model_name, model_name)
        dim = _DINO_EMBED_DIM.get(actual_name)
        if dim is not None:
            return dim

        # Fallback for unexpected variants or wrappers without a known static mapping.
        dummy = Image.new("RGB", (224, 224))
        return int(self.encode_images([dummy], batch_size=1).shape[1])

    def load_backbone_checkpoint(self, path: str) -> None:
        """Hot-swap backbone weights in-place.

        Loads a backbone state_dict from path and replaces the current model weights.
        Python reference assignment to app.state.dino_model is GIL-atomic, so
        in-flight inference calls that captured the old reference complete normally.

        Raises RuntimeError on any load failure (caller's model is unchanged).
        """
        import torch as _torch
        state = _torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        _set_dino_xformers_enabled(str(self.device).startswith("cuda"))
        self.logger.info("DINO: hot-swapped backbone checkpoint %s", path)

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            try:
                actual_device = next(self.model.parameters()).device
            except StopIteration:
                actual_device = torch.device(self.device)
            try:
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(actual_device)
                # On CPU the model may still be in FP16 (offloaded from CUDA); cast to match.
                if not str(actual_device).startswith("cuda"):
                    _mdtype = next(self.model.parameters()).dtype
                    if tensors.dtype != _mdtype:
                        tensors = tensors.to(_mdtype)
                with torch.no_grad():
                    try:
                        if settings.USE_FP16 and str(actual_device).startswith("cuda"):
                            with torch.amp.autocast("cuda"):
                                feats = self.model(tensors)
                        else:
                            feats = self.model(tensors)
                    except NotImplementedError as exc:
                        if "memory_efficient_attention_forward" not in str(exc):
                            raise
                        self.logger.warning(
                            "DINO: xFormers attention unavailable on %s; retrying with PyTorch attention",
                            actual_device,
                        )
                        _set_dino_xformers_enabled(False)
                        feats = self.model(tensors)
            except Exception as exc:
                if not is_cuda_oom(exc) or not str(actual_device).startswith("cuda"):
                    raise
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner
                log_oom_banner(self.logger, "DINOv3 image encoding", "moving backbone to CPU for remaining batches")
                self.model.cpu()
                _set_dino_xformers_enabled(False)
                actual_device = torch.device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(actual_device)
                with torch.no_grad():
                    feats = self.model(tensors)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.detach().cpu().numpy())
        if not embeddings:
            return np.zeros((0, self._embed_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def image_dim(self) -> int:
        return self._embed_dim



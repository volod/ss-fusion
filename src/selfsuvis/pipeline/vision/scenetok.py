"""SceneTok streaming scene encoder + segmentation decoder client.

SceneTok (arxiv 2602.18882) compresses a set of multi-view frames into a small
set of permutation-invariant latent tokens using a multi-view encoder, then
renders novel viewpoints or segmentation masks via a rectified flow decoder.

Two runtime modes:
  sidecar  — HTTP calls to a thin FastAPI wrapper
             (python -m selfsuvis.scripts.scenetok_server).
             Set SCENETOK_API_URL in .env.  Preferred on single-GPU setups.
  local    — loads the SceneTok checkpoint in-process via the scenetok package.
             Falls back gracefully if the package or checkpoint is absent.

The segmentation decoder replaces the RGB head with a mask prediction head.
Base checkpoints produce novel-view RGB renders; pass a fine-tuned checkpoint
to get per-frame segmentation masks from SCENETOK_MODE=masks.
"""

import base64
import gc
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings

logger = get_logger(__name__)

_SIDECAR_PROCESS_ENDPOINT = "/process"
_SIDECAR_HEALTH_ENDPOINT = "/health"


def _encode_image_b64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_b64_png(b64: str) -> Optional[np.ndarray]:
    """Decode a base64 PNG string into an HxWxC uint8 array."""
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        logger.debug("SceneTok: PNG decode failed: %s", exc)
        return None


def _decode_b64_npz(b64: str) -> Optional[np.ndarray]:
    try:
        data = base64.b64decode(b64)
        npz = np.load(io.BytesIO(data))
        key = list(npz.files)[0]
        return npz[key]
    except Exception as exc:
        logger.debug("SceneTok: token npz decode failed: %s", exc)
        return None


def _vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


class SceneTokModel:
    """SceneTok streaming scene encoder + segmentation decoder.

    Prefers a running sidecar (SCENETOK_API_URL); falls back to loading the
    checkpoint in-process when the sidecar is absent.  Requires ~24 GB VRAM
    for local inference; skips automatically on smaller cards unless sidecar is
    configured.
    """

    _MIN_LOCAL_VRAM_GB = 20.0

    def __init__(self) -> None:
        self._encoder = None
        self._decoder = None
        self._device = resolve_device()
        self._load_failed = False

    # ── availability ──────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        if not getattr(settings, "SCENETOK_ENABLED", False):
            return False
        if getattr(settings, "SCENETOK_API_URL", ""):
            return True
        if _vram_gb() >= self._MIN_LOCAL_VRAM_GB:
            return self._scenetok_package_available()
        return False

    def _scenetok_package_available(self) -> bool:
        try:
            import importlib
            return importlib.util.find_spec("scenetok") is not None
        except Exception:
            return False

    # ── sidecar mode ──────────────────────────────────────────────────────────

    def _sidecar_healthy(self) -> bool:
        api_url = str(getattr(settings, "SCENETOK_API_URL", "") or "")
        if not api_url:
            return False
        try:
            import httpx
            resp = httpx.get(
                f"{api_url.rstrip('/')}{_SIDECAR_HEALTH_ENDPOINT}",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _call_sidecar(
        self,
        frames: List[Tuple[str, float]],
        images: List[Image.Image],
        mode: str,
    ) -> Dict[str, Any]:
        api_url = str(getattr(settings, "SCENETOK_API_URL", "") or "")
        checkpoint = str(getattr(settings, "SCENETOK_CHECKPOINT", "va-videodc_re10k") or "va-videodc_re10k")
        timeout = float(getattr(settings, "SCENETOK_TIMEOUT_SEC", 300) or 300)

        payload = {
            "frames": [
                {"t_sec": t_sec, "b64_jpeg": _encode_image_b64(img)}
                for (_, t_sec), img in zip(frames, images)
            ],
            "checkpoint": checkpoint,
            "mode": mode,
        }
        try:
            import httpx
            resp = httpx.post(
                f"{api_url.rstrip('/')}{_SIDECAR_PROCESS_ENDPOINT}",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("SceneTok sidecar call failed: %s", exc)
            return {"service_unavailable": True, "reason": str(exc)}

    # ── local torch mode ──────────────────────────────────────────────────────

    def _load_local(self):
        if self._encoder is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from scenetok import SceneTokEncoder, SceneTokDecoder  # type: ignore[import]

            checkpoint = str(getattr(settings, "SCENETOK_CHECKPOINT", "va-videodc_re10k") or "va-videodc_re10k")
            cache_dir = Path.home() / ".cache" / "selfsuvis" / "scenetok"
            ckpt_path = cache_dir / f"{checkpoint}.ckpt"
            if not ckpt_path.exists():
                logger.warning(
                    "SceneTok checkpoint not found at %s — run "
                    "scripts/prepare_models.py --scenetok to download it",
                    ckpt_path,
                )
                self._load_failed = True
                return False

            dtype = torch.float16 if self._device != "cpu" and getattr(settings, "USE_FP16", True) else torch.float32
            self._encoder = SceneTokEncoder.from_checkpoint(ckpt_path, dtype=dtype).to(self._device).eval()
            self._decoder = SceneTokDecoder.from_checkpoint(ckpt_path, dtype=dtype).to(self._device).eval()
            logger.info("SceneTok local model loaded: %s on %s", checkpoint, self._device)
            return True
        except ImportError:
            logger.info(
                "scenetok package not installed — local mode unavailable. "
                "Set SCENETOK_API_URL for sidecar mode or install scenetok."
            )
            self._load_failed = True
            return False
        except Exception as exc:
            logger.warning("SceneTok local model load failed: %s", exc)
            self._load_failed = True
            return False

    def _run_local(
        self,
        frames: List[Tuple[str, float]],
        images: List[Image.Image],
        mode: str,
    ) -> Dict[str, Any]:
        if not self._load_local():
            return {"service_unavailable": True, "reason": "local model unavailable"}
        try:
            import torch

            with torch.no_grad():
                tokens = self._encoder.encode(images, device=self._device)
                results = self._decoder.decode(
                    tokens,
                    timestamps=[t for _, t in frames],
                    mode=mode,
                )

            tokens_buf = io.BytesIO()
            np.savez_compressed(tokens_buf, tokens=tokens.cpu().float().numpy())
            tokens_b64 = base64.b64encode(tokens_buf.getvalue()).decode("ascii")

            decoded_results = []
            for (_, t_sec), mask_or_view in zip(frames, results):
                arr = mask_or_view.cpu().numpy()
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                img_out = Image.fromarray(arr)
                buf = io.BytesIO()
                img_out.save(buf, format="PNG")
                decoded_results.append({
                    "t_sec": t_sec,
                    "b64_png": base64.b64encode(buf.getvalue()).decode("ascii"),
                })

            return {
                "tokens_b64_npz": tokens_b64,
                "n_tokens": int(tokens.shape[-2]) if tokens.ndim >= 2 else 0,
                "results": decoded_results,
            }
        except Exception as exc:
            logger.warning("SceneTok local inference failed: %s", exc, exc_info=True)
            return {"service_unavailable": True, "reason": str(exc)}

    # ── public API ────────────────────────────────────────────────────────────

    def encode_decode(
        self,
        frames: List[Tuple[str, float]],
        images: List[Image.Image],
        mode: str = "masks",
    ) -> Dict[str, Any]:
        """Encode frames into scene tokens then decode to masks or novel views.

        Args:
            frames:  list of (frame_path, t_sec) pairs (same length as images).
            images:  corresponding PIL images.
            mode:    "masks" (segmentation decoder) or "rgb" (novel view decoder).

        Returns dict with keys:
            tokens_b64_npz   base64-encoded npz of scene tokens
            n_tokens         number of latent tokens
            results          list of {"t_sec": float, "b64_png": str}
            service_unavailable  True on failure (optional key)
        """
        if not self.is_enabled():
            return {"service_unavailable": True, "reason": "SceneTok disabled"}

        api_url = str(getattr(settings, "SCENETOK_API_URL", "") or "")
        if api_url:
            if self._sidecar_healthy():
                return self._call_sidecar(frames, images, mode)
            logger.warning(
                "SceneTok sidecar at %s is not responding — falling back to local torch", api_url
            )

        return self._run_local(frames, images, mode)

    def release(self) -> None:
        self._encoder = None
        self._decoder = None
        self._load_failed = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["SceneTokModel"]

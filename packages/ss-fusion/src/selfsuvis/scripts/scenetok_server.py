"""Thin FastAPI wrapper around local SceneTok inference.

Exposes:
  GET  /health   — liveness / readiness probe
  POST /process  — encode frames into SceneTok tokens and decode masks or RGB

The local pipeline client in ``selfsuvis.pipeline.vision.scenetok`` expects
this exact contract.
"""

import asyncio
import base64
import gc
import io
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from selfsuvis.pipeline.core import resolve_device, settings

app = FastAPI(title="selfsuvis-scenetok", version="1.0.0")


def _decode_b64_jpeg(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _encode_png_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _checkpoint_path(name: str) -> Path:
    data_dir = Path(getattr(settings, "DATA_DIR", "./.data"))
    cache_dir = Path(os.getenv("CACHE_DIR", str(data_dir / ".cache"))) / "selfsuvis" / "scenetok"
    ckpt = name if name.endswith(".ckpt") else f"{name}.ckpt"
    return cache_dir / ckpt


class FramePayload(BaseModel):
    t_sec: float = Field(..., description="Frame timestamp in seconds.")
    b64_jpeg: str = Field(..., description="Base64-encoded JPEG frame.")


class ProcessRequest(BaseModel):
    frames: list[FramePayload] = Field(default_factory=list)
    checkpoint: str = Field(
        default_factory=lambda: str(
            getattr(settings, "SCENETOK_CHECKPOINT", "va-videodc_re10k") or "va-videodc_re10k"
        )
    )
    mode: Literal["masks", "rgb"] = Field(
        default_factory=lambda: str(getattr(settings, "SCENETOK_MODE", "masks") or "masks")
    )


class DecodedFrame(BaseModel):
    t_sec: float
    b64_png: str


class ProcessResponse(BaseModel):
    tokens_b64_npz: str
    n_tokens: int
    results: list[DecodedFrame]


class SceneTokService:
    """Lazy-loading SceneTok runner with one active checkpoint at a time."""

    def __init__(self) -> None:
        self._encoder = None
        self._decoder = None
        self._checkpoint = ""
        self._device = resolve_device()
        self._lock = asyncio.Lock()

    def _dtype(self):
        import torch

        return (
            torch.float16
            if self._device != "cpu" and getattr(settings, "USE_FP16", True)
            else torch.float32
        )

    def _release_locked(self) -> None:
        self._encoder = None
        self._decoder = None
        self._checkpoint = ""
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

    def _load_locked(self, checkpoint: str) -> None:
        if (
            self._encoder is not None
            and self._decoder is not None
            and self._checkpoint == checkpoint
        ):
            return

        ckpt_path = _checkpoint_path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SceneTok checkpoint not found at {ckpt_path}. "
                "Run `APP_ENV=dev .venv/bin/python -m selfsuvis.scripts.prepare_models --scenetok` first."
            )

        try:
            from scenetok import SceneTokDecoder, SceneTokEncoder  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "scenetok package is not installed. Install the upstream package before starting the sidecar."
            ) from exc

        if self._checkpoint and self._checkpoint != checkpoint:
            self._release_locked()

        dtype = self._dtype()
        self._encoder = (
            SceneTokEncoder.from_checkpoint(ckpt_path, dtype=dtype).to(self._device).eval()
        )
        self._decoder = (
            SceneTokDecoder.from_checkpoint(ckpt_path, dtype=dtype).to(self._device).eval()
        )
        self._checkpoint = checkpoint

    def _run_locked(self, req: ProcessRequest) -> dict[str, Any]:
        import torch

        if not req.frames:
            raise ValueError("No frames provided.")

        self._load_locked(req.checkpoint)

        images: list[Image.Image] = []
        timestamps: list[float] = []
        for frame in req.frames:
            images.append(_decode_b64_jpeg(frame.b64_jpeg))
            timestamps.append(frame.t_sec)

        with torch.no_grad():
            tokens = self._encoder.encode(images, device=self._device)
            outputs = self._decoder.decode(tokens, timestamps=timestamps, mode=req.mode)

        tokens_np = tokens.detach().cpu().float().numpy()
        tokens_buf = io.BytesIO()
        np.savez_compressed(tokens_buf, tokens=tokens_np)

        results: list[dict[str, Any]] = []
        for t_sec, item in zip(timestamps, outputs):
            arr = item.detach().cpu().numpy()
            results.append({"t_sec": t_sec, "b64_png": _encode_png_b64(arr)})

        return {
            "tokens_b64_npz": base64.b64encode(tokens_buf.getvalue()).decode("ascii"),
            "n_tokens": int(tokens.shape[-2]) if getattr(tokens, "ndim", 0) >= 2 else 0,
            "results": results,
        }

    async def health(self) -> dict[str, Any]:
        checkpoint = str(
            getattr(settings, "SCENETOK_CHECKPOINT", "va-videodc_re10k") or "va-videodc_re10k"
        )
        ckpt_path = _checkpoint_path(checkpoint)
        return {
            "status": "ok",
            "service": "scenetok",
            "device": self._device,
            "loaded_checkpoint": self._checkpoint,
            "default_checkpoint": checkpoint,
            "checkpoint_exists": ckpt_path.exists(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }

    async def process(self, req: ProcessRequest) -> dict[str, Any]:
        async with self._lock:
            try:
                return await asyncio.to_thread(self._run_locked, req)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"SceneTok inference failed: {exc}"
                ) from exc


SERVICE = SceneTokService()


@app.get("/health")
async def health() -> dict[str, Any]:
    return await SERVICE.health()


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    out = await SERVICE.process(req)
    return ProcessResponse(**out)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8040"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

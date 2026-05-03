"""Startup preflight checks for local and production runs."""

import importlib.util
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.scripts import prepare_models as model_prep

logger = get_logger(__name__)


@dataclass
class PreflightReport:
    """Accumulated startup preflight findings."""

    scope: str
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_check(self, message: str) -> None:
        self.checks.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        summary = "\n".join(f"- {item}" for item in self.errors)
        raise RuntimeError(f"{self.scope} preflight failed:\n{summary}")


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _check_module(report: PreflightReport, module_name: str, hint: str) -> None:
    if _has_module(module_name):
        report.add_check(f"python module present: {module_name}")
    else:
        report.add_error(f"missing Python dependency '{module_name}' ({hint})")


def _check_cached(report: PreflightReport, label: str, check_fn: Any, *, hint: str) -> None:
    try:
        cached = bool(check_fn())
    except Exception:
        cached = False
    if cached:
        report.add_check(f"cached: {label}")
    else:
        report.add_error(f"missing cached artifact for {label} ({hint})")


def _check_tcp_service(report: PreflightReport, label: str, host: str, port: int, *, fatal: bool) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            report.add_check(f"service reachable: {label} ({host}:{port})")
            return
    except OSError:
        pass
    message = f"{label} is not reachable at {host}:{port}"
    if fatal:
        report.add_error(message)
    else:
        report.add_warning(message)


def _is_local_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


def _resolve_auto_model(task: str, override: str) -> str:
    override = (override or "").strip()
    if override and override.lower() != "auto":
        return override
    if task == "asr":
        from selfsuvis.pipeline.vision.registry import auto_select, detect_resources

        return auto_select("asr", detect_resources()) or "openai/whisper-large-v3-turbo"
    return model_prep._resolve_hf_model(task, "")


def _check_ollama_sidecar_model(
    report: PreflightReport,
    *,
    label: str,
    url: str,
    backend: str,
    model_name: str,
) -> None:
    if not url or not _is_local_url(url):
        return
    if backend.lower() != "ollama" and "11434" not in url:
        return
    _check_cached(
        report,
        f"{label} Ollama model {model_name}",
        lambda: model_prep._is_ollama_model_cached(model_name),
        hint="run `python -m selfsuvis.scripts.prepare_models --reasoning/--unidrive` or `ollama pull`",
    )


def _check_vllm_sidecar_model(
    report: PreflightReport,
    *,
    label: str,
    url: str,
    backend: str,
    model_name: str,
) -> None:
    if not url or not _is_local_url(url):
        return
    if backend.lower() != "vllm":
        return
    if "/" not in model_name:
        return
    _check_cached(
        report,
        f"{label} vLLM model {model_name}",
        lambda: model_prep._is_hf_cached(model_name),
        hint="run `python -m selfsuvis.scripts.prepare_models --all` or cache the model manually",
    )


def run_local_preflight(args: Any) -> PreflightReport:
    """Verify local pipeline requirements before the run starts."""
    report = PreflightReport(scope="local pipeline")

    _check_module(report, "open_clip", "required by OpenCLIP frame embedding")
    _check_module(report, "transformers", "required by HuggingFace-backed local models")
    _check_module(report, "timm", "required by EfficientViT stage-2 distillation")
    _check_cached(
        report,
        f"OpenCLIP {settings.OPENCLIP_MODEL}/{settings.OPENCLIP_PRETRAINED}",
        lambda: model_prep._is_openclip_cached(settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED),
        hint="run `python -m selfsuvis.scripts.prepare_models --clip`",
    )
    _check_cached(
        report,
        "DINOv2/v3 torch hub archive",
        lambda: model_prep._is_dino_hub_cached("dinov3_vitb14"),
        hint="run `python -m selfsuvis.scripts.prepare_models --dino`",
    )

    if getattr(args, "asr", False):
        asr_model = _resolve_auto_model("asr", getattr(args, "asr_model", "") or settings.ASR_MODEL)
        _check_cached(
            report,
            f"ASR {asr_model}",
            lambda model=asr_model: model_prep._is_hf_cached(model),
            hint="run `python -m selfsuvis.scripts.prepare_models --whisper`",
        )

    if getattr(args, "ocr", False) and not settings.OCR_API_URL:
        ocr_model = _resolve_auto_model("ocr", getattr(args, "ocr_model", "") or settings.OCR_MODEL)
        if ocr_model:
            _check_cached(
                report,
                f"OCR {ocr_model}",
                lambda model=ocr_model: model_prep._is_hf_cached(model),
                hint="run `python -m selfsuvis.scripts.prepare_models --ocr`",
            )

    if getattr(args, "depth", False):
        depth_model = _resolve_auto_model("depth", getattr(args, "depth_model", "") or settings.DEPTH_MODEL)
        if depth_model:
            _check_cached(
                report,
                f"Depth {depth_model}",
                lambda model=depth_model: model_prep._is_hf_cached(model),
                hint="run `python -m selfsuvis.scripts.prepare_models --depth`",
            )

    if getattr(args, "detection", False):
        detection_model = _resolve_auto_model(
            "detection",
            getattr(args, "detection_model", "") or settings.DETECTION_MODEL,
        )
        if detection_model:
            _check_cached(
                report,
                f"Detection {detection_model}",
                lambda model=detection_model: model_prep._is_hf_cached(model),
                hint="run `python -m selfsuvis.scripts.prepare_models --detection`",
            )
        if settings.YOLO_ENABLED:
            _check_module(report, "ultralytics", "required by YOLO+SAM and drone-detection training")
            _check_cached(
                report,
                "YOLO11 weights",
                lambda: model_prep._is_yolo_cached(settings.YOLO_MODEL),
                hint="run `python -m selfsuvis.scripts.prepare_models --yolo`",
            )
        if settings.SAM_ENABLED:
            sam_model = settings.SAM_MODEL if settings.SAM_MODEL != "auto" else "facebook/sam2-hiera-large"
            _check_cached(
                report,
                f"SAM {sam_model}",
                lambda model=sam_model: model_prep._is_hf_cached(model)
                or model_prep._is_hf_cached("facebook/sam2-hiera-large"),
                hint="run `python -m selfsuvis.scripts.prepare_models --sam`",
            )

    if getattr(args, "world_model", False):
        world_model = _resolve_auto_model(
            "world_model",
            getattr(args, "world_model_id", "") or settings.WORLD_MODEL,
        )
        if world_model:
            _check_cached(
                report,
                f"World model {world_model}",
                lambda model=world_model: model_prep._is_hf_cached(model),
                hint="run `python -m selfsuvis.scripts.prepare_models --world-model`",
            )

    qwen_enabled = getattr(args, "qwen", False) or bool(settings.QWEN_API_URL)
    if qwen_enabled:
        _check_ollama_sidecar_model(
            report,
            label="Qwen",
            url=getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL,
            backend=getattr(args, "qwen_backend", "") or settings.QWEN_BACKEND,
            model_name=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
        )
        _check_vllm_sidecar_model(
            report,
            label="Qwen",
            url=getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL,
            backend=getattr(args, "qwen_backend", "") or settings.QWEN_BACKEND,
            model_name=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
        )

    if settings.GEMMA_API_URL:
        _check_ollama_sidecar_model(
            report,
            label="Gemma",
            url=getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
            backend=getattr(args, "gemma_api_backend", "") or settings.GEMMA_API_BACKEND,
            model_name=getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
        )
        _check_vllm_sidecar_model(
            report,
            label="Gemma",
            url=getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
            backend=getattr(args, "gemma_api_backend", "") or settings.GEMMA_API_BACKEND,
            model_name=getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
        )

    if settings.REASONING_API_URL:
        _check_ollama_sidecar_model(
            report,
            label="Reasoning",
            url=getattr(args, "reasoning_api_url", "") or settings.REASONING_API_URL,
            backend=getattr(args, "reasoning_backend", "") or settings.REASONING_BACKEND,
            model_name=getattr(args, "reasoning_model", "") or settings.REASONING_MODEL,
        )

    if getattr(args, "unidrive", False):
        unidrive_url = getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL
        unidrive_backend = getattr(args, "unidrive_backend", "") or settings.UNIDRIVE_BACKEND
        unidrive_model = getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL
        if unidrive_url:
            _check_ollama_sidecar_model(
                report,
                label="UniDrive",
                url=unidrive_url,
                backend=unidrive_backend,
                model_name=unidrive_model,
            )
            _check_vllm_sidecar_model(
                report,
                label="UniDrive",
                url=unidrive_url,
                backend=unidrive_backend,
                model_name=unidrive_model,
            )
        else:
            _check_cached(
                report,
                f"UniDrive {unidrive_model}",
                lambda model=unidrive_model: model_prep._is_hf_cached(model),
                hint="run `python -m selfsuvis.scripts.prepare_models --unidrive`",
            )

    if getattr(args, "scenetok", False) or settings.SCENETOK_ENABLED:
        if settings.SCENETOK_API_URL:
            report.add_check("SceneTok sidecar configured")
        else:
            try:
                from selfsuvis.pipeline.vision.scenetok import SceneTokModel

                min_local_vram_gb = float(getattr(SceneTokModel, "_MIN_LOCAL_VRAM_GB", 20.0))
            except Exception:
                min_local_vram_gb = 20.0
            try:
                from selfsuvis.pipeline.vision.registry import detect_vram_gb

                total_vram_gb = float(detect_vram_gb() or 0.0)
            except Exception:
                total_vram_gb = 0.0

            if total_vram_gb < min_local_vram_gb:
                report.add_warning(
                    "SceneTok is enabled but local execution is not possible on this GPU "
                    f"({total_vram_gb:.1f} GiB < {min_local_vram_gb:.1f} GiB) and no SCENETOK_API_URL is set; "
                    "the step will be skipped"
                )
            else:
                if _can_import("scenetok"):
                    report.add_check("python module importable: scenetok")
                else:
                    report.add_error(
                        "missing importable Python module 'scenetok' "
                        "(required for local in-process SceneTok on >=20 GiB GPUs)"
                    )
                _check_cached(
                    report,
                    f"SceneTok {settings.SCENETOK_CHECKPOINT}",
                    lambda: model_prep._is_scenetok_cached(settings.SCENETOK_CHECKPOINT),
                    hint="run `python -m selfsuvis.scripts.prepare_models --scenetok`",
                )

    if getattr(args, "drone_detection", False):
        _check_module(report, "ultralytics", "required by drone-detection training")
        _check_module(report, "onnxruntime", "required by ONNX quantization/export checks")
        _check_module(report, "onnxslim", "required to avoid mid-run Ultralytics auto-install during ONNX export")
        _check_cached(
            report,
            "YOLOv8n training weights",
            lambda: model_prep._is_yolo_cached("yolov8n"),
            hint="place yolov8n.pt in ~/.cache/ultralytics or warm it once before the run",
        )
        drone_cache = Path(args.output_dir).resolve() / "_drone_detection_cache"
        if not (drone_cache / "train_images").exists():
            report.add_warning(
                "drone-detection dataset cache is empty; the first run will still download the Seraphim batch"
            )

    _check_tcp_service(
        report,
        "Qdrant",
        settings.QDRANT_HOST,
        settings.QDRANT_PORT,
        fatal=False,
    )

    mapper_url = settings.MAPPER_API_URL
    if mapper_url:
        parsed = urlparse(mapper_url)
        if parsed.hostname and parsed.port:
            _check_tcp_service(report, "Mapper API", parsed.hostname, parsed.port, fatal=False)

    return report


def run_production_preflight(component: str) -> PreflightReport:
    """Verify the configured production runtime before startup."""
    report = PreflightReport(scope=f"{component} runtime")

    if settings.MODEL_NAME == "openclip":
        _check_module(report, "open_clip", "required by MODEL_NAME=openclip")
        _check_cached(
            report,
            f"OpenCLIP {settings.OPENCLIP_MODEL}/{settings.OPENCLIP_PRETRAINED}",
            lambda: model_prep._is_openclip_cached(settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED),
            hint="run `python -m selfsuvis.scripts.prepare_models --clip`",
        )
    elif settings.MODEL_NAME in {"dinov2", "dinov3"}:
        _check_cached(
            report,
            "DINOv2/v3 torch hub archive",
            lambda: model_prep._is_dino_hub_cached(settings.MODEL_NAME),
            hint="run `python -m selfsuvis.scripts.prepare_models --dino`",
        )
    elif settings.MODEL_NAME == "gemma":
        _check_cached(
            report,
            f"Gemma {settings.GEMMA_MODEL_ID}",
            lambda: model_prep._is_gemma_cached(settings.GEMMA_MODEL_ID),
            hint="run `python -m selfsuvis.scripts.prepare_models --gemma`",
        )

    if settings.QDRANT_HOST and settings.QDRANT_PORT:
        _check_tcp_service(report, "Qdrant", settings.QDRANT_HOST, settings.QDRANT_PORT, fatal=False)

    return report


def log_preflight(report: PreflightReport) -> None:
    """Log a preflight report in a compact startup-friendly format."""
    logger.info(
        "%s preflight: %d checks, %d warnings, %d errors",
        report.scope,
        len(report.checks),
        len(report.warnings),
        len(report.errors),
    )
    for item in report.warnings:
        logger.warning("preflight: %s", item)
    for item in report.errors:
        logger.error("preflight: %s", item)

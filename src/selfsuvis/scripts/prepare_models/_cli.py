"""CLI: argument parser, main entry point, and completion summary."""

import argparse
import os
import sys

from selfsuvis.pipeline.core.logging import get_logger

from ._auth import _with_auth_retry
from ._cache import _is_hf_cached, _is_openclip_cached, _is_dino_hub_cached, _is_florence2_complete, _verify_models
from ._dino import _download_dino
from ._downloaders import (
    _download_depth,
    _download_detection,
    _download_florence,
    _download_ocr,
    _download_openclip,
    _download_unidrive,
    _download_whisper,
    _download_world_model,
)
from ._flash_attn import _install_flash_attn
from ._gemma import _download_gemma, _is_gemma_cached
from ._ollama import (
    _REASONING_DEFAULT_MODEL,
    _UNIDRIVE_DEFAULT_MODEL,
    _download_ollama_model,
    _is_ollama_model_cached,
    _resolve_unidrive_backend,
    _resolve_unidrive_prepare_model,
)
from ._special import (
    _SCENETOK_CHECKPOINT_VARIANTS,
    _SCENETOK_DEFAULT_CHECKPOINT,
    _download_sam,
    _download_scenetok,
    _download_yolo,
    _is_scenetok_cached,
    _is_yolo_cached,
)

log = get_logger("prepare_models")


def _resolve_hf_model(task: str, override: str) -> str:
    """Return the model ID to use: *override* if set, else auto-select by VRAM."""
    mid = override.strip()
    if mid:
        return mid
    from selfsuvis.pipeline.vision.registry import auto_select, detect_resources

    selected = auto_select(task, detect_resources())
    if selected:
        log.info("%s auto-selected model: %s", task, selected)
    return selected or ""


def _default_all_if_no_selection(args: argparse.Namespace) -> argparse.Namespace:
    selected = (
        args.clip
        or args.dino
        or args.gemma
        or args.flash_attn
        or args.whisper
        or args.florence
        or args.ocr
        or args.depth
        or args.detection
        or args.world_model
        or args.unidrive
        or args.reasoning
        or args.yolo
        or args.sam
        or args.scenetok
        or args.all
    )
    if not selected:
        args.all = True
    return args


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-download all model weights for selfsuvis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clip", action="store_true", help="Download OpenCLIP weights")
    p.add_argument("--dino", action="store_true", help="Download DINOv2/v3 hub weights")
    p.add_argument(
        "--gemma",
        action="store_true",
        help="Download Gemma open-weight model (step 03; requires HF_TOKEN for gated access)",
    )
    _default_gemma = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    p.add_argument(
        "--gemma-model",
        default=_default_gemma,
        metavar="MODEL_ID",
        help=(
            "Gemma model repo ID to cache (requires HF_TOKEN in .env and license accepted). "
            "Multimodal (vision+text): google/gemma-3-4b-it (~8 GiB, default), "
            "google/gemma-3-12b-it (~24 GiB). "
            "Text-only: google/gemma-3-1b-it (~2 GiB, no image encoding). "
            "Ollama sidecar: gemma4:e4b (set GEMMA_API_URL=http://localhost:11434/v1)."
        ),
    )
    p.add_argument(
        "--flash-attn",
        action="store_true",
        help="Install flash-attn (CUDA required; uses prebuilt wheel or compiles)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download/verify everything (default when no other flag is given)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Check cache status for all requested models without downloading",
    )
    p.add_argument(
        "--device",
        default=os.getenv("DEVICE", "auto"),
        choices=["cpu", "cuda", "auto"],
        help="Torch device for weight loading",
    )
    _default_dino = os.getenv("DINO_MODEL", "dinov2_vitb14,dinov3_vitb14").split(",")
    p.add_argument(
        "--dino-model",
        nargs="+",
        default=_default_dino,
        metavar="MODEL",
        help="DINO model names to warm up",
    )
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "hub", "hf"],
        help="DINO weight source: 'auto' = local → GitHub → HF",
    )
    p.add_argument("--whisper", action="store_true", help="Download Whisper ASR model (step 05)")
    p.add_argument(
        "--whisper-model",
        default=os.getenv("ASR_MODEL", "auto"),
        metavar="MODEL_ID",
        help="Whisper/ASR model ID to cache, or 'auto' to match runtime auto-selection",
    )
    p.add_argument(
        "--florence", action="store_true", help="Download Florence-2 captioning model (step 04)"
    )
    _default_florence = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")
    p.add_argument(
        "--florence-model",
        default=_default_florence,
        metavar="MODEL_ID",
        help="Florence-2 model ID to cache",
    )
    p.add_argument(
        "--ocr", action="store_true", help="Download OCR model (step 06; auto-selects by VRAM)"
    )
    p.add_argument(
        "--ocr-model",
        default="",
        metavar="MODEL_ID",
        help=(
            "OCR model ID to cache. Empty = auto-select by VRAM. "
            "Examples: microsoft/trocr-base-printed, ucaslcl/GOT-OCR2_0, "
            "microsoft/Phi-3.5-vision-instruct"
        ),
    )
    p.add_argument(
        "--depth",
        action="store_true",
        help="Download depth estimation model (step 07; auto-selects by VRAM)",
    )
    p.add_argument(
        "--depth-model",
        default="",
        metavar="MODEL_ID",
        help=(
            "Depth model ID to cache. Empty = auto-select by VRAM. "
            "Examples: depth-anything/Depth-Anything-V2-Small-hf, "
            "depth-anything/Depth-Anything-V2-Large-hf"
        ),
    )
    p.add_argument(
        "--detection",
        action="store_true",
        help="Download object detection model (step 08; auto-selects by VRAM)",
    )
    p.add_argument(
        "--detection-model",
        default="",
        metavar="MODEL_ID",
        help=(
            "Detection model ID to cache. Empty = auto-select by VRAM. "
            "Examples: PekingU/rtdetr_r50vd, IDEA-Research/grounding-dino-base"
        ),
    )
    p.add_argument(
        "--world-model",
        action="store_true",
        help="Download world model for video embeddings (step 11; auto-selects by VRAM)",
    )
    p.add_argument(
        "--world-model-id",
        default="",
        metavar="MODEL_ID",
        help=(
            "World model ID to cache. Empty = auto-select by VRAM. "
            "Examples: MCG-NJU/videomae-base, facebook/vjepa2-vitl-fpc64-256"
        ),
    )
    _default_unidrive = os.getenv("UNIDRIVE_MODEL", _UNIDRIVE_DEFAULT_MODEL)
    p.add_argument(
        "--unidrive", action="store_true", help="Download UniDriveVLA expert model assets (step 13)"
    )
    p.add_argument(
        "--unidrive-model",
        default=_default_unidrive,
        metavar="MODEL_ID",
        help=(
            "UniDriveVLA model repo ID to cache for external bridge / sidecar use. "
            f"Default: {_UNIDRIVE_DEFAULT_MODEL}"
        ),
    )
    p.add_argument(
        "--unidrive-backend",
        default=os.getenv("UNIDRIVE_BACKEND", "auto"),
        choices=["auto", "ollama", "vllm"],
        help=(
            "Backend used for UniDrive prep. "
            "'ollama' pulls an Ollama tag, 'vllm' caches HF weights, "
            "'auto' prefers vllm for HF UniDrive repos and Ollama for Ollama tags."
        ),
    )
    _default_reasoning = os.getenv("REASONING_MODEL", _REASONING_DEFAULT_MODEL)
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="Pull the Ollama reasoning model used by the agentic-flow audit (step 24)",
    )
    p.add_argument(
        "--reasoning-model",
        default=_default_reasoning,
        metavar="OLLAMA_TAG",
        help=(
            "Ollama tag to pull for the reasoning/audit step. "
            f"Default: {_REASONING_DEFAULT_MODEL} (~8 GB). "
            "Alternative: deepseek-r1:14b (~9 GB). "
            "Set REASONING_MODEL env var to override the default."
        ),
    )
    p.add_argument(
        "--yolo",
        action="store_true",
        help="Download YOLO11 detection model (step 09; default model: yolo11l.pt ~48 MB)",
    )
    p.add_argument(
        "--yolo-model",
        default="yolo11l",
        metavar="MODEL",
        help=(
            "YOLO model filename to cache (without .pt extension). "
            "Default: yolo11l (~48 MB, 25.3 M params). "
            "Options: yolo11n (6 MB) | yolo11s (18 MB) | yolo11m (38 MB) "
            "| yolo11l (48 MB) | yolo11x (109 MB)"
        ),
    )
    p.add_argument(
        "--sam",
        action="store_true",
        help="Download SAM3/SAM2 segmentation model (step 09; tries sam3 then sam2 fallback)",
    )
    p.add_argument(
        "--sam-model",
        default="facebook/sam3",
        metavar="MODEL_ID",
        help=(
            "SAM model repo ID to cache. "
            "Default: facebook/sam3 (falls back to facebook/sam2-hiera-large if access not granted). "
            "Options: facebook/sam3 | facebook/sam2-hiera-large | "
            "facebook/sam2-hiera-base-plus"
        ),
    )
    _default_scenetok = os.getenv("SCENETOK_CHECKPOINT", _SCENETOK_DEFAULT_CHECKPOINT)
    p.add_argument(
        "--scenetok",
        action="store_true",
        help=(
            "Install scenetok package, download HF dependencies, and cache checkpoint "
            "(step 14 — streaming scene encoder + segmentation decoder; "
            "requires ~24 GB VRAM to run)"
        ),
    )
    p.add_argument(
        "--scenetok-checkpoint",
        default=_default_scenetok,
        metavar="CHECKPOINT",
        help=(
            "SceneTok checkpoint variant to cache. "
            f"Default: {_SCENETOK_DEFAULT_CHECKPOINT} (RealEstate10K). "
            f"Options: {' | '.join(_SCENETOK_CHECKPOINT_VARIANTS)}. "
            "Set SCENETOK_CHECKPOINT env var to override the default."
        ),
    )
    return p


def main() -> None:
    args = _default_all_if_no_selection(_build_parser().parse_args())
    do_clip = args.clip or args.all
    do_dino = args.dino or args.all
    do_gemma = args.gemma or args.all
    do_flash_attn = args.flash_attn or args.all
    do_whisper = args.whisper or args.all
    do_florence = args.florence or args.all
    do_ocr = args.ocr or args.all
    do_depth = args.depth or args.all
    do_detection = args.detection or args.all
    do_world_model = args.world_model or args.all
    do_unidrive = args.unidrive or args.all
    do_reasoning = args.reasoning or args.all
    do_yolo = args.yolo or args.all
    do_sam = args.sam or args.all
    do_scenetok = args.scenetok or args.all

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Auto device → %s", device)

    from selfsuvis.pipeline.core.config import settings

    errors: list = []
    attempted: list[tuple[str, str]] = []

    whisper_id = args.whisper_model
    if (whisper_id or "").strip().lower() in ("", "auto"):
        from selfsuvis.pipeline.vision.registry import auto_select, detect_resources

        whisper_id = auto_select("asr", detect_resources()) or "openai/whisper-large-v3-turbo"
        log.info("ASR auto-selected model: %s", whisper_id)
    florence_id = args.florence_model
    ocr_id = _resolve_hf_model("ocr", args.ocr_model) if do_ocr else ""
    depth_id = _resolve_hf_model("depth", args.depth_model) if do_depth else ""
    detection_id = _resolve_hf_model("detection", args.detection_model) if do_detection else ""
    world_id = _resolve_hf_model("world_model", args.world_model_id) if do_world_model else ""
    unidrive_backend = ""
    unidrive_id = ""
    if do_unidrive:
        try:
            unidrive_backend = _resolve_unidrive_backend(args.unidrive_backend, args.unidrive_model)
            unidrive_id = _resolve_unidrive_prepare_model(args.unidrive_model, unidrive_backend)
            log.info("UniDrive prepare backend: %s  model=%s", unidrive_backend, unidrive_id)
        except Exception as exc:
            log.warning("UniDrive skipped: %s", exc)
            do_unidrive = False

    # -- Verify mode -----------------------------------------------------------
    if args.verify:
        log.info("Verifying model cache (no downloads) …")
        specs = []
        if do_clip:
            specs.append(
                (
                    f"OpenCLIP {settings.OPENCLIP_MODEL}/{settings.OPENCLIP_PRETRAINED}",
                    lambda: _is_openclip_cached(
                        settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED
                    ),
                )
            )
        if do_dino:
            for dm in args.dino_model:
                dm_copy = dm
                specs.append((f"DINOv2/v3 {dm_copy}", lambda d=dm_copy: _is_dino_hub_cached(d)))
        if do_gemma:
            gm = args.gemma_model
            specs.append((f"Gemma {gm}", lambda m=gm: _is_gemma_cached(m)))
        if do_whisper:
            specs.append((f"Whisper {whisper_id}", lambda m=whisper_id: _is_hf_cached(m)))
        if do_florence:
            specs.append(
                (
                    f"Florence-2 {florence_id}",
                    lambda m=florence_id: _is_hf_cached(m) and _is_florence2_complete(m),
                )
            )
        if do_ocr and ocr_id:
            specs.append((f"OCR {ocr_id}", lambda m=ocr_id: _is_hf_cached(m)))
        if do_depth and depth_id:
            specs.append((f"Depth {depth_id}", lambda m=depth_id: _is_hf_cached(m)))
        if do_detection and detection_id:
            specs.append((f"Detection {detection_id}", lambda m=detection_id: _is_hf_cached(m)))
        if do_world_model and world_id:
            specs.append((f"WorldModel {world_id}", lambda m=world_id: _is_hf_cached(m)))
        if do_unidrive and unidrive_id:
            if unidrive_backend == "ollama":
                specs.append(
                    (
                        f"UniDriveVLA(Ollama) {unidrive_id}",
                        lambda m=unidrive_id: _is_ollama_model_cached(m),
                    )
                )
            else:
                specs.append(
                    (f"UniDriveVLA(vLLM) {unidrive_id}", lambda m=unidrive_id: _is_hf_cached(m))
                )
        if do_reasoning:
            rm = args.reasoning_model
            specs.append((f"Reasoning(Ollama) {rm}", lambda m=rm: _is_ollama_model_cached(m)))
        if do_yolo:
            ym = args.yolo_model
            specs.append((f"YOLO11 {ym}", lambda m=ym: _is_yolo_cached(m)))
        if do_sam:
            sm = args.sam_model
            _SAM2_FALLBACK = "facebook/sam2-hiera-large"

            def _sam_cached(m=sm, fb=_SAM2_FALLBACK):
                return _is_hf_cached(m) or _is_hf_cached(fb)

            label = f"SAM {sm} (or {_SAM2_FALLBACK} fallback)"
            specs.append((label, _sam_cached))
        if do_scenetok:
            sc = args.scenetok_checkpoint
            specs.append((f"SceneTok {sc}", lambda c=sc: _is_scenetok_cached(c)))

        ok, missing = _verify_models(specs)
        for label in ok:
            log.info("  [ok] CACHED    %s", label)
        for label in missing:
            log.warning("  ✗ MISSING   %s", label)

        if missing:
            log.error(
                "%d model(s) not cached — run without --verify to download them.", len(missing)
            )
            sys.exit(1)
        log.info("All %d model(s) verified in cache.", len(ok))
        return

    # -- Download mode ---------------------------------------------------------
    if do_flash_attn:
        try:
            _install_flash_attn()
        except Exception as exc:
            log.error("flash-attn installation failed: %s", exc)
            errors.append(("flash-attn", exc))

    if do_clip:
        label = f"OpenCLIP {settings.OPENCLIP_MODEL}/{settings.OPENCLIP_PRETRAINED}"
        attempted.append((label, label))
        try:
            _download_openclip(settings.OPENCLIP_MODEL, settings.OPENCLIP_PRETRAINED, device)
        except Exception as exc:
            log.error("OpenCLIP download failed: %s", exc)
            errors.append((label, exc))

    if do_dino:
        for dino_model in args.dino_model:
            label = f"DINO {dino_model}"
            attempted.append((label, label))
            try:
                _download_dino(dino_model, device, source=args.source)
            except Exception as exc:
                log.error("DINO [%s] download failed: %s", dino_model, exc)
                import torch as _t

                hub_dir = _t.hub.get_dir()
                log.error(
                    "  Recovery options:\n"
                    "  1. Hugging Face:  selfsuvis-models --dino --source hf\n"
                    "  2. Manual clone:  git clone https://github.com/facebookresearch/dinov2 "
                    "%s/facebookresearch_dinov2_main",
                    hub_dir,
                )
                errors.append((label, exc))

    if do_gemma:
        label = f"Gemma {args.gemma_model}"
        attempted.append((label, label))
        try:
            _download_gemma(args.gemma_model)
        except Exception as exc:
            log.error("Gemma download failed: %s", exc)
            errors.append((label, exc))

    if do_whisper:
        label = f"Whisper {whisper_id}"
        attempted.append((label, label))
        try:
            _with_auth_retry("Whisper", whisper_id, lambda: _download_whisper(whisper_id))
        except Exception as exc:
            log.error("Whisper download failed: %s", exc)
            errors.append((label, exc))

    if do_florence:
        label = f"Florence-2 {florence_id}"
        attempted.append((label, label))
        try:
            _with_auth_retry("Florence-2", florence_id, lambda: _download_florence(florence_id))
        except Exception as exc:
            log.error("Florence-2 download failed: %s", exc)
            errors.append((label, exc))

    if do_ocr:
        label = f"OCR {ocr_id}" if ocr_id else "OCR"
        attempted.append((label, label))
        if not ocr_id:
            log.error("OCR: could not determine a model ID — pass --ocr-model explicitly")
            errors.append((label, ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"OCR ({ocr_id})", ocr_id, lambda: _download_ocr(ocr_id))
            except Exception as exc:
                log.error("OCR model [%s] download failed: %s", ocr_id, exc)
                errors.append((label, exc))

    if do_depth:
        label = f"Depth {depth_id}" if depth_id else "Depth"
        attempted.append((label, label))
        if not depth_id:
            log.error("Depth: could not determine a model ID — pass --depth-model explicitly")
            errors.append((label, ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(f"Depth ({depth_id})", depth_id, lambda: _download_depth(depth_id))
            except Exception as exc:
                log.error("Depth model [%s] download failed: %s", depth_id, exc)
                errors.append((label, exc))

    if do_detection:
        label = f"Detection {detection_id}" if detection_id else "Detection"
        attempted.append((label, label))
        if not detection_id:
            log.error(
                "Detection: could not determine a model ID — pass --detection-model explicitly"
            )
            errors.append((label, ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(
                    f"Detection ({detection_id})",
                    detection_id,
                    lambda: _download_detection(detection_id),
                )
            except Exception as exc:
                log.error("Detection model [%s] download failed: %s", detection_id, exc)
                errors.append((label, exc))

    if do_world_model:
        label = f"WorldModel {world_id}" if world_id else "WorldModel"
        attempted.append((label, label))
        if not world_id:
            log.error(
                "WorldModel: could not determine a model ID — pass --world-model-id explicitly"
            )
            errors.append((label, ValueError("no model ID")))
        else:
            try:
                _with_auth_retry(
                    f"WorldModel ({world_id})", world_id, lambda: _download_world_model(world_id)
                )
            except Exception as exc:
                log.error("World model [%s] download failed: %s", world_id, exc)
                errors.append((label, exc))

    if do_unidrive:
        label = f"UniDriveVLA {unidrive_id}" if unidrive_id else "UniDriveVLA"
        attempted.append((label, label))
        if not unidrive_id:
            log.error(
                "UniDriveVLA: could not determine a model ID — pass --unidrive-model explicitly"
            )
            errors.append((label, ValueError("no model ID")))
        else:
            try:
                if unidrive_backend == "ollama":
                    _download_ollama_model(unidrive_id)
                else:
                    _with_auth_retry(
                        f"UniDriveVLA ({unidrive_id})",
                        unidrive_id,
                        lambda: _download_unidrive(unidrive_id),
                    )
            except Exception as exc:
                log.error(
                    "UniDriveVLA [%s via %s] download failed: %s",
                    unidrive_id,
                    unidrive_backend or "unknown",
                    exc,
                )
                errors.append((label, exc))

    if do_reasoning:
        reasoning_id = args.reasoning_model
        label = f"Reasoning(Ollama) {reasoning_id}"
        attempted.append((label, label))
        try:
            _download_ollama_model(reasoning_id)
        except Exception as exc:
            log.error("Reasoning model [%s] pull failed: %s", reasoning_id, exc)
            errors.append((label, exc))

    if do_yolo:
        yolo_id = args.yolo_model
        label = f"YOLO11 {yolo_id}"
        attempted.append((label, label))
        try:
            _download_yolo(yolo_id)
        except Exception as exc:
            log.error("YOLO11 [%s] download failed: %s", yolo_id, exc)
            errors.append((label, exc))

    if do_sam:
        sam_id = args.sam_model
        label = f"SAM {sam_id}"
        attempted.append((label, label))
        try:
            _with_auth_retry(f"SAM ({sam_id})", sam_id, lambda: _download_sam(sam_id))
        except Exception as exc:
            log.error("SAM [%s] download failed: %s", sam_id, exc)
            errors.append((label, exc))

    if do_scenetok:
        scenetok_ckpt = args.scenetok_checkpoint
        label = f"SceneTok {scenetok_ckpt}"
        attempted.append((label, label))
        try:
            _download_scenetok(scenetok_ckpt)
        except Exception as exc:
            log.error("SceneTok [%s] download failed: %s", scenetok_ckpt, exc)
            errors.append((label, exc))

    _print_completion_summary(attempted, errors)
    if errors:
        sys.exit(1)


def _print_completion_summary(
    attempted: "list[tuple[str, str]]", errors: "list[tuple[str, Exception]]"
) -> None:
    """Print a compact table of every model attempted, its status, and the exit verdict."""
    failed_names = {name for name, _ in errors}
    ok = [name for name, _ in attempted if name not in failed_names]
    failed = [(name, exc) for name, exc in errors]

    col = 48
    line = "-" * (col + 12)
    print(f"\n{line}", flush=True)
    print("  selfsuvis-models — completion summary", flush=True)
    print(line, flush=True)
    for name in ok:
        label = name[:col].ljust(col)
        print(f"  [ok]    {label}", flush=True)
    for name, exc in failed:
        label = name[:col].ljust(col)
        short = str(exc).split("\n")[0][:60]
        print(f"  [FAIL]  {label}  {short}", flush=True)
    print(line, flush=True)
    if failed:
        print(
            f"  {len(ok)} succeeded  /  {len(failed)} FAILED"
            "  — re-run the failed steps individually.",
            flush=True,
        )
        print(line, flush=True)
        log.error("%d model(s) failed — see summary above.", len(failed))
    else:
        print(f"  All {len(ok)} model(s) ready.  Pipeline can now run offline.", flush=True)
        print(line, flush=True)
        log.info("All models ready.")


if __name__ == "__main__":
    main()

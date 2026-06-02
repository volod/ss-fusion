"""Local full-analysis pipeline — public entry point.

All heavy logic lives in the runner_helpers/ submodule:
  runner_helpers/_analytics.py   — run analytics payload & emission
  runner_helpers/_init.py        — model/store init, video discovery
  runner_helpers/_compare.py     — compare/describe steps
  runner_helpers/_agentic.py     — agentic-trace helpers, prompt builders
  runner_helpers/_synthesis.py   — video synthesis & agentic-flow artifact
  runner_helpers/_pipeline.py    — per-video orchestrator (run_video_pipeline)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import resolve_device, settings
from selfsuvis.pipeline.core.logging import get_logger, log_pipeline_finished
from selfsuvis.pipeline.mapping.viewer import _HAS_MPL, view_npz

from ..steps.common import (
    _banner,
    _configure_logging,
    _configure_warnings,
    _step,
)

# -- Re-exports: keep every previously-public symbol importable from runner --
from .runner_helpers import (  # noqa: F401
    _append_agentic_step,
    _build_context_prompt,
    _emit_local_run_analytics,
    find_videos,
    init_models,
    init_store,
    resolve_local_videos,
    run_video_pipeline,
    _run_video_pipeline_safe,
    step_agentic_flow_artifact,
    step_compare_and_describe,
    step_multi_model_compare,
    step_video_synthesis,
    _TOTAL_STEPS,
    _SSL_GATE_MAX_LOSS,
    _VIDEO_EXTS,
)

_log = get_logger(__name__)


def run_local(args: Any) -> None:
    """Run the local full-analysis and training pipeline.

    Called by ``main.py --mode local``.
    Env vars must be set by the caller (via :func:`apply_local_env`) **before**
    this module is imported.
    """
    from ..steps.caption import (
        _compute_sidecar_timeout,
        _list_ollama_models,
        _recommend_gemma_sidecar_models,
        _resolve_ollama_gemma_model,
        _resolve_ollama_reasoning_model,
        _unload_known_sidecars,
    )
    from ..steps.report import print_run_stats, write_final_stats_md

    _configure_logging()
    _configure_warnings()

    output_dir = Path(args.output_dir).resolve()

    # --view-npz shortcut: just visualise existing NPZ files
    if getattr(args, "view_npz", None) is not None:
        if not _HAS_MPL:
            _log.error("matplotlib is required for the 3D viewer.  Install: pip install matplotlib")
            sys.exit(1)
        view_npz(args.view_npz if args.view_npz is not None else "", output_dir)
        return

    t_start = time.time()
    source_label, videos = resolve_local_videos(args)

    _banner("ssv_vdp — Local Full Analysis and Training Pipeline")
    _log.info("Input source     : %s", source_label)
    _log.info("Output directory : %s", output_dir)
    _log.info("Device           : %s", args.device)
    _log.info("Epochs           : %d", args.epochs)
    _log.info("Qdrant           : %s", "disabled" if args.no_qdrant else "auto-detect")
    _log.info("SfM              : %s", "disabled" if args.no_sfm else "auto-detect (pycolmap)")
    multimodal_active = [
        args.asr,
        args.ocr,
        args.depth,
        args.detection,
        args.world_model,
        args.qwen,
        getattr(args, "unidrive", False),
    ]
    if any(multimodal_active):
        _log.info(
            "Multimodal steps : %s",
            " ".join(
                s
                for s, e in [
                    ("ASR", args.asr),
                    ("OCR", args.ocr),
                    ("Depth", args.depth),
                    ("Detection", args.detection),
                    ("WorldModel", args.world_model),
                    ("Qwen", args.qwen),
                    ("UniDriveVLA", getattr(args, "unidrive", False)),
                ]
                if e
            ),
        )

    _log.info("Found %d video(s): %s", len(videos), [v.name for v in videos])

    device = resolve_device(args.device)
    _log.info("Using device: %s", device)

    from selfsuvis.pipeline.vision.registry import detect_resources  # noqa: PLC0415

    if device == "cuda":
        _unload_known_sidecars(
            [
                (
                    getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                    getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
                ),
                (
                    getattr(args, "qwen_api_url", "") or getattr(settings, "QWEN_API_URL", ""),
                    getattr(args, "qwen_model", "") or getattr(settings, "QWEN_MODEL", ""),
                ),
                (
                    getattr(args, "unidrive_api_url", "")
                    or getattr(settings, "UNIDRIVE_API_URL", ""),
                    getattr(args, "unidrive_model", "") or getattr(settings, "UNIDRIVE_MODEL", ""),
                ),
                (
                    getattr(args, "reasoning_api_url", "")
                    or getattr(settings, "REASONING_API_URL", ""),
                    getattr(args, "reasoning_model", "")
                    or getattr(settings, "REASONING_MODEL", ""),
                ),
            ]
        )
    resources = detect_resources()
    _log.info(
        "Detected resources: VRAM total %.1f GiB | VRAM free %.1f GiB | RAM %.1f GiB",
        resources.get("vram_gb", 0.0),
        resources.get("free_vram_gb", 0.0),
        resources.get("ram_gb", 0.0),
    )
    if device == "cuda" and resources.get("vram_gb", 0.0) <= 0.0:
        _log.warning(
            "CUDA was requested but VRAM auto-detection returned 0.0 GiB. "
            "If the NVIDIA driver is temporarily inaccessible, set GPU_TOTAL_GB_HINT "
            "and optionally GPU_FREE_GB_HINT to preserve correct model planning."
        )

    explicit_gemma_model = getattr(args, "gemma_api_model", "") or os.getenv("GEMMA_API_MODEL", "")
    explicit_reasoning_model = getattr(args, "reasoning_model", "") or os.getenv(
        "REASONING_MODEL", ""
    )
    auto_analysis_model, auto_reasoning_model = _recommend_gemma_sidecar_models(resources)
    if not explicit_gemma_model:
        os.environ["GEMMA_API_MODEL"] = auto_analysis_model
        settings.GEMMA_API_MODEL = auto_analysis_model  # type: ignore[misc]
    if not explicit_reasoning_model:
        os.environ["REASONING_MODEL"] = auto_reasoning_model
        settings.REASONING_MODEL = auto_reasoning_model  # type: ignore[misc]
    if not os.getenv("REASONING_API_URL") and not getattr(args, "reasoning_api_url", ""):
        fallback_reasoning_url = (
            getattr(args, "gemma_api_url", "")
            or settings.GEMMA_API_URL
            or getattr(args, "qwen_api_url", "")
            or settings.QWEN_API_URL
        )
        if fallback_reasoning_url:
            os.environ["REASONING_API_URL"] = fallback_reasoning_url
            settings.REASONING_API_URL = fallback_reasoning_url  # type: ignore[misc]

    _log.info(
        "Local pipeline LLM plan: analysis model=%s | reasoning model=%s",
        settings.GEMMA_API_MODEL or auto_analysis_model,
        settings.REASONING_MODEL or auto_reasoning_model,
    )

    # Pre-flight: if a Gemma API URL is configured, verify it responds before
    # loading any models.  Fail loudly rather than silently skipping later.
    _gemma_url = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    if _gemma_url:
        _gemma_model_cfg = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
        # Auto-resolve: swap for a model that's actually available in Ollama
        _gemma_model = _resolve_ollama_gemma_model(_gemma_url, _gemma_model_cfg)
        if _gemma_model != _gemma_model_cfg:
            # Persist resolution so all downstream steps see the correct model
            os.environ["GEMMA_API_MODEL"] = _gemma_model
            settings.GEMMA_API_MODEL = _gemma_model  # type: ignore[misc]
        _PREFLIGHT_TIMEOUT = _compute_sidecar_timeout(_gemma_model, _gemma_url, resources)
        _log.info(
            "Gemma API pre-flight check (url=%s  model=%s) … (timeout=%.0fs)",
            _gemma_url,
            _gemma_model,
            _PREFLIGHT_TIMEOUT,
        )
        try:
            import httpx as _httpx

            _r = _httpx.post(
                f"{_gemma_url.rstrip('/')}/chat/completions",
                json={
                    "model": _gemma_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=_PREFLIGHT_TIMEOUT,
            )
            if _r.status_code == 404:
                _log.error(
                    "Gemma model '%s' not found in Ollama (HTTP 404). "
                    "Pull it with: ollama pull %s\n"
                    "Available models: %s",
                    _gemma_model,
                    _gemma_model,
                    _list_ollama_models(_gemma_url),
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Gemma API pre-flight failed (HTTP %d). "
                    "Ensure Ollama is running: ollama pull %s",
                    _r.status_code,
                    _gemma_model,
                )
                sys.exit(1)
            _log.info(
                "  [ok] Gemma API reachable (HTTP %d  model=%s)", _r.status_code, _gemma_model
            )
        except Exception as _exc:
            _log.error(
                "Gemma API pre-flight error: %s. Check that Ollama is running at %s",
                _exc,
                _gemma_url,
            )
            sys.exit(1)

    _reasoning_url = getattr(args, "reasoning_api_url", "") or getattr(
        settings, "REASONING_API_URL", ""
    )
    if _reasoning_url:
        _reasoning_model_cfg = getattr(args, "reasoning_model", "") or getattr(
            settings, "REASONING_MODEL", ""
        )
        if (
            getattr(args, "reasoning_backend", "") or getattr(settings, "REASONING_BACKEND", "")
        ).lower() == "ollama" or ":11434" in _reasoning_url:
            _reasoning_model = _resolve_ollama_reasoning_model(_reasoning_url, _reasoning_model_cfg)
        else:
            _reasoning_model = _reasoning_model_cfg
        if _reasoning_model != _reasoning_model_cfg:
            os.environ["REASONING_MODEL"] = _reasoning_model
            settings.REASONING_MODEL = _reasoning_model  # type: ignore[misc]
        _log.info(
            "Reasoning API pre-flight check (url=%s  model=%s) …",
            _reasoning_url,
            _reasoning_model,
        )
        _reasoning_preflight_timeout = _compute_sidecar_timeout(
            _reasoning_model, _reasoning_url, resources
        )
        try:
            import httpx as _httpx

            _r = _httpx.post(
                f"{_reasoning_url.rstrip('/')}/chat/completions",
                json={
                    "model": _reasoning_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=_reasoning_preflight_timeout,
            )
            if _r.status_code == 404:
                _log.error(
                    "Reasoning model '%s' not found at %s (HTTP 404). "
                    "Pull or serve it before running the local pipeline.",
                    _reasoning_model,
                    _reasoning_url,
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Reasoning API pre-flight failed (HTTP %d) for model '%s'.",
                    _r.status_code,
                    _reasoning_model,
                )
                sys.exit(1)
            _log.info(
                "  [ok] Reasoning API reachable (HTTP %d  model=%s)",
                _r.status_code,
                _reasoning_model,
            )
        except Exception as _exc:
            _log.error(
                "Reasoning API pre-flight error: %s. Check endpoint %s",
                _exc,
                _reasoning_url,
            )
            sys.exit(1)

    t_init = time.time()
    models = init_models(device)
    store, is_qdrant = init_store(models, use_qdrant=not args.no_qdrant)
    init_elapsed = time.time() - t_init

    per_video_stats: list[dict[str, Any]] = []
    try:
        for i, video_path in enumerate(videos, 1):
            _banner(f"Video {i}/{len(videos)}: {video_path.name}")
            try:
                vstats = _run_video_pipeline_safe(
                    args, video_path, output_dir, models, store, is_qdrant, device
                )
            except KeyboardInterrupt:
                raise
            per_video_stats.append(vstats)

    except KeyboardInterrupt:
        _log.warning("")
        _log.warning("Interrupted by user (Ctrl-C) -- shutting down gracefully ...")
        _log.warning("  %d/%d video(s) completed.", len(per_video_stats), len(videos))
        total_elapsed = time.time() - t_start
        if per_video_stats:
            stats_path = output_dir / "final_stats.md"
            from selfsuvis.pipeline.fusion import persist_threat_memory

            from ..steps.global_threat import step_global_threat
            from ..steps.threat_eval import write_threat_calibration, write_threat_eval_summary

            global_threat_result = step_global_threat(output_dir, per_video_stats)
            persist_threat_memory(output_dir, per_video_stats, global_threat_result)
            write_threat_calibration(output_dir, per_video_stats)
            write_threat_eval_summary(output_dir, per_video_stats)
            write_final_stats_md(stats_path, per_video_stats, total_elapsed)
            print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)
            _log.warning("  Partial results written to: %s", stats_path)
        _log.warning("  Re-run to process remaining videos.")
        log_pipeline_finished(total_elapsed)
        sys.exit(130)

    if not args.no_view:
        view_npz("", output_dir)

    total_elapsed = time.time() - t_start
    stats_path = output_dir / "final_stats.md"
    from selfsuvis.pipeline.fusion import persist_threat_memory

    from ..steps.global_threat import step_global_threat
    from ..steps.model_advisor import write_model_run_advisor
    from ..steps.threat_eval import write_threat_calibration, write_threat_eval_summary

    global_threat_result = step_global_threat(output_dir, per_video_stats)
    persist_threat_memory(output_dir, per_video_stats, global_threat_result)
    write_threat_calibration(output_dir, per_video_stats)
    write_threat_eval_summary(output_dir, per_video_stats)
    _step(33, _TOTAL_STEPS, "Model/run advisor → model_run_advisor.md")
    t_advisor = time.monotonic()
    env_values = {
        key: str(getattr(settings, key, "") or os.getenv(key, ""))
        for key in (
            "APP_ENV",
            "GEMMA_API_URL",
            "GEMMA_API_BACKEND",
            "GEMMA_API_MODEL",
            "QWEN_API_URL",
            "QWEN_BACKEND",
            "QWEN_MODEL",
            "REASONING_API_URL",
            "REASONING_BACKEND",
            "REASONING_MODEL",
            "UNIDRIVE_ENABLED",
            "UNIDRIVE_API_URL",
            "UNIDRIVE_BACKEND",
            "UNIDRIVE_MODEL",
        )
    }
    write_model_run_advisor(
        output_dir,
        per_video_stats,
        resources=resources,
        env_values=env_values,
    )
    if per_video_stats:
        per_video_stats[-1].setdefault("timings", {})["AB_model_advisor"] = (
            time.monotonic() - t_advisor
        )
    write_final_stats_md(stats_path, per_video_stats, total_elapsed)
    print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)

    _log.info("  Final statistics: %s", stats_path)
    _log.info("  Model run advisor: %s", output_dir / "model_run_advisor.md")
    _log.info("  Global threat summary: %s", output_dir / "global_threat_summary.json")
    _log.info("  Threat memory: %s", output_dir / "threat_memory")
    _log.info("  Threat calibration: %s", output_dir / "threat_calibration.json")
    _log.info("  Threat evaluation: %s", output_dir / "threat_eval_summary.json")
    _log.info("")
    _log.info("  Next steps:")
    _log.info(
        "    • Edge inference:  EdgeClassifier('edge_models/dino_local.onnx', 'edge_models/gallery.npz')"
    )
    _log.info("    • Full stack:      make up")
    _log.info("    • Fine-tune rerun: DINO_CHECKPOINT=<path> python main.py --mode local")
    _log.info("")
    log_pipeline_finished(total_elapsed)

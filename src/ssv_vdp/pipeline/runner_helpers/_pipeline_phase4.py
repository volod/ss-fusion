"""Phase 4 — Finalization: Steps 27-34 + final stats.

Multi-model comparison, local threat/policy, video synthesis, agentic flow audit,
drone detection, drone audio, drau range eval, and final stats collection.
"""

import json
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import _step, _Timer
from ._agentic import _append_agentic_step

_log = get_logger(__name__)

_TOTAL_STEPS = 35


def run_phase4(
    *,
    args: Any,
    video_path: Path,
    video_dir: Path,
    video_name: str,
    output_dir: Path,
    device: str,
    frame_list: list[tuple[str, float]],
    clip_dino_on_gpu: bool,
    # results from phase 2
    j: dict[str, Any],
    qwen_result: dict[str, Any],
    unidrive_result: dict[str, Any],
    threat_primitives_result: dict[str, Any],
    physical_state_result: dict[str, Any],
    base_results: list[dict[str, Any]],
    # results from phase 3
    ssl_gate_passed: bool,
    ft_results: list[dict[str, Any]],
    # shared mutable state
    stats: dict[str, Any],
    T: dict[str, Any],
    video_context: dict[str, Any],
    agentic_trace: list[dict[str, Any]],
) -> None:
    """Run Steps 27-34 and collect final stats. Mutates stats in place."""
    from ...steps.caption import (
        _offload_models_to_cpu,
        _unload_known_sidecars,
        get_runtime_telemetry,
    )
    from ._analytics import _emit_local_run_analytics
    from ._compare import step_multi_model_compare
    from ._synthesis import step_agentic_flow_artifact, step_video_synthesis

    # Step 27: Multi-model comparison — Gemma vs Qwen vs UniDriveVLA
    if not qwen_result.get("skipped") and not unidrive_result.get("skipped"):
        _step(27, _TOTAL_STEPS, "Multi-model comparison → multi_model_comparison.md")
        with _Timer(T, "T_multimodel"):
            mm = step_multi_model_compare(video_name, video_dir, j, qwen_result, unidrive_result)
        video_context["multi_model_comparison"] = mm
        _append_agentic_step(
            agentic_trace,
            step_id="23",
            title="Multi-model comparison",
            description="Compare Gemma, Qwen, and UniDriveVLA outputs and expose UniDrive mixture-of-experts agreement signals.",
            status="ok",
            context_inputs=["Gemma summary", "Qwen structured scene facts", "UniDrive expert outputs"],
            context_outputs=[
                f"{mm.get('matched_frames', 0)} matched comparison frames",
                f"Qwen/UniDrive agreement {mm.get('mean_qwen_unidrive_agreement', 0.0):.3f}",
                f"{mm.get('high_risk_frames', 0)} high-risk UniDrive frames",
            ],
            risks=[
                "timestamp-nearest matching can compare slightly different moments",
                "token-overlap agreement is a coarse proxy for semantic agreement",
                "expert consensus may under-report minority expert concerns",
            ],
            artifacts=["multi_model_comparison.md"],
        )
    else:
        T["T_multimodel"] = 0.0
        _step(27, _TOTAL_STEPS, "Multi-model comparison (skipped — requires Qwen and UniDrive)")
        _append_agentic_step(
            agentic_trace,
            step_id="23",
            title="Multi-model comparison",
            description="Compare Gemma, Qwen, and UniDriveVLA outputs and expose UniDrive mixture-of-experts agreement signals.",
            status="skipped",
            context_inputs=["Qwen and UniDrive outputs"],
            context_outputs=["no cross-model comparison artifact"],
            risks=["cross-model disagreement remains implicit"],
            artifacts=[],
        )

    # Step 28: Local threat aggregation
    _step(28, _TOTAL_STEPS, "Local threat inference → local_threat_assessment.json")
    from ...steps.threat.local_threat import step_local_threat

    with _Timer(T, "PS_local_threat"):
        local_threat_result = step_local_threat(
            threat_primitives_result=threat_primitives_result,
            video_dir=video_dir,
            video_name=video_name,
            unidrive_rows=video_context.get("unidrive_analysis", []),
            physical_state=physical_state_result,
        )
    video_context["local_threat"] = local_threat_result
    _append_agentic_step(
        agentic_trace,
        step_id="27",
        title="Local threat inference",
        description="Aggregate persisted threat primitives across the full video window into a policy-free threat estimate.",
        status="ok" if not local_threat_result.get("skipped") else "skipped",
        context_inputs=["threat primitives", "temporal persistence threshold"],
        context_outputs=[
            f"local threat score {float(local_threat_result.get('local_threat_score', 0.0)):.3f}",
            f"automation confidence {float(local_threat_result.get('automation_confidence', 1.0)):.3f}",
        ]
        if not local_threat_result.get("skipped")
        else ["no active local threat output"],
        risks=[
            "persistence threshold can suppress short but real hazards",
            "clip-level aggregation can hide when a threat is localized to a brief segment",
            "threat estimate can be over-trusted if policy and sensor-health checks are skipped downstream",
        ],
        artifacts=["local_threat_assessment.json"] if not local_threat_result.get("skipped") else [],
    )

    # Step 29: Action policy
    _step(29, _TOTAL_STEPS, "Action policy → policy_decision.json")
    from ...steps.threat.policy import step_policy

    with _Timer(T, "PS_policy"):
        policy_result = step_policy(
            local_threat_result,
            video_dir,
            video_name,
            sensor_health={
                "degraded": float(local_threat_result.get("trust_penalty", 0.0) or 0.0) >= 0.30,
                "health_warnings": [
                    conflict.get("pattern", "unknown")
                    for conflict in (local_threat_result.get("source_pair_conflicts") or [])[:3]
                ],
                "missing_sensors": [],
            },
        )
    video_context["policy_decision"] = policy_result
    _append_agentic_step(
        agentic_trace,
        step_id="28",
        title="Action policy",
        description="Map the threat estimate, confidence, and sensor-health context into a fixed action vocabulary without changing the threat score semantics.",
        status="ok" if not policy_result.get("skipped") else "skipped",
        context_inputs=["local threat estimate", "automation confidence", "sensor-health indicators"],
        context_outputs=[
            f"recommended action {policy_result.get('recommended_action', 'continue')}",
            f"policy reason {policy_result.get('policy_reason', 'n/a')}",
        ]
        if not policy_result.get("skipped")
        else ["no policy decision"],
        risks=[
            "policy defaults may not match mission-specific objectives",
            "sensor-health heuristics can over-trigger inspect-sensor in noisy environments",
        ],
        artifacts=["policy_decision.json"] if not policy_result.get("skipped") else [],
    )

    # Step 30: Video synthesis
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu({})
    _step(30, _TOTAL_STEPS, "Video synthesis (ontology + narrative) → video_synthesis.md")
    _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    with _Timer(T, "Z_synthesis"):
        step_video_synthesis(video_name, video_dir, video_context, api_url=_qwen_url, model=_qwen_model)
    _append_agentic_step(
        agentic_trace,
        step_id="29",
        title="Video synthesis",
        description="Use accumulated multimodal context to generate a structured ontology and narrative summary of the whole video.",
        status="ok" if _qwen_url else "skipped",
        context_inputs=[
            "Gemma summary", "captions, ASR, OCR, detections, Qwen frame reasoning",
            "local threat assessment", "retrieval description and map summary",
        ],
        context_outputs=["video ontology", "global narrative summary"] if _qwen_url else ["no synthesis output"],
        risks=[
            "final narrative can collapse uncertain evidence into a single confident story",
            "contradictions across modalities may be hidden in the synthesized summary",
            "wrong high-level framing can mask the original source of context errors",
        ],
        artifacts=["video_synthesis.md", "video_ontology.json"] if _qwen_url else [],
    )

    # Step 31: Agentic flow audit
    _step(31, _TOTAL_STEPS, "Agentic flow audit → agentic_flow.md")
    _agentic_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
        or getattr(args, "gemma_api_url", "")
        or settings.GEMMA_API_URL
        or _qwen_url
    )
    _agentic_model = (
        getattr(args, "reasoning_model", "")
        or getattr(settings, "REASONING_MODEL", "")
        or getattr(args, "gemma_api_model", "")
        or settings.GEMMA_API_MODEL
        or _qwen_model
    )
    _append_agentic_step(
        agentic_trace,
        step_id="30",
        title="Agentic flow audit",
        description="Audit the full context chain, explain step-to-step reasoning state, and register per-step risks of misidentification and wrong context.",
        status="ok",
        context_inputs=["complete pipeline trace", "all accumulated artifacts and summaries"],
        context_outputs=["agentic_flow.md audit report"],
        risks=[
            "reasoning model can restate upstream errors coherently",
            "audit quality depends on provenance captured from earlier steps",
            "fallback deterministic summary is less nuanced than the LLM audit",
        ],
        artifacts=["agentic_flow.md"],
    )
    with _Timer(T, "AA_agentic"):
        step_agentic_flow_artifact(
            video_name, video_dir, video_context, api_url=_agentic_url, model=_agentic_model
        )
    if device == "cuda":
        _unload_known_sidecars(
            [
                (_agentic_url, _agentic_model),
                (_qwen_url, _qwen_model),
                (getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL, getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL),
                (getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL, getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL),
            ]
        )

    # Step 32: Drone detection edge model training
    _drone_enabled = getattr(args, "drone_detection", None)
    if _drone_enabled is None:
        _drone_enabled = True  # on by default when not explicitly disabled
    if _drone_enabled:
        from ...steps.edge.drone_detection import step_drone_detection_training
        _step(32, _TOTAL_STEPS, "Drone detection training → drone_detection/")
        _append_agentic_step(
            agentic_trace,
            step_id="31",
            title="Drone detection edge training",
            description=(
                "Train YOLOv8n on seraphim-drone-detection-dataset + mission hard negatives; "
                "export ONNX fp32 (Cortex-A76) and int8 (RV1106G3 NPU)."
            ),
            status="ok",
            context_inputs=["extracted mission frames", "seraphim HF dataset batch_001"],
            context_outputs=["drone_yolo8n_a76.onnx", "drone_yolo8n_rv1106_int8.onnx", "drone_detection_report.md"],
            risks=[
                "small dataset subset limits generalisation",
                "false positives increase without sufficient hard negatives",
                "rknn-toolkit2 required for full NPU deployment on RV1106G3",
            ],
            artifacts=["drone_detection/drone_detection_report.md"],
        )
        with _Timer(T, "AC_drone_detection"):
            drone_result = step_drone_detection_training(
                frame_list, video_name, video_dir, output_dir, device, args
            )
        stats["drone_detection"] = drone_result
        _log.info(
            "  Drone detection: map50=%.4f | fp32=%s | int8=%s | rknn=%s",
            drone_result.get("map50", float("nan")),
            "[ok]" if drone_result.get("model_fp32") else "✗",
            "[ok]" if drone_result.get("model_int8") else "✗",
            "[ok]" if drone_result.get("model_rknn") else "skipped",
        )
    else:
        _step(32, _TOTAL_STEPS, "Drone detection training (skipped — pass --drone-detection to enable)")

    # Step 33: Drone audio detection model training
    _audio_enabled = getattr(args, "drone_audio", None)
    if _audio_enabled is None:
        _audio_enabled = True  # on by default when not explicitly disabled
    if _audio_enabled:
        from ...steps.edge.drone_audio import step_drone_audio_training
        _step(33, _TOTAL_STEPS, "Drone audio training → drone_audio/")
        _append_agentic_step(
            agentic_trace,
            step_id="33",
            title="Drone audio detection model training",
            description=(
                "Train DroneAudioCNN (small 2-D CNN on MFCC features) on "
                "geronimobasso/drone-audio-detection-samples cached in "
                ".data/drone-audio-data/; export ONNX for edge inference."
            ),
            status="ok",
            context_inputs=[".data/drone-audio-data/train/drone/*.wav", ".data/drone-audio-data/train/no_drone/*.wav"],
            context_outputs=["drone_audio_cnn.pt", "drone_audio_cnn.onnx", "drone_audio_report.md"],
            risks=[
                "datasets library required for first-time HF download",
                "small dataset may limit generalisation to novel drone types",
                "run ssv-prepare-audio first for best results",
            ],
            artifacts=["drone_audio/drone_audio_report.md"],
        )
        with _Timer(T, "AC_drone_audio"):
            audio_result = step_drone_audio_training(video_dir, output_dir, device, args)
        stats["drone_audio"] = audio_result
        _log.info(
            "  Drone audio: val_acc=%.3f  val_f1=%.3f  onnx=%s",
            audio_result.get("val_acc", float("nan")),
            audio_result.get("val_f1", float("nan")),
            "[ok]" if audio_result.get("model_onnx") else "✗",
        )
    else:
        _step(33, _TOTAL_STEPS, "Drone audio training (skipped — pass --drone-audio to enable)")

    # Step 34: drau range-detection evaluation
    _drau_enabled = getattr(args, "drau_eval", None)
    if _drau_enabled is None:
        # Auto: run if the ONNX from step 33 exists.
        _drau_onnx = video_dir / "drone_audio" / "drone_audio_cnn.onnx"
        _drau_enabled = _drau_onnx.exists()
    if _drau_enabled:
        from ...steps.edge.drau_eval import step_drau_range_eval
        _step(34, _TOTAL_STEPS, "drau range eval → drone_audio/drau_range_report.md")
        _append_agentic_step(
            agentic_trace,
            step_id="34",
            title="drau range-detection evaluation",
            description=(
                "Evaluate DroneAudioCNN ONNX model at simulated distances using "
                "the drau physics model (github.com/volod/drau): inverse-square "
                "amplitude scaling + ISO 9613-1 atmospheric absorption. "
                "Exports drau_edge_test.py for numpy+scipy+onnxruntime-only inference."
            ),
            status="ok",
            context_inputs=[
                "drone_audio/drone_audio_cnn.onnx (from step 33)",
                "synthetic quadcopter audio at 9 distances (1-200 m)",
            ],
            context_outputs=[
                "drau_range_report.md (detection probability vs distance)",
                "drau_edge_test.py (standalone edge script, no PyTorch)",
            ],
            risks=[
                "synthetic signal is a simplification; real drone audio varies by model and rotor configuration",
                "onnxruntime must be installed for inference; skips gracefully if absent",
                "detection range estimate is a model characteristic, not a deployment guarantee",
            ],
            artifacts=["drone_audio/drau_range_report.md", "drone_audio/drau_edge_test.py"],
        )
        with _Timer(T, "AC_drau_eval"):
            drau_result = step_drau_range_eval(video_dir, output_dir, args)
        stats["drau_eval"] = drau_result
        if drau_result.get("skipped"):
            _log.info("  drau eval: skipped (%s)", drau_result.get("reason", ""))
        else:
            _log.info(
                "  drau eval: detection_range=%s m  elapsed=%.1fs",
                drau_result.get("detection_range_m", "n/a"),
                drau_result.get("elapsed_sec", 0.0),
            )
    else:
        _step(34, _TOTAL_STEPS, "drau eval (skipped — ONNX not found; pass --drau-eval to force)")

    # Final stats
    stats["pipeline_sec"] = sum(T.values())
    runtime_metrics = get_runtime_telemetry()
    stats["vram_wait_time_sec"] = runtime_metrics.get("vram_wait_time_sec", 0.0)
    stats["restore_failures"] = int(runtime_metrics.get("restore_failures", 0.0))
    (video_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2), encoding="utf-8"
    )
    stats["analysis_summary"] = _emit_local_run_analytics(video_dir) or {}
    stats["video_dir"] = str(video_dir)

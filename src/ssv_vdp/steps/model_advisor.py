"""Post-run model and runtime advisor for local pipeline runs."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import write_json_artifact, write_markdown_artifact


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _collect_summaries(per_video: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for video in per_video:
        summary = dict(video.get("analysis_summary", {}) or {})
        if not summary:
            continue
        summary.setdefault("video_name", video.get("name", "unknown"))
        summaries.append(summary)
    return summaries


def _recommend_qwen_model(vram_gb: float, free_vram_gb: float, ram_gb: float) -> str:
    if vram_gb >= 24 or free_vram_gb >= 18:
        return "qwen2.5vl:7b"
    if vram_gb >= 12 or free_vram_gb >= 10 or ram_gb >= 48:
        return "qwen2.5vl:7b"
    return "qwen2.5vl:3b"


def _recommend_reasoning_model(vram_gb: float, free_vram_gb: float, ram_gb: float) -> str:
    if vram_gb >= 24 or free_vram_gb >= 18:
        return "deepseek-r1:14b"
    if vram_gb >= 12 or free_vram_gb >= 10 or ram_gb >= 48:
        return "qwen3:14b"
    if ram_gb >= 32:
        return "qwen3:8b"
    return "gemma4:e4b"


def _collect_drone_detection_summaries(
    per_video: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for v in per_video:
        dd = dict(v.get("drone_detection") or {})
        if dd and not dd.get("skipped"):
            dd.setdefault("video_name", v.get("name", "unknown"))
            results.append(dd)
    return results


def build_model_run_advisor(
    per_video: Sequence[Mapping[str, Any]],
    *,
    resources: Mapping[str, Any],
    env_values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a post-run optimization plan from analytics summaries."""

    env = dict(env_values or {})
    summaries = _collect_summaries(per_video)
    vram_gb = _as_float(resources.get("vram_gb"))
    free_vram_gb = _as_float(resources.get("free_vram_gb"), vram_gb)
    ram_gb = _as_float(resources.get("ram_gb"))

    qwen_parse_errors = sum(
        _as_int((s.get("run_health", {}) or {}).get("qwen_parse_error_count")) for s in summaries
    )
    qwen_coverages = [
        _as_float((s.get("run_health", {}) or {}).get("qwen_caption_coverage"))
        for s in summaries
        if "run_health" in s
    ]
    min_qwen_coverage = min(qwen_coverages) if qwen_coverages else 0.0
    degraded_maps = [
        s
        for s in summaries
        if bool((s.get("map_stats", {}) or {}).get("degraded"))
        or _as_float((s.get("diagnostics", {}) or {}).get("map_pose_coverage")) <= 0.0
    ]
    artifact_mb_per_min = max(
        [_as_float((s.get("diagnostics", {}) or {}).get("artifact_mb_per_min")) for s in summaries]
        or [0.0]
    )

    recommended_qwen = _recommend_qwen_model(vram_gb, free_vram_gb, ram_gb)
    recommended_reasoning = _recommend_reasoning_model(vram_gb, free_vram_gb, ram_gb)

    findings: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    if qwen_parse_errors or min_qwen_coverage < 0.5:
        findings.append(
            {
                "severity": "high",
                "code": "qwen_structured_captioning_degraded",
                "detail": (
                    f"Qwen structured captioning had {qwen_parse_errors} parse error(s) "
                    f"and minimum coverage {min_qwen_coverage:.1%}."
                ),
            }
        )
        recommendations.append(
            {
                "area": "vlm_captioning",
                "action": "Use a stronger Qwen-VL model and the LangGraph retry path.",
                "env": {
                    "QWEN_API_URL": "http://localhost:11434/v1",
                    "QWEN_BACKEND": "ollama",
                    "QWEN_MODEL": recommended_qwen,
                    "UNIDRIVE_ENABLED": "true",
                    "UNIDRIVE_API_URL": "http://localhost:11434/v1",
                    "UNIDRIVE_BACKEND": "ollama",
                    "UNIDRIVE_MODEL": recommended_qwen,
                },
                "why": (
                    "The 3B Qwen-VL tier is fast but brittle for strict JSON output. "
                    "The 7B tier is slower, but on a 12 GB GPU it is usually workable "
                    "when Ollama keeps only one model loaded at a time."
                ),
            }
        )

    if degraded_maps:
        findings.append(
            {
                "severity": "medium",
                "code": "sfm_pose_recovery_degraded",
                "detail": (
                    f"{len(degraded_maps)} video(s) produced degraded maps or zero SfM pose coverage."
                ),
            }
        )
        recommendations.append(
            {
                "area": "mapping",
                "action": "Treat the current map as semantic pseudo-3D and improve capture before tuning models.",
                "env": {},
                "why": (
                    "Zero SfM poses usually comes from short clips, low parallax, nadir-only motion, "
                    "motion blur, or low texture. A larger model does not fix missing geometric constraints."
                ),
                "capture_guidance": [
                    "Use 30-90 seconds instead of a 10 second clip.",
                    "Fly or move with lateral/parallax motion, not only straight-line or static hover.",
                    "Keep overlap high and avoid abrupt yaw-only turns.",
                    "Disable `--no-sfm` for quality runs; use `--no-gsplat` while debugging capture quality.",
                ],
            }
        )

    if artifact_mb_per_min > 4096:
        findings.append(
            {
                "severity": "low",
                "code": "artifact_volume_high",
                "detail": f"Artifact density reached {artifact_mb_per_min:.0f} MB/min.",
            }
        )
        recommendations.append(
            {
                "area": "storage",
                "action": "Use fast-iteration flags while tuning models, then run the full recipe once.",
                "env": {},
                "why": "Short tuning cycles should avoid repeated large checkpoints and map outputs.",
            }
        )

    env_updates = {
        "GEMMA_API_URL": env.get("GEMMA_API_URL", "http://localhost:11434/v1"),
        "GEMMA_API_BACKEND": env.get("GEMMA_API_BACKEND", "ollama"),
        "GEMMA_API_MODEL": env.get("GEMMA_API_MODEL", "gemma4:e4b"),
        "QWEN_API_URL": "http://localhost:11434/v1",
        "QWEN_BACKEND": "ollama",
        "QWEN_MODEL": recommended_qwen,
        "REASONING_API_URL": "http://localhost:11434/v1",
        "REASONING_BACKEND": "ollama",
        "REASONING_MODEL": recommended_reasoning,
        "UNIDRIVE_ENABLED": "true",
        "UNIDRIVE_API_URL": "http://localhost:11434/v1",
        "UNIDRIVE_BACKEND": "ollama",
        "UNIDRIVE_MODEL": recommended_qwen,
    }

    # Drone detection edge deployment insights
    drone_summaries = _collect_drone_detection_summaries(per_video)
    edge_profile: dict[str, Any] = {}
    if drone_summaries:
        dd = drone_summaries[-1]
        map50 = _as_float(dd.get("map50"), float("nan"))
        has_fp32 = bool(dd.get("model_fp32"))
        has_int8 = bool(dd.get("model_int8"))
        has_rknn = bool(dd.get("model_rknn"))
        edge_profile = {
            "map50": map50,
            "a76_onnx": has_fp32,
            "rv1106_int8": has_int8,
            "rv1106_rknn": has_rknn,
        }
        import math as _math

        if not _math.isnan(map50) and map50 < 0.50:
            findings.append(
                {
                    "severity": "medium",
                    "code": "drone_detection_low_map50",
                    "detail": (
                        f"YOLOv8n drone detector reached mAP@50={map50:.3f} on the demo subset. "
                        "Download the full dataset (batch_001–004) and retrain with --drone-detection."
                    ),
                }
            )
            recommendations.append(
                {
                    "area": "edge_drone_detection",
                    "action": "Expand training data and retrain drone detector.",
                    "env": {},
                    "why": (
                        f"mAP@50={map50:.3f} is below the 0.50 threshold suitable for deployment. "
                        "The demo run used only batch_001 (~400 images). "
                        "Use the full seraphim dataset for production-quality weights."
                    ),
                }
            )
        if not has_rknn:
            recommendations.append(
                {
                    "area": "edge_drone_detection",
                    "action": "Install rknn-toolkit2 to generate the RV1106G3 NPU model.",
                    "env": {},
                    "why": (
                        "The RKNN model was not generated because rknn-toolkit2 is not installed. "
                        "Without it, the int8 ONNX fallback runs on the RV1106G3 Cortex-A7 core "
                        "at ~3-5× lower throughput than the NPU path."
                    ),
                }
            )

    # Sequential VLLM quality-graph profile for model orchestration
    seq_vllm_profile = {
        "graph_mode": "sequential",
        "recommended_order": [
            {
                "step": "gemma_analysis",
                "model": env_updates["GEMMA_API_MODEL"],
                "when": "scene_understanding",
            },
            {
                "step": "qwen_captioning",
                "model": env_updates["QWEN_MODEL"],
                "when": "detailed_captioning",
            },
            {"step": "unidrive", "model": env_updates["UNIDRIVE_MODEL"], "when": "vla_planning"},
            {
                "step": "reasoning_audit",
                "model": env_updates["REASONING_MODEL"],
                "when": "final_audit",
            },
        ],
        "vllm_env": {
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_KEEP_ALIVE": "0",
        },
        "rationale": (
            "Keep OLLAMA_MAX_LOADED_MODELS=1 so each VLLM step gets the full VRAM budget. "
            f"On {vram_gb:.0f} GiB VRAM, running models sequentially avoids OOM "
            "and eliminates inter-model interference in KV-cache."
        ),
    }

    if not findings:
        recommendations.append(
            {
                "area": "general",
                "action": "Keep the current model plan; no high-confidence model-size change was indicated.",
                "env": {},
                "why": "Run-health warnings and coverage metrics are within expected bounds.",
            }
        )

    return {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "resources": {
            "vram_gb": vram_gb,
            "free_vram_gb": free_vram_gb,
            "ram_gb": ram_gb,
        },
        "current_env": {
            key: env.get(key, "")
            for key in (
                "GEMMA_API_MODEL",
                "QWEN_MODEL",
                "REASONING_MODEL",
                "UNIDRIVE_ENABLED",
                "UNIDRIVE_MODEL",
            )
        },
        "findings": findings,
        "recommendations": recommendations,
        "recommended_env_updates": env_updates,
        "edge_deployment": edge_profile,
        "sequential_vllm_graph_profile": seq_vllm_profile,
        "recommended_ollama_pulls": list(
            dict.fromkeys(
                [
                    env_updates["GEMMA_API_MODEL"],
                    env_updates["QWEN_MODEL"],
                    env_updates["REASONING_MODEL"],
                ]
            )
        ),
        "recommended_rerun": {
            "env": {
                "SELFSUVIS_USE_GRAPH": "1",
                "APP_ENV": env.get("APP_ENV", "dev") or "dev",
            },
            "command": (
                ".venv/bin/selfsuvis --mode local --videos-dir .data/videos "
                "--qwen --unidrive --world-model --rfdetr-model base --drone-detection"
            ),
        },
    }


def write_model_run_advisor(
    output_dir: Path,
    per_video: Sequence[Mapping[str, Any]],
    *,
    resources: Mapping[str, Any],
    env_values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write model_run_advisor.json and model_run_advisor.md."""

    advisor = build_model_run_advisor(per_video, resources=resources, env_values=env_values)
    json_path = output_dir / "model_run_advisor.json"
    md_path = output_dir / "model_run_advisor.md"
    write_json_artifact(json_path, advisor)

    lines = [
        "# Model Run Advisor",
        "",
        f"Generated: {advisor['generated_at']}",
        "",
        "## Hardware",
        "",
        (
            f"- VRAM: {advisor['resources']['vram_gb']:.1f} GiB total, "
            f"{advisor['resources']['free_vram_gb']:.1f} GiB free at planning time"
        ),
        f"- RAM: {advisor['resources']['ram_gb']:.1f} GiB",
        "",
        "## Findings",
        "",
    ]
    if advisor["findings"]:
        for finding in advisor["findings"]:
            lines.append(f"- **{finding['severity']}** `{finding['code']}`: {finding['detail']}")
    else:
        lines.append("- No high-confidence run-health issues found.")

    lines += ["", "## Recommended `.env` Updates", "", "```env"]
    for key, value in advisor["recommended_env_updates"].items():
        lines.append(f"{key}={value}")
    lines += ["```", "", "## Pull / Serve Models", "", "```bash"]
    lines.append(
        "OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=0 ollama serve"
    )
    for model in advisor["recommended_ollama_pulls"]:
        lines.append(f"ollama pull {model}")
    lines += ["```", "", "## Recommended Rerun", "", "```bash"]
    for key, value in advisor["recommended_rerun"]["env"].items():
        lines.append(f"export {key}={value}")
    lines.append(advisor["recommended_rerun"]["command"])
    lines += ["```", "", "## Rationale", ""]
    for recommendation in advisor["recommendations"]:
        lines.append(
            f"- **{recommendation['area']}**: {recommendation['action']} {recommendation['why']}"
        )
        for item in recommendation.get("capture_guidance", []):
            lines.append(f"  - {item}")

    # Edge deployment section
    ep = advisor.get("edge_deployment", {})
    if ep:
        import math as _math

        lines += ["", "## Edge Deployment — Drone Detection", ""]
        map50 = ep.get("map50", float("nan"))
        lines.append(
            f"- mAP@50: **{map50:.3f}**"
            if not _math.isnan(map50)
            else "- mAP@50: n/a (training skipped)"
        )
        lines.append(
            f"- Cortex-A76 ONNX fp32: {'[ok] generated' if ep.get('a76_onnx') else '✗ missing'}"
        )
        lines.append(
            f"- RV1106G3 ONNX int8:   {'[ok] generated' if ep.get('rv1106_int8') else '✗ missing'}"
        )
        lines.append(
            f"- RV1106G3 RKNN:        {'[ok] generated' if ep.get('rv1106_rknn') else '[warn] install rknn-toolkit2'}"
        )

    # Sequential VLLM graph profile
    sq = advisor.get("sequential_vllm_graph_profile", {})
    if sq:
        lines += ["", "## Sequential VLLM Graph Profile", "", sq.get("rationale", ""), ""]
        lines += ["| Step | Model | When |", "|------|-------|------|"]
        for entry in sq.get("recommended_order", []):
            lines.append(f"| {entry['step']} | `{entry['model']}` | {entry['when']} |")
        lines += ["", "```bash"]
        for k, v in sq.get("vllm_env", {}).items():
            lines.append(f"export {k}={v}")
        lines += ["ollama serve", "```"]

    write_markdown_artifact(md_path, lines)
    return advisor

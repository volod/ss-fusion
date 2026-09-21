"""Legacy local-pipeline adapter: map a planned spec onto `ssv --mode local`."""

from pathlib import Path
from typing import Any

from ss_kernel.catalog import TIMING_KEYS
from ss_kernel.contracts import PipelineSpec, StepManifest
from ss_kernel.plan import Plan

_BOOL_CLI = (
    "no_qdrant",
    "no_asr",
    "no_ocr",
    "no_depth",
    "no_detection",
    "no_world_model",
    "no_unidrive",
    "no_qwen",
    "no_scenetok",
    "no_yolo",
    "no_sam",
    "no_rfdetr",
    "no_sfm",
    "no_gsplat",
    "no_distill",
    "no_onnx",
    "no_caption",
    "no_cosmos3",
    "no_drone_detection",
    "no_drone_audio",
    "no_drau_eval",
)


def legacy_step(context: dict[str, Any]) -> dict[str, Any]:
    del context
    raise RuntimeError("legacy steps run as a batch via run_legacy_batch")


def build_legacy_argv(
    spec: PipelineSpec,
    manifests: dict[str, StepManifest],
    plan: Plan,
    video: Path,
    output_dir: Path,
) -> list[str]:
    """ssv argv from spec.parameters.legacy_cli plus skip_flags for shed steps."""
    flags = dict(spec.parameters.get("legacy_cli") or {})
    extra: set[str] = set()
    shed_ids = {item.step_id for item in plan.shed}
    for step in spec.steps:
        if step.step_id in shed_ids:
            extra.update(manifests[step.manifest_id].skip_flags)
    argv = [
        "--mode",
        "local",
        "--video",
        str(video),
        "--output-dir",
        str(output_dir),
        "--device",
        str(flags.get("device") or "cuda"),
        "--epochs",
        str(int(flags.get("epochs") or 1)),
    ]
    present: set[str] = set()
    for key in _BOOL_CLI:
        dashed = key.replace("_", "-")
        if flags.get(key) or dashed in extra:
            argv.append(f"--{dashed}")
            present.add(dashed)
    for dashed in sorted(extra):
        if dashed not in present:
            argv.append(f"--{dashed}")
    return argv


def collect_local_result(output_dir: Path) -> dict[str, Any]:
    """Coverage, frame count, and relative artifact paths from a local run dir."""
    summaries = list(output_dir.rglob("analysis_summary.json"))
    coverage = ""
    n_frames = 0
    video_artifact_count = 0
    video_dir: Path | None = None
    if summaries:
        import json

        summary_path = summaries[0]
        video_dir = summary_path.parent
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        health = payload.get("run_health") or {}
        coverage = (
            f"{100.0 * float(health.get('florence_caption_coverage', 0.0)):.0f}/"
            f"{100.0 * float(health.get('qwen_caption_coverage', 0.0)):.0f}/"
            f"{100.0 * float(health.get('asr_coverage', 0.0)):.0f}/"
            f"{100.0 * float(health.get('ocr_coverage', 0.0)):.0f}%"
        )
        n_frames = int(payload.get("n_frames") or 0)
        video_artifact_count = int(payload.get("artifact_count") or 0)
    artifacts: list[str] = []
    scan_root = video_dir if video_dir is not None else output_dir
    for path in sorted(scan_root.rglob("*")):
        if path.is_file():
            artifacts.append(path.relative_to(scan_root).as_posix())
    timings: dict[str, int] = {}
    return {
        "coverage": coverage,
        "n_frames": n_frames,
        "video_artifact_count": video_artifact_count,
        "video_artifact_list": artifacts,
        "timings": timings,
        "timing_keys": TIMING_KEYS,
    }


def run_legacy_batch(
    spec: PipelineSpec,
    manifests: dict[str, StepManifest],
    plan: Plan,
    *,
    run_dir: Path,
    video: Path | None,
) -> dict[str, Any]:
    """Run the monolith local pipeline for remaining legacy-wrapped steps."""
    if video is None:
        raise ValueError("legacy local adapter requires --video")
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_legacy_argv(spec, manifests, plan, video, output_dir)
    from ssv_vdp.commands.parser import build_parser
    from ssv_vdp.local_env import apply_local_env

    args = build_parser().parse_args(argv)
    apply_local_env(args)
    from selfsuvis.pipeline.core import log_preflight, run_local_preflight
    from selfsuvis.pipeline.vision.registry import auto_select, detect_resources

    def _select_model(task: str) -> str:
        return auto_select(task, detect_resources()) or ""

    report = run_local_preflight(args, select_model=_select_model)
    log_preflight(report)
    if report.errors:
        raise RuntimeError("local pipeline preflight failed")
    from ssv_vdp import run_local

    run_local(args)
    return collect_local_result(output_dir)

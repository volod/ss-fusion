"""Packaged entry point for local-run analytics."""

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse a selfsuvis local-run output directory.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a local run output directory (e.g. .data/local_runs/drone_mission).",
    )
    parser.add_argument(
        "--charts-dir",
        default=None,
        help="Directory for individual PNG charts (defaults to --run-dir).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the HTML report.",
    )
    parser.add_argument(
        "--report-filename",
        default="analysis_report.html",
        help="Filename for the HTML report (default: analysis_report.html).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a compact machine-readable summary JSON.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not getattr(args, "run_dir", None):
        raise SystemExit("--run-dir is required when --mode analyse")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/selfsuvis-matplotlib")

    from selfsuvis.analytics import LocalRunLoader
    from selfsuvis.visualization import (
        generate_report,
        plot_detections,
        plot_embedding_pca,
        plot_similarity_matrix,
        plot_timeline,
        plot_training_curves,
    )

    run_dir = Path(args.run_dir)
    charts_dir = Path(args.charts_dir) if args.charts_dir else run_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {run_dir} …")
    summary = LocalRunLoader(run_dir).load()

    print(f"  Video     : {summary.video_name}")
    print(
        f"  Frames    : {summary.n_frames}  ({summary.duration_sec:.1f}s @ {summary.fps:.1f} fps)"
    )
    print(f"  Domain    : {summary.domain or '—'}")
    print(f"  Category  : {summary.top_category or '—'}")
    print(f"  Artifacts : {summary.artifact_inventory.total_files} files")
    print(
        f"  Quality   : {summary.diagnostics.quality_score:.1f}/100  "
        f"(modality {100.0 * summary.diagnostics.modality_completeness:.0f}%)"
    )
    if summary.detection_stats:
        ds = summary.detection_stats
        print(f"  Detections: {ds.total_objects} objects  (mean {ds.mean_per_frame:.1f}/frame)")
    if summary.temporal_stats:
        ts = summary.temporal_stats
        print(
            f"  Surprise  : mean={ts.mean_surprise:.3f}  "
            f"std={summary.diagnostics.surprise_std:.3f}  peaks={len(ts.peak_frames)}"
        )
    if summary.training_stats:
        tr = summary.training_stats
        print(f"  SSL loss  : {tr.ssl_best_loss:.4f}  |  distill R@1={tr.distill_best_r1:.3f}")
    if summary.run_health.warnings:
        print("  Warnings  :")
        for warning in summary.run_health.warnings:
            print(f"    - {warning}")

    print("\nGenerating charts …")
    chart_fns = [
        ("timeline", plot_timeline, (summary,)),
        ("detections", plot_detections, (summary,)),
        ("embedding_pca", plot_embedding_pca, (run_dir,)),
        ("similarity_matrix", plot_similarity_matrix, (run_dir,)),
        ("training", plot_training_curves, (summary,)),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, fn, fn_args in chart_fns:
        out = charts_dir / f"{name}.png"
        try:
            fig = fn(*fn_args)
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [ok] {out.name}")
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")

    if not args.no_report:
        print("\nGenerating HTML report …")
        report_path = generate_report(
            summary,
            out_dir=run_dir,
            report_filename=args.report_filename,
        )
        print(f"  [ok] {report_path}")

    if args.summary_json:
        payload = {
            "video_name": summary.video_name,
            "n_frames": summary.n_frames,
            "duration_sec": summary.duration_sec,
            "domain": summary.domain,
            "top_category": summary.top_category,
            "artifact_count": summary.artifact_inventory.total_files,
            "diagnostics": {
                "quality_score": summary.diagnostics.quality_score,
                "modality_completeness": summary.diagnostics.modality_completeness,
                "tracking_fragmentation": summary.diagnostics.tracking_fragmentation,
                "map_pose_coverage": summary.diagnostics.map_pose_coverage,
                "adaptation_efficiency": summary.diagnostics.adaptation_efficiency,
            },
            "warnings": summary.run_health.warnings,
            "has_3d_map": summary.has_3d_map,
            "has_edge_model": summary.has_edge_model,
        }
        out_path = Path(args.summary_json)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  [ok] {out_path}")

    print("\nDone.")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

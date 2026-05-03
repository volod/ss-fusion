"""Generate a self-contained HTML analysis report from a RunSummary."""


import base64
import io
import logging
from pathlib import Path

from selfsuvis.analytics.models import RunSummary

logger = logging.getLogger(__name__)

_CSS = """
body{font-family:sans-serif;margin:0;padding:20px;background:#f8f9fa;color:#222}
h1{background:#2c3e50;color:#fff;padding:16px 20px;margin:-20px -20px 20px;font-size:1.5em}
h2{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-top:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:12px 0}
.card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
.card .label{font-size:.8em;color:#777;text-transform:uppercase;letter-spacing:.05em}
.card .value{font-size:1.4em;font-weight:700;color:#2c3e50;margin-top:4px}
img{max-width:100%;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.15)}
.chart-row{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}
.chart-row>div{flex:1;min-width:300px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px 12px;text-align:left}
th{background:#ecf0f1}
footer{color:#aaa;font-size:.8em;margin-top:32px;text-align:center}
.warnings{background:#fff3cd;border:1px solid #ffe69c;color:#664d03;padding:12px 14px;border-radius:8px}
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" />'


def generate_report(
    summary: RunSummary,
    out_dir: str | Path | None = None,
    report_filename: str = "analysis_report.html",
) -> Path:
    """Write a self-contained HTML report into out_dir (defaults to run_dir).

    All charts are embedded as base64 PNGs so the file is portable.

    Returns the path to the written HTML file.
    """
    import matplotlib
    matplotlib.use("Agg")

    from selfsuvis.visualization.detections import plot_detections
    from selfsuvis.visualization.embeddings import plot_embedding_pca, plot_similarity_matrix
    from selfsuvis.visualization.timeline import plot_timeline
    from selfsuvis.visualization.training import plot_training_curves

    out_dir = Path(out_dir) if out_dir else Path(summary.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / report_filename

    charts: dict[str, str] = {}
    for name, fn, args in [
        ("timeline",    plot_timeline,         (summary,)),
        ("detections",  plot_detections,       (summary,)),
        ("pca",         plot_embedding_pca,    (summary.run_dir,)),
        ("sim_matrix",  plot_similarity_matrix,(summary.run_dir,)),
        ("training",    plot_training_curves,  (summary,)),
    ]:
        try:
            fig = fn(*args)
            charts[name] = _fig_to_b64(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            logger.warning("Chart '%s' failed: %s", name, exc)

    ds = summary.detection_stats
    ts = summary.temporal_stats
    tr = summary.training_stats
    th = summary.tracking_stats
    emb = summary.embedding_stats
    diag = summary.diagnostics

    # Build HTML
    sections: list[str] = []

    # --- Summary cards ---
    cards_html = _cards([
        ("Video", summary.video_name),
        ("Frames", str(summary.n_frames)),
        ("Duration", f"{summary.duration_sec:.1f}s"),
        ("FPS", f"{summary.fps:.1f}"),
        ("Domain", summary.domain or "—"),
        ("Scene", summary.scene_complexity or "—"),
        ("Top category", summary.top_category or "—"),
        ("Quality score", f"{diag.quality_score:.1f}/100"),
        ("Modality coverage", f"{100.0 * diag.modality_completeness:.0f}%"),
        ("3D map", "yes" if summary.has_3d_map else "no"),
        ("Edge model", "yes" if summary.has_edge_model else "no"),
        ("Artifacts", str(summary.artifact_inventory.total_files)),
    ])
    sections.append(f"<h2>Overview</h2>{cards_html}")

    if summary.run_health.warnings:
        warnings_html = "".join(f"<li>{warning}</li>" for warning in summary.run_health.warnings)
        sections.append(f"<h2>Run Health</h2><div class='warnings'><ul>{warnings_html}</ul></div>")

    # --- Timeline ---
    if "timeline" in charts:
        sections.append(
            f'<h2>Temporal Analysis</h2>'
            f'<div class="chart-row"><div>{_img_tag(charts["timeline"])}</div></div>'
        )
        if ts:
            peak_ts = [f"{summary.frames[i].t_sec:.1f}s" for i in ts.peak_frames
                       if i < len(summary.frames)]
            sections.append(
                f"<p>Mean surprise: <b>{ts.mean_surprise:.3f}</b> | "
                f"Peak frames (top 10%): {', '.join(peak_ts) or '—'}</p>"
            )

    # --- Detections ---
    if "detections" in charts:
        sections.append(
            f'<h2>Object Detections</h2>'
            f'<div class="chart-row"><div>{_img_tag(charts["detections"])}</div></div>'
        )
        if ds:
            by_class_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in ds.by_class.items()
            )
            sections.append(
                f"<table><tr><th>Class</th><th>Count</th></tr>{by_class_rows}</table>"
            )

    # --- Embeddings ---
    emb_charts = [c for c in ("pca", "sim_matrix") if c in charts]
    if emb_charts:
        row = "".join(f'<div>{_img_tag(charts[c])}</div>' for c in emb_charts)
        sections.append(f'<h2>Embedding Space</h2><div class="chart-row">{row}</div>')

    # --- Training ---
    if "training" in charts:
        sections.append(
            f'<h2>Model Training</h2>'
            f'<div class="chart-row"><div>{_img_tag(charts["training"])}</div></div>'
        )
        if tr:
            sections.append(
                f"<p>SSL best loss: <b>{tr.ssl_best_loss:.4f}</b> | "
                f"Distill R@1: <b>{tr.distill_best_r1:.3f}</b> | "
                f"Compression: <b>{tr.distill_compression:.1f}×</b> | "
                f"ONNX size: <b>{tr.onnx_mb:.1f} MB</b></p>"
            )

    if th or emb or summary.map_stats:
        cards: list[tuple[str, str]] = []
        if th:
            cards.extend([
                ("Tracking model", th.model or "—"),
                ("Track IDs", str(th.unique_track_ids)),
                ("SAM masks", str(th.sam_masks_total)),
                ("Fragmentation", f"{diag.tracking_fragmentation:.3f}"),
                ("Track persistence", f"{diag.track_persistence:.3f}"),
            ])
        if emb:
            cards.extend([
                ("Embeddings", f"{emb.n_embeddings} × {emb.embedding_dim}"),
                ("Mean NN sim", f"{emb.mean_neighbour_similarity:.3f}"),
            ])
        if summary.map_stats:
            cards.extend([
                ("Map method", summary.map_stats.method or "—"),
                ("Map points", str(summary.map_stats.points)),
                ("Map poses", str(summary.map_stats.poses)),
                ("Points / pose", f"{diag.map_points_per_pose:.1f}"),
                ("Pose coverage", f"{100.0 * diag.map_pose_coverage:.0f}%"),
            ])
        sections.append(f"<h2>Derived Artifacts</h2>{_cards(cards)}")

    diagnostics_cards = _cards([
        ("Detection density", f"{diag.detection_density_per_frame:.2f}/frame"),
        ("Detection CV", f"{diag.detection_count_cv:.2f}"),
        ("Class entropy", f"{diag.detection_entropy_norm:.2f}"),
        ("Surprise std", f"{diag.surprise_std:.3f}"),
        ("Peak rate", f"{100.0 * diag.surprise_peak_rate:.0f}%"),
        ("Peak/object overlap", f"{100.0 * diag.surprise_detection_overlap:.0f}%"),
        ("Adapt efficiency", f"{diag.adaptation_efficiency:.3f}"),
        ("Artifacts / frame", f"{diag.artifact_density_per_frame:.2f}"),
    ])
    sections.append(f"<h2>Analytics Diagnostics</h2>{diagnostics_cards}")

    # --- Artifacts table ---
    artifacts = _list_artifacts(summary)
    if artifacts:
        rows = "".join(f"<tr><td>{name}</td><td>{size}</td></tr>" for name, size in artifacts)
        sections.append(
            f"<h2>Output Artifacts</h2>"
            f"<table><tr><th>File</th><th>Size</th></tr>{rows}</table>"
        )

    body = "\n".join(sections)
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>selfsuvis — {summary.video_name}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>selfsuvis Local Run Report — {summary.video_name}</h1>"
        f"{body}"
        f"<footer>Generated by selfsuvis analytics</footer>"
        f"</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path


def _cards(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="grid">{cards}</div>'


def _list_artifacts(summary: RunSummary) -> list[tuple[str, str]]:
    rows = []
    for artifact in summary.artifact_inventory.files:
        size_kb = artifact.size_bytes / 1024
        label = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        rows.append((artifact.path, label))
    return rows

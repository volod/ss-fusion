"""Interactive 3D point-cloud viewer for sparse_map.npz files.

Uses matplotlib (optional dependency). Falls back gracefully when matplotlib
is not installed.

Public API
----------
open_3d_viewers(viewer_data)
    Open one figure per entry in viewer_data and block until all closed.
    Each entry: {"title": str, "points": ndarray(N,3), "colours": ndarray(N,3)}

collect_npz_files(path_str, output_dir) -> List[Path]
    Resolve one or more sparse_map.npz paths from a flexible path_str.

view_npz(path_str, output_dir)
    Load NPZ file(s) and open interactive viewer.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

try:
    import sys

    import matplotlib

    matplotlib.use("TkAgg" if sys.platform != "linux" else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mwidgets
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def build_viewer_figure(title: str, points: np.ndarray, colours: np.ndarray):
    """Build a dark-theme matplotlib 3D scatter figure with a Close button."""
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    if len(points) > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colours.clip(0, 1),
            s=4,
            linewidths=0,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(f"{len(points)} points", color="white", fontsize=9, pad=4)

    btn_ax = fig.add_axes([0.82, 0.02, 0.14, 0.05])
    btn = mwidgets.Button(btn_ax, "Close Viewer", color="#c0392b", hovercolor="#e74c3c")
    btn.label.set_color("white")
    btn.on_clicked(lambda _evt: plt.close(fig))

    return fig


def open_3d_viewers(viewer_data: list[dict[str, Any]]) -> None:
    """Open one matplotlib figure per entry; block until all windows are closed."""
    if not _HAS_MPL:
        logger.warning("matplotlib unavailable — skipping 3D viewers")
        return
    if not viewer_data:
        return

    logger.info("Opening %d 3D viewer(s) — close each window to continue", len(viewer_data))
    for vd in viewer_data:
        pts = vd.get("points", np.zeros((1, 3)))
        col = vd.get("colours", np.ones((1, 3)))
        build_viewer_figure(vd["title"], pts, col)
        logger.info("  Opened: %s", vd["title"])

    try:
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
    except Exception as exc:
        logger.warning("3D viewer error: %s", exc)

    logger.info("All 3D viewers closed.")


def collect_npz_files(path_str: str, output_dir: Path) -> list[Path]:
    """Resolve sparse_map.npz files from *path_str*.

    - Empty string → scan *output_dir* recursively for all sparse_map.npz
    - Directory    → look for <dir>/3d_map/sparse_map.npz (or recurse)
    - *.npz file   → use directly
    """
    if not path_str:
        found = sorted(output_dir.rglob("sparse_map.npz"))
        if not found:
            logger.error("No sparse_map.npz files found under %s", output_dir)
        return found

    p = Path(path_str)
    if p.is_file() and p.suffix == ".npz":
        return [p]
    if p.is_dir():
        candidate = p / "3d_map" / "sparse_map.npz"
        if candidate.exists():
            return [candidate]
        found = sorted(p.rglob("sparse_map.npz"))
        if not found:
            logger.error("No sparse_map.npz found in %s", p)
        return found
    logger.error("Path does not exist or is not a .npz file: %s", path_str)
    return []


def view_npz(path_str: str, output_dir: Path) -> None:
    """Load sparse_map.npz file(s) and open the interactive 3D viewer.

    Parameters
    ----------
    path_str   : "" to scan output_dir, a directory, or a specific .npz path
    output_dir : fallback root used when path_str is empty
    """
    if not _HAS_MPL:
        logger.warning("matplotlib is required for the 3D viewer — skipping.")
        logger.warning("  Install: pip install matplotlib")
        return

    npz_files = collect_npz_files(path_str, output_dir)
    if not npz_files:
        return

    viewer_data: list[dict[str, Any]] = []
    for npz_path in npz_files:
        try:
            data = np.load(str(npz_path))
            points = data["points"]
            colours = data["colours"]

            parts = npz_path.parts
            try:
                idx = parts.index("3d_map")
                label = parts[idx - 1] if idx > 0 else npz_path.stem
            except ValueError:
                label = npz_path.parent.name

            method = "unknown"
            stats_file = npz_path.parent / "map_stats.json"
            if stats_file.exists():
                try:
                    method = json.loads(stats_file.read_text())["method"]
                except Exception:
                    pass

            viewer_data.append(
                {
                    "title": f"3D Map — {label}  ({method})  [{len(points)} pts]",
                    "points": points,
                    "colours": colours,
                }
            )
            logger.info("Loaded %s: %d points  method=%s", npz_path, len(points), method)
        except Exception as exc:
            logger.error("Failed to load %s: %s", npz_path, exc)

    if not viewer_data:
        logger.warning("No valid NPZ files could be loaded — skipping 3D viewer.")
        return

    open_3d_viewers(viewer_data)

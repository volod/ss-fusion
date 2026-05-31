"""Validate that SSL fine-tuning improved temporal recall on held-out videos.

Usage:
    python scripts/validate_ssl_improvement.py \\
        --base-model /path/to/base_dino.pt \\
        --finetuned /path/to/finetuned_dino.pt \\
        --video-dirs /data/val/video_a /data/val/video_b /data/val/video_c

Exit codes:
    0 — gate passed (fine-tuned model improves R@1 on ≥ _GATE_VIDEOS videos)
    1 — gate failed
"""

import argparse
from pathlib import Path

import numpy as np

# Gate constants — edit carefully, tests assert exact values.
_GATE_VIDEOS = 2  # minimum number of validation videos that must improve
_GATE_DELTA = 0.02  # minimum R@1 improvement (strict >)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_DEFAULT_WINDOW = 3  # temporal window for R@1 (±window frames are considered positives)


def _collect_frames(directory) -> list[Path]:
    """Return sorted list of image paths under `directory` (recursive)."""
    root = Path(directory)
    frames = [p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()]
    return sorted(frames)


def recall_at_1(embeddings: np.ndarray, window: int = _DEFAULT_WINDOW) -> float:
    """Temporal Recall@1 — fraction of frames whose nearest neighbour falls within ±window.

    Args:
        embeddings: (N, D) float32 array of L2-normalised frame embeddings in
                    temporal order.
        window:     Number of adjacent frames on each side that count as positives.

    Returns:
        float in [0.0, 1.0]; 0.0 when N < 2.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # exclude self-match

    nn_idx = np.argmax(sims, axis=1)
    hits = sum(1 for i, j in enumerate(nn_idx) if abs(i - j) <= window)
    return float(hits) / n


def _embed_frames(model, frames: list[Path]) -> np.ndarray:
    """Encode frames with a DINOEmbedder; return L2-normalised (N, D) array."""
    from PIL import Image as PILImage

    images = [PILImage.open(p).convert("RGB") for p in frames]
    vecs = model.encode_images(images)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms < 1e-8, 1.0, norms)


def _evaluate_video(base_model, ft_model, video_dir: Path) -> dict:
    frames = _collect_frames(video_dir)
    if not frames:
        return {
            "frames": 0,
            "r1_base": 0.0,
            "r1_ft": 0.0,
            "delta_median": 0.0,
            "gate_passed": False,
        }

    embs_base = _embed_frames(base_model, frames)
    embs_ft = _embed_frames(ft_model, frames)
    r1_base = recall_at_1(embs_base)
    r1_ft = recall_at_1(embs_ft)
    delta = r1_ft - r1_base
    return {
        "frames": len(frames),
        "r1_base": r1_base,
        "r1_ft": r1_ft,
        "delta_median": delta,
        "gate_passed": delta > _GATE_DELTA,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate SSL fine-tune improvement")
    p.add_argument("--base-model", required=True)
    p.add_argument("--finetuned", required=True)
    p.add_argument("--video-dirs", nargs="+", required=True)
    p.add_argument("--model-name", default="dinov3")
    return p


def main() -> int:
    import json as _json

    from selfsuvis.models.dino_model import DINOEmbedder
    from selfsuvis.pipeline.core import get_dino_model_name

    args = _build_parser().parse_args()
    dino_name = get_dino_model_name(args.model_name)

    base_model = DINOEmbedder(dino_name)
    ft_model = DINOEmbedder(dino_name)
    ft_model.load_backbone_checkpoint(args.finetuned)

    results = {}
    for vdir in args.video_dirs:
        results[vdir] = _evaluate_video(base_model, ft_model, Path(vdir))

    n_passed = sum(1 for r in results.values() if r["gate_passed"])
    gate = n_passed >= _GATE_VIDEOS
    print(_json.dumps({"results": results, "n_passed": n_passed, "gate": gate}, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())

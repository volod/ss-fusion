"""Generate synthetic 3DGS PLY test fixtures.

Usage:
    python scripts/generate_test_splat.py <output.ply> [--n 200] [--seed 0]
                                          [--lat 48.0] [--lon 11.0] [--alt 100.0]
                                          [--radius 10.0]
"""
from __future__ import annotations

import argparse

import numpy as np

from selfsuvis.pipeline.mapping.splat_io import _SPLAT_DTYPE, write_splat, write_splat_metadata

_DEFAULT_N = 200
_DEFAULT_RADIUS_M = 10.0
_DEFAULT_LAT = 48.0
_DEFAULT_LON = 11.0
_DEFAULT_ALT = 100.0
_DEFAULT_SEED = 0


def generate_splat(
    path: str,
    n_gaussians: int = _DEFAULT_N,
    origin_lat: float = _DEFAULT_LAT,
    origin_lon: float = _DEFAULT_LON,
    origin_alt: float = _DEFAULT_ALT,
    radius_m: float = _DEFAULT_RADIUS_M,
    seed: int = _DEFAULT_SEED,
) -> None:
    """Write a synthetic 3DGS PLY with random Gaussians centred at the origin.

    All positions are drawn from a uniform disc of radius `radius_m` in the XY
    plane (Z ∈ [-1, 1]).  Opacity and scale are set to neutral defaults.
    Metadata (origin GPS + n_gaussians) is written as a companion JSON file.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros(n_gaussians, dtype=_SPLAT_DTYPE)

    # Positions: uniform disc in XY, small Z variation
    angles = rng.uniform(0, 2 * np.pi, n_gaussians)
    radii = rng.uniform(0, radius_m, n_gaussians)
    data["x"] = (radii * np.cos(angles)).astype(np.float32)
    data["y"] = (radii * np.sin(angles)).astype(np.float32)
    data["z"] = rng.uniform(-1.0, 1.0, n_gaussians).astype(np.float32)

    # Neutral opacity (logit ~0 → sigmoid ~0.5) and scale (log-encoded ~1m)
    data["opacity"] = np.zeros(n_gaussians, dtype=np.float32)
    data["scale_0"] = data["scale_1"] = data["scale_2"] = np.full(n_gaussians, -1.0, dtype=np.float32)

    # Identity quaternion (WXYZ): w=1, x=y=z=0
    data["rot_0"] = 1.0

    write_splat(path, data)
    write_splat_metadata(path, origin_lat, origin_lon, origin_alt,
                         extra={"n_gaussians": n_gaussians})


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a synthetic splat PLY for testing")
    p.add_argument("output", help="Output .ply path")
    p.add_argument("--n", type=int, default=_DEFAULT_N, dest="n_gaussians")
    p.add_argument("--lat", type=float, default=_DEFAULT_LAT, dest="origin_lat")
    p.add_argument("--lon", type=float, default=_DEFAULT_LON, dest="origin_lon")
    p.add_argument("--alt", type=float, default=_DEFAULT_ALT, dest="origin_alt")
    p.add_argument("--radius", type=float, default=_DEFAULT_RADIUS_M, dest="radius_m")
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_splat(
        args.output,
        n_gaussians=args.n_gaussians,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        origin_alt=args.origin_alt,
        radius_m=args.radius_m,
        seed=args.seed,
    )
    print(f"Written {args.n_gaussians} Gaussians → {args.output}")

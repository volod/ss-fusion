"""Helpers for stable sector and route indexing across videos."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

_EARTH_RADIUS_M = 6_378_137.0


def enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> tuple[float, float, float]:
    """Approximate ENU -> geodetic conversion for local-scale trajectories."""
    lat_rad = math.radians(origin_lat)
    dlat = north_m / _EARTH_RADIUS_M
    dlon = east_m / (_EARTH_RADIUS_M * max(1e-6, math.cos(lat_rad)))
    lat = origin_lat + math.degrees(dlat)
    lon = origin_lon + math.degrees(dlon)
    alt = origin_alt + up_m
    return lat, lon, alt


def latlon_to_sector_id(
    lat: float,
    lon: float,
    tile_size_m: float = 50.0,
) -> str:
    """Quantize geodetic coordinates into a stable sector/tile identifier."""
    tile_deg_lat = tile_size_m / 111_320.0
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    tile_deg_lon = tile_size_m / (111_320.0 * cos_lat)
    lat_idx = int(math.floor(lat / max(tile_deg_lat, 1e-9)))
    lon_idx = int(math.floor(lon / max(tile_deg_lon, 1e-9)))
    return f"sector_{lat_idx}_{lon_idx}"


def sectorize_global_positions(
    origin_lla: dict[str, float],
    positions_enu_m: Sequence[dict[str, float]],
    tile_size_m: float = 50.0,
) -> list[dict[str, Any]]:
    """Map ENU trajectory samples into global sector IDs."""
    out: list[dict[str, Any]] = []
    origin_lat = float(origin_lla["lat"])
    origin_lon = float(origin_lla["lon"])
    origin_alt = float(origin_lla.get("alt", 0.0))
    for idx, pos in enumerate(positions_enu_m):
        east = float(pos.get("x", 0.0))
        north = float(pos.get("y", 0.0))
        up = float(pos.get("z", 0.0))
        lat, lon, alt = enu_to_geodetic(east, north, up, origin_lat, origin_lon, origin_alt)
        out.append(
            {
                "sample_index": idx,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "sector_id": latlon_to_sector_id(lat, lon, tile_size_m=tile_size_m),
            }
        )
    return out


def unique_sector_sequence(sector_samples: Sequence[dict[str, Any]]) -> list[str]:
    """Collapse consecutive duplicates while preserving route order."""
    sequence: list[str] = []
    prev = None
    for item in sector_samples:
        sector_id = str(item.get("sector_id", ""))
        if not sector_id or sector_id == prev:
            continue
        sequence.append(sector_id)
        prev = sector_id
    return sequence


def build_route_segment_id(video_name: str, sector_sequence: Sequence[str]) -> str:
    if not sector_sequence:
        return f"route_{video_name}"
    start = sector_sequence[0]
    end = sector_sequence[-1]
    if start == end:
        return f"route_{start}"
    return f"route_{start}__{end}"


def build_sector_adjacency(sector_sequence: Sequence[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left, right in zip(sector_sequence, sector_sequence[1:]):
        if not left or not right or left == right:
            continue
        edges.append({"from_sector": left, "to_sector": right, "weight": 1.0})
    return edges

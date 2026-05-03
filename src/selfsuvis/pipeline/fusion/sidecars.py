from collections.abc import Iterable
from typing import Any

from selfsuvis.pipeline.core.sidecars import load_video_jsonl_sidecar


def load_imu_sidecar(video_path: str) -> list[dict[str, Any]]:
    return load_video_jsonl_sidecar(video_path, ".imu.jsonl")


def load_baro_sidecar(video_path: str) -> list[dict[str, Any]]:
    return load_video_jsonl_sidecar(video_path, ".baro.jsonl")


def pressure_hpa_to_altitude_m(pressure_hpa: float) -> float:
    # ISA approximation near sea level; good enough for a local fusion prior.
    return 44330.0 * (1.0 - (pressure_hpa / 1013.25) ** 0.1903)


def normalize_baro_rows(rows: Iterable[dict[str, Any]], origin_alt_m: float) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for row in rows:
        t_sec = float(row.get("t", row.get("timestamp", 0.0)) or 0.0)
        if "alt_m" in row:
            alt_m = float(row["alt_m"])
        elif "pressure_hpa" in row:
            alt_m = pressure_hpa_to_altitude_m(float(row["pressure_hpa"]))
        else:
            continue
        normalized.append({"t_sec": t_sec, "alt_enu_m": alt_m - origin_alt_m})
    return normalized

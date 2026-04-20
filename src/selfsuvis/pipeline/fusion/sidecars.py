from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    rows.sort(key=lambda row: float(row.get("t", row.get("timestamp", 0.0)) or 0.0))
    return rows


def _sidecar_path(video_path: str, suffix: str) -> Path:
    return Path(video_path).with_suffix(suffix)


def load_imu_sidecar(video_path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(_sidecar_path(video_path, ".imu.jsonl"))


def load_baro_sidecar(video_path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(_sidecar_path(video_path, ".baro.jsonl"))


def pressure_hpa_to_altitude_m(pressure_hpa: float) -> float:
    # ISA approximation near sea level; good enough for a local fusion prior.
    return 44330.0 * (1.0 - (pressure_hpa / 1013.25) ** 0.1903)


def normalize_baro_rows(rows: Iterable[Dict[str, Any]], origin_alt_m: float) -> List[Dict[str, float]]:
    normalized: List[Dict[str, float]] = []
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

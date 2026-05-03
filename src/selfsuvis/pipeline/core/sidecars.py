"""Shared sidecar file and HTTP client helpers."""

import json
from pathlib import Path
from typing import Any

import httpx


def sidecar_path(video_path: str | Path, suffix: str) -> Path:
    return Path(video_path).with_suffix(suffix)


def load_jsonl_sidecar(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate = Path(path)
    if not candidate.is_file():
        return rows
    with candidate.open(encoding="utf-8") as handle:
        for line in handle:
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


def load_video_jsonl_sidecar(video_path: str | Path, suffix: str) -> list[dict[str, Any]]:
    return load_jsonl_sidecar(sidecar_path(video_path, suffix))


class HttpSidecarClient:
    """Small shared base for HTTP-backed sidecars."""

    def __init__(
        self,
        *,
        backend_name: str,
        base_url: str | None,
        timeout_sec: float,
    ) -> None:
        self._backend_name = str(backend_name or "").strip().lower()
        self._base_url = str(base_url or "").rstrip("/")
        self._timeout_sec = float(timeout_sec)

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url)

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any | None:
        if not self.is_configured:
            return None
        async with httpx.AsyncClient(timeout=self._timeout_sec) as client:
            resp = await client.request(method, f"{self._base_url}{path}", json=payload)
            if allow_404 and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def unwrap_dict_payload(data: Any, *, field: str | None = None) -> dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        if field and isinstance(data.get(field), dict):
            return data[field]
        return data

    async def stats(self) -> dict[str, Any]:
        if not self.is_configured:
            return {"configured": False, "backend": self._backend_name}
        data = await self.request_json("GET", "/stats")
        if isinstance(data, dict):
            return {"configured": True, "backend": self._backend_name, **data}
        return {"configured": True, "backend": self._backend_name}

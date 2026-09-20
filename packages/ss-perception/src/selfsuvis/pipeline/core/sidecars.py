"""Shared sidecar file and HTTP client helpers."""

from ss_kit.sidecar import (
    HttpSidecarClient,
    load_jsonl_sidecar,
    load_media_jsonl_sidecar,
    sidecar_path,
)

load_video_jsonl_sidecar = load_media_jsonl_sidecar

__all__ = [
    "HttpSidecarClient",
    "load_jsonl_sidecar",
    "load_video_jsonl_sidecar",
    "sidecar_path",
]

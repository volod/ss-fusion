"""Canonical SHA-256 helpers (ASCII JSON, lowercase hex)."""

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_hex(value: Any) -> str:
    if isinstance(value, bytes):
        return sha256_bytes(value)
    if isinstance(value, str):
        return sha256_bytes(value.encode("utf-8"))
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(blob.encode("utf-8"))

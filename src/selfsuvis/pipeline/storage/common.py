"""Shared helpers for asyncpg-backed storage modules."""

import json
from collections.abc import Iterable
from typing import Any


def jsonb(value: Any, *, default: Any = None) -> str:
    if value is None:
        value = default
    return json.dumps(value)


def jsonb_optional(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def decoded_json(value: Any, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def row_dict(row: Any) -> dict[str, Any] | None:
    return dict(row) if row else None


def row_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]

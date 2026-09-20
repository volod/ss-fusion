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


_UINT64 = 1 << 64
_INT64_MAX = (1 << 63) - 1


def qdrant_id_to_pg(value: Any) -> int | str | None:
    """Encode a Qdrant uint64 point id as PostgreSQL signed BIGINT.

    Non-numeric ids (UUID or test strings) are returned unchanged.
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return value
    if n > _INT64_MAX:
        n -= _UINT64
    return n


def qdrant_id_from_pg(value: Any) -> int | str | None:
    """Decode a PostgreSQL BIGINT back to a Qdrant uint64 point id."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return value
    if n < 0:
        n += _UINT64
    return n

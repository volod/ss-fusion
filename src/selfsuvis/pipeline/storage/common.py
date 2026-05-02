"""Shared helpers for asyncpg-backed storage modules."""

import json
from typing import Any, Dict, Iterable, List, Optional


def jsonb(value: Any, *, default: Any = None) -> str:
    if value is None:
        value = default
    return json.dumps(value)


def jsonb_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value)


def decoded_json(value: Any, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def row_dict(row: Any) -> Optional[Dict[str, Any]]:
    return dict(row) if row else None


def row_dicts(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]

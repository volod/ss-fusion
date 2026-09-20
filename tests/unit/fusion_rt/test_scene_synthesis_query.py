import asyncio
import re
from typing import Any

from selfsuvis.fusion_rt.scene_synthesis import SceneSynthesizer
from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from selfsuvis.scripts.migrate_postgres import _SCHEMA

SQL_WORDS = {
    "select", "from", "where", "order", "by", "desc", "limit", "as", "now", "interval",
    "minutes", "scene_timeline",
}  # fmt: skip


def _table_columns(table: str) -> set[str]:
    create = next(sql for sql in _SCHEMA if f"CREATE TABLE IF NOT EXISTS {table} (" in sql)
    body = create.split("(", 1)[1]
    return {line.split()[0] for line in body.splitlines() if line.strip() and line[:1] == " "}


def test_recent_captions_query_uses_existing_columns() -> None:
    queries: list[str] = []

    class _Conn:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
            queries.append(query)
            return [{"mission_id": "m1", "ts": "t", "caption": "c", "facts_json": None}]

    class _Acquire:
        async def __aenter__(self) -> _Conn:
            return _Conn()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

    synthesizer = SceneSynthesizer(CombinedSiteSnapshot(), db_pool=_Pool())
    rows = asyncio.run(synthesizer._fetch_recent_captions())

    assert rows[0]["ts"] == "t"
    columns = _table_columns("scene_timeline")
    assert "created_at" in columns
    words = set(re.findall(r"[a-z_]+", queries[0].lower())) - SQL_WORDS
    assert words - {"ts"} <= columns, words - columns
    assert re.search(r"created_at\s+as\s+ts", queries[0], re.IGNORECASE)

"""Idempotent PostgreSQL DDL helpers shared by video and fusion migrations."""

from selfsuvis.pipeline.core.db_urls import admin_database_url, database_name
from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


async def apply_statements(conn, statements, *, verbose: bool = True) -> None:
    """Apply every idempotent DDL statement to an open connection."""
    total = len(statements)
    for i, sql in enumerate(statements, 1):
        stmt = sql.strip()
        label = stmt.split("\n")[0][:80].strip()
        try:
            await conn.execute(stmt)
        except Exception as exc:
            # Concurrent CREATE TABLE IF NOT EXISTS can still race on pg_type.
            if type(exc).__name__ != "UniqueViolationError":
                raise
            if not stmt.upper().lstrip().startswith("CREATE"):
                raise
            if verbose:
                print(f"  [{i:02d}/{total}] {label} (already exists)")
            continue
        if verbose:
            print(f"  [{i:02d}/{total}] {label}")


async def with_advisory_lock(conn, lock_key: int, body) -> None:
    """Run ``body`` while holding a session-level advisory lock."""
    await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
    try:
        await body()
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)


async def ensure_database(url: str, *, verbose: bool = True) -> None:
    """Create the database named in ``url`` if it does not already exist.

    Connects to the ``postgres`` database on the same host. Existing Docker
    volumes skip ``initdb.d``, so API/worker startup must call this before
    applying fusion DDL. The role in ``url`` must be allowed to CREATE DATABASE
    (the compose ``POSTGRES_USER`` is a superuser).
    """
    import asyncpg

    db_name = database_name(url)
    if not db_name or db_name in {"postgres", "template0", "template1"}:
        return
    admin_url = admin_database_url(url)
    conn = await asyncpg.connect(admin_url)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if exists:
            return
        await conn.execute(f'CREATE DATABASE "{db_name}"')
        if verbose:
            print(f"Created database {db_name}")
        else:
            logger.info("created database %s", db_name)
    except Exception as exc:
        if type(exc).__name__ == "DuplicateDatabaseError":
            return
        raise
    finally:
        await conn.close()

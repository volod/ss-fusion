"""Parse and rewrite PostgreSQL DSNs without importing a service owner."""

from urllib.parse import urlsplit, urlunsplit

_DB_NAME_RE = None


def _db_name_ok(name: str) -> bool:
    global _DB_NAME_RE
    if _DB_NAME_RE is None:
        import re

        _DB_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    return bool(_DB_NAME_RE.fullmatch(name))


def database_name(url: str) -> str:
    """Return the database name from a PostgreSQL URL path."""
    path = urlsplit(url).path.lstrip("/")
    return path.split("/")[0].split("?")[0]


def sibling_database_url(url: str, db_name: str) -> str:
    """Return ``url`` with the database name replaced by ``db_name``.

    Query strings and credentials are preserved. ``db_name`` must be a plain
    SQL identifier (letters, digits, underscore).
    """
    if not url:
        raise ValueError("database URL is empty")
    if not _db_name_ok(db_name):
        raise ValueError(f"invalid database name: {db_name!r}")
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{db_name}", parts.query, parts.fragment))


def admin_database_url(url: str, admin_db: str = "postgres") -> str:
    """Return a DSN that connects to ``admin_db`` on the same host as ``url``."""
    return sibling_database_url(url, admin_db)

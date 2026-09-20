"""Unit tests for PostgreSQL DSN helpers."""

import pytest

from selfsuvis.pipeline.core.db_urls import (
    admin_database_url,
    database_name,
    sibling_database_url,
)


def test_sibling_database_url_replaces_path_keeps_query() -> None:
    url = "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis?sslmode=disable"
    out = sibling_database_url(url, "selfsuvis_fusion")
    assert out == "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis_fusion?sslmode=disable"


def test_database_name_and_admin_url() -> None:
    url = "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis"
    assert database_name(url) == "selfsuvis"
    assert admin_database_url(url) == "postgresql://selfsuvis:selfsuvis@localhost:5432/postgres"


def test_sibling_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        sibling_database_url("postgresql://u:p@h:5432/selfsuvis", "bad-name")

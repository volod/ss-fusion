"""Qdrant uint64 point ids stored as PostgreSQL signed BIGINT."""

from selfsuvis.pipeline.storage.common import qdrant_id_from_pg, qdrant_id_to_pg

OVERFLOW_ID = 13967740624921136041


def test_qdrant_id_roundtrip_fits_int64() -> None:
    encoded = qdrant_id_to_pg(OVERFLOW_ID)
    assert encoded is not None
    assert -(1 << 63) <= encoded <= (1 << 63) - 1
    assert qdrant_id_from_pg(encoded) == OVERFLOW_ID


def test_qdrant_id_none_passthrough() -> None:
    assert qdrant_id_to_pg(None) is None
    assert qdrant_id_from_pg(None) is None


def test_qdrant_id_non_numeric_passthrough() -> None:
    assert qdrant_id_from_pg("q0") == "q0"
    assert qdrant_id_to_pg("uuid-1") == "uuid-1"

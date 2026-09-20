"""Fusion-owned PostgreSQL schema (sensor keys, events, zones, rules, incidents)."""

import asyncpg

from selfsuvis.pipeline.core.ddl import apply_statements, ensure_database, with_advisory_lock

FUSION_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sensor_keys (
        key_hash    TEXT PRIMARY KEY,
        sensor_id   TEXT NOT NULL,
        scopes      TEXT[] NOT NULL DEFAULT '{ingest}',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_sensor_keys_sensor_id ON sensor_keys (sensor_id)",
    """
    CREATE TABLE IF NOT EXISTS site_events (
        event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ts          TIMESTAMPTZ NOT NULL,
        zone_id     TEXT NOT NULL,
        sensor_id   TEXT NOT NULL,
        modality    TEXT NOT NULL
                    CHECK (modality IN ('camera','audio','rf','thermal','vibration','custom')),
        confidence  FLOAT CHECK (confidence BETWEEN 0.0 AND 1.0),
        payload     JSONB NOT NULL DEFAULT '{}',
        artifact_uri TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_site_events_ts_zone ON site_events (ts DESC, zone_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_site_events_modality_ts ON site_events (modality, zone_id, ts DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS zones (
        zone_id     TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        description TEXT,
        map_x       INTEGER,
        map_y       INTEGER,
        map_w       INTEGER,
        map_h       INTEGER,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS fusion_rules (
        rule_id         TEXT PRIMARY KEY,
        label           TEXT NOT NULL,
        modalities      TEXT[] NOT NULL,
        zone_id         TEXT,
        window_s        INTEGER NOT NULL DEFAULT 30,
        min_confidence  FLOAT NOT NULL DEFAULT 0.5,
        enabled         BOOLEAN NOT NULL DEFAULT TRUE,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS incidents (
        incident_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        ts               TIMESTAMPTZ NOT NULL,
        zone_id          TEXT NOT NULL,
        modalities       TEXT[] NOT NULL,
        confidence       FLOAT NOT NULL,
        risk_level       TEXT NOT NULL
                         CHECK (risk_level IN ('low','medium','high','critical')),
        summary_text     TEXT,
        evidence_refs    JSONB NOT NULL,
        rule_id          TEXT,
        acknowledged_at  TIMESTAMPTZ,
        dismissed_at     TIMESTAMPTZ,
        dismissal_reason TEXT,
        created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_incidents_zone_ts ON incidents (zone_id, ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents (ts DESC)",
    """
CREATE INDEX IF NOT EXISTS idx_incidents_summary_fts ON incidents USING GIN(to_tsvector('english', COALESCE(summary_text, '')))
    """,
    """
    CREATE TABLE IF NOT EXISTS incident_notes (
        note_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        incident_id UUID NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
        body        TEXT NOT NULL,
        operator_id TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_incident_notes_incident_id ON incident_notes (incident_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_incident_notes_body_fts ON incident_notes USING GIN(to_tsvector('english', body))
    """,
]

FUSION_SCHEMA_LOCK_KEY = 87263402


async def apply_schema(conn, *, verbose: bool = True) -> None:
    """Apply every fusion DDL statement to an open connection."""
    await apply_statements(conn, FUSION_SCHEMA, verbose=verbose)


async def migrate(url: str, *, verbose: bool = True) -> None:
    """Create the fusion database if needed, then apply the fusion schema."""
    await ensure_database(url, verbose=verbose)
    if verbose:
        print(f"Connecting to: {url.split('@')[-1]}")
    conn = await asyncpg.connect(url)
    try:

        async def _body():
            await apply_schema(conn, verbose=verbose)

        await with_advisory_lock(conn, FUSION_SCHEMA_LOCK_KEY, _body)
        if verbose:
            print(f"\nFusion schema bootstrap complete - {len(FUSION_SCHEMA)} statements applied.")
    finally:
        await conn.close()

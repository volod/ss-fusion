"""Generate and provision a sensor key.

Usage:
    python -m selfsuvis.scripts.add_sensor_key --sensor-id <id> [--scopes ingest,...]

Generates a random key, stores its SHA-256 hash in sensor_keys, and prints the
raw key value once. The raw key is never stored; if lost, delete the row and
re-provision.
"""

import argparse
import asyncio
import hashlib
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import asyncpg

from selfsuvis.pipeline.core.env import env_str, load_layered_env

load_layered_env(anchor_file=__file__, app_env=os.getenv("APP_ENV", "prod"))

DATABASE_URL = env_str(
    "DATABASE_URL",
    "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
)


async def add_key(sensor_id: str, scopes: list[str]) -> None:
    raw_key = secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            """
            INSERT INTO sensor_keys (key_hash, sensor_id, scopes)
            VALUES ($1, $2, $3)
            """,
            key_hash,
            sensor_id,
            scopes,
        )
    finally:
        await conn.close()

    print(f"Sensor key for '{sensor_id}' provisioned.")
    print("Raw key (save this — it will not be shown again):")
    print(f"\n  {raw_key}\n")
    print(f"Scopes: {scopes}")
    print(f"Send as HTTP header: X-Sensor-Key: {raw_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a sensor API key")
    parser.add_argument("--sensor-id", required=True, help="Sensor identifier")
    parser.add_argument(
        "--scopes",
        default="ingest",
        help="Comma-separated scopes (default: ingest)",
    )
    args = parser.parse_args()
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    asyncio.run(add_key(args.sensor_id, scopes))


if __name__ == "__main__":
    main()

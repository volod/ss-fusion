"""Seed test events for Phase 3A correlator development.

Creates test zones and posts synthetic sensor events to the ingest API.
Use this script when Phase 2 adapters are not yet writing live events.

Usage:
    python -m selfsuvis.scripts.seed_test_events [--api-url http://localhost:8000]
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx


async def seed(api_url: str, api_key: str) -> None:
    headers = {"X-Api-Key": api_key} if api_key else {}

    zones = [
        {"zone_id": "north-gate", "label": "North Gate"},
        {"zone_id": "south-perimeter", "label": "South Perimeter"},
        {"zone_id": "east-entrance", "label": "East Entrance"},
    ]

    events = [
        {"modality": "camera", "zone_id": "north-gate", "sensor_id": "cam-01", "confidence": 0.85},
        {"modality": "audio", "zone_id": "north-gate", "sensor_id": "mic-01", "confidence": 0.72},
        {"modality": "rf", "zone_id": "north-gate", "sensor_id": "rf-01", "confidence": 0.90},
        {
            "modality": "camera",
            "zone_id": "south-perimeter",
            "sensor_id": "cam-02",
            "confidence": 0.60,
        },
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Create zones
        for zone in zones:
            resp = await client.post(
                f"{api_url}/api/v1/zones",
                json=zone,
                headers=headers,
            )
            if resp.status_code == 201:
                print(f"  Created zone: {zone['zone_id']}")
            elif resp.status_code == 409:
                print(f"  Zone already exists: {zone['zone_id']}")
            else:
                print(f"  Zone {zone['zone_id']}: {resp.status_code}")

        # Post events
        ts = datetime.now(timezone.utc).isoformat()
        for evt in events:
            resp = await client.post(
                f"{api_url}/api/v1/events/{evt['modality']}",
                json={
                    "ts": ts,
                    "zone_id": evt["zone_id"],
                    "sensor_id": evt["sensor_id"],
                    "confidence": evt["confidence"],
                },
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                print(f"  Event {evt['modality']}@{evt['zone_id']}: {data['event_id']}")
            else:
                print(f"  Event {evt['modality']}@{evt['zone_id']}: {resp.status_code} {resp.text}")

    print(f"\nSeeded {len(zones)} zones and {len(events)} events.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed test events for correlator development")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    api_key = os.getenv("API_KEY", "")
    asyncio.run(seed(args.api_url, api_key))


if __name__ == "__main__":
    main()

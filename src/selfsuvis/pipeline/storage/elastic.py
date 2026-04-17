import json
from typing import Any, Dict, Iterable, List

import requests

from selfsuvis.pipeline.core.logging import get_logger


def _mapping() -> Dict[str, Any]:
    return {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "video_name": {"type": "keyword"},
                "frame_index": {"type": "integer"},
                "t_sec": {"type": "float"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "frame_path": {"type": "keyword"},
                "description": {"type": "text"},
                "segments": {
                    "type": "nested",
                    "properties": {
                        "segment_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "bbox": {"type": "integer"},
                        "mean_color": {"type": "integer"},
                        "area": {"type": "integer"},
                    },
                },
                "entities": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "bbox": {"type": "integer"},
                        "dominant_color": {"type": "keyword"},
                    },
                },
                "tracks": {
                    "type": "nested",
                    "properties": {
                        "track_id": {"type": "integer"},
                        "segment_id": {"type": "keyword"},
                        "label": {"type": "keyword"},
                        "bbox": {"type": "integer"},
                    },
                },
                "warnings": {"type": "keyword"},
                "ontology_entities": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "keyword"},
                        "count": {"type": "integer"},
                        "first_seen": {"type": "integer"},
                        "last_seen": {"type": "integer"},
                        "dominant_color": {"type": "keyword"},
                    },
                },
                "metadata": {
                    "type": "object",
                    "dynamic": "strict",
                    "properties": {
                        "created_at": {"type": "date"},
                        "output_dir": {"type": "keyword"},
                        "pipeline": {"type": "keyword"},
                        "model_type": {"type": "keyword"},
                        "sampling": {
                            "type": "object",
                            "dynamic": "strict",
                            "properties": {
                                "mode": {"type": "keyword"},
                                "interval_sec": {"type": "float"},
                                "min_interval_sec": {"type": "float"},
                                "max_gap_sec": {"type": "float"},
                                "diff_threshold": {"type": "float"},
                                "probe_fps": {"type": "float"},
                            },
                        },
                        "source": {
                            "type": "object",
                            "dynamic": "strict",
                            "properties": {
                                "video_path": {"type": "keyword"},
                                "stream_source": {"type": "keyword"},
                            },
                        },
                    },
                },
            },
        }
    }


def ensure_index(es_url: str, index: str) -> None:
    logger = get_logger(__name__)
    resp = requests.head(f"{es_url}/{index}", timeout=10)
    if resp.status_code == 200:
        return
    if resp.status_code not in (404, 400):
        raise RuntimeError(f"unexpected status checking index: {resp.status_code} {resp.text}")
    payload = _mapping()
    resp = requests.put(f"{es_url}/{index}", json=payload, timeout=10)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"failed to create index: {resp.status_code} {resp.text}")
    logger.info("Created index=%s", index)


def _bulk_lines(index: str, records: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for rec in records:
        lines.append(json.dumps({"index": {"_index": index}}, ensure_ascii=True))
        lines.append(json.dumps(rec, ensure_ascii=True))
    return "\n".join(lines) + "\n"


def bulk_index_jsonl(es_url: str, index: str, jsonl_path: str, batch_size: int = 500) -> None:
    logger = get_logger(__name__)
    ensure_index(es_url, index)

    batch: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                _post_bulk(es_url, index, batch)
                logger.info("Indexed batch size=%s", len(batch))
                batch = []
        if batch:
            _post_bulk(es_url, index, batch)
            logger.info("Indexed batch size=%s", len(batch))


def _post_bulk(es_url: str, index: str, records: List[Dict[str, Any]]) -> None:
    body = _bulk_lines(index, records)
    headers = {"Content-Type": "application/x-ndjson"}
    resp = requests.post(f"{es_url}/_bulk", data=body, headers=headers, timeout=30)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"bulk indexing failed: {resp.status_code} {resp.text}")
    payload = resp.json()
    if payload.get("errors"):
        raise RuntimeError(f"bulk indexing errors: {payload}")

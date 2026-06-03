"""Embedding and search steps for the local full-analysis pipeline."""

import time

# _log_vram_snapshot lives in steps_caption; import lazily inside functions to
# avoid circular imports at module level.
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.media import extract_frames

from ..common import _log


def step_extract_frames(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    fps: float,
) -> dict[str, Any]:
    """Step 01: extract frames via ffmpeg, write metadata JSON."""
    import json

    _log.info("Extracting frames from %s at %.1f fps …", video_path.name, fps)
    t0 = time.time()
    frame_list = extract_frames(str(video_path), video_id)
    elapsed = time.time() - t0
    from ..caption import _log_vram_snapshot as _lvs

    _lvs("after frame extraction")
    meta = {
        "video": str(video_path),
        "video_id": video_id,
        "fps": fps,
        "frame_count": len(frame_list),
        "duration_sec": frame_list[-1][1] if frame_list else 0.0,
        "frames": [{"path": p, "t_sec": t} for p, t in frame_list],
        "extracted_at": datetime.now().isoformat(),
    }
    meta_path = video_dir / "frames_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log.info("  [ok] %d frames extracted in %.1fs → %s", len(frame_list), elapsed, meta_path)
    return {"frame_list": frame_list, "elapsed_sec": elapsed, "meta": meta}


def _embed_and_flush(
    batch_pil: list[Image.Image],
    batch_meta: list[tuple[str, float]],
    video_id: str,
    clip_model: OpenCLIPEmbedder,
    dino_model: Any,
    store: Any,
    is_qdrant: bool,
) -> int:
    if not batch_pil:
        return 0
    clip_embeds = clip_model.encode_images(batch_pil)
    dino_embeds = dino_model.encode_images(batch_pil) if dino_model else None
    if is_qdrant:
        from selfsuvis.pipeline.core.optional_deps import require_qdrant_models

        from selfsuvis.pipeline.core.utils import stable_point_id
        qmodels = require_qdrant_models()

        points = []
        for i, (fp, t_sec) in enumerate(batch_meta):
            vectors: dict[str, Any] = {"clip": clip_embeds[i].tolist()}
            if dino_embeds is not None:
                vectors["dino"] = dino_embeds[i].tolist()
            points.append(
                qmodels.PointStruct(
                    id=stable_point_id(video_id, fp),
                    vector=vectors,
                    payload={"frame_path": fp, "t_sec": t_sec, "video_id": video_id},
                )
            )
        store.upsert_points(points)
    else:
        for i, (fp, t_sec) in enumerate(batch_meta):
            vec = dino_embeds[i] if dino_embeds is not None else clip_embeds[i]
            store.add(vec, {"frame_path": fp, "t_sec": t_sec, "video_id": video_id})
    return len(batch_pil)


def step_index_to_store(
    video_path: Path,
    video_id: str,
    store: Any,
    is_qdrant: bool,
    models: dict[str, Any],
    frame_list: list[tuple[str, float]],
) -> dict[str, Any]:
    """Step 02: embed frames and upsert into Qdrant or InMemoryStore."""
    t0 = time.time()
    dest = "Qdrant" if is_qdrant else "in-memory store"
    _log.info("Embedding %d frames into %s …", len(frame_list), dest)
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    batch_pil: list[Image.Image] = []
    batch_meta: list[tuple[str, float]] = []
    indexed = 0
    for fp, t_sec in frame_list:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        batch_pil.append(img)
        batch_meta.append((fp, t_sec))
        if len(batch_pil) >= 32:
            indexed += _embed_and_flush(
                batch_pil, batch_meta, video_id, clip_model, dino_model, store, is_qdrant
            )
            batch_pil, batch_meta = [], []
    indexed += _embed_and_flush(
        batch_pil, batch_meta, video_id, clip_model, dino_model, store, is_qdrant
    )
    elapsed = time.time() - t0
    _log.info("  [ok] %d frames indexed into %s in %.1fs", indexed, dest, elapsed)
    return {"indexed": indexed, "elapsed_sec": elapsed}


def _pick_query_frame(frame_list: list[tuple[str, float]]) -> tuple[str, float]:
    return frame_list[len(frame_list) // 2]


def _embed_query(frame_path: str, models: dict[str, Any], use_dino: bool = True) -> np.ndarray:
    img = Image.open(frame_path).convert("RGB")
    if use_dino and models.get("dino"):
        return models["dino"].encode_images([img])[0]
    return models["clip"].encode_images([img])[0]


def _search(
    query_vec: np.ndarray,
    store: Any,
    is_qdrant: bool,
    top_k: int,
    video_id: str,
    vector_name: str = "clip",
    exclude_frame_path: str = "",
    exclude_t_sec: float | None = None,
    min_time_gap_sec: float = 1.0,
) -> list[dict[str, Any]]:
    limit = top_k
    if exclude_frame_path:
        limit += 1
    if exclude_t_sec is not None and min_time_gap_sec > 0.0:
        # Temporal filtering happens after vector retrieval, so fetch a wider
        # candidate pool to avoid empty rankings on short near-duplicate clips.
        limit = max(limit + 8, top_k * 4)
    if is_qdrant:
        from selfsuvis.pipeline.core.optional_deps import require_qdrant_models

        qmodels = require_qdrant_models()
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="video_id",
                    match=qmodels.MatchValue(value=video_id),
                )
            ]
        )
        raw = store.search(vector_name, query_vec, limit=limit, payload_filter=filt)
        results = [{"score": p.score, "payload": p.payload} for p in raw]
    else:
        results = store.search(query_vec, limit=limit)
    if exclude_frame_path:
        results = [
            r for r in results if (r.get("payload", r).get("frame_path", "")) != exclude_frame_path
        ]
    if exclude_t_sec is not None and min_time_gap_sec > 0.0:
        results = [
            r
            for r in results
            if abs(float((r.get("payload", r).get("t_sec", -1.0)) or -1.0) - exclude_t_sec)
            >= min_time_gap_sec
        ]
    return results[:top_k]


def _run_search_test(
    *,
    store: Any,
    is_qdrant: bool,
    models: dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
    report_name: str,
    model_label: str,
    query_frame: str,
    query_t_sec: float,
) -> dict[str, Any]:
    from ..report import write_search_md

    out_md = video_dir / report_name
    use_dino = models.get("dino") is not None
    t0 = time.time()
    query_vec = _embed_query(query_frame, models, use_dino=use_dino)
    results = _search(
        query_vec,
        store,
        is_qdrant,
        top_k,
        video_id,
        vector_name="dino" if use_dino else "clip",
        exclude_frame_path=query_frame,
        exclude_t_sec=query_t_sec,
    )
    elapsed = time.time() - t0
    write_search_md(out_md, video_name, model_label, query_frame, results, query_t_sec)
    return {"results": results, "elapsed_sec": elapsed}


def step_base_model_search_test(
    frame_list: list[tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    """Step 14: embed query with base model, search, write base_search.md."""
    qfp, qt = _pick_query_frame(frame_list)
    _log.info("Query frame: %s (t=%.2fs)", Path(qfp).name, qt)
    label = (
        "Base DINOv3 (pretrained)" if models.get("dino") is not None else "Base CLIP (pretrained)"
    )
    result = _run_search_test(
        store=store,
        is_qdrant=is_qdrant,
        models=models,
        video_id=video_id,
        video_name=video_name,
        video_dir=video_dir,
        top_k=top_k,
        report_name="base_search.md",
        model_label=label,
        query_frame=qfp,
        query_t_sec=qt,
    )
    results = result["results"]
    elapsed = float(result["elapsed_sec"])
    _log.info(
        "  [ok] Search in %.2fs → top score %.4f", elapsed, results[0]["score"] if results else 0
    )
    return {"results": results, "query_frame": qfp, "query_t_sec": qt}


def step_finetuned_model_search_test(
    frame_list: list[tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: dict[str, Any],
    query_frame: str,
    query_t_sec: float,
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    """Step 19: search with fine-tuned DINO, write finetuned_search.md."""
    result = _run_search_test(
        store=store,
        is_qdrant=is_qdrant,
        models=models,
        video_id=video_id,
        video_name=video_name,
        video_dir=video_dir,
        top_k=top_k,
        report_name="finetuned_search.md",
        model_label="Fine-tuned DINOv3 (SSL adapted)",
        query_frame=query_frame,
        query_t_sec=query_t_sec,
    )
    results = result["results"]
    ft_infer_ms = float(result["elapsed_sec"]) * 1000 / max(len(frame_list), 1)
    return {"results": results, "infer_ms": ft_infer_ms}

"""Steps 05-08 — ASR transcription, OCR, Depth estimation, Object detection."""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..caption_helpers.ocr import _fallback_ocr_frame_sample, _select_ocr_candidate_frames
from ..caption_helpers.vram import _log_vram_snapshot
from ..common import _run_batched_frame_inference, write_markdown_artifact

_log = get_logger("pipeline.local.caption")


def step_asr_transcription(
    video_path: Path,
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 05: extract audio, run Whisper ASR."""
    from datetime import datetime

    from ..common import _RUNNER_LABEL

    out_md = video_dir / "asr_subtitles.md"
    result: dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    try:
        from selfsuvis.pipeline.media.audio import extract_audio, map_subtitles_to_frames
        from selfsuvis.pipeline.vision.asr import ASRModel
    except ImportError as exc:
        _log.warning("  ASR unavailable (%s) — skipping", exc)
        return result
    asr = ASRModel()
    _log_vram_snapshot("before ASR model use")
    if not asr.is_enabled():
        _log.info("  ASR disabled (ASR_ENABLED=false) — skipping")
        return result
    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Extracting audio from %s …", video_path.name)
    wav_path = extract_audio(str(video_path), str(audio_dir))
    if not wav_path:
        _log.warning("  No audio stream found in %s — ASR skipped", video_path.name)
        return result
    _log.info("Transcribing audio with %s …", asr.model_id)
    t0 = time.time()
    segments = asr.transcribe(wav_path)
    elapsed = time.time() - t0
    if not segments:
        _log.warning("  ASR returned no segments for %s", video_path.name)
        return result
    frame_timestamps = [t for _, t in frame_list]
    subtitle_map = map_subtitles_to_frames(
        segments, frame_timestamps, window_sec=settings.ASR_SUBTITLE_WINDOW_SEC
    )
    covered = sum(1 for t in frame_timestamps if t in subtitle_map)
    _log.info(
        "  [ok] ASR: %d segments → %d/%d frames have subtitles (%.1fs, model=%s)",
        len(segments),
        covered,
        len(frame_list),
        elapsed,
        asr.model_id,
    )
    _log_vram_snapshot("after ASR model use")
    lines = [
        f"# ASR Subtitles — {video_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{asr.model_id}`",
        f"Segments: {len(segments)}  |  Frames with subtitles: {covered}/{len(frame_list)}",
        f"Elapsed: {elapsed:.1f}s",
        "",
        "## Subtitle Segments",
        "",
        "| Start (s) | End (s) | Text |",
        "|-----------|---------|------|",
    ]
    for seg in segments:
        ts = seg.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        start = float(ts[0]) if len(ts) > 0 and ts[0] is not None else 0.0
        end = float(ts[1]) if len(ts) > 1 and ts[1] is not None else start
        text = seg.get("text", "").strip().replace("|", "\\|")
        lines.append(f"| {start:.2f} | {end:.2f} | {text} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · ASR step 05*"]
    write_markdown_artifact(out_md, lines)
    result.update(
        {
            "skipped": False,
            "subtitle_map": subtitle_map,
            "segments": segments,
            "elapsed_sec": elapsed,
            "covered_frames": covered,
        }
    )
    return result


def step_ocr_extraction(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    caption_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Step 06: visible text extraction per frame."""
    result: dict[str, Any] = {"skipped": True, "ocr_results": []}
    try:
        from selfsuvis.pipeline.vision.ocr import OCRModel
    except ImportError as exc:
        _log.warning("  OCR unavailable (%s) — skipping", exc)
        return result
    ocr = OCRModel()
    _log_vram_snapshot("before OCR model use")
    if not ocr.is_enabled():
        _log.info("  OCR disabled (OCR_ENABLED=false) — skipping")
        return result
    _log.info("Running OCR on %d frames (model=%s) …", len(frame_list), ocr.model_id)
    t0 = time.time()
    threshold = settings.OCR_MIN_CAPTION_CONFIDENCE
    max_ocr = int(settings.OCR_MAX_FRAMES)
    selected_frame_list, skipped_by_caption, ranking = _select_ocr_candidate_frames(
        frame_list=frame_list,
        caption_results=caption_results,
        ocr_model_id=ocr.model_id,
        threshold=threshold,
        max_ocr=max_ocr,
    )
    if ranking:
        top_score = max(float(item.get("score", 0.0) or 0.0) for item in ranking)
        _log.info(
            "  OCR ranked selection: %d/%d frames kept (top score %.2f, OCR_MAX_FRAMES=%d)",
            len(selected_frame_list),
            len(frame_list),
            top_score,
            max_ocr,
        )
    elif threshold > 0.0:
        if len(selected_frame_list) < len(frame_list):
            _log.info(
                "  OCR caption prescreen unavailable (ran concurrently) — "
                "capped to %d/%d evenly spaced frames (OCR_MAX_FRAMES=%d)",
                len(selected_frame_list),
                len(frame_list),
                max_ocr,
            )
        else:
            _log.info(
                "  OCR ranked selection unavailable — using all %d frames",
                len(selected_frame_list),
            )

    if not selected_frame_list and frame_list:
        selected_frame_list = _fallback_ocr_frame_sample(frame_list)
        selected_paths = {fp for fp, _ in selected_frame_list}
        for fp, meta in skipped_by_caption.items():
            if fp in selected_paths:
                meta.pop("ocr_skipped_by_caption", None)
                meta["ocr_prescreen_fallback"] = True
        _log.info(
            "  OCR prescreen fallback: selected %d evenly spaced frames because caption prescreen skipped everything",
            len(selected_frame_list),
        )

    processed_results = _run_batched_frame_inference(
        selected_frame_list,
        batch_size=settings.OCR_BATCH_SIZE,
        batch_fn=lambda _batch, imgs: ocr.extract_text_batch(imgs),
        warning_label="OCR",
        error_result={"ocr_text": "", "ocr_error": True},
    )
    processed_by_frame = {str(r["frame_path"]): r for r in processed_results}
    ocr_results: list[dict[str, Any]] = []
    for fp, t_sec in frame_list:
        if fp in processed_by_frame:
            ocr_results.append(processed_by_frame[fp])
        else:
            ocr_results.append(
                skipped_by_caption.get(
                    fp,
                    {"frame_path": fp, "t_sec": t_sec, "ocr_text": "", "ocr_error": True},
                )
            )
    elapsed = time.time() - t0
    non_empty = sum(1 for r in ocr_results if r.get("ocr_text"))
    _log.info("  [ok] OCR: %d/%d frames have text in %.1fs", non_empty, len(frame_list), elapsed)
    result.update(
        {
            "skipped": False,
            "ocr_results": ocr_results,
            "non_empty": non_empty,
            "elapsed_sec": elapsed,
        }
    )
    ocr.release()
    _log_vram_snapshot("after OCR model use")
    return result


def step_depth_estimation(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 07: depth estimation per frame."""
    result: dict[str, Any] = {"skipped": True, "depth_results": []}
    try:
        from selfsuvis.pipeline.vision.depth import DepthModel
    except ImportError as exc:
        _log.warning("  Depth model unavailable (%s) — skipping", exc)
        return result
    depth_model = DepthModel()
    _log_vram_snapshot("before depth model use")
    if not depth_model.is_enabled():
        _log.info("  Depth disabled (DEPTH_ENABLED=false) — skipping")
        return result
    _log.info(
        "Running depth estimation on %d frames (model=%s) …", len(frame_list), depth_model.model_id
    )
    t0 = time.time()
    depth_results = _run_batched_frame_inference(
        frame_list,
        batch_size=max(1, int(getattr(settings, "DEPTH_BATCH_SIZE", 8) or 8)),
        batch_fn=lambda _batch, imgs: depth_model.estimate_batch(imgs),
        warning_label="Depth",
        error_result={"depth_error": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in depth_results
        if not r.get("depth_error")
        and not r.get("depth_unavailable")
        and not r.get("depth_disabled")
    )
    _log.info("  [ok] Depth: %d/%d frames estimated in %.1fs", ok, len(frame_list), elapsed)
    result.update(
        {"skipped": False, "depth_results": depth_results, "ok_count": ok, "elapsed_sec": elapsed}
    )
    depth_model.release()
    _log_vram_snapshot("after depth model use")
    return result


def step_object_detection(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> dict[str, Any]:
    """Step 08: object detection per frame."""
    result: dict[str, Any] = {"skipped": True, "detection_results": []}
    try:
        from selfsuvis.pipeline.vision.detection import DetectionModel
    except ImportError as exc:
        _log.warning("  Detection model unavailable (%s) — skipping", exc)
        return result
    det_model = DetectionModel()
    _log_vram_snapshot("before detection model use")
    if not det_model.is_enabled():
        _log.info("  Detection disabled (DETECTION_ENABLED=false) — skipping")
        return result
    _log.info(
        "Running object detection on %d frames (model=%s) …", len(frame_list), det_model.model_id
    )
    t0 = time.time()
    det_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: det_model.detect_batch(imgs),
        warning_label="Detection",
        error_result={"detection_error": True},
    )
    elapsed = time.time() - t0
    total_objs = sum(len(r.get("detections", [])) for r in det_results)
    ok = sum(
        1
        for r in det_results
        if not r.get("detection_error")
        and not r.get("detection_unavailable")
        and not r.get("detection_disabled")
    )
    _log.info(
        "  [ok] Detection: %d objects across %d/%d frames in %.1fs",
        total_objs,
        ok,
        len(frame_list),
        elapsed,
    )
    result.update(
        {
            "skipped": False,
            "detection_results": det_results,
            "total_objects": total_objs,
            "ok_count": ok,
            "elapsed_sec": elapsed,
        }
    )
    det_model.release()
    _log_vram_snapshot("after detection model use")
    return result

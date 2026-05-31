"""CLI runner for the agentic video processing pipeline (file and stream modes)."""

import argparse
import os
from typing import Any

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.core.optional_deps import require_cv2
from selfsuvis.pipeline.media.frames import (
    FrameRecord,
    extract_frames_adaptive,
    extract_frames_fixed,
    extract_stream_frames,
)
from selfsuvis.pipeline.storage.elastic import bulk_index_jsonl


def _run_metadata(
    args: argparse.Namespace,
    mode: str,
    video_path: str | None = None,
    stream_source: str | None = None,
) -> dict[str, Any]:
    """Build run_metadata dict for file or stream mode."""
    if mode == "stream":
        return {
            "pipeline": "v1",
            "source": {"video_path": None, "stream_source": stream_source or ""},
            "sampling": {
                "mode": "stream-adaptive",
                "interval_sec": None,
                "min_interval_sec": args.min_interval,
                "max_gap_sec": args.max_gap,
                "diff_threshold": args.diff_threshold,
                "probe_fps": args.probe_fps,
            },
        }
    return {
        "pipeline": "v1",
        "source": {"video_path": video_path, "stream_source": stream_source},
        "sampling": {
            "mode": "adaptive" if args.adaptive else "fixed",
            "interval_sec": args.interval if not args.adaptive else None,
            "min_interval_sec": args.min_interval if args.adaptive else None,
            "max_gap_sec": args.max_gap if args.adaptive else None,
            "diff_threshold": args.diff_threshold if args.adaptive else None,
            "probe_fps": args.probe_fps if args.adaptive else None,
        },
    }


def _jsonl_path(result: dict[str, Any] | None, out_dir: str, video_name: str) -> str:
    """Return jsonl path from process_frames result or default path."""
    if result and "jsonl_path" in result:
        return result["jsonl_path"]
    return os.path.join(out_dir, f"{video_name}.jsonl")


def _list_videos(path: str) -> list[str]:
    """List video files under path (recursive)."""
    videos: list[str] = []
    for root, _, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in settings.VIDEO_EXTS:
                videos.append(os.path.join(root, name))
    return videos


def _safe_stem(path: str) -> str:
    """Return filename without extension."""
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _parse_steps(raw: str | None) -> list[str]:
    """Parse comma-separated steps; default extract,describe,index."""
    if not raw:
        return ["extract", "describe", "index"]
    steps = [s.strip() for s in raw.split(",") if s.strip()]
    valid = {"extract", "describe", "index"}
    for s in steps:
        if s not in valid:
            raise ValueError(f"Unknown step: {s}")
    return steps


def _load_existing_frames(out_dir: str) -> list[FrameRecord]:
    """Load FrameRecords from existing frame_*.png files in out_dir."""
    cv2 = require_cv2()
    frames: list[FrameRecord] = []
    if not os.path.isdir(out_dir):
        return frames
    for name in sorted(os.listdir(out_dir)):
        if not name.startswith("frame_") or not name.endswith(".png"):
            continue
        parts = name.split("_")
        if len(parts) < 3:
            continue
        ms_part = parts[2].replace("ms.png", "")
        try:
            t_sec = int(ms_part) / 1000.0
        except ValueError:
            t_sec = 0.0
        path = os.path.join(out_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        index = len(frames)
        frames.append(FrameRecord(path=path, t_sec=t_sec, index=index, width=w, height=h))
    return frames


def run_file_mode(args: argparse.Namespace) -> None:
    """Run pipeline in file mode (single file or directory)."""
    logger = get_logger(__name__)
    input_path = args.input or args.dir
    if not input_path:
        input_path = "video_test"
    steps = _parse_steps(args.steps)

    if os.path.isdir(input_path):
        videos = _list_videos(input_path)
    else:
        videos = [input_path]

    if not videos:
        logger.warning("No videos found in %s", input_path)
        return

    for video_path in videos:
        video_name = _safe_stem(video_path)
        out_dir = os.path.join(args.output_dir, video_name)
        run_metadata = _run_metadata(args, "file", video_path=os.path.abspath(video_path))
        frames: list[FrameRecord] = []
        if "extract" in steps:
            if args.adaptive:
                frames = extract_frames_adaptive(
                    video_path,
                    out_dir,
                    min_interval_sec=args.min_interval,
                    max_gap_sec=args.max_gap,
                    diff_threshold=args.diff_threshold,
                    probe_fps=args.probe_fps,
                )
            else:
                frames = extract_frames_fixed(
                    video_path,
                    out_dir,
                    interval_sec=args.interval,
                )
        elif "describe" in steps:
            frames = _load_existing_frames(out_dir)

        result = None
        if "describe" in steps:
            from ssv_vdp.agentic.processor import process_frames

            result = process_frames(
                video_name,
                frames,
                out_dir,
                run_metadata=run_metadata,
                model_type=args.model_type,
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
                labels_file=args.labels_file,
                verbose=args.verbose,
            )
        if "index" in steps and args.es_url:
            index_name = args.es_index or f"{video_name}_frames"
            bulk_index_jsonl(
                args.es_url,
                index_name,
                _jsonl_path(result, out_dir, video_name),
            )


def run_stream_mode(args: argparse.Namespace) -> None:
    """Run pipeline in stream mode."""
    logger = get_logger(__name__)
    if not args.source:
        raise ValueError("--source is required for stream mode")
    steps = _parse_steps(args.steps)
    if "extract" not in steps:
        raise ValueError("stream mode requires extract step")

    video_name = args.stream_name or "stream"
    out_dir = os.path.join(args.output_dir, video_name)
    run_metadata = _run_metadata(args, "stream", stream_source=str(args.source))
    frames = extract_stream_frames(
        args.source,
        out_dir,
        min_interval_sec=args.min_interval,
        max_gap_sec=args.max_gap,
        diff_threshold=args.diff_threshold,
        probe_fps=args.probe_fps,
        max_frames=args.max_frames,
    )
    result = None
    if "describe" in steps:
        from ssv_vdp.agentic.processor import process_frames

        result = process_frames(
            video_name,
            frames,
            out_dir,
            run_metadata=run_metadata,
            model_type=args.model_type,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            labels_file=args.labels_file,
            verbose=args.verbose,
        )
    if "index" in steps and args.es_url:
        index_name = args.es_index or f"{video_name}_frames"
        bulk_index_jsonl(
            args.es_url,
            index_name,
            _jsonl_path(result, out_dir, video_name),
        )
    logger.info("Stream mode completed frames=%s", len(frames))

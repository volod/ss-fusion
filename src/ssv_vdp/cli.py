"""Entry point for the ssv_vdp CLI (ssv command).

Modes:
  local   — run the full local analysis / training orchestration
  file    — process a video file or directory (default)
  stream  — process a live RTSP/device stream
  analyse — generate charts / report for an existing local run
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress transformers lazy-loader __warningregistry__ noise before any import
# triggers the transformers package.  Must be set before the first transformers
# import; deferred imports in main() mean this module-level assignment is early enough.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated",
    category=FutureWarning,
)


def _validate_local_inputs(args) -> None:
    # args.videos_dir is resolved by apply_local_env before this is called:
    # - None (not supplied)  → DATA_DIR/videos or .data/videos
    # - explicit value       → used as-is, never rewritten
    videos_dir = Path(args.videos_dir)
    if videos_dir.is_dir():
        return
    data_dir = os.environ.get("DATA_DIR", ".data")
    default_suggestion = os.path.join(data_dir, "videos")
    raise SystemExit(
        f"Videos directory does not exist: {videos_dir}\n"
        f"Use the local data directory:  --videos-dir {default_suggestion}\n"
        f"Create it with:  mkdir -p {default_suggestion}"
    )


def _dispatch(args) -> None:
    if args.mode == "local":
        from ssv_vdp.local_env import apply_local_env  # noqa: PLC0415

        apply_local_env(args)
        _validate_local_inputs(args)
        from selfsuvis.pipeline.core import log_preflight, run_local_preflight  # noqa: PLC0415

        report = run_local_preflight(args)
        log_preflight(report)
        if report.errors:
            print(
                "\nRun the commands shown above to cache the missing models,"
                " then re-run the pipeline.\n"
                "Full option reference:  ssv-models --help",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        from ssv_vdp import run_local  # noqa: PLC0415

        run_local(args)
    elif args.mode == "file":
        from ssv_vdp.commands.runner import run_file_mode  # noqa: PLC0415

        run_file_mode(args)
    elif args.mode == "analyse":
        from ssv_vdp.scripts.analyse_local_run import run  # noqa: PLC0415

        run(args)
    else:
        from ssv_vdp.commands.runner import run_stream_mode  # noqa: PLC0415

        run_stream_mode(args)


def main() -> None:
    from ssv_vdp.commands.parser import build_parser

    parser = build_parser()
    args = parser.parse_args()
    try:
        _dispatch(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

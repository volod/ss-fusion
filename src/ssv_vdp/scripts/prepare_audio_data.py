"""Download and split the drone audio detection dataset into train/val/test splits.

Downloads geronimobasso/drone-audio-detection-samples from HuggingFace and
organises it as:

    <data_dir>/
        train/
            drone/    *.wav
            no_drone/ *.wav
        val/
            drone/    *.wav
            no_drone/ *.wav
        test/
            drone/    *.wav
            no_drone/ *.wav

Re-running is safe — already-present WAV files are not re-written.
Use this as a first-class setup step analogous to prepare_models.py.

Usage
-----
    # Default path (.data/drone-audio-data):
    python -m selfsuvis.scripts.prepare_audio_data

    # Custom path:
    python -m selfsuvis.scripts.prepare_audio_data --data-dir /mnt/data/audio

    # Verify an existing split (no network, no writes):
    python -m selfsuvis.scripts.prepare_audio_data --verify

    # Limit samples per class (useful for quick smoke-tests):
    python -m selfsuvis.scripts.prepare_audio_data --max-per-class 100

Environment
-----------
    DRONE_AUDIO_DATA_DIR    Override default data directory
    HF_TOKEN                HuggingFace token (not required for this public dataset)
"""

import argparse
import os
from pathlib import Path


def _default_data_dir() -> str:
    try:
        from selfsuvis.pipeline.core.config import settings

        return settings.DRONE_AUDIO_DATA_DIR
    except Exception:
        return os.path.join(".data", "drone-audio-data")


def _verify(data_dir: Path) -> bool:
    """Return True if all three splits exist and are non-empty."""
    ok = True
    for split in ("train", "val", "test"):
        for label in ("drone", "no_drone"):
            d = data_dir / split / label
            if not d.is_dir():
                print(f"  MISSING  {d}")
                ok = False
                continue
            files = list(d.glob("*.wav"))
            if files:
                print(f"  OK  {d}  ({len(files)} files)")
            else:
                print(f"  EMPTY  {d}")
                ok = False
    return ok


def run(
    data_dir: str,
    target_sr: int = 22050,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    max_per_class: int = 0,
) -> None:
    """Download the HF dataset and write WAVs to *data_dir* split directories."""
    from ssv_vdp.scripts.split_drone_audio_data import run as split_run

    split_run(
        data_dir=data_dir,
        target_sr=target_sr,
        val_frac=val_frac,
        test_frac=test_frac,
        max_per_class=max_per_class,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and split the drone audio detection dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default=_default_data_dir(),
        help="Output directory (default: .data/drone-audio-data or DRONE_AUDIO_DATA_DIR env var)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sample rate Hz (default: 22050)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.10,
        help="Fraction for test split (default: 0.10)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Max samples per class (0 = unlimited)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing split without downloading",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.verify:
        print(f"Verifying drone audio dataset at: {data_dir}")
        ok = _verify(data_dir)
        if ok:
            print("Dataset is complete.")
        else:
            print("Dataset is incomplete or missing. Re-run without --verify to download.")
            sys.exit(1)
        return

    if data_dir.exists():
        existing = list(data_dir.glob("**/*.wav"))
        if existing:
            print(f"Found {len(existing)} WAV files in {data_dir} — verifying completeness ...")
            if _verify(data_dir):
                print(
                    "Dataset already complete. Use --verify to re-check, or delete the directory to re-download."
                )
                return

    print(f"Preparing drone audio dataset → {data_dir}")
    run(
        data_dir=str(data_dir),
        target_sr=args.sr,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_per_class=args.max_per_class,
    )

    print("\nVerifying result ...")
    ok = _verify(data_dir)
    if ok:
        print(f"\nDrone audio dataset ready at: {data_dir}")
        print("Train the DroneAudioCNN with:")
        print("  selfsuvis --mode local --videos-dir .data/videos --drone-audio")
    else:
        print("\nWarning: some splits appear to be incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()

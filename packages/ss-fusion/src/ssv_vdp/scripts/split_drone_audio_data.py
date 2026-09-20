"""Split the drone-audio-detection-samples HuggingFace dataset into
train / val / test subdirectories under DRONE_AUDIO_DATA_DIR.

Usage:
    python -m selfsuvis.scripts.split_drone_audio_data [options]
    scripts/split_drone_audio_data.sh [options]

Output structure:
    <data-dir>/
      train/
        drone/    *.wav
        no_drone/ *.wav
      val/
        drone/    *.wav
        no_drone/ *.wav
      test/
        drone/    *.wav
        no_drone/ *.wav

The script is idempotent: existing WAV files are not overwritten.
"""

import argparse
import math
import sys
from pathlib import Path

_HF_REPO = "geronimobasso/drone-audio-detection-samples"
_DEFAULT_SR = 22050
_VAL_FRAC = 0.15
_TEST_FRAC = 0.10


def _resample_linear(arr, sr_in: int, sr_out: int):
    import numpy as np

    if sr_in == sr_out:
        return arr
    n_out = int(len(arr) * sr_out / sr_in)
    return np.interp(
        np.linspace(0, len(arr) - 1, n_out),
        range(len(arr)),
        arr,
    ).astype(arr.dtype)


def _write_wav(path: Path, arr, sr: int) -> None:
    import numpy as np
    from scipy.io import wavfile

    pcm = (arr * 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


def _label_to_dirname(label: int, label_names: list[str]) -> str:
    if label_names and label < len(label_names):
        name = label_names[label].strip().lower().replace("-", "_").replace(" ", "_")
        negative_tokens = (
            "no_drone",
            "non_drone",
            "not_drone",
            "negative",
            "background",
            "ambient",
            "noise",
            "no_uav",
            "non_uav",
        )
        positive_tokens = ("drone", "uav", "quadcopter", "positive")
        if any(token in name for token in negative_tokens):
            return "no_drone"
        if any(token in name for token in positive_tokens):
            return "drone"
        return "no_drone"
    return "drone" if label == 1 else "no_drone"


def run(
    data_dir: Path, target_sr: int, val_frac: float, test_frac: float, max_per_class: int
) -> None:
    import numpy as np

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: 'datasets' library required. Install with:\n  pip install datasets soundfile\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading {_HF_REPO} …")
    try:
        ds_full = load_dataset(_HF_REPO, cache_dir=str(data_dir / "_hf_cache"))
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Detect label names
    label_names: list[str] = []
    first_split = list(ds_full.keys())[0]
    try:
        feat = ds_full[first_split].features.get("label")
        if hasattr(feat, "names"):
            label_names = feat.names
    except Exception:
        pass
    print(f"Splits found: {list(ds_full.keys())}")
    print(f"Label names: {label_names or '(int: 0=no_drone, 1=drone assumed)'}")

    # Merge all splits into one pool (we do our own train/val/test split)
    all_samples: list[dict] = []
    for split_name, split_ds in ds_full.items():
        for sample in split_ds:
            all_samples.append(dict(sample))
    print(f"Total samples: {len(all_samples)}")

    # Group by class
    by_class: dict[str, list[dict]] = {}
    for s in all_samples:
        key = _label_to_dirname(int(s.get("label", 0)), label_names)
        by_class.setdefault(key, []).append(s)

    for cls, samples in by_class.items():
        print(f"  {cls}: {len(samples)} samples")

    # Split each class independently (preserves balance)
    import random

    random.seed(42)

    written_total = 0
    for cls, samples in by_class.items():
        random.shuffle(samples)
        if max_per_class > 0:
            samples = samples[:max_per_class]

        n = len(samples)
        n_test = max(1, math.floor(n * test_frac))
        n_val = max(1, math.floor(n * val_frac))
        n_train = n - n_test - n_val

        splits = {
            "train": samples[:n_train],
            "val": samples[n_train : n_train + n_val],
            "test": samples[n_train + n_val :],
        }

        for split_name, split_samples in splits.items():
            out_dir = data_dir / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for i, sample in enumerate(split_samples):
                audio = sample.get("audio")
                if audio is None:
                    continue
                arr = np.array(audio["array"], dtype=np.float32)
                sr_in = int(audio["sampling_rate"])
                arr = _resample_linear(arr, sr_in, target_sr)
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                # Normalise to [-1, 1]
                peak = np.abs(arr).max()
                if peak > 0:
                    arr = arr / peak

                out_path = out_dir / f"{split_name}_{cls}_{i:06d}.wav"
                if out_path.exists():
                    continue
                _write_wav(out_path, arr, target_sr)
                written += 1
                written_total += 1

            print(
                f"  {split_name}/{cls}: {len(split_samples)} samples  ({written} written, {len(split_samples) - written} already exist)"
            )

    print(f"\nDone. {written_total} WAV files written to {data_dir}")
    print("\nDirectory layout:")
    for split in ("train", "val", "test"):
        for cls in ("drone", "no_drone"):
            d = data_dir / split / cls
            n = len(list(d.glob("*.wav"))) if d.exists() else 0
            print(f"  {split}/{cls}/  {n} files")


def main() -> None:
    from selfsuvis.pipeline.core.config import settings

    parser = argparse.ArgumentParser(
        description="Download and split drone-audio-detection-samples dataset"
    )
    parser.add_argument(
        "--data-dir",
        default=settings.DRONE_AUDIO_DATA_DIR,
        help=f"Output directory for WAV files (default: {settings.DRONE_AUDIO_DATA_DIR})",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=_DEFAULT_SR,
        help=f"Target sample rate in Hz (default: {_DEFAULT_SR})",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=_VAL_FRAC,
        help=f"Fraction of data for validation (default: {_VAL_FRAC})",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=_TEST_FRAC,
        help=f"Fraction of data for test (default: {_TEST_FRAC})",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Max samples per class (0 = unlimited)",
    )
    args = parser.parse_args()
    run(Path(args.data_dir), args.sr, args.val_frac, args.test_frac, args.max_per_class)


if __name__ == "__main__":
    main()

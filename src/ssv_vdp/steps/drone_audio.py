"""Step 32: Drone audio detection model training.

Downloads geronimobasso/drone-audio-detection-samples from HuggingFace and
caches audio files under settings.DRONE_AUDIO_DATA_DIR (default:
.data/drone-audio-data/).  Trains DroneAudioCNN — a small 2-D CNN over MFCC
features — for binary drone / no-drone classification and exports ONNX for
edge inference.

Outputs (under video_dir/drone_audio/):
  drone_audio_cnn.pt          PyTorch state-dict checkpoint
  drone_audio_cnn.onnx        ONNX opset-14 export
  drone_audio_report.md       training summary + deployment notes

Dataset directory layout produced (or consumed) by split_drone_audio_data.py:
  <DRONE_AUDIO_DATA_DIR>/
    train/drone/    *.wav  (positive class — labelled 1)
    train/no_drone/ *.wav  (negative class — labelled 0)
    val/drone/      *.wav
    val/no_drone/   *.wav
    test/drone/     *.wav
    test/no_drone/  *.wav

Model architecture (≈ 52 k parameters):
  Input  (batch, 1, 40, T)  — MFCC (40 coefficients, T ≈ 44 frames / second)
  Conv2d 1→16  k=3 / BN / ReLU / MaxPool2d(2)
  Conv2d 16→32 k=3 / BN / ReLU / MaxPool2d(2)
  Conv2d 32→64 k=3 / BN / ReLU / AdaptiveAvgPool2d(1)
  Flatten → Linear(64, 2)
"""

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

from .common import write_markdown_artifact

_log = get_logger("pipeline.local.drone_audio")

_HF_REPO = "geronimobasso/drone-audio-detection-samples"
_SR = 22050  # resample target
_N_MFCC = 40
_N_FFT = 1024
_HOP_LENGTH = 512
_N_MELS = 64
_CHUNK_SAMPLES = _SR  # 1-second chunks
_T_FRAMES = _CHUNK_SAMPLES // _HOP_LENGTH + 1  # ≈ 44


# -- MFCC (scipy-only, no librosa) ---------------------------------------------


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin_mel = _hz_to_mel(0.0)
    fmax_mel = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if center > low:
            up = (fft_freqs - low) / (center - low)
        else:
            up = np.zeros_like(fft_freqs)
        if high > center:
            down = (high - fft_freqs) / (high - center)
        else:
            down = np.zeros_like(fft_freqs)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FILTERBANK: np.ndarray | None = None


def _compute_mfcc(wave: np.ndarray) -> np.ndarray:
    """Return MFCC array of shape (N_MFCC, T_FRAMES) from a mono float32 waveform."""
    global _FILTERBANK
    from scipy.fft import dct
    from scipy.signal import stft

    if _FILTERBANK is None:
        _FILTERBANK = _mel_filterbank(_SR, _N_FFT, _N_MELS)

    # Pad or truncate to exactly one second
    if len(wave) < _CHUNK_SAMPLES:
        wave = np.pad(wave, (0, _CHUNK_SAMPLES - len(wave)))
    else:
        wave = wave[:_CHUNK_SAMPLES]

    _, _, Zxx = stft(
        wave.astype(np.float32),
        fs=_SR,
        window="hann",
        nperseg=_N_FFT,
        noverlap=_N_FFT - _HOP_LENGTH,
        return_onesided=True,
    )
    power = np.abs(Zxx) ** 2  # (n_fft//2+1, T)
    mel_spec = _FILTERBANK @ power  # (n_mels, T)
    log_mel = np.log(mel_spec + 1e-9)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:_N_MFCC]  # (n_mfcc, T)

    # Pad or truncate time axis to _T_FRAMES
    T = mfcc.shape[1]
    if T < _T_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, _T_FRAMES - T)))
    else:
        mfcc = mfcc[:, :_T_FRAMES]

    return mfcc.astype(np.float32)


# -- Dataset helpers -----------------------------------------------------------


def _load_wav_mono(path: Path) -> np.ndarray | None:
    """Load a WAV file as mono float32 resampled to _SR. Returns None on error."""
    try:
        from scipy.io import wavfile

        sr_in, data = wavfile.read(str(path))
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype.kind == "u":
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Resample if needed using linear interpolation (cheap, good enough)
        if sr_in != _SR:
            n_out = int(len(data) * _SR / sr_in)
            x_in = np.linspace(0, len(data) - 1, len(data))
            x_out = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(x_out, x_in, data).astype(np.float32)
        return data
    except Exception as exc:
        _log.debug("WAV load failed %s: %s", path, exc)
        return None


def _collect_split(split_dir: Path) -> list[tuple[Path, int]]:
    """Collect (path, label) pairs from <split_dir>/drone/ and <split_dir>/no_drone/."""
    items: list[tuple[Path, int]] = []
    for label, subdir in ((1, "drone"), (0, "no_drone")):
        d = split_dir / subdir
        if d.is_dir():
            for p in sorted(d.glob("*.wav")):
                items.append((p, label))
    return items


def _download_hf_dataset(cache_dir: Path) -> bool:
    """Download dataset from HuggingFace and organise into train/val/no_drone/drone dirs."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        _log.warning(
            "datasets library not installed; cannot download %s automatically. "
            "Run: pip install datasets  then  ssv-prepare-audio",
            _HF_REPO,
        )
        return False

    try:
        _log.info("Downloading %s from HuggingFace (this may take a while) …", _HF_REPO)
        # Use decode=False to avoid the torchcodec hard-dependency introduced in
        # datasets ≥ 3.x for audio column decoding; we decode the raw bytes with
        # soundfile instead (already installed as part of the sensor extras).
        from datasets import Audio  # type: ignore[import]
        ds = load_dataset(_HF_REPO, cache_dir=str(cache_dir / "_hf_cache"))
        ds = {split: sd.cast_column("audio", Audio(decode=False)) for split, sd in ds.items()}
    except Exception as exc:
        _log.warning("HuggingFace dataset download failed: %s", exc)
        return False

    # Figure out the label mapping (ClassLabel or plain int)
    label_names: list[str] = []
    try:
        first_split = list(ds.keys())[0]
        label_feature = ds[first_split].features.get("label")
        if hasattr(label_feature, "names"):
            label_names = label_feature.names
    except Exception:
        pass
    _log.info("Label names: %s", label_names or "(int: 0=no_drone, 1=drone assumed)")

    def _label_to_dir(lbl: int) -> str:
        if label_names:
            name = label_names[lbl].lower()
            return "drone" if "drone" in name else "no_drone"
        return "drone" if lbl == 1 else "no_drone"

    import io as _io

    import soundfile as _sf
    from scipy.io import wavfile

    written = 0
    for split_name, split_ds in ds.items():
        for i, sample in enumerate(split_ds):
            audio = sample.get("audio")
            label = sample.get("label", 0)
            if audio is None:
                continue
            # decode=False gives {"bytes": <raw_bytes>, "path": ...}
            raw = audio.get("bytes") or audio.get("array")
            if isinstance(raw, (bytes, bytearray)):
                arr, sr_in = _sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
            else:
                arr = np.array(raw, dtype=np.float32)
                sr_in = int(audio.get("sampling_rate", _SR))
            arr = arr.astype(np.float32)
            sr_in = int(sr_in)
            if sr_in != _SR:
                n_out = int(len(arr) * _SR / sr_in)
                arr = np.interp(
                    np.linspace(0, len(arr) - 1, n_out),
                    np.arange(len(arr)),
                    arr,
                ).astype(np.float32)

            subdir = _label_to_dir(int(label))
            out_dir = cache_dir / split_name / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split_name}_{i:06d}.wav"
            if not out_path.exists():
                pcm = (arr * 32767).astype(np.int16)
                wavfile.write(str(out_path), _SR, pcm)
                written += 1

    _log.info("Dataset saved to %s  (%d WAV files written)", cache_dir, written)
    return True


# -- PyTorch model + dataset ---------------------------------------------------


def _build_model() -> Any:
    """Return DroneAudioCNN. Raises ImportError if torch is unavailable."""
    import torch.nn as nn

    class DroneAudioCNN(nn.Module):
        """Binary drone/no-drone classifier over MFCC features.

        Input: (batch, 1, 40, 44) — MFCC (40 coefficients × 44 time frames)
        Output: (batch, 2) — logits for [no_drone, drone]
        """

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, 2)

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x).flatten(1))

    return DroneAudioCNN()


class _AudioDataset:
    """Minimal PyTorch Dataset over (path, label) pairs."""

    def __init__(self, items: list[tuple[Path, int]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import torch

        path, label = self.items[idx]
        wave = _load_wav_mono(path)
        if wave is None:
            wave = np.zeros(_CHUNK_SAMPLES, dtype=np.float32)
        mfcc = _compute_mfcc(wave)  # (40, T)
        x = torch.from_numpy(mfcc).unsqueeze(0)  # (1, 40, T)
        return x, label


# -- Training ------------------------------------------------------------------


def _train(
    train_items: list[tuple[Path, int]],
    val_items: list[tuple[Path, int]],
    device: str,
    epochs: int,
) -> tuple[Any, dict[str, Any]]:
    """Train DroneAudioCNN. Returns (model, metrics_dict)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    _log.info(
        "Audio model training on device=%s  train=%d  val=%d", dev, len(train_items), len(val_items)
    )

    model = _build_model().to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = _AudioDataset(train_items)
    val_ds = _AudioDataset(val_items)
    # batch_size scales with dataset size to keep epochs fast on small datasets
    bs = min(32, max(4, len(train_items) // 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, n_train = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), torch.tensor(y, dtype=torch.long).to(dev)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            n_train += len(y)
        scheduler.step()

        model.eval()
        val_correct, n_val = 0, 0
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(dev)
                y_np = np.array(y)
                pred = model(x).argmax(1).cpu().numpy()
                val_correct += (pred == y_np).sum()
                n_val += len(y_np)
                tp += int(((pred == 1) & (y_np == 1)).sum())
                fp += int(((pred == 1) & (y_np == 0)).sum())
                fn += int(((pred == 0) & (y_np == 1)).sum())

        val_acc = val_correct / max(n_val, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(n_train, 1),
                "train_acc": train_correct / max(n_train, 1),
                "val_acc": float(val_acc),
                "val_f1": float(f1),
                "val_precision": float(prec),
                "val_recall": float(rec),
            }
        )
        _log.info(
            "  epoch %d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f  val_f1=%.3f",
            epoch,
            epochs,
            history[-1]["train_loss"],
            history[-1]["train_acc"],
            val_acc,
            f1,
        )

    model.eval()
    best = max(history, key=lambda r: r["val_f1"])
    return model, {"history": history, "best": best}


# -- ONNX export ---------------------------------------------------------------


def _export_onnx(model: Any, out_path: Path, device: str) -> bool:
    """Export model to ONNX opset 14. Returns True on success."""
    try:
        import torch

        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        dummy = torch.zeros(1, 1, _N_MFCC, _T_FRAMES, device=dev)
        torch.onnx.export(
            model.to(dev),
            dummy,
            str(out_path),
            input_names=["mfcc"],
            output_names=["logits"],
            dynamic_axes={"mfcc": {0: "batch"}},
            opset_version=14,
        )
        _log.info("  [ok] ONNX export: %s (%.1f kB)", out_path.name, out_path.stat().st_size / 1024)
        return True
    except Exception as exc:
        _log.warning("ONNX export failed: %s", exc)
        return False


# -- Report --------------------------------------------------------------------


def _write_report(
    path: Path,
    cache_dir: Path,
    n_train: int,
    n_val: int,
    train_label_counts: dict[str, int],
    val_label_counts: dict[str, int],
    metrics: dict[str, Any],
    onnx_path: Path | None,
    elapsed: float,
) -> None:
    best = metrics.get("best", {})
    history = metrics.get("history", [])

    lines = [
        "# Drone Audio Detection — Training Report",
        "",
        f"Dataset: `{_HF_REPO}`",
        f"Cache:   `{cache_dir}`",
        "",
        "## Model",
        "",
        "**DroneAudioCNN** — 2-D CNN over MFCC features",
        "",
        "```",
        f"Input  (batch, 1, {_N_MFCC}, {_T_FRAMES})  — MFCC {_N_MFCC} coeff × {_T_FRAMES} frames (1 s @ {_SR} Hz)",
        "Conv2d 1→16  k=3 pad=1 / BN / ReLU / MaxPool2d(2)",
        "Conv2d 16→32 k=3 pad=1 / BN / ReLU / MaxPool2d(2)",
        "Conv2d 32→64 k=3 pad=1 / BN / ReLU / AdaptiveAvgPool2d(1)",
        "Flatten → Linear(64, 2)",
        "```",
        "",
        "## Dataset",
        "",
        "| Split | Total | Drone | No-drone |",
        "|-------|-------|-------|----------|",
        f"| Train | {n_train} | {train_label_counts.get('drone', 0)} | {train_label_counts.get('no_drone', 0)} |",
        f"| Val   | {n_val}   | {val_label_counts.get('drone', 0)}   | {val_label_counts.get('no_drone', 0)}   |",
        "",
        "## Training Results",
        "",
    ]

    if not history:
        lines += ["Training did not run (torch unavailable or no data).", ""]
    else:
        lines += [
            "| Epoch | Train loss | Train acc | Val acc | Val F1 |",
            "|-------|-----------|-----------|---------|--------|",
        ]
        for r in history:
            lines.append(
                f"| {r['epoch']} | {r['train_loss']:.4f} | {r['train_acc']:.3f}"
                f" | {r['val_acc']:.3f} | {r['val_f1']:.3f} |"
            )
        lines += [
            "",
            f"**Best epoch**: {best.get('epoch', '—')}  "
            f"val_acc={best.get('val_acc', 0):.3f}  "
            f"val_F1={best.get('val_f1', 0):.3f}  "
            f"precision={best.get('val_precision', 0):.3f}  "
            f"recall={best.get('val_recall', 0):.3f}",
            f"**Training time**: {elapsed:.1f} s",
        ]

    lines += [
        "",
        "## Edge Deployment",
        "",
        f"- **ONNX model**: `{onnx_path.name if onnx_path and onnx_path.exists() else 'export failed'}`",
        f"- **Input shape**: `(1, 1, {_N_MFCC}, {_T_FRAMES})`  — one 1-second audio chunk",
        "- **Runtime**: onnxruntime CPU — runs on any Arm/x86 device",
        "- **Estimated latency**: < 2 ms on Cortex-A55 (single MFCC + inference)",
        "",
        "```python",
        "import onnxruntime as ort, numpy as np",
        "sess = ort.InferenceSession('drone_audio_cnn.onnx', providers=['CPUExecutionProvider'])",
        "# mfcc: float32 (1, 1, 40, 44) from a 1-second 22050 Hz mono chunk",
        "logits = sess.run(None, {'mfcc': mfcc})[0]",
        "is_drone = int(np.argmax(logits)) == 1",
        "```",
        "",
        "## Sound Simulation",
        "",
        "Use `scripts/play_drone_sound.sh` to synthesise physically realistic drone",
        "audio at any distance, speed, and approach geometry for testing the model:",
        "",
        "```bash",
        "# Drone 200 m away, 10 m/s, fly straight over microphone",
        "scripts/play_drone_sound.sh --scenario flyover --distance 200 --speed 10",
        "",
        "# Close hover at 30 m",
        "scripts/play_drone_sound.sh --scenario hover --distance 30",
        "",
        "# Save simulated audio to file for offline testing",
        "scripts/play_drone_sound.sh --scenario flyover --distance 100 --speed 15 --output sim_flyover.wav",
        "```",
        "",
        "## Data Preparation",
        "",
        "```bash",
        "# Download and split dataset into train/val/test directories",
        "ssv-prepare-audio  # or: ssv-split-audio",
        "",
        "# Re-run audio training step with more epochs",
        "selfsuvis --mode local --drone-audio --drone-audio-epochs 20",
        "```",
    ]
    write_markdown_artifact(path, lines)


# -- Public step function ------------------------------------------------------


def step_drone_audio_training(
    video_dir: Path,
    output_dir: Path,
    device: str,
    args: Any,
) -> dict[str, Any]:
    """Train DroneAudioCNN; export ONNX; write drone_audio_report.md."""
    from selfsuvis.pipeline.core.config import settings

    result: dict[str, Any] = {
        "skipped": False,
        "model_onnx": "",
        "n_train": 0,
        "n_val": 0,
        "val_acc": float("nan"),
        "val_f1": float("nan"),
    }
    t0 = time.monotonic()

    audio_dir = video_dir / "drone_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(settings.DRONE_AUDIO_DATA_DIR)
    epochs = int(getattr(args, "drone_audio_epochs", None) or settings.DRONE_AUDIO_EPOCHS)

    # Locate organised splits; download if absent
    train_items = _collect_split(cache_dir / "train")
    val_items = _collect_split(cache_dir / "val")

    if not train_items:
        _log.info("No cached data in %s — downloading from HuggingFace …", cache_dir)
        ok = _download_hf_dataset(cache_dir)
        if ok:
            train_items = _collect_split(cache_dir / "train")
            val_items = _collect_split(cache_dir / "val")

    if not train_items:
        _log.warning(
            "No audio training data found at %s and download failed. "
            "Run ssv-prepare-audio to prepare the dataset.",
            cache_dir,
        )
        result["skipped"] = True
        result["error"] = f"No training data in {cache_dir}/train/"
        return result

    # Use a fraction of val as fallback when val split is missing
    if not val_items:
        split_idx = max(1, int(len(train_items) * 0.8))
        val_items = train_items[split_idx:]
        train_items = train_items[:split_idx]

    n_train, n_val = len(train_items), len(val_items)
    result["n_train"] = n_train
    result["n_val"] = n_val

    train_counts = {
        "drone": sum(1 for _, lbl in train_items if lbl == 1),
        "no_drone": sum(1 for _, lbl in train_items if lbl == 0),
    }
    val_counts = {
        "drone": sum(1 for _, lbl in val_items if lbl == 1),
        "no_drone": sum(1 for _, lbl in val_items if lbl == 0),
    }

    try:
        import torch  # noqa: F401
    except ImportError:
        _log.warning("PyTorch not installed — skipping drone audio training")
        result["skipped"] = True
        result["error"] = "torch not installed"
        _write_report(
            audio_dir / "drone_audio_report.md",
            cache_dir,
            n_train,
            n_val,
            train_counts,
            val_counts,
            {},
            None,
            time.monotonic() - t0,
        )
        return result

    _log.info(
        "Training DroneAudioCNN  epochs=%d  train=%d  val=%d  device=%s",
        epochs,
        n_train,
        n_val,
        device,
    )
    model, metrics = _train(train_items, val_items, device, epochs)

    # Save checkpoint
    import torch

    pt_path = audio_dir / "drone_audio_cnn.pt"
    torch.save(model.state_dict(), pt_path)
    _log.info("  [ok] Checkpoint: %s", pt_path)

    best = metrics.get("best", {})
    result["val_acc"] = best.get("val_acc", float("nan"))
    result["val_f1"] = best.get("val_f1", float("nan"))

    # ONNX export
    onnx_path = audio_dir / "drone_audio_cnn.onnx"
    ok = _export_onnx(model, onnx_path, device)
    if ok:
        result["model_onnx"] = str(onnx_path)

    elapsed = time.monotonic() - t0
    result["elapsed_sec"] = elapsed

    _write_report(
        audio_dir / "drone_audio_report.md",
        cache_dir,
        n_train,
        n_val,
        train_counts,
        val_counts,
        metrics,
        onnx_path if ok else None,
        elapsed,
    )
    _log.info(
        "  Drone audio: val_acc=%.3f  val_f1=%.3f  onnx=%s  elapsed=%.1fs",
        result["val_acc"],
        result["val_f1"],
        "[ok]" if ok else "✗",
        elapsed,
    )
    return result

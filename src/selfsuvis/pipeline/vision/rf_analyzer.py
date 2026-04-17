"""RF signal analyzer — TorchSig-based IQ signal characterization.

Analyzes IQ (in-phase/quadrature) recordings captured alongside mission video
and produces per-frame signal metrics stored in ``frame_facts_json["rf_signal"]``.

Disabled by default (``RF_ENABLED=false``).  Enable with::

    RF_ENABLED=true RF_SAMPLE_RATE=1000000 python worker/main.py

IQ data sources (checked in order for each video):
  1. ``<video_basename>.iq``         — raw interleaved float32 I/Q samples
  2. ``<video_basename>.sigmf-data`` — SigMF binary (requires matching ``.sigmf-meta``)
  3. Audio track from video          — real-valued proxy (16 kHz WAV), useful when no
                                       dedicated SDR capture is available

Output written to ``frame_facts_json["rf_signal"]`` per frame::

    {
        "snr_db":           12.3,   # estimated signal-to-noise ratio (dB)
        "spectral_flatness": 0.71,  # 0 = tonal, 1 = flat noise (Wiener measure)
        "peak_freq_ratio":   0.25,  # normalised peak-energy bin (0–1 of bandwidth)
        "occupied_bw_ratio": 0.38,  # fraction of spectrum above noise floor
        "modulation_class":  "QAM16",    # optional — requires RF_CLASSIFIER_CHECKPOINT
        "modulation_confidence": 0.85,   # optional
        "source":          "iq_file",    # iq_file | sigmf | audio_proxy
        "sample_rate":      1000000,
    }

Modulation classifier
---------------------
TorchSig does not ship a pre-trained classifier; you must train one using
``torchsig.datasets`` (see ``docs/rf_training.md``) and export it as a
TorchScript ``.pt`` file.  Point ``RF_CLASSIFIER_CHECKPOINT`` at that file to
enable modulation classification.  Without it the pass still runs and stores
the four signal-quality metrics above.
"""
from __future__ import annotations

import gc
import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)

# Modulation class labels in the same order TorchSig uses internally.
# Override via RF_CLASSIFIER_CLASSES env var (JSON list).
_DEFAULT_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-DSB-WC", "AM-DSB-SC", "AM-USB", "AM-LSB",
    "FM", "GMSK", "OQPSK",
]


# ── IQ file discovery ─────────────────────────────────────────────────────────


def _find_iq_sidecar(video_path: str) -> Tuple[Optional[str], str]:
    """Return (iq_path, source_tag) for the first sidecar found, or (None, '')."""
    base = os.path.splitext(video_path)[0]
    candidates = [
        (base + ".iq", "iq_file"),
        (base + ".bin", "iq_file"),
        (base + ".sigmf-data", "sigmf"),
    ]
    for path, tag in candidates:
        if os.path.isfile(path):
            return path, tag
    return None, ""


def _load_iq_file(path: str, source: str) -> Tuple[Optional[np.ndarray], float]:
    """Load IQ samples as complex64 array.  Returns (samples, sample_rate)."""
    sample_rate = float(settings.RF_SAMPLE_RATE)
    try:
        if source == "sigmf":
            meta_path = path.replace(".sigmf-data", ".sigmf-meta")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                sr = (
                    meta.get("global", {}).get("core:sample_rate")
                    or meta.get("global", {}).get("sample_rate")
                )
                if sr:
                    sample_rate = float(sr)
            raw = np.fromfile(path, dtype=np.float32)
        else:
            raw = np.fromfile(path, dtype=np.float32)

        if raw.size < 2:
            return None, sample_rate
        # Interleaved I/Q → complex
        samples = raw[0::2] + 1j * raw[1::2]
        return samples.astype(np.complex64), sample_rate
    except Exception:
        logger.warning("RF: failed to load IQ file %s", path, exc_info=True)
        return None, sample_rate


def _load_audio_proxy(wav_path: str) -> Tuple[Optional[np.ndarray], float]:
    """Load a WAV file as a real-valued proxy signal (treated as I-only baseband)."""
    try:
        import wave
        with wave.open(wav_path, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        dtype = np.int16 if sampwidth == 2 else np.int32
        pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if nchannels > 1:
            pcm = pcm[0::nchannels]
        # Normalise to [-1, 1] and treat as real-only complex signal
        peak = float(np.iinfo(dtype).max)
        pcm /= peak
        samples = (pcm + 0j).astype(np.complex64)
        return samples, float(framerate)
    except Exception:
        logger.warning("RF: failed to load audio proxy", exc_info=True)
        return None, 16000.0


# ── Signal feature extraction ─────────────────────────────────────────────────


def _extract_features(
    segment: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
) -> Dict[str, Any]:
    """Compute spectral features from one IQ segment using numpy FFT.

    Falls back gracefully to torchsig.transforms.Spectrogram if available,
    but the numpy path is always used as the base since not all environments
    will have torchsig installed.
    """
    if segment.size < nperseg:
        return {"rf_insufficient_samples": True}

    # Try torchsig spectrogram transform first (better windowing options).
    spec_db: Optional[np.ndarray] = _torchsig_spectrogram(segment, nperseg)

    if spec_db is None:
        # Pure numpy STFT fallback
        spec_db = _numpy_spectrogram(segment, nperseg)

    if spec_db is None or spec_db.size == 0:
        return {"rf_spectrogram_error": True}

    # Average across time → power spectral density vector
    psd = spec_db.mean(axis=-1) if spec_db.ndim == 2 else spec_db

    # SNR estimate: top 10 % bins as signal, bottom 10 % as noise floor
    sorted_psd = np.sort(psd)
    n = len(sorted_psd)
    noise_floor = float(sorted_psd[: max(1, n // 10)].mean())
    signal_peak = float(sorted_psd[max(0, n - n // 10) :].mean())
    snr_db = float(np.clip(signal_peak - noise_floor, -30.0, 60.0))

    # Spectral flatness (Wiener entropy): geometric mean / arithmetic mean
    # Use linear power, clipped to avoid log(0)
    lin = np.power(10.0, psd / 10.0).clip(1e-12)
    geo_mean = float(np.exp(np.log(lin).mean()))
    arith_mean = float(lin.mean())
    spectral_flatness = float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0))

    # Peak frequency bin (normalised 0–1)
    peak_idx = int(np.argmax(psd))
    peak_freq_ratio = float(peak_idx / max(1, len(psd) - 1))

    # Occupied bandwidth: fraction of bins above noise_floor + 3 dB
    threshold = noise_floor + 3.0
    occupied = float((psd > threshold).mean())

    return {
        "snr_db": round(snr_db, 2),
        "spectral_flatness": round(spectral_flatness, 4),
        "peak_freq_ratio": round(peak_freq_ratio, 4),
        "occupied_bw_ratio": round(occupied, 4),
    }


def _numpy_spectrogram(samples: np.ndarray, nperseg: int) -> Optional[np.ndarray]:
    """Compute magnitude spectrogram (dB) via numpy FFT with Hann window."""
    try:
        window = np.hanning(nperseg).astype(np.float32)
        n_frames = len(samples) // nperseg
        if n_frames < 1:
            return None
        frames = samples[: n_frames * nperseg].reshape(n_frames, nperseg)
        windowed = frames * window
        fft_out = np.fft.fft(windowed, axis=1)
        mag = np.abs(fft_out[:, : nperseg // 2]).astype(np.float32)
        mag = np.where(mag > 0, mag, 1e-12)
        spec_db = 20.0 * np.log10(mag)
        # shape: (freq_bins, time_frames) to match torchsig convention
        return spec_db.T
    except Exception:
        return None


def _torchsig_spectrogram(samples: np.ndarray, nperseg: int) -> Optional[np.ndarray]:
    """Try to compute spectrogram via torchsig transforms; return None if unavailable."""
    try:
        import torch
        from torchsig.transforms.signal_processing import Spectrogram  # type: ignore[import]

        spec_transform = Spectrogram(nperseg=nperseg)
        tensor = torch.from_numpy(
            np.stack([samples.real, samples.imag], axis=0).astype(np.float32)
        )
        result = spec_transform(tensor)
        arr = result.numpy() if hasattr(result, "numpy") else np.array(result)
        # torchsig returns linear magnitude; convert to dB
        arr = np.where(arr > 0, arr, 1e-12)
        return 20.0 * np.log10(arr).astype(np.float32)
    except (ImportError, AttributeError, Exception):
        return None


# ── Modulation classifier (optional) ─────────────────────────────────────────


def _load_classifier(checkpoint_path: str):
    """Load a TorchScript modulation classifier from *checkpoint_path*."""
    try:
        import torch
        model = torch.jit.load(checkpoint_path, map_location="cpu")
        model.eval()
        logger.info("RF: loaded modulation classifier from %s", checkpoint_path)
        return model
    except Exception:
        logger.warning(
            "RF: failed to load classifier from %s — modulation labelling disabled",
            checkpoint_path,
            exc_info=True,
        )
        return None


def _classify_segment(
    segment: np.ndarray,
    classifier,
    classes: List[str],
    nperseg: int = 256,
) -> Tuple[str, float]:
    """Run modulation classifier; returns (class_name, confidence)."""
    try:
        import torch

        spec_db = _torchsig_spectrogram(segment, nperseg) or _numpy_spectrogram(segment, nperseg)
        if spec_db is None:
            return "unknown", 0.0

        # Normalise to [0, 1] for classifier input
        s_min, s_max = spec_db.min(), spec_db.max()
        if s_max > s_min:
            spec_norm = (spec_db - s_min) / (s_max - s_min)
        else:
            spec_norm = spec_db

        tensor = torch.from_numpy(spec_norm[np.newaxis, np.newaxis].astype(np.float32))
        with torch.no_grad():
            logits = classifier(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        idx = int(np.argmax(probs))
        label = classes[idx] if idx < len(classes) else f"class_{idx}"
        return label, float(round(probs[idx], 4))
    except Exception:
        logger.debug("RF: classification failed", exc_info=True)
        return "unknown", 0.0


# ── Main analyzer class ───────────────────────────────────────────────────────


class RFSignalAnalyzer:
    """Per-mission RF signal analyzer.

    Loads IQ data once per video, then slices it per frame timestamp.
    Falls back to audio proxy when no IQ sidecar is present.
    """

    def __init__(self) -> None:
        self._classifier = None
        self._classes: List[str] = _DEFAULT_CLASSES
        self._load_classifier_if_configured()

    def is_enabled(self) -> bool:
        return settings.RF_ENABLED

    # ── public API ────────────────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: str,
        frame_timestamps: List[float],
        audio_wav_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze all frame timestamps for *video_path*.

        Returns a list of result dicts (one per timestamp), in the same order.
        Each dict is ready to merge into ``frame_facts_json``.
        """
        samples, sample_rate, source = self._load_signal(video_path, audio_wav_path)

        if samples is None:
            logger.info("RF: no IQ data available for %s — pass skipped", video_path)
            return [{}] * len(frame_timestamps)

        logger.info(
            "RF: analyzing %d frames from %s (source=%s, sr=%.0f Hz, %.1f s total)",
            len(frame_timestamps),
            video_path,
            source,
            sample_rate,
            len(samples) / sample_rate,
        )

        window_samples = int(settings.RF_WINDOW_SEC * sample_rate)
        results: List[Dict[str, Any]] = []

        for t_sec in frame_timestamps:
            start = int(t_sec * sample_rate)
            end = start + window_samples
            if start >= len(samples):
                results.append({})
                continue

            segment = samples[start:end]
            features = _extract_features(segment, sample_rate, nperseg=settings.RF_NPERSEG)

            if self._classifier is not None and "snr_db" in features:
                mod_class, mod_conf = _classify_segment(
                    segment, self._classifier, self._classes, nperseg=settings.RF_NPERSEG
                )
                features["modulation_class"] = mod_class
                features["modulation_confidence"] = mod_conf

            features["source"] = source
            features["sample_rate"] = int(sample_rate)
            results.append({"rf_signal": features})

        good = sum(1 for r in results if r.get("rf_signal", {}).get("snr_db") is not None)
        logger.info("RF: analysis complete — %d/%d frames have signal metrics", good, len(results))
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_signal(
        self,
        video_path: str,
        audio_wav_path: Optional[str],
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """Return (samples, sample_rate, source_tag) from best available source."""
        iq_path, source = _find_iq_sidecar(video_path)
        if iq_path:
            samples, sr = _load_iq_file(iq_path, source)
            if samples is not None and samples.size > 0:
                return samples, sr, source

        if audio_wav_path and os.path.isfile(audio_wav_path):
            samples, sr = _load_audio_proxy(audio_wav_path)
            if samples is not None and samples.size > 0:
                return samples, sr, "audio_proxy"

        return None, float(settings.RF_SAMPLE_RATE), ""

    def _load_classifier_if_configured(self) -> None:
        ckpt = settings.RF_CLASSIFIER_CHECKPOINT
        if not ckpt:
            return
        self._classifier = _load_classifier(ckpt)

        classes_json = settings.RF_CLASSIFIER_CLASSES
        if classes_json:
            try:
                parsed = json.loads(classes_json)
                if isinstance(parsed, list) and parsed:
                    self._classes = [str(c) for c in parsed]
            except Exception:
                logger.warning("RF: RF_CLASSIFIER_CLASSES is not valid JSON; using defaults")

    def release(self) -> None:
        """Free classifier memory."""
        self._classifier = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

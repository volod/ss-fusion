"""Step 33: drau range-detection evaluation (github.com/volod/drau).

Evaluates the trained DroneAudioCNN ONNX model across simulated acoustic distances
using the drau physics model:

  - Inverse-square amplitude scaling (1/d relative to 1 m reference)
  - ISO 9613-1 atmospheric absorption at 20 degC, 70% RH (first-order lowpass)

Generates synthetic quadcopter audio (sum of blade-pass harmonics) at each test
distance, computes MFCC features (reusing steps_drone_audio constants), runs ONNX
inference without PyTorch, and plots a detection-probability vs distance curve.

Also exports drau_edge_test.py alongside the ONNX model — a standalone script
that requires only numpy, scipy, and onnxruntime (no PyTorch, no selfsuvis).

Outputs (under video_dir/drone_audio/):
  drau_range_report.md    detection probability vs distance
  drau_edge_test.py       standalone edge inference script
"""

import math
import time
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger
from ..common import write_markdown_artifact

# Reuse MFCC constants and implementation from the training step.
from .drone_audio import (
    _N_MFCC,
    _T_FRAMES,
    _SR,
    _N_FFT,
    _HOP_LENGTH,
    _N_MELS,
    _compute_mfcc,
    _load_wav_mono,
    _collect_split,
)

_log = get_logger("pipeline.local.drau_eval")

# Test distances in metres (mirrors drau's session distance range).
_TEST_DISTANCES_M = [1, 5, 10, 25, 50, 75, 100, 150, 200]
# Synthetic signals generated per distance to average out phase randomness.
_N_SYNTHETIC = 8
# Typical small quadcopter blade-pass fundamental frequency (Hz).
_DRONE_FUNDAMENTAL_HZ = 300.0
# Number of harmonics in synthetic signal.
_N_HARMONICS = 7

# ISO 9613-1 absorption reference — 2 kHz at 20 degC, 70% RH (dB/m).
_AIR_ABS_ALPHA_2KHZ = 1.9e-2
# Reference distance for inverse-square normalisation (metres).
_REF_DIST_M = 1.0
# RMS target before distance scaling (linear; -18 dBFS).
_RMS_TARGET = 10.0 ** (-18.0 / 20.0)


# -- Physics (ported from drau/src/drau/detection_test/player.py) --------------


def _rms(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    return max(rms, 1e-12)


def _normalize_to_rms(signal: np.ndarray, target: float = _RMS_TARGET) -> np.ndarray:
    return (signal * (target / _rms(signal))).astype(np.float32)


def _air_absorption_cutoff_hz(distance_m: float) -> float:
    """ISO 9613-1 first-order lowpass cutoff matching HF attenuation at distance_m.

    Derivation: absorption alpha ~ (f/f_ref)^2 * alpha_ref.
    3 dB point: distance_m * alpha_ref * (f/2000)^2 = 3
    => f_3dB = 2000 * sqrt(3 / (distance_m * alpha_ref))
    """
    d = max(distance_m, 0.01)
    f2 = 3.0 / (d * _AIR_ABS_ALPHA_2KHZ)
    cutoff = 2000.0 * math.sqrt(max(f2, 1e-6))
    return max(100.0, min(cutoff, _SR / 2.0 * 0.95))


def _apply_distance_physics(
    signal: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    """Apply inverse-square gain and atmospheric HF absorption for distance_m.

    Mirrors drau's play_at_distance() without the sounddevice playback:
      1. Normalize to -18 dBFS RMS (reference level).
      2. Apply inverse-distance amplitude gain: gain = ref_dist / distance_m.
      3. Apply ISO 9613-1 first-order lowpass for atmospheric HF rolloff.
    """
    from scipy.signal import butter, lfilter

    normed = _normalize_to_rms(signal)
    gain = _REF_DIST_M / max(distance_m, 0.01)
    scaled = (normed * gain).astype(np.float32)

    cutoff = _air_absorption_cutoff_hz(distance_m)
    nyq = _SR / 2.0
    if cutoff < nyq * 0.9:
        b, a = butter(1, cutoff / nyq, btype="low")
        scaled = lfilter(b, a, scaled).astype(np.float32)

    return scaled


# -- Synthetic drone signal ----------------------------------------------------


def _synthesize_drone(
    fundamental_hz: float = _DRONE_FUNDAMENTAL_HZ,
    n_harmonics: int = _N_HARMONICS,
    duration_s: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Synthetic quadcopter: sum of blade-pass harmonics with random phase.

    Amplitude of k-th harmonic decays as 1/k (aeroacoustics approximation).
    White motor noise added at -26 dB relative to fundamental.
    """
    rng = np.random.default_rng(seed)
    n = int(_SR * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        signal += (1.0 / k) * np.sin(2.0 * math.pi * k * fundamental_hz * t + phase)
    noise_std = 10.0 ** (-26.0 / 20.0)
    signal += noise_std * rng.standard_normal(n)
    peak = np.max(np.abs(signal))
    if peak > 0.0:
        signal /= peak
    return signal.astype(np.float32)


# -- ONNX inference (no PyTorch) -----------------------------------------------


def _load_onnx_session(onnx_path: Path) -> Any | None:
    """Load ONNX model. Returns session or None if onnxruntime unavailable."""
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception as exc:
        _log.warning("onnxruntime not available: %s", exc)
        return None


def _infer_drone_prob(session: Any, wave: np.ndarray) -> float:
    """Return P(drone) from one 1-second waveform via ONNX inference."""
    mfcc = _compute_mfcc(wave)  # (N_MFCC, T_FRAMES)
    inp = mfcc[np.newaxis, np.newaxis, :, :]  # (1, 1, N_MFCC, T_FRAMES)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: inp})[0][0]  # (2,)
    # Softmax to get probability
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    return float(probs[1])  # index 1 = drone class


# -- Range evaluation ----------------------------------------------------------


def _evaluate_range(
    session: Any,
    real_samples: list[Path],
) -> dict[int, dict[str, float]]:
    """Evaluate detection probability at each test distance.

    Uses synthetic signals for all distances. If real drone WAV samples are
    available they are also processed (averaged into synthetic results).

    Returns: {distance_m: {"p_drone": float, "n_signals": int}}
    """
    results: dict[int, dict[str, float]] = {}

    for dist in _TEST_DISTANCES_M:
        probs: list[float] = []

        # Synthetic signals
        for seed in range(_N_SYNTHETIC):
            wave = _synthesize_drone(seed=seed)
            wave_dist = _apply_distance_physics(wave, float(dist))
            probs.append(_infer_drone_prob(session, wave_dist))

        # Real samples (up to 8)
        for wav_path in real_samples[:8]:
            wave = _load_wav_mono(wav_path)
            if wave is not None:
                wave_dist = _apply_distance_physics(wave, float(dist))
                probs.append(_infer_drone_prob(session, wave_dist))

        results[dist] = {
            "p_drone": float(np.mean(probs)),
            "p_drone_std": float(np.std(probs)),
            "n_signals": len(probs),
        }
        _log.info(
            "  d=%3d m  P(drone)=%.3f +/- %.3f  (n=%d)",
            dist,
            results[dist]["p_drone"],
            results[dist]["p_drone_std"],
            results[dist]["n_signals"],
        )

    return results


def _detection_range_m(results: dict[int, dict[str, float]], threshold: float = 0.5) -> int | None:
    """Return the largest distance at which P(drone) >= threshold."""
    last = None
    for d in sorted(results):
        if results[d]["p_drone"] >= threshold:
            last = d
    return last


# -- ASCII distance plot -------------------------------------------------------


def _ascii_bar(p: float, width: int = 30) -> str:
    filled = round(p * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _build_report_lines(
    onnx_path: Path,
    results: dict[int, dict[str, float]],
    elapsed: float,
    n_real: int,
) -> list[str]:
    det_range = _detection_range_m(results)
    lines = [
        "# drau Range-Detection Evaluation Report",
        "",
        "Source: github.com/volod/drau",
        "Physics: inverse-square amplitude + ISO 9613-1 atmospheric absorption (20 degC, 70% RH)",
        "",
        f"Model: `{onnx_path.name}`",
        f"Evaluation: {_N_SYNTHETIC} synthetic signals/distance + {n_real} real samples/distance",
        f"Elapsed: {elapsed:.1f} s",
        "",
        "## Detection Probability vs Distance",
        "",
        "| Distance (m) | P(drone) | Std | Signals | Bar |",
        "|-------------|---------|-----|---------|-----|",
    ]
    for d in sorted(results):
        r = results[d]
        p = r["p_drone"]
        bar = _ascii_bar(p)
        lines.append(
            f"| {d:>4} | {p:.3f} | {r['p_drone_std']:.3f} | {r['n_signals']} | {bar} |"
        )
    lines += [
        "",
        f"**Estimated detection range** (P >= 0.50): "
        + (f"{det_range} m" if det_range is not None else "< 1 m (model not loaded or untrained)"),
        "",
        "## Signal Model",
        "",
        f"- Fundamental frequency: {_DRONE_FUNDAMENTAL_HZ:.0f} Hz (quadcopter blade-pass)",
        f"- Harmonics: {_N_HARMONICS}",
        "- Distance law: A(d) = A(1m) / d   (inverse-square pressure amplitude)",
        "- Atmospheric absorption: first-order IIR lowpass",
        "  f_3dB(d) = 2000 * sqrt(3 / (d * 0.019))   [ISO 9613-1, alpha_2kHz = 0.019 dB/m]",
        "",
        "### f_3dB rolloff by distance",
        "",
        "| Distance (m) | f_3dB (Hz) | Effect |",
        "|-------------|-----------|--------|",
    ]
    for d in _TEST_DISTANCES_M:
        f3 = _air_absorption_cutoff_hz(float(d))
        if f3 >= _SR / 2.0 * 0.95:
            effect = "no significant rolloff"
        elif f3 >= 4000:
            effect = "minor HF rolloff"
        elif f3 >= 2000:
            effect = "moderate HF rolloff"
        else:
            effect = "heavy HF attenuation"
        lines.append(f"| {d:>4} | {f3:>8.0f} | {effect} |")

    lines += [
        "",
        "## Edge Inference Script",
        "",
        "A standalone inference script is exported to `drau_edge_test.py` alongside",
        "the ONNX model. Requirements: numpy, scipy, onnxruntime (no PyTorch).",
        "",
        "Copy both files to any Arm/x86 edge device and run:",
        "",
        "```bash",
        "pip install numpy scipy onnxruntime",
        "python drau_edge_test.py drone_audio_cnn.onnx path/to/audio.wav",
        "",
        "# With distance simulation (apply drau physics before inference):",
        "python drau_edge_test.py drone_audio_cnn.onnx path/to/audio.wav --distance 50",
        "```",
        "",
        "## Next Steps",
        "",
        "1. **More data**: run `scripts/split_drone_audio_data.sh` to expand the dataset beyond the HF auto-download",
        "2. **Real hardware test**: use drau's `make run-session` to test physical microphones against the trained model",
        "3. **Threshold tuning**: adjust the confidence threshold in `drau_edge_test.py` to balance FP vs FN for your deployment distance",
        "4. **TFLite export**: convert `drone_audio_cnn.onnx` to TFLite for devices that prefer the TensorFlow Lite runtime",
    ]
    return lines


# -- Edge test script generation -----------------------------------------------


def _write_edge_test_script(dest: Path, onnx_name: str) -> None:
    """Write drau_edge_test.py — a standalone edge inference script.

    Deps: numpy, scipy, onnxruntime only. No PyTorch, no selfsuvis.
    Embeds the MFCC pipeline and drau distance physics inline.
    """
    script = textwrap.dedent(f"""\
        \"\"\"drau edge inference test script.

        Evaluates the DroneAudioCNN ONNX model on a WAV file.
        Optionally applies drau distance physics before inference.

        Source: selfsuvis pipeline (github.com/volod/drau physics model)

        Requirements: numpy scipy onnxruntime
        Install: pip install numpy scipy onnxruntime

        Usage:
          python drau_edge_test.py <model.onnx> <audio.wav> [--distance <m>]
          python drau_edge_test.py {onnx_name} recording.wav
          python drau_edge_test.py {onnx_name} recording.wav --distance 100
        \"\"\"
        import argparse
        import math
        import sys
        from pathlib import Path

        import numpy as np

        # -- Constants (must match training in steps_drone_audio.py) ----------
        _SR = {_SR}
        _N_MFCC = {_N_MFCC}
        _N_MELS = {_N_MELS}
        _N_FFT = {_N_FFT}
        _HOP_LENGTH = {_HOP_LENGTH}
        _T_FRAMES = {_T_FRAMES}
        _CONF_THRESH = 0.5
        _RMS_TARGET = 10.0 ** (-18.0 / 20.0)
        _AIR_ABS_ALPHA_2KHZ = 1.9e-2
        _REF_DIST_M = 1.0


        # -- MFCC (scipy-only, no librosa) ------------------------------------

        def _hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def _mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        def _mel_filterbank():
            fmin_mel = _hz_to_mel(0.0)
            fmax_mel = _hz_to_mel(_SR / 2.0)
            mel_pts = np.linspace(fmin_mel, fmax_mel, _N_MELS + 2)
            hz_pts = np.array([_mel_to_hz(m) for m in mel_pts])
            fft_freqs = np.linspace(0.0, _SR / 2.0, _N_FFT // 2 + 1)
            fb = np.zeros((_N_MELS, _N_FFT // 2 + 1), dtype=np.float32)
            for i in range(_N_MELS):
                low, ctr, high = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
                up = (fft_freqs - low) / max(ctr - low, 1e-9)
                down = (high - fft_freqs) / max(high - ctr, 1e-9)
                fb[i] = np.maximum(0.0, np.minimum(up, down))
            return fb

        _FILTERBANK = _mel_filterbank()

        def compute_mfcc(wave):
            from scipy.fft import dct
            from scipy.signal import stft
            chunk = _SR  # 1-second window
            if len(wave) < chunk:
                wave = np.pad(wave, (0, chunk - len(wave)))
            else:
                wave = wave[:chunk]
            _, _, Zxx = stft(
                wave.astype(np.float32),
                fs=_SR, window="hann",
                nperseg=_N_FFT, noverlap=_N_FFT - _HOP_LENGTH,
                return_onesided=True,
            )
            power = np.abs(Zxx) ** 2
            mel = _FILTERBANK @ power
            log_mel = np.log(mel + 1e-9)
            mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:_N_MFCC]
            T = mfcc.shape[1]
            if T < _T_FRAMES:
                mfcc = np.pad(mfcc, ((0, 0), (0, _T_FRAMES - T)))
            else:
                mfcc = mfcc[:, :_T_FRAMES]
            return mfcc.astype(np.float32)


        # -- Distance physics (drau ISO 9613-1 model) -------------------------

        def _rms(s):
            return float(np.sqrt(np.mean(s.astype(np.float64) ** 2))) or 1e-12

        def apply_distance(signal, distance_m):
            from scipy.signal import butter, lfilter
            normed = signal * (_RMS_TARGET / _rms(signal))
            gain = _REF_DIST_M / max(distance_m, 0.01)
            scaled = (normed * gain).astype(np.float32)
            d = max(distance_m, 0.01)
            f2 = 3.0 / (d * _AIR_ABS_ALPHA_2KHZ)
            cutoff = 2000.0 * math.sqrt(max(f2, 1e-6))
            cutoff = max(100.0, min(cutoff, _SR / 2.0 * 0.95))
            nyq = _SR / 2.0
            if cutoff < nyq * 0.9:
                b, a = butter(1, cutoff / nyq, btype="low")
                scaled = lfilter(b, a, scaled).astype(np.float32)
            return scaled


        # -- WAV loader -------------------------------------------------------

        def load_wav(path):
            try:
                import soundfile as sf
                audio, sr_in = sf.read(str(path), dtype="float32")
            except Exception:
                from scipy.io import wavfile
                sr_in, audio = wavfile.read(str(path))
                if audio.dtype.kind in ("i", "u"):
                    info = np.iinfo(audio.dtype)
                    audio = audio.astype(np.float32) / info.max
                else:
                    audio = audio.astype(np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr_in != _SR:
                n_out = int(len(audio) * _SR / sr_in)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            return audio


        # -- ONNX inference ---------------------------------------------------

        def infer(session, wave):
            mfcc = compute_mfcc(wave)
            inp = mfcc[np.newaxis, np.newaxis, :, :]
            name = session.get_inputs()[0].name
            logits = session.run(None, {{name: inp}})[0][0]
            e = np.exp(logits - np.max(logits))
            return float((e / e.sum())[1])


        # -- Main -------------------------------------------------------------

        def main():
            ap = argparse.ArgumentParser(description="drau edge inference test")
            ap.add_argument("model", help="Path to drone_audio_cnn.onnx")
            ap.add_argument("audio", help="Path to WAV file")
            ap.add_argument(
                "--distance",
                type=float,
                default=None,
                metavar="M",
                help="Simulate drone at this distance in metres (applies drau physics)",
            )
            ap.add_argument(
                "--threshold",
                type=float,
                default=_CONF_THRESH,
                metavar="T",
                help="Detection confidence threshold (default: %(default)s)",
            )
            args = ap.parse_args()

            try:
                import onnxruntime as ort
            except ImportError:
                print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
                sys.exit(1)

            model_path = Path(args.model)
            audio_path = Path(args.audio)
            if not model_path.exists():
                print(f"ERROR: model not found: {{model_path}}")
                sys.exit(1)
            if not audio_path.exists():
                print(f"ERROR: audio not found: {{audio_path}}")
                sys.exit(1)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

            wave = load_wav(audio_path)

            if args.distance is not None:
                wave = apply_distance(wave, args.distance)
                dist_label = f" (simulated at {{args.distance:.0f}} m)"
            else:
                dist_label = ""

            prob = infer(sess, wave)
            label = "DRONE" if prob >= args.threshold else "no_drone"
            print(f"{{label}}  confidence={{prob:.3f}}{{dist_label}}")
            sys.exit(0 if label == "DRONE" else 1)


        if __name__ == "__main__":
            main()
        """)
    dest.write_text(script, encoding="utf-8")


# -- Public step function ------------------------------------------------------


def step_drau_range_eval(
    video_dir: Path,
    output_dir: Path,
    args: Any,
) -> dict[str, Any]:
    """Evaluate DroneAudioCNN ONNX at simulated distances using drau physics.

    Requires drone_audio_cnn.onnx from step_drone_audio_training (Step 32).
    Skips gracefully if the ONNX model is absent or onnxruntime is unavailable.
    """
    from selfsuvis.pipeline.core.config import settings

    result: dict[str, Any] = {
        "skipped": False,
        "detection_range_m": None,
        "n_distances": 0,
        "onnx_found": False,
    }
    t0 = time.monotonic()

    audio_dir = video_dir / "drone_audio"
    onnx_path = audio_dir / "drone_audio_cnn.onnx"

    if not onnx_path.exists():
        _log.info(
            "drau eval: drone_audio_cnn.onnx not found at %s — step 32 must run first",
            onnx_path,
        )
        result["skipped"] = True
        result["reason"] = "drone_audio_cnn.onnx not found (run with --drone-audio first)"
        return result

    result["onnx_found"] = True
    _log.info("drau range eval: loading ONNX model from %s", onnx_path)
    session = _load_onnx_session(onnx_path)
    if session is None:
        result["skipped"] = True
        result["reason"] = "onnxruntime not available"
        return result

    # Collect real drone samples from dataset cache for supplementary testing.
    cache_dir = Path(settings.DRONE_AUDIO_DATA_DIR)
    real_samples: list[Path] = []
    for split in ("train", "val", "test"):
        drone_dir = cache_dir / split / "drone"
        if drone_dir.is_dir():
            real_samples.extend(sorted(drone_dir.glob("*.wav"))[:4])
    n_real = min(len(real_samples), 8)
    _log.info("drau eval: %d real drone samples available for supplementary testing", n_real)

    _log.info("drau eval: evaluating %d distances …", len(_TEST_DISTANCES_M))
    range_results = _evaluate_range(session, real_samples[:8])
    result["n_distances"] = len(range_results)

    det_range = _detection_range_m(range_results)
    result["detection_range_m"] = det_range
    if det_range is not None:
        _log.info("  [ok] Estimated detection range: %d m", det_range)
    else:
        _log.info("  [warn] Detection probability below 0.5 at all distances")

    # Write range report
    audio_dir.mkdir(parents=True, exist_ok=True)
    report_path = audio_dir / "drau_range_report.md"
    report_lines = _build_report_lines(onnx_path, range_results, time.monotonic() - t0, n_real)
    write_markdown_artifact(report_path, report_lines)
    result["report"] = str(report_path)
    _log.info("  [ok] Report: %s", report_path)

    # Write standalone edge test script
    edge_script = audio_dir / "drau_edge_test.py"
    _write_edge_test_script(edge_script, onnx_path.name)
    result["edge_script"] = str(edge_script)
    _log.info("  [ok] Edge script: %s", edge_script)

    result["elapsed_sec"] = time.monotonic() - t0
    return result

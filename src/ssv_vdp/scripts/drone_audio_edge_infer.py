"""Standalone edge inference script for DroneAudioCNN.

Evaluates a WAV file against the trained ONNX model.
Optionally applies drau distance physics (inverse-square + ISO 9613-1 absorption).

Requirements: numpy scipy onnxruntime
No PyTorch, no selfsuvis imports required.

Usage:
  python -m selfsuvis.scripts.drone_audio_edge_infer <model.onnx> <audio.wav>
  python -m selfsuvis.scripts.drone_audio_edge_infer <model.onnx> <audio.wav> --distance 100
  python -m selfsuvis.scripts.drone_audio_edge_infer <model.onnx> <audio.wav> --scan

See also: scripts/audio/drone_audio_edge_test.sh
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# -- Constants (must match training constants in steps_drone_audio.py) ---------
_SR = 22050
_N_MFCC = 40
_N_MELS = 64
_N_FFT = 1024
_HOP_LENGTH = 512
_T_FRAMES = _SR // _HOP_LENGTH + 1  # 44
_CONF_THRESH = 0.5
_RMS_TARGET = 10.0 ** (-18.0 / 20.0)
_AIR_ABS_ALPHA_2KHZ = 1.9e-2
_REF_DIST_M = 1.0
_SCAN_DISTANCES_M = [1, 5, 10, 25, 50, 75, 100, 150, 200]


# -- Mel filterbank (build once) -----------------------------------------------


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_filterbank() -> np.ndarray:
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


_FILTERBANK = _build_filterbank()


# -- MFCC (scipy-only, no librosa) ---------------------------------------------


def compute_mfcc(wave: np.ndarray) -> np.ndarray:
    """Return (N_MFCC, T_FRAMES) float32 MFCC from a mono waveform at _SR Hz."""
    from scipy.fft import dct
    from scipy.signal import stft

    chunk = _SR  # 1-second window
    if len(wave) < chunk:
        wave = np.pad(wave, (0, chunk - len(wave)))
    else:
        wave = wave[:chunk]

    _, _, Zxx = stft(
        wave.astype(np.float32),
        fs=_SR,
        window="hann",
        nperseg=_N_FFT,
        noverlap=_N_FFT - _HOP_LENGTH,
        return_onesided=True,
    )
    power = np.abs(Zxx) ** 2
    mel_spec = _FILTERBANK @ power
    log_mel = np.log(mel_spec + 1e-9)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:_N_MFCC]

    T = mfcc.shape[1]
    if T < _T_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, _T_FRAMES - T)))
    else:
        mfcc = mfcc[:, :_T_FRAMES]
    return mfcc.astype(np.float32)


# -- Distance physics (drau ISO 9613-1 model) ----------------------------------


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) or 1e-12


def apply_distance_physics(signal: np.ndarray, distance_m: float) -> np.ndarray:
    """Apply drau physics: inverse-square gain + ISO 9613-1 atmospheric absorption.

    1. Normalize to -18 dBFS RMS (reference level).
    2. Scale by 1/distance_m (inverse pressure-distance law).
    3. Apply first-order lowpass for HF atmospheric absorption.
    """
    from scipy.signal import butter, lfilter

    normed = signal * (_RMS_TARGET / _rms(signal))
    gain = _REF_DIST_M / max(distance_m, 0.01)
    scaled = (normed * gain).astype(np.float32)

    # ISO 9613-1 lowpass cutoff
    d = max(distance_m, 0.01)
    f2 = 3.0 / (d * _AIR_ABS_ALPHA_2KHZ)
    cutoff = 2000.0 * math.sqrt(max(f2, 1e-6))
    cutoff = max(100.0, min(cutoff, _SR / 2.0 * 0.95))
    nyq = _SR / 2.0
    if cutoff < nyq * 0.9:
        b, a = butter(1, cutoff / nyq, btype="low")
        scaled = lfilter(b, a, scaled).astype(np.float32)
    return scaled


# -- WAV loader ----------------------------------------------------------------


def load_wav_mono(path: Path) -> np.ndarray:
    """Load WAV as mono float32 at _SR Hz. Raises on failure."""
    try:
        import soundfile as sf
        audio, sr_in = sf.read(str(path), dtype="float32")
    except Exception:
        from scipy.io import wavfile
        sr_in, audio = wavfile.read(str(path))
        if audio.dtype.kind in ("i", "u"):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
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


# -- ONNX inference ------------------------------------------------------------


def load_session(model_path: Path):
    """Load ONNX session with single-threaded CPU provider."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def infer(session, wave: np.ndarray) -> float:
    """Return P(drone) for a 1-second mono waveform at _SR Hz."""
    mfcc = compute_mfcc(wave)
    inp = mfcc[np.newaxis, np.newaxis, :, :]  # (1, 1, N_MFCC, T_FRAMES)
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: inp})[0][0]
    e = np.exp(logits - np.max(logits))
    return float((e / e.sum())[1])  # index 1 = drone class


# -- Main ----------------------------------------------------------------------


def _print_result(label: str, prob: float, dist_label: str = "") -> None:
    bar_width = 30
    filled = round(prob * bar_width)
    bar = "[" + "#" * filled + "." * (bar_width - filled) + "]"
    print(f"{label:<9}  confidence={prob:.3f}  {bar}{dist_label}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DroneAudioCNN edge inference (numpy + scipy + onnxruntime only)"
    )
    ap.add_argument("model", help="Path to drone_audio_cnn.onnx")
    ap.add_argument("audio", help="Path to WAV audio file")
    ap.add_argument(
        "--distance",
        type=float,
        default=None,
        metavar="M",
        help="Simulate drone at this distance in metres (applies drau physics before inference)",
    )
    ap.add_argument(
        "--scan",
        action="store_true",
        help="Scan all standard distances and print a range-detection table",
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
        import onnxruntime  # noqa: F401
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    for p, label in [(model_path, "model"), (audio_path, "audio")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    try:
        from scipy.signal import stft  # noqa: F401
    except ImportError:
        print("ERROR: scipy not installed. Run: pip install scipy")
        sys.exit(1)

    session = load_session(model_path)
    wave = load_wav_mono(audio_path)

    if args.scan:
        print(f"\nRange scan: {audio_path.name}")
        print(f"{'Distance (m)':<14}  {'P(drone)':<10}  {'Label':<9}  Bar")
        print("-" * 65)
        last_detected = None
        for d in _SCAN_DISTANCES_M:
            w = apply_distance_physics(wave, float(d))
            prob = infer(session, w)
            label = "DRONE" if prob >= args.threshold else "no_drone"
            if label == "DRONE":
                last_detected = d
            bar_w = 30
            filled = round(prob * bar_w)
            bar = "[" + "#" * filled + "." * (bar_w - filled) + "]"
            print(f"{d:<14}  {prob:<10.3f}  {label:<9}  {bar}")
        print()
        if last_detected is not None:
            print(f"Estimated detection range: {last_detected} m (threshold={args.threshold})")
        else:
            print("No detection at any distance.")
        sys.exit(0)

    if args.distance is not None:
        wave = apply_distance_physics(wave, args.distance)
        dist_label = f"  (simulated at {args.distance:.0f} m)"
    else:
        dist_label = ""

    prob = infer(session, wave)
    label = "DRONE" if prob >= args.threshold else "no_drone"
    _print_result(label, prob, dist_label)
    sys.exit(0 if label == "DRONE" else 1)


if __name__ == "__main__":
    main()

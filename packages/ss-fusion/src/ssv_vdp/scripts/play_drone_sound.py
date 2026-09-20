"""Simulate and play physically-realistic drone audio.

Models a drone as a point source moving through 3-D space.  Applies:
  - Inverse-square-law amplitude decay  (20 dB per distance doubling)
  - Atmospheric absorption             (~0.5 dB / 100 m typical)
  - Doppler pitch shift                (emission-time interpolation)
  - Speaker/system-volume compensation (best-effort playback calibration)

Usage:
    python -m selfsuvis.scripts.play_drone_sound [options]
    scripts/play_drone_sound.sh [options]

Examples:
    # Drone flies from 200 m, directly over mic, at 10 m/s
    scripts/play_drone_sound.sh --scenario flyover --distance 200 --speed 10

    # Hovering drone at 30 m directly above
    scripts/play_drone_sound.sh --scenario hover --distance 30

    # Circular flight at 50 m radius
    scripts/play_drone_sound.sh --scenario circle --distance 50 --speed 8

    # Save instead of playing
    scripts/play_drone_sound.sh --scenario flyover --distance 100 --output sim.wav

Physics parameters:
    --source-db     Reference dBSPL at 1 m (default 85) — typical multirotor
    --speaker-ref-db Estimated speaker dBSPL at full-scale / 100% OS volume
    --system-volume Current output volume 0.0..1.0, or auto-detect it
    --mic-type      Microphone used by the acoustic probe / detector
    --player-type   Playback device type
    --probe-distance-m Distance from speaker to probe microphone
    --c             Speed of sound in m/s (default 343)
    --atm-db        Atmospheric absorption dB per 100 m (default 0.5)

The simulated audio uses a real drone WAV from the cache directory when
available; otherwise a synthetic drone-like tone is synthesised.
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_SR = 22050
_SOURCE_DB_DEFAULT = 85.0  # dBSPL at 1 m — typical multirotor
_C_DEFAULT = 343.0  # m/s — speed of sound at 20 °C
_ATM_DB_DEFAULT = 0.5  # dB / 100 m
_REF_DIST = 1.0  # m — reference distance for source_db
_SPEAKER_REF_DB_DEFAULT = 85.0
_PLAYBACK_HEADROOM = 0.95
_MIC_TYPES = ("measurement", "acoustic", "embedded", "headset", "phone", "unknown")
_PLAYER_TYPES = ("single-speaker", "stereo-speakers", "laptop", "phone", "headphones", "unknown")


# -- Source audio --------------------------------------------------------------


def _find_drone_wav(data_dir: Path) -> Path | None:
    """Return the first available drone WAV from the dataset cache."""
    for subdir in ("train/drone", "val/drone", "test/drone", "drone"):
        d = data_dir / subdir
        if d.is_dir():
            wavs = sorted(d.glob("*.wav"))
            if wavs:
                return wavs[0]
    return None


def _load_wav_mono(path: Path, target_sr: int) -> np.ndarray | None:
    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        elif data.dtype.kind == "u":
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != target_sr:
            n_out = int(len(data) * target_sr / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, n_out),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        return data
    except Exception as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
        return None


def _synthesise_drone(duration_s: float, sr: int) -> np.ndarray:
    """Generate a synthetic drone-like tone (fundamental + harmonics + noise)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    # Typical multirotor: ~80-120 Hz blade-pass fundamental + odd harmonics + broadband
    f0 = 95.0
    signal = (
        0.50 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 4 * f0 * t)
        + 0.08 * np.sin(2 * np.pi * 6 * f0 * t)
        + 0.05 * np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    )
    return signal / np.abs(signal).max()


def _loop_audio(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Tile audio to cover n_samples by looping with cross-fade."""
    reps = math.ceil(n_samples / len(audio)) + 1
    long = np.tile(audio, reps)
    return long[:n_samples].copy()


def _normalise_source(audio: np.ndarray) -> np.ndarray:
    """Keep source samples in a stable full-scale range before physical gain."""
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio = audio / peak
    return audio


def _run_volume_command(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _current_system_volume() -> tuple[float | None, str]:
    """Best-effort output volume probe.

    Returns a linear volume in [0, 1] and the backend name.  This is used as a
    calibration input, not as an exact acoustic measurement.
    """
    wpctl = _run_volume_command(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
    if wpctl:
        match = re.search(r"Volume:\s*([0-9]*\.?[0-9]+)", wpctl)
        if match:
            if "MUTED" in wpctl.upper():
                return 0.0, "wpctl"
            return max(0.0, min(float(match.group(1)), 1.0)), "wpctl"

    pactl = _run_volume_command(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    if pactl:
        volumes = [int(value) for value in re.findall(r"(\d+)%", pactl)]
        if volumes:
            return max(0.0, min(float(sum(volumes) / len(volumes)) / 100.0, 1.0)), "pactl"

    amixer = _run_volume_command(["amixer", "get", "Master"])
    if amixer:
        volumes = [int(value) for value in re.findall(r"\[(\d+)%\]", amixer)]
        if volumes:
            muted = "[off]" in amixer.lower()
            if muted:
                return 0.0, "amixer"
            return max(0.0, min(float(sum(volumes) / len(volumes)) / 100.0, 1.0)), "amixer"

    return None, "unknown"


def _resolve_system_volume(value: str) -> tuple[float, str]:
    if value == "auto":
        detected, backend = _current_system_volume()
        if detected is None:
            return 1.0, "unknown; assuming 100%"
        return detected, backend
    volume = float(value)
    if not 0.0 <= volume <= 1.0:
        raise ValueError("--system-volume must be auto or a value between 0.0 and 1.0")
    return volume, "manual"


def _placement_lines(mic_type: str, player_type: str, probe_distance_m: float) -> list[str]:
    question = (
        "Calibration question: Which microphone will record the emulated drone sound, "
        "and where will it be during the test?"
    )
    lines = [question]

    if player_type == "headphones":
        lines.append(
            "Headphones: do not use open-air calibration; put the mic in the same coupler/ear "
            "position used for the real test, or use speakers instead."
        )
    elif player_type == "laptop":
        lines.append(
            "Laptop speaker: put the mic on-axis with the loudest speaker grille, keep the "
            "laptop on the same surface used for the test, and avoid blocking ports or vents."
        )
    elif player_type == "phone":
        lines.append(
            "Phone player: aim the mic at the phone speaker port and keep the phone case/orientation "
            "unchanged during calibration and playback."
        )
    elif player_type == "stereo-speakers":
        lines.append(
            "Stereo speakers: put the mic centered between left/right speakers at tweeter height, "
            "or use the exact detector position if the test setup is fixed."
        )
    else:
        lines.append(
            "Speaker player: put the mic on-axis with the speaker center at the same height as "
            "the driver, with a clear line of sight."
        )

    if mic_type in {"measurement", "acoustic"}:
        lines.append(
            "Measurement/acoustic mic: point the capsule at the speaker, use a stand, disable AGC "
            "if available, and start at 1.0 m unless testing the real detector position."
        )
    elif mic_type == "embedded":
        lines.append(
            "Embedded mic: place the whole device where the detector will sit, keep mic holes "
            "uncovered, and do not rest the mic side directly on a reflective tabletop."
        )
    elif mic_type == "headset":
        lines.append(
            "Headset mic: place only the boom capsule at the intended probe point. A 30 cm "
            "speaker distance is acceptable only if that is the real detector geometry."
        )
    elif mic_type == "phone":
        lines.append(
            "Phone mic: point the phone's primary mic toward the speaker, keep the same case, "
            "and turn off voice enhancement/noise suppression where possible."
        )
    else:
        lines.append(
            "Unknown mic: use the real detector position, keep orientation fixed, and avoid "
            "touching the mic or speaker after setting levels."
        )

    if probe_distance_m < 0.75:
        lines.append(
            f"Near-field note: {probe_distance_m:.2f} m is much closer than the 1 m speaker "
            "reference; reflections and bass response can dominate, so treat this as a "
            "geometry-specific calibration."
        )
    elif probe_distance_m > 2.0:
        lines.append(
            f"Room note: {probe_distance_m:.2f} m is far enough that room reflections matter; "
            "prefer the exact detector position over a generic reference distance."
        )
    else:
        lines.append(
            f"Reference placement: {probe_distance_m:.2f} m is suitable for a simple speaker "
            "reference if the mic is on-axis and unobstructed."
        )

    return lines


def _speaker_ref_at_probe_distance(speaker_ref_db: float, probe_distance_m: float) -> float:
    """Adjust a 1 m speaker reference for the chosen microphone placement distance."""
    distance = max(probe_distance_m, 0.05)
    return speaker_ref_db - 20.0 * math.log10(distance / _REF_DIST)


def _print_placement_guidance(mic_type: str, player_type: str, probe_distance_m: float) -> None:
    print("\nProbe placement")
    for line in _placement_lines(mic_type, player_type, probe_distance_m):
        print(f"  {line}")
    print()


def _apply_distance_playback_gain(
    audio: np.ndarray,
    distance_m: np.ndarray,
    source_db: float,
    atm_db_per_100m: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Convert simulated received SPL into output samples.

    speaker_ref_db is an approximate SPL produced by a full-scale signal at
    100% system output volume.  system_volume compensates for the current OS
    speaker setting so the requested distance maps to a plausible playback
    level on the current machine.  Without speaker calibration this remains an
    estimate, but it preserves distance-dependent loudness.
    """
    distance_m = np.maximum(distance_m.astype(np.float32), 0.1)
    received_db = (
        source_db - 20.0 * np.log10(distance_m / _REF_DIST) - atm_db_per_100m * distance_m / 100.0
    )
    effective_volume = max(system_volume, 0.05)
    gain = (10.0 ** ((received_db - speaker_ref_db) / 20.0)) / effective_volume
    out = audio * gain.astype(np.float32)
    unclipped_peak = float(np.abs(out).max())
    if unclipped_peak > _PLAYBACK_HEADROOM:
        out = np.tanh(out / _PLAYBACK_HEADROOM) * _PLAYBACK_HEADROOM
    return out.astype(np.float32), float(received_db.max()), unclipped_peak


# -- Physics -------------------------------------------------------------------


def _simulate(
    source_audio: np.ndarray,
    sr: int,
    x_traj: np.ndarray,  # drone x position at each output sample (m)
    y_offset: float,  # lateral / height offset — closest approach distance (m)
    z_offset: float,  # altitude offset (m) — adds to dist but not to x_traj
    source_db: float,
    c: float,
    atm_db_per_100m: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Apply inverse-square amplitude + Doppler to source_audio.

    x_traj has the same length as source_audio and describes the drone's
    x-coordinate over time.  The observer is at (0, y_offset, z_offset).
    """
    n = len(x_traj)
    # Distance observer ← drone at observation time
    d_obs = np.sqrt(x_traj**2 + y_offset**2 + z_offset**2)
    d_obs = np.maximum(d_obs, 0.1)  # avoid division by zero

    # Emission time: iterate to solve  t_obs = t_em + d(t_em)/c
    t_obs = np.arange(n, dtype=np.float64) / sr
    t_em = t_obs.copy()
    for _ in range(5):
        x_em = np.interp(t_em, t_obs, x_traj)
        d_em = np.sqrt(x_em**2 + y_offset**2 + z_offset**2)
        d_em = np.maximum(d_em, 0.1)
        t_em = t_obs - d_em / c

    # Resample source at emission times (Doppler)
    t_em_samples = np.clip(t_em * sr, 0, n - 1)
    doppler_audio = np.interp(t_em_samples, np.arange(n), source_audio).astype(np.float32)

    return _apply_distance_playback_gain(
        doppler_audio,
        d_obs,
        source_db,
        atm_db_per_100m,
        speaker_ref_db,
        system_volume,
    )


# -- Scenario builders ---------------------------------------------------------


def _scenario_flyover(
    source_audio: np.ndarray,
    sr: int,
    source_db: float,
    c: float,
    atm: float,
    distance: float,
    speed: float,
    duration: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Drone flies in a straight line; closest approach = distance (lateral offset)."""
    n = int(duration * sr)
    audio = _loop_audio(source_audio, n)
    x_start = -0.5 * speed * duration
    t = np.arange(n, dtype=np.float64) / sr
    x_traj = (x_start + speed * t).astype(np.float32)
    return _simulate(
        audio,
        sr,
        x_traj,
        y_offset=distance,
        z_offset=0.0,
        source_db=source_db,
        c=c,
        atm_db_per_100m=atm,
        speaker_ref_db=speaker_ref_db,
        system_volume=system_volume,
    )


def _scenario_approach(
    source_audio: np.ndarray,
    sr: int,
    source_db: float,
    c: float,
    atm: float,
    distance: float,
    speed: float,
    duration: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Drone approaches from distance, passes mic at d_min=5m, flies away."""
    n = int(duration * sr)
    audio = _loop_audio(source_audio, n)
    t = np.arange(n, dtype=np.float64) / sr
    # Start at x=-distance, pass x=0 at mid-point
    x_start = -distance
    x_traj = (x_start + speed * t).astype(np.float32)
    return _simulate(
        audio,
        sr,
        x_traj,
        y_offset=5.0,
        z_offset=0.0,
        source_db=source_db,
        c=c,
        atm_db_per_100m=atm,
        speaker_ref_db=speaker_ref_db,
        system_volume=system_volume,
    )


def _scenario_hover(
    source_audio: np.ndarray,
    sr: int,
    source_db: float,
    c: float,
    atm: float,
    distance: float,
    duration: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Drone hovers at fixed distance directly above the microphone."""
    n = int(duration * sr)
    audio = _loop_audio(source_audio, n)
    x_traj = np.zeros(n, dtype=np.float32)
    return _simulate(
        audio,
        sr,
        x_traj,
        y_offset=0.0,
        z_offset=distance,
        source_db=source_db,
        c=c,
        atm_db_per_100m=atm,
        speaker_ref_db=speaker_ref_db,
        system_volume=system_volume,
    )


def _scenario_circle(
    source_audio: np.ndarray,
    sr: int,
    source_db: float,
    c: float,
    atm: float,
    distance: float,
    speed: float,
    duration: float,
    speaker_ref_db: float,
    system_volume: float,
) -> tuple[np.ndarray, float, float]:
    """Drone circles at constant radius (= distance) around the microphone."""
    n = int(duration * sr)
    audio = _loop_audio(source_audio, n)
    t = np.arange(n, dtype=np.float64) / sr
    omega = speed / distance  # angular velocity rad/s
    x_traj = (distance * np.cos(omega * t)).astype(np.float32)
    y_offset_arr = (distance * np.abs(np.sin(omega * t))).astype(np.float32)
    # For circle, lateral offset changes: treat as a loop of flyovers
    # Simplify: use the instantaneous y component
    # _simulate expects constant y; vectorise manually
    d_obs = np.sqrt(x_traj**2 + y_offset_arr**2)
    d_obs = np.maximum(d_obs, 0.1)
    t_obs = t
    t_em = t_obs.copy()
    for _ in range(5):
        x_em = np.interp(t_em, t_obs, x_traj)
        ye_em = np.interp(t_em, t_obs, y_offset_arr)
        d_em = np.sqrt(x_em**2 + ye_em**2)
        d_em = np.maximum(d_em, 0.1)
        t_em = t_obs - d_em / c
    t_em_samples = np.clip(t_em * sr, 0, n - 1)
    doppler = np.interp(t_em_samples, np.arange(n), audio).astype(np.float32)
    return _apply_distance_playback_gain(
        doppler,
        d_obs,
        source_db,
        atm,
        speaker_ref_db,
        system_volume,
    )


# -- Output --------------------------------------------------------------------


def _play(audio: np.ndarray, sr: int) -> None:
    try:
        import sounddevice as sd  # type: ignore[import]

        print(f"Playing {len(audio) / sr:.1f} s …  (press Ctrl-C to stop)")
        sd.play(audio, samplerate=sr)
        sd.wait()
    except ImportError:
        print(
            "sounddevice not installed — cannot play audio.\n"
            "Install with:  pip install sounddevice\n"
            "Or use --output to save to a WAV file.",
            file=sys.stderr,
        )


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    from scipy.io import wavfile

    pcm = (audio * 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
    print(f"Saved: {path}  ({len(audio) / sr:.1f} s)")


def _print_summary(
    scenario: str,
    distance: float,
    speed: float,
    source_db: float,
    c: float,
    atm: float,
    duration: float,
    speaker_ref_db: float,
    effective_speaker_ref_db: float,
    system_volume: float,
    volume_backend: str,
    mic_type: str,
    player_type: str,
    probe_distance_m: float,
) -> None:
    print("\nDrone audio simulation")
    print(f"  scenario   : {scenario}")
    print(f"  distance   : {distance} m  (closest approach / hover / orbit radius)")
    print(f"  speed      : {speed} m/s")
    print(f"  source dB  : {source_db} dBSPL @ 1 m")
    print(f"  speaker ref: {speaker_ref_db} dBSPL @ full-scale, 100% system volume")
    print(f"  probe setup: {mic_type} mic, {player_type}, {probe_distance_m:.2f} m")
    print(f"  probe ref  : {effective_speaker_ref_db:.1f} dBSPL at probe position")
    print(f"  sys volume : {system_volume * 100:.0f}% ({volume_backend})")
    if system_volume <= 0.0:
        print("  warning    : output is muted; playback cannot produce the requested SPL")
    print(f"  duration   : {duration} s")

    for d in (distance * 0.5, distance, distance * 2):
        if d > 0:
            spread_loss = 20 * math.log10(max(d, 0.01))
            atm_loss = atm * d / 100.0
            total_db = source_db - spread_loss - atm_loss
            print(f"  level @ {d:6.0f} m : {total_db:.1f} dBSPL")
    print()


def _print_playback_stats(max_received_db: float, unclipped_peak: float) -> None:
    print(f"  peak target: {max_received_db:.1f} dBSPL")
    print(f"  peak sample: {min(unclipped_peak, _PLAYBACK_HEADROOM):.3f} full-scale")
    if unclipped_peak > _PLAYBACK_HEADROOM:
        print("  limiter    : active (requested playback exceeds digital headroom)")
    print()


# -- Main ----------------------------------------------------------------------


def main() -> None:
    from selfsuvis.pipeline.core.config import settings

    parser = argparse.ArgumentParser(description="Simulate drone audio with physics")
    parser.add_argument(
        "--scenario",
        choices=["flyover", "approach", "hover", "circle"],
        default="flyover",
        help="Flight scenario (default: flyover)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=200.0,
        help="Initial / closest-approach / hover / orbit distance in m (default: 200)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=10.0,
        help="Drone speed in m/s (default: 10; ignored for hover)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Simulation duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--source-db",
        type=float,
        default=_SOURCE_DB_DEFAULT,
        help=f"Drone source level dBSPL at 1 m (default: {_SOURCE_DB_DEFAULT})",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=_C_DEFAULT,
        help=f"Speed of sound m/s (default: {_C_DEFAULT})",
    )
    parser.add_argument(
        "--atm-db",
        type=float,
        default=_ATM_DB_DEFAULT,
        help=f"Atmospheric absorption dB per 100 m (default: {_ATM_DB_DEFAULT})",
    )
    parser.add_argument(
        "--speaker-ref-db",
        type=float,
        default=_SPEAKER_REF_DB_DEFAULT,
        help=(
            "Estimated dBSPL from this speaker at full-scale audio and 100%% system "
            f"volume (default: {_SPEAKER_REF_DB_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--system-volume",
        default="auto",
        help=(
            "Current output volume as 0.0..1.0, or auto to detect wpctl/pactl/amixer "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--mic-type",
        choices=_MIC_TYPES,
        default="unknown",
        help="Microphone used by the acoustic probe/detector (default: unknown)",
    )
    parser.add_argument(
        "--player-type",
        choices=_PLAYER_TYPES,
        default="unknown",
        help="Playback device used as the speaker source (default: unknown)",
    )
    parser.add_argument(
        "--probe-distance-m",
        type=float,
        default=1.0,
        help=(
            "Distance from speaker to the probe microphone during acoustic calibration "
            "(default: 1.0)"
        ),
    )
    parser.add_argument(
        "--placement-help",
        action="store_true",
        help="Print microphone/speaker placement guidance and exit",
    )
    parser.add_argument(
        "--sample",
        help="Path to a drone WAV file.  Default: first file found in DRONE_AUDIO_DATA_DIR",
    )
    parser.add_argument(
        "--output",
        help="Save simulated audio to this WAV file instead of playing",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.DRONE_AUDIO_DATA_DIR,
        help="Dataset cache directory (default: DRONE_AUDIO_DATA_DIR from settings)",
    )
    args = parser.parse_args()
    try:
        system_volume, volume_backend = _resolve_system_volume(args.system_volume)
    except ValueError as exc:
        parser.error(str(exc))
    if args.probe_distance_m <= 0.0:
        parser.error("--probe-distance-m must be greater than 0")
    effective_speaker_ref_db = _speaker_ref_at_probe_distance(
        args.speaker_ref_db,
        args.probe_distance_m,
    )
    if args.placement_help:
        _print_placement_guidance(args.mic_type, args.player_type, args.probe_distance_m)
        return

    # Load source audio
    source_path = Path(args.sample) if args.sample else _find_drone_wav(Path(args.data_dir))
    if source_path and source_path.exists():
        print(f"Source audio: {source_path}")
        source = _load_wav_mono(source_path, _SR)
    else:
        print("No drone WAV found — using synthetic tone (run split_drone_audio_data.sh first)")
        source = None

    if source is None or len(source) == 0:
        source = _synthesise_drone(5.0, _SR)
    source = _normalise_source(source)

    _print_summary(
        args.scenario,
        args.distance,
        args.speed,
        args.source_db,
        args.c,
        args.atm_db,
        args.duration,
        args.speaker_ref_db,
        effective_speaker_ref_db,
        system_volume,
        volume_backend,
        args.mic_type,
        args.player_type,
        args.probe_distance_m,
    )
    _print_placement_guidance(args.mic_type, args.player_type, args.probe_distance_m)

    if args.scenario == "flyover":
        out, max_received_db, unclipped_peak = _scenario_flyover(
            source,
            _SR,
            args.source_db,
            args.c,
            args.atm_db,
            args.distance,
            args.speed,
            args.duration,
            effective_speaker_ref_db,
            system_volume,
        )
    elif args.scenario == "approach":
        out, max_received_db, unclipped_peak = _scenario_approach(
            source,
            _SR,
            args.source_db,
            args.c,
            args.atm_db,
            args.distance,
            args.speed,
            args.duration,
            effective_speaker_ref_db,
            system_volume,
        )
    elif args.scenario == "hover":
        out, max_received_db, unclipped_peak = _scenario_hover(
            source,
            _SR,
            args.source_db,
            args.c,
            args.atm_db,
            args.distance,
            args.duration,
            effective_speaker_ref_db,
            system_volume,
        )
    else:  # circle
        out, max_received_db, unclipped_peak = _scenario_circle(
            source,
            _SR,
            args.source_db,
            args.c,
            args.atm_db,
            args.distance,
            args.speed,
            args.duration,
            effective_speaker_ref_db,
            system_volume,
        )

    _print_playback_stats(max_received_db, unclipped_peak)

    if args.output:
        _save_wav(Path(args.output), out, _SR)
    else:
        _play(out, _SR)


if __name__ == "__main__":
    main()

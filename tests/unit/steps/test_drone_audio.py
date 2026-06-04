import numpy as np
import pytest
from types import SimpleNamespace


def test_label_name_to_dir_handles_no_drone_before_drone():
    from ssv_vdp.steps.edge.drone_audio import _label_name_to_dir, _label_to_dir

    assert _label_name_to_dir("no_drone") == "no_drone"
    assert _label_name_to_dir("non-drone background") == "no_drone"
    assert _label_name_to_dir("drone") == "drone"
    assert _label_to_dir(0, ["no_drone", "drone"]) == "no_drone"
    assert _label_to_dir(1, ["no_drone", "drone"]) == "drone"


def test_drone_audio_training_skips_one_class_dataset(tmp_path, monkeypatch):
    import numpy as np
    from scipy.io import wavfile

    from selfsuvis.pipeline.core.config import settings
    from ssv_vdp.steps.edge import drone_audio

    data_dir = tmp_path / "audio-data"
    train_no_drone = data_dir / "train" / "no_drone"
    val_no_drone = data_dir / "val" / "no_drone"
    train_no_drone.mkdir(parents=True)
    val_no_drone.mkdir(parents=True)
    silence = np.zeros(drone_audio._SR, dtype=np.int16)
    wavfile.write(train_no_drone / "a.wav", drone_audio._SR, silence)
    wavfile.write(val_no_drone / "b.wav", drone_audio._SR, silence)

    monkeypatch.setattr(settings, "DRONE_AUDIO_DATA_DIR", str(data_dir), raising=False)
    monkeypatch.setattr(drone_audio, "_download_hf_dataset", lambda cache_dir: False)

    result = drone_audio.step_drone_audio_training(
        tmp_path / "run",
        tmp_path,
        "cpu",
        SimpleNamespace(drone_audio_epochs=1),
    )

    assert result["skipped"] is True
    assert "both drone and no_drone" in result["error"]
    assert not (tmp_path / "run" / "drone_audio" / "drone_audio_cnn.onnx").exists()
    assert (tmp_path / "run" / "drone_audio" / "drone_audio_report.md").exists()


# ---------------------------------------------------------------------------
# MFCC computation: output shape contract required by the DroneAudioCNN input
# ---------------------------------------------------------------------------


def test_compute_mfcc_output_shape_is_always_fixed():
    from ssv_vdp.steps.edge.drone_audio import _N_MFCC, _SR, _T_FRAMES, _compute_mfcc

    wave = np.random.randn(_SR).astype(np.float32)
    mfcc = _compute_mfcc(wave)

    assert mfcc.shape == (_N_MFCC, _T_FRAMES), (
        f"DroneAudioCNN expects ({_N_MFCC}, {_T_FRAMES}), got {mfcc.shape}"
    )
    assert mfcc.dtype == np.float32


def test_compute_mfcc_short_waveform_padded_to_fixed_shape():
    from ssv_vdp.steps.edge.drone_audio import _N_MFCC, _T_FRAMES, _compute_mfcc

    wave = np.zeros(100, dtype=np.float32)  # much shorter than 1 second
    mfcc = _compute_mfcc(wave)

    assert mfcc.shape == (_N_MFCC, _T_FRAMES)


def test_compute_mfcc_long_waveform_truncated_to_fixed_shape():
    from ssv_vdp.steps.edge.drone_audio import _N_MFCC, _SR, _T_FRAMES, _compute_mfcc

    wave = np.random.randn(3 * _SR).astype(np.float32)  # 3 seconds — must be truncated
    mfcc = _compute_mfcc(wave)

    assert mfcc.shape == (_N_MFCC, _T_FRAMES)


# ---------------------------------------------------------------------------
# WAV loading: dtype normalisation
# ---------------------------------------------------------------------------


def test_load_wav_mono_int16_normalised_to_float32(tmp_path):
    from scipy.io import wavfile

    from ssv_vdp.steps.edge.drone_audio import _SR, _load_wav_mono

    # Write a 1-second WAV at maximum int16 amplitude.
    pcm = np.full(_SR, 32767, dtype=np.int16)
    path = tmp_path / "test.wav"
    wavfile.write(str(path), _SR, pcm)

    wave = _load_wav_mono(path)

    assert wave is not None
    assert wave.dtype == np.float32
    assert np.allclose(wave, 1.0, atol=1e-4), "int16 max should normalise to ~1.0"


# ---------------------------------------------------------------------------
# _collect_split: label assignment
# ---------------------------------------------------------------------------


def test_collect_split_assigns_drone_label_1_and_no_drone_label_0(tmp_path):
    from scipy.io import wavfile

    from ssv_vdp.steps.edge.drone_audio import _SR, _collect_split

    drone_dir = tmp_path / "drone"
    no_drone_dir = tmp_path / "no_drone"
    drone_dir.mkdir()
    no_drone_dir.mkdir()

    silence = np.zeros(100, dtype=np.int16)
    wavfile.write(str(drone_dir / "d1.wav"), _SR, silence)
    wavfile.write(str(no_drone_dir / "n1.wav"), _SR, silence)

    items = _collect_split(tmp_path)

    label_map = {p.stem: lbl for p, lbl in items}
    assert label_map["d1"] == 1
    assert label_map["n1"] == 0


# ---------------------------------------------------------------------------
# val split fallback: when val/ is missing, 20% of train is split off.
# _collect_split yields drone items before no_drone (class-ordered), so the
# last 20% of a balanced 5+5 set ends up containing only no_drone items.
# The class-balance guard then correctly rejects training.
# ---------------------------------------------------------------------------


def test_val_split_fallback_rejects_when_split_produces_one_class_val(tmp_path, monkeypatch):
    from scipy.io import wavfile

    from selfsuvis.pipeline.core.config import settings
    from ssv_vdp.steps.edge import drone_audio

    data_dir = tmp_path / "audio-data"
    for subdir in ("drone", "no_drone"):
        d = data_dir / "train" / subdir
        d.mkdir(parents=True)
        silence = np.zeros(drone_audio._SR, dtype=np.int16)
        for i in range(5):
            wavfile.write(str(d / f"{subdir}_{i}.wav"), drone_audio._SR, silence)
    # Deliberately no val/ directory

    monkeypatch.setattr(settings, "DRONE_AUDIO_DATA_DIR", str(data_dir), raising=False)
    monkeypatch.setattr(drone_audio, "_download_hf_dataset", lambda cache_dir: False)

    result = drone_audio.step_drone_audio_training(
        tmp_path / "run", tmp_path, "cpu", SimpleNamespace(drone_audio_epochs=1),
    )

    # The 80/20 split puts drone first and no_drone last, so the 20% tail
    # contains only no_drone — the balance guard correctly rejects training.
    assert result["skipped"] is True
    assert "both drone and no_drone" in result["error"]
    assert (tmp_path / "run" / "drone_audio" / "drone_audio_report.md").exists()

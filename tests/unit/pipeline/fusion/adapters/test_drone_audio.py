"""Unit tests for DroneAudioAdapter."""

import asyncio
from unittest.mock import MagicMock, patch


def test_adapter_disabled_when_model_path_unset():
    with patch("selfsuvis.pipeline.core.settings") as mock_settings:
        mock_settings.DRONE_AUDIO_MODEL_PATH = ""
        mock_settings.DRONE_AUDIO_WATCH_DIR = "/some/dir"

        from selfsuvis.pipeline.fusion.adapters.drone_audio import DroneAudioAdapter

        adapter = DroneAudioAdapter()
        adapter._model_path = ""
        adapter.enabled = bool(adapter._model_path and adapter._watch_dir)
        assert not adapter.enabled


def test_adapter_disabled_when_watch_dir_unset():
    with patch("selfsuvis.pipeline.core.settings") as mock_settings:
        mock_settings.DRONE_AUDIO_MODEL_PATH = "/some/model.onnx"
        mock_settings.DRONE_AUDIO_WATCH_DIR = ""

        from selfsuvis.pipeline.fusion.adapters.drone_audio import DroneAudioAdapter

        adapter = DroneAudioAdapter()
        adapter._watch_dir = ""
        adapter.enabled = bool(adapter._model_path and adapter._watch_dir)
        assert not adapter.enabled


async def test_start_returns_immediately_when_disabled():
    from selfsuvis.pipeline.fusion.adapters.drone_audio import DroneAudioAdapter

    adapter = DroneAudioAdapter()
    adapter.enabled = False
    # Should complete without hanging
    await asyncio.wait_for(adapter.start(), timeout=1.0)


async def test_processed_subdir_created(tmp_path):
    """After processing, file should be in processed/ subdir."""
    from unittest.mock import AsyncMock

    from selfsuvis.pipeline.fusion.adapters.drone_audio import DroneAudioAdapter

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF")  # Minimal fake WAV

    adapter = DroneAudioAdapter()
    adapter.enabled = True
    adapter._model_path = "fake.onnx"
    adapter._watch_dir = str(tmp_path)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch.object(adapter, "_run_inference", return_value=0.9):
        with patch.object(adapter, "_record_event"):
            await adapter._process_file(wav, mock_client)

    processed = tmp_path / "processed" / "test.wav"
    assert processed.exists()
    assert not wav.exists()

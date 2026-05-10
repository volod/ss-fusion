"""DroneAudioAdapter — runs the SV-21 ONNX drone detection model on WAV files.

Configuration (both env vars must be set; adapter is disabled if either is empty):
    DRONE_AUDIO_MODEL_PATH — absolute path to drone_audio_cnn.onnx produced by
        the SV-21 training pipeline (step_drone_audio_training). This file is NOT
        shipped in the repo; run training first to produce it.
    DRONE_AUDIO_WATCH_DIR — absolute path to directory polled every 5 s for .wav
        files. After processing, each file is moved to a `processed/` subdirectory
        to prevent re-processing on the next poll.

Workflow:
    1. Poll DRONE_AUDIO_WATCH_DIR every 5 s for *.wav files.
    2. Load each WAV, run ONNX inference.
    3. If drone confidence >= 0.5, POST an `audio` EventEnvelope to the ingest API.
    4. Move processed file to DRONE_AUDIO_WATCH_DIR/processed/.
"""

import asyncio
import shutil
from pathlib import Path

import httpx

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.fusion.adapters.base import SensorAdapter
from selfsuvis.pipeline.fusion.adapters.registry import registry

logger = get_logger(__name__)

_POLL_INTERVAL_S = 5.0
_CONFIDENCE_THRESHOLD = 0.5
_INGEST_URL = "http://localhost:8000/api/v1/events/audio"


class DroneAudioAdapter(SensorAdapter):
    """Polls a watch directory for WAV files and runs ONNX drone audio detection.

    Disabled when DRONE_AUDIO_MODEL_PATH or DRONE_AUDIO_WATCH_DIR is not set.
    The ONNX model is the training output from the SV-21 pipeline; it does not
    exist in the repo by default.
    """

    modality = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._model_path = settings.DRONE_AUDIO_MODEL_PATH
        self._watch_dir = settings.DRONE_AUDIO_WATCH_DIR
        self.enabled = bool(self._model_path and self._watch_dir)
        self._session = None

    def _load_model(self):
        import onnxruntime as ort

        self._session = ort.InferenceSession(self._model_path)
        logger.info("DroneAudioAdapter: loaded ONNX model from %s", self._model_path)

    def _run_inference(self, wav_path: Path) -> float:
        """Return drone confidence score (0.0-1.0) for the given WAV file."""
        import numpy as np

        try:
            import soundfile as sf

            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:
            import scipy.io.wavfile as wav_io

            sr, audio = wav_io.read(str(wav_path))
            audio = audio.astype("float32") / 32768.0

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_name = self._session.get_inputs()[0].name
        result = self._session.run(None, {input_name: audio[np.newaxis, :]})[0]
        return float(result.flat[0])

    async def _process_file(self, wav_path: Path, client: httpx.AsyncClient) -> None:
        processed_dir = wav_path.parent / "processed"
        processed_dir.mkdir(exist_ok=True)

        try:
            confidence = await asyncio.get_event_loop().run_in_executor(
                None, self._run_inference, wav_path
            )
            logger.debug("DroneAudioAdapter: %s confidence=%.3f", wav_path.name, confidence)

            if confidence >= _CONFIDENCE_THRESHOLD:
                from datetime import datetime, timezone

                payload = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "zone_id": "unknown",
                    "sensor_id": "drone_audio",
                    "confidence": confidence,
                    "payload": {"source_file": wav_path.name},
                }
                try:
                    api_key = settings.API_KEY
                    headers = {"X-Api-Key": api_key} if api_key else {}
                    await client.post(_INGEST_URL, json=payload, headers=headers)
                    self._record_event()
                except Exception as exc:
                    logger.warning("DroneAudioAdapter: ingest POST failed: %s", exc)
        finally:
            dest = processed_dir / wav_path.name
            try:
                shutil.move(str(wav_path), str(dest))
            except Exception as exc:
                logger.warning("DroneAudioAdapter: could not move %s: %s", wav_path.name, exc)

    async def start(self) -> None:
        if not self.enabled:
            logger.debug(
                "DroneAudioAdapter disabled (DRONE_AUDIO_MODEL_PATH=%r, DRONE_AUDIO_WATCH_DIR=%r)",
                self._model_path,
                self._watch_dir,
            )
            return

        try:
            self._load_model()
        except Exception as exc:
            logger.error("DroneAudioAdapter: model load failed, adapter disabled: %s", exc)
            self.enabled = False
            return

        watch = Path(self._watch_dir)
        logger.info("DroneAudioAdapter: polling %s every %.0fs", watch, _POLL_INTERVAL_S)

        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    wav_files = sorted(watch.glob("*.wav"))
                    for wav_path in wav_files:
                        await self._process_file(wav_path, client)
                except Exception as exc:
                    logger.error("DroneAudioAdapter: poll error: %s", exc)
                await asyncio.sleep(_POLL_INTERVAL_S)


registry.register("drone_audio", DroneAudioAdapter())

# Whisper ASR Runbook

> Covers: enabling audio transcription, GPU-aware auto-selection,
> subtitle injection into Qwen context, and language detection.

---

## 1. Architecture overview

```
VideoIndexer
  └─ ASR pass (once per video, before captioning)
       ├─ ffmpeg: extract audio → 16 kHz mono WAV (ASR_AUDIO_DIR)
       ├─ WhisperModel.transcribe()      ← loaded in worker VRAM
       │    chunked: ASR_CHUNK_LENGTH_SEC=30
       └─ subtitle_text mapped to frame timestamps (±ASR_SUBTITLE_WINDOW_SEC=3s)
            → frames.subtitle_text column
            → injected into Qwen prompt as "[Audio context at this moment]: ..."
```

ASR is **disabled by default** (`ASR_ENABLED=false`). Enable when videos have
informative audio (voice narration, radio comms, ATC, operator callouts).

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `ASR_ENABLED` | `false` | Enable ASR pass |
| `ASR_MODEL` | `auto` | Model ID or `auto` for GPU-aware selection |
| `ASR_LANGUAGE` | `""` | ISO-639-1 language code (`en`, `fr`, etc.) or blank for auto-detect |
| `ASR_BATCH_SIZE` | `8` | Audio chunks processed in parallel |
| `ASR_CHUNK_LENGTH_SEC` | `30` | Seconds of audio per Whisper chunk |
| `ASR_SUBTITLE_WINDOW_SEC` | `3.0` | ±seconds when matching subtitle segments to frames |
| `ASR_AUDIO_DIR` | `.data/audio` | Directory for extracted WAV files |

---

## 3. GPU-aware auto-selection

When `ASR_MODEL=auto`, the pipeline reads available VRAM and selects:

| VRAM available | Selected model | Speed |
|---|---|---|
| < 2 GB or CPU | `openai/whisper-tiny` | ~32× real-time |
| 2–3 GB | `openai/whisper-base` | ~16× real-time |
| 3–5 GB | `openai/whisper-small` | ~6× real-time |
| 5–8 GB | `distil-whisper/distil-large-v3` | ~6× real-time (best speed/quality) |
| 8–12 GB | `openai/whisper-large-v3-turbo` | ~8× real-time (pruned decoder) |
| > 12 GB | `openai/whisper-large-v3` | Best accuracy |

---

## 4. Quick start

```bash
# Enable with auto model selection
ASR_ENABLED=true ssv --mode local

# Explicit model + language
ASR_ENABLED=true ASR_MODEL=openai/whisper-large-v3-turbo ASR_LANGUAGE=en ssv --mode local

# Download weights
python -m selfsuvis.scripts.prepare_models --asr
python -m selfsuvis.scripts.prepare_models --asr --asr-model openai/whisper-large-v3-turbo
```

---

## 5. Health check

```bash
# Test transcription on a short WAV
python -c "
import os; os.environ['ASR_ENABLED']='true'
from pipeline.vision.asr import WhisperASR
m = WhisperASR()
print('Backend:', m._model_id if hasattr(m,'_model_id') else 'loaded')
"
```

---

## 6. Verifying subtitle injection

After a run, check that subtitles were injected:

```bash
# Look for subtitle_text in frame output
grep -r "subtitle_text" output/<video>/ | head -5
grep "Audio context" output/<video>/detailed_captions.md | head -3
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ASR pass skipped` | `ASR_ENABLED=false` | Set `ASR_ENABLED=true` |
| Empty `subtitle_text` for all frames | Video has no audio track | Expected; ASR silently skips |
| Transcription is wrong language | Auto-detect failed | Set `ASR_LANGUAGE=en` explicitly |
| `CUDA out of memory` | Whisper too large for available VRAM | Use `ASR_MODEL=auto` or set smaller model explicitly |
| Very slow: >10 min for 1h video | CPU inference | Ensure `DEVICE=cuda`; Whisper-large on CPU takes ~3× real-time |
| Audio extraction fails | ffmpeg not on PATH | Install ffmpeg: `apt install ffmpeg` |

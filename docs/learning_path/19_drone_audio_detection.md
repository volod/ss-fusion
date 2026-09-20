# 19 — Drone Audio Detection: DroneAudioCNN

**Pipeline step:** Step 32 of the local runner  
**Source:** `src/ssv_vdp/steps/drone_audio.py`  
**Dataset:** `geronimobasso/drone-audio-detection-samples` (HuggingFace, public)  
**Dataset prep:** `scripts/split_drone_audio_data.sh` → `.data/drone-audio-data/`  
**Outputs:** `drone_audio/drone_audio_cnn.pt`, `drone_audio/drone_audio_cnn.onnx`, `drone_audio/drone_audio_report.md`

---

## What this step does

Step 32 trains a small binary CNN classifier — **DroneAudioCNN** — to tell apart "drone present"
from "no drone" using one-second audio chunks. The trained ONNX model is a lightweight
inference artifact that can run on CPU at the edge, alongside the YOLO and RF-DETR visual
detectors, to provide an independent acoustic evidence channel.

The full sequence:

1. Locate the split dataset under `.data/drone-audio-data/` (or download it on first run)
2. Compute **MFCC features** from each 1-second WAV chunk (40 coefficients × 44 frames)
3. Train **DroneAudioCNN** for up to `DRONE_AUDIO_EPOCHS` epochs (default: 10)
4. Evaluate on the validation split; log F1 / precision / recall per epoch
5. Export the trained model to **ONNX** (opset 14, dynamic batch axis)
6. Write a **markdown training report** with architecture summary, dataset stats, per-epoch metrics, and ONNX usage snippet

---

## Why MFCC — and why without librosa

**Mel-Frequency Cepstral Coefficients** compress the perceptually relevant frequency content of an
audio signal into a compact fixed-size feature map. For drone detection the acoustic signature is
dominated by rotor blade-pass frequency harmonics (80–200 Hz fundamental, overtones up to ~4 kHz),
which MFCC captures naturally because the mel scale emphasises low frequencies where rotor tones sit.

This codebase computes MFCC **from scratch** using only `scipy` and `numpy`, avoiding `librosa` as
a hard dependency:

1. **STFT** via `scipy.signal.stft` — converts the 1-second waveform to a complex spectrogram
2. **Mel filterbank** — a hand-built matrix that maps FFT frequency bins to 40 mel-spaced bands
   using the Hz ↔ mel conversion: `m = 2595 · log₁₀(1 + f/700)`
3. **Log energy** — `log(filterbank @ |STFT|² + ε)` gives log-mel spectrogram
4. **DCT-II** via `scipy.fft.dct` — decorrelates the log-mel features and yields the final MFCC

Output shape: `(40, 44)` per chunk (40 coefficients × 44 time frames for 1 s at 22050 Hz with a
512-sample hop). Frames are zero-padded or truncated to exactly 44 columns so the ONNX export has
a fixed spatial dimension with a dynamic batch axis.

---

## DroneAudioCNN architecture

```
Input: (batch, 1, 40, 44)          # 1-channel MFCC image

Conv2d(1→16, 3×3, pad=1) → BN → ReLU → MaxPool2d(2)   → (batch, 16, 20, 22)
Conv2d(16→32, 3×3, pad=1) → BN → ReLU → MaxPool2d(2)  → (batch, 32, 10, 11)
Conv2d(32→64, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool2d(1) → (batch, 64, 1, 1)
Flatten → Linear(64, 2)            # logits for [no_drone, drone]
```

Total parameters: ≈ 52,000. Fits on any microcontroller class system that can run ONNX Runtime.

Training uses:
- **Optimiser:** Adam (lr = 1e-3)
- **LR schedule:** CosineAnnealingLR over `epochs`
- **Loss:** CrossEntropyLoss
- **Validation metrics:** accuracy, F1, precision, recall per epoch

---

## Dataset layout

After `split_drone_audio_data.sh` (or `ssv-prepare-audio`):

```
.data/drone-audio-data/
  train/
    drone/        *.wav   (≈75% of drone samples)
    no_drone/     *.wav   (≈75% of background samples)
  val/
    drone/        *.wav   (≈15%)
    no_drone/     *.wav
  test/
    drone/        *.wav   (≈10%)
    no_drone/     *.wav
```

The split is stratified per class — each class is shuffled independently, so class balance is
preserved across all three splits regardless of dataset size.

---

## Inference with the ONNX model

```python
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import stft, resample
from scipy.fft import dct

SR = 22050
N_MFCC = 40
N_FFT = 1024
HOP = 512
N_MELS = 40
T_FRAMES = 44
CHUNK = SR  # 1 second


def load_wav(path):
    rate, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data /= 32768.0
    if rate != SR:
        n = int(len(data) * SR / rate)
        data = resample(data, n)
    return data


def compute_mfcc(wave):
    _, _, Zxx = stft(wave[:CHUNK], fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    power = np.abs(Zxx) ** 2
    # Mel filterbank (simplified — see steps_drone_audio.py for full implementation)
    mel_power = mel_filterbank @ power  # shape: (N_MELS, T)
    log_mel = np.log(mel_power + 1e-8)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    if mfcc.shape[1] < T_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, T_FRAMES - mfcc.shape[1])))
    return mfcc[:, :T_FRAMES].astype(np.float32)


sess = ort.InferenceSession(".data/local_runs/drone_mission/drone_audio/drone_audio_cnn.onnx")
wave = load_wav("audio_chunk.wav")
mfcc = compute_mfcc(wave)[None, None, :, :]  # (1, 1, 40, 44)
logits = sess.run(["logits"], {"mfcc": mfcc})[0]
drone_prob = float(np.exp(logits[0, 1]) / np.exp(logits[0]).sum())
print(f"drone probability: {drone_prob:.3f}")
```

The `drone_audio_report.md` artifact (written after training) contains a self-contained version
of this snippet with the actual ONNX path.

---

## Physics-based audio simulation

The `play_drone_sound.sh` script / `ssv-play-drone` entry point applies three physical
effects to a drone audio sample (or a synthetic rotor tone) to simulate what the microphone
would hear at a given distance, approach angle, and speed:

| Effect | Formula |
|---|---|
| Inverse-square law amplitude | `SPL(d) = source_db − 20·log₁₀(d/r_ref)` where `r_ref = 1 m` |
| Atmospheric absorption | `A_atm = 10^(−α·d/100/20)` where `α` = dB/100 m (default 0.5) |
| Doppler pitch shift | Emission-time iteration: `t_em = t_obs − d(t_em)/c`, 5 rounds, then `np.interp` resampling |
| Playback calibration | Output dBFS is estimated from target SPL, `--speaker-ref-db`, current `--system-volume`, and probe mic placement |

**Scenarios:**

| Scenario | Description |
|---|---|
| `flyover` | Drone flies a straight lateral path; closest approach = `--distance` |
| `approach` | Drone approaches head-on from `--distance`, passes at 5 m, recedes |
| `hover` | Drone hovers at fixed altitude `--distance` above the microphone |
| `circle` | Drone circles the microphone at orbit radius `--distance` |

```bash
# Simulate and play a 200 m flyover at 10 m/s:
./scripts/play_drone_sound.sh --scenario flyover --distance 200 --speed 10

# Save to WAV for model testing:
./scripts/play_drone_sound.sh --scenario approach --distance 300 --speed 15 \
  --output .data/reports/test_approach_300m.wav

# Hover at 50 m for 20 seconds:
./scripts/play_drone_sound.sh --scenario hover --distance 50 --duration 20

# Calibrate for a quieter speaker or override the auto-detected OS volume:
./scripts/play_drone_sound.sh --scenario flyover --distance 80 \
  --speaker-ref-db 78 --system-volume 0.6

# Get setup guidance for a headset mic 30 cm from a laptop speaker:
./scripts/play_drone_sound.sh --placement-help \
  --mic-type headset --player-type laptop --probe-distance-m 0.3
```

The simulation prints a dB summary table showing received SPL at 0.5×, 1×, and 2× the
specified distance, plus the detected output volume and resulting playback headroom. The
speaker calibration is only an estimate unless measured with an SPL meter, but unlike peak
normalisation it preserves the expected loudness drop as the drone distance increases.

For acoustic checks, answer this setup question first: which microphone will record the
emulated drone sound, and where will it be during the test? Use `--mic-type`, `--player-type`,
and `--probe-distance-m` to make that geometry explicit. A measurement or external acoustic
mic should normally be 1 m on-axis from the speaker. An embedded mic should be tested with the
whole device in its real detector position. A headset boom mic at 30 cm is valid only when that
close speaker-to-mic layout is the intended detector geometry; otherwise move the capsule to the
same on-axis reference position as the other microphones.

---

## What to inspect after the step runs

```bash
# Training report with architecture, metrics table, and ONNX usage snippet:
cat .data/local_runs/drone_mission/drone_audio/drone_audio_report.md

# Verify ONNX model is loadable and has correct input/output shapes:
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession(
    '.data/local_runs/drone_mission/drone_audio/drone_audio_cnn.onnx')
for inp in sess.get_inputs():
    print('input :', inp.name, inp.shape, inp.type)
for out in sess.get_outputs():
    print('output:', out.name, out.shape, out.type)
"
# Expected:
#   input : mfcc ['batch', 1, 40, 44] tensor(float)
#   output: logits ['batch', 2] tensor(float)
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Step 32 skipped: data dir empty` | Dataset not prepared | Run `ssv-prepare-audio` first |
| `datasets package not found` | HF client not installed | `uv pip install "datasets>=2.18"` |
| `soundfile not found` | WAV writer missing | `uv pip install "soundfile>=0.12"` |
| Very low val F1 after 10 epochs | Too few epochs or imbalanced split | Increase `--drone-audio-epochs` or check split ratios |
| ONNX export fails | `torch` not importable | Training requires PyTorch; Step 32 is skipped on CPU-only envs without torch |

---

## Connections to other steps

- **Step 19 (acoustic):** reads `.audio.wav` sidecar for acoustic feature extraction.
  DroneAudioCNN targets the same input format — a live or recorded WAV stream — but is a
  **trained classifier** rather than a hand-tuned feature extractor.
- **Step 30 (YOLO drone detection):** trains a YOLOv8n visual drone detector. DroneAudioCNN
  provides an orthogonal acoustic evidence channel that complements the visual one.
- **Step 32 ONNX → edge deployment:** the exported `drone_audio_cnn.onnx` targets the same
  Cortex-A76 / RV1106G3 edge stack as the YOLO fp32 / int8 ONNX exports from Step 30.
- **coop Step 19 acoustic analysis:** the acoustic MFCC features computed here share the
  same mel-filterbank mathematics as the beamforming and GCC-PHAT bearing estimation in the
  coop acoustic sensor stack.

---

## Further reading

- [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) — drone detection training, ONNX export, and edge deployment context
- [17_essential_technology_stack.md](17_essential_technology_stack.md) — ONNX Runtime edge inference, RKNN NPU conversion
- [Drone detection runbook](../runbooks/drone-detection.md) — operational runbook for the visual detector (YOLOv8n, hard negatives, int8 calibration)
- `src/ssv_vdp/steps/drone_audio.py` — full implementation
- `src/ssv_vdp/scripts/split_drone_audio_data.py` — dataset download and splitting
- `src/ssv_vdp/scripts/play_drone_sound.py` — physics simulation engine

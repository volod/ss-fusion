# 20 -- drau Range-Detection Evaluation

**Pipeline step:** Step 33 of the local runner
**Source:** `src/ssv_vdp/steps/drau_eval.py`
**Dataset:** `geronimobasso/drone-audio-detection-samples` (HuggingFace, public; reused from Step 32)
**Physics reference:** `github.com/volod/drau` (inverse-square + ISO 9613-1 atmospheric absorption)
**Outputs:**
- `drone_audio/drau_range_report.md` -- detection probability vs distance table
- `drone_audio/drau_edge_test.py` -- standalone edge inference script (no PyTorch)

---

## What this step does

Step 33 answers a question that Step 32 (DroneAudioCNN training) leaves open:

> At what acoustic distance does the trained model stop reliably detecting drones?

It does this by taking the ONNX model produced by Step 32 and evaluating it at simulated distances using the drau physics model from `github.com/volod/drau`. The drau project is an acoustic drone detection testing harness that plays real drone audio at calibrated distances through a speaker and records microphone responses. Step 33 applies the same physics simulation without hardware -- synthesising test signals in software.

---

## Physics model (ported from drau)

drau models two physical processes that degrade drone audio with distance:

### 1. Inverse-square amplitude scaling

In free-field propagation, sound pressure amplitude drops as 1/distance. The step normalises each synthetic signal to a reference level (-18 dBFS RMS at 1 m) and then applies:

```
A(d) = A(1m) / d
```

At 100 m this is a 40 dB drop. At 200 m it is 46 dB. Beyond 50-100 m the signal approaches the noise floor.

### 2. Atmospheric absorption (ISO 9613-1, 20 degC, 70% RH)

Air absorbs high-frequency sound. The absorption coefficient scales approximately as f^2 (frequency squared), meaning treble frequencies attenuate much faster than bass. ISO 9613-1 at typical outdoor conditions gives approximately:

| Frequency | Absorption |
|-----------|-----------|
| 500 Hz    | 0.8 dB/100 m |
| 2 kHz     | 1.9 dB/100 m |
| 4 kHz     | 4.0 dB/100 m |
| 8 kHz     | 9.2 dB/100 m |

Step 33 implements this as a first-order IIR lowpass filter whose cutoff frequency drops with distance:

```
f_3dB(d) = 2000 * sqrt(3 / (d * 0.019))   [Hz]
```

This means at 200 m, frequencies above ~1800 Hz are already 3 dB down before the inverse-distance loss is applied.

---

## Synthetic quadcopter signal

drau uses real recorded drone audio. Step 33 supplements real samples (if the dataset cache from Step 32 is available) with a synthetic quadcopter model:

- Fundamental frequency: 300 Hz (typical small quadcopter blade-pass)
- 7 harmonics (amplitudes decay as 1/k)
- Motor noise at -26 dB relative to fundamental
- Phase randomised per signal to average out coherent artefacts

Eight synthetic signals are generated per test distance to reduce variance. If real drone WAV files are available from the Step 32 dataset cache, up to 8 real samples per distance are also evaluated and averaged in.

---

## ONNX inference path (no PyTorch)

Step 33 uses `onnxruntime` directly for inference -- PyTorch is not required. This is intentional: edge devices that run the model at deployment time also need only `numpy`, `scipy`, and `onnxruntime`. The evaluation step demonstrates and validates this minimal-dependency path.

The MFCC computation is identical to Step 32 (same constants, same scipy STFT + mel filterbank + DCT-II pipeline). The model output is softmaxed to P(drone) in [0, 1].

---

## Range curve and report

`drau_range_report.md` contains:

1. A detection-probability vs distance table (ASCII bar chart)
2. The estimated detection range (largest distance at which P >= 0.5)
3. The per-distance f_3dB lowpass cutoff from the ISO 9613-1 model
4. Usage instructions for the edge test script

Example output:

```
| Distance (m) | P(drone) | Std | Signals | Bar                            |
|-------------|---------|-----|---------|--------------------------------|
|    1 | 0.972 | 0.011 | 8 | [#############################.] |
|    5 | 0.941 | 0.024 | 8 | [############################..] |
|   10 | 0.883 | 0.041 | 8 | [##########################.....] |
|   25 | 0.712 | 0.065 | 8 | [#####################.........] |
|   50 | 0.531 | 0.089 | 8 | [################..............] |
|   75 | 0.381 | 0.093 | 8 | [###########....................] |
|  100 | 0.241 | 0.078 | 8 | [#######........................] |
|  150 | 0.112 | 0.044 | 8 | [###............................] |
|  200 | 0.058 | 0.029 | 8 | [#.............................] |

Estimated detection range (P >= 0.50): 50 m
```

The detection range from the synthetic test is a lower bound on real-world performance -- the synthetic signal lacks the acoustic complexity (Doppler shift, turbulence, motor modulation) that real drone audio provides, and the simple IIR absorption model is a first-order approximation. Real detection range depends on background noise level, microphone sensitivity, and drone type.

---

## Edge test script

`drau_edge_test.py` is generated alongside the ONNX model. It is fully standalone:

- No PyTorch required
- No selfsuvis imports
- Deps: `numpy scipy onnxruntime`

Copy both files to any Arm/x86 edge device:

```bash
# Install minimal deps on edge device
pip install numpy scipy onnxruntime

# Single file test
python drau_edge_test.py drone_audio_cnn.onnx recording.wav

# With distance simulation
python drau_edge_test.py drone_audio_cnn.onnx recording.wav --distance 50

# Expected output:
DRONE      confidence=0.872  [##########################....]
```

The `ssv_vdp.scripts.drone_audio_edge_infer` module provides the same functionality with an additional `--scan` mode that prints the full range table for a given WAV file:

```bash
scripts/audio/drone_audio_edge_test.sh drone_audio_cnn.onnx recording.wav --scan
```

---

## drau hardware testing

The full drau toolkit (`github.com/volod/drau`) adds physical speaker and microphone testing on top of the software-only evaluation in Step 33. To test a real microphone array:

```bash
# Clone drau and install
git clone https://github.com/volod/drau
cd drau && make venv && make cache-data

# Run a calibrated detection session against microphones under test
make run-session SAMPLES=20 MICS=2 DIST_MAX=150
```

drau plays drone audio through a speaker at simulated distances (applying the same inverse-square physics as Step 33) and records which microphone systems detect it. The output is a CSV of detection results by distance and sound class, which drau's `make analyse` converts to accuracy, precision, recall, F1, and per-distance grading plots.

This is the intended workflow for validating the ONNX model's real-world detection range before deploying it to edge hardware.

---

## Step dependencies

| Dependency | Required | Notes |
|------------|----------|-------|
| Step 32 | Yes | Must produce `drone_audio_cnn.onnx`; Step 33 skips gracefully if absent |
| `onnxruntime` | Yes | Required for inference; skips if unavailable |
| `scipy` | Yes | Required for STFT and butterworth filter |
| PyTorch | No | Not needed at inference time |
| drau repo | No | Only needed for real microphone hardware testing |

---

## Related

- [Step 32 deep dive: DroneAudioCNN](19_drone_audio_detection.md)
- [Drone detection runbook](../runbooks/drone-detection.md) -- Step 30 YOLOv8n visual detection
- [Drone detection edge targets](../runbooks/drone-detection.md) -- ONNX fp32 / int8 / RKNN targets
- [Threat primitives and local inference](15_threat_primitives_local_inference.md) -- how acoustic detection feeds the threat layer
- `coop IoT edge monitoring` -- acoustic event detection in the live edge stack (SoundAnalyzer)

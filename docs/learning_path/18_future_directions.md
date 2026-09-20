# Future Directions — What Is Not Yet Implemented

This document covers the advanced themes that are **not yet implemented** in the
current codebase. It is the final stop on the learning path, after you understand
the current 32-step local runner, the fusion math, and the coop IoT layer.

Read this when you want to reason about where the system should go next, not when
you want to understand what it does today.

---

## What Is Already Done (And Where To Read About It)

Before engaging with the not-yet-implemented themes, confirm you understand what
already exists:

| Theme | Status | Deep-dive document |
|---|---|---|
| Track-aware SSL fine-tuning | Implemented | [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md) |
| Threat primitives with two-source gate | Implemented | [15_threat_primitives_local_inference.md](15_threat_primitives_local_inference.md) |
| Probabilistic Kalman/RTS/SfM fusion | Implemented | [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) |
| coop MQTT/LoRaWAN/Frigate realtime | Implemented | `16_coop_pilot_iot_edge_monitoring.md` |
| Knowledge distillation and ONNX edge export | Implemented | [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md) |

The themes below are genuinely open. They are not design-document placeholders —
they are unsolved engineering problems that require real thought before any code is
written.

---

## 1. Full Cross-Modal Temporal SSL

### What is implemented

The current SSL step uses two positive-pair sources:
- Track-based pairs: two crops of the same RF-DETR track ID across nearby frames
- Standard augmentation-based pairs (random crop, color jitter, flip)

Cycle-consistency prevents embedding drift along long tracks.

### What is not implemented

**Multi-modal positive pairs** from sources other than visual track identity:

- Two frames that describe the same GPS waypoint (within a radius) from different
  flight altitudes should have similar embeddings — the scene changes but the
  location is the same.
- A frame where the IMU reports near-zero angular rate (stable hover) should form a
  positive pair with adjacent frames at the same hover point — no visual change
  should produce no embedding change.
- A frame where the acoustic sidecar records a consistent tonal signature and an
  adjacent frame with the same signature should be positive — independent of
  anything visible.

The difficulty is that multi-modal positive pairs require calibrated alignment
between modalities. A GPS waypoint radius chosen too loosely produces false
positives (different scenes at the same GPS region); too tight and you get no
training signal. The same calibration problem applies to IMU angular rate
thresholds and acoustic signature similarity.

**Cross-view contrastive learning** from multiple cameras at the same site:
- A frame from camera A and a frame from camera B that observe the same object at
  the same time should have embeddings closer than frames from different objects.
- This requires synchronised timestamps and known camera-to-camera geometry.

### What to study before attempting this

- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning"
  (MoCo, CVPR 2020) — the negative-queue design used in the current SSL step.
  [arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722)
- Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision" (CLIP, ICML 2021) — multi-modal contrastive alignment.
  [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- Zhu et al., "UniVLP: A Unified Vision-Language Pre-Training Framework for Joint
  Visual Grounding and Image-Text Matching" (2021) — cross-modal alignment.
- The calibration section in [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md)
  before any multi-modal alignment work.

### Why it matters

The current SSL step makes the visual encoder better at identifying objects it has
seen before in this mission. Full cross-modal SSL would also make it better at
identifying *locations* and *conditions* — the same GPS region under different
lighting, the same acoustic environment at different times — which is what the
platform needs to reason about mission persistence and revisitation.

---

## 2. Environmental Field Models As Uncertain State

### What is implemented

The current physical-state layer tracks:
- Platform pose (position, velocity) via Kalman filter
- Per-object state (bounding box, velocity) via per-track Kalman filter
- Scene-level summaries (near-field density, visibility) from model outputs

### What is not implemented

**Environmental fields**: continuous spatial distributions of physical quantities
that are not point observations. Examples:

- **RF signal field**: the signal strength from a suspected emitter as a function of
  position. A single RF reading is a scalar; a field model is a 2D or 3D map that
  estimates the full spatial distribution of that signal, with uncertainty growing
  with distance from any measured point.
- **Gas concentration field**: a plume model that estimates where gas concentration
  exceeds a threshold across the area, given sparse sensor readings at known
  positions.
- **Acoustic pressure field**: a spatial map of ambient sound levels estimated from
  directional microphone readings.

Each of these is a state estimation problem: the unknown is a function over space
(or space-time), and measurements reduce uncertainty at observed locations while
uncertainty grows elsewhere over time as conditions change.

### Technical prerequisites

Environmental field estimation uses **Gaussian Process Regression** (GPR):
- The GP prior defines smoothness assumptions (how quickly the field changes with
  distance — the kernel function)
- Each sensor reading is a noisy observation of the field at a known location
- The GP posterior gives a mean field estimate and a per-location uncertainty

The output at each map cell is a Gaussian distribution over the field value.
Threshold exceedance probability (probability that concentration exceeds a safety
limit) can then be computed analytically from the posterior.

### What to study before attempting this

- Rasmussen and Williams, "Gaussian Processes for Machine Learning" (MIT Press,
  2006) — the standard reference. Chapters 1-3 are sufficient for field estimation.
  [gaussianprocess.org/gpml/](http://www.gaussianprocess.org/gpml/)
- Thrun, Burgard, Fox, "Probabilistic Robotics" (MIT Press, 2005) — Chapter 9 on
  occupancy grids provides the spatial uncertainty framework.
- `sklearn.gaussian_process` — the simplest Python implementation for 2D field
  estimation with no external dependencies.

### Why it matters

The current system can say "RF sensor detected signal at 2.4 GHz at position X".
With field models it can say "the estimated source is at position Y with 80%
confidence, and the signal-safe zone ends at radius R". That changes the quality
of the threat assessment from a point observation to a spatial risk estimate.

---

## 3. Calibration And Contradiction Handling

### What is implemented

The threat primitive layer explicitly tracks disagreement between evidence sources
and surfaces it in the operator view. The two-source gate rejects single-source
signals. The audit step flags synthesis contradictions.

### What is not implemented

**Systematic cross-sensor calibration** and **formal contradiction modeling**:

- **Temporal calibration**: when a camera and an IMU disagree about whether the
  platform is stationary, is it because the IMU clock drifts 50 ms from video
  timestamps? The current code logs a warning; it does not estimate and correct the
  clock offset from data.
- **Spatial calibration**: when a camera detects an object at bearing 15° and the
  RF sensor detects an emission that should come from bearing 25°, is the
  discrepancy within calibration uncertainty or is it evidence of two different
  sources? The current code cannot answer this without known camera-to-RF boresight
  alignment.
- **Formal contradiction modeling**: a structured representation of which sensor
  pairs disagree, by how much, and whether the disagreement is explained by a known
  calibration offset, a known sensor limitation (e.g. depth model never reports
  accurate scale), or a genuine environmental event.

### Technical prerequisites

- **Factor graphs and belief propagation**: the standard tool for multi-sensor
  fusion with explicit uncertainty and contradiction modeling. A factor graph
  connects sensor nodes to shared latent state nodes through factor nodes that
  encode the noise model of each measurement.
- **Chi-squared test for innovation**: the Kalman filter already produces an
  innovation (difference between predicted and measured value) at each update. A
  chi-squared test on the innovation detects when a measurement is inconsistent with
  the current state estimate. This is the formal version of what the Mahalanobis
  gate approximates.

### What to study before attempting this

- Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking and
  Navigation" (Wiley, 2001) — the formal treatment of chi-squared gating and
  multi-sensor consistency.
- The existing Mahalanobis gating in [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md) —
  understand what the gate already does before adding new consistency checks.

### Why it matters

Without calibration, contradictions between sensors are silent: one wins, one is
discarded, or they are averaged. With calibration and contradiction modeling, a
disagreement between camera and RF bearing is a signal that either needs resolution
(one sensor is wrong) or is itself informative (two independent emitters). The
current system cannot distinguish these cases.

---

## 4. Global Threat Inference Across Missions

### What is implemented

Each video run produces a local threat assessment: primitives scoped to the current
clip, a clip-level threat level, and a fixed-vocabulary action recommendation. The
coop layer produces a site-level threat summary from live sensor feeds.

### What is not implemented

**Cross-mission threat persistence and global inference**:

- A track anomaly in mission A and a similar track anomaly in mission B at the same
  GPS sector two days later are two independent events. Neither the local pipeline
  nor coop currently links them into a cross-mission pattern.
- Global threat inference would maintain a persistent threat map across missions:
  for each spatial sector, a history of observed threat primitive types, scores, and
  timestamps. A new mission that activates the same primitive in the same sector
  increases the sector's long-term threat estimate; a clean mission decreases it.
- This requires defining a temporal decay function (how quickly old evidence loses
  weight) and a spatial binning scheme (how finely to discretize the site into
  sectors for persistent tracking).

### Technical prerequisites

- Bayesian filter over discrete states: treating sector threat level as a hidden
  variable with transitions governed by prior decay and updates from new mission
  evidence.
- The coop rolling site state (already implemented) is the closest existing
  analogue. The difference is time horizon: rolling state covers minutes; global
  threat maps cover days or weeks.
- The Qdrant vector database can serve as the backing store for mission-indexed
  threat primitives, enabling semantic search over historical threats.

### Why it matters

Single-mission threat assessment answers "is this clip dangerous?" Global threat
inference answers "is this *area* becoming more dangerous over time, even if no
single mission crosses a threshold?" That is the difference between tactical
situational awareness and strategic site monitoring.

---

## Pre-Extension Checklist

Before writing any new runtime behavior, inspect a completed local run and answer:

- Which artifacts are strong evidence and which are model guesses?
- Which steps degraded, skipped, or used fallbacks?
- Does `analysis_summary.json` report missing modality coverage?
- Do `threat_primitives.json`, `local_threat_assessment.json`, and
  `policy_decision.json` cite enough independent evidence?
- Does the map or pose estimate support the spatial claims in the synthesis report?
- Does the `agentic_flow.md` audit log surface any step where evidence and
  narrative diverge?

If you cannot answer every question from artifacts, study the relevant learning-path
documents before designing an extension. Adding a new model or sidecar type before
the existing evidence pipeline is well-understood produces a system that is harder
to debug, not a better one.

---

## Related Docs

- [Temporal SSL and physical state](14_temporal_ssl_physical_state.md) — what is
  already implemented in track-aware SSL
- [Threat primitives and local inference](15_threat_primitives_local_inference.md) —
  what is already implemented in structured evidence gating
- [Probabilistic fusion deep dive](12_probabilistic_fusion_deep_dive.md) — the math
  foundation required before any calibration or field-model work
- `coop IoT edge monitoring` — the live
  site-state layer that global threat inference would extend

---

`← coop IoT edge monitoring` | [Essential technology stack →](17_essential_technology_stack.md)

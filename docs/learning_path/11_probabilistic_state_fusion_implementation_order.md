# Probabilistic State Fusion — Implementation Status

This document records what has been delivered and what is not yet done.

For the detailed math behind each component, see
[12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md).

## What Was Delivered

All five planned phases are complete.

### Phase 0 — Documentation and Interfaces ✓

- Requirements deep dive (this directory, doc 10)
- Architecture deep dive (doc 11)
- Implementation-order deep dive (this doc)
- `fusion/measurements.py` — `PlatformMeasurement` typed interface
- `fusion/state.py` — `PlatformFusionResult`, `PlatformPosteriorSample`,
  `FullFusionResult`
- Config surface in `pipeline/core/config.py` under `STATE_FUSION_*`,
  `OBJECT_FUSION_*`, `MAP_FUSION_*`

### Phase 1 — Platform-State Fusion MVP ✓

- `fusion/sidecars.py` — IMU and barometer loaders
- `fusion/filters/platform.py` — 6-state constant-velocity KF with GPS,
  barometer, and IMU-acceleration inputs
- `fusion/summaries.py` — `run_platform_state_fusion()`
- `workflows/local/steps_fusion.py` — `step_platform_state_fusion()`
- Artifact: `state_fusion.json` / `state_fusion.md`
- Runner integration in `runner.py` between steps 5 and 6

### Phase 2 — Visual-Pose Constraint Integration ✓

- `fusion/visual_pose.py` — Umeyama Sim(3) alignment
- `sfm_position` measurement kind accepted by `update_position()`
- Minimum frame threshold (`STATE_FUSION_SFM_MIN_FRAMES`, default 6)
- Measurement covariance derived from alignment RMSE
- Silently skips when fewer than 6 co-located SfM+GPS frames exist

### Phase 3 — Probabilistic Object-State Fusion ✓

- `fusion/filters/object_filter.py` — per-object constant-velocity KF
  in normalised image space with Mahalanobis gating
- `fusion/object_state.py` — full pipeline: Hungarian assignment, track
  lifecycle, per-track RTS smoother
- `ObjectFusionResult` with per-frame smoothed bounding boxes and
  velocity estimates

### Phase 4 — Map-State Fusion ✓

- `fusion/filters/rts_smoother.py` — Rauch-Tung-Striebel backward smoother
- `fusion/map_state.py` — platform KF with semantic noise adaptation +
  SfM visual constraints + RTS smoothing
- `MapFusionResult` with per-frame `covariance_diag` (6 diagonal elements
  of the 6×6 posterior covariance matrix)
- Comparison logging: raw vs smoothed mean covariance trace

### Phase 5 — Semantic Priors and Higher-Level Reasoning ✓

- `fusion/semantic_priors.py` — `SemanticPrior` dataclass and
  `build_semantic_prior()` function
- Scene type → process noise scale (14 scene categories, range 0.6×–2.0×)
- RSSM mean surprise → temporal noise scale (linear, 1.0×–3.0× over [0,1])
- Urban canyon detection (buildings, overpasses) → GPS noise 2.5× inflation
- Per-label speed caps for velocity clamping (pedestrian 3.5 m/s,
  car 40 m/s, etc.)

### Orchestration ✓

- `fusion/summaries.py` — `run_full_state_fusion()` runs all four layers
  in correct dependency order
- `steps_fusion.py` — `step_full_state_fusion()` called from runner after
  step 15 (SfM join), receives SfM positions, tracking results, Gemma
  analysis, Qwen captions, and RSSM surprise
- Artifact: `full_state_fusion.json`

---

## What Is Not Yet Done

### Dedicated unit tests

The filter math is covered by smoke tests in the implementation. Proper
synthetic-trajectory unit test files do not yet exist.

Planned files:

| File | Tests |
|---|---|
| `tests/unit/pipeline/fusion/test_platform_filter.py` | Circular trajectory posterior within 3σ of ground truth |
| `tests/unit/pipeline/fusion/test_rts_smoother.py` | Smoothed covariance ≤ filtered covariance at each step |
| `tests/unit/pipeline/fusion/test_umeyama.py` | Identity alignment on noise-free data, verify RMSE = 0 |
| `tests/unit/pipeline/fusion/test_object_filter.py` | Single straight-line track, Mahalanobis gate acceptance |
| `tests/unit/pipeline/fusion/test_semantic_priors.py` | Scene type → expected scale values |

### Online calibration

GPS noise is currently a fixed config value. Incorporating fix-type
and HDOP from the GPS receiver into the measurement covariance would
tighten or widen the estimate appropriately.

### World-frame object projection

Object state is currently in normalised image coordinates. Projecting
bounding-box centroids into the world frame using platform pose +
camera intrinsics would enable world-frame track positions. This requires
calibrated camera-to-platform extrinsics.

### Loop closure

When the platform revisits a GPS-overlapping region across multiple missions,
the current smoothed trajectory does not incorporate multi-mission loop
constraints. This is planned as a map-state extension once cross-mission
smoothing becomes a priority.

### Factor graph backend

The current smoothing is RTS, which is linear-Gaussian and optimal for the
CV process model. A factor-graph backend (e.g. GTSAM or hand-rolled) would
support non-Gaussian process models, better handle multi-mission loop
closures, and enable delayed-state smoothing with landmarks.

---

## Validation Sequence For Each Phase

For any future extension of the fusion stack, use this order:

1. Smoke test on synthetic data — verify the filter can track a known trajectory
2. Unit tests — test each component in isolation
3. Artifact inspection — run the local pipeline on a real video and inspect
   `full_state_fusion.json` manually
4. Regression comparison — compare summary diagnostics against a stored baseline
5. Integration — verify the runner integration does not break other steps

---

## Related Deep Dives

- [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md)
- [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md)
- [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md)

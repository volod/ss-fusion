# Probabilistic State Fusion Requirements

This deep dive documents the formal requirements for probabilistic state fusion
in `selfsuvis` and tracks which requirements have been met.

Read this before or alongside:

- [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md)
- [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md)

## Implementation Status

All five delivery phases are now complete. The implementation is in
`src/selfsuvis/pipeline/fusion/`. The four active fusion layers are:

| Layer | Module | Status |
|---|---|---|
| Platform-state KF (GPS + IMU + baro) | `fusion/filters/platform.py` | ✓ complete |
| Visual-pose constraints (SfM → ENU) | `fusion/visual_pose.py` | ✓ complete |
| Object-state fusion (KF + Mahalanobis + RTS) | `fusion/object_state.py` | ✓ complete |
| Map-state RTS smoothing | `fusion/map_state.py` | ✓ complete |
| Semantic priors (VLM-grounded noise) | `fusion/semantic_priors.py` | ✓ complete |

---

## Why This Exists

The earlier stack did:

- timestamp alignment to frame time
- modality-specific feature extraction
- context aggregation in `VideoKnowledge`
- downstream reasoning over the aggregated context

That is useful, but it is not the same thing as maintaining a posterior over a
latent world state.

Full probabilistic state fusion means the system explicitly estimates a hidden
state `x_t` over time and updates its belief using observations `z_t` and a
process model.

---

## Requirement 1 — Formal Latent State ✓

The system must define a state vector, not just a collection of artifacts.

**What was delivered:**

- Platform state: `x = [px, py, pz, vx, vy, vz]` in ENU frame.
  Implemented in `fusion/filters/platform.py`.
- Object state: `x = [cx, cy, w, h, vcx, vcy]` per tracked object in
  normalised image coordinates.
  Implemented in `fusion/filters/object_filter.py`.
- Map state: smoothed platform trajectory with full 6×6 covariance per frame.
  Implemented in `fusion/map_state.py`.

---

## Requirement 2 — Common Time Base ✓

All probabilistic fusion depends on consistent timing.

**What was delivered:**

- All measurements carry a `t_sec` float.
- The event timeline in `summaries.py` sorts measurements by `(t_sec, priority)`
  before processing, handling measurement kinds that arrive at the same timestamp.
- IMU accelerations are processed before GPS updates at equal timestamps.
- Frame sample events are given the lowest priority, so they are processed after
  all measurements at the same time.

---

## Requirement 3 — Coordinate Frames And Transform Graph ✓

Every fused measurement must be expressed in a known frame.

**What was delivered:**

- GPS measurements are converted from WGS-84 (lat/lon/alt) to ENU via
  `pipeline/mapping/gps_registration.py`.
- SfM poses are camera-from-world; camera centres are extracted as
  `p = -R^T @ t` in the SfM local frame.
- Sim(3) alignment in `fusion/visual_pose.py` maps SfM local → ENU using the
  Umeyama algorithm (optimal rotation + translation + scale).
- Object state lives in normalised image coordinates. No world-frame projection
  is attempted, which is the correct conservative choice until camera intrinsics
  and platform pose are jointly calibrated.

---

## Requirement 4 — Process Model ✓

**What was delivered:**

- Constant-velocity model with optional acceleration control input (IMU).
- `F = I_6, F[:3, 3:] = I_3 * dt` — position integrates velocity over `dt`.
- `B[:3, :] = 0.5 * I_3 * dt²`, `B[3:, :] = I_3 * dt` — acceleration input.
- Process noise `Q = diag([q_pos]*3, [q_vel]*3)` where `q = (std * dt)²`.
- The semantic prior multiplies `std` by a scene-type-specific scale before
  `Q` is computed, so noisier scenes produce larger `Q` automatically.

---

## Requirement 5 — Measurement Models ✓

**What was delivered:**

| Measurement kind | Observation model | Module |
|---|---|---|
| `gps_position` | `H = [I3 | 0]`, `z = [px, py, pz]` | `summaries.py` |
| `barometer_altitude` | `H = [0 0 1 0 0 0]`, `z = [pz]` | `summaries.py` |
| `imu_accel` | Not a Kalman update; sets `current_accel_enu` in filter | `summaries.py` |
| `sfm_position` | Same `H` as GPS; covariance from alignment RMSE | `visual_pose.py` |
| Object bbox | `H = [I4 | 0_{4×2}]`, `z = [cx, cy, w, h]` | `object_filter.py` |

---

## Requirement 6 — Noise And Uncertainty Calibration ✓

**What was delivered:**

- All noise parameters live in `config.py` under `STATE_FUSION_*` and
  `OBJECT_FUSION_*` prefixes.
- GPS noise: `STATE_FUSION_GPS_POS_STD_M` (default 5 m).
- Barometer: `STATE_FUSION_BARO_ALT_STD_M` (default 2.5 m).
- IMU: `STATE_FUSION_IMU_ACCEL_STD_MPS2` (default 1.5 m/s²).
- SfM position: `STATE_FUSION_SFM_POS_STD_M` (default 2 m) plus
  alignment RMSE.
- Object bbox: `OBJECT_FUSION_OBS_NOISE` (default 0.005 normalised coords).
- Semantic prior additionally scales GPS and process noise at runtime based
  on scene type and RSSM surprise.

---

## Requirement 7 — Typed Measurements ✓

**What was delivered:**

`PlatformMeasurement` in `fusion/measurements.py` carries:

- `kind` — measurement type string
- `t_sec` — timestamp
- `values` — tuple of floats
- `covariance` — full matrix as tuple of tuples
- `source` — which sensor / loader
- `frame` — coordinate frame string
- `quality` — `"nominal"` or `"approx_world_frame"`

---

## Requirement 8 — Data Association ✓

**What was delivered:**

- Per-object Kalman filter in `fusion/filters/object_filter.py`.
- Gating: Mahalanobis distance² compared against chi-squared threshold
  13.28 (4 DOF, p = 0.99).
- Assignment: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
  on the cost matrix of Mahalanobis distances.
- Infeasible cells (outside gate) are set to a large sentinel value so the
  Hungarian solver avoids them.
- Track lifecycle: tentative (< 3 hits), confirmed (≥ 3 hits), deleted
  (≥ 5 consecutive misses). Tentative tracks are included in the forward
  pass but RTS smoothing only runs over confirmed tracks.

---

## Requirement 9 — Map Representation With Uncertainty ✓

**What was delivered:**

- `MapFusionResult` in `fusion/map_state.py` carries per-frame smoothed
  states with full covariance diagonal.
- The RTS backward smoother reduces both lag and posterior covariance across
  the entire trajectory.
- Visual pose constraints from SfM tighten position uncertainty between GPS
  readings.
- `full_state_fusion.json` persists the complete smoothed trajectory.

---

## Requirement 10 — Diagnostic Residuals ✓

**What was delivered:**

- `filter_.innovation_norms` records the norm of each residual.
- `diagnostics` dict in `PlatformFusionResult` includes:
  `mean_innovation_norm`, `max_innovation_norm`, `predict_steps`,
  `measurement_total`.
- `MapFusionResult.diagnostics` adds: `mean_cov_trace_raw`,
  `mean_cov_trace_smoothed`, `sfm_measurements`, `gps_measurements`,
  `process_noise_scale`, `gps_noise_scale`.
- `ObjectFusionResult.diagnostics` reports: `total_tracks`,
  `confirmed_tracks`, `rts_smoothed_tracks`, `frames_processed`.

---

## Requirement 11 — Posterior Persistence ✓

**What was delivered:**

- `state_fusion.json` — platform-only GPS fusion (backward-compatible).
- `full_state_fusion.json` — all four layers combined, written by
  `step_full_state_fusion()` in `steps_fusion.py`.
- Contains: per-frame smoothed states, per-frame object states, semantic
  prior metadata, and diagnostics for all subsystems.

---

## Requirement 12 — Semantic Observation Policy ✓

**What was delivered:**

- Semantic outputs (Gemma, Qwen, RSSM) are used as **priors on noise parameters**,
  not as direct Kalman measurement updates.
- `SemanticPrior` in `fusion/semantic_priors.py` produces scalars for `Q`
  and `R` multiplication, not additional measurement vectors.
- Object speed priors clamp the velocity state post-update.
- This keeps the physical filter from being contaminated by hallucinations
  while still benefiting from semantic knowledge.

---

## Requirement 13 — Degradation Modes ✓

**What was delivered:**

- GPS only: works. The baseline path in `run_platform_state_fusion()`.
- GPS + baro: works. Barometer altitude updates the altitude dimension.
- GPS + IMU + baro: works. IMU sets `current_accel_enu` as control input.
- GPS + SfM: works. Umeyama alignment produces extra position measurements.
- No GPS, no telemetry: `run_platform_state_fusion` exits with
  `status="skipped"` and a clear reason string.
- Object fusion with no tracking results: `ObjectFusionResult` exits with
  `status="skipped"`.
- All failures are non-fatal to the local runner.

---

## Requirement 14 — Testability

**Partial.** The filter math is tested via the smoke tests in the
implementation. Dedicated unit test files for synthetic trajectory validation
are still a gap.

Planned unit tests (not yet written):

- `test_platform_filter.py` — synthetic circular trajectory, verify
  posterior tracks ground truth within 3σ.
- `test_rts_smoother.py` — verify smoothed covariance ≤ filtered covariance
  at each step.
- `test_umeyama.py` — identity alignment on noise-free data, verify RMSE = 0.
- `test_object_filter.py` — single straight-line track, verify Mahalanobis
  gate accepts true track and rejects distant spurious detection.
- `test_semantic_priors.py` — verify urban scene → proc_scale > 1.0,
  RSSM surprise 0.5 → temporal_scale 2.0.

---

## Related Deep Dives

- [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md)
- [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md)
- [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md)

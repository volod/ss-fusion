# Threat Primitive Layer — Local Inference From Physical State

This document explains the threat primitive step introduced in
`ssv_vdp/steps/threat_primitives.py`.

It covers:
1. Why free-text hazards are insufficient for decision-making
2. The structured primitive schema and what each field means
3. The two-source gate and why it exists
4. How each primitive is computed and what evidence it draws on
5. What to inspect when a primitive is surprising or missing

Related code: `ssv_vdp/steps/threat_primitives.py`
Related docs: [Physical scene layer](14_temporal_ssl_physical_state.md),
[Tracking and mapping](05_tracking_mapping_steps_21_27.md),
[Future directions](18_future_directions.md).

---

## 1. Why Free-Text Hazards Are Insufficient

UniDriveVLA's planning expert emits a list like:

```json
"hazards": ["pedestrian crossing ahead", "wet road surface", "limited visibility"]
```

This is useful for a human reader but unsuitable for downstream decision-making because:

**1. No score.** You cannot rank hazards, compare across missions, or set an alert
threshold.  "Limited visibility" might mean 10 % occlusion or 90 % — the string does not
say.

**2. No uncertainty.** A single-sensor signal can produce false positives.
A camera-based model that has never seen a particular object class may hallucinate a hazard.
There is no indication of how confident the model is.

**3. No evidence attribution.** When a hazard appears, there is no record of which sensor
or algorithm produced it.  If the camera model is wrong, you cannot trace the error back
to its source or exclude it from future runs.

**4. No spatial or temporal localization.** The list does not say *where* or *for how long*
the hazard persists.  You cannot correlate it with a frame, a time window, or a 3D map region.

**5. Single-source.** A free-text list from one model is a single point of failure.
It is impossible to cross-validate against an independent signal.

The threat primitive layer addresses all five limitations with a structured schema and a
two-source gate.

---

## 2. The Primitive Schema

Each emitted primitive is a dict with these fields:

```json
{
  "type":                 "collision_risk",
  "score":                0.42,
  "uncertainty":          0.12,
  "spatial_support":      ["frames/vid/frame_0011.jpg", "frames/vid/frame_0012.jpg"],
  "temporal_persistence": 7,
  "evidence_sources":     ["near_field_occupancy", "object_velocity"]
}
```

| Field | Type | Meaning |
|---|---|---|
| `type` | str | One of `collision_risk`, `visibility_degradation`, `track_anomaly`, `pose_uncertain` |
| `score` | float [0, 1] | Severity — 0 is benign, 1 is worst possible |
| `uncertainty` | float [0, 1] | How much the score could be wrong — 0 = certain, 1 = completely unknown |
| `spatial_support` | list[str] | Frame paths where the condition was observed |
| `temporal_persistence` | int | Maximum consecutive-frame run where the condition held |
| `evidence_sources` | list[str] | Named independent signals that agree (see §3) |

### Why score AND uncertainty?

A hazard with `score=0.8, uncertainty=0.05` is a well-supported high-severity finding —
act on it.

A hazard with `score=0.8, uncertainty=0.45` is a high-severity guess — flag it for human
review but do not trigger an automated response.

Uncertainty-aware scoring is standard in probabilistic robotics (see Thrun, Burgard, Fox
"Probabilistic Robotics", MIT Press 2005).  The `uncertainty` field is the pipeline's
analogue of a standard deviation on the score estimate.

### Why spatial_support and temporal_persistence?

**spatial_support** maps the primitive back to specific frames.  You can:
- Load those frames to manually verify the signal
- Cross-reference with the 3D map to localise the hazard in world coordinates
- Use them as hard-negative samples in the next SSL training run

**temporal_persistence** (max consecutive-frame run) distinguishes a brief spike
(flicker = likely false alarm) from a sustained condition (multiple consecutive
frames = more reliable).

A rule of thumb: distrust any primitive where `temporal_persistence == 1`.  A
single-frame event can be motion blur, a sensor dropout, or a tracker glitch.

---

## 3. The Two-Source Gate

**Rule**: a primitive is only emitted if `len(evidence_sources) >= 2`.

If only one signal triggers the threshold, the primitive is suppressed entirely — it does
not appear in `threat_primitives.json`.

### Why two sources?

This is a minimum-corroboration requirement adapted from DARPA's "independent evidence"
doctrine in autonomous system certification.

The core argument is:

> A signal that cannot be corroborated by an independent measurement pathway is a
> single-point failure mode.  Emitting it as a threat creates false alarms that erode
> operator trust and, in automated systems, trigger unnecessary interventions.

Each primitive type draws from evidence sources computed by different subsystems:

| Primitive | Possible evidence sources | Subsystems involved |
|---|---|---|
| `collision_risk` | `near_field_occupancy`, `object_velocity`, `free_space_estimate` | Object KF (fusion), depth estimation |
| `visibility_degradation` | `depth_failure_rate`, `caption_confidence` | Depth estimator, Florence-2 |
| `track_anomaly` | `track_breaks`, `short_track_length` | RF-DETR tracker, object KF |
| `pose_uncertain` | `kalman_pose_confidence`, `sfm_quality_degraded`, `sfm_failure_rate` | Platform KF, pycolmap SfM |

Because the sources for each primitive come from *different models* (RF-DETR vs Florence,
Kalman vs SfM), a false alarm from one source is unlikely to be replicated by the other.
Two independent false alarms in the same clip occurring simultaneously is much rarer than
a single false alarm.

### What the gate does NOT prevent

The gate prevents single-sensor false alarms, not correlated failures.
If the camera lens is covered, both `depth_failure_rate` and `caption_confidence`
will drop simultaneously — the gate will pass `visibility_degradation` even though
the underlying cause is a single hardware fault.

This is intentional: a lens obstruction genuinely degrades the platform's situational
awareness and should be flagged even if the root cause is mechanical.

---

## 4. How Each Primitive Is Computed

### 4.1 `collision_risk`

**Physical meaning**: tracked objects are dense in the near-field and approaching.

**Primary signal**: `near_field_occupancy_density` from `physical_state_summary.json`
(fraction of the central image region `[0.3, 0.7]²` covered by confirmed/smoothed tracks).

**Score formula**:
```
score = min(1.0,
    near_field_density × 0.50
    + min(1.0, mean_velocity / 0.05) × 0.30
    + (1.0 − free_space_estimate) × 0.20
)
```

**Uncertainty**: normalized `mean_bbox_uncertainty` from the object Kalman filter.
High Kalman covariance = more bbox uncertainty = less confidence in occupancy measurement.

**Per-frame spatial support**: computed by replaying the per-frame object states from
`full_fusion_result["per_frame_object_states"]`, aligned with the tracked frame paths from
`gemma_tracking_results.json`.

**Evidence thresholds**:
- `near_field_occupancy > 0.15` → "near_field_occupancy"
- `mean_velocity > 0.02` (normalised/frame) → "object_velocity"
- `free_space_estimate < 0.70` → "free_space_estimate"

### 4.2 `visibility_degradation`

**Physical meaning**: the platform's sensors are returning degraded or unreliable observations.

**Sources**:
- `depth_failure_rate`: fraction of frames where the depth estimator returned an error,
  unavailable, or disabled status. Computed from `depth_results.json`.
- `caption_confidence`: mean Florence-2 caption confidence across all keyframes.
  Low confidence indicates the captioning model is uncertain — often due to motion blur,
  overexposure, or heavy occlusion.

**Score formula**:
```
score = min(1.0, depth_fail_rate × 0.50 + (1.0 − mean_caption_conf) × 0.50)
```

**Uncertainty**: standard deviation of caption confidence across frames.  High variance
(some frames very confident, others not) = lower-certainty score than consistently low
confidence.

**Spatial support**: union of frames where depth failed and frames where caption confidence
was below threshold.

### 4.3 `track_anomaly`

**Physical meaning**: the tracker is losing and re-acquiring objects, suggesting occlusion,
fast motion, or detection noise.

**Sources**:
- `track_breaks`: fraction of consecutive track-ID appearance pairs that have a gap
  (track_id present at frame i, absent at frame i+1, present again later).
  Computed by scanning `gemma_tracking_results.json` frame-by-frame.
- `short_track_length`: mean number of frames each track ID appears in.
  Short mean length indicates most tracks are ephemeral — either the detector is unreliable
  or objects move too fast to be continuously tracked.

**Score formula**:
```
score = min(1.0,
    min(1.0, break_rate / 0.30) × 0.60
    + min(1.0, max(0, 5 − mean_track_len) / 5) × 0.40
)
```

**Uncertainty**: lower when tracking was successfully used in the physical state fusion
(`physical_state["tracking_used"] == True`); higher when tracking was skipped or degraded.

**Spatial support**: the frame indices that fall inside a continuity gap (frames where a
confirmed track was expected but absent).

**Evidence thresholds**:
- `break_rate > 0.08` (8% of consecutive track pairs have gaps) → "track_breaks"
- `mean_track_length < 5` frames → "short_track_length"

### 4.4 `pose_uncertain`

**Physical meaning**: the platform does not know where it is.

**Sources**:
- `kalman_pose_confidence`: the platform Kalman pose confidence from
  `physical_state_summary.json`. Below 0.40 means the Kalman covariance trace
  is high — the filter has not received enough corroborating GPS/IMU observations.
- `sfm_quality_degraded`: the 3D map step (`stats["map_degraded"]`) flagged the
  reconstruction as degraded (< 50 points or < 20 SfM poses).
- `sfm_failure_rate`: fraction of video frames without a registered SfM pose
  (`1 − sfm_poses / n_frames > 0.30`).

**Score formula**:
```
score = min(1.0,
    (1.0 − pose_confidence) × 0.60
    + min(1.0, sfm_fail_rate / 0.60) × 0.40
)
```

**Uncertainty**: decreases as more sources agree:
- 2 sources → uncertainty = 0.20
- 3 sources → uncertainty = 0.10

**Spatial support**: all frames in the video (pose uncertainty is a clip-level property —
the platform does not know where it is for the entire clip, not just at specific frames).

---

## 5. Reading `threat_primitives.json`

```json
{
  "skipped": false,
  "primitives": [
    {
      "type": "collision_risk",
      "score": 0.38,
      "uncertainty": 0.09,
      "spatial_support": ["frames/vid/frame_0015.jpg", ...],
      "temporal_persistence": 12,
      "evidence_sources": ["near_field_occupancy", "free_space_estimate"]
    },
    {
      "type": "pose_uncertain",
      "score": 0.61,
      "uncertainty": 0.20,
      "spatial_support": ["frames/vid/frame_0001.jpg", ...],
      "temporal_persistence": 47,
      "evidence_sources": ["kalman_pose_confidence", "sfm_quality_degraded"]
    }
  ],
  "summary": {
    "n_primitives": 2,
    "types_detected": ["collision_risk", "pose_uncertain"],
    "overall_threat_level": "medium"
  },
  "elapsed_sec": 0.003
}
```

**Overall threat level** mapping:

| max(score) | Level |
|---|---|
| ≥ 0.70 | high |
| ≥ 0.50 | medium |
| ≥ 0.25 | low |
| < 0.25 | none |

### When no primitives are emitted

`n_primitives == 0` is the expected result for a clean run with good GPS, a working depth
sensor, stable tracking, and clear scene captions.

If you expect a primitive but it is missing, check:
- Was the relevant upstream step skipped? (depth disabled, RF-DETR disabled, no GPS)
- Did only one source trigger? (two are required — print the evidence values manually)
- Are the thresholds too conservative? (`_COLL_OCC_THRESH = 0.15` etc. in the source)

### When `skipped: true`

All four upstream inputs (physical state, depth, tracking, full fusion) were skipped.
No inputs = no evidence = no primitives. This is correct behaviour.

---

## 6. Common Failure Modes

| Symptom | Likely cause | What to check |
|---|---|---|
| No `collision_risk` emitted despite visible congestion | Occupancy below 0.15 or velocity below 0.02 | Check `physical_state_summary.json` → `near_field_occupancy_density` and `tracked_object_velocities.mean`; lower thresholds if needed |
| `visibility_degradation` emitted on every run | Florence-2 caption confidence is always low for your video domain | Check `scene_captions.md`; caption confidence is 0.75 for non-Florence (OCR path); consider calibrating `_VIS_CAPTION_THRESH` |
| `track_anomaly` emitted even with smooth tracking | `mean_track_length < 5` because the scene has many short-lived objects | Check `gemma_tracking_summary.md` → `mean_track_length_frames`; raise `_TRACK_SHORTLEN_THRESH` for crowded scenes |
| `pose_uncertain` always absent despite no GPS | SfM succeeded (enough poses, enough points) and Kalman falls back to prior with moderate confidence | Check `sfm_poses` vs `n_frames`; if SfM ran on dense frames (not keyframes), pose count can be high even without GPS |
| `uncertainty` is always 0.40 for `track_anomaly` | `physical_state["tracking_used"]` is False (tracking step skipped) | Verify `RFDETR_ENABLED=true` and `gemma_tracking_results.json` exists |

---

## 7. Temporal Persistence And Action Vocabulary

The primitive layer is still *per-primitive*.  Area 4 adds a separate
`steps_local_threat.py` aggregation pass that answers the operational question:

> across the whole clip, which persisted threats matter enough to influence action?

The local-threat step applies a **persistence filter** before scoring.  A primitive
must appear in at least **N frames** (default `N = 3`) before it counts toward the
clip-level threat score.  This suppresses one-frame spikes caused by blur, tracker
glitches, or temporary sensor dropout.

### Why persistence is a separate gate

The primitive schema already has `temporal_persistence`, but that field is descriptive:
it tells you how long the condition lasted.  The local-threat step turns that into a
decision rule:

- below the persistence threshold: log the primitive but do not let it influence action
- above the threshold: include it in the clip-level threat score

This separation is deliberate.  The primitive layer answers "did this condition happen?".
The local-threat layer answers "did it persist long enough to matter?".

### Why the action vocabulary is fixed

`recommended_action` is intentionally restricted to:

- `continue`
- `reduce_speed`
- `reroute`
- `abort`
- `inspect_sensor`

The rationale is operational discipline.  Free-text recommendations are easy for a human
to read but hard for downstream logic to validate, rank, or simulate.  A fixed vocabulary:

- keeps action semantics stable across missions
- makes policy testing possible (`if action == "reroute" ...`)
- prevents the model from inventing novel actions that the autonomy stack cannot execute

The current mapping is deterministic:

- high `visibility_degradation` or `pose_uncertain` with no dominant collision signal
  tends to produce `inspect_sensor`
- moderate persistent `collision_risk` tends to produce `reduce_speed`
- stronger persistent `collision_risk` tends to produce `reroute`
- severe `collision_risk` combined with degraded visibility or pose confidence tends to
  produce `abort`
- absence of persisted threats yields `continue`

This is not a planner.  It is a local, auditable action prior derived from persisted
threat evidence.

---

## 8. Integration With Downstream Steps

The threat primitives result is stored in `video_context["threat_primitives"]` and is
available to:

- **Step 25 — Local threat inference**: persisted primitives are collapsed into one
  clip-level score and one fixed-vocabulary action recommendation.
- **Step 26 — Video synthesis**: the LLM can reference both the primitive summary and
  the clip-level local-threat assessment in its narrative.
  The `overall_threat_level` and `types_detected` fields are included in the synthesis context.
- **Step 27 — Agentic flow audit**: the audit LLM can reason about which primitives were
  emitted, which survived the persistence filter, and whether the recommended action is
  consistent with the visual evidence.
- **Threat-primitive-aware retrieval** (future): spatial_support frame paths can be used
  as a focused query set when searching for similar conditions across the mission archive.

---

## 9. Provenance And Disagreement Display

Area 5 extends the human-facing output so operators can inspect not just the final
threat score, but also the evidence path that produced it.

The key design principle is:

> do not silently resolve contradictions between sensors and models when the
> contradiction itself is operationally important.

For each persisted threat, the report now surfaces:

- the threat type
- the score and uncertainty
- the contributing sensor/model families
- the disagreeing sources
- the fixed-vocabulary recommended action

### Why disagreement belongs in the operator view

If the occupancy-derived signal says the near field is dense but UniDriveVLA says the
drivable area is clear, the correct operator experience is **not** a hidden weighted
average.  The correct experience is an explicit contradiction:

- occupancy stack: dense / constrained
- UniDrive perception: clear

That contradiction changes how a human should interpret the recommendation.  A
`reroute` or `abort` recommendation with strong disagreement should be treated
differently from the same recommendation with unanimous agreement.

### Provenance as trust calibration

Provenance is not only for debugging after the fact.  It is a real-time trust signal.

When the report says:

- `sensor_sources = object-state fusion, depth estimation`
- `disagreeing_sources = UniDriveVLA perception: drivable_area=clear`

the operator can immediately infer:

1. this threat is grounded in geometric / tracking evidence rather than only text generation
2. one model family disagrees, so automation confidence should be reduced
3. the next action is to inspect the supporting frames rather than accept the recommendation blindly

This is why the disagreement column is paired with score and uncertainty.  It acts as a
human-readable confidence decomposition rather than a single opaque scalar.

---

## 10. What A Human Should Study Here

### The two-source gate in practice

Read the evidence source table in §3 and ask: for a specific mission run, which two sources
happened to agree and why?

Load `threat_primitives.json` from a real run, pick the emitted primitive with the highest
score, and manually verify each evidence source:
1. Open the `spatial_support` frames in an image viewer.
2. Cross-reference with `physical_state_summary.json` for occupancy/velocity.
3. Cross-reference with `depth_results.json` (or `state_fusion.json`) for pose/depth.

If the evidence sources agree but the frames look benign, the threshold for one source is
too low.  If the frames look dangerous but no primitive was emitted, one of the two required
sources was below threshold — add a third source or lower an existing one.

### Uncertainty as a decision criterion

The `uncertainty` field is not decorative — it is the signal for *whether to act*.

A reasonable operational policy:
- `score > 0.50 AND uncertainty < 0.20` → automated flag + notify operator
- `score > 0.50 AND uncertainty ≥ 0.20` → queue for human review; do not automate
- `score ≤ 0.50` → log, do not notify

This two-threshold (score × uncertainty) policy is standard in nuclear safety systems
(double-blind confirmations) and aviation TCAS (two independent transponder agreements
before a resolution advisory).

---

## Related Docs

- [Physical scene layer](14_temporal_ssl_physical_state.md) — how physical_state_summary is computed
- [Tracking and mapping steps](05_tracking_mapping_steps_21_27.md) — RF-DETR track IDs and the tracker lifecycle
- [Probabilistic fusion deep dive](12_probabilistic_fusion_deep_dive.md) — Kalman/RTS/Mahalanobis math
- [Future directions](18_future_directions.md) — global threat inference, cross-mission persistence, and realtime mesh expansion

---

[← Temporal SSL and physical state](14_temporal_ssl_physical_state.md) | [Future directions →](18_future_directions.md)

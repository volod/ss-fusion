# Learning Path

The old single-file learning path has been split into a short path plus a deep-dive directory.

Start here:

- [Short local path](../guide/local_path.md)
- [Learning path deep dives](README.md)
- [Day-by-day syllabus](00_day_by_day_syllabus.md)

Recommended reading order:

1. [Short local path](../guide/local_path.md)
2. [Runtime and study guide](01_runtime_and_study_guide.md)
3. [Perception core, Steps 1-8](02_perception_core_steps_01_08.md)
4. [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md)
5. [Sensors and fusion, Steps 9-20](04_sensor_steps_09_20.md)
6. [Tracking and mapping, Steps 21-27](05_tracking_mapping_steps_21_27.md)
7. [Adaptation and audit, Steps 28-35](06_adaptation_eval_steps_28_35.md)
8. [Agentic knowledge flow](07_agentic_knowledge_flow.md)
9. [Local run artifact analysis](08_local_run_artifact_analysis.md)
10. [Local analytics math and methodology](13_local_analytics_math_methodology.md)
11. [Temporal SSL and physical state](14_temporal_ssl_physical_state.md)
12. [Threat primitives and local inference](15_threat_primitives_local_inference.md)
13. `ss-sens IoT edge monitoring`
14. [Essential technology stack](17_essential_technology_stack.md)
15. [Future directions: cross-modal SSL, environmental fields, calibration, global threats](18_future_directions.md)

If you were sent here by an older script or document, use this page as the compatibility entry point.

## Probabilistic State Fusion

The fusion subsystem is fully implemented. Four layers run after the 3D map
step on every local run and write `full_state_fusion.json`:

| Layer | Description |
|---|---|
| Semantic priors | VLM scene type + RSSM surprise → adaptive Kalman noise |
| Platform KF | GPS + IMU + baro constant-velocity Kalman filter |
| Map-state (RTS) | Adds SfM visual-pose constraints; RTS backward smoother |
| Object-state | Per-track KF + Mahalanobis gating + Hungarian + RTS |

Deep dives:

- [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md)
- [Requirements and status](09_probabilistic_state_fusion_requirements.md)
- [Architecture and data flow](10_probabilistic_state_fusion_architecture.md)
- [Delivery status and gaps](11_probabilistic_state_fusion_implementation_order.md)
- [**Mathematical deep dive**](12_probabilistic_fusion_deep_dive.md) ← start here if you want to understand the math

## Post-run analysis

After completing your first local run, use the analytics toolkit to inspect results:

- `Analytics & Visualization guide` — charts, HTML report, Python API
- [Local analytics math and methodology](13_local_analytics_math_methodology.md) — derived metrics, equations, and interpretation rules
- [Future directions: cross-modal SSL, environmental fields, calibration, global threats](18_future_directions.md) — not-yet-implemented advanced themes to study after the current local stack is understood
- [Essential technology stack](17_essential_technology_stack.md) — extended human-readable guide to the core technologies behind the current implementation

## ss-sens IoT Edge Monitoring

The `ss-sens` layer adds continuous site awareness on top of the mission-indexing
pipeline. It connects Mosquitto MQTT, ChirpStack LoRaWAN uplinks, Frigate NVR events,
MediaMTX RTSP bridge sessions, acoustic FFT/Whisper analysis, rolling site-state
aggregation, scene synthesis, and realtime threat-sector ingestion.

Start with:

- `coop IoT edge monitoring deep dive`
- [Day-by-day syllabus, Days 36-42](00_day_by_day_syllabus.md#week-6-iot-edge-monitoring-with-coop)
- [coop getting started](https://github.com/volod/ss-sens/blob/v0.1.0/docs/ss-sens/getting-started.md)
- [coop integration guide](https://github.com/volod/ss-sens/blob/v0.1.0/docs/ss-sens/integration.md)

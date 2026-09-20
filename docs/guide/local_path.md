# Local Learning Path

This is the short version of the local `ssv --mode local` study path.
Use it when you want the essentials for each step first, then jump into deeper material only where needed.

Deep-dive entry points:

- [Learning path index](../learning_path/README.md)
- [Day-by-day syllabus](../learning_path/00_day_by_day_syllabus.md)
- [Runtime and study guide](../learning_path/01_runtime_and_study_guide.md)
- [Perception core, Steps 1-8](../learning_path/02_perception_core_steps_01_08.md)
- [Sensor fusion fundamentals](../learning_path/03_sensor_fusion_fundamentals.md)
- [Probabilistic state fusion requirements](../learning_path/09_probabilistic_state_fusion_requirements.md)
- [Probabilistic state fusion architecture](../learning_path/10_probabilistic_state_fusion_architecture.md)
- [Probabilistic state fusion implementation order](../learning_path/11_probabilistic_state_fusion_implementation_order.md)
- [Physical sensors and fusion, Steps 9-20](../learning_path/04_sensor_steps_09_20.md)
- [Tracking, world models, and 3D mapping, Steps 21-27](../learning_path/05_tracking_mapping_steps_21_27.md)
- [Adaptation, evaluation, and audit, Steps 28-36](../learning_path/06_adaptation_eval_steps_28_35.md)
- [Agentic knowledge flow](../learning_path/07_agentic_knowledge_flow.md)
- [Local analytics math and methodology](../learning_path/13_local_analytics_math_methodology.md)
- [Temporal SSL and physical state](../learning_path/14_temporal_ssl_physical_state.md)
- [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md)
- `coop IoT edge monitoring`
- [Essential technology stack](../learning_path/17_essential_technology_stack.md)
- [Future directions: cross-modal SSL, environmental fields, calibration, global threats](../learning_path/18_future_directions.md)

## How To Use This Path

1. Read the step summary below once from top to bottom.
2. Pick the deep-dive document for the phase you are working on.
3. Before the sensor phase, read the sensor-fusion fundamentals session once so clocks, calibration, and contradiction handling are already in your head.
4. Use the day-by-day syllabus if you want a realistic study plan instead of reading everything at once.
5. Read the essential-technology guide to connect the codebase to the underlying tools before choosing advanced research topics.
6. Once the current runtime makes sense, use the advanced-directions document to decide what to study next instead of reading research topics randomly.

## Human Recommendations

For a human learner, the highest-return sequence is:

1. Learn the current runner and its artifacts well enough to spot silent failure.
2. Learn temporal alignment, uncertainty, and coordinate frames before adding new modalities.
3. Learn self-supervised representation learning before reaching for larger multimodal models.
4. Learn physical state estimation before trying to infer “threats” from captions and detections alone.
5. Treat advanced threat reasoning as a systems-and-inference problem, not only an LLM prompt problem.

## The 36 Steps, Short Version

| Step | Essential purpose | Go deeper |
|---|---|---|
| 1. Frame extraction | Decode the video into frames. This fixes the temporal resolution for everything else. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-1-frame-extraction) |
| 2. Vector store indexing | Embed frames with CLIP and DINO, then index them for retrieval and comparison. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-2-vector-store-indexing) |
| 3. Gemma multimodal analysis | Build video-level scene understanding, scene changes, and text-image retrieval context. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-3-gemma-multimodal-analysis) |
| 4. Florence captioning | Produce a readable scene caption for each key frame. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-4-florence-scene-captioning) |
| 5. ASR transcription | Turn speech into timestamped text aligned with the video. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-5-asr-transcription) |
| 6. OCR text extraction | Recover visible text from signs, dashboards, overlays, and labels. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-6-ocr-text-extraction) |
| 7. Depth estimation | Add a cheap geometric prior: near, far, cluttered, open. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-7-depth-estimation) |
| 8. Object detection | Turn scenes into object instances with boxes and labels. | [Perception core](../learning_path/02_perception_core_steps_01_08.md#step-8-object-detection) |
| 9. RF / SDR sensing | Inspect the radio environment around the mission when IQ data exists. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-9-rf--sdr-sensing) |
| 10. Thermal sensing | Add heat signatures that RGB alone cannot reveal. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-10-thermal--infrared-imaging) |
| 11. Multispectral sensing | Add non-RGB bands for material and vegetation cues. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-11-multispectral--hyperspectral-imaging) |
| 12. Event camera sensing | Represent change as asynchronous events instead of normal frames. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-12-event-camera-neuromorphic-sensing) |
| 13. LiDAR sensing | Add active ranging and point geometry. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-13-lidar--active-ranging) |
| 14. Radar sensing | Add motion and range structure that works in poor visibility. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-14-radar-fmcw--doppler--sar) |
| 15. GNSS-R and satellite reception | Add signal-based environmental and traffic evidence beyond the camera. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-15-gnss-r-and-satellite-signal-reception) |
| 16. Inertial and barometric sensing | Add motion, orientation drift, and altitude pressure context. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-16-inertial-and-barometric-sensing) |
| 17. Atmospheric sensing | Add weather and ambient environmental context. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-17-atmospheric--environmental-sensing) |
| 18. Chemical / gas / radiation sensing | Add invisible hazard indicators. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-18-chemical--gas--radiation-sensing) |
| 19. Acoustic sensing | Add sound evidence from engines, speech, impacts, and ambience. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-19-acoustic-sensing) |
| 20. Sensor fusion | Merge all side channels into one time-aligned context block. | [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md#step-20-sensor-fusion-analysis) |
| 20a. Physical state summary | Aggregate depth, tracking, and pose into a clip-level physical state: pose confidence, occupancy, object velocities, free-space estimate. Writes `physical_state_summary.json`. | [Temporal SSL and physical state](../learning_path/14_temporal_ssl_physical_state.md) |
| 20b. Environmental field state | Estimate coarse hazard fields (visibility, RF, thermal) from captions, depth, and sensor sidecars. Writes `field_state_summary.json`. | [Temporal SSL and physical state](../learning_path/14_temporal_ssl_physical_state.md) |
| 20c. Threat primitives | Combine physical state, field state, and multi-modal evidence into structured, evidence-gated threat primitives. Writes `threat_primitives.json`. | [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md) |
| 21. YOLO + SAM detection and segmentation | Refine object localization and add masks for spatial structure. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-21-yolo--sam-detection-and-segmentation) |
| 22. Gemma directed tracking | Use language-guided context to focus tracking on what matters. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-22-gemma-directed-tracking) |
| 23. World model embeddings | Move from isolated frames to temporal clip representations. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-23-world-model-video-embeddings) |
| 24. Qwen detailed captioning | Build dense per-frame reasoning from accumulated context. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-24-qwen-detailed-captioning) |
| 25. UniDriveVLA expert analysis | Add domain-specific understanding, perception, and planning structure. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-25-unidrivevla-expert-analysis) |
| 26. Base model search test | Check whether the baseline embedding space retrieves useful neighbors. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-26-base-model-search-test) |
| 27. 3D map and Gaussian Splat | Turn 2D evidence into reusable geometry and scene structure. | [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md#step-27-3d-map-and-gaussian-splat) |
| 27a. Local threat inference | Aggregate persisted threat primitives across the full video into a clip-level threat score and automation confidence. Writes `local_threat_assessment.json`. | [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md) |
| 27b. Action policy | Map threat score and sensor-health context to a fixed operator action vocabulary (`continue` / `reduce_speed` / `reroute` / `abort` / `inspect_sensor`). Writes `policy_decision.json`. | [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md) |
| 28. SSL DINO fine-tuning | Adapt the representation to the current mission without labels. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-28-ssl-dino-fine-tuning) |
| 29. Knowledge distillation | Compress the strong teacher into a smaller deployment model. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-29-knowledge-distillation) |
| 30. Drone detection edge training | Train a YOLOv8n drone detector from a public dataset plus mission hard negatives; export ONNX fp32 for Arm Cortex-A76 and int8 for Rockchip RV1106G3. | [Runbook](../runbooks/drone-detection.md) · [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-30-drone-detection-edge-training) |
| 31. ONNX export and gallery build | Package the adapted model for lightweight inference. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-31-onnx-export-and-gallery-build) |
| 32. Fine-tuned search test | Measure whether adaptation actually improved retrieval. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-32-fine-tuned-search-test) |
| 33. Model comparison and video description | Compare baseline vs adapted behavior and produce a clip-level summary. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-33-model-comparison-and-video-description) |
| 34. Multi-model comparison | Check agreement and disagreement across major multimodal analyzers. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-34-multi-model-comparison) |
| 35. Video synthesis | Turn many artifacts into one human-readable report. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-35-video-synthesis) |
| 36. Agentic flow audit | Explain how context moved through the pipeline and where risk can propagate. | [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md#step-36-agentic-flow-audit) |

## coop Extension Steps

These steps are not part of one `ssv --mode local` video run. They are the
reasonable next learning layer after Step 36: take the same evidence concepts from
the local pipeline and study how they behave in a continuous IoT site-awareness
runtime.

| Step | Essential purpose | Go deeper |
|---|---|---|
| 37. Coop stack bootstrap and health | Start Mosquitto, ChirpStack, Frigate, Redis, Postgres, and the REST bridge; verify container health and credentials before debugging higher-level code. | [coop getting started](https://github.com/volod/ss-sens/blob/v0.1.0/docs/ss-sens/getting-started.md) |
| 38. MQTT and LoRaWAN ingestion | Trace ChirpStack MQTT uplinks into `SensorReading` objects; learn which fields are physical measurements vs radio-link metadata. | `coop deep dive` |
| 39. Frigate event ingestion | Trace Frigate MQTT detection events into `CameraEvent` objects and rolling camera summaries. | [coop integration](https://github.com/volod/ss-sens/blob/v0.1.0/docs/ss-sens/integration.md) |
| 40. Rolling site state | Understand `SiteStateAggregator`: timestamp eviction, per-device deques, `/site/state`, `/site/sensors`, and `/site/cameras`. | `coop deep dive` |
| 41. RTSP bridge and acoustic evidence | Bridge Frigate streams through MediaMTX for live captioning, then add FFT/Whisper acoustic observations as synthetic camera events. | `coop deep dive` |
| 42. Site mesh and scene synthesis | Build GPS-proximity sensor graphs, query `/site/mesh`, and fuse live state plus `scene_timeline` captions into `/site/synthesis`. | `coop deep dive` |
| 43. Realtime threat bridge and analytics | Convert coop readings into `SensorEvent` / `ThreatEvent`, inspect `/site/threat`, and use `ss-sens-analytics` to diagnose stack health. | `coop deep dive` |

The mental bridge from local to coop is:

- Local Steps 1-8 teach how raw media becomes structured visual/audio/text evidence.
- Local Steps 9-20 teach sidecar sensor alignment, uncertainty, and fusion.
- Local Steps 20c and 27a-27b teach evidence-gated threat primitives and policy actions.
- Coop Steps 37-43 apply those ideas to live MQTT/RTSP streams, rolling windows, and sector-level threat snapshots.

## Realistic Day-By-Day Syllabus

Use this if you want a practical study sequence instead of trying to absorb all 36 steps at once.
For the longer version, see [the full syllabus](../learning_path/00_day_by_day_syllabus.md).

| Day | Focus |
|---|---|
| 1 | Repo overview, `README.md`, runtime shape, local outputs |
| 2 | Step 1: video basics, FPS, sampling, `ffmpeg`, frame timestamps |
| 3 | Step 2: embeddings, cosine similarity, CLIP vs DINOv3, vector stores |
| 4 | Step 3: Gemma multimodal reasoning, scene classification, `VideoKnowledge` |
| 5 | Step 4: Florence captioning; domain hints; compare your captions vs Florence |
| 6 | Steps 5-6: ASR and OCR as non-visual evidence; context injection |
| 7 | Steps 7-8: depth as geometric prior; detection and entity inventory |
| 8 | Step 9: RF basics, IQ, spectrograms, spectral flatness |
| 9 | Steps 10-12: thermal, multispectral, event cameras — not just more cameras |
| 10 | Steps 13-15: LiDAR, FMCW radar, GNSS-R, DOP |
| 11 | Steps 16-19: IMU, atmospheric, gas/radiation, acoustic sensing |
| 12 | Sensor fusion fundamentals: clocks, frames, calibration, uncertainty, then Step 20 timestamp alignment and fusion design |
| 13 | Steps 21-22: segmentation, IoU tracking, language-guided perception |
| 14 | Steps 23-25: clip embeddings, Qwen rolling state, UniDriveVLA |
| 15 | Steps 26-27: retrieval P@K and 3D Gaussian Splat |
| 16 | Step 28: SSL DINO fine-tuning, EMA teacher, loss sparkline |
| 17 | Steps 29-30: knowledge distillation and ONNX export |
| 18 | Steps 31-33: evaluation delta and cross-model comparison |
| 19 | Steps 34-35: synthesis sourcing and agentic flow audit |
| 20 | End-to-end run: predict artifacts before running, check after |
| 21 | Write your own one-page pipeline explanation from memory |
| 22-28 | Application week: custom queries, failure inventory, architecture extension (see full syllabus) |
| 29-35 | Advanced extension: self-supervised temporal learning, physical models, global threat inference, sensor-mesh runtime, and threat calibration — read `threat_primitives.json`, `local_threat_assessment.json`, and `global_threat_summary.json` after a full run |
| 36-42 | coop extension: bootstrap the IoT stack, trace MQTT/RTSP evidence, inspect `/site/state`, `/site/mesh`, `/site/synthesis`, `/site/threat`, and run `ss-sens-analytics` |

## Recommended Reading Order

If you only have time for a fast pass:

1. [Runtime and study guide](../learning_path/01_runtime_and_study_guide.md)
2. [Perception core](../learning_path/02_perception_core_steps_01_08.md)
3. [Sensor fusion fundamentals](../learning_path/03_sensor_fusion_fundamentals.md)
4. [Sensors and fusion](../learning_path/04_sensor_steps_09_20.md)
5. [Tracking and mapping](../learning_path/05_tracking_mapping_steps_21_27.md)
6. [Adaptation and audit](../learning_path/06_adaptation_eval_steps_28_35.md)
7. [Agentic knowledge flow](../learning_path/07_agentic_knowledge_flow.md)
8. [Local analytics math and methodology](../learning_path/13_local_analytics_math_methodology.md)
9. [Temporal SSL and physical state](../learning_path/14_temporal_ssl_physical_state.md)
10. [Threat primitives and local inference](../learning_path/15_threat_primitives_local_inference.md)
11. `coop IoT edge monitoring`
12. [Essential technology stack](../learning_path/17_essential_technology_stack.md)
13. [Future directions: cross-modal SSL, environmental fields, calibration, global threats](../learning_path/18_future_directions.md)

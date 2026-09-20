# Sensor Fusion Fundamentals Knowledge Session

This session is the bridge between "I know what each sensor measures" and
"I understand how a multimodal autonomy stack should combine them without
lying to itself."

Read this before or alongside
[04_sensor_steps_09_20.md](04_sensor_steps_09_20.md).

The goal is not to turn you into a specialist in Kalman filtering or Bayesian
state estimation in one sitting. The goal is to give you the minimum mental
model required to reason correctly about timestamp alignment, calibration,
uncertainty, contradictions, and evidence reuse in `selfsuvis`.

## Why This Session Exists

Most sensor-fusion failures are not caused by advanced math mistakes.
They are caused by simpler errors:

- two sensors were not time-aligned
- one sensor was treated as ground truth when it was only a weak cue
- a missing sensor was silently interpreted as a negative observation
- a camera-frame fact was mixed with an airframe-frame fact
- a high-latency source was fused as if it were instantaneous

That is exactly why this session matters for the local learning path.

`selfsuvis` currently uses a practical fusion style:

- it **collects** evidence from different modalities
- it **aligns** evidence onto the video timeline
- it **integrates** the evidence into `VideoKnowledge`
- it lets downstream models reason over a structured context block

It is not primarily doing full probabilistic state fusion in the classic
robotics sense. It is doing **context fusion** and **evidence accumulation**
with some geometric and temporal alignment.

## 1. What Fusion Is And Is Not

Sensor fusion means combining information from multiple sources to produce a
better estimate, decision, or context than any single source can provide.

Three common levels:

1. **Raw-data fusion**
   Combine measurements early, close to the sensor signal.
   Example: combining stereo images into a disparity map.
2. **Feature-level fusion**
   Combine extracted representations.
   Example: concatenate image features with IMU features before a model head.
3. **Decision/context-level fusion**
   Keep modality-specific outputs, then combine them later.
   Example: visual detection + OCR + ASR + weather note in one prompt context.

`selfsuvis` mostly operates at level 3, with some structured geometric support
from detection, depth, tracking, and mapping.

Important distinction:

- **fusion** tries to reduce uncertainty or improve confidence
- **integration** simply makes multiple pieces of evidence available together

In this repo, many steps are better described as integration than strict fusion.
That is fine. It is still the correct architecture for a system that must work
with optional, heterogeneous, and frequently missing sidecars.

## 2. The Five Foundations You Must Track

When reading any multimodal pipeline, keep these five questions in your head:

1. What physical quantity does this sensor actually measure?
2. In what coordinate frame is that quantity expressed?
3. On what clock or timestamp base was it recorded?
4. What is the uncertainty or reliability of the reading?
5. What downstream step will over-trust this if it is wrong?

If you cannot answer those five questions, you are not ready to fuse the sensor.

## 3. Time Alignment Comes First

The first hard problem in fusion is usually not model architecture.
It is time.

Every sensor has its own sample rate and latency:

- RGB video might be 2 fps after extraction
- IMU might be 100-400 Hz
- GPS might be 1-10 Hz
- weather API data may be effectively static for minutes
- OCR/Qwen outputs are post-hoc annotations on selected frames

`selfsuvis` uses the frame timestamp `t_sec` as the main alignment spine.
That has practical consequences:

- a fast sensor is downsampled or nearest-matched to the frame grid
- a slow sensor may be reused across many frames
- a delayed sensor can look contradictory even when it is correct

Questions to ask:

- What is the allowable lag window?
- Is the sensor causal or post-processed?
- Is the sensor reading describing the same instant as the frame?

Practical rule:

- if a disagreement disappears when you shift one stream by a constant offset,
  you likely have a synchronization problem, not a sensing problem

## 4. Coordinate Frames Matter

A measurement only makes sense inside a frame convention.

Common frames:

- **camera frame**: what the lens sees
- **image plane**: pixel coordinates
- **body frame**: the vehicle or drone airframe
- **world frame**: ENU, map, GPS, or SfM reconstruction frame

Examples:

- IMU acceleration is usually in the body frame
- detections are usually in image coordinates
- GPS is in a geographic world frame
- 3D map points are in a reconstructed or mapped world frame

Common mistake:

- comparing a gimbal-stabilized camera observation to an airframe IMU reading
  as if they described the same orientation

Before trusting any multimodal correlation, ask:

- are these two signals even expressed in compatible frames?

## 5. Calibration Is Not Optional

There are three types of calibration you should care about:

1. **Intrinsic calibration**
   Sensor-internal parameters such as focal length or lens distortion.
2. **Extrinsic calibration**
   The rigid transform between two sensors.
3. **Temporal calibration**
   The clock offset and latency relationship between sensors.

You can have perfect algorithms and still get nonsense if calibration is wrong.

Examples:

- RGB and thermal are useless for overlap analysis if the spatial registration is off
- LiDAR-to-camera projection is misleading if extrinsics drifted
- GPS-to-video association is wrong if timestamps are offset by even a second during motion

The practical learning takeaway:

- calibration errors often look like model errors
- they are not model errors

## 6. Uncertainty Beats Confidence Theater

Every modality has failure conditions.
The correct fusion mindset is not "which sensor is best?"
It is "under what conditions does each sensor become unreliable?"

Examples:

- monocular depth is weak on scale and textureless regions
- LiDAR degrades in rain, dust, or sparse returns
- radar has multipath and clutter problems
- OCR fails on blur, tiny text, or oblique views
- ASR fails on noise or overlapping speech
- thermal fails on reflective surfaces and low thermal contrast

If you only keep the mean estimate and ignore the reliability context, you are
not doing real fusion. You are doing optimism.

For this repo, the practical proxy for uncertainty is often:

- model confidence
- sensor availability
- quality flags
- step-level warnings
- human sanity checks against artifacts

## 7. Complementary Sensors Beat Redundant Sensors

Good fusion pairs sensors whose failures are different.

Examples:

- RGB + thermal
- monocular depth + LiDAR
- camera detections + radar velocity
- ASR + OCR + scene captioning

Less useful pairings are ones that fail for the same reason.

Examples:

- two RGB models both trained on similar internet images
- OCR and captioning on the same blurry frame without any other modality

Ask this whenever a new modality is proposed:

- what failure mode does it cover that the current stack does not?

If the answer is "none," it may not be worth the operational complexity.

## 8. Missing Data Must Stay Missing

Absence of evidence is not evidence of absence.

This matters in `selfsuvis` because many sensor stages are optional.

Good behavior:

- omit a modality line from the context when the sensor is absent
- mark the artifact as unavailable
- let downstream reasoning stay conditional

Bad behavior:

- write zeros as if they were real readings
- claim "no object" because a sensor did not run
- fuse stale data as if it were current

When reading code or artifacts, distinguish:

- sensor absent
- sensor present but no event detected
- sensor present but reading unreliable

Those are different states.

## 9. Contradictions Are Valuable

A contradiction between sensors is not always a bug.
Sometimes it is the most important signal in the mission.

Examples:

- radar sees motion where RGB sees nothing
- thermal shows a hot vehicle that looks parked and inert visually
- OCR indicates a hazard sign but the detector missed the object
- GPS jump disagrees with IMU continuity, implying GPS corruption

Your job as a learner is to decide which class of contradiction you are seeing:

1. true anomaly
2. calibration / sync error
3. sensor-specific failure mode
4. model hallucination or bad prompt context

That classification skill is a major part of multimodal engineering.

## 10. What `selfsuvis` Actually Does Today

At a practical level, the current local pipeline uses these fusion ideas:

- `VideoKnowledge` accumulates evidence across stages
- `context_for_frame(t_sec)` assembles the per-frame multimodal context
- OCR and ASR inject textual side evidence
- depth, detection, tracking, and world-model steps provide geometric and temporal priors
- later captioning and audit stages reason over the accumulated context rather than over raw sensor feeds directly

That means the most important fusion artifacts for a human are often not raw arrays.
They are:

- markdown summaries
- JSON per-frame or per-video structures
- the exact context string handed to a reasoning model

In this repo, understanding fusion means understanding **evidence flow**.

## 11. Practical Checklist For Any New Sensor

Before adding a new modality to the pipeline, answer:

1. What physical signal is measured?
2. What are the units?
3. What is the native sample rate?
4. What timestamp source is used?
5. What coordinate frame is used?
6. What are the main failure modes?
7. How is uncertainty exposed?
8. What artifact will a human inspect after a run?
9. What later stage consumes it?
10. What is the safe behavior when it is absent?

If you cannot answer all ten, the integration is incomplete.

## 12. Knowledge Session Exercises

Use these before diving into the full sensor phase.

**Exercise 1: classify evidence**

Take one artifact from a real run and label it as:

- raw measurement
- derived feature
- model inference
- fused context

Do this for:

- `asr_subtitles.md`
- `scene_captions.md`
- `multimodal_features.md`
- `detailed_captions.md`

**Exercise 2: trace one frame**

For one timestamped frame, write down:

- what the frame itself shows
- what OCR adds
- what ASR adds
- what detections add
- what later Qwen text adds

Then separate "observed" from "inferred."

**Exercise 3: contradiction drill**

Invent one realistic contradiction for each pair:

- RGB vs thermal
- monocular depth vs LiDAR
- GPS vs IMU
- OCR vs detector

For each contradiction, say whether the most likely explanation is:

- sync/calibration issue
- modality failure
- real-world anomaly

## 13. What To Read Next

After this session:

1. Read [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md)
2. Re-read the `VideoKnowledge` and `context_for_frame()` flow in
   [01_runtime_and_study_guide.md](01_runtime_and_study_guide.md)
3. Then continue to
   [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md)

## Further Reading

- Thrun, Burgard & Fox, *Probabilistic Robotics*:
  still the best conceptual base for thinking about uncertainty, state, and fusion.
- Barfoot, *State Estimation for Robotics*:
  the right next step if you want the math done properly.
- OpenVINS paper:
  useful for seeing how visual-inertial fusion becomes an operational system.

The purpose of this knowledge session is not to replace those sources.
It is to make the rest of this repo legible before you need the full theory.

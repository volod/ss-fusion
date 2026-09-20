# Physical Sensors And Fusion: Steps 9-20

This phase expands perception beyond RGB video.
The goal is not to force every mission to use every sensor.
The goal is to understand how the pipeline can absorb side-channel evidence when it exists, and how to design for graceful degradation when it does not.

A practical rule: **treat each sensor as a hypothesis generator, not a ground truth source.**
The pipeline merges them in Step 20 precisely because no single sensor is always right.

Before you study individual modalities, read
[Sensor Fusion Fundamentals](03_sensor_fusion_fundamentals.md).
That session gives the conceptual base for everything in this file:
clocks, coordinate frames, calibration, uncertainty, contradiction handling,
and the difference between integration and true fusion.

## Current implementation reality

This document is intentionally broader than the code path taken by a typical
single-video local run.

The current local runner:

- may skip many of these steps entirely when no sidecar data exists
- uses the video frame timeline as the main alignment spine
- accumulates evidence in `VideoKnowledge`
- performs mostly context-level fusion rather than full probabilistic state fusion

That means the right human question is usually not "did the sensor mathematically
fuse with everything else?" It is "what evidence became available, how was it
aligned, and how will a later stage over-trust it if it is wrong?"

## Deep-dive themes for this phase

As you read Steps 9-20, keep four themes in view:

1. **Physical quantity**
   What is each modality actually measuring: emitted radiation, reflected light,
   pressure, acceleration, RF energy, acoustic pressure, or estimated semantics?
2. **Alignment**
   How does that measurement get tied to frame time `t_sec` and to the scene?
3. **Reliability**
   Under what environmental conditions does the modality become misleading?
4. **Downstream reuse**
   Which later step consumes this evidence: Qwen context, tracking, mapping,
   anomaly flagging, or human report synthesis?

---

<a id="step-9-rf--sdr-sensing"></a>
## Step 9. RF / SDR sensing

**What it does:**
Ingest IQ captures (raw radio-frequency samples) or an audio proxy, compute spectrograms and signal statistics, and derive radio-environment features: occupied bandwidth, SNR, spectral flatness, peak frequencies.

**Why it matters:**
The radio environment around a mission can be operationally significant even when nothing visually changes.
Sudden frequency occupation or jamming signatures can indicate activity that no camera sees.
For drone missions, detecting interference in control-link bands (2.4 GHz, 5.8 GHz) is a safety signal.

**Implementation:**
- [`pipeline/vision/rf_analyzer.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/rf_analyzer.py)
- ``scripts/ssv/ssv-prepare-sensor-data.sh``

**Key concepts:**

*IQ data:*
An SDR captures signal as complex samples: I (in-phase) and Q (quadrature) components.
The combination gives the full amplitude and phase of the signal, not just its magnitude.
Without IQ you can only measure signal strength; with IQ you can measure modulation, bandwidth, and Doppler shift.

*Spectrogram:*
Computed by sliding a short-time Fourier transform (STFT) over the IQ stream.
Result: a 2D array of time vs frequency vs power.
The pipeline extracts summary statistics from this rather than storing the full spectrogram.

*Spectral flatness:*
A measure of how "noise-like" a signal is.
White noise has spectral flatness near 1. A pure tone has flatness near 0.
Jamming signals tend to be broadband (flatness near 1) while communications signals are narrowband (flatness near 0).

*SNR:*
Signal-to-noise ratio in dB. Low SNR means the signal is buried in noise; estimates are unreliable below ~10 dB.

**Output artifact:**
RF feature JSON per time window: `{center_freq, bandwidth, snr_db, flatness, peak_power}` plus a time-aligned map to `t_sec`.

**Human focus:**
- Understand IQ basics: I and Q are the real and imaginary parts of a complex baseband signal.
- Learn to read a spectrogram: x-axis = time, y-axis = frequency, color = power in dB.
- Know the difference between occupied bandwidth (how wide is the emission?) and SNR (how clean is the measurement?).
- Understand spectral flatness as a modulation indicator.
- Learn where this step is likely absent: most consumer drone videos have no IQ sidecar.

**Common failure modes:**
- No IQ capture in mission → step is skipped; RF context is empty.
- Sample rate mismatch between IQ capture and expected SDR rate → frequency axis is scaled wrong.
- Very short capture window → spectral resolution is poor (frequency bins are wide).
- Wideband interference saturates receiver → all SNR estimates are unreliable.

---

<a id="step-10-thermal--infrared-imaging"></a>
## Step 10. Thermal / infrared imaging

**What it does:**
Ingest thermal (LWIR, ~8-14 µm wavelength) or near-infrared frames alongside visible-light frames.
Align them spatially to the RGB camera. Derive heat signature features: hot spots, thermal gradients, temperature zones.

**Why it matters:**
Thermal radiation reveals what RGB cannot:
living agents behind foliage or at night, engine heat in parked vehicles, heat leaks in buildings, and body temperature of personnel.
A vehicle that looks inactive in visible light may show a running engine in thermal.

**Implementation:**
- [`pipeline/vision/factory.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/factory.py) — sensor adapter registration
- Thermal frames are processed as grayscale imagery with specialized normalization
- Sidecar files: `.seq` (FLIR), `.tiff` 16-bit, or proprietary drone thermal stream

**Key concepts:**

*Wavelength bands:*
- Near-infrared (NIR, 0.7-1.4 µm): reflected solar radiation, sensitive to vegetation (NIR is highly reflective for healthy leaves). Not temperature measurement.
- Short-wave IR (SWIR, 1.4-3 µm): sensitive to moisture and glass penetration.
- Mid-wave IR (MWIR, 3-5 µm): good for detecting hot objects (engines, fires) against cool backgrounds.
- Long-wave IR (LWIR, 8-14 µm): thermal radiation from room-temperature objects; what most drone thermal cameras use.

*Radiometric vs non-radiometric thermal:*
Radiometric cameras return absolute temperature per pixel (in Kelvin or Celsius).
Non-radiometric cameras return relative IR intensity. Most affordable drone thermal cameras are non-radiometric.
The pipeline normalizes both to [0, 1] intensity range.

*Spatial alignment (registration):*
Thermal and RGB cameras have different fields of view and lens centers.
Alignment requires either factory calibration parameters or homography estimation.
Misalignment is common and causes thermal overlay errors.

**Output artifact:**
Per-frame thermal feature JSON: `{hot_spot_count, max_temp_relative, mean_temp_relative, thermal_zones}`.
Overlay image: thermal colormap (e.g., "inferno" or "rainbow") saved beside the RGB frame.

**Human focus:**
- Understand emissivity: different materials emit different fractions of their true thermal radiation. Metal is reflective and appears cold even when hot.
- Learn the difference between detecting temperature and detecting heat contrast.
- Understand when thermal is most useful: at night, in smoke, or against foliage.
- Learn where thermal fails: reflective surfaces, cold environments where all objects are near ambient, and very hot scenes where contrast collapses.

**Common failure modes:**
- Thermal camera not present → step skipped; no thermal context.
- Non-radiometric camera with automatic gain control → intensity values shift between frames, making temporal comparison unreliable.
- Spatial misregistration → thermal hot spot is attributed to wrong RGB pixel region.
- High ambient temperature (desert noon) → temperature contrast between objects collapses; thermal advantage disappears.

---

<a id="step-11-multispectral--hyperspectral-imaging"></a>
## Step 11. Multispectral / hyperspectral imaging

**What it does:**
Ingest imagery with more than three spectral bands (e.g., Red, Green, Blue, NIR, RedEdge for multispectral; hundreds of narrow bands for hyperspectral).
Compute vegetation indices (NDVI), material indices (NDWI, NDSI), and spectral signatures for each pixel.

**Why it matters:**
Materials that look identical in RGB often separate in spectral space.
Healthy vs stressed vegetation, different soil compositions, water vs non-water surfaces, certain minerals, and camouflage materials all have distinct spectral signatures.
This is widely used in agriculture, forestry, geology, and military reconnaissance.

**Implementation:**
- Multi-band TIFF ingestion via `rasterio` or `GDAL`
- Index computation: NDVI = (NIR - Red) / (NIR + Red)
- Spectral signature extraction and comparison to a reference library

**Key concepts:**

*Spectral index:*
A ratio of two or more bands designed to highlight a specific material or condition.
- NDVI (Normalized Difference Vegetation Index): measures plant health using NIR and Red. Range -1 to +1; healthy vegetation > 0.4.
- NDWI (Normalized Difference Water Index): highlights water bodies.
- NDSI (Normalized Difference Snow Index): highlights snow/ice.

*Hyperspectral vs multispectral:*
Multispectral: 4-20 bands, typically pre-defined wavelengths. Lightweight cameras, common on drones.
Hyperspectral: 100-500 narrow contiguous bands. Large data volume, requires specialized processing (PCA, matched filter).
The pipeline currently targets multispectral; hyperspectral support is aspirational.

*Band registration:*
Multi-band cameras often have separate sensors per band with slight spatial offsets.
Band-to-band registration (alignment) must happen before index computation.
Errors produce "rainbow" artifacts around object edges.

**Output artifact:**
Per-frame spectral index map (GeoTIFF) and summary statistics: `{ndvi_mean, ndvi_std, water_fraction, stressed_veg_fraction}`.

**Human focus:**
- Learn what NDVI looks like and what a healthy vs stressed vegetation field looks like in it.
- Understand why spectral signatures can reveal camouflage: paint reflects differently than natural vegetation in NIR.
- Know the key limitation: multispectral cameras need calibration panels or atmospheric correction for cross-flight comparison.

**Common failure modes:**
- No multispectral payload → step skipped.
- Poor illumination or shadow → spectral ratios are distorted in shadowed pixels.
- Band registration errors → index maps have edge artifacts.
- Atmospheric scattering (high-altitude flights) → apparent reflectance values shift; NDVI underestimates vegetation.

---

<a id="step-12-event-camera-neuromorphic-sensing"></a>
## Step 12. Event camera (neuromorphic sensing)

**What it does:**
Ingest asynchronous event streams from a Dynamic Vision Sensor (DVS).
Convert events to a time-surface or event frame representation.
Extract motion speed, direction of movement, and temporal texture features.

**Why it matters:**
Standard cameras capture frames at a fixed rate regardless of scene activity.
Event cameras fire a pixel event only when that pixel's brightness changes, with microsecond resolution.
This makes them far better than frame cameras for:
- Very fast motion (propellers, projectiles, fast vehicles)
- High dynamic range scenes (bright sun + dark shadow in the same frame)
- Low latency detection (events arrive in real time, not at 30 fps)

**Implementation:**
- Event stream ingest from `.aedat`, `.raw`, or `.rosbag` formats
- Time-surface computation: each pixel stores the timestamp of its last event
- Event frame accumulation: sum events over a window for visualization

**Key concepts:**

*Event polarity:*
Each event carries a polarity: positive (brightness increased) or negative (brightness decreased).
Motion boundaries produce pairs of positive-then-negative events as the edge passes the pixel.
This gives edge detection for "free" without any image processing.

*Time surface:*
A 2D image where each pixel stores the timestamp of its most recent event.
Recent events appear bright; pixels with no recent events fade.
This is the natural representation for training event-based neural networks.

*No motion → no signal:*
A static scene produces almost no events (only noise).
This is a feature (efficient, no redundant data) and a limitation (the camera goes "blind" to stationary objects).

**Output artifact:**
Per-time-window event statistics: `{event_rate, dominant_motion_direction, mean_velocity_px_per_sec}`.
Accumulated event frames (optional): grayscale images for visualization.

**Human focus:**
- Understand why event cameras are fundamentally different from frame cameras (asynchronous, pixel-independent, microsecond timestamps).
- Learn when this sensor adds value vs when it is redundant (only useful for fast motion or HDR scenes).
- Know that training data for event cameras is still scarce; most pretrained models are not production-ready.

**Common failure modes:**
- No event camera in mission → step skipped.
- Very slow motion → event rate near zero; little useful signal.
- Extreme illumination change (camera pans toward sun) → massive event burst; event rate statistics become meaningless.
- Noise events (sensor defects) → "hot pixels" emit constant events and must be filtered.

---

<a id="step-13-lidar--active-ranging"></a>
## Step 13. LiDAR / active ranging

**What it does:**
Ingest a LiDAR point cloud (`.pcd`, `.las`, `.bag`, or per-frame `.bin`) aligned to the mission timestamp.
Compute per-frame range statistics: minimum range, maximum range, point density, ground plane height, and obstacle height profile.
Optionally perform ground plane extraction and object cluster detection.

**Why it matters:**
LiDAR provides explicit distance measurements rather than inferred monocular estimates.
It directly measures the 3D structure of the scene, enabling:
- Accurate obstacle detection at precise distances
- Ground plane extraction even on uneven terrain
- Height measurement of objects and terrain features
- Point cloud registration for map building

**Implementation:**
- Point cloud ingestion via `open3d` or `pyntcloud`
- Timestamp-synchronized to frame list via `t_sec` interpolation
- Ground segmentation via RANSAC plane fitting
- Cluster detection via DBSCAN or Euclidean clustering

**Key concepts:**

*Mechanical vs solid-state LiDAR:*
Mechanical (spinning) LiDAR (e.g., Velodyne VLP-16): 360° horizontal coverage, 16-128 scan lines, 10-20 Hz.
Solid-state LiDAR (e.g., Livox Avia): narrow field of view (~70°), no spinning parts, higher durability.
The pipeline accepts both but handles their different point cloud densities differently.

*Returns and intensity:*
Each LiDAR pulse may produce 0, 1, or multiple returns per ray (first return, last return, strongest return).
Multiple returns happen when a pulse passes through foliage and hits the ground below.
Return intensity indicates surface reflectivity.

*Point cloud alignment to RGB:*
To combine LiDAR structure with RGB texture, you need the extrinsic calibration (rotation + translation) between LiDAR and camera.
Without calibration, point clouds and images can only be used independently.

**Output artifact:**
Per-frame LiDAR statistics JSON: `{min_range_m, max_range_m, point_count, ground_height_m, cluster_count}`.
Optional: downsampled point cloud for 3D map use in Step 27.

**Human focus:**
- Understand the physical operation: a pulsed laser measures time-of-flight to compute range.
- Learn point cloud density vs range tradeoff: farther objects have sparser measurements.
- Understand why LiDAR beats monocular depth: it gives absolute metric distance, not relative estimates.
- Know the main failure mode: rain, fog, and dust scatter the laser, causing range errors or total signal loss.

**Common failure modes:**
- No LiDAR → step skipped; monocular depth from Step 7 is the only geometry.
- Time sync error between LiDAR and camera → point cloud appears offset in time; obstacle positions are wrong.
- Velodyne spinning at 10 Hz vs camera at 30 Hz → point cloud undersamples during fast movement.
- Retroreflective surfaces (road markings, safety vests) → overdriven returns with range errors.

---

<a id="step-14-radar-fmcw--doppler--sar"></a>
## Step 14. Radar (FMCW / Doppler / SAR)

**What it does:**
Ingest radar range-Doppler data from an FMCW (Frequency Modulated Continuous Wave) or pulse radar.
Extract range profiles, velocity estimates (Doppler), and optionally SAR (Synthetic Aperture Radar) imagery.

**Why it matters:**
Radar works in conditions where cameras and LiDAR fail: rain, fog, smoke, dust, and darkness.
FMCW radar provides simultaneous range and velocity measurements, making it ideal for tracking moving targets.
SAR produces high-resolution imagery independent of illumination, enabling day/night/all-weather mapping.

**Implementation:**
- Raw FMCW cube processing: fast time (range), slow time (Doppler), antenna (angle)
- Range-FFT + Doppler-FFT to produce a range-Doppler map
- CFAR (Constant False Alarm Rate) detection for target extraction
- Angle estimation from multiple receive antennas (MIMO)

**Key concepts:**

*FMCW principle:*
The radar transmits a signal whose frequency rises linearly (a chirp).
The reflected signal arrives at the receiver with a time delay proportional to range.
Mixing the transmitted and received chirp produces a beat frequency proportional to range.

*Doppler shift:*
A moving target shifts the reflected chirp frequency proportionally to its radial velocity.
Positive Doppler = target approaching. Negative = receding.
Doppler gives velocity "for free" without multiple frames.

*Range resolution:*
Range resolution = c / (2 × bandwidth). Higher bandwidth → finer resolution.
A 77 GHz automotive radar with 4 GHz bandwidth resolves targets ~4 cm apart.

*CFAR detection:*
Targets are detected by comparing each range-Doppler cell to the statistical noise floor of neighboring cells.
This adapts to varying clutter levels automatically.

**Output artifact:**
Per-time-window radar detections: `{range_m, velocity_mps, azimuth_deg, snr_db}` for each detected target.
Optional: range-Doppler heatmap image.

**Human focus:**
- Learn the FMCW chirp principle: increasing frequency + time delay → beat frequency = range.
- Understand range-Doppler maps: two axes give simultaneous range and velocity of every target.
- Know the key radar artifacts: range sidelobes, velocity ambiguity (aliasing), multipath reflections.
- Understand when radar is worth the complexity: primarily for all-weather operation and velocity measurement.

**Common failure modes:**
- No radar → step skipped.
- Static clutter (ground, buildings) overwhelms detection → Doppler filtering needed to separate stationary from moving.
- Velocity ambiguity: targets faster than `PRF/2` wrap around in Doppler space.
- Multipath: ground reflection interferes with direct path signal, causing ghost targets.

---

<a id="step-15-gnss-r-and-satellite-signal-reception"></a>
## Step 15. GNSS-R and satellite signal reception

**What it does:**
Ingest GPS/GNSS position data from the platform, extract position accuracy metrics, and optionally process GNSS-R (reflectometry) data.
Pull external evidence from satellite-derived products: AIS shipping data, ADS-B aircraft tracking, space weather indices.

**Why it matters:**
This step broadens context beyond the camera platform itself.
GNSS provides the absolute geographic reference that anchors all other observations to real-world coordinates.
GNSS-R adds soil moisture, sea surface height, and ice cover sensing from reflected navigation signals.
External satellite feeds (AIS, ADS-B) add activity evidence that the onboard sensors cannot observe.

**Implementation:**
- GPS sidecar extraction: `pipeline/gps_extractor.py`
- GNSS accuracy metrics: HDOP, VDOP, satellite count, fix type
- Optional AIS/ADS-B integration via external API (not enabled by default)

**Key concepts:**

*GNSS dilution of precision (DOP):*
DOP values describe how satellite geometry affects position accuracy.
HDOP < 1 is excellent. HDOP 1-2 is good. HDOP > 5 means poor geometry; position error grows significantly.
DOP is not the same as actual error: low DOP + good signal = accurate; low DOP + multipath = still wrong.

*GNSS-R (reflectometry):*
GNSS signals reflected off the Earth's surface carry information about the surface properties.
Delay-Doppler maps of the reflected signal can measure soil moisture, sea roughness, and snow depth.
This is a specialized technique requiring dual-antenna receivers; rare in standard drone missions.

*AIS and ADS-B:*
AIS (Automatic Identification System): maritime vessel transponder data. Available globally.
ADS-B (Automatic Dependent Surveillance-Broadcast): aircraft transponder data. Available globally.
Both provide activity context for missions near ports, airports, or shipping lanes.

**Output artifact:**
Per-frame GPS quality JSON: `{lat, lon, alt_m, hdop, vdop, fix_type, satellite_count}`.
Optional: AIS/ADS-B contact list for the mission area during the mission time window.

**Human focus:**
- Understand DOP: why satellite geometry matters as much as signal strength for position accuracy.
- Know what GNSS fix types mean: no fix, 2D fix, 3D fix, DGPS, RTK.
- Understand the difference between GNSS accuracy (reported by the receiver) and GNSS precision (actual position error after all sources of error).
- Learn when GNSS fails: urban canyons, dense foliage, electromagnetic interference, spoofing.

**Common failure modes:**
- No GPS sidecar → GPS context is null; map registration fails.
- GPS-denied environment (indoor, jammed) → position is reported but wrong; downstream map is corrupted.
- Time synchronization between GPS and video clock → timestamp offset causes trajectory errors.
- GNSS-R requires specific hardware → almost always unavailable in practice.

---

<a id="step-16-inertial-and-barometric-sensing"></a>
## Step 16. Inertial and barometric sensing

**What it does:**
Ingest IMU (Inertial Measurement Unit) data: accelerometer, gyroscope, and optionally magnetometer.
Ingest barometric altitude from pressure sensor.
Compute platform motion state: velocity estimates, angular rate, attitude (roll/pitch/yaw), and altitude profile.

**Why it matters:**
The IMU anchors visual observations to how the platform actually moved.
Without IMU, a camera tilt looks like a scene change; with IMU, it is identified as a platform rotation.
IMU data is essential for:
- Distinguishing camera motion from scene motion in optical flow
- Deblurring frames during high-acceleration maneuvers
- Visual-inertial odometry (VIO) for GPS-denied navigation
- Validating that GPS position jumps are real vs sensor glitches

**Implementation:**
- IMU sidecar ingestion from `.csv`, `.bin`, or proprietary drone flight log format
- Complementary filter or Madgwick filter for attitude estimation
- Barometric altitude integration with GPS altitude cross-check

**Key concepts:**

*IMU noise model:*
Accelerometers and gyroscopes have two dominant noise sources:
- White noise: random measurement fluctuations at each sample.
- Bias drift: slow, unpredictable drift in the zero reading over time.
Integration of accelerometer readings accumulates error quadratically; integration of gyroscope readings drifts linearly.
This is why IMU alone is useless for long-range position estimation without GPS or vision correction.

*Complementary filter:*
Combines accelerometer (stable long-term, noisy short-term) with gyroscope (accurate short-term, drifts long-term) using a frequency-domain filter.
Result: stable attitude estimate that is accurate in both short bursts and over time.

*Barometric altitude:*
Air pressure decreases with altitude: roughly 12 Pa per meter near sea level.
Barometric altitude is accurate for relative height changes but drifts with weather pressure changes.
Combined with GPS altitude, it gives a smooth altitude trace.

**Output artifact:**
Per-timestamp IMU state: `{roll_deg, pitch_deg, yaw_deg, accel_mss, angular_rate_dps, baro_alt_m}`.
Derived motion label: `hover`, `ascending`, `descending`, `fast_translation`, `rotation`.

**Human focus:**
- Understand the difference between sensor-frame acceleration and world-frame acceleration (requires attitude first).
- Learn why IMU integration produces position estimates that are only useful for short intervals (seconds, not minutes).
- Know the complementary filter as the practical alternative to a full EKF for attitude.
- Understand why IMU and GPS must be time-synchronized: a 100 ms offset at 10 m/s flight speed is 1 meter of position error.

**Common failure modes:**
- No IMU sidecar → platform motion is unobservable; optical flow appears mixed.
- IMU not calibrated (factory default gains) → attitude estimates are systematically wrong.
- Vibration from motors → high-frequency noise in accelerometer; attitude filter oscillates.
- Gimbal isolation → IMU measures airframe motion but camera is stabilized; both must be logged.

---

<a id="step-17-atmospheric--environmental-sensing"></a>
## Step 17. Atmospheric / environmental sensing

**What it does:**
Ingest local weather data from an onboard environmental sensor or an external API: temperature, humidity, wind speed and direction, barometric pressure, UV index, precipitation.
Classify current atmospheric condition: clear, cloudy, foggy, windy, or rainy.
Flag frames where weather degrades sensor performance.

**Why it matters:**
Weather changes both what the mission can achieve and how reliably other sensors perform.
Rain degrades LiDAR and radar accuracy.
Fog reduces camera range and saturates thermal contrast.
High wind affects platform stability, introducing vibration in all sensors.
Knowing weather state lets the pipeline weight sensor outputs appropriately.

**Implementation:**
- Onboard sensor: temperature/humidity/pressure from flight controller log
- External API: `pipeline/net_utils.py` fetch from weather provider if GPS coordinates are known
- Condition classifier: rule-based from humidity, visibility, wind speed thresholds

**Key concepts:**

*Turbulence and vibration:*
At wind speeds above ~10 m/s, rotor-wing drones experience significant vibration.
This vibration appears as camera shake, IMU noise spikes, and GPS position jitter.
Weather context explains these artifacts without attributing them to sensor failure.

*Humidity and thermal contrast:*
High relative humidity increases atmospheric absorption in the LWIR band.
At very high humidity, thermal camera range decreases significantly; distant objects lose contrast.

*Visibility and range:*
Fog or haze reduces the range at which objects can be resolved visually.
The camera sees a flat, low-contrast scene beyond a certain range.
Knowing the predicted or measured visibility allows the pipeline to flag frames where depth estimates are likely wrong.

**Output artifact:**
Per-mission atmospheric summary: `{temperature_c, humidity_pct, wind_speed_mps, wind_dir_deg, visibility_km, condition_label}`.
Per-frame weather quality flag: `{sensor_degradation_risk}` for frames where conditions are likely to impair sensor accuracy.

**Human focus:**
- Understand how wind speed affects drone stability and what that means for frame quality.
- Know the atmospheric windows: wavelengths where the atmosphere is transparent (visible, NIR, LWIR) vs absorbing (MWIR in humid conditions).
- Learn when to trust vs discount sensor readings based on weather.

**Common failure modes:**
- No onboard weather sensor → conditions unknown; API fallback depends on GPS coordinates being valid.
- API rate limiting → weather not fetched; condition label defaults to "unknown".
- Pressure used for altitude (barometry) is confused with pressure used for weather: they are the same sensor but serve different purposes.

---

<a id="step-18-chemical--gas--radiation-sensing"></a>
## Step 18. Chemical / gas / radiation sensing

**What it does:**
Ingest measurements from gas sensors (CO, CO₂, methane, VOCs, O₃), radiation detectors (Geiger-Müller, scintillator), or chemical agent detectors.
Threshold-alert on readings above safety or operational limits.
Geo-tag anomalous readings with GPS coordinates for spatial mapping.

**Why it matters:**
These are often the highest-value "invisible" signals in a mission log.
A visual inspection of a refinery from above cannot detect a methane leak.
A radiation anomaly near an infrastructure target is not visible in any camera.
When present, these sensors have the clearest and most actionable output of any modality.

**Implementation:**
- Sensor sidecar ingestion from CSV, serial port log, or proprietary detector format
- Unit conversion and calibration: raw ADC → PPM (gas), CPM → µSv/h (radiation)
- Geo-tagging: align sensor timestamp to GPS track
- Threshold alerting: configurable per-sensor alert levels

**Key concepts:**

*Gas sensor operating principles:*
- Electrochemical: measures gas by the current produced when it reacts at an electrode. Accurate but slow (minutes to equilibrate).
- Metal oxide semiconductor (MOS): changes resistance in presence of gas. Fast but sensitive to humidity and temperature.
- Infrared absorption (NDIR): measures gas absorption of an IR beam. Very specific per gas type, not affected by humidity.

*Radiation detection:*
- Geiger-Müller tube: counts individual ionizing particles. Very sensitive at low cost. Does not distinguish particle types.
- Scintillation detector: measures energy spectrum of radiation. Can identify isotopes.
- CPM to dose conversion requires geometry and detector efficiency factors.

*Sensor cross-sensitivity:*
Most gas sensors respond to multiple species. A CO sensor also responds to hydrogen and VOCs.
Cross-sensitivity tables from the manufacturer must be used to correct readings.

**Output artifact:**
Per-sample sensor readings: `{timestamp, lat, lon, gas_ppm, radiation_usv_h, alert_triggered}`.
Spatial alert map: KML or GeoJSON file with alert locations overlaid on the mission GPS track.

**Human focus:**
- Understand electrochemical sensor warm-up time: cold readings during first 30-60 seconds are unreliable.
- Learn when MOS sensors are unreliable: high humidity, rain, or condensation.
- Know the difference between detection (is it there?) and quantification (how much?): most drone sensors do the former.
- Understand why geo-tagging precision matters: a 10-meter GPS error puts a gas plume in the wrong location.

**Common failure modes:**
- No sensor payload → step skipped; all chemical/radiation context is absent.
- Sensor not warmed up → first readings are falsely high or low.
- GPS-sensor timestamp mismatch → readings are geo-tagged to wrong locations.
- Wind plume dilution → sensor is downwind but concentration is below detection threshold even with an active source.

---

<a id="step-19-acoustic-sensing"></a>
## Step 19. Acoustic sensing

**What it does:**
Separate from ASR (Step 5, which handles speech): ingest audio and extract non-speech acoustic features.
Classify environmental sounds: vehicle engines, aircraft, gunfire, machinery, wildlife, water, wind.
Detect acoustic events with onset timestamps.

**Why it matters:**
Sound often confirms or contradicts what the camera shows.
A vehicle that looks stationary in the frame but has a running engine is audible.
An aircraft that has left the camera frame is still acoustically present.
Gunfire or impact sounds can precede visible effects by multiple frames.
Acoustic event timestamps provide an independent temporal anchor for correlating other sensors.

**Implementation:**
- Audio feature extraction: MFCC, mel spectrogram, chroma, spectral centroid
- Environmental sound classification: pretrained model (YAMNet, VGGish, BEATs)
- Onset detection: energy envelope + spectral flux thresholding
- Separation of ASR-band speech from environmental audio: VAD gating

**Key concepts:**

*MFCCs (Mel Frequency Cepstral Coefficients):*
The standard feature representation for audio classification.
Computed by: FFT → mel filterbank → log → discrete cosine transform.
MFCCs capture the spectral envelope of sound (how energy is distributed across frequency) in a perceptually-scaled way.

*Acoustic event detection vs scene classification:*
Event detection: something happened at time T (onset + offset). E.g., "gunfire at 12.3 s".
Scene classification: what kind of environment is this audio? E.g., "urban traffic". Operates on windows.
Both are useful; the pipeline uses both.

*Propeller wash and wind noise:*
Drone flights have high-energy low-frequency noise from propellers.
This masks other sounds at close range.
High-pass filtering or notch filtering removes the rotor fundamental frequency, revealing quieter sounds.

**Output artifact:**
Per-window acoustic features: `{dominant_class, confidence, event_list, onset_timestamps}`.
Acoustic timeline: list of detected events with onset/offset times and class labels.

**Human focus:**
- Understand MFCCs as the audio equivalent of visual feature descriptors.
- Learn the difference between sound event detection (temporal) and audio scene classification (global).
- Know the dominant failure mode: propeller noise from the drone itself masks almost everything else below 5 kHz.
- Understand why acoustic timestamps are valuable: they are independent of camera rate and GPS accuracy.

**Common failure modes:**
- Drone rotor noise overwhelms everything → classification fails unless rotor notch filter is applied.
- Low sample rate audio (8 kHz) from cheap microphone → cannot resolve frequencies above 4 kHz; most acoustic events are attenuated.
- Wind noise in microphone → broadband noise masks acoustic events.
- No external microphone (only onboard) → all sound is dominated by airframe vibration.

---

<a id="step-20-sensor-fusion-analysis"></a>
## Step 20. Sensor fusion analysis

**What it does:**
Time-align all sensor evidence collected in Steps 9-19 to a common timeline, merge them into a unified per-frame context block, resolve contradictions, and flag high-confidence anomalies.

**Why it matters:**
Independent sensors are useful.
Time-aligned sensors are much more useful: a radar detection at 120 m at 2.4 s can be linked to a LiDAR cluster at 2.4 s and a visual object detection at the same timestamp.
Agreement across modalities increases confidence.
Disagreement across modalities is informative: it either means one sensor is wrong or that something is genuinely strange.

**Implementation:**
- [`ssv_vdp/steps/common.py`](../../packages/ss-fusion/src/ssv_vdp/steps/common.py) — `VideoKnowledge` accumulator
- Timestamp alignment: each sensor stream is resampled or nearest-matched to the frame `t_sec` grid
- Lag tolerance: configurable per-sensor offset to account for processing delay

**Key concepts:**

*Timestamp alignment:*
Each sensor records at its own rate and clock.
The pipeline aligns everything to the video frame timestamps (`t_sec`) as the master clock.
A sensor reading within ±N seconds of a frame timestamp is considered "aligned" to that frame.
N varies per sensor: for LiDAR (10 Hz), ±0.05 s; for weather API (1/hour), ±1800 s.

*Fusion vs integration:*
Fusion: combine multiple sensor outputs into a single representation (e.g., weighted average of radar + LiDAR range estimates).
Integration: add a new independent observation alongside existing ones (e.g., acoustic event + visual detection at the same timestamp).
The pipeline mostly does integration (each sensor result is kept separately in `VideoKnowledge`) and lets Qwen reason across them.

*Missing data handling:*
When a sensor is absent or fails, the pipeline must not propagate nulls as if they were observations.
Missing data is explicitly marked absent; Qwen receives no context line for that sensor, rather than a confusing zero.

*Contradiction detection:*
When two sensors disagree (e.g., LiDAR says obstacle at 50 m, radar says nothing at 50 m), the pipeline logs the contradiction.
This is a signal for human review, not for automatic resolution.

**Output artifact:**
The `VideoKnowledge` accumulator with all sensor evidence merged.
For each frame `t_sec`: the full `context_for_frame(t_sec)` string that Qwen receives.
Contradiction log: list of timestamp + modality pairs where sensors disagreed.

**Human focus:**
- Trace one fused frame context from raw sensor sidecar inputs to the final `context_for_frame()` string.
- Understand timestamp alignment windows and how they interact with sensor update rates.
- Learn to distinguish genuine sensor disagreement from time synchronization error (same root cause, different interpretation).
- Know when fusion reduces uncertainty vs when it just adds noise.

**Common failure modes:**
- All sensors except RGB camera are absent → fusion step is a no-op; only camera evidence matters.
- Clock drift between sensors → temporal alignment fails; sensor readings are matched to wrong frames.
- Lag in slower sensors (weather API, GPS at 1 Hz) → readings appear stale; short events are missed.
- Fusing contradictory sensors without flagging → Qwen receives inconsistent context and produces unreliable output.

---

## What A Human Should Learn In This Phase

Do not try to become a specialist in every sensor first.
The practical study order:

1. Learn what each sensor is physically measuring (wavelength, electrical quantity, acoustic pressure, etc.).
2. Learn what each sensor cannot tell you (its blind spots, failure conditions, and physical limits).
3. Learn how the pipeline stores each sensor's evidence in `VideoKnowledge`.
4. Learn how fusion aligns them in time and what happens when they disagree.

The most valuable skill in this phase is understanding **sensor failure modes** — not just the happy path.
A system that trusts a failed sensor is more dangerous than one with no sensor at all.

## How To Extract Real Value From The Sensor Phase

If you are studying this phase as an engineer rather than as a sensor specialist,
use this pattern for every modality:

1. Name the physical measurement.
2. Name the likely artifact written by the step.
3. Name the dominant failure mode.
4. Name one later stage that benefits from the sensor.
5. Name one way the pipeline could be misled by the sensor.

That exercise produces much better understanding than memorizing sensor catalogs.

## Related Docs

- [Sensor fusion fundamentals](03_sensor_fusion_fundamentals.md)
- [Perception core: Steps 1-8](02_perception_core_steps_01_08.md)
- [Agentic knowledge flow](07_agentic_knowledge_flow.md)
- [Tracking and mapping: Steps 21-27](05_tracking_mapping_steps_21_27.md)
- [Pipeline architecture](../reference/pipeline.md)
- [Day-by-day syllabus](00_day_by_day_syllabus.md)

---

## Learning Resources — Sensors and Fusion (Steps 9-20)

Resources are ordered basics → deep dive. Physical measurement sections include the underlying principles because understanding what a sensor actually measures is the prerequisite for understanding when and why it fails.

---

### Step 9 — RF and Software-Defined Radio

**Why it matters:** RF sensing is the only modality that detects emitters invisible to all optical sensors. A drone carrying a radio transceiver, a communications relay, or a radar altimeter will appear in IQ data when it is invisible to RGB.

**Basics**
- Ettus Research, *USRP B200/B210 Getting Started Guide* — introduces IQ sampling, bandwidth, center frequency, and gain. Directly applicable to any SDR-based data source.
- Proakis & Salehi, *Communication Systems Engineering* (2nd ed., Prentice Hall, 2001). Chapter 2 (signals and systems in the frequency domain) and Chapter 6 (bandpass signals, IQ representation) are the minimum background.

**Core paper and library**
- TorchSig documentation and tutorials: [torchsig.com](https://torchsig.com). The library used in Step 9. Tutorial notebooks explain IQ preprocessing, spectrogram generation, and CNN-based signal classification.
- West & O'Shea, "Deep Architectures for Modulation Recognition" (2017). Seminal paper establishing deep learning for automatic modulation classification (AMC) — the direct predecessor to TorchSig's model zoo. [arxiv.org/abs/1703.09197](https://arxiv.org/abs/1703.09197)

**Deep dive**
- O'Shea & Hoydis, "An Introduction to Deep Learning for the Physical Layer" (2017). Treats the whole communication chain as an autoencoder — a unifying framework for understanding why DNNs work on IQ data. [arxiv.org/abs/1702.00832](https://arxiv.org/abs/1702.00832)

---

### Step 10 — Thermal and Infrared Imaging

**Why it matters:** A thermal camera detects emitted radiation (not reflected light), so it works at night, through smoke, and reveals heat signatures invisible to RGB. Spatial alignment with RGB is non-trivial: optics differ, and a calibration error of 5 pixels destroys cross-modal IoU thresholds.

**Basics**
- Vollmer & Möllmann, *Infrared Thermal Imaging: Fundamentals, Research and Applications* (Wiley-VCH, 2017). Chapters 1-3 cover the Planck curve, emissivity, and the 8-14 µm LWIR atmospheric window. The minimum physics background before working with radiometric cameras.
- FLIR Lepton 3.5 Application Note — explains radiometric vs non-radiometric output modes, FFC (flat-field correction), and pseudo-colour palettes. Available from Teledyne FLIR.

**Core paper**
- Treible et al., "CATS: A Color and Thermal Stereo Benchmark" (2017). Benchmark for RGB-thermal alignment and cross-modal object detection — directly relevant to the IoU ≥ 0.4 threshold used in `cross_modal_detections`. [arxiv.org/abs/1801.09558](https://arxiv.org/abs/1801.09558)

**Deep dive**
- Zhang et al., "Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks" (2019). State-of-the-art RGB-thermal fusion for pedestrian detection. Explains mid-fusion vs late-fusion architectures. [arxiv.org/abs/2007.03539](https://arxiv.org/abs/2007.03539)

---

### Step 11 — Multispectral and Hyperspectral Imaging

**Why it matters:** Spectral indices (NDVI, NDWI, NDSI) encode physical properties that are invisible to RGB and thermal. NDVI < 0 on vegetation that looked healthy to RGB is an early drought indicator — the kind of change detection that drives the pipeline's repeat-mission value.

**Basics**
- NASA Earth Observatory — "How to Interpret a False-Color Satellite Image." Builds intuition for what each band combination reveals before working with index formulas. [earthobservatory.nasa.gov](https://earthobservatory.nasa.gov/features/FalseColor)
- Rouse et al., "Monitoring Vegetation Systems in the Great Plains with ERTS" (1974). The original NDVI paper. Two pages. Read it — it establishes the (NIR - R) / (NIR + R) ratio that every modern precision-agriculture pipeline still uses.

**Core paper**
- Audebert et al., "Deep Learning for Classification of Hyperspectral Data: A Comparative Review" (2019). Comprehensive survey of CNN, RNN, and GAN approaches to hyperspectral classification — maps the current methods landscape. [arxiv.org/abs/1904.10674](https://arxiv.org/abs/1904.10674)

**Deep dive**
- Signoroni et al., "Deep Learning Meets Hyperspectral Image Analysis: A Multidisciplinary Review" (2019). Covers the continuum from multispectral (4-12 bands) to hyperspectral (100s of bands). [arxiv.org/abs/1903.02176](https://arxiv.org/abs/1903.02176)

---

### Step 12 — Event Cameras (Neuromorphic Sensing)

**Why it matters:** Event cameras respond to per-pixel brightness changes asynchronously, with microsecond latency and 120 dB dynamic range. They are the only sensor that can reliably track a propeller or a fast-moving object without motion blur — and they go "blind" to static scenes, which is the exact inverse of RGB failure modes.

**Basics**
- Lichtsteiner et al., "A 128×128 120 dB 15 µs Latency Asynchronous Temporal Contrast Vision Sensor" (2008). The original Dynamic Vision Sensor (DVS) paper. Four pages that explain the event polarity convention and the temporal contrast mechanism used in every event camera since.
- The Prophesee Metavision SDK documentation — the primary open-source toolkit for event data preprocessing, time-surface generation, and event-to-frame conversion. [docs.prophesee.ai](https://docs.prophesee.ai)

**Core survey**
- Gallego et al., "Event-based Vision: A Survey" (IEEE TPAMI, 2022). The definitive reference: covers event representations (time surfaces, voxel grids, event frames), reconstruction, optical flow, SLAM, and object recognition. 50 pages, but Sections 2-4 are the essential reading. [arxiv.org/abs/1904.08405](https://arxiv.org/abs/1904.08405)

**Deep dive**
- Gehrig et al., "E-RAFT: Dense Optical Flow from Event Cameras" (3DV, 2021). Shows how event streams enable sub-millisecond optical flow — the temporal resolution that makes event cameras compelling for drone guidance. [arxiv.org/abs/2108.10552](https://arxiv.org/abs/2108.10552)

---

### Step 13 — LiDAR and Active Ranging

**Why it matters:** LiDAR provides metric range — the only modality (alongside radar and sonar) with absolute depth at a specific point. A point cloud from one pass registered to a point cloud from a second pass directly reveals structural changes (new obstacles, missing structures).

**Basics**
- Velodyne LiDAR HDL-64E User Manual. Explains the rotating mirror mechanism, azimuth/elevation grid, return intensity, and dual-return modes. The physical model is the same for solid-state variants.
- PCL (Point Cloud Library) documentation: [pointclouds.org/documentation](https://pointclouds.org/documentation). The reference library for point cloud preprocessing, filtering (VoxelGrid, PassThrough), and ground-plane segmentation (RANSAC, progressive morphological filter).

**Core paper**
- Zhou & Tuzel, "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection" (2018). Explains voxel-based 3D feature extraction — the architecture class underlying all deep LiDAR detectors. Useful for understanding the data structure even if not running VoxelNet. [arxiv.org/abs/1711.06396](https://arxiv.org/abs/1711.06396)

**Deep dive**
- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (2017). The foundational paper for learning directly from unordered point clouds without voxelization. [arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)
- Zhang et al., "Deep Learning on Point Clouds: Towards 3D Perception" (survey, 2019). Maps all major 3D perception architectures (PointNet, VoxelNet, PointPillars, SECOND) and benchmarks. [arxiv.org/abs/1912.12033](https://arxiv.org/abs/1912.12033)

---

### Step 14 — Radar (FMCW, Doppler, SAR)

**Why it matters:** Radar penetrates fog, rain, and smoke. FMCW radar provides both range and radial velocity (Doppler) in the same chirp — the only sensor that simultaneously knows where an object is and how fast it is approaching. This is safety-critical for any weather-degraded mission.

**Basics**
- Mahafza, *Radar Systems Analysis and Design Using MATLAB* (3rd ed., CRC Press, 2013). Chapters 1-3 (radar equation, range resolution) and Chapter 7 (FMCW) are the minimum background. The range-resolution formula `δR = c/(2B)` and velocity resolution `δv = λ/(2T)` tell you what your radar hardware can and cannot resolve before writing any code.
- Texas Instruments, "Introduction to mmWave Sensing: FMCW Radars" (application note mmWAVE-SDK). Four pages, equations and intuition. Start here for FMCW specifics.

**Core paper**
- Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving" (2020). Not a radar paper per se, but the dataset and evaluation protocol that established how radar integrates with LiDAR and camera in a joint perception stack — directly relevant to the multi-modal fusion design in Step 20. [arxiv.org/abs/1929.02165](https://arxiv.org/abs/1929.02165) — note: correct arXiv ID is [1929.02165](https://arxiv.org/abs/1929.02165).

**Deep dive**
- Ouaknine et al., "CARRADA Dataset: Camera and Automotive Radar with Range-Angle-Doppler Annotations" (2020). Explains the range-Doppler, range-angle, and angle-Doppler tensor representations. [arxiv.org/abs/2005.01667](https://arxiv.org/abs/2005.01667)

---

### Step 15 — GNSS-R and Satellite Signal Reception

**Why it matters:** GPS position is the spatial anchor for every cross-mission comparison. Without GPS, change detection degrades to per-video comparisons without geographic registration. GNSS-R is a secondary path that uses reflected GNSS signals as a passive radar — surface roughness and soil moisture from opportunistic bistatic geometry.

**Basics**
- IS-GPS-200 (Interface Specification, current revision). The authoritative document on GPS signal structure, C/A code, navigation message, and dilution of precision (DOP). Section 3 is the key reference for understanding HDOP/VDOP.
- u-blox, "GPS Compendium" (application note GPS-X-02007). 175-page accessible introduction to GPS, differential correction, and RTK. Freely downloadable from u-blox.

**Core paper**
- Larson et al., "GPS Multipath and Its Use for Studying Earth Surfaces" (2009). Explains the GNSS-R technique — how multipath interference in geodetic receivers encodes soil moisture and snow depth. The physical basis for Step 15's `gnssr.bin` processing. [doi: 10.2174/1876825300902010001]

**Deep dive**
- Gleason & Gebre-Egziabher, *GNSS Applications and Methods* (Artech House, 2009). Covers SBAS, RTK, and GNSS-R in a unified framework — useful for understanding the full capability envelope.

---

### Step 16 — Inertial and Barometric Sensing

**Why it matters:** IMU data underpins every dead-reckoning pose estimate between GPS fixes. The bias drift challenge is quantitative: a consumer-grade MEMS gyroscope drifts ~1 deg/s — ten seconds of GPS gap means 10° of heading error if the filter is not correctly modelled.

**Basics**
- Titterton & Weston, *Strapdown Inertial Navigation Technology* (2nd ed., IET, 2004). Chapters 3-5 (accelerometers, gyroscopes, error sources). The standard reference for IMU noise models (Allan variance, bias instability, angle random walk).
- Madgwick, "An efficient orientation filter for inertial and inertial/magnetic sensor arrays" (2010). The complementary filter used in most AHRS (attitude and heading reference systems) running on embedded hardware. 10 pages. [x-io.co.uk/res/doc/madgwick_internal_report.pdf](https://x-io.co.uk/res/doc/madgwick_internal_report.pdf)

**Core paper**
- Forster et al., "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation" (IJRR, 2017). The preintegration theory used by VINS-Mono and ORB-SLAM3 — explains how to combine IMU and camera data without linearisation errors. [arxiv.org/abs/1512.02363](https://arxiv.org/abs/1512.02363)

**Deep dive**
- Barfoot, *State Estimation for Robotics* (Cambridge, 2017). The mathematically rigorous treatment of Lie groups (SO(3), SE(3)), batch nonlinear least squares, and the Kalman filter as a special case. Chapter 9 (IMU integration on manifold) is directly applicable.

---

### Step 17 — Atmospheric and Environmental Sensing

**Why it matters:** Weather conditions do not just affect human comfort — they quantitatively degrade every sensor. The `weather_factor` in the fusion confidence formula (`SENSOR_FUSION_MAX_LAG_MS`) exists because high humidity attenuates RF, wind causes motion blur, and rain floods radar with clutter. Ignoring weather produces overconfident fusion outputs.

**Basics**
- WMO, *Guide to Instruments and Methods of Observation* (WMO-No. 8, 2018). The authoritative reference for temperature, humidity, wind, and pressure measurement principles. Chapter 1 covers sensor types, calibration, and uncertainty quantification.
- Campbellsci application notes on atmospheric sensor calibration — practical notes for the specific sensor models commonly attached to UAS platforms.

**Deep dive**
- Heinle et al., "Automatic cloud classification of whole sky images" (2010). Example of the vision-based weather classification that feeds atmospheric context in Step 17. [doi: 10.5194/amt-3-557-2010]

---

### Step 18 — Chemical, Gas, and Radiation Sensing

**Why it matters:** A gas sensor reading of CO > 50 ppm or dose_rate > 1 µSv/h is a hard safety trigger in the pipeline — it overrides `al_score` and forces `al_tag = "needs_annotation"` regardless of visual uncertainty. Understanding sensor response time and cross-sensitivity is required to interpret these readings correctly.

**Basics**
- Figaro Inc., "Gas Sensor Fundamentals" (application note). Explains MOS (metal oxide semiconductor) sensor response curves, cross-sensitivity, and temperature compensation.
- Alphasense, "EC4 Series 4-electrode sensor design and circuit configuration." Explains electrochemical gas sensor operation for CO, NO₂, O₃ — the sensor types most common on small UAS.

**Core paper**
- De Vito et al., "On-field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario" (Sensors & Actuators B, 2008). Demonstrates the drift and cross-sensitivity problems that make raw gas sensor readings unreliable without co-located calibration — motivates why the pipeline logs raw readings and relies on threshold flags rather than calibrated ppm values.

---

### Step 19 — Acoustic Sensing

**Why it matters:** Acoustic sensing captures events invisible to all optical sensors: aircraft passing overhead, structural impacts, propeller distress. The primary challenge is propeller noise masking — at 2m distance from a quadrotor, propeller SPL (~100 dB) saturates most microphones at acoustic events of interest.

**Basics**
- Google, "YAMNet: A Pretrained Deep Net that Works with the AudioSet Dataset" — model card and API: [tfhub.dev/google/yamnet/1](https://tfhub.dev/google/yamnet/1). YAMNet classifies 521 audio event classes; understanding its class hierarchy is required for interpreting Step 19 outputs.
- Virtanen et al., "Computational Analysis of Sound Scenes and Events" (Springer, 2018). Chapter 2 (MFCC features) and Chapter 5 (sound event detection) cover the feature extraction and detection methods used in acoustic steps.

**Core paper**
- Gemmeke et al., "AudioSet: An ontology and human-labeled dataset for audio events" (ICASSP, 2017). The training dataset and ontology that YAMNet is trained on. Understanding the class hierarchy explains which outdoor acoustic events the model will and won't catch. [doi: 10.1109/ICASSP.2017.7952261]

**Deep dive**
- Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" (2020). The large-scale audio representation model family — the audio equivalent of CLIP for audio. [arxiv.org/abs/1912.10211](https://arxiv.org/abs/1912.10211)

---

### Step 20 — Sensor Fusion

**Why it matters:** This is the step where all sensor hypotheses are reconciled. The fusion output determines which frames are flagged for annotation and which pose source is trusted. Getting this wrong (e.g., trusting a GPS-only pose with HDOP > 4 in a canyon) cascades into incorrect global map entries and wrong change detections on future missions.

**Basics**
- Thrun, Burgard & Fox, *Probabilistic Robotics* (MIT Press, 2005). Chapters 3-4 (Gaussian filters, Kalman filter, extended Kalman filter). These are the conceptual foundation for every fusion filter in this pipeline. Available through most university library systems.
- Julier & Uhlmann, "A New Extension of the Kalman Filter to Nonlinear Systems" (1997). The UKF paper — explains why EKF linearisation fails for highly nonlinear pose estimation and what the unscented transform does instead. [doi: 10.1117/12.280797]

**Core paper**
- Geneva et al., "OpenVINS: A Research Platform for Visual-Inertial Estimation" (ICRA, 2020). Describes the EKF-based visual-inertial fusion system most similar to what `pose_source: ekf_imu_gps` represents. [arxiv.org/abs/1908.01012](https://arxiv.org/abs/1908.01012)

**Deep dive**
- Huang, "Review and Analysis of Multi-sensor Fusion Approaches for Autonomous Ground Vehicles" (2019). Survey of tight vs loose coupling, graph-based vs filter-based, and multi-hypothesis fusion — directly maps to the `pose_source` selection logic in the pipeline. [arxiv.org/abs/1906.02971](https://arxiv.org/abs/1906.02971)
- Sun et al., "Scalability in Perception for Autonomous Driving" (2020). The nuScenes/Waymo ecosystem that established multi-sensor detection evaluation metrics — context for why `cross_modal_agreement` is a meaningful quality signal. [arxiv.org/abs/1912.00844](https://arxiv.org/abs/1912.00844)

---

## Perspective Directions For Physical Modeling And Realtime Threats

Once you understand Steps 9-20 as context fusion, the next technical step is to ask:

- how do these modalities become explicit beliefs about the physical world?

The strongest directions are:

1. **State estimation**
   Promote sensor evidence into explicit platform, object, and scene state with uncertainty.
2. **Field estimation**
   Model hazards that are not objects: RF intensity, turbulence, smoke, dust, gas, radiation, and thermal gradients.
3. **Local threat inference**
   Turn aligned state into immediate safety outputs: collision risk, jamming likelihood, degraded pose trust, unstable terrain, nearby plume or hotspot.
4. **Global threat aggregation**
   Aggregate local alerts across time, space, and multiple nodes into sector-level risk, route advisories, and mission-level hazard maps.

The critical human lesson is:

- physical modeling is where multimodal evidence becomes operationally useful

Without this layer, the system may describe the world well but still fail to answer:

- what is dangerous now?
- what is getting more dangerous over the next minute?
- what should the platform or operator do differently?

For the dedicated roadmap, see
[18_future_directions.md](18_future_directions.md).

# Artificial Vision — Unified Terrestrial & Hybrid Flash Station Baseline (`av-scan`)

**Aligned with `av-ios` Commit `d5d4306` Cross-Repository Parity Specification**

---

## 1. Cross-Repository Ecosystem Parity Overview

All three scanning products in the Artificial Vision ecosystem (**AV scanAR**, **AV Scan iOS**, and **AV Scan / AV SLAM**) share a unified mathematical, state-machine, and on-disk station architecture across three operational paradigms:

| Feature / Subsystem | AV scanAR (Jetson Orin) | AV Scan iOS (`av-ios`) | AV Scan / AV SLAM (`av-scan`) |
| :--- | :--- | :--- | :--- |
| **Mode A — Continuous Mobile SLAM** | FAST-LIO2 + RTAB-Map mobile mapping | FAST-LIVO2 + Pose Graph mobile SLAM | Modular multi-sensor SLAM (`AiryLioBackend`, `GeminiRgbdBackend`, `VitureVioBackend`, `LingBotSlamBackend`) |
| **Mode B — Standalone Terrestrial Station** | $60^\circ \times 6$ stops ($360^\circ$), $5.0\text{ s}$ stationary accumulation per stop | $60^\circ \times 6$ stops ($360^\circ$), $5.0\text{ s}$ stationary accumulation per stop | $60^\circ \times 6$ stops ($360^\circ$), $5.0\text{ s}$ stationary accumulation per stop |
| **Mode C — Dynamic Hybrid "Flash Station"** | On-the-fly `Flash 60°` during active SLAM anchored at $T_{\text{world\_anchor}}$ | On-the-fly `Flash 60°` during active SLAM anchored at $T_{\text{world\_anchor}}$ | On-the-fly `Flash 60°` during active SLAM anchored at $T_{\text{world\_anchor}}$ |
| **Primary 3D Sensor** | RoboSense Airy (96-beam + 200 Hz IMU) | RoboSense Airy (96-beam + 200 Hz IMU) | RoboSense RS-Airy (`robosense_airy_01`), Orbbec Gemini 336L (`orbbec_gemini_336l_01`), VITURE Luma Ultra (`viture_luma_ultra_01`), Composite `scanar_01` |
| **LiDAR Mechanical Mount** | Calibrated pitch ($\sim 45^\circ$ forward tilt in `production_extrinsics.yaml` / `scanar_profile.json`) | Rigid clamp, $90^\circ$ pitch (forward-facing, cable down) | Profile-driven extrinsic matrix (`ScannerConfig` / `DeviceProfile` / `TerrestrialDeviceMountProfile`) |
| **RGB Camera** | Orbbec Gemini 336L RGB/Stereo | Apple iPad / iPhone wide-angle camera | Orbbec Gemini 336L / Generic UVC Camera (with automatic LiDAR reflectivity panorama fallback when no RGB camera is active) |
| **IMU Stillness Gate** | Dual-criterion gyro + accel gate (200 Hz Airy IMU) | Dual-criterion gyro + accel gate (200 Hz Airy IMU) | Dual-criterion gyro + accel gate with automatic $1.0\text{ s}$ LiDAR-timestamp settling fallback for LiDAR-only profiles |
| **Motorized Tripod Support** | Non-blocking Manual / Detent or Auto Bluetooth/Serial Motor | Non-blocking Manual / Detent or Auto Bluetooth/Serial Motor | Non-blocking Manual / Detent (`ManualRotationController`) or Auto Motor (`AutoMotorRotationController`) |
| **Station Coordinate Frame** | Right-handed metric: $+Z$ Up, $+X$ Forward at $0^\circ$, $+Y$ Left | Right-handed metric: $+Z$ Up, $+X$ Forward at $0^\circ$, $+Y$ Left | Right-handed metric: $+Z$ Up, $+X$ Forward at $0^\circ$, $+Y$ Left |
| **Output Deliverables** | `station_manifest.json` (v1.0), `pointcloud.ply`, `station.e57`, `cache/lod0.bin`, `raw/stop_*.ply` | `station_manifest.json` (v1.0), `pointcloud.ply`, `station.e57`, `cache/lod0.bin`, `raw/stop_*.ply` | `station_manifest.json` (v1.0), `pointcloud.ply`, `station.e57`, `cache/lod0.bin`, `raw/stop_*.ply`, `pano/pano_rgb.jpg`, `pano/coverage_mask.png` |

---

## 2. Detected Sensors, Profiles & Extrinsics in `av-scan`

### 2.1 Validated Sensor Profiles (`profiles/validated/`)
1. **RoboSense RS-Airy (`profiles/validated/airy_profile.json`, ID: `robosense_airy_01`)**:
   - **Streams**: `lidar_points` (10 Hz `XYZI_PCL`), `internal_imu` (200 Hz `6DOF_IMU`).
   - **Internal Extrinsics (`lidar_to_imu`)**: $t = [0.004250, 0.004180, -0.004460]\text{ m}$, $q = [0.71145376, -0.70271848, 0.0018789, 0.00409338]$.
   - **Canonical AV Frame**: `airy_base_link` ($+X$ Forward, $+Y$ Left, $+Z$ Up).
2. **scanAR Composite Hybrid Rig (`profiles/validated/scanar_profile.json`, ID: `scanar_01`)**:
   - **Component Profiles**: `robosense_airy_01` + `orbbec_gemini_336l_01`.
   - **Calibrated Extrinsic (`gemini_336l_to_airy_canonical`)**: Roll $-2.7^\circ$, Pitch $-51.0^\circ$, Yaw $79.5^\circ$:
     $$T_{\text{airy\_gemini}} = \begin{bmatrix} 0.114685 & -0.975492 & -0.187784 & 0.004 \\ 0.618782 & 0.218029 & -0.754700 & 0.168 \\ 0.777146 & -0.029645 & 0.628622 & -0.017 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
3. **Orbbec Gemini 336L (`profiles/validated/gemini_336l_profile.json`, ID: `orbbec_gemini_336l_01`)**:
   - Active stereo RGB-D + optical RGB stream (`RGB8`, 30 Hz).
4. **VITURE Luma Ultra (`profiles/validated/viture_ultra_profile.json`, ID: `viture_luma_ultra_01`)** & **Generic UVC Camera (`core/drivers/uvc/generic_uvc_sensor_adapter.hpp`)**.

---

## 3. Universal Station Mathematical Geometry & Forward Kinematic Chain

### 3.1 Station Coordinate System
Each terrestrial or hybrid flash station defines its right-handed metric origin at the tripod rotation axis:
- **$+Z_{\text{station}}$**: Upward along the gravity-aligned vertical tripod rotation axis.
- **$+X_{\text{station}}$**: Forward line-of-sight at the $\theta = 0^\circ$ reference stop.
- **$+Y_{\text{station}}$**: Left ($90^\circ$ counter-clockwise around $+Z_{\text{station}}$).

### 3.2 Forward Kinematic Chain & Camera Optical Frame Mapping
When sensor extrinsics $T_{\text{cam\_lidar}}$ transform points from the LiDAR frame into the Camera Optical Frame ($+X_{\text{cam}}$ Right, $+Y_{\text{cam}}$ Down, $+Z_{\text{cam}}$ Forward), `RotationGeometry::cameraOpticalToStationZero()` converts from the Camera Optical Frame into the Station Reference Frame at $\theta = 0^\circ$ before applying the turntable rotation $R_z(\theta_i)$ around $+Z_{\text{station}}$:

$$R_{\text{body\_cam}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$$

which maps:
- $+Z_{\text{cam}}$ (Forward) $\longrightarrow +X_{\text{station}}$ (Forward)
- $-X_{\text{cam}}$ (Left) $\longrightarrow +Y_{\text{station}}$ (Left)
- $-Y_{\text{cam}}$ (Up) $\longrightarrow +Z_{\text{station}}$ (Up)

The complete station transform for stop $i$ at angle $\theta_i$ is:

$$T_{\text{station\_lidar}}(\theta_i) = T_{\text{station\_axis}} \cdot R_z(\theta_i) \cdot T_{\text{axis\_body}} \cdot T_{\text{body\_cam}} \cdot T_{\text{cam\_lidar}}$$

When a profile stores extrinsics directly as $T_{\text{body\_lidar}}$ in an $X$-Forward, $Y$-Left, $Z$-Up body frame (`extrinsic_is_camera_optical = false`), `RotationGeometry` composes:

$$T_{\text{station\_lidar}}(\theta_i) = T_{\text{station\_axis}} \cdot R_z(\theta_i) \cdot T_{\text{axis\_body}} \cdot T_{\text{body\_lidar}}$$

In both representations, $R_z(\theta_i)$ strictly rotates horizontally around the vertical tripod axis $+Z_{\text{station}}$, never around the camera optical axis.

---

## 4. Station Capture Engine, Stillness Gate & Overlap Refinement

1. **Default Acquisition Parameters**:
   - `nominal_step_deg`: $60.0^\circ$ ($6$ stops for $360^\circ$: $0^\circ, 60^\circ, 120^\circ, 180^\circ, 240^\circ, 300^\circ$)
   - `total_expected_stops`: $6$
   - `accumulation_sec_per_stop`: $5.0\text{ s}$
   - `settling_delay_sec`: $1.0\text{ s}$
   - `max_motion_angular_vel_rads`: $0.05\text{ rad/s}$
   - **Cable Safety Guard**: Wraps angle at $360^\circ$ and returns motor to $0^\circ$ on station completion so cables are never twisted past one revolution.
2. **Dual-Criterion IMU Stillness Gate & Unified LiDAR Clock**:
   - Requires $\|\omega\| < 0.05\text{ rad/s}$ AND $|\|a\| - 9.81| < 0.85\text{ m/s}^2$ continuously for `settling_delay_sec` ($1.0\text{ s}$) before transitioning from `WAITING_FOR_STILLNESS` to `ACCUMULATING_STOP`.
   - If a bump or rotation occurs during `ACCUMULATING_STOP` ($\|\omega\| > 0.125\text{ rad/s}$ or $|\|a\| - 9.81| > 2.0\text{ m/s}^2$), immediately clears `current_stop_points_`, resets to `WAITING_FOR_STILLNESS`, and re-accumulates cleanly once still.
   - **Unified Clock**: Accumulation duration ($5.0\text{ s}$) is measured strictly using LiDAR frame timestamps (`frame_end_sec - stop_start_frame_sec >= 5.0`), never mixing IMU and LiDAR clock domains.
   - **LiDAR-Only Fallback**: If no IMU packets have been received, transitions automatically from `WAITING_FOR_STILLNESS` to `ACCUMULATING_STOP` after $1.0\text{ s}$ of LiDAR frame timestamps.
3. **Constrained 1-DOF Overlap Registration Refinement**:
   - Anchors Stop 0 at $0.0^\circ$ and refines each stop $k \ge 1$ against a $15\text{ cm}$ spatial voxel centroid hash map of stop $k - 1$ over $\Delta\theta \in [-3.0^\circ, +3.0^\circ]$ around $+Z_{\text{station}}$ with $\ge 25$ inliers.
4. **3D Azimuth & Elevation Coverage Analyzer**:
   - Bins merged station points ($r \ge 0.25\text{ m}$) into 36 azimuth sectors ($10^\circ$ each, $[0^\circ, 360^\circ)$) and 18 elevation sectors ($10^\circ$ each, $[-90^\circ, +90^\circ]$).

---

## 5. UI & Hybrid SLAM Hooks (`gui/qml_bridge.*` & `gui/qml/AppWindow.qml`)

- **Mode Selector**: `[Mobile SLAM | Terrestrial Station]` toggle in `ctrlBar`.
- **Non-Blocking Motor / Turntable Confirmation Dialog**:
  1. `Confirm Manual / Detent Head Attached` (never blocks terrestrial or flash station capture when no electronic motor is attached).
  2. `Connect Auto Bluetooth / Serial Motor` (programmatic motor control with automatic fallback to manual).
- **Dynamic Hybrid `Flash 60°` Button During Active SLAM**:
  - Locks current SLAM pose as $T_{\text{world\_anchor}} = T_{\text{world\_body}}(t_{\text{pause}})$.
  - Pauses SLAM trajectory propagation and routes stationary packets to `TerrestrialStationManager`.
  - Streams live accumulating station points transformed by $T_{\text{world\_anchor}}$ into the 3D/2D viewport alongside the global SLAM map.
  - On completion, writes `stations/flash_st_<id>/`, injects the verified high-weight anchor submap into the SLAM map and project session WAL, resets SLAM front-end pose to $T_{\text{world\_anchor}}$, and resumes mobile tracking.

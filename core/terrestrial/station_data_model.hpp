#pragma once

#include "core/terrestrial/rotation_geometry.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace av::core::terrestrial {

enum class RotationMode : uint8_t {
    MANUAL_INDEX = 0,    // Operator manually turns mount to 60 deg detent / nominal mark
    PRESET_INTERVAL = 1, // Turntable runs programmed manufacturer preset stops
    AUTOMATIC_MOTOR = 2  // Software commands Bluetooth / Serial motor with automatic fallback to manual
};

enum class MotionState : uint8_t {
    STATIONARY = 0,      // Scanner is stable and motionless; safe to accumulate
    SETTLING = 1,        // Motion recently ceased; waiting for tripod vibrations to damp
    ROTATING = 2,        // Motor or operator is currently rotating scanner
    DISTURBED = 3        // Excessive shock, wind buffeting, or tripod bump detected
};

enum class MotorVerificationStatus : uint8_t {
    UNVERIFIED = 0,
    MANUAL_CONFIRMED = 1,
    AUTOMATIC_CONNECTED = 2
};

enum class AcquisitionMode : uint8_t {
    MOBILE_SLAM = 0,          // Mode A: Continuous Mobile SLAM
    TERRESTRIAL_STATION = 1,  // Mode B: Standalone Terrestrial Station (60 deg x 6 stops at 5.0s)
    HYBRID_FLASH_STATION = 2  // Mode C: Single-shot stationary Flash Scan during active SLAM (no motor/rotation)
};

struct LidarPointXYZIRT {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float intensity{1.0f};
    uint8_t r{255};
    uint8_t g{255};
    uint8_t b{255};
    uint16_t ring{0};
    float time_offset_sec{0.0f};
    bool colorized{false};
};

struct LidarFrame {
    double timestamp_start_sec{0.0};
    double timestamp_end_sec{0.0};
    uint64_t sequence_id{0};
    std::string frame_id{"airy_lidar"};
    std::vector<LidarPointXYZIRT> points;
};

struct ImuMeasurement {
    double timestamp_sec{0.0};
    math::Vec3d gyro_rads{0.0, 0.0, 0.0};   // rad/s
    math::Vec3d accel_mps2{0.0, 0.0, 9.81}; // m/s^2
};

struct DeviceMountProfile {
    std::string profile_id{"robosense_airy_01"};
    std::string display_name{"AV Scan Rig (RoboSense RS-Airy)"};
    std::string host_app{"AV Scan"};
    std::string host_device{"Linux Workstation / Jetson Orin"};
    std::string sensor_model{"RoboSense Airy"};
    std::string sensor_serial{"AIRY-2024-99812"};

    // Calibrated extrinsic transform T_cam_lidar (Camera Optical Frame <- LiDAR Frame)
    // or T_body_lidar when extrinsic_is_camera_optical == false.
    math::SE3d T_cam_lidar{math::SE3d::Identity()};
    bool extrinsic_is_camera_optical{true};

    // Default cameraOpticalToStationZero().inverse() so identity body-aligned LiDAR frames
    // map +X_lidar -> +X_station, +Y_lidar -> +Y_station, +Z_lidar -> +Z_station when transformed
    // through T_body_cam * T_cam_lidar:
    static math::SE3d bodyAlignedAsCameraOpticalExtrinsic(const math::SE3d& T_body_lidar = math::SE3d::Identity()) {
        return RotationGeometry::cameraOpticalToStationZero().inverse() * T_body_lidar;
    }

    // Factory for scanAR composite rig (calibrated ~45-51 deg pitch from scanar_profile.json)
    static DeviceMountProfile makeScanArProfile() {
        DeviceMountProfile prof;
        prof.profile_id = "scanar_01";
        prof.display_name = "scanarMini scanAR (Airy + Gemini 336L)";
        prof.host_app = "AV Scan";
        prof.host_device = "Jetson Orin NX 16GB (scanar-01)";
        prof.sensor_model = "RoboSense Airy + Orbbec Gemini 336L";
        prof.sensor_serial = "AIRY-2024-99812+CPC6463000XW";

        // Calibrated transform_4x4 from profiles/validated/scanar_profile.json
        // gemini_336l_to_airy_canonical (T_airy_gemini); T_cam_lidar = (T_airy_gemini)^-1
        std::array<double, 16> T_airy_gemini_arr = {
            0.114685, -0.975492, -0.187784,  0.004,
            0.618782,  0.218029, -0.754700,  0.168,
            0.777146, -0.029645,  0.628622, -0.017,
            0.0,       0.0,       0.0,       1.0
        };
        math::SE3d T_airy_gemini = math::SE3d::FromRowMajor4x4(T_airy_gemini_arr);
        prof.T_cam_lidar = T_airy_gemini.inverse();
        prof.extrinsic_is_camera_optical = true;
        return prof;
    }

    // Factory for standalone RoboSense Airy LiDAR profile
    static DeviceMountProfile makeAiryProfile() {
        DeviceMountProfile prof;
        prof.profile_id = "robosense_airy_01";
        prof.display_name = "RoboSense RS-Airy";
        prof.host_app = "AV Scan";
        prof.host_device = "AV Scan Host";
        prof.sensor_model = "RoboSense Airy";
        prof.sensor_serial = "AIRY-2024-99812";
        prof.T_cam_lidar = bodyAlignedAsCameraOpticalExtrinsic(math::SE3d::Identity());
        prof.extrinsic_is_camera_optical = true;
        return prof;
    }

    // Factory for iOS 90-deg forward mount parity test
    static DeviceMountProfile makeIos90DegProfile() {
        DeviceMountProfile prof;
        prof.profile_id = "ipad_10th_gen_airy_rigid";
        prof.display_name = "iPad 10th Gen + RoboSense Airy";
        prof.host_app = "AV Scan";
        prof.host_device = "iPad 10th Gen (A2696)";
        prof.sensor_model = "RoboSense Airy";
        prof.sensor_serial = "08:48:57:09:82:89";
        // 90-deg forward pitch in body frame mapped into camera optical frame
        math::Quatd q_pitch90 = math::Quatd::FromEuler(0.0, 90.0 * math::kDegToRad, 0.0);
        math::SE3d T_body_lidar(math::SO3d(q_pitch90), math::Vec3d(0.0, 0.0, 0.0));
        prof.T_cam_lidar = bodyAlignedAsCameraOpticalExtrinsic(T_body_lidar);
        prof.extrinsic_is_camera_optical = true;
        return prof;
    }
};

struct StopObservation {
    uint32_t stop_index{0};
    double nominal_angle_deg{0.0};
    double measured_angle_deg{0.0};
    bool has_measured_angle{false};

    double timestamp_start_sec{0.0};
    double timestamp_end_sec{0.0};

    std::string raw_lidar_relpath{};
    std::string rgb_image_relpath{};

    std::vector<LidarPointXYZIRT> stationary_points;
    math::SE3d T_station_lidar{math::SE3d::Identity()};

    double registration_residual_rmse_m{0.0};
    bool registration_converged{false};
};

struct TerrestrialStationConfig {
    std::string project_id{"PROJECT_001"};
    std::string station_id{"st_001"};
    std::string station_name{"Station 01"};
    std::string mode{"terrestrial"};          // "terrestrial" or "flash_scan"

    // Single-shot Flash Scan during Mobile SLAM:
    // Accumulates stationary frames from the current pose ONLY (1 stop, no motor/turntable rotation).
    bool is_single_shot_flash{false};
    math::SE3d T_body_lidar{math::SE3d::Identity()};

    RotationMode rotation_mode{RotationMode::MANUAL_INDEX};
    double nominal_step_deg{60.0};            // Default 6-stop 360 degree scan (60 deg at 5.0 sec)
    size_t total_expected_stops{6};           // (360 / 60) = 6 stops (0, 60, 120, 180, 240, 300)
    double accumulation_sec_per_stop{5.0};    // 5.0 seconds stationary accumulation per stop
    double settling_delay_sec{1.0};           // 1.0 second continuous stillness required

    double max_motion_angular_vel_rads{0.05}; // Dual-criterion IMU gyro stillness threshold (0.05 rad/s)
    bool enable_overlap_refinement{true};
    bool enable_camera_capture{true};
};

struct TerrestrialStationMetadata {
    std::string schema_version{"1.0"};
    std::string project_id{"PROJECT_001"};
    std::string station_id{"st_001"};
    std::string station_name{"Station 01"};
    std::string mode{"terrestrial"};
    std::string created_utc{};

    std::string host_app{"AV Scan"};
    std::string host_device{"AV Scan Host"};
    std::string sensor_model{"RoboSense Airy"};
    std::string sensor_serial{"AIRY-2024-99812"};

    RotationMode rotation_mode{RotationMode::MANUAL_INDEX};
    double nominal_step_deg{60.0};
    size_t total_stops{0};
    size_t expected_stops{6};
    double accumulation_sec_per_stop{5.0};
    size_t total_merged_points{0};

    double overall_rmse_m{0.0};
    bool is_closed_360{false};
    double azimuth_coverage_ratio{1.0};
    double elevation_coverage_ratio{0.85};
    std::string coverage_warning{};

    std::string pointcloud_ply_relpath{"pointcloud.ply"};
    std::string pointcloud_lod0_relpath{"cache/lod0.bin"};
    std::string station_e57_relpath{"station.e57"};
    std::string panorama_rgb_relpath{"pano/pano_rgb.jpg"};
    std::string coverage_mask_relpath{"pano/coverage_mask.png"};
};

} // namespace av::core::terrestrial

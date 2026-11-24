/**
 * @file test_terrestrial_station.cpp
 * @brief Comprehensive unit & conformance test suite for AV Scan Terrestrial Station Mode (Mode B)
 *        and Dynamic Hybrid "Flash Station" Mode (Mode C), aligned with AV Scan iOS (commit d5d4306).
 *
 * Tests:
 *   1. Forward Kinematic Chain & Coordinate Frame Mapping (Camera Optical + Body Frame -> Station Frame)
 *   2. Dual-Criterion IMU Stillness Gate, Bump Reset & LiDAR-Only Settling Timer Fallback
 *   3. Full 6-Stop 360-deg Station Capture, 1-DOF Overlap Refinement & On-Disk File/Binary Export
 *   4. E57 ASTM E2807 Page/CRC32C, Binary PLY & LOD0 (15-byte record) Round-Trip Integrity
 *   5. Non-Blocking Rotation Controllers (Manual/Detent + Auto Motor Fallback + 360-deg Cable Safety)
 */

#include "avscan/terrestrial/IRotationController.hpp"
#include "avscan/terrestrial/RotationGeometry.hpp"
#include "avscan/terrestrial/StationDataModel.hpp"
#include "avscan/terrestrial/TerrestrialStationManager.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace av::core::terrestrial;

namespace {

int g_tests_passed = 0;
int g_tests_failed = 0;

#define EXPECT_TRUE(cond, msg)                                                              \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "[FAIL] " << __FUNCTION__ << ":" << __LINE__ << " - " << (msg)     \
                      << " (" #cond ")\n";                                                  \
            ++g_tests_failed;                                                               \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

#define EXPECT_NEAR(val1, val2, tol, msg)                                                   \
    do {                                                                                    \
        const double _v1 = static_cast<double>(val1);                                       \
        const double _v2 = static_cast<double>(val2);                                       \
        if (std::fabs(_v1 - _v2) > (tol)) {                                                 \
            std::cerr << "[FAIL] " << __FUNCTION__ << ":" << __LINE__ << " - " << (msg)     \
                      << " (got " << _v1 << ", expected " << _v2 << " +/- " << (tol)        \
                      << ")\n";                                                             \
            ++g_tests_failed;                                                               \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

// 1. Test Forward Kinematic Chain & Coordinate Frame Mapping
bool testForwardKinematicChainAndCoordinateFrames() {
    std::cout << "[RUN ] testForwardKinematicChainAndCoordinateFrames\n";

    // Verify R_body_cam maps Camera Optical (+X Right, +Y Down, +Z Forward)
    // to Station Reference Frame at theta = 0 deg (+X Forward, +Y Left, +Z Up)
    const math::SE3d T_body_cam = RotationGeometry::cameraOpticalToStationZero();

    const math::Vec3d p_cam_fwd(0.0, 0.0, 5.0);   // +Z_cam (Forward)
    const math::Vec3d p_cam_right(2.0, 0.0, 0.0); // +X_cam (Right -> -Y_station)
    const math::Vec3d p_cam_down(0.0, 3.0, 0.0);  // +Y_cam (Down  -> -Z_station)

    const math::Vec3d p_st_fwd   = T_body_cam * p_cam_fwd;
    const math::Vec3d p_st_right = T_body_cam * p_cam_right;
    const math::Vec3d p_st_down  = T_body_cam * p_cam_down;

    EXPECT_NEAR(p_st_fwd.x, 5.0, 1e-9, "+Z_cam maps to +X_station");
    EXPECT_NEAR(p_st_fwd.y, 0.0, 1e-9, "+Z_cam Y is 0");
    EXPECT_NEAR(p_st_fwd.z, 0.0, 1e-9, "+Z_cam Z is 0");

    EXPECT_NEAR(p_st_right.x, 0.0, 1e-9, "+X_cam X is 0");
    EXPECT_NEAR(p_st_right.y, -2.0, 1e-9, "+X_cam maps to -Y_station");
    EXPECT_NEAR(p_st_right.z, 0.0, 1e-9, "+X_cam Z is 0");

    EXPECT_NEAR(p_st_down.x, 0.0, 1e-9, "+Y_cam X is 0");
    EXPECT_NEAR(p_st_down.y, 0.0, 1e-9, "+Y_cam Y is 0");
    EXPECT_NEAR(p_st_down.z, -3.0, 1e-9, "+Y_cam maps to -Z_station");

    // Verify 90-deg pitch iOS profile (where +Z_lidar points along +Z_cam forward)
    // rotates horizontally in the XY plane when turntable rotates around +Z_station,
    // never rolling around the optical axis!
    const DeviceMountProfile ios_profile = DeviceMountProfile::makeIos90DegProfile();
    TripodAxisCalibration calib;
    calib.T_axis_body_offset_m = math::Vec3d(0.0, 0.0, 0.05);

    const math::Vec3d pt_lidar_forward(0.0, 0.0, 10.0); // 10m out along LiDAR Z (which is forward in iOS 90-deg mount)
    const math::SE3d T_0 = RotationGeometry::computeStationLidarTransform(
        0.0, ios_profile.T_cam_lidar, calib, ios_profile.extrinsic_is_camera_optical);
    const math::SE3d T_90 = RotationGeometry::computeStationLidarTransform(
        90.0, ios_profile.T_cam_lidar, calib, ios_profile.extrinsic_is_camera_optical);
    const math::SE3d T_180 = RotationGeometry::computeStationLidarTransform(
        180.0, ios_profile.T_cam_lidar, calib, ios_profile.extrinsic_is_camera_optical);

    const math::Vec3d p0   = T_0 * pt_lidar_forward;
    const math::Vec3d p90  = T_90 * pt_lidar_forward;
    const math::Vec3d p180 = T_180 * pt_lidar_forward;

    // At 0 deg: point should lie along +X_station (~10m) at constant height Z (+0.05m offset)
    EXPECT_NEAR(p0.x, 10.0, 1e-6, "At 0 deg, forward ray is along +X_station");
    EXPECT_NEAR(p0.y, 0.0, 1e-6, "At 0 deg, Y_station is 0");
    EXPECT_NEAR(p0.z, 0.05, 1e-6, "At 0 deg, Z_station is tripod height offset 0.05m");

    // At 90 deg CCW around +Z_station: point should lie along +Y_station (~10m) at the EXACT SAME Z height
    EXPECT_NEAR(p90.x, 0.0, 1e-6, "At 90 deg, forward ray X_station is 0");
    EXPECT_NEAR(p90.y, 10.0, 1e-6, "At 90 deg, forward ray is along +Y_station");
    EXPECT_NEAR(p90.z, 0.05, 1e-6, "At 90 deg, Z_station remains invariant at 0.05m");

    // At 180 deg CCW around +Z_station: point should lie along -X_station (-10m) at the EXACT SAME Z height
    EXPECT_NEAR(p180.x, -10.0, 1e-6, "At 180 deg, forward ray is along -X_station");
    EXPECT_NEAR(p180.y, 0.0, 1e-6, "At 180 deg, Y_station is 0");
    EXPECT_NEAR(p180.z, 0.05, 1e-6, "At 180 deg, Z_station remains invariant at 0.05m");

    // Verify arbitrary runtime-calibrated variable mount extrinsic T_cam_lidar
    // (detachable mount with non-zero roll/pitch/yaw + translation) preserves Z across 0/90/180 deg
    const math::SE3d T_runtime_var(
        math::SO3d(math::Quatd::FromEuler(0.12, -0.48, 0.25)),
        math::Vec3d(0.03, -0.02, 0.07));
    const math::Vec3d p_var_0   = RotationGeometry::computeStationLidarTransform(0.0, T_runtime_var, calib, true) * math::Vec3d(3.0, 1.0, 2.0);
    const math::Vec3d p_var_90  = RotationGeometry::computeStationLidarTransform(90.0, T_runtime_var, calib, true) * math::Vec3d(3.0, 1.0, 2.0);
    const math::Vec3d p_var_180 = RotationGeometry::computeStationLidarTransform(180.0, T_runtime_var, calib, true) * math::Vec3d(3.0, 1.0, 2.0);
    EXPECT_NEAR(p_var_90.z, p_var_0.z, 1e-6, "Variable runtime mount preserves Z at 90 deg");
    EXPECT_NEAR(p_var_180.z, p_var_0.z, 1e-6, "Variable runtime mount preserves Z at 180 deg");

    // Also test ScanAR 45-deg forward tilt profile and Airy upright profile:
    // rotating through 6 stops (0, 60, 120, 180, 240, 300) must preserve Z_station!
    const DeviceMountProfile scanar_profile = DeviceMountProfile::makeScanArProfile();
    const DeviceMountProfile airy_profile   = DeviceMountProfile::makeAiryProfile();
    const math::Vec3d sample_pt(4.0, -1.5, 2.0);

    const math::SE3d T_scanar_0 = RotationGeometry::computeStationLidarTransform(
        0.0, scanar_profile.T_cam_lidar, calib, scanar_profile.extrinsic_is_camera_optical);
    const math::SE3d T_airy_0 = RotationGeometry::computeStationLidarTransform(
        0.0, airy_profile.T_cam_lidar, calib, airy_profile.extrinsic_is_camera_optical);
    const math::Vec3d scanar_0 = RotationGeometry::transformPointToStation(sample_pt, T_scanar_0);
    const math::Vec3d airy_0   = RotationGeometry::transformPointToStation(sample_pt, T_airy_0);
    const double r_scanar_xy = std::hypot(scanar_0.x, scanar_0.y);
    const double r_airy_xy   = std::hypot(airy_0.x, airy_0.y);

    for (int i = 1; i < 6; ++i) {
        const double deg = i * 60.0;
        const math::SE3d T_s = RotationGeometry::computeStationLidarTransform(
            deg, scanar_profile.T_cam_lidar, calib, scanar_profile.extrinsic_is_camera_optical);
        const math::SE3d T_a = RotationGeometry::computeStationLidarTransform(
            deg, airy_profile.T_cam_lidar, calib, airy_profile.extrinsic_is_camera_optical);
        const math::Vec3d p_s = RotationGeometry::transformPointToStation(sample_pt, T_s);
        const math::Vec3d p_a = RotationGeometry::transformPointToStation(sample_pt, T_a);

        EXPECT_NEAR(p_s.z, scanar_0.z, 1e-6, "ScanAR 45-deg profile preserves Z across turntable stops");
        EXPECT_NEAR(std::hypot(p_s.x, p_s.y), r_scanar_xy, 1e-6, "ScanAR XY radius preserved");
        EXPECT_NEAR(p_a.z, airy_0.z, 1e-6, "Airy body profile preserves Z across turntable stops");
        EXPECT_NEAR(std::hypot(p_a.x, p_a.y), r_airy_xy, 1e-6, "Airy XY radius preserved");
    }

    std::cout << "[PASS] testForwardKinematicChainAndCoordinateFrames\n";
    ++g_tests_passed;
    return true;
}

// 2. Test Dual-Criterion IMU Stillness Gate, Bump Reset & LiDAR-Only Fallback
bool testStillnessGateBumpResetAndLidarFallback() {
    std::cout << "[RUN ] testStillnessGateBumpResetAndLidarFallback\n";

    const fs::path temp_root = fs::temp_directory_path() / "av_scan_test_stillness";
    fs::remove_all(temp_root);

    TerrestrialStationConfig cfg;
    cfg.project_id                = "proj_stillness";
    cfg.station_id                = "st_still_001";
    cfg.nominal_step_deg          = 60.0;
    cfg.total_expected_stops      = 6;
    cfg.accumulation_sec_per_stop = 5.0;
    cfg.settling_delay_sec        = 1.0;

    TerrestrialStationManager mgr;
    const DeviceMountProfile profile = DeviceMountProfile::makeAiryProfile();
    EXPECT_TRUE(mgr.startNewStation(temp_root.string(), cfg, profile), "startNewStation succeeded");
    mgr.beginCurrentStopAccumulation();
    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::WAITING_FOR_STILLNESS,
                "Initial state is WAITING_FOR_STILLNESS");

    // 2a. Feed high-motion IMU samples -> must stay in WAITING_FOR_STILLNESS
    for (int i = 0; i < 15; ++i) {
        ImuMeasurement imu;
        imu.timestamp_sec = i * 0.1;
        imu.gyro_rads     = math::Vec3d(0.35, 0.0, 0.0); // > 0.05 rad/s threshold
        imu.accel_mps2    = math::Vec3d(0.0, 0.0, 9.81);
        mgr.pushImuMeasurement(imu);
    }
    EXPECT_TRUE(!mgr.isStationary(), "Scanner is not stationary during 0.35 rad/s rotation");
    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::WAITING_FOR_STILLNESS,
                "Remains in WAITING_FOR_STILLNESS while moving");

    // 2b. Feed stationary IMU samples for >= 1.0s settling_delay_sec
    for (int i = 0; i <= 12; ++i) {
        ImuMeasurement imu;
        imu.timestamp_sec = 2.0 + i * 0.1;
        imu.gyro_rads     = math::Vec3d(0.005, -0.004, 0.002); // ||w|| < 0.05 rad/s
        imu.accel_mps2    = math::Vec3d(0.0, 0.0, 9.81);       // | ||a|| - 9.81 | < 0.85 m/s^2
        mgr.pushImuMeasurement(imu);
    }
    EXPECT_TRUE(mgr.isStationary(), "Scanner is stationary after 1.2s of quiet IMU");

    // Push first LiDAR frame at t = 10.0s -> transitions to ACCUMULATING_STOP
    LidarFrame frame1;
    frame1.timestamp_start_sec = 10.0;
    frame1.timestamp_end_sec   = 10.1;
    LidarPointXYZIRT pt;
    pt.x = 3.0f; pt.y = 1.0f; pt.z = 0.5f; pt.intensity = 120.0f;
    frame1.points.push_back(pt);
    mgr.pushLidarFrame(frame1);
    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::ACCUMULATING_STOP,
                "Transitioned to ACCUMULATING_STOP");

    // Push LiDAR frame at t = 12.5s (halfway through 5.0s window)
    LidarFrame frame2 = frame1;
    frame2.timestamp_start_sec = 12.5;
    frame2.timestamp_end_sec   = 12.6;
    mgr.pushLidarFrame(frame2);
    EXPECT_NEAR(mgr.getCurrentAccumulationProgress(), 2.6 / 5.0, 0.05, "Progress is ~52%");

    // 2c. Simulate a bump mid-accumulation (||w|| > 0.125 rad/s)
    ImuMeasurement bump_imu;
    bump_imu.timestamp_sec = 12.7;
    bump_imu.gyro_rads     = math::Vec3d(0.0, 0.0, 0.45); // > 0.125 rad/s bump threshold
    bump_imu.accel_mps2    = math::Vec3d(0.0, 0.0, 9.81);
    mgr.pushImuMeasurement(bump_imu);

    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::WAITING_FOR_STILLNESS,
                "Bump mid-accumulation resets state to WAITING_FOR_STILLNESS");
    EXPECT_NEAR(mgr.getCurrentAccumulationProgress(), 0.0, 1e-6,
                "Bump clears partial stop accumulation progress to 0.0");

    // 2d. Test LiDAR-only sensor profile (no IMU packets at all)
    TerrestrialStationManager lidar_only_mgr;
    cfg.station_id = "st_lidar_only_001";
    EXPECT_TRUE(lidar_only_mgr.startNewStation(temp_root.string(), cfg, profile),
                "startNewStation for LiDAR-only succeeded");
    lidar_only_mgr.beginCurrentStopAccumulation();

    // Feed LiDAR frames over 1.1s without any IMU packets -> automatic settling fallback
    for (int i = 0; i <= 11; ++i) {
        LidarFrame lf;
        lf.timestamp_start_sec = 100.0 + i * 0.1;
        lf.timestamp_end_sec   = 100.0 + i * 0.1 + 0.05;
        lf.points.push_back(pt);
        lidar_only_mgr.pushLidarFrame(lf);
    }
    EXPECT_TRUE(lidar_only_mgr.getCaptureState() == TerrestrialCaptureState::ACCUMULATING_STOP,
                "LiDAR-only fallback transitions to ACCUMULATING_STOP after 1.0s of LiDAR frames");

    // Feed up to 5.0s of accumulation after transition -> auto-completes stop
    for (int i = 12; i <= 65; ++i) {
        LidarFrame lf;
        lf.timestamp_start_sec = 100.0 + i * 0.1;
        lf.timestamp_end_sec   = 100.0 + i * 0.1 + 0.05;
        lf.points.push_back(pt);
        lidar_only_mgr.pushLidarFrame(lf);
        if (lidar_only_mgr.getCaptureState() == TerrestrialCaptureState::STOP_COMPLETED) {
            break;
        }
    }
    EXPECT_TRUE(lidar_only_mgr.getCaptureState() == TerrestrialCaptureState::STOP_COMPLETED,
                "Stop completes after 5.0s of LiDAR frame timestamps");

    fs::remove_all(temp_root);
    std::cout << "[PASS] testStillnessGateBumpResetAndLidarFallback\n";
    ++g_tests_passed;
    return true;
}

// 3 & 4. Test Full 6-Stop 360-deg Station Capture, 1-DOF Overlap Refinement, and On-Disk Outputs
bool testFullSixStopStationRefinementAndExports() {
    std::cout << "[RUN ] testFullSixStopStationRefinementAndExports\n";

    const fs::path temp_root = fs::temp_directory_path() / "av_scan_test_full_station";
    fs::remove_all(temp_root);

    TerrestrialStationConfig cfg;
    cfg.project_id                = "proj_terrestrial_full";
    cfg.station_id                = "station_0001";
    cfg.station_name              = "Lobby Center Station 1";
    cfg.mode                      = "terrestrial";
    cfg.nominal_step_deg          = 60.0;
    cfg.total_expected_stops      = 6;
    cfg.accumulation_sec_per_stop = 5.0;
    cfg.settling_delay_sec        = 1.0;
    cfg.enable_overlap_refinement = true;
    cfg.enable_camera_capture     = true;

    TerrestrialStationManager mgr;
    const DeviceMountProfile profile = DeviceMountProfile::makeAiryProfile();
    EXPECT_TRUE(mgr.startNewStation(temp_root.string(), cfg, profile), "startNewStation succeeded");

    // Generate a static room layout in Station Frame (octagonal room walls + floor + ceiling features)
    struct WorldPoint {
        math::Vec3d pos;
        uint8_t r, g, b;
        float intensity;
    };
    std::vector<WorldPoint> world_scene;
    for (int az_deg = 0; az_deg < 360; az_deg += 4) {
        for (int el_deg = -35; el_deg <= 55; el_deg += 5) {
            const double az = az_deg * math::kDegToRad;
            const double el = el_deg * math::kDegToRad;
            // Slightly asymmetric room radius so 1-DOF overlap alignment has strong geometric lock
            const double radius = 5.0 + 0.6 * std::cos(2.0 * az) + 0.3 * std::sin(3.0 * az);
            WorldPoint wp;
            wp.pos = math::Vec3d(
                radius * std::cos(el) * std::cos(az),
                radius * std::cos(el) * std::sin(az),
                radius * std::sin(el) + 0.05);
            wp.r         = static_cast<uint8_t>((az_deg * 255) / 360);
            wp.g         = static_cast<uint8_t>(((el_deg + 45) * 255) / 100);
            wp.b         = 180;
            wp.intensity = 100.0f + static_cast<float>(az_deg % 100);
            world_scene.push_back(wp);
        }
    }

    // Synthetic physical turntable detent offsets per stop, including +1.5 deg error on Stop 1
    // to verify constrained 1-DOF overlap refinement recovers 61.5 deg within <= 0.35 deg
    const double true_angle_offsets_deg[6] = {0.0, 1.5, -0.4, 0.3, -0.5, 0.4};
    TripodAxisCalibration axis_calib;

    for (int stop = 0; stop < 6; ++stop) {
        const double nominal_deg = stop * 60.0;
        const double true_deg    = nominal_deg + true_angle_offsets_deg[stop];

        mgr.beginCurrentStopAccumulation();

        // Feed quiet IMU for 1.1s
        const double base_t = stop * 20.0;
        for (int k = 0; k <= 11; ++k) {
            ImuMeasurement imu;
            imu.timestamp_sec = base_t + k * 0.1;
            imu.gyro_rads     = math::Vec3d(0.0, 0.0, 0.0);
            imu.accel_mps2    = math::Vec3d(0.0, 0.0, 9.81);
            mgr.pushImuMeasurement(imu);
        }

        // Transform world scene into the LiDAR sensor frame at `true_deg`
        const math::SE3d T_station_lidar_true = RotationGeometry::computeStationLidarTransform(
            true_deg, profile.T_cam_lidar, axis_calib, profile.extrinsic_is_camera_optical);
        const math::SE3d T_lidar_station_true = T_station_lidar_true.inverse();

        LidarFrame frame;
        frame.frame_id = static_cast<uint64_t>(stop + 1);
        for (const auto& wp : world_scene) {
            const math::Vec3d p_lidar = T_lidar_station_true * wp.pos;
            LidarPointXYZIRT pt;
            pt.x         = static_cast<float>(p_lidar.x);
            pt.y         = static_cast<float>(p_lidar.y);
            pt.z         = static_cast<float>(p_lidar.z);
            pt.intensity = wp.intensity;
            pt.r         = wp.r;
            pt.g         = wp.g;
            pt.b         = wp.b;
            pt.colorized = true;
            frame.points.push_back(pt);
        }

        // Push start and end LiDAR frames spanning 5.1 seconds
        frame.timestamp_start_sec = base_t + 2.0;
        frame.timestamp_end_sec   = base_t + 2.1;
        mgr.pushLidarFrame(frame);

        // Push synthetic RGB frame for this stop
        EXPECT_TRUE(!mgr.hasCapturedStopRgb(), "RGB flag initially false for stop");
        std::vector<uint8_t> rgb(64 * 32 * 3, static_cast<uint8_t>(40 + stop * 30));
        mgr.pushCameraRgbBuffer(rgb, 64, 32);
        EXPECT_TRUE(mgr.hasCapturedStopRgb(), "RGB flag becomes true after pushCameraRgbBuffer");

        LidarFrame frame_end = frame;
        frame_end.timestamp_start_sec = base_t + 7.0;
        frame_end.timestamp_end_sec   = base_t + 7.1; // 7.1 - 2.0 = 5.1s >= 5.0s
        mgr.pushLidarFrame(frame_end);

        EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::STOP_COMPLETED,
                    "Stop auto-completed after 5.0s accumulation");

        // Verify raw checkpoint PLY and RGB image were persisted immediately
        char raw_ply_name[64];
        std::snprintf(raw_ply_name, sizeof(raw_ply_name), "stop_%03d_%03ddeg.ply",
                      stop, static_cast<int>(nominal_deg));
        const fs::path raw_ply_path = fs::path(mgr.getStationDirectory()) / "raw" / raw_ply_name;
        EXPECT_TRUE(fs::exists(raw_ply_path), "Per-stop raw checkpoint PLY exists on disk");

        char raw_img_name[64];
        std::snprintf(raw_img_name, sizeof(raw_img_name), "stop_%03d.jpg", stop);
        const fs::path raw_img_path = fs::path(mgr.getStationDirectory()) / "raw" / "images" / raw_img_name;
        EXPECT_TRUE(fs::exists(raw_img_path), "Per-stop raw RGB JPEG exists on disk");

        if (stop < 5) {
            EXPECT_TRUE(mgr.advanceToNextStop(), "advanceToNextStop succeeded");
        }
    }

    // Finalize & save station
    std::string err_msg;
    EXPECT_TRUE(mgr.finalizeAndSaveStation(err_msg),
                ("finalizeAndSaveStation succeeded: " + err_msg).c_str());
    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::STATION_COMPLETED,
                "State is STATION_COMPLETED");

    // Verify 1-DOF overlap refinement converged and recovered Stop 1's +1.5 deg detent error within <= 0.35 deg
    const auto& stops = mgr.getCompletedStops();
    EXPECT_TRUE(stops.size() == 6, "All 6 stops recorded");
    EXPECT_NEAR(stops[1].measured_angle_deg, 61.5, 0.35,
                "Stop 1 recovers synthetic +1.5 deg detent error (61.5 deg) within <= 0.35 deg");
    for (size_t i = 1; i < stops.size(); ++i) {
        EXPECT_TRUE(stops[i].registration_converged, "Stop 1-DOF overlap refinement converged");
        EXPECT_TRUE(stops[i].registration_residual_rmse_m < 0.15,
                    "Stop registration RMSE is within sub-voxel tolerance");
    }

    // Verify coverage metrics
    const TerrestrialStationMetadata meta = mgr.getStationMetadata();
    EXPECT_TRUE(meta.is_closed_360, "Station is marked as closed 360");
    EXPECT_TRUE(meta.azimuth_coverage_ratio >= 0.95, "Azimuth coverage >= 95% across 36 sectors");
    EXPECT_TRUE(meta.elevation_coverage_ratio > 0.30, "Elevation coverage > 30%");

    // Verify all required on-disk artifacts exist
    const fs::path st_dir(mgr.getStationDirectory());
    const fs::path manifest_path = st_dir / "station_manifest.json";
    const fs::path ply_path      = st_dir / "pointcloud.ply";
    const fs::path e57_path      = st_dir / "station.e57";
    const fs::path lod0_path     = st_dir / "cache" / "lod0.bin";
    const fs::path pano_path     = st_dir / "pano" / "pano_rgb.jpg";
    const fs::path mask_path     = st_dir / "pano" / "coverage_mask.png";
    const fs::path keyplan_path  = st_dir / "floors" / "floor_0_keyplan.png";

    EXPECT_TRUE(fs::exists(manifest_path), "station_manifest.json exists");
    EXPECT_TRUE(fs::exists(ply_path), "pointcloud.ply exists");
    EXPECT_TRUE(fs::exists(e57_path), "station.e57 exists");
    EXPECT_TRUE(fs::exists(lod0_path), "cache/lod0.bin exists");
    EXPECT_TRUE(fs::exists(pano_path), "pano/pano_rgb.jpg exists");
    EXPECT_TRUE(fs::exists(mask_path), "pano/coverage_mask.png exists");
    EXPECT_TRUE(fs::exists(keyplan_path), "floors/floor_0_keyplan.png exists");

    // Verify genuine PNG signature (\x89PNG\r\n\x1a\n) + IHDR + IEND chunks on keyplan and coverage mask
    auto verifyGenuinePng = [](const fs::path& p, uint8_t expected_color_type) -> bool {
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs.is_open()) return false;
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)),
                                   std::istreambuf_iterator<char>());
        if (bytes.size() < 45) return false;
        const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        for (int i = 0; i < 8; ++i) {
            if (bytes[i] != sig[i]) return false;
        }
        // First chunk must be IHDR (length = 13, type = "IHDR")
        if (bytes[8] != 0 || bytes[9] != 0 || bytes[10] != 0 || bytes[11] != 13) return false;
        if (bytes[12] != 'I' || bytes[13] != 'H' || bytes[14] != 'D' || bytes[15] != 'R') return false;
        // Color type is byte 25 (16 + 4 width + 4 height + 1 bit_depth)
        if (bytes[25] != expected_color_type) return false;
        // Last 12 bytes must be IEND chunk (0 length + "IEND" + 4-byte CRC)
        const size_t n = bytes.size();
        if (bytes[n - 8] != 'I' || bytes[n - 7] != 'E' || bytes[n - 6] != 'N' || bytes[n - 5] != 'D') {
            return false;
        }
        return true;
    };
    EXPECT_TRUE(verifyGenuinePng(keyplan_path, 2 /* RGB24 */),
                "floors/floor_0_keyplan.png has valid PNG signature, 24-bit RGB IHDR, and IEND");
    EXPECT_TRUE(verifyGenuinePng(mask_path, 0 /* Gray8 */),
                "pano/coverage_mask.png has valid PNG signature, Gray8 IHDR, and IEND");

    // Verify station_manifest.json v1.0 schema keys
    {
        std::ifstream ifs(manifest_path);
        std::stringstream ss;
        ss << ifs.rdbuf();
        const std::string json_str = ss.str();

        EXPECT_TRUE(json_str.find("\"schema_version\": \"1.0\"") != std::string::npos,
                    "Manifest has schema_version 1.0");
        EXPECT_TRUE(json_str.find("\"station_id\": \"station_0001\"") != std::string::npos,
                    "Manifest has station_id");
        EXPECT_TRUE(json_str.find("\"hardware_profile\"") != std::string::npos,
                    "Manifest has hardware_profile");
        EXPECT_TRUE(json_str.find("\"coordinate_system\"") != std::string::npos,
                    "Manifest has coordinate_system");
        EXPECT_TRUE(json_str.find("\"rotation_spec\"") != std::string::npos,
                    "Manifest has rotation_spec");
        EXPECT_TRUE(json_str.find("\"coverage\"") != std::string::npos,
                    "Manifest has coverage");
        EXPECT_TRUE(json_str.find("\"registration_quality\"") != std::string::npos,
                    "Manifest has registration_quality");
        EXPECT_TRUE(json_str.find("\"refined_angle_deg\"") != std::string::npos,
                    "Manifest has refined_angle_deg per stop");
        EXPECT_TRUE(json_str.find("\"point_cloud_lod0\": \"cache/lod0.bin\"") != std::string::npos,
                    "Manifest has point_cloud_lod0: cache/lod0.bin");
        EXPECT_TRUE(json_str.find("\"floor_keyplan_png\": \"floors/floor_0_keyplan.png\"") != std::string::npos,
                    "Manifest has floor_keyplan_png: floors/floor_0_keyplan.png");
    }

    // Round-trip test for pointcloud.ply
    std::vector<LidarPointXYZIRT> ply_points;
    EXPECT_TRUE(TerrestrialStationManager::readFromPly(ply_path.string(), ply_points),
                "readFromPly succeeded");
    EXPECT_TRUE(ply_points.size() == meta.total_merged_points,
                "PLY round-trip point count matches merged station cloud");

    // Round-trip test for station.e57 (verifying ASTM E2807 header, CRC32C pages & XML trailer)
    std::vector<LidarPointXYZIRT> e57_points;
    EXPECT_TRUE(TerrestrialStationManager::readFromE57(e57_path.string(), e57_points),
                "readFromE57 succeeded with valid CRC32C page checksums");
    EXPECT_TRUE(e57_points.size() == meta.total_merged_points,
                "E57 round-trip point count matches merged station cloud");
    EXPECT_NEAR(e57_points.front().x, ply_points.front().x, 1e-5, "E57 first point X matches PLY");
    EXPECT_NEAR(e57_points.front().y, ply_points.front().y, 1e-5, "E57 first point Y matches PLY");
    EXPECT_NEAR(e57_points.front().z, ply_points.front().z, 1e-5, "E57 first point Z matches PLY");

    // Round-trip test for cache/lod0.bin (4-byte uint32 count + N * 15-byte packed records)
    std::vector<LidarPointXYZIRT> lod0_points;
    EXPECT_TRUE(TerrestrialStationManager::readLod0Bin(lod0_path.string(), lod0_points),
                "readLod0Bin succeeded");
    EXPECT_TRUE(!lod0_points.empty(), "LOD0 points non-empty");
    const uintmax_t expected_lod0_bytes = 4 + lod0_points.size() * 15;
    EXPECT_TRUE(fs::file_size(lod0_path) == expected_lod0_bytes,
                "cache/lod0.bin exact byte size is 4 + N * 15");

    fs::remove_all(temp_root);
    std::cout << "[PASS] testFullSixStopStationRefinementAndExports\n";
    ++g_tests_passed;
    return true;
}

// 5. Test Auto-Incrementing Directory-Scanning Station IDs & Repeat-Stop RGB Flag Reset
bool testAutoIncrementStationIdsAndRepeatRgbReset() {
    std::cout << "[RUN ] testAutoIncrementStationIdsAndRepeatRgbReset\n";

    const fs::path temp_root = fs::temp_directory_path() / "av_scan_test_station_ids";
    fs::remove_all(temp_root);

    // First standalone & flash IDs on empty project
    EXPECT_TRUE(TerrestrialStationManager::nextAvailableStationIdWithPrefix(temp_root.string(), "st_") == "st_001",
                "First standalone station ID is st_001");
    EXPECT_TRUE(TerrestrialStationManager::nextAvailableStationIdWithPrefix(temp_root.string(), "flash_st_") == "flash_st_001",
                "First flash station ID is flash_st_001");

    // Create st_001, st_002 and flash_st_001 on disk -> next allocations must never overwrite
    fs::create_directories(temp_root / "stations" / "st_001");
    fs::create_directories(temp_root / "stations" / "st_002");
    fs::create_directories(temp_root / "stations" / "flash_st_001");

    EXPECT_TRUE(TerrestrialStationManager::nextAvailableStationIdWithPrefix(temp_root.string(), "st_") == "st_003",
                "Next standalone station ID skips existing st_001/st_002 -> st_003");
    EXPECT_TRUE(TerrestrialStationManager::nextAvailableStationIdWithPrefix(temp_root.string(), "flash_st_") == "flash_st_002",
                "Next flash station ID skips existing flash_st_001 -> flash_st_002");

    // Verify repeatCurrentStop resets the per-stop RGB capture flag so a fresh image is captured
    TerrestrialStationConfig cfg;
    cfg.project_id = "proj_repeat";
    cfg.station_id = TerrestrialStationManager::nextAvailableStationIdWithPrefix(temp_root.string(), "st_");
    TerrestrialStationManager mgr;
    EXPECT_TRUE(mgr.startNewStation(temp_root.string(), cfg, DeviceMountProfile::makeAiryProfile()),
                "startNewStation succeeded");
    mgr.beginCurrentStopAccumulation();
    EXPECT_TRUE(!mgr.hasCapturedStopRgb(), "hasCapturedStopRgb starts false");
    std::vector<uint8_t> rgb(16 * 16 * 3, 128);
    mgr.pushCameraRgbBuffer(rgb, 16, 16);
    EXPECT_TRUE(mgr.hasCapturedStopRgb(), "hasCapturedStopRgb is true after capture");
    EXPECT_TRUE(mgr.repeatCurrentStop(), "repeatCurrentStop succeeded");
    EXPECT_TRUE(!mgr.hasCapturedStopRgb(), "repeatCurrentStop resets hasCapturedStopRgb to false");

    fs::remove_all(temp_root);
    std::cout << "[PASS] testAutoIncrementStationIdsAndRepeatRgbReset\n";
    ++g_tests_passed;
    return true;
}

// 6. Test Non-Blocking Rotation Controllers & 360-deg Cable Safety Guard
bool testRotationControllersAndCableSafety() {
    std::cout << "[RUN ] testRotationControllersAndCableSafety\n";

    // Manual / Detent controller is always available without blocking
    ManualRotationController manual_ctrl;
    EXPECT_TRUE(manual_ctrl.connect(), "ManualRotationController connects immediately");
    EXPECT_TRUE(manual_ctrl.setZeroPosition(), "ManualRotationController setZeroPosition");
    EXPECT_TRUE(manual_ctrl.moveToAngleDeg(60.0), "Manual move to 60 deg");
    double deg = -1.0;
    EXPECT_TRUE(!manual_ctrl.getCurrentAngleDeg(deg),
                "Manual controller reports false for hardware encoder while populating nominal angle");
    EXPECT_NEAR(deg, 60.0, 1e-6, "Current manual angle is 60 deg");

    // Test 360-deg cable safety guard on ManualRotationController
    EXPECT_TRUE(manual_ctrl.moveToAngleDeg(420.0), "Move to 420 deg wraps safely for cable guard");
    manual_ctrl.getCurrentAngleDeg(deg);
    EXPECT_TRUE(deg >= 0.0 && deg < 360.0, "Wrapped angle stays within [0, 360) deg");

    // AutoMotorRotationController when hardware is NOT online must fall back cleanly
    // without blocking terrestrial capture
    AutoMotorRotationController auto_ctrl("Syrp Genie Mini II", false /* hardware_online */);
    EXPECT_TRUE(auto_ctrl.connect(),
                "AutoMotorRotationController connects non-blockingly via manual fallback");
    EXPECT_TRUE(auto_ctrl.getRotationMode() == RotationMode::MANUAL_INDEX,
                "Offline motor controller reports MANUAL_INDEX fallback mode");
    EXPECT_TRUE(auto_ctrl.moveRelativeDeg(60.0), "Fallback moveRelativeDeg(60.0) succeeds");
    EXPECT_TRUE(!auto_ctrl.getCurrentAngleDeg(deg),
                "Offline motor controller reports no hardware encoder while populating fallback angle");
    EXPECT_NEAR(deg, 60.0, 1e-6, "Angle tracked at 60 deg in fallback mode");

    // AutoMotorRotationController when hardware IS online reports encoder feedback
    AutoMotorRotationController online_ctrl("Syrp Genie Mini II", true /* hardware_online */);
    EXPECT_TRUE(online_ctrl.connect(), "Online motor controller connects");
    EXPECT_TRUE(online_ctrl.getRotationMode() == RotationMode::AUTOMATIC_MOTOR,
                "Online motor controller reports AUTOMATIC_MOTOR mode");
    EXPECT_TRUE(online_ctrl.moveToAngleDeg(120.0), "Online motor moveToAngleDeg(120.0)");
    EXPECT_TRUE(online_ctrl.getCurrentAngleDeg(deg), "Online motor reports measured encoder angle");
    EXPECT_NEAR(deg, 120.0, 1e-6, "Measured motor angle is 120 deg");

    std::cout << "[PASS] testRotationControllersAndCableSafety\n";
    ++g_tests_passed;
    return true;
}

// 7. Test Single-Shot Motor-Free Flash Scan (Decoupled from 6x60 deg Terrestrial Mode)
bool testSingleShotMotorFreeFlashScan() {
    std::cout << "[RUN ] testSingleShotMotorFreeFlashScan\n";

    class SpyRotationController : public IRotationController {
    public:
        int set_zero_calls{0};
        int move_calls{0};
        int stop_calls{0};
        bool connect() override { return true; }
        void disconnect() override {}
        bool isConnected() const override { return true; }
        MotorCapabilities getCapabilities() const override { return MotorCapabilities{}; }
        RotationMode getRotationMode() const override { return RotationMode::MANUAL_INDEX; }
        bool setZeroPosition() override { ++set_zero_calls; return true; }
        bool moveToAngleDeg(double) override { ++move_calls; return true; }
        bool moveRelativeDeg(double) override { ++move_calls; return true; }
        bool getCurrentAngleDeg(double& deg) const override { deg = 0.0; return false; }
        MotionState getMotionState() const override { return MotionState::STATIONARY; }
        void setOperatorPromptCallback(PromptOperatorCallback) override {}
        void stopMotion() override { ++stop_calls; }
    };

    const fs::path temp_root = fs::temp_directory_path() / "av_scan_test_single_shot_flash";
    fs::remove_all(temp_root);

    auto spy = std::make_shared<SpyRotationController>();
    TerrestrialStationConfig cfg;
    cfg.project_id = "proj_flash";
    cfg.station_id = "flash_st_001";
    cfg.station_name = "Flash Scan 01";
    cfg.is_single_shot_flash = true;
    cfg.T_body_lidar = math::SE3d(math::SO3d(math::Quatd::FromEuler(0.0, 0.0, 90.0 * math::kDegToRad)),
                                  math::Vec3d(0.10, -0.05, 0.20));
    cfg.settling_delay_sec = 0.5;
    cfg.accumulation_sec_per_stop = 2.0;

    TerrestrialStationManager mgr(cfg, spy);
    EXPECT_TRUE(mgr.startNewStation(temp_root.string(), cfg, DeviceMountProfile::makeAiryProfile()),
                "startNewStation succeeded for single-shot Flash Scan");
    EXPECT_TRUE(spy->set_zero_calls == 0, "Single-shot Flash Scan must NEVER reset motor/turntable");

    // Feed stillness + 2.1s of stationary LiDAR frames
    double t = 10.0;
    for (int i = 0; i < 30; ++i) {
        ImuMeasurement imu;
        imu.timestamp_sec = t;
        imu.gyro_rads = math::Vec3d(0.001, -0.001, 0.0);
        imu.accel_mps2 = math::Vec3d(0.0, 0.0, 9.81);
        mgr.pushImuMeasurement(imu);

        LidarFrame frame;
        frame.timestamp_start_sec = t;
        frame.timestamp_end_sec = t + 0.1;
        LidarPointXYZIRT pt;
        pt.x = 2.0f; pt.y = 0.0f; pt.z = 0.5f;
        frame.points.push_back(pt);
        mgr.pushLidarFrame(frame);
        t += 0.1;
        if (mgr.getCaptureState() == TerrestrialCaptureState::STATION_COMPLETED) break;
    }

    EXPECT_TRUE(mgr.getCaptureState() == TerrestrialCaptureState::STATION_COMPLETED,
                "Single-shot Flash Scan reaches STATION_COMPLETED directly after 1 stop");
    EXPECT_TRUE(spy->move_calls == 0 && spy->set_zero_calls == 0,
                "Single-shot Flash Scan makes zero motor commands");

    std::string err;
    EXPECT_TRUE(mgr.finalizeAndSaveStation(err), "finalizeAndSaveStation succeeded for Flash Scan");
    EXPECT_TRUE(spy->set_zero_calls == 0, "finalizeAndSaveStation must NEVER unwind/command motor for Flash Scan");

    const auto meta = mgr.getStationMetadata();
    EXPECT_TRUE(meta.mode == "flash_scan", "Flash Scan metadata mode is flash_scan");
    EXPECT_TRUE(meta.expected_stops == 1 && meta.total_stops == 1, "Flash Scan has exactly 1 stop");

    const auto cloud = mgr.buildStationPointCloud();
    EXPECT_TRUE(!cloud.empty(), "Flash Scan cloud non-empty");
    // (2,0,0.5) rotated +90 deg around Z -> (0, 2, 0.5) + (0.10, -0.05, 0.20) = (0.10, 1.95, 0.70)
    EXPECT_NEAR(cloud.front().x, 0.10, 1e-4, "Flash point transformed directly by T_body_lidar X");
    EXPECT_NEAR(cloud.front().y, 1.95, 1e-4, "Flash point transformed directly by T_body_lidar Y");
    EXPECT_NEAR(cloud.front().z, 0.70, 1e-4, "Flash point transformed directly by T_body_lidar Z");

    {
        std::ifstream mf(temp_root / "stations" / "flash_st_001" / "station_manifest.json");
        std::stringstream ss;
        ss << mf.rdbuf();
        const std::string json_str = ss.str();
        EXPECT_TRUE(json_str.find("\"mode\": \"flash_scan\"") != std::string::npos,
                    "Manifest records mode=flash_scan");
        EXPECT_TRUE(json_str.find("\"mode\": \"none_stationary\"") != std::string::npos,
                    "Manifest records rotation_spec.mode=none_stationary");
    }

    fs::remove_all(temp_root);
    std::cout << "[PASS] testSingleShotMotorFreeFlashScan\n";
    ++g_tests_passed;
    return true;
}

} // namespace

int main() {
    std::cout << "=================================================================\n";
    std::cout << " AV Scan Terrestrial & Hybrid Flash Station Conformance Suite\n";
    std::cout << "=================================================================\n";

    testForwardKinematicChainAndCoordinateFrames();
    testStillnessGateBumpResetAndLidarFallback();
    testFullSixStopStationRefinementAndExports();
    testAutoIncrementStationIdsAndRepeatRgbReset();
    testRotationControllersAndCableSafety();
    testSingleShotMotorFreeFlashScan();

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << " Passed: " << g_tests_passed << " | Failed: " << g_tests_failed << "\n";
    std::cout << "=================================================================\n";

    return (g_tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

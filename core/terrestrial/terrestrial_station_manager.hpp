#pragma once

#include "core/terrestrial/irotation_controller.hpp"
#include "core/terrestrial/rotation_geometry.hpp"
#include "core/terrestrial/station_data_model.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace av::core::terrestrial {

enum class TerrestrialCaptureState : uint8_t {
    IDLE = 0,
    WAITING_FOR_STILLNESS = 1,
    ACCUMULATING_STOP = 2,
    STOP_COMPLETED = 3,
    STATION_COMPLETED = 4,
    STATION_ERROR = 5
};

inline std::string to_string(TerrestrialCaptureState s) {
    switch (s) {
        case TerrestrialCaptureState::IDLE: return "IDLE";
        case TerrestrialCaptureState::WAITING_FOR_STILLNESS: return "WAITING_FOR_STILLNESS";
        case TerrestrialCaptureState::ACCUMULATING_STOP: return "ACCUMULATING_STOP";
        case TerrestrialCaptureState::STOP_COMPLETED: return "STOP_COMPLETED";
        case TerrestrialCaptureState::STATION_COMPLETED: return "STATION_COMPLETED";
        case TerrestrialCaptureState::STATION_ERROR: return "STATION_ERROR";
        default: return "IDLE";
    }
}

class TerrestrialStationManager {
public:
    explicit TerrestrialStationManager(
        const TerrestrialStationConfig& config = TerrestrialStationConfig{},
        std::shared_ptr<IRotationController> rotation_ctrl = nullptr);

    bool startNewStation(
        const std::string& project_dir,
        const TerrestrialStationConfig& config,
        const DeviceMountProfile& mount_profile);

    void setTripodAxisCalibration(const TripodAxisCalibration& axis_calib);
    TripodAxisCalibration getTripodAxisCalibration() const;

    // Live sensor stream ingestion
    void pushLidarFrame(const LidarFrame& frame);
    void pushImuMeasurement(const ImuMeasurement& imu);
    void pushCameraImage(const std::string& image_file_path);
    void pushCameraRgbBuffer(const std::vector<uint8_t>& rgb_pixels, int width, int height);

    // Operator & state machine control triggers (Manual or Automated)
    bool beginCurrentStopAccumulation();
    bool finishCurrentStop();
    bool advanceToNextStop();
    bool repeatCurrentStop();
    bool cancelStation();
    bool finalizeAndSaveStation(std::string& out_error);

    // Constrained 1-DOF rotation registration refinement around the tripod +Z_station axis
    bool refineStopAnglesViaOverlap(double& out_mean_rmse_m);

    // 3D Azimuth & Elevation coverage analyzer across 36 azimuth x 18 elevation sectors (r >= 0.25m)
    void computeCoverageMetrics(double& out_az_cov, double& out_el_cov, std::string& out_warning) const;

    // Directory-scanning station ID allocator (st_001, st_002, ... / flash_st_001, flash_st_002, ...)
    static std::string nextAvailableStationIdWithPrefix(
        const std::string& project_dir,
        const std::string& prefix);

    // State queries
    TerrestrialCaptureState getCaptureState() const;
    uint32_t getCurrentStopIndex() const;
    double getCurrentNominalAngleDeg() const;
    double getCurrentAccumulationProgress() const; // 0.0 to 1.0
    double getImuMotionEnergy() const;
    bool isStationary() const;
    bool hasCapturedStopRgb() const;
    std::string getStationDirectory() const;

    const std::vector<StopObservation>& getCompletedStops() const;
    std::vector<LidarPointXYZIRT> buildStationPointCloud() const;
    TerrestrialStationMetadata getStationMetadata() const;

    // Static exporters & validators for station artifacts
    static bool exportToPly(const std::string& file_path,
                            const std::vector<LidarPointXYZIRT>& points,
                            bool binary_format = true);
    static bool readFromPly(const std::string& file_path,
                            std::vector<LidarPointXYZIRT>& out_points);

    static bool exportToE57(const std::string& file_path,
                            const std::vector<LidarPointXYZIRT>& points,
                            const std::string& scan_guid = "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                            const std::string& scan_name = "AVScan_Terrestrial_Station");
    static bool readFromE57(const std::string& file_path,
                            std::vector<LidarPointXYZIRT>& out_points);
    static uint32_t computeE57Crc32c(const uint8_t* data, size_t len);

    static size_t writeLod0Bin(const std::string& bin_path,
                               const std::vector<LidarPointXYZIRT>& points,
                               size_t max_points = 350000);
    static bool readLod0Bin(const std::string& bin_path,
                            std::vector<LidarPointXYZIRT>& out_points);

    static bool generatePanoramaAndCoverageMask(
        const std::vector<LidarPointXYZIRT>& merged_station_points,
        const std::string& pano_jpg_path,
        const std::string& mask_png_path,
        int width = 1024,
        int height = 512);

    static bool generateFloorKeyplanPng(
        const std::vector<LidarPointXYZIRT>& merged_station_points,
        const std::string& keyplan_png_path,
        int width = 1024,
        int height = 1024);

private:
    bool finishCurrentStopLocked();
    bool writeStationArtifactsLocked(std::string& out_error);
    math::SE3d computeStopTransformLocked(double angle_deg) const;

    TerrestrialStationConfig config_;
    DeviceMountProfile mount_profile_;
    TripodAxisCalibration axis_calib_;
    std::shared_ptr<IRotationController> rotation_ctrl_;

    std::string project_dir_;
    std::string station_dir_;

    mutable std::mutex mutex_;
    TerrestrialCaptureState state_{TerrestrialCaptureState::IDLE};
    uint32_t current_stop_idx_{0};
    double current_nominal_angle_deg_{0.0};
    double stop_start_time_sec_{0.0};
    double latest_frame_end_sec_{0.0};

    // Dual-criterion IMU stillness gate state
    double latest_gyro_norm_{0.0};
    double latest_accel_dev_{0.0};
    double stillness_duration_sec_{0.0};
    double last_imu_time_sec_{0.0};
    bool has_received_imu_{false};

    // Temporary accumulation buffer for active stop
    std::vector<LidarPointXYZIRT> current_stop_points_;
    std::string current_stop_image_path_;
    std::vector<uint8_t> current_stop_rgb_pixels_;
    int current_stop_rgb_width_{0};
    int current_stop_rgb_height_{0};
    bool stop_rgb_captured_{false};

    // Completed stops
    std::vector<StopObservation> stops_;
    double refined_overall_rmse_m_{0.0};
};

} // namespace av::core::terrestrial

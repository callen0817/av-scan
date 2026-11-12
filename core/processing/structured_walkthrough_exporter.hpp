#pragma once

#include "core/schemas/spatial_data.hpp"
#include "core/schemas/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace av::core::processing {

struct FlashStationAnchor {
    std::string anchor_id;
    std::string label;
    schemas::Vector3D position;
    double yaw_rad{0.0};
    bool z_is_sensor_height{false};
};

struct PosedRgbKeyframe {
    std::string frame_id;
    uint64_t timestamp_ns{0};
    std::string sensor_id;
    // Camera-to-World rigid transform: p_world = R_wc * p_cam + t_wc
    std::array<double, 16> T_world_cam{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    double fx{600.0};
    double fy{600.0};
    double cx{640.0};
    double cy{360.0};
    int width{0};
    int height{0};
    double sharpness_score{100.0};
    std::vector<uint8_t> rgb_pixels; // width * height * 3 (RGB8)
};

struct WalkthroughExportConfig {
    double station_spacing_m{2.5};
    double sensor_eye_height_m{1.60};
    double min_wall_clearance_m{0.45};
    double min_range_m{0.25};
    double max_range_m{35.0};
    int scan_rows{180};
    int scan_cols{720};
    int pano_width{2048};
    int pano_height{1024};
    int jpeg_quality{90};
    bool export_e57{true};
    bool export_standardized_clouds{true};
    bool enable_small_gap_fill{true};
    bool enforce_gravity_z_up{false};
    std::string project_id;
    std::string project_name{"AV Scan Project"};
    std::string scan_date;
    std::string scan_id;
    std::string av_showcase_project_dir;
    double capture_duration_seconds{0.0};
    size_t cslam_peers_count{0};
    const std::vector<schemas::PointXYZI>* airy_points{nullptr};
    const std::vector<schemas::PointXYZI>* stereo_points{nullptr};
    const std::vector<schemas::PointXYZI>* cslam_merged_points{nullptr};
};

struct E57ValidationSummary {
    bool valid{false};
    uint64_t physical_size_bytes{0};
    size_t total_pages{0};
    size_t crc_verified_pages{0};
    size_t station_count{0};
    size_t image_count{0};
    bool associated_guids_matched{false};
    std::string error_message;
};

struct WalkthroughBundleResult {
    bool success{false};
    std::string bundle_dir;
    std::string bundle_json_path;
    std::string stations_manifest_path;
    std::string scan_metadata_path;
    std::string structured_e57_path;
    std::string lod0_bin_path;
    std::string avproj_path;
    std::string pointcloud_airy_ply_path;
    std::string pointcloud_airy_pcd_path;
    std::string pointcloud_stereo_ply_path;
    std::string pointcloud_stereo_pcd_path;
    std::string pointcloud_fused_rgb_ply_path;
    std::string pointcloud_fused_rgb_pcd_path;
    std::string trajectory_csv_path;
    std::string trajectory_tum_path;
    size_t station_count{0};
    size_t floor_count{0};
    size_t lod0_points_written{0};
    uint64_t e57_size_bytes{0};
    bool is_monochrome_lidar_only{false};
    double registration_rmse_m{0.0};
    double registration_overlap{1.0};
    double export_time_ms{0.0};
    std::string error_message;
};

class StructuredWalkthroughExporter {
public:
    /**
     * Computes Castagnoli CRC-32C (reflected polynomial 0x82F63B78) per ASTM E2807-11.
     */
    static uint32_t computeCrc32c(const uint8_t* data, size_t length);

    /**
     * Decodes a JPEG file into an 8-bit RGB buffer (width * height * 3).
     */
    static bool decodeJpegFileRgb(
        const std::string& jpeg_path,
        std::vector<uint8_t>& out_rgb,
        int& out_width,
        int& out_height);

    /**
     * Computes true Black-and-White (R = G = B) grayscale colors for uncolored / LiDAR-only
     * points using LiDAR reflectivity/intensity and elevation shading.
     * Returns true if the cloud is monochrome (no true RGB colors present).
     */
    static bool populateMonochromeGrayscaleForUncoloredPoints(
        std::vector<schemas::PointXYZI>& points,
        std::vector<bool>* out_point_has_true_rgb = nullptr);

    /**
     * Validates an ASTM E2807-11 Structured E57 file:
     *   - Verifies every 1024-byte physical page's big-endian Castagnoli CRC-32C trailer
     *   - Parses the XML footer and checks data3D <-> images2D associatedData3DGuid linkage
     */
    static E57ValidationSummary validateStructuredE57(const std::string& e57_path);

    /**
     * Writes AV Showcase binary point cloud cache (`cache/lod0.bin`):
     *   uint32_t numPoints + N * packed 15-byte <float x, float y, float z, uint8 r, g, b>.
     */
    static size_t writeLod0Bin(
        const std::string& bin_path,
        const std::vector<schemas::PointXYZI>& points,
        size_t max_points = 350000);

    static bool readLod0Bin(
        const std::string& bin_path,
        std::vector<schemas::PointXYZI>& out_points);

    /**
     * Gravity-preserving 4-DOF ([tx, ty, tz, yaw]) coarse-to-fine ICP registration
     * aligning `source_points` into `reference_points` (Project Baseline Scan #0).
     */
    static std::array<double, 16> computeTransformToProjectIcp(
        const std::vector<schemas::PointXYZI>& source_points,
        const std::vector<schemas::PointXYZI>& reference_points,
        double& out_rmse_m,
        double& out_overlap_ratio);

    /**
     * Full end-to-end export of:
     *   - Virtual Scan Stations (Flash anchors + trajectory + navigable floor fallback)
     *   - ASTM E2807-11 360° Equirectangular Panoramas (hybrid keyframe reprojection + LiDAR splatting,
     *     strictly non-fabricated pure #000000 black for unobserved regions)
     *   - ASTM E2807-11 Structured E57 (`structured_stations.e57`)
     *   - Multi-story floor detection + wall-occlusion-checked 8-sector & staircase nav graph
     *   - Metric-calibrated 2D architectural wall-slice Key Plan PNGs (`floors/floor_<id>_keyplan.png`)
     *   - Binary point cloud cache (`cache/lod0.bin`) + `av_walkthrough_bundle.json` (v1.1)
     *   - Optional direct creation / 4D historical append to `<av_showcase_project_dir>/project.avproj`
     */
    static WalkthroughBundleResult exportWalkthroughBundle(
        const std::string& output_dir,
        const std::vector<schemas::PointXYZI>& points,
        const std::vector<schemas::Pose3D>& trajectory = {},
        const std::vector<FlashStationAnchor>& flash_anchors = {},
        const std::vector<PosedRgbKeyframe>& keyframes = {},
        const WalkthroughExportConfig& config = WalkthroughExportConfig(),
        const std::vector<schemas::PointXYZI>* reference_baseline_points = nullptr);

    /**
     * Appends or updates a generated `av_walkthrough_bundle.json` directory directly inside an
     * AV Showcase project folder (`<av_showcase_project_dir>/project.avproj` + `Scans/<scan_key>/`),
     * automatically running 4-DOF ICP against Scan #0's `lod0.bin` when `auto_align_to_baseline` is true.
     */
    static std::string exportOrAppendAvShowcaseProject(
        const std::string& av_showcase_project_dir,
        const std::string& bundle_dir,
        bool auto_align_to_baseline = true,
        double* out_rmse_m = nullptr,
        double* out_overlap_ratio = nullptr);
};

} // namespace av::core::processing

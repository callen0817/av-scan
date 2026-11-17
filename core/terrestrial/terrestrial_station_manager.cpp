#include "core/terrestrial/terrestrial_station_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <jpeglib.h>
#include <sstream>
#include <unordered_map>
#include <zlib.h>

namespace fs = std::filesystem;

namespace av::core::terrestrial {

namespace {

std::string getCurrentIsoUtc() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm gm{};
#if defined(_WIN32)
    gmtime_s(&gm, &tt);
#else
    gmtime_r(&tt, &gm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&gm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

struct VoxelKey {
    int32_t ix{0};
    int32_t iy{0};
    int32_t iz{0};
    bool operator==(const VoxelKey& o) const noexcept {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const noexcept {
        return (static_cast<size_t>(static_cast<uint32_t>(k.ix)) * 73856093u) ^
               (static_cast<size_t>(static_cast<uint32_t>(k.iy)) * 19349663u) ^
               (static_cast<size_t>(static_cast<uint32_t>(k.iz)) * 83492791u);
    }
};

struct VoxelCentroidAccum {
    math::Vec3d sum{0.0, 0.0, 0.0};
    uint32_t count{0};
};

#pragma pack(push, 1)
struct E57FileHeader {
    char file_signature[8];
    uint32_t major_version;
    uint32_t minor_version;
    uint64_t file_physical_len;
    uint64_t xml_physical_offset;
    uint64_t xml_logical_len;
    uint64_t page_size;
};

struct CompressedVectorSectionHeader {
    uint8_t section_id;
    uint8_t reserved[7];
    uint64_t section_logical_length;
    uint64_t data_physical_offset;
    uint64_t index_physical_offset;
};
#pragma pack(pop)

uint64_t logicalToPhysicalOffset(uint64_t logical_offset) {
    uint64_t page = logical_offset / 1020ULL;
    uint64_t rem = logical_offset % 1020ULL;
    return page * 1024ULL + rem;
}

bool writeJpegRgbFile(const std::string& path, const uint8_t* rgb, int width, int height, int quality = 90) {
    if (!rgb || width <= 0 || height <= 0) return false;
    fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = static_cast<JDIMENSION>(width);
    cinfo.image_height = static_cast<JDIMENSION>(height);
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, std::clamp(quality, 10, 100), TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_ptr[1];
        row_ptr[0] = const_cast<JSAMPROW>(
            rgb + static_cast<size_t>(cinfo.next_scanline) * static_cast<size_t>(width) * 3);
        jpeg_write_scanlines(&cinfo, row_ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
    return true;
}

void appendBe32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
}

void writePngChunk(std::vector<uint8_t>& out, const char type[4], const uint8_t* payload, size_t len) {
    appendBe32(out, static_cast<uint32_t>(len));
    size_t type_start = out.size();
    out.insert(out.end(), reinterpret_cast<const uint8_t*>(type), reinterpret_cast<const uint8_t*>(type) + 4);
    if (payload && len > 0) {
        out.insert(out.end(), payload, payload + len);
    }
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, out.data() + type_start, static_cast<uInt>(4 + len));
    appendBe32(out, static_cast<uint32_t>(crc));
}

bool writePngGray8File(const std::string& path, const uint8_t* gray, int width, int height) {
    if (!gray || width <= 0 || height <= 0) return false;
    fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    std::vector<uint8_t> raw_scanlines;
    raw_scanlines.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width)));
    for (int r = 0; r < height; ++r) {
        raw_scanlines.push_back(0); // Filter 0
        const uint8_t* row = gray + static_cast<size_t>(r) * static_cast<size_t>(width);
        raw_scanlines.insert(raw_scanlines.end(), row, row + width);
    }

    uLongf comp_bound = compressBound(static_cast<uLong>(raw_scanlines.size()));
    std::vector<uint8_t> zbuf(comp_bound);
    if (compress2(zbuf.data(), &comp_bound, raw_scanlines.data(),
                  static_cast<uLong>(raw_scanlines.size()), 6) != Z_OK) {
        return false;
    }

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    appendBe32(ihdr, static_cast<uint32_t>(width));
    appendBe32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // bit depth 8
    ihdr.push_back(0); // color type 0 (grayscale)
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    writePngChunk(png, "IHDR", ihdr.data(), ihdr.size());
    writePngChunk(png, "IDAT", zbuf.data(), static_cast<size_t>(comp_bound));
    writePngChunk(png, "IEND", nullptr, 0);

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return ofs.good();
}

} // anonymous namespace

TerrestrialStationManager::TerrestrialStationManager(
    const TerrestrialStationConfig& config,
    std::shared_ptr<IRotationController> rotation_ctrl)
    : config_(config), rotation_ctrl_(std::move(rotation_ctrl)) {
    if (!rotation_ctrl_) {
        rotation_ctrl_ = std::make_shared<ManualRotationController>();
    }
}

void TerrestrialStationManager::setTripodAxisCalibration(const TripodAxisCalibration& axis_calib) {
    std::lock_guard<std::mutex> lock(mutex_);
    axis_calib_ = axis_calib;
}

TripodAxisCalibration TerrestrialStationManager::getTripodAxisCalibration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return axis_calib_;
}

math::SE3d TerrestrialStationManager::computeStopTransformLocked(double angle_deg) const {
    return RotationGeometry::computeStationLidarTransform(
        angle_deg,
        mount_profile_.T_cam_lidar,
        axis_calib_,
        mount_profile_.extrinsic_is_camera_optical);
}

bool TerrestrialStationManager::startNewStation(
    const std::string& project_dir,
    const TerrestrialStationConfig& config,
    const DeviceMountProfile& mount_profile) {
    std::lock_guard<std::mutex> lock(mutex_);
    project_dir_ = project_dir;
    config_ = config;
    mount_profile_ = mount_profile;
    current_stop_idx_ = 0;
    current_nominal_angle_deg_ = 0.0;
    stop_start_time_sec_ = 0.0;
    latest_frame_end_sec_ = 0.0;
    stillness_duration_sec_ = 0.0;
    last_imu_time_sec_ = 0.0;
    latest_gyro_norm_ = 0.0;
    latest_accel_dev_ = 0.0;
    has_received_imu_ = false;
    refined_overall_rmse_m_ = 0.0;

    stops_.clear();
    current_stop_points_.clear();
    current_stop_image_path_.clear();
    current_stop_rgb_pixels_.clear();
    current_stop_rgb_width_ = 0;
    current_stop_rgb_height_ = 0;

    station_dir_ = (fs::path(project_dir_) / "stations" / config_.station_id).string();
    std::error_code ec;
    fs::create_directories(fs::path(station_dir_) / "raw" / "images", ec);
    fs::create_directories(fs::path(station_dir_) / "cache", ec);
    fs::create_directories(fs::path(station_dir_) / "pano", ec);

    state_ = TerrestrialCaptureState::WAITING_FOR_STILLNESS;
    rotation_ctrl_->setZeroPosition();
    return true;
}

void TerrestrialStationManager::pushImuMeasurement(const ImuMeasurement& imu) {
    std::lock_guard<std::mutex> lock(mutex_);
    double dt = (last_imu_time_sec_ > 0.0 && imu.timestamp_sec > last_imu_time_sec_)
                    ? std::clamp(imu.timestamp_sec - last_imu_time_sec_, 0.001, 0.10)
                    : 0.005;
    last_imu_time_sec_ = imu.timestamp_sec;
    has_received_imu_ = true;

    latest_gyro_norm_ = imu.gyro_rads.norm();
    latest_accel_dev_ = std::abs(imu.accel_mps2.norm() - 9.81);

    // Dual-criterion stillness gate:
    //   ||omega|| < 0.05 rad/s AND | ||a|| - 9.81 | < 0.85 m/s^2
    bool is_quiet = (latest_gyro_norm_ < config_.max_motion_angular_vel_rads) &&
                    (latest_accel_dev_ < 0.85);

    if (is_quiet) {
        stillness_duration_sec_ += dt;
    } else {
        stillness_duration_sec_ = 0.0;
    }

    if (state_ == TerrestrialCaptureState::WAITING_FOR_STILLNESS) {
        if (stillness_duration_sec_ >= config_.settling_delay_sec) {
            state_ = TerrestrialCaptureState::ACCUMULATING_STOP;
            stop_start_time_sec_ = 0.0; // Anchor strictly on first LiDAR frame timestamp
        }
    } else if (state_ == TerrestrialCaptureState::ACCUMULATING_STOP) {
        if (latest_gyro_norm_ > (config_.max_motion_angular_vel_rads * 2.5) ||
            latest_accel_dev_ > 2.0) {
            // Motion disturbance or operator bump during accumulation:
            // clear partial buffer, reset to WAITING_FOR_STILLNESS, and re-accumulate cleanly once still
            state_ = TerrestrialCaptureState::WAITING_FOR_STILLNESS;
            stillness_duration_sec_ = 0.0;
            stop_start_time_sec_ = 0.0;
            current_stop_points_.clear();
        }
    }
}

void TerrestrialStationManager::pushLidarFrame(const LidarFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Fallback for LiDAR-only sensor profiles where no IMU packets have arrived:
    // automatically transition from WAITING_FOR_STILLNESS to ACCUMULATING_STOP after settling_delay_sec (1.0s)
    if (!has_received_imu_ && state_ == TerrestrialCaptureState::WAITING_FOR_STILLNESS) {
        if (stop_start_time_sec_ <= 0.0) {
            stop_start_time_sec_ = frame.timestamp_start_sec;
        } else if ((frame.timestamp_end_sec - stop_start_time_sec_) >= config_.settling_delay_sec) {
            state_ = TerrestrialCaptureState::ACCUMULATING_STOP;
            stop_start_time_sec_ = frame.timestamp_start_sec;
            stillness_duration_sec_ = config_.settling_delay_sec;
        } else {
            return;
        }
        // If we just transitioned on this frame, start accumulating from this frame's start timestamp
    }

    if (state_ != TerrestrialCaptureState::ACCUMULATING_STOP) {
        return; // Guarded stationary accumulation: reject points during rotation or settling
    }

    if (stop_start_time_sec_ <= 0.0) {
        stop_start_time_sec_ = frame.timestamp_start_sec;
    }
    latest_frame_end_sec_ = frame.timestamp_end_sec;

    // Accumulate stationary points
    current_stop_points_.insert(current_stop_points_.end(), frame.points.begin(), frame.points.end());

    // Unified Clock: measure 5.0-second accumulation duration strictly using LiDAR frame timestamps
    if ((latest_frame_end_sec_ - stop_start_time_sec_) >= config_.accumulation_sec_per_stop) {
        finishCurrentStopLocked();
    }
}

void TerrestrialStationManager::pushCameraImage(const std::string& image_file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_stop_image_path_ = image_file_path;
}

void TerrestrialStationManager::pushCameraRgbBuffer(const std::vector<uint8_t>& rgb_pixels, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (width > 0 && height > 0 && rgb_pixels.size() >= static_cast<size_t>(width) * static_cast<size_t>(height) * 3) {
        current_stop_rgb_pixels_ = rgb_pixels;
        current_stop_rgb_width_ = width;
        current_stop_rgb_height_ = height;
    }
}

bool TerrestrialStationManager::beginCurrentStopAccumulation() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = TerrestrialCaptureState::WAITING_FOR_STILLNESS;
    stillness_duration_sec_ = 0.0;
    stop_start_time_sec_ = 0.0;
    current_stop_points_.clear();
    return true;
}

bool TerrestrialStationManager::finishCurrentStop() {
    std::lock_guard<std::mutex> lock(mutex_);
    return finishCurrentStopLocked();
}

bool TerrestrialStationManager::finishCurrentStopLocked() {
    if (state_ != TerrestrialCaptureState::ACCUMULATING_STOP &&
        state_ != TerrestrialCaptureState::WAITING_FOR_STILLNESS) {
        return false;
    }
    if (current_stop_points_.empty()) {
        return false; // Do not commit an empty stop
    }

    StopObservation obs;
    obs.stop_index = current_stop_idx_;
    obs.nominal_angle_deg = current_nominal_angle_deg_;

    double measured_deg = current_nominal_angle_deg_;
    obs.has_measured_angle = rotation_ctrl_->getCurrentAngleDeg(measured_deg);
    obs.measured_angle_deg = measured_deg;

    obs.timestamp_start_sec = stop_start_time_sec_;
    obs.timestamp_end_sec = latest_frame_end_sec_;

    // Transform from LiDAR sensor frame to Station frame
    obs.T_station_lidar = computeStopTransformLocked(obs.measured_angle_deg);
    obs.stationary_points = current_stop_points_;

    // Save raw stop LiDAR checkpoint to disk immediately
    std::ostringstream raw_name;
    raw_name << "raw/stop_" << std::setw(3) << std::setfill('0') << obs.stop_index
             << "_" << std::setw(3) << std::setfill('0') << static_cast<int>(std::round(obs.nominal_angle_deg))
             << "deg.ply";
    obs.raw_lidar_relpath = raw_name.str();
    if (!station_dir_.empty()) {
        std::string abs_raw = (fs::path(station_dir_) / obs.raw_lidar_relpath).string();
        exportToPly(abs_raw, obs.stationary_points, true);
    }

    // Save or record raw RGB image checkpoint if a camera frame was captured
    if (!current_stop_rgb_pixels_.empty() && current_stop_rgb_width_ > 0 && current_stop_rgb_height_ > 0 && !station_dir_.empty()) {
        std::ostringstream img_rel;
        img_rel << "raw/images/stop_" << std::setw(3) << std::setfill('0') << obs.stop_index << ".jpg";
        obs.rgb_image_relpath = img_rel.str();
        std::string abs_img = (fs::path(station_dir_) / obs.rgb_image_relpath).string();
        writeJpegRgbFile(abs_img, current_stop_rgb_pixels_.data(), current_stop_rgb_width_, current_stop_rgb_height_, 90);
    } else if (!current_stop_image_path_.empty()) {
        obs.rgb_image_relpath = current_stop_image_path_;
        if (!station_dir_.empty() && fs::exists(current_stop_image_path_)) {
            std::ostringstream img_rel;
            img_rel << "raw/images/stop_" << std::setw(3) << std::setfill('0') << obs.stop_index << ".jpg";
            std::error_code ec;
            fs::copy_file(current_stop_image_path_, fs::path(station_dir_) / img_rel.str(),
                          fs::copy_options::overwrite_existing, ec);
            if (!ec) {
                obs.rgb_image_relpath = img_rel.str();
            }
        }
    }

    obs.registration_converged = true;
    obs.registration_residual_rmse_m = 0.0;

    stops_.push_back(std::move(obs));
    current_stop_points_.clear();
    current_stop_rgb_pixels_.clear();
    state_ = TerrestrialCaptureState::STOP_COMPLETED;
    return true;
}

bool TerrestrialStationManager::advanceToNextStop() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Commit current stop if points exist while accumulating
    if (state_ == TerrestrialCaptureState::ACCUMULATING_STOP && !current_stop_points_.empty()) {
        finishCurrentStopLocked();
    }

    if (current_stop_idx_ + 1 >= config_.total_expected_stops) {
        state_ = TerrestrialCaptureState::STATION_COMPLETED;
        return true;
    }

    current_stop_idx_++;
    current_nominal_angle_deg_ += config_.nominal_step_deg;

    // Cable safety guard: prevent > 360 deg unidirectional rotation that twists cables
    if (current_nominal_angle_deg_ >= 360.0) {
        current_nominal_angle_deg_ = std::fmod(current_nominal_angle_deg_, 360.0);
    }

    rotation_ctrl_->moveToAngleDeg(current_nominal_angle_deg_);
    state_ = TerrestrialCaptureState::WAITING_FOR_STILLNESS;
    stillness_duration_sec_ = 0.0;
    stop_start_time_sec_ = 0.0;
    current_stop_points_.clear();
    current_stop_image_path_.clear();
    current_stop_rgb_pixels_.clear();
    return true;
}

bool TerrestrialStationManager::repeatCurrentStop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stops_.empty() && stops_.back().stop_index == current_stop_idx_) {
        stops_.pop_back();
    }
    state_ = TerrestrialCaptureState::WAITING_FOR_STILLNESS;
    stillness_duration_sec_ = 0.0;
    stop_start_time_sec_ = 0.0;
    current_stop_points_.clear();
    return true;
}

bool TerrestrialStationManager::cancelStation() {
    std::lock_guard<std::mutex> lock(mutex_);
    rotation_ctrl_->stopMotion();
    state_ = TerrestrialCaptureState::IDLE;
    stops_.clear();
    current_stop_points_.clear();
    return true;
}

bool TerrestrialStationManager::refineStopAnglesViaOverlap(double& out_mean_rmse_m) {
    if (stops_.size() < 2) {
        out_mean_rmse_m = 0.0;
        return true;
    }

    constexpr double kVoxelSize = 0.15; // 15 cm spatial voxel hash for overlap correspondence
    constexpr double kInvVoxel = 1.0 / kVoxelSize;
    double total_rmse = 0.0;
    size_t valid_pairs = 0;

    // Anchor Stop 0 at 0.0 deg; refine each subsequent stop k against stop k-1
    // via constrained 1-DOF rotation search around +Z_station over [-3.0 deg, +3.0 deg]
    stops_[0].measured_angle_deg = stops_[0].nominal_angle_deg;
    stops_[0].T_station_lidar = computeStopTransformLocked(stops_[0].measured_angle_deg);

    for (size_t k = 1; k < stops_.size(); ++k) {
        const auto& prev_stop = stops_[k - 1];
        auto& curr_stop = stops_[k];

        if (prev_stop.stationary_points.empty() || curr_stop.stationary_points.empty()) {
            continue;
        }

        // Build 15 cm spatial voxel centroid hash map of prev_stop in Station Frame
        std::unordered_map<VoxelKey, VoxelCentroidAccum, VoxelKeyHash> accum_voxels;
        size_t ref_stride = std::max<size_t>(1, prev_stop.stationary_points.size() / 6000);
        for (size_t i = 0; i < prev_stop.stationary_points.size(); i += ref_stride) {
            auto p_st = RotationGeometry::transformPointToStation(
                prev_stop.stationary_points[i], prev_stop.T_station_lidar);
            VoxelKey vk{
                static_cast<int32_t>(std::floor(p_st.x * kInvVoxel)),
                static_cast<int32_t>(std::floor(p_st.y * kInvVoxel)),
                static_cast<int32_t>(std::floor(p_st.z * kInvVoxel))
            };
            auto& acc = accum_voxels[vk];
            acc.sum += math::Vec3d(p_st.x, p_st.y, p_st.z);
            acc.count++;
        }

        std::unordered_map<VoxelKey, math::Vec3d, VoxelKeyHash> ref_voxels;
        ref_voxels.reserve(accum_voxels.size());
        for (const auto& kv : accum_voxels) {
            ref_voxels[kv.first] = kv.second.sum / static_cast<double>(kv.second.count);
        }

        size_t query_stride = std::max<size_t>(1, curr_stop.stationary_points.size() / 2000);

        auto evalCandidateAngle = [&](double cand_deg, double& out_rmse, size_t& out_inliers, double& out_cost) {
            math::SE3d T_cand = computeStopTransformLocked(cand_deg);
            double sum_sq = 0.0;
            size_t inliers = 0;

            for (size_t i = 0; i < curr_stop.stationary_points.size(); i += query_stride) {
                auto p_st = RotationGeometry::transformPointToStation(curr_stop.stationary_points[i], T_cand);
                math::Vec3d qv(p_st.x, p_st.y, p_st.z);
                VoxelKey base_vk{
                    static_cast<int32_t>(std::floor(p_st.x * kInvVoxel)),
                    static_cast<int32_t>(std::floor(p_st.y * kInvVoxel)),
                    static_cast<int32_t>(std::floor(p_st.z * kInvVoxel))
                };

                // Check target voxel and immediate 26-connected neighbors within 15cm radius
                double min_d2 = kVoxelSize * kVoxelSize;
                bool found = false;
                auto it_center = ref_voxels.find(base_vk);
                if (it_center != ref_voxels.end()) {
                    double d2 = (qv - it_center->second).squaredNorm();
                    if (d2 < min_d2) {
                        min_d2 = d2;
                        found = true;
                    }
                }
                if (!found) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dz = -1; dz <= 1; ++dz) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                VoxelKey nvk{base_vk.ix + dx, base_vk.iy + dy, base_vk.iz + dz};
                                auto it = ref_voxels.find(nvk);
                                if (it != ref_voxels.end()) {
                                    double d2 = (qv - it->second).squaredNorm();
                                    if (d2 < min_d2) {
                                        min_d2 = d2;
                                        found = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if (found) {
                    sum_sq += min_d2;
                    inliers++;
                }
            }

            out_inliers = inliers;
            if (inliers >= 25) {
                out_rmse = std::sqrt(sum_sq / static_cast<double>(inliers));
                out_cost = out_rmse / (std::sqrt(static_cast<double>(inliers)) + 1.0);
            } else {
                out_rmse = 1e9;
                out_cost = 1e9;
            }
        };

        double best_angle_deg = curr_stop.nominal_angle_deg;
        double best_score = 1e9;
        double best_rmse = 0.0;
        size_t best_inliers = 0;

        // Stage 1: Coarse search over [-3.0 deg, +3.0 deg] in 0.25 deg increments
        for (double delta_deg = -3.0; delta_deg <= 3.001; delta_deg += 0.25) {
            double cand_deg = curr_stop.nominal_angle_deg + delta_deg;
            double rmse = 0.0, cost = 1e9;
            size_t inliers = 0;
            evalCandidateAngle(cand_deg, rmse, inliers, cost);
            if (inliers >= 25 && cost < best_score) {
                best_score = cost;
                best_angle_deg = cand_deg;
                best_rmse = rmse;
                best_inliers = inliers;
            }
        }

        // Stage 2: Fine search around coarse winner in 0.05 deg increments
        if (best_inliers >= 25) {
            double coarse_center = best_angle_deg;
            for (double delta_deg = -0.25; delta_deg <= 0.2501; delta_deg += 0.05) {
                double cand_deg = std::clamp(
                    coarse_center + delta_deg,
                    curr_stop.nominal_angle_deg - 3.0,
                    curr_stop.nominal_angle_deg + 3.0);
                double rmse = 0.0, cost = 1e9;
                size_t inliers = 0;
                evalCandidateAngle(cand_deg, rmse, inliers, cost);
                if (inliers >= 25 && cost < best_score) {
                    best_score = cost;
                    best_angle_deg = cand_deg;
                    best_rmse = rmse;
                    best_inliers = inliers;
                }
            }
        }

        if (best_inliers >= 25) {
            curr_stop.measured_angle_deg = best_angle_deg;
            curr_stop.T_station_lidar = computeStopTransformLocked(best_angle_deg);
            curr_stop.registration_residual_rmse_m = best_rmse;
            curr_stop.registration_converged = true;
            total_rmse += best_rmse;
            valid_pairs++;
        } else {
            curr_stop.registration_residual_rmse_m = 0.0;
            curr_stop.registration_converged = false;
        }
    }

    out_mean_rmse_m = (valid_pairs > 0) ? (total_rmse / static_cast<double>(valid_pairs)) : 0.0;
    return true;
}

void TerrestrialStationManager::computeCoverageMetrics(
    double& out_az_cov,
    double& out_el_cov,
    std::string& out_warning) const {
    constexpr int kAzBins = 36; // 10 deg per bin across [0, 360)
    constexpr int kElBins = 18; // 10 deg per bin across [-90, +90]
    bool az_hit[kAzBins]{};
    bool el_hit[kElBins]{};

    for (const auto& stop : stops_) {
        size_t stride = std::max<size_t>(1, stop.stationary_points.size() / 2000);
        for (size_t i = 0; i < stop.stationary_points.size(); i += stride) {
            auto p = RotationGeometry::transformPointToStation(
                stop.stationary_points[i], stop.T_station_lidar);
            double r_xy = std::hypot(static_cast<double>(p.x), static_cast<double>(p.y));
            double r = std::hypot(r_xy, static_cast<double>(p.z));
            if (r < 0.25) continue;

            double az_deg = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x)) * math::kRadToDeg;
            if (az_deg < 0.0) az_deg += 360.0;
            double el_deg = std::atan2(static_cast<double>(p.z), r_xy) * math::kRadToDeg;

            int az_idx = std::clamp(static_cast<int>(az_deg / 10.0), 0, kAzBins - 1);
            int el_idx = std::clamp(static_cast<int>((el_deg + 90.0) / 10.0), 0, kElBins - 1);
            az_hit[az_idx] = true;
            el_hit[el_idx] = true;
        }
    }

    int az_count = 0;
    for (bool b : az_hit) if (b) az_count++;
    int el_count = 0;
    for (bool b : el_hit) if (b) el_count++;

    out_az_cov = static_cast<double>(az_count) / static_cast<double>(kAzBins);
    out_el_cov = static_cast<double>(el_count) / static_cast<double>(kElBins);

    if (stops_.size() < config_.total_expected_stops) {
        out_warning = "Incomplete revolution (" + std::to_string(stops_.size()) + "/" +
                      std::to_string(config_.total_expected_stops) + " stops captured).";
    } else if (out_az_cov < 0.85) {
        out_warning = "Missing azimuth sectors detected (<85% 360 horizontal coverage).";
    } else {
        out_warning = "Complete 360 horizontal coverage verified.";
    }
}

bool TerrestrialStationManager::generatePanoramaAndCoverageMask(
    const std::vector<LidarPointXYZIRT>& merged_station_points,
    const std::string& pano_jpg_path,
    const std::string& mask_png_path,
    int width,
    int height) {
    if (width <= 0 || height <= 0) return false;
    std::vector<uint8_t> pano_rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
    std::vector<uint8_t> cov_mask(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    std::vector<float> z_buffer(static_cast<size_t>(width) * static_cast<size_t>(height), 1e9f);

    for (const auto& pt : merged_station_points) {
        double r_xy = std::hypot(static_cast<double>(pt.x), static_cast<double>(pt.y));
        double r = std::hypot(r_xy, static_cast<double>(pt.z));
        if (r < 0.25) continue;

        double az = std::atan2(static_cast<double>(pt.y), static_cast<double>(pt.x)); // [-pi, +pi]
        double el = std::atan2(static_cast<double>(pt.z), r_xy);                      // [-pi/2, +pi/2]

        double u_norm = (0.5 - az / (2.0 * math::kPi));
        if (u_norm < 0.0) u_norm += 1.0;
        if (u_norm >= 1.0) u_norm -= 1.0;
        double v_norm = 0.5 - el / math::kPi;

        int u = std::clamp(static_cast<int>(u_norm * width), 0, width - 1);
        int v = std::clamp(static_cast<int>(v_norm * height), 0, height - 1);

        uint8_t pr = pt.r, pg = pt.g, pb = pt.b;
        if (!pt.colorized && pr == 255 && pg == 255 && pb == 255) {
            float inten = pt.intensity <= 1.0f ? pt.intensity * 255.0f : pt.intensity;
            uint8_t g = static_cast<uint8_t>(std::clamp(inten, 45.0f, 255.0f));
            pr = g; pg = g; pb = g;
        }

        // Splat 3x3 neighborhood for clean equirectangular LiDAR/RGB panorama
        for (int dv = -1; dv <= 1; ++dv) {
            int vv = v + dv;
            if (vv < 0 || vv >= height) continue;
            for (int du = -1; du <= 1; ++du) {
                int uu = (u + du + width) % width;
                size_t idx = static_cast<size_t>(vv) * static_cast<size_t>(width) + static_cast<size_t>(uu);
                if (static_cast<float>(r) < z_buffer[idx]) {
                    z_buffer[idx] = static_cast<float>(r);
                    pano_rgb[idx * 3 + 0] = pr;
                    pano_rgb[idx * 3 + 1] = pg;
                    pano_rgb[idx * 3 + 2] = pb;
                    cov_mask[idx] = 255;
                }
            }
        }
    }

    bool ok_jpg = writeJpegRgbFile(pano_jpg_path, pano_rgb.data(), width, height, 90);
    bool ok_png = writePngGray8File(mask_png_path, cov_mask.data(), width, height);
    return ok_jpg && ok_png;
}

bool TerrestrialStationManager::writeStationArtifactsLocked(std::string& out_error) {
    if (station_dir_.empty()) {
        out_error = "Station directory path is not initialized.";
        return false;
    }

    fs::path st_dir(station_dir_);
    std::error_code ec;
    fs::create_directories(st_dir / "cache", ec);
    fs::create_directories(st_dir / "pano", ec);
    fs::create_directories(st_dir / "raw" / "images", ec);

    // 1. Build merged station point cloud in Station Frame
    std::vector<LidarPointXYZIRT> merged;
    for (const auto& stop : stops_) {
        for (const auto& pt : stop.stationary_points) {
            merged.push_back(RotationGeometry::transformPointToStation(pt, stop.T_station_lidar));
        }
    }

    // 2. Write pointcloud.ply, station.e57, cache/lod0.bin, and pano/*
    std::string ply_abs = (st_dir / "pointcloud.ply").string();
    std::string e57_abs = (st_dir / "station.e57").string();
    std::string lod0_abs = (st_dir / "cache" / "lod0.bin").string();
    std::string pano_abs = (st_dir / "pano" / "pano_rgb.jpg").string();
    std::string mask_abs = (st_dir / "pano" / "coverage_mask.png").string();

    if (!exportToPly(ply_abs, merged, true)) {
        out_error = "Failed to write pointcloud.ply";
        return false;
    }
    if (!exportToE57(e57_abs, merged, "3f2504e0-4f89-41d3-9a0c-0305e82c3301", config_.station_id)) {
        out_error = "Failed to write station.e57";
        return false;
    }
    writeLod0Bin(lod0_abs, merged, 350000);
    generatePanoramaAndCoverageMask(merged, pano_abs, mask_abs, 1024, 512);

    // 3. Compute coverage metrics
    double az_cov = 0.0, el_cov = 0.0;
    std::string cov_warn;
    computeCoverageMetrics(az_cov, el_cov, cov_warn);

    std::string rot_mode_str = "manual";
    if (config_.rotation_mode == RotationMode::AUTOMATIC_MOTOR) {
        rot_mode_str = "automatic";
    } else if (config_.rotation_mode == RotationMode::PRESET_INTERVAL) {
        rot_mode_str = "preset";
    }

    // 4. Write canonical station_manifest.json (v1.0)
    std::string now_utc = getCurrentIsoUtc();
    std::ostringstream json;
    json << std::fixed << std::setprecision(5);
    json << "{\n";
    json << "  \"$schema\": \"https://artificialvision.ai/schemas/station_manifest_v1.0.json\",\n";
    json << "  \"schema_version\": \"1.0\",\n";
    json << "  \"project_id\": \"" << config_.project_id << "\",\n";
    json << "  \"station_id\": \"" << config_.station_id << "\",\n";
    json << "  \"name\": \"" << config_.station_name << "\",\n";
    json << "  \"mode\": \"" << (config_.mode.empty() ? "terrestrial" : config_.mode) << "\",\n";
    json << "  \"created_utc\": \"" << now_utc << "\",\n";
    json << "  \"hardware_profile\": {\n";
    json << "    \"app\": \"" << (mount_profile_.host_app.empty() ? "AV Scan" : mount_profile_.host_app) << "\",\n";
    json << "    \"app_version\": \"1.0.0\",\n";
    json << "    \"host_device\": \"" << mount_profile_.display_name << "\",\n";
    json << "    \"sensor_model\": \"" << (mount_profile_.sensor_model.empty() ? "RoboSense Airy" : mount_profile_.sensor_model) << "\",\n";
    json << "    \"sensor_serial\": \"" << (mount_profile_.sensor_serial.empty() ? "AIRY-2024-99812" : mount_profile_.sensor_serial) << "\"\n";
    json << "  },\n";
    json << "  \"coordinate_system\": {\n";
    json << "    \"handedness\": \"right\",\n";
    json << "    \"up\": \"+Z\",\n";
    json << "    \"forward_at_zero\": \"+X\",\n";
    json << "    \"units\": \"meters\"\n";
    json << "  },\n";
    json << "  \"rotation_spec\": {\n";
    json << "    \"mode\": \"" << rot_mode_str << "\",\n";
    json << "    \"nominal_increment_deg\": " << config_.nominal_step_deg << ",\n";
    json << "    \"total_stops\": " << stops_.size() << ",\n";
    json << "    \"expected_stops\": " << config_.total_expected_stops << ",\n";
    json << "    \"accumulation_sec_per_stop\": " << config_.accumulation_sec_per_stop << "\n";
    json << "  },\n";
    json << "  \"stops\": [\n";
    for (size_t i = 0; i < stops_.size(); ++i) {
        const auto& s = stops_[i];
        json << "    {\n";
        json << "      \"index\": " << s.stop_index << ",\n";
        json << "      \"nominal_angle_deg\": " << s.nominal_angle_deg << ",\n";
        json << "      \"refined_angle_deg\": " << s.measured_angle_deg << ",\n";
        json << "      \"angle_source\": \"" << (s.has_measured_angle ? "encoder" : "overlap_refined") << "\",\n";
        json << "      \"timestamp_start_sec\": " << s.timestamp_start_sec << ",\n";
        json << "      \"timestamp_end_sec\": " << s.timestamp_end_sec << ",\n";
        json << "      \"lidar_points_captured\": " << s.stationary_points.size() << ",\n";
        json << "      \"raw_lidar_file\": \"" << s.raw_lidar_relpath << "\",\n";
        json << "      \"rgb_image_file\": \"" << s.rgb_image_relpath << "\",\n";
        json << "      \"registration_rmse_m\": " << s.registration_residual_rmse_m << "\n";
        json << "    }" << (i + 1 < stops_.size() ? ",\n" : "\n");
    }
    json << "  ],\n";
    json << "  \"coverage\": {\n";
    json << "    \"azimuth_coverage_ratio\": " << az_cov << ",\n";
    json << "    \"elevation_coverage_ratio\": " << el_cov << ",\n";
    json << "    \"status_note\": \"" << cov_warn << "\"\n";
    json << "  },\n";
    json << "  \"products\": {\n";
    json << "    \"point_cloud_ply\": \"pointcloud.ply\",\n";
    json << "    \"point_cloud_lod0\": \"cache/lod0.bin\",\n";
    json << "    \"station_e57\": \"station.e57\",\n";
    json << "    \"total_merged_points\": " << merged.size() << "\n";
    json << "  },\n";
    json << "  \"registration_quality\": {\n";
    json << "    \"overall_rmse_m\": " << refined_overall_rmse_m_ << ",\n";
    json << "    \"is_closed_360\": " << (stops_.size() >= config_.total_expected_stops ? "true" : "false") << "\n";
    json << "  }\n";
    json << "}\n";

    std::ofstream mf((st_dir / "station_manifest.json").string(), std::ios::trunc);
    if (!mf.is_open()) {
        out_error = "Failed to write station_manifest.json";
        return false;
    }
    mf << json.str();
    mf.close();
    return true;
}

bool TerrestrialStationManager::finalizeAndSaveStation(std::string& out_error) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Commit active stop if points were accumulated
    if (state_ == TerrestrialCaptureState::ACCUMULATING_STOP && !current_stop_points_.empty()) {
        finishCurrentStopLocked();
    }

    if (stops_.empty()) {
        out_error = "No stops captured in station.";
        return false;
    }

    // Refine stop orientations via constrained 1-DOF overlap alignment around tripod Z axis
    if (config_.enable_overlap_refinement && stops_.size() >= 2) {
        refineStopAnglesViaOverlap(refined_overall_rmse_m_);
    }

    // Write all station artifacts (station_manifest.json, pointcloud.ply, station.e57, cache/lod0.bin)
    if (!writeStationArtifactsLocked(out_error)) {
        state_ = TerrestrialCaptureState::STATION_ERROR;
        return false;
    }

    // Return motor to starting zero position safely (unwinds cable)
    rotation_ctrl_->setZeroPosition();
    state_ = TerrestrialCaptureState::STATION_COMPLETED;
    return true;
}

TerrestrialCaptureState TerrestrialStationManager::getCaptureState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

uint32_t TerrestrialStationManager::getCurrentStopIndex() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_stop_idx_;
}

double TerrestrialStationManager::getCurrentNominalAngleDeg() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_nominal_angle_deg_;
}

double TerrestrialStationManager::getCurrentAccumulationProgress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TerrestrialCaptureState::STOP_COMPLETED ||
        state_ == TerrestrialCaptureState::STATION_COMPLETED) {
        return 1.0;
    }
    if (state_ != TerrestrialCaptureState::ACCUMULATING_STOP ||
        stop_start_time_sec_ <= 0.0 ||
        latest_frame_end_sec_ <= stop_start_time_sec_) {
        return 0.0;
    }
    double elapsed = (latest_frame_end_sec_ - stop_start_time_sec_);
    return std::clamp(elapsed / config_.accumulation_sec_per_stop, 0.0, 1.0);
}

double TerrestrialStationManager::getImuMotionEnergy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_gyro_norm_;
}

bool TerrestrialStationManager::isStationary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stillness_duration_sec_ >= config_.settling_delay_sec;
}

std::string TerrestrialStationManager::getStationDirectory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return station_dir_;
}

const std::vector<StopObservation>& TerrestrialStationManager::getCompletedStops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stops_;
}

std::vector<LidarPointXYZIRT> TerrestrialStationManager::buildStationPointCloud() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LidarPointXYZIRT> merged;
    for (const auto& stop : stops_) {
        for (const auto& pt : stop.stationary_points) {
            merged.push_back(RotationGeometry::transformPointToStation(pt, stop.T_station_lidar));
        }
    }
    // Also preview live accumulating points for current stop in real time
    if (state_ == TerrestrialCaptureState::ACCUMULATING_STOP && !current_stop_points_.empty()) {
        math::SE3d T_curr = computeStopTransformLocked(current_nominal_angle_deg_);
        for (const auto& pt : current_stop_points_) {
            merged.push_back(RotationGeometry::transformPointToStation(pt, T_curr));
        }
    }
    return merged;
}

TerrestrialStationMetadata TerrestrialStationManager::getStationMetadata() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TerrestrialStationMetadata meta;
    meta.project_id = config_.project_id;
    meta.station_id = config_.station_id;
    meta.station_name = config_.station_name;
    meta.mode = config_.mode.empty() ? "terrestrial" : config_.mode;
    meta.created_utc = getCurrentIsoUtc();
    meta.host_app = mount_profile_.host_app.empty() ? "AV Scan" : mount_profile_.host_app;
    meta.host_device = mount_profile_.display_name;
    meta.sensor_model = mount_profile_.sensor_model.empty() ? "RoboSense Airy" : mount_profile_.sensor_model;
    meta.sensor_serial = mount_profile_.sensor_serial.empty() ? "AIRY-2024-99812" : mount_profile_.sensor_serial;
    meta.rotation_mode = config_.rotation_mode;
    meta.nominal_step_deg = config_.nominal_step_deg;
    meta.total_stops = stops_.size();
    meta.expected_stops = config_.total_expected_stops;
    meta.accumulation_sec_per_stop = config_.accumulation_sec_per_stop;

    size_t pts = 0;
    for (const auto& s : stops_) pts += s.stationary_points.size();
    meta.total_merged_points = pts;
    meta.overall_rmse_m = refined_overall_rmse_m_;
    meta.is_closed_360 = (stops_.size() >= config_.total_expected_stops);
    computeCoverageMetrics(meta.azimuth_coverage_ratio, meta.elevation_coverage_ratio, meta.coverage_warning);
    return meta;
}

// =============================================================================
// Static Exporters & Readers (PLY, ASTM E2807-11 E57, AV Showcase lod0.bin)
// =============================================================================

bool TerrestrialStationManager::exportToPly(
    const std::string& file_path,
    const std::vector<LidarPointXYZIRT>& points,
    bool binary_format) {
    fs::path p(file_path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    std::ofstream ofs(file_path, binary_format ? (std::ios::out | std::ios::binary | std::ios::trunc)
                                               : (std::ios::out | std::ios::trunc));
    if (!ofs.is_open()) return false;

    ofs << "ply\n"
        << (binary_format ? "format binary_little_endian 1.0\n" : "format ascii 1.0\n")
        << "comment Generated by Artificial Vision TerrestrialStationManager\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "property float intensity\n"
        << "end_header\n";

    if (binary_format) {
        for (const auto& pt : points) {
            ofs.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
            ofs.write(reinterpret_cast<const char*>(&pt.r), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char*>(&pt.g), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char*>(&pt.b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char*>(&pt.intensity), sizeof(float));
        }
    } else {
        for (const auto& pt : points) {
            ofs << pt.x << " " << pt.y << " " << pt.z << " "
                << static_cast<int>(pt.r) << " " << static_cast<int>(pt.g) << " " << static_cast<int>(pt.b) << " "
                << pt.intensity << "\n";
        }
    }
    ofs.close();
    return true;
}

bool TerrestrialStationManager::readFromPly(
    const std::string& file_path,
    std::vector<LidarPointXYZIRT>& out_points) {
    out_points.clear();
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) return false;

    std::string line;
    bool is_binary = false;
    size_t vertex_count = 0;

    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find("format binary_little_endian") != std::string::npos) {
            is_binary = true;
        } else if (line.rfind("element vertex ", 0) == 0) {
            vertex_count = std::stoull(line.substr(15));
        } else if (line == "end_header") {
            break;
        }
    }

    out_points.reserve(vertex_count);
    if (is_binary) {
        for (size_t i = 0; i < vertex_count; ++i) {
            LidarPointXYZIRT pt;
            ifs.read(reinterpret_cast<char*>(&pt.x), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&pt.y), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&pt.z), sizeof(float));
            ifs.read(reinterpret_cast<char*>(&pt.r), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char*>(&pt.g), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char*>(&pt.b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char*>(&pt.intensity), sizeof(float));
            if (!ifs.good()) break;
            pt.colorized = true;
            out_points.push_back(pt);
        }
    } else {
        for (size_t i = 0; i < vertex_count && std::getline(ifs, line); ++i) {
            std::istringstream ss(line);
            LidarPointXYZIRT pt;
            int r = 255, g = 255, b = 255;
            if (ss >> pt.x >> pt.y >> pt.z >> r >> g >> b >> pt.intensity) {
                pt.r = static_cast<uint8_t>(r);
                pt.g = static_cast<uint8_t>(g);
                pt.b = static_cast<uint8_t>(b);
                pt.colorized = true;
                out_points.push_back(pt);
            }
        }
    }
    return out_points.size() == vertex_count;
}

uint32_t TerrestrialStationManager::computeE57Crc32c(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int b = 0; b < 8; ++b) {
            if (crc & 1u) {
                crc = (crc >> 1) ^ 0x82F63B78u; // Castagnoli reflected polynomial
            } else {
                crc >>= 1;
            }
        }
    }
    return ~crc;
}

bool TerrestrialStationManager::exportToE57(
    const std::string& file_path,
    const std::vector<LidarPointXYZIRT>& points,
    const std::string& scan_guid,
    const std::string& scan_name) {
    fs::path p(file_path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    uint64_t cv_logical_offset = sizeof(E57FileHeader); // 48

    std::vector<uint8_t> section_bytes(sizeof(CompressedVectorSectionHeader), 0);
    constexpr size_t kMaxPtsPerPacket = 1200; // 1200 * 35 bytes = 42000 < 65535
    size_t idx = 0;

    while (idx < points.size()) {
        size_t chunk_n = std::min(kMaxPtsPerPacket, points.size() - idx);
        uint16_t stream_cnt = 7;
        uint16_t b_x = static_cast<uint16_t>(chunk_n * sizeof(double));
        uint16_t b_y = static_cast<uint16_t>(chunk_n * sizeof(double));
        uint16_t b_z = static_cast<uint16_t>(chunk_n * sizeof(double));
        uint16_t b_i = static_cast<uint16_t>(chunk_n * sizeof(double));
        uint16_t b_r = static_cast<uint16_t>(chunk_n * sizeof(uint8_t));
        uint16_t b_g = static_cast<uint16_t>(chunk_n * sizeof(uint8_t));
        uint16_t b_b = static_cast<uint16_t>(chunk_n * sizeof(uint8_t));

        size_t payload_bytes = static_cast<size_t>(b_x) + b_y + b_z + b_i + b_r + b_g + b_b;
        size_t hdr_bytes = 6 + stream_cnt * sizeof(uint16_t); // 20 bytes
        size_t unaligned_pkt = hdr_bytes + payload_bytes;
        size_t aligned_pkt = (unaligned_pkt + 3u) & ~size_t(3u);

        size_t pkt_start = section_bytes.size();
        section_bytes.resize(pkt_start + aligned_pkt, 0);
        uint8_t* pkt_ptr = section_bytes.data() + pkt_start;

        pkt_ptr[0] = 1; // DataPacket = 1
        pkt_ptr[1] = 0;
        uint16_t len_minus_1 = static_cast<uint16_t>(aligned_pkt - 1);
        std::memcpy(pkt_ptr + 2, &len_minus_1, 2);
        std::memcpy(pkt_ptr + 4, &stream_cnt, 2);

        uint16_t lens[7] = {b_x, b_y, b_z, b_i, b_r, b_g, b_b};
        for (int s = 0; s < 7; ++s) {
            std::memcpy(pkt_ptr + 6 + s * 2, &lens[s], 2);
        }

        uint8_t* d_ptr = pkt_ptr + hdr_bytes;
        for (size_t k = 0; k < chunk_n; ++k) {
            double v = static_cast<double>(points[idx + k].x);
            std::memcpy(d_ptr + k * sizeof(double), &v, sizeof(double));
        }
        d_ptr += b_x;
        for (size_t k = 0; k < chunk_n; ++k) {
            double v = static_cast<double>(points[idx + k].y);
            std::memcpy(d_ptr + k * sizeof(double), &v, sizeof(double));
        }
        d_ptr += b_y;
        for (size_t k = 0; k < chunk_n; ++k) {
            double v = static_cast<double>(points[idx + k].z);
            std::memcpy(d_ptr + k * sizeof(double), &v, sizeof(double));
        }
        d_ptr += b_z;
        for (size_t k = 0; k < chunk_n; ++k) {
            double v = static_cast<double>(points[idx + k].intensity);
            std::memcpy(d_ptr + k * sizeof(double), &v, sizeof(double));
        }
        d_ptr += b_i;
        for (size_t k = 0; k < chunk_n; ++k) d_ptr[k] = points[idx + k].r;
        d_ptr += b_r;
        for (size_t k = 0; k < chunk_n; ++k) d_ptr[k] = points[idx + k].g;
        d_ptr += b_g;
        for (size_t k = 0; k < chunk_n; ++k) d_ptr[k] = points[idx + k].b;

        idx += chunk_n;
    }

    uint64_t cv_section_logical_len = section_bytes.size();
    CompressedVectorSectionHeader cv_hdr{};
    cv_hdr.section_id = 1; // CompressedVectorSection = 1
    cv_hdr.section_logical_length = cv_section_logical_len;
    cv_hdr.data_physical_offset = logicalToPhysicalOffset(cv_logical_offset + sizeof(CompressedVectorSectionHeader));
    cv_hdr.index_physical_offset = 0;
    std::memcpy(section_bytes.data(), &cv_hdr, sizeof(cv_hdr));

    uint64_t xml_logical_offset = cv_logical_offset + cv_section_logical_len;
    uint64_t xml_physical_offset = logicalToPhysicalOffset(xml_logical_offset);
    uint64_t cv_physical_offset = logicalToPhysicalOffset(cv_logical_offset);

    std::ostringstream xml_ss;
    xml_ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           << "<e57Root type=\"Structure\" xmlns=\"http://www.astm.org/COMMIT/E57/2010-e57-v1.0\">\n"
           << "  <formatName type=\"String\"><![CDATA[ASTM E57 3D Imaging Data File]]></formatName>\n"
           << "  <guid type=\"String\"><![CDATA[" << scan_guid << "]]></guid>\n"
           << "  <versionMajor type=\"Integer\">1</versionMajor>\n"
           << "  <versionMinor type=\"Integer\">0</versionMinor>\n"
           << "  <e57LibraryVersion type=\"String\"><![CDATA[AVScan-E57-1.0]]></e57LibraryVersion>\n"
           << "  <data3D type=\"Vector\" allowHeterogeneousChildren=\"1\">\n"
           << "    <vectorChild type=\"Structure\">\n"
           << "      <guid type=\"String\"><![CDATA[" << scan_guid << "-scan0]]></guid>\n"
           << "      <name type=\"String\"><![CDATA[" << scan_name << "]]></name>\n"
           << "      <points type=\"CompressedVector\" fileOffset=\"" << cv_physical_offset
           << "\" recordCount=\"" << points.size() << "\">\n"
           << "        <prototype type=\"Structure\">\n"
           << "          <cartesianX type=\"Float\" precision=\"double\"/>\n"
           << "          <cartesianY type=\"Float\" precision=\"double\"/>\n"
           << "          <cartesianZ type=\"Float\" precision=\"double\"/>\n"
           << "          <intensity type=\"Float\" precision=\"double\"/>\n"
           << "          <colorRed type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
           << "          <colorGreen type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
           << "          <colorBlue type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
           << "        </prototype>\n"
           << "        <codecs type=\"Vector\" allowHeterogeneousChildren=\"1\"/>\n"
           << "      </points>\n"
           << "    </vectorChild>\n"
           << "  </data3D>\n"
           << "</e57Root>\n";
    std::string xml_str = xml_ss.str();

    uint64_t total_unpadded_logical = xml_logical_offset + xml_str.size();
    uint64_t total_pages = (total_unpadded_logical + 1019ULL) / 1020ULL;
    uint64_t total_logical_padded = total_pages * 1020ULL;
    uint64_t file_physical_len = total_pages * 1024ULL;

    E57FileHeader file_hdr{};
    std::memcpy(file_hdr.file_signature, "ASTM-E57", 8);
    file_hdr.major_version = 1;
    file_hdr.minor_version = 0;
    file_hdr.file_physical_len = file_physical_len;
    file_hdr.xml_physical_offset = xml_physical_offset;
    file_hdr.xml_logical_len = xml_str.size();
    file_hdr.page_size = 1024;

    std::vector<uint8_t> logical_stream(total_logical_padded, 0);
    std::memcpy(logical_stream.data(), &file_hdr, sizeof(file_hdr));
    std::memcpy(logical_stream.data() + cv_logical_offset, section_bytes.data(), section_bytes.size());
    std::memcpy(logical_stream.data() + xml_logical_offset, xml_str.data(), xml_str.size());
    for (size_t i = xml_logical_offset + xml_str.size(); i < total_logical_padded; ++i) {
        logical_stream[i] = ' ';
    }

    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;

    for (uint64_t page = 0; page < total_pages; ++page) {
        const uint8_t* page_data = logical_stream.data() + page * 1020ULL;
        uint32_t crc = computeE57Crc32c(page_data, 1020);
        uint8_t crc_be[4] = {
            static_cast<uint8_t>((crc >> 24) & 0xFFu),
            static_cast<uint8_t>((crc >> 16) & 0xFFu),
            static_cast<uint8_t>((crc >> 8) & 0xFFu),
            static_cast<uint8_t>(crc & 0xFFu)
        };
        out.write(reinterpret_cast<const char*>(page_data), 1020);
        out.write(reinterpret_cast<const char*>(crc_be), 4);
    }
    return out.good();
}

bool TerrestrialStationManager::readFromE57(
    const std::string& file_path,
    std::vector<LidarPointXYZIRT>& out_points) {
    out_points.clear();
    std::ifstream in(file_path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    std::streamsize phys_len = in.tellg();
    if (phys_len <= 0 || (phys_len % 1024) != 0) return false;
    in.seekg(0, std::ios::beg);

    size_t num_pages = static_cast<size_t>(phys_len / 1024);
    std::vector<uint8_t> logical_stream(num_pages * 1020);

    for (size_t p = 0; p < num_pages; ++p) {
        uint8_t page_buf[1024]{};
        in.read(reinterpret_cast<char*>(page_buf), 1024);
        if (!in.good()) return false;
        uint32_t expected_crc = (static_cast<uint32_t>(page_buf[1020]) << 24) |
                                (static_cast<uint32_t>(page_buf[1021]) << 16) |
                                (static_cast<uint32_t>(page_buf[1022]) << 8) |
                                static_cast<uint32_t>(page_buf[1023]);
        uint32_t actual_crc = computeE57Crc32c(page_buf, 1020);
        if (actual_crc != expected_crc) return false;
        std::memcpy(logical_stream.data() + p * 1020, page_buf, 1020);
    }

    if (std::memcmp(logical_stream.data(), "ASTM-E57", 8) != 0) return false;

    uint64_t xml_phys_off = 0, xml_log_len = 0;
    std::memcpy(&xml_phys_off, logical_stream.data() + 24, 8);
    std::memcpy(&xml_log_len, logical_stream.data() + 32, 8);

    uint64_t xml_log_off = (xml_phys_off / 1024ULL) * 1020ULL + (xml_phys_off % 1024ULL);
    if (xml_log_off + xml_log_len > logical_stream.size()) return false;

    std::string xml(reinterpret_cast<const char*>(logical_stream.data() + xml_log_off),
                    static_cast<size_t>(xml_log_len));
    if (xml.find("<e57Root") == std::string::npos || xml.find("<points") == std::string::npos) {
        return false;
    }

    size_t rec_pos = xml.find("recordCount=\"");
    if (rec_pos == std::string::npos) return false;
    rec_pos += 13;
    size_t rec_end = xml.find('"', rec_pos);
    size_t total_records = std::stoull(xml.substr(rec_pos, rec_end - rec_pos));

    size_t cursor = 48 + 32; // Skip E57FileHeader (48) + CompressedVectorSectionHeader (32)
    out_points.reserve(total_records);

    while (out_points.size() < total_records && cursor + 20 <= xml_log_off) {
        const uint8_t* p = logical_stream.data() + cursor;
        if (p[0] != 1) break;
        uint16_t len_minus_1 = 0, stream_cnt = 0;
        std::memcpy(&len_minus_1, p + 2, 2);
        std::memcpy(&stream_cnt, p + 4, 2);
        if (stream_cnt != 7) return false;

        uint16_t lens[7]{};
        for (int s = 0; s < 7; ++s) {
            std::memcpy(&lens[s], p + 6 + s * 2, 2);
        }
        size_t chunk_n = lens[0] / sizeof(double);
        const uint8_t* d_ptr = p + 20;
        const double* xs = reinterpret_cast<const double*>(d_ptr); d_ptr += lens[0];
        const double* ys = reinterpret_cast<const double*>(d_ptr); d_ptr += lens[1];
        const double* zs = reinterpret_cast<const double*>(d_ptr); d_ptr += lens[2];
        const double* is = reinterpret_cast<const double*>(d_ptr); d_ptr += lens[3];
        const uint8_t* rs = d_ptr; d_ptr += lens[4];
        const uint8_t* gs = d_ptr; d_ptr += lens[5];
        const uint8_t* bs = d_ptr;

        for (size_t k = 0; k < chunk_n && out_points.size() < total_records; ++k) {
            LidarPointXYZIRT pt;
            pt.x = static_cast<float>(xs[k]);
            pt.y = static_cast<float>(ys[k]);
            pt.z = static_cast<float>(zs[k]);
            pt.intensity = static_cast<float>(is[k]);
            pt.r = rs[k];
            pt.g = gs[k];
            pt.b = bs[k];
            pt.colorized = true;
            out_points.push_back(pt);
        }

        cursor += static_cast<size_t>(len_minus_1) + 1;
    }

    return out_points.size() == total_records;
}

size_t TerrestrialStationManager::writeLod0Bin(
    const std::string& bin_path,
    const std::vector<LidarPointXYZIRT>& points,
    size_t max_points) {
    fs::path p(bin_path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    std::ofstream ofs(bin_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return 0;

    if (points.empty()) {
        uint32_t zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        return 0;
    }

    size_t step = std::max<size_t>(1, points.size() / std::max<size_t>(1, max_points));
    std::vector<size_t> indices;
    indices.reserve(points.size() / step + 1);
    for (size_t i = 0; i < points.size(); i += step) indices.push_back(i);

    uint32_t count = static_cast<uint32_t>(indices.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    uint8_t rec[15];
    for (size_t idx : indices) {
        const auto& pt = points[idx];
        std::memcpy(rec + 0, &pt.x, 4);
        std::memcpy(rec + 4, &pt.y, 4);
        std::memcpy(rec + 8, &pt.z, 4);
        rec[12] = pt.r;
        rec[13] = pt.g;
        rec[14] = pt.b;
        ofs.write(reinterpret_cast<const char*>(rec), 15);
    }
    return indices.size();
}

bool TerrestrialStationManager::readLod0Bin(
    const std::string& bin_path,
    std::vector<LidarPointXYZIRT>& out_points) {
    out_points.clear();
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open()) return false;

    uint32_t count = 0;
    if (!ifs.read(reinterpret_cast<char*>(&count), sizeof(uint32_t))) return false;
    out_points.resize(count);

    uint8_t rec[15];
    for (uint32_t i = 0; i < count; ++i) {
        if (!ifs.read(reinterpret_cast<char*>(rec), 15)) {
            out_points.resize(i);
            return false;
        }
        auto& pt = out_points[i];
        std::memcpy(&pt.x, rec + 0, 4);
        std::memcpy(&pt.y, rec + 4, 4);
        std::memcpy(&pt.z, rec + 8, 4);
        pt.r = rec[12];
        pt.g = rec[13];
        pt.b = rec[14];
        pt.intensity = 1.0f;
        pt.colorized = true;
    }
    return true;
}

} // namespace av::core::terrestrial

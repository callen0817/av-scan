#include <QCoreApplication>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/processing/point_cloud_processor.hpp"
#include "core/processing/structured_walkthrough_exporter.hpp"
#include "core/storage/storage_engine.hpp"
#include "gui/qml_bridge.hpp"
#include "platform/linux/linux_platform.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace av::core::processing;
using namespace av::core::schemas;

#define CHECK(cond, msg)                                                                   \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::cerr << "[INTEROP FAIL] " << msg << " (line " << __LINE__ << ")\n";       \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

namespace {

void addBoxSurface(
    std::vector<PointXYZI>& pts,
    double x0, double x1,
    double y0, double y1,
    double z0, double z1,
    double step,
    uint8_t r, uint8_t g, uint8_t b,
    bool floor_on = true,
    bool ceiling_on = true)
{
    if (floor_on) {
        for (double x = x0; x <= x1 + 1e-6; x += step) {
            for (double y = y0; y <= y1 + 1e-6; y += step) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z0), 0.8f, r, g, b, true});
            }
        }
    }
    if (ceiling_on) {
        for (double x = x0; x <= x1 + 1e-6; x += step) {
            for (double y = y0; y <= y1 + 1e-6; y += step) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z1), 0.8f, 190, 190, 195, true});
            }
        }
    }
}

void addWallY(
    std::vector<PointXYZI>& pts,
    double y_fixed,
    double x0, double x1,
    double z0, double z1,
    double step,
    uint8_t r, uint8_t g, uint8_t b,
    double door_x0 = 1e9, double door_x1 = -1e9, double door_z1 = 2.1)
{
    for (double x = x0; x <= x1 + 1e-6; x += step) {
        for (double z = z0; z <= z1 + 1e-6; z += step) {
            if (x >= door_x0 && x <= door_x1 && z <= door_z1) continue;
            pts.push_back({static_cast<float>(x), static_cast<float>(y_fixed), static_cast<float>(z), 0.9f, r, g, b, true});
        }
    }
}

void addWallX(
    std::vector<PointXYZI>& pts,
    double x_fixed,
    double y0, double y1,
    double z0, double z1,
    double step,
    uint8_t r, uint8_t g, uint8_t b)
{
    for (double y = y0; y <= y1 + 1e-6; y += step) {
        for (double z = z0; z <= z1 + 1e-6; z += step) {
            pts.push_back({static_cast<float>(x_fixed), static_cast<float>(y), static_cast<float>(z), 0.9f, r, g, b, true});
        }
    }
}

std::vector<PointXYZI> buildTwoStoryBuildingCloud(bool remodel_april = false) {
    std::vector<PointXYZI> pts;
    const double step = 0.11;

    // Level 1 main hall: x in [-4.0, 4.0], y in [-2.0, 1.0], z in [0.0, 2.7]
    addBoxSurface(pts, -4.0, 4.0, -2.0, 1.0, 0.0, 2.7, step, 145, 140, 132, true, true);
    addWallY(pts, -2.0, -4.0, 4.0, 0.0, 2.7, step, 175, 175, 180);
    addWallX(pts, -4.0, -2.0, 1.0, 0.0, 2.7, step, 165, 170, 180);
    addWallX(pts,  4.0, -2.0, 1.0, 0.0, 2.7, step, 185, 170, 155);

    // Solid interior partition wall between Hallway and Bathroom (y = 1.00 .. 1.24),
    // with doorway ONLY at East end x in [3.0, 3.85]
    for (double yw : {1.00, 1.12, 1.24}) {
        addWallY(pts, yw, -4.0, 4.0, 0.0, 2.7, 0.08, 210, 210, 215, 3.0, 3.85, 2.2);
    }

    // Level 1 Bathroom: x in [0.5, 4.0], y in [1.24, 3.8], z in [0.0, 2.7]
    addBoxSurface(pts, 0.5, 4.0, 1.24, 3.8, 0.0, 2.7, step, 160, 170, 175, true, true);
    addWallX(pts, 0.5, 1.24, 3.8, 0.0, 2.7, step, 180, 180, 185);
    // East bathroom wall: warm amber
    addWallX(pts, 4.0, 1.24, 3.8, 0.0, 2.7, step, 240, 165, 45);
    // North bathroom vanity wall (y = 3.8): distinct bright cyan tile
    addWallY(pts, 3.8, 0.5, 4.0, 0.0, 2.7, 0.08, 35, 215, 245);

    if (remodel_april) {
        // Add a small linen cabinet on the West bathroom wall in April dataset
        addBoxSurface(pts, 0.6, 1.1, 1.5, 2.1, 0.0, 1.8, 0.09, 200, 150, 110, true, true);
    }

    // Staircase rising from Level 1 (-3.2, -0.5, 0.0) to Level 2 (-3.2, -0.5, 3.2)
    for (int s = 0; s <= 16; ++s) {
        const double frac = static_cast<double>(s) / 16.0;
        const double z_step = frac * 3.2;
        const double x_c = -3.4 + frac * 0.8;
        for (double dx = -0.35; dx <= 0.35; dx += 0.10) {
            for (double dy = -0.55; dy <= 0.55; dy += 0.10) {
                pts.push_back({static_cast<float>(x_c + dx), static_cast<float>(-0.4 + dy), static_cast<float>(z_step), 0.85f, 170, 155, 130, true});
            }
        }
    }

    // Level 2 upper floor: x in [-4.0, 2.0], y in [-2.0, 2.0], z in [3.2, 5.8]
    addBoxSurface(pts, -4.0, 2.0, -2.0, 2.0, 3.2, 5.8, step, 135, 145, 160, true, true);
    addWallY(pts, -2.0, -4.0, 2.0, 3.2, 5.8, step, 175, 180, 190);
    addWallY(pts,  2.0, -4.0, 2.0, 3.2, 5.8, step, 175, 180, 190);
    addWallX(pts, -4.0, -2.0, 2.0, 3.2, 5.8, step, 170, 175, 185);
    addWallX(pts,  2.0, -2.0, 2.0, 3.2, 5.8, step, 170, 175, 185);

    return pts;
}

Pose3D makeYawPose(uint64_t ts, double x, double y, double z, double yaw_rad) {
    Pose3D p;
    p.timestamp_ns = ts;
    p.position = {x, y, z};
    p.orientation = {0.0, 0.0, std::sin(yaw_rad * 0.5), std::cos(yaw_rad * 0.5)};
    return p;
}

std::vector<PointXYZI> transformCloudRigid4Dof(
    const std::vector<PointXYZI>& cloud,
    double yaw_rad, double tx, double ty, double tz)
{
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    std::vector<PointXYZI> out;
    out.reserve(cloud.size());
    for (const auto& p : cloud) {
        PointXYZI q = p;
        q.x = static_cast<float>(c * p.x - s * p.y + tx);
        q.y = static_cast<float>(s * p.x + c * p.y + ty);
        q.z = static_cast<float>(p.z + tz);
        out.push_back(q);
    }
    return out;
}

bool checkPngSignature(const fs::path& png_path) {
    std::ifstream ifs(png_path, std::ios::binary);
    if (!ifs.is_open()) return false;
    uint8_t sig[8]{};
    ifs.read(reinterpret_cast<char*>(sig), 8);
    if (ifs.gcount() != 8) return false;
    const uint8_t expected[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    for (int i = 0; i < 8; ++i) {
        if (sig[i] != expected[i]) return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    std::cout << "=====================================================================\n"
              << "  AV SCAN — ECOSYSTEM WALKTHROUGH INTEROPERABILITY ACCEPTANCE SUITE  \n"
              << "=====================================================================\n";

    fs::path sandbox = fs::current_path() / "test_walkthrough_interop_sandbox";
    std::error_code ec;
    if (fs::exists(sandbox, ec)) {
        fs::remove_all(sandbox, ec);
    }
    fs::create_directories(sandbox, ec);

    const fs::path avproj_dir = sandbox / "AV_Showcase_Project";
    const fs::path dec_bundle_dir = sandbox / "bundle_2026_12_04";
    const fs::path apr_bundle_dir = sandbox / "bundle_2027_04_13";

    // =========================================================================
    // GATE 1: Multi-Story Floor Detection, Wall-Occlusion Graph, KeyPlans,
    //         lod0.bin & ASTM E2807-11 Structured E57 Verification (2026-12-04)
    // =========================================================================
    std::cout << "\n[GATE 1] Exporting Baseline Multi-Story Dataset (2026-12-04)...\n";
    auto cloud_dec = buildTwoStoryBuildingCloud(false);

    std::vector<Pose3D> traj_dec = {
        makeYawPose(1000000ULL,  1.5, -0.4, 1.60, 0.0),          // South of solid bathroom wall
        makeYawPose(2000000ULL,  3.4,  0.2, 1.60, M_PI * 0.5),   // In front of bathroom East doorway
        makeYawPose(3000000ULL,  3.4,  2.2, 1.60, M_PI * 0.75),  // Inside bathroom East
        makeYawPose(4000000ULL, -3.0, -0.4, 1.60, M_PI),         // Bottom of stairs (Level 1)
        makeYawPose(5000000ULL, -2.4, -0.4, 4.80, 0.0),          // Top of stairs (Level 2)
        makeYawPose(6000000ULL,  0.2,  0.5, 4.80, 0.0)           // Upper room (Level 2)
    };

    std::vector<FlashStationAnchor> flash_dec = {
        {"flash_bath", "Bathroom Center", {2.2, 2.5, 1.60}, M_PI * 0.5, true}
    };

    WalkthroughExportConfig cfg_dec;
    cfg_dec.station_spacing_m = 1.6;
    cfg_dec.min_wall_clearance_m = 0.35;
    cfg_dec.pano_width = 1024;
    cfg_dec.pano_height = 512;
    cfg_dec.project_id = "proj_duplex_01";
    cfg_dec.project_name = "Duplex Renovation";
    cfg_dec.scan_date = "2026-12-04";
    cfg_dec.scan_id = "scan_dec04";
    cfg_dec.av_showcase_project_dir = avproj_dir.string();

    auto res_dec = StructuredWalkthroughExporter::exportWalkthroughBundle(
        dec_bundle_dir.string(), cloud_dec, traj_dec, flash_dec, {}, cfg_dec, nullptr);

    CHECK(res_dec.success, "Baseline walkthrough bundle export failed: " + res_dec.error_message);
    CHECK(res_dec.floor_count == 2, "Expected 2 stories (Level 1 & Level 2), got " + std::to_string(res_dec.floor_count));
    CHECK(res_dec.station_count >= 5, "Expected at least 5 stations across 2 floors");

    // Validate ASTM E2807-11 Structured E57 CRC-32C pages & associatedData3DGuid
    auto e57_val = StructuredWalkthroughExporter::validateStructuredE57(res_dec.structured_e57_path);
    std::cout << "  E57 Pages verified: " << e57_val.crc_verified_pages << "/" << e57_val.total_pages
              << " | Stations: " << e57_val.station_count
              << " | Panos: " << e57_val.image_count << "\n";
    CHECK(e57_val.valid, "Structured E57 failed ASTM E2807-11 validation: " + e57_val.error_message);
    CHECK(e57_val.crc_verified_pages == e57_val.total_pages && e57_val.total_pages > 4,
          "All E57 1024-byte physical pages must pass Castagnoli CRC-32C check");
    CHECK(e57_val.associated_guids_matched, "images2D associatedData3DGuid must match data3D guids");

    // Validate cache/lod0.bin exact binary format (4 + 15 * N bytes)
    const uint64_t lod0_bytes = fs::file_size(res_dec.lod0_bin_path, ec);
    CHECK(lod0_bytes == 4ULL + 15ULL * res_dec.lod0_points_written,
          "lod0.bin byte size must equal 4 + 15 * numPoints");
    std::vector<PointXYZI> roundtrip_lod0;
    CHECK(StructuredWalkthroughExporter::readLod0Bin(res_dec.lod0_bin_path, roundtrip_lod0),
          "readLod0Bin failed");
    CHECK(roundtrip_lod0.size() == res_dec.lod0_points_written, "lod0.bin point count mismatch");

    // Inspect av_walkthrough_bundle.json for multi-floor keyplans, wall occlusion, and stair links
    std::ifstream b_ifs(res_dec.bundle_json_path);
    json bundle_dec = json::parse(b_ifs);
    CHECK(bundle_dec["format"] == "av_spatial_bundle", "Bundle format must be av_spatial_bundle");
    CHECK(bundle_dec["schema_version"] == "1.1", "Bundle schema_version must be 1.1");
    CHECK(bundle_dec["nav_graph"] == "los_v1", "Bundle nav_graph must be los_v1");
    CHECK(bundle_dec.contains("registration_metrics"), "Bundle must include registration_metrics");
    CHECK(bundle_dec["transform_to_project"].is_array() && bundle_dec["transform_to_project"].size() == 16,
          "Bundle transform_to_project must be a flat 16-float row-major array");
    CHECK(bundle_dec["floors"].size() == 2, "Bundle JSON must list 2 floors");

    // Verify all standardized av-scanar + AV Structure files exist on baseline dataset
    CHECK(fs::exists(res_dec.pointcloud_airy_ply_path) && fs::file_size(res_dec.pointcloud_airy_ply_path, ec) > 64,
          "pointcloud_airy.ply must exist and be non-empty");
    CHECK(fs::exists(res_dec.pointcloud_airy_pcd_path) && fs::file_size(res_dec.pointcloud_airy_pcd_path, ec) > 64,
          "pointcloud_airy.pcd must exist and be non-empty");
    CHECK(fs::exists(res_dec.pointcloud_stereo_ply_path) && fs::file_size(res_dec.pointcloud_stereo_ply_path, ec) > 64,
          "pointcloud_stereo.ply must exist and be non-empty for RGB cloud");
    CHECK(fs::exists(res_dec.pointcloud_stereo_pcd_path) && fs::file_size(res_dec.pointcloud_stereo_pcd_path, ec) > 64,
          "pointcloud_stereo.pcd must exist and be non-empty for RGB cloud");
    CHECK(fs::exists(res_dec.pointcloud_fused_rgb_ply_path) && fs::file_size(res_dec.pointcloud_fused_rgb_ply_path, ec) > 64,
          "pointcloud_fused_rgb.ply must exist and be non-empty");
    CHECK(fs::exists(res_dec.pointcloud_fused_rgb_pcd_path) && fs::file_size(res_dec.pointcloud_fused_rgb_pcd_path, ec) > 64,
          "pointcloud_fused_rgb.pcd must exist and be non-empty");
    CHECK(fs::exists(res_dec.trajectory_csv_path) && fs::file_size(res_dec.trajectory_csv_path, ec) > 32,
          "trajectory.csv must exist and be non-empty");
    CHECK(fs::exists(res_dec.trajectory_tum_path) && fs::file_size(res_dec.trajectory_tum_path, ec) > 32,
          "trajectory_tum.txt must exist and be non-empty");
    CHECK(fs::exists(dec_bundle_dir / "flash_anchors.json"), "flash_anchors.json must exist");
    CHECK(fs::exists(res_dec.scan_metadata_path), "scan_metadata.json must exist");
    CHECK(fs::exists(res_dec.stations_manifest_path), "stations_manifest.json must exist");

    {
        std::ifstream sm_ifs(res_dec.stations_manifest_path);
        json st_manifest = json::parse(sm_ifs);
        CHECK(st_manifest["schema"] == "ArtificialVision.AVStructure.StationsManifest.v1",
              "stations_manifest.json schema must match ArtificialVision.AVStructure.StationsManifest.v1");
        CHECK(st_manifest["stations"].size() == res_dec.station_count,
              "stations_manifest.json station count must match exported station count");
        for (const auto& st_entry : st_manifest["stations"]) {
            const auto& files = st_entry["files"];
            CHECK(fs::exists(dec_bundle_dir / files["panorama_rgb"].get<std::string>()),
                  "Missing station panorama_rgb.jpg");
            CHECK(fs::exists(dec_bundle_dir / files["depth_map_mm"].get<std::string>()),
                  "Missing station depth_map_mm.png");
            CHECK(fs::exists(dec_bundle_dir / files["coverage_mask"].get<std::string>()),
                  "Missing station coverage_mask.png");
            CHECK(fs::exists(dec_bundle_dir / files["provenance_code"].get<std::string>()),
                  "Missing station provenance_code.png");
            CHECK(fs::exists(dec_bundle_dir / files["structured_scan_ply"].get<std::string>()),
                  "Missing station structured_scan.ply");
        }
    }

    for (const auto& fl : bundle_dec["floors"]) {
        CHECK(fl.contains("floor_elevation"), "Floor metadata must include floor_elevation");
        fs::path kp_path = dec_bundle_dir / fl["keyplan_path"].get<std::string>();
        CHECK(fs::exists(kp_path), "Missing floor keyplan PNG: " + kp_path.string());
        CHECK(checkPngSignature(kp_path), "Invalid PNG header on floor keyplan: " + kp_path.string());
        CHECK(fl.contains("world_bounds"), "Floor metadata must include metric world_bounds");
        CHECK(fl["world_bounds"]["max_x"].get<double>() > fl["world_bounds"]["min_x"].get<double>(),
              "Keyplan world_bounds X span must be positive");
    }

    // Verify wall occlusion: station near Hallway South (1.5, -0.4) must NOT link directly through solid wall to Bathroom (2.2, 2.5)
    std::string bath_id;
    std::string hall_south_id;
    bool found_stair_link = false;
    for (const auto& p : bundle_dec["panos"]) {
        const double px = p["position"][0].get<double>();
        const double py = p["position"][1].get<double>();
        const double pz = p["position"][2].get<double>();
        if (pz < 2.6 && std::hypot(px - 2.2, py - 2.5) < 0.45) {
            bath_id = p["id"].get<std::string>();
        }
        if (pz < 2.6 && py < 0.2 && std::fabs(px - 1.5) < 0.75) {
            hall_south_id = p["id"].get<std::string>();
        }
        if (p.value("stair_link", false)) {
            for (const auto& nid : p["neighbors"]) {
                for (const auto& q : bundle_dec["panos"]) {
                    if (q["id"] == nid && q["floor_id"] != p["floor_id"]) {
                        found_stair_link = true;
                    }
                }
            }
        }
    }
    CHECK(!bath_id.empty(), "Flash station in bathroom must be preserved");
    CHECK(!hall_south_id.empty(), "Hallway station south of solid wall must exist");
    CHECK(found_stair_link, "Multi-story navigation graph must include cross-floor staircase link with stair_link=true");

    for (const auto& p : bundle_dec["panos"]) {
        if (p["id"] == hall_south_id) {
            for (const auto& nid : p["neighbors"]) {
                CHECK(nid.get<std::string>() != bath_id,
                      "Wall-occlusion check failed: link penetrated solid bathroom partition wall!");
            }
        }
    }
    std::cout << "  [PASS] Gate 1: 2 floors, keyplans, wall-occlusion, stair link, lod0.bin & E57 verified.\n";

    // =========================================================================
    // GATE 2: 4D Historical Time-Travel Continuity (2026-12-04 -> 2027-04-13)
    //         with Arbitrary Rotated/Shifted SLAM Origin & Opposite Walk Yaw
    // =========================================================================
    std::cout << "\n[GATE 2] Exporting Remodeled Dataset (2027-04-13) with +30 deg / shifted SLAM frame...\n";
    // Simulate Scan #2 starting in a different room facing +30 deg yaw and shifted by (+4.2, -2.8, +0.35)m
    // Note: T_apr_from_canon moves canonical world -> April local SLAM frame;
    // ICP `transform_to_project` must recover the inverse (April local -> Canonical Project Baseline)!
    const double slam_yaw = 30.0 * M_PI / 180.0;
    const double slam_tx = 4.20;
    const double slam_ty = -2.80;
    const double slam_tz = 0.35;

    auto cloud_apr_canon = buildTwoStoryBuildingCloud(true);
    auto cloud_apr_local = transformCloudRigid4Dof(cloud_apr_canon, slam_yaw, slam_tx, slam_ty, slam_tz);

    auto toAprLocalPt = [&](double cx, double cy, double cz) -> Vector3D {
        const double c = std::cos(slam_yaw);
        const double s = std::sin(slam_yaw);
        return {c * cx - s * cy + slam_tx, s * cx + c * cy + slam_ty, cz + slam_tz};
    };

    // Walk in the opposite direction (-Y / South-facing yaw in bathroom instead of +Y North)
    Vector3D bath_apr_pos = toAprLocalPt(2.22, 2.48, 1.60);
    Vector3D door_apr_pos = toAprLocalPt(3.40, 0.25, 1.60);
    Vector3D upper_apr_pos = toAprLocalPt(-0.20, 0.40, 4.80);

    std::vector<Pose3D> traj_apr = {
        makeYawPose(1000000ULL, bath_apr_pos.x, bath_apr_pos.y, bath_apr_pos.z, slam_yaw - M_PI * 0.5),
        makeYawPose(2000000ULL, door_apr_pos.x, door_apr_pos.y, door_apr_pos.z, slam_yaw - M_PI * 0.5),
        makeYawPose(3000000ULL, upper_apr_pos.x, upper_apr_pos.y, upper_apr_pos.z, slam_yaw + M_PI)
    };

    WalkthroughExportConfig cfg_apr = cfg_dec;
    cfg_apr.scan_date = "2027-04-13";
    cfg_apr.scan_id = "scan_apr13";

    auto res_apr = StructuredWalkthroughExporter::exportWalkthroughBundle(
        apr_bundle_dir.string(), cloud_apr_local, traj_apr, {}, {}, cfg_apr, nullptr);

    CHECK(res_apr.success, "April historical bundle export failed: " + res_apr.error_message);
    CHECK(fs::exists(res_apr.avproj_path), "project.avproj must exist after multi-date export");

    std::ifstream proj_ifs(res_apr.avproj_path);
    json avproj = json::parse(proj_ifs);
    CHECK(avproj["schema_version"] == "1.1", "project.avproj schema_version must be 1.1");
    CHECK(avproj["baseline_scan_id"] == "scan_dec04", "project.avproj baseline_scan_id must be scan_dec04");
    CHECK(avproj["scans"].size() == 2, "project.avproj must contain 2 chronological historical scans");
    CHECK(avproj["scans"][0]["scan_date"] == "2026-12-04", "Scan #0 must be 2026-12-04");
    CHECK(avproj["scans"][1]["scan_date"] == "2027-04-13", "Scan #1 must be 2027-04-13");
    CHECK(avproj["scans"][1]["scan_dir"] == "Scans/2027-04-13", "Scan #1 scan_dir must be Scans/2027-04-13");

    // Verify 4D Time-Travel Teleportation (AV Showcase AutoStack in Canonical Project Space):
    // User is standing in the Bathroom in 2026-12-04 at canonical (2.20, 2.50, 1.60) facing North (+Y vanity).
    const auto& T_arr = avproj["scans"][1]["transform_to_project"];
    CHECK(T_arr.is_array() && T_arr.size() == 16, "Scan #1 must store flat 16-float transform_to_project");
    double T_proj_from_apr[16]{};
    for (int i = 0; i < 16; ++i) T_proj_from_apr[i] = T_arr[i].get<double>();
    const double recovered_yaw_deg = std::atan2(T_proj_from_apr[4], T_proj_from_apr[0]) * 180.0 / M_PI;
    CHECK(std::fabs(recovered_yaw_deg - (-30.0)) < 2.0,
          "4-DOF ICP must recover -30 deg yaw inverse alignment, got " + std::to_string(recovered_yaw_deg));

    const fs::path apr_in_proj_bundle =
        avproj_dir / avproj["scans"][1]["scan_dir"].get<std::string>() / "av_walkthrough_bundle.json";
    CHECK(fs::exists(apr_in_proj_bundle), "Companion av_walkthrough_bundle.json must exist inside Scans/2027-04-13");
    std::ifstream apr_b_ifs(apr_in_proj_bundle);
    json apr_bundle_proj = json::parse(apr_b_ifs);

    // Stations inside project.avproj / Scans/2027-04-13 are pre-transformed into Canonical Project Space
    double best_bath_dist_m = 1e9;
    for (const auto& p : apr_bundle_proj["panos"]) {
        const double wx = p["position"][0].get<double>();
        const double wy = p["position"][1].get<double>();
        const double wz = p["position"][2].get<double>();
        const double d = std::sqrt((wx - 2.20) * (wx - 2.20) + (wy - 2.50) * (wy - 2.50) + (wz - 1.60) * (wz - 1.60));
        if (d < best_bath_dist_m) best_bath_dist_m = d;
    }

    std::cout << "  4D Time-Travel Bathroom Position Match Error: " << (best_bath_dist_m * 100.0)
              << " cm | Recovered Yaw: " << recovered_yaw_deg
              << " deg | ICP RMSE: " << (res_apr.registration_rmse_m * 100.0) << " cm\n";
    CHECK(best_bath_dist_m < 0.12,
          "Switching from Dec 4 to Apr 13 must land in the exact same bathroom within 12 cm!");
    std::cout << "  [PASS] Gate 2: 4D Historical Time-Travel Continuity verified (< 12 cm).\n";

    // =========================================================================
    // GATE 3: PointCloudProcessor::exportPointCloud(E57_STRUCTURED) & QmlBridge
    // =========================================================================
    std::cout << "\n[GATE 3] Verifying PointCloudProcessor E57 export & QmlBridge walkthrough integration...\n";
    const fs::path direct_e57 = sandbox / "direct_export.e57";
    auto pcp_res = PointCloudProcessor::exportPointCloud(
        cloud_dec, direct_e57.string(), ExportFormat::E57_STRUCTURED);
    CHECK(pcp_res.success, "PointCloudProcessor E57_STRUCTURED export failed: " + pcp_res.error_message);
    CHECK(fs::exists(direct_e57), "Direct exported .e57 file missing");
    auto direct_val = StructuredWalkthroughExporter::validateStructuredE57(direct_e57.string());
    CHECK(direct_val.valid, "Direct exported .e57 failed ASTM E2807-11 validation");

    auto storage = std::make_shared<av::core::storage::StorageEngine>();
    CHECK(storage->initialize((sandbox / "av_storage").string()), "StorageEngine initialize failed");
    auto platform = std::make_shared<av::platform::linux_os::LinuxPlatformAdapter>();
    av::gui::QmlBridge bridge(storage, platform);
    CHECK(bridge.createProject("BridgeWalkthroughProj", "QmlBridge E2E Interop Test"),
          "QmlBridge createProject failed");
    bridge.dropFlashStationAnchor("Lobby Entry");
    CHECK(bridge.flashAnchorCount() == 1, "QmlBridge flashAnchorCount should increment to 1");
    bridge.clearFlashStationAnchors();
    CHECK(bridge.flashAnchorCount() == 0, "QmlBridge clearFlashStationAnchors should reset to 0");
    std::cout << "  [PASS] Gate 3: PointCloudProcessor & QmlBridge APIs verified.\n";

    // =========================================================================
    // GATE 4: LiDAR-Only (No RGB) Point Cloud & Black-and-White Panorama Parity
    // =========================================================================
    std::cout << "\n[GATE 4] Verifying LiDAR-Only (No RGB) Standardized Outputs & B&W Panoramas...\n";
    std::vector<PointXYZI> cloud_lidar_only = cloud_dec;
    for (size_t i = 0; i < cloud_lidar_only.size(); ++i) {
        auto& pt = cloud_lidar_only[i];
        pt.has_color = false;
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
        // Vary LiDAR reflectivity/intensity realistically across geometry
        pt.intensity = static_cast<float>(0.20 + 0.75 * (0.5 + 0.5 * std::sin(pt.x * 0.9 + pt.y * 0.7 + pt.z * 1.1)));
    }

    const fs::path lidar_bundle_dir = sandbox / "bundle_lidar_only";
    WalkthroughExportConfig cfg_lidar = cfg_dec;
    cfg_lidar.scan_date = "2027-06-01";
    cfg_lidar.scan_id = "scan_lidar_bw";
    cfg_lidar.av_showcase_project_dir = (sandbox / "AV_Showcase_LiDAR_Only").string();

    auto res_lidar = StructuredWalkthroughExporter::exportWalkthroughBundle(
        lidar_bundle_dir.string(), cloud_lidar_only, traj_dec, flash_dec, {}, cfg_lidar, nullptr);

    CHECK(res_lidar.success, "LiDAR-only walkthrough bundle export failed: " + res_lidar.error_message);
    CHECK(res_lidar.is_monochrome_lidar_only, "LiDAR-only export must set is_monochrome_lidar_only = true");

    // Verify all standardized av-scanar files exist for LiDAR-only scan
    CHECK(fs::exists(res_lidar.pointcloud_airy_ply_path), "LiDAR-only pointcloud_airy.ply must exist");
    CHECK(fs::exists(res_lidar.pointcloud_airy_pcd_path), "LiDAR-only pointcloud_airy.pcd must exist");
    CHECK(fs::exists(res_lidar.pointcloud_stereo_ply_path), "LiDAR-only pointcloud_stereo.ply must exist (0-point header)");
    CHECK(fs::exists(res_lidar.pointcloud_stereo_pcd_path), "LiDAR-only pointcloud_stereo.pcd must exist (0-point header)");
    CHECK(fs::exists(res_lidar.pointcloud_fused_rgb_ply_path), "LiDAR-only pointcloud_fused_rgb.ply must exist");
    CHECK(fs::exists(res_lidar.pointcloud_fused_rgb_pcd_path), "LiDAR-only pointcloud_fused_rgb.pcd must exist");
    CHECK(fs::exists(res_lidar.scan_metadata_path), "LiDAR-only scan_metadata.json must exist");
    CHECK(fs::exists(res_lidar.stations_manifest_path), "LiDAR-only stations_manifest.json must exist");

    {
        std::ifstream sm_ifs(res_lidar.scan_metadata_path);
        json sm_lidar = json::parse(sm_ifs);
        CHECK(sm_lidar["is_monochrome_lidar_only"].get<bool>() == true,
              "scan_metadata.json must record is_monochrome_lidar_only = true");
        CHECK(sm_lidar["airy_points"].get<size_t>() == cloud_lidar_only.size(),
              "scan_metadata.json airy_points must match LiDAR point count");
        CHECK(sm_lidar["stereo_points"].get<size_t>() == 0,
              "scan_metadata.json stereo_points must be 0 for LiDAR-only scan");
        CHECK(sm_lidar["fused_points"].get<size_t>() == cloud_lidar_only.size(),
              "scan_metadata.json fused_points must match LiDAR point count");
    }

    {
        std::ifstream lb_ifs(res_lidar.bundle_json_path);
        json bundle_lidar = json::parse(lb_ifs);
        for (const auto& p : bundle_lidar["panos"]) {
            CHECK(p["provenance"] == "GEOMETRY_UNCOLORED",
                  "LiDAR-only panorama provenance must be GEOMETRY_UNCOLORED, got " + p["provenance"].get<std::string>());
            fs::path pano_jpg = lidar_bundle_dir / p["image_path"].get<std::string>();
            std::vector<uint8_t> rgb_px;
            int pw = 0, ph = 0;
            CHECK(StructuredWalkthroughExporter::decodeJpegFileRgb(pano_jpg.string(), rgb_px, pw, ph),
                  "Failed to decode LiDAR-only panorama JPEG: " + pano_jpg.string());
            CHECK(pw == cfg_lidar.pano_width && ph == cfg_lidar.pano_height,
                  "Decoded LiDAR-only panorama dimensions mismatch");

            size_t nonzero_bw_px = 0;
            size_t zero_black_px = 0;
            std::set<uint8_t> distinct_grays;
            const size_t total_px = static_cast<size_t>(pw) * static_cast<size_t>(ph);
            for (size_t i = 0; i < total_px; ++i) {
                uint8_t r = rgb_px[i * 3 + 0];
                uint8_t g = rgb_px[i * 3 + 1];
                uint8_t b = rgb_px[i * 3 + 2];
                CHECK(r == g && g == b,
                      "LiDAR-only panorama pixel must be strictly Black-and-White (R == G == B)!");
                if (r > 0) {
                    ++nonzero_bw_px;
                    distinct_grays.insert(r);
                } else {
                    ++zero_black_px;
                }
            }
            CHECK(nonzero_bw_px > total_px / 10,
                  "LiDAR-only B&W panorama must contain observed grayscale geometry pixels");
            CHECK(zero_black_px > 0,
                  "LiDAR-only B&W panorama must preserve pure black (0,0,0) for unobserved angles");
            CHECK(distinct_grays.size() >= 12,
                  "LiDAR-only B&W panorama must have rich grayscale shading variation, got " +
                  std::to_string(distinct_grays.size()) + " levels");
        }
    }

    // Verify pointcloud_fused_rgb.pcd and cache/lod0.bin are also strictly Black-and-White (R == G == B) with grayscale shading
    {
        std::vector<PointXYZI> lidar_lod0;
        CHECK(StructuredWalkthroughExporter::readLod0Bin(res_lidar.lod0_bin_path, lidar_lod0),
              "Failed to read LiDAR-only lod0.bin");
        std::set<uint8_t> lod0_grays;
        for (const auto& pt : lidar_lod0) {
            CHECK(pt.r == pt.g && pt.g == pt.b, "LiDAR-only lod0.bin point must have R == G == B");
            lod0_grays.insert(pt.r);
        }
        CHECK(lod0_grays.size() >= 10, "LiDAR-only lod0.bin must have grayscale intensity/elevation shading variation");

        std::vector<PointXYZI> lidar_fused_pcd;
        CHECK(PointCloudProcessor::loadPcd(res_lidar.pointcloud_fused_rgb_pcd_path, lidar_fused_pcd),
              "Failed to load LiDAR-only pointcloud_fused_rgb.pcd");
        CHECK(lidar_fused_pcd.size() == cloud_lidar_only.size(),
              "LiDAR-only pointcloud_fused_rgb.pcd point count mismatch");
        for (const auto& pt : lidar_fused_pcd) {
            CHECK(pt.r == pt.g && pt.g == pt.b,
                  "LiDAR-only pointcloud_fused_rgb.pcd point must have R == G == B");
        }
    }
    std::cout << "  [PASS] Gate 4: LiDAR-only standardized outputs & strictly B&W (R=G=B) panoramas/clouds verified.\n";

    fs::remove_all(sandbox, ec);
    std::cout << "\n=====================================================================\n"
              << "  ALL ECOSYSTEM WALKTHROUGH INTEROPERABILITY GATES PASSED (100%)     \n"
              << "=====================================================================\n";
    return 0;
}

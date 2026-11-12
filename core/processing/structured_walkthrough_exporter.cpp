#include "core/processing/structured_walkthrough_exporter.hpp"
#include "core/processing/point_cloud_processor.hpp"
#include "core/logging/logger.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdio>
#include <jpeglib.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace av::core::processing {

namespace fs = std::filesystem;

namespace {

constexpr size_t E57_PHYSICAL_PAGE_SIZE = 1024;
constexpr size_t E57_LOGICAL_PAGE_SIZE = 1020;
constexpr uint8_t E57_BLOB_SECTION_ID = 0;
constexpr uint8_t E57_COMPRESSED_VECTOR_SECTION_ID = 1;
constexpr uint8_t E57_DATA_PACKET_TYPE = 1;
constexpr double kPi = 3.14159265358979323846;

size_t align4(size_t offset) {
    size_t rem = offset % 4;
    return rem == 0 ? offset : (offset + (4 - rem));
}

uint64_t logicalToPhysicalOffset(uint64_t logical_offset) {
    uint64_t page_idx = logical_offset / E57_LOGICAL_PAGE_SIZE;
    uint64_t page_rem = logical_offset % E57_LOGICAL_PAGE_SIZE;
    return page_idx * E57_PHYSICAL_PAGE_SIZE + page_rem;
}

uint64_t physicalToLogicalOffset(uint64_t physical_offset) {
    uint64_t page_idx = physical_offset / E57_PHYSICAL_PAGE_SIZE;
    uint64_t page_rem = physical_offset % E57_PHYSICAL_PAGE_SIZE;
    return page_idx * E57_LOGICAL_PAGE_SIZE + std::min<uint64_t>(page_rem, E57_LOGICAL_PAGE_SIZE - 1);
}

void appendLe16(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFFu));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}

void appendLe32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void appendLe64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void appendBe32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    buf.push_back(static_cast<uint8_t>(v & 0xFFu));
}

std::string makeDeterministicUuid(const std::string& seed_str, bool braced = false) {
    // Deterministic 128-bit FNV-1a style hash formatted as RFC 4122 v5-like UUID
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL ^ 0x9E3779B97F4A7C15ULL;
    for (unsigned char c : seed_str) {
        h1 ^= static_cast<uint64_t>(c);
        h1 *= 1099511628211ULL;
        h2 ^= static_cast<uint64_t>(c) + 0x9E3779B97F4A7C15ULL + (h2 << 6) + (h2 >> 2);
    }
    uint8_t b[16];
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<uint8_t>((h1 >> (8 * i)) & 0xFFu);
        b[8 + i] = static_cast<uint8_t>((h2 >> (8 * i)) & 0xFFu);
    }
    b[6] = static_cast<uint8_t>((b[6] & 0x0Fu) | 0x50u);
    b[8] = static_cast<uint8_t>((b[8] & 0x3Fu) | 0x80u);

    std::ostringstream oss;
    if (braced) oss << "{";
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) oss << "-";
        oss << std::setw(2) << static_cast<int>(b[i]);
    }
    if (braced) oss << "}";
    return oss.str();
}

std::string currentUtcIsoString() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
    gmtime_r(&now, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

std::string currentLocalDateString(bool with_time = false) {
    std::time_t now = std::time(nullptr);
    std::tm tm_loc{};
    localtime_r(&now, &tm_loc);
    char buf[32];
    std::strftime(buf, sizeof(buf), with_time ? "%Y-%m-%d_%H%M" : "%Y-%m-%d", &tm_loc);
    return std::string(buf);
}

std::vector<uint8_t> encodeJpegRgb(const uint8_t* rgb, int width, int height, int quality) {
    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned char* outbuffer = nullptr;
    unsigned long outsize = 0;
    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = static_cast<JDIMENSION>(width);
    cinfo.image_height = static_cast<JDIMENSION>(height);
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, std::clamp(quality, 10, 100), TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_ptr[1];
        row_ptr[0] = const_cast<JSAMPROW>(rgb + static_cast<size_t>(cinfo.next_scanline) * static_cast<size_t>(width) * 3);
        jpeg_write_scanlines(&cinfo, row_ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    std::vector<uint8_t> result;
    if (outbuffer && outsize > 0) {
        result.assign(outbuffer, outbuffer + outsize);
    }
    if (outbuffer) {
        std::free(outbuffer);
    }
    jpeg_destroy_compress(&cinfo);
    return result;
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

bool writePngRgb8(const std::string& path, const uint8_t* rgb, int width, int height) {
    if (width <= 0 || height <= 0 || !rgb) return false;
    std::vector<uint8_t> raw_scanlines;
    raw_scanlines.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * 3));
    for (int r = 0; r < height; ++r) {
        raw_scanlines.push_back(0); // Filter type 0 (None)
        const uint8_t* row = rgb + static_cast<size_t>(r) * static_cast<size_t>(width) * 3;
        raw_scanlines.insert(raw_scanlines.end(), row, row + static_cast<size_t>(width) * 3);
    }
    uLongf comp_bound = compressBound(static_cast<uLong>(raw_scanlines.size()));
    std::vector<uint8_t> zbuf(comp_bound);
    if (compress2(zbuf.data(), &comp_bound, raw_scanlines.data(), static_cast<uLong>(raw_scanlines.size()), 6) != Z_OK) {
        return false;
    }

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    appendBe32(ihdr, static_cast<uint32_t>(width));
    appendBe32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // bit depth 8
    ihdr.push_back(2); // color type 2 (Truecolor RGB)
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    writePngChunk(png, "IHDR", ihdr.data(), ihdr.size());
    writePngChunk(png, "IDAT", zbuf.data(), static_cast<size_t>(comp_bound));
    writePngChunk(png, "IEND", nullptr, 0);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return ofs.good();
}

bool writePngGray16(const std::string& path, const uint16_t* gray16, int width, int height) {
    if (width <= 0 || height <= 0 || !gray16) return false;
    std::vector<uint8_t> raw_scanlines;
    raw_scanlines.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * 2));
    for (int r = 0; r < height; ++r) {
        raw_scanlines.push_back(0);
        const uint16_t* row = gray16 + static_cast<size_t>(r) * static_cast<size_t>(width);
        for (int c = 0; c < width; ++c) {
            uint16_t v = row[c];
            raw_scanlines.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu)); // PNG 16-bit is big-endian
            raw_scanlines.push_back(static_cast<uint8_t>(v & 0xFFu));
        }
    }
    uLongf comp_bound = compressBound(static_cast<uLong>(raw_scanlines.size()));
    std::vector<uint8_t> zbuf(comp_bound);
    if (compress2(zbuf.data(), &comp_bound, raw_scanlines.data(), static_cast<uLong>(raw_scanlines.size()), 6) != Z_OK) {
        return false;
    }

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    appendBe32(ihdr, static_cast<uint32_t>(width));
    appendBe32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(16); // bit depth 16
    ihdr.push_back(0);  // color type 0 (Grayscale)
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    writePngChunk(png, "IHDR", ihdr.data(), ihdr.size());
    writePngChunk(png, "IDAT", zbuf.data(), static_cast<size_t>(comp_bound));
    writePngChunk(png, "IEND", nullptr, 0);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return ofs.good();
}

bool writePngGray8(const std::string& path, const uint8_t* gray8, int width, int height) {
    if (width <= 0 || height <= 0 || !gray8) return false;
    std::vector<uint8_t> raw_scanlines;
    raw_scanlines.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width)));
    for (int r = 0; r < height; ++r) {
        raw_scanlines.push_back(0); // Filter type 0 (None)
        const uint8_t* row = gray8 + static_cast<size_t>(r) * static_cast<size_t>(width);
        raw_scanlines.insert(raw_scanlines.end(), row, row + static_cast<size_t>(width));
    }
    uLongf comp_bound = compressBound(static_cast<uLong>(raw_scanlines.size()));
    std::vector<uint8_t> zbuf(comp_bound);
    if (compress2(zbuf.data(), &comp_bound, raw_scanlines.data(), static_cast<uLong>(raw_scanlines.size()), 6) != Z_OK) {
        return false;
    }

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    appendBe32(ihdr, static_cast<uint32_t>(width));
    appendBe32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // bit depth 8
    ihdr.push_back(0); // color type 0 (Grayscale)
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    writePngChunk(png, "IHDR", ihdr.data(), ihdr.size());
    writePngChunk(png, "IDAT", zbuf.data(), static_cast<size_t>(comp_bound));
    writePngChunk(png, "IEND", nullptr, 0);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return ofs.good();
}

bool writeStandardizedPlyAscii(
    const std::string& filepath,
    const std::vector<schemas::PointXYZI>& points,
    bool with_rgb) {
    fs::create_directories(fs::path(filepath).parent_path());
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;

    ofs << "ply\n"
        << "format ascii 1.0\n"
        << "comment Generated by AV Scan Ecosystem Engine\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n";
    if (with_rgb) {
        ofs << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n";
    }
    ofs << "end_header\n";

    ofs << std::fixed << std::setprecision(4);
    for (const auto& pt : points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
        if (with_rgb) {
            ofs << pt.x << " " << pt.y << " " << pt.z << " "
                << static_cast<int>(pt.r) << " "
                << static_cast<int>(pt.g) << " "
                << static_cast<int>(pt.b) << "\n";
        } else {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }
    }
    return ofs.good();
}

bool writeStandardizedPcdAscii(
    const std::string& filepath,
    const std::vector<schemas::PointXYZI>& points,
    bool with_rgb) {
    fs::create_directories(fs::path(filepath).parent_path());
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;

    ofs << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n";
    if (with_rgb) {
        ofs << "FIELDS x y z rgb\n"
            << "SIZE 4 4 4 4\n"
            << "TYPE F F F U\n"
            << "COUNT 1 1 1 1\n";
    } else {
        ofs << "FIELDS x y z\n"
            << "SIZE 4 4 4\n"
            << "TYPE F F F\n"
            << "COUNT 1 1 1\n";
    }
    ofs << "WIDTH " << points.size() << "\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << points.size() << "\n"
        << "DATA ascii\n";

    ofs << std::fixed << std::setprecision(4);
    for (const auto& pt : points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
        if (with_rgb) {
            uint32_t packed_rgb = (static_cast<uint32_t>(pt.r) << 16) |
                                  (static_cast<uint32_t>(pt.g) << 8) |
                                  static_cast<uint32_t>(pt.b);
            ofs << pt.x << " " << pt.y << " " << pt.z << " " << packed_rgb << "\n";
        } else {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }
    }
    return ofs.good();
}

struct E57StationData {
    std::string station_id;
    std::string pano_id;
    std::string description;
    std::string selection_mode{"auto_trajectory"};
    std::string scan_guid;
    std::string image_guid;
    Eigen::Vector3d pos_world{0.0, 0.0, 0.0};
    double yaw_rad{0.0};
    std::array<double, 4> quat_wxyz{1.0, 0.0, 0.0, 0.0};
    std::array<double, 4> quat_xyzw{0.0, 0.0, 0.0, 1.0};
    int scan_rows{0};
    int scan_cols{0};
    std::vector<float> scan_x;
    std::vector<float> scan_y;
    std::vector<float> scan_z;
    std::vector<uint8_t> scan_invalid;
    std::vector<uint8_t> scan_r;
    std::vector<uint8_t> scan_g;
    std::vector<uint8_t> scan_b;
    size_t valid_scan_points{0};
    int pano_w{0};
    int pano_h{0};
    std::vector<uint8_t> pano_rgb;
    std::vector<float> pano_depth_m;
    std::vector<uint8_t> pano_prov_code;
    std::vector<uint8_t> pano_jpeg;
    double valid_pixel_ratio{0.0};
    double photo_rgb_ratio{0.0};
    double pointcloud_rgb_ratio{0.0};
    double depth_fill_ratio{0.0};
    double geometry_uncolored_ratio{0.0};
    double unobserved_ratio{1.0};
    std::string provenance{"SYNTHETIC_FROM_CLOUD"};
};

bool writeBinaryPlyStructuredScan(const std::string& filepath, const E57StationData& st) {
    fs::create_directories(fs::path(filepath).parent_path());
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) return false;

    const size_t total_cells = static_cast<size_t>(st.scan_rows) * static_cast<size_t>(st.scan_cols);
    size_t valid_cnt = 0;
    for (size_t i = 0; i < total_cells; ++i) {
        if (st.scan_invalid[i] == 0) ++valid_cnt;
    }

    std::ostringstream hdr;
    hdr << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "comment Artificial Vision AV Structure Structured Station Scan\n"
        << "comment station_id " << st.station_id << "\n"
        << "comment rows " << st.scan_rows << " cols " << st.scan_cols << "\n"
        << "element vertex " << valid_cnt << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "property ushort row_index\n"
        << "property ushort col_index\n"
        << "end_header\n";
    std::string hdr_str = hdr.str();
    ofs.write(hdr_str.data(), static_cast<std::streamsize>(hdr_str.size()));

    Eigen::Matrix3d R_ws;
    R_ws << std::cos(st.yaw_rad), -std::sin(st.yaw_rad), 0.0,
            std::sin(st.yaw_rad),  std::cos(st.yaw_rad), 0.0,
            0.0,                   0.0,                  1.0;

    uint8_t rec[19];
    for (size_t idx = 0; idx < total_cells; ++idx) {
        if (st.scan_invalid[idx] != 0) continue;
        Eigen::Vector3d p_loc(st.scan_x[idx], st.scan_y[idx], st.scan_z[idx]);
        Eigen::Vector3d p_w = R_ws * p_loc + st.pos_world;
        float wx = static_cast<float>(p_w.x());
        float wy = static_cast<float>(p_w.y());
        float wz = static_cast<float>(p_w.z());
        uint16_t r_idx = static_cast<uint16_t>(idx / static_cast<size_t>(st.scan_cols));
        uint16_t c_idx = static_cast<uint16_t>(idx % static_cast<size_t>(st.scan_cols));
        std::memcpy(rec + 0, &wx, 4);
        std::memcpy(rec + 4, &wy, 4);
        std::memcpy(rec + 8, &wz, 4);
        rec[12] = st.scan_r[idx];
        rec[13] = st.scan_g[idx];
        rec[14] = st.scan_b[idx];
        std::memcpy(rec + 15, &r_idx, 2);
        std::memcpy(rec + 17, &c_idx, 2);
        ofs.write(reinterpret_cast<const char*>(rec), 19);
    }
    return ofs.good();
}

bool writeStructuredE57File(
    const std::string& filepath,
    const std::vector<E57StationData>& stations,
    uint64_t& out_file_bytes) {
    std::vector<uint8_t> logical_stream(48, 0); // 48-byte E57FileHeader placeholder

    std::ostringstream data3d_xml;
    std::ostringstream images2d_xml;

    for (const auto& st : stations) {
        const size_t record_count = static_cast<size_t>(st.scan_rows) * static_cast<size_t>(st.scan_cols);
        float xmin = 0.0f, ymin = 0.0f, zmin = 0.0f;
        float xmax = 0.0f, ymax = 0.0f, zmax = 0.0f;
        bool first_valid = true;
        for (size_t i = 0; i < record_count; ++i) {
            if (st.scan_invalid[i] == 0) {
                if (first_valid) {
                    xmin = xmax = st.scan_x[i];
                    ymin = ymax = st.scan_y[i];
                    zmin = zmax = st.scan_z[i];
                    first_valid = false;
                } else {
                    xmin = std::min(xmin, st.scan_x[i]);
                    xmax = std::max(xmax, st.scan_x[i]);
                    ymin = std::min(ymin, st.scan_y[i]);
                    ymax = std::max(ymax, st.scan_y[i]);
                    zmin = std::min(zmin, st.scan_z[i]);
                    zmax = std::max(zmax, st.scan_z[i]);
                }
            }
        }

        while (logical_stream.size() % 4 != 0) logical_stream.push_back(0);
        const uint64_t cv_logical_offset = logical_stream.size();
        const uint64_t cv_physical_offset = logicalToPhysicalOffset(cv_logical_offset);
        const uint64_t data_physical_offset = logicalToPhysicalOffset(cv_logical_offset + 32);

        std::vector<uint8_t> packets;
        constexpr size_t chunk_records = 2500;
        constexpr uint16_t num_streams = 9;

        for (size_t start = 0; start < std::max<size_t>(1, record_count); start += chunk_records) {
            size_t end = std::min(record_count, start + chunk_records);
            size_t cnt = (record_count == 0) ? 0 : (end - start);

            uint16_t b_lens[num_streams] = {
                static_cast<uint16_t>(cnt * 4),
                static_cast<uint16_t>(cnt * 4),
                static_cast<uint16_t>(cnt * 4),
                static_cast<uint16_t>(cnt * 1),
                static_cast<uint16_t>(cnt * 2),
                static_cast<uint16_t>(cnt * 2),
                static_cast<uint16_t>(cnt * 1),
                static_cast<uint16_t>(cnt * 1),
                static_cast<uint16_t>(cnt * 1)
            };
            size_t raw_pkt_len = 6 + 2 * num_streams;
            for (uint16_t bl : b_lens) raw_pkt_len += bl;
            size_t padded_pkt_len = align4(raw_pkt_len);

            packets.push_back(E57_DATA_PACKET_TYPE);
            packets.push_back(0);
            appendLe16(packets, static_cast<uint16_t>(padded_pkt_len - 1));
            appendLe16(packets, num_streams);
            for (uint16_t bl : b_lens) appendLe16(packets, bl);

            if (cnt > 0) {
                auto appendFloats = [&](const std::vector<float>& arr) {
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(arr.data() + start);
                    packets.insert(packets.end(), p, p + cnt * sizeof(float));
                };
                appendFloats(st.scan_x);
                appendFloats(st.scan_y);
                appendFloats(st.scan_z);
                packets.insert(packets.end(), st.scan_invalid.begin() + static_cast<ptrdiff_t>(start),
                               st.scan_invalid.begin() + static_cast<ptrdiff_t>(end));
                for (size_t idx = start; idx < end; ++idx) {
                    uint16_t r_i = static_cast<uint16_t>(idx / static_cast<size_t>(st.scan_cols));
                    appendLe16(packets, r_i);
                }
                for (size_t idx = start; idx < end; ++idx) {
                    uint16_t c_i = static_cast<uint16_t>(idx % static_cast<size_t>(st.scan_cols));
                    appendLe16(packets, c_i);
                }
                packets.insert(packets.end(), st.scan_r.begin() + static_cast<ptrdiff_t>(start),
                               st.scan_r.begin() + static_cast<ptrdiff_t>(end));
                packets.insert(packets.end(), st.scan_g.begin() + static_cast<ptrdiff_t>(start),
                               st.scan_g.begin() + static_cast<ptrdiff_t>(end));
                packets.insert(packets.end(), st.scan_b.begin() + static_cast<ptrdiff_t>(start),
                               st.scan_b.begin() + static_cast<ptrdiff_t>(end));
            }
            while (packets.size() % 4 != 0) packets.push_back(0);
        }

        uint64_t sec_logical_len = align4(32 + packets.size());
        logical_stream.push_back(E57_COMPRESSED_VECTOR_SECTION_ID);
        for (int i = 0; i < 7; ++i) logical_stream.push_back(0);
        appendLe64(logical_stream, sec_logical_len);
        appendLe64(logical_stream, data_physical_offset);
        appendLe64(logical_stream, 0);
        logical_stream.insert(logical_stream.end(), packets.begin(), packets.end());
        while (logical_stream.size() < cv_logical_offset + sec_logical_len) logical_stream.push_back(0);

        data3d_xml << std::fixed << std::setprecision(6)
                   << "    <vectorChild type=\"Structure\">\n"
                   << "      <guid type=\"String\"><![CDATA[" << st.scan_guid << "]]></guid>\n"
                   << "      <name type=\"String\"><![CDATA[" << st.pano_id << "]]></name>\n"
                   << "      <description type=\"String\"><![CDATA[" << st.description << "]]></description>\n"
                   << "      <sensorVendor type=\"String\"><![CDATA[Artificial Vision]]></sensorVendor>\n"
                   << "      <sensorModel type=\"String\"><![CDATA[AV Scan Virtual Station]]></sensorModel>\n"
                   << "      <indexBounds type=\"Structure\">\n"
                   << "        <rowMinimum type=\"Integer\">0</rowMinimum>\n"
                   << "        <rowMaximum type=\"Integer\">" << std::max(0, st.scan_rows - 1) << "</rowMaximum>\n"
                   << "        <columnMinimum type=\"Integer\">0</columnMinimum>\n"
                   << "        <columnMaximum type=\"Integer\">" << std::max(0, st.scan_cols - 1) << "</columnMaximum>\n"
                   << "        <returnMinimum type=\"Integer\">0</returnMinimum>\n"
                   << "        <returnMaximum type=\"Integer\">0</returnMaximum>\n"
                   << "      </indexBounds>\n"
                   << "      <cartesianBounds type=\"Structure\">\n"
                   << "        <xMinimum type=\"Float\">" << xmin << "</xMinimum>\n"
                   << "        <xMaximum type=\"Float\">" << xmax << "</xMaximum>\n"
                   << "        <yMinimum type=\"Float\">" << ymin << "</yMinimum>\n"
                   << "        <yMaximum type=\"Float\">" << ymax << "</yMaximum>\n"
                   << "        <zMinimum type=\"Float\">" << zmin << "</zMinimum>\n"
                   << "        <zMaximum type=\"Float\">" << zmax << "</zMaximum>\n"
                   << "      </cartesianBounds>\n"
                   << "      <colorLimits type=\"Structure\">\n"
                   << "        <colorRedMinimum type=\"Integer\">0</colorRedMinimum>\n"
                   << "        <colorRedMaximum type=\"Integer\">255</colorRedMaximum>\n"
                   << "        <colorGreenMinimum type=\"Integer\">0</colorGreenMinimum>\n"
                   << "        <colorGreenMaximum type=\"Integer\">255</colorGreenMaximum>\n"
                   << "        <colorBlueMinimum type=\"Integer\">0</colorBlueMinimum>\n"
                   << "        <colorBlueMaximum type=\"Integer\">255</colorBlueMaximum>\n"
                   << "      </colorLimits>\n"
                   << "      <pose type=\"Structure\">\n"
                   << "        <rotation type=\"Structure\">\n"
                   << std::setprecision(8)
                   << "          <w type=\"Float\">" << st.quat_wxyz[0] << "</w>\n"
                   << "          <x type=\"Float\">" << st.quat_wxyz[1] << "</x>\n"
                   << "          <y type=\"Float\">" << st.quat_wxyz[2] << "</y>\n"
                   << "          <z type=\"Float\">" << st.quat_wxyz[3] << "</z>\n"
                   << "        </rotation>\n"
                   << std::setprecision(6)
                   << "        <translation type=\"Structure\">\n"
                   << "          <x type=\"Float\">" << st.pos_world.x() << "</x>\n"
                   << "          <y type=\"Float\">" << st.pos_world.y() << "</y>\n"
                   << "          <z type=\"Float\">" << st.pos_world.z() << "</z>\n"
                   << "        </translation>\n"
                   << "      </pose>\n"
                   << "      <points type=\"CompressedVector\" fileOffset=\"" << cv_physical_offset
                   << "\" recordCount=\"" << record_count << "\">\n"
                   << "        <prototype type=\"Structure\">\n"
                   << "          <cartesianX type=\"Float\" precision=\"single\"/>\n"
                   << "          <cartesianY type=\"Float\" precision=\"single\"/>\n"
                   << "          <cartesianZ type=\"Float\" precision=\"single\"/>\n"
                   << "          <cartesianInvalidState type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
                   << "          <rowIndex type=\"Integer\" minimum=\"0\" maximum=\"65535\"/>\n"
                   << "          <columnIndex type=\"Integer\" minimum=\"0\" maximum=\"65535\"/>\n"
                   << "          <colorRed type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
                   << "          <colorGreen type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
                   << "          <colorBlue type=\"Integer\" minimum=\"0\" maximum=\"255\"/>\n"
                   << "        </prototype>\n"
                   << "        <codecs type=\"Vector\" allowHeterogeneousChildren=\"1\">\n"
                   << "        </codecs>\n"
                   << "      </points>\n"
                   << "    </vectorChild>\n";

        if (!st.pano_jpeg.empty() && st.pano_w > 0 && st.pano_h > 0) {
            while (logical_stream.size() % 4 != 0) logical_stream.push_back(0);
            uint64_t blob_logical_offset = logical_stream.size();
            uint64_t blob_physical_offset = logicalToPhysicalOffset(blob_logical_offset);
            uint64_t blob_sec_len = align4(16 + st.pano_jpeg.size());

            logical_stream.push_back(E57_BLOB_SECTION_ID);
            for (int i = 0; i < 7; ++i) logical_stream.push_back(0);
            appendLe64(logical_stream, blob_sec_len);
            logical_stream.insert(logical_stream.end(), st.pano_jpeg.begin(), st.pano_jpeg.end());
            while (logical_stream.size() < blob_logical_offset + blob_sec_len) logical_stream.push_back(0);

            double px_w = (2.0 * kPi) / static_cast<double>(st.pano_w);
            double px_h = kPi / static_cast<double>(st.pano_h);

            images2d_xml << std::fixed
                         << "    <vectorChild type=\"Structure\">\n"
                         << "      <guid type=\"String\"><![CDATA[" << st.image_guid << "]]></guid>\n"
                         << "      <name type=\"String\"><![CDATA[" << st.pano_id << "_panorama]]></name>\n"
                         << "      <description type=\"String\"><![CDATA[360 Equirectangular Panorama associated with " << st.pano_id << "]]></description>\n"
                         << "      <associatedData3DGuid type=\"String\"><![CDATA[" << st.scan_guid << "]]></associatedData3DGuid>\n"
                         << "      <pose type=\"Structure\">\n"
                         << "        <rotation type=\"Structure\">\n"
                         << std::setprecision(8)
                         << "          <w type=\"Float\">" << st.quat_wxyz[0] << "</w>\n"
                         << "          <x type=\"Float\">" << st.quat_wxyz[1] << "</x>\n"
                         << "          <y type=\"Float\">" << st.quat_wxyz[2] << "</y>\n"
                         << "          <z type=\"Float\">" << st.quat_wxyz[3] << "</z>\n"
                         << "        </rotation>\n"
                         << std::setprecision(6)
                         << "        <translation type=\"Structure\">\n"
                         << "          <x type=\"Float\">" << st.pos_world.x() << "</x>\n"
                         << "          <y type=\"Float\">" << st.pos_world.y() << "</y>\n"
                         << "          <z type=\"Float\">" << st.pos_world.z() << "</z>\n"
                         << "        </translation>\n"
                         << "      </pose>\n"
                         << "      <sphericalRepresentation type=\"Structure\">\n"
                         << "        <jpegImage type=\"Blob\" fileOffset=\"" << blob_physical_offset
                         << "\" length=\"" << st.pano_jpeg.size() << "\"/>\n"
                         << "        <imageHeight type=\"Integer\">" << st.pano_h << "</imageHeight>\n"
                         << "        <imageWidth type=\"Integer\">" << st.pano_w << "</imageWidth>\n"
                         << std::setprecision(10)
                         << "        <pixelHeight type=\"Float\">" << px_h << "</pixelHeight>\n"
                         << "        <pixelWidth type=\"Float\">" << px_w << "</pixelWidth>\n"
                         << "      </sphericalRepresentation>\n"
                         << "    </vectorChild>\n";
        }
    }

    while (logical_stream.size() % 4 != 0) logical_stream.push_back(0);
    std::string root_guid = makeDeterministicUuid("av-scan-e57-root-" + filepath, true);
    double gps_epoch = static_cast<double>(std::time(nullptr)) - 315964800.0;

    std::ostringstream xml_ss;
    xml_ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           << "<e57Root type=\"Structure\" xmlns=\"http://www.astm.org/COMMIT/E57/2010-e57-v1.0\">\n"
           << "  <formatName type=\"String\"><![CDATA[ASTM E57 3D Imaging Data File]]></formatName>\n"
           << "  <guid type=\"String\"><![CDATA[" << root_guid << "]]></guid>\n"
           << "  <versionMajor type=\"Integer\">1</versionMajor>\n"
           << "  <versionMinor type=\"Integer\">0</versionMinor>\n"
           << "  <e57LibraryVersion type=\"String\"><![CDATA[ArtificialVision-AVScan-1.1]]></e57LibraryVersion>\n"
           << "  <coordinateMetadata type=\"String\"><![CDATA[Gravity-Aligned Right-Handed Metric World Frame (meters)]]></coordinateMetadata>\n"
           << "  <creationDateTime type=\"Structure\">\n"
           << "    <dateTimeValue type=\"Float\">" << std::fixed << std::setprecision(3) << gps_epoch << "</dateTimeValue>\n"
           << "    <isAtomicClockReferenced type=\"Integer\">0</isAtomicClockReferenced>\n"
           << "  </creationDateTime>\n"
           << "  <data3D type=\"Vector\" allowHeterogeneousChildren=\"1\">\n"
           << data3d_xml.str()
           << "  </data3D>\n"
           << "  <images2D type=\"Vector\" allowHeterogeneousChildren=\"1\">\n"
           << images2d_xml.str()
           << "  </images2D>\n"
           << "</e57Root>\n";

    std::string xml_str = xml_ss.str();
    uint64_t xml_logical_offset = logical_stream.size();
    uint64_t xml_physical_offset = logicalToPhysicalOffset(xml_logical_offset);
    uint64_t xml_logical_length = xml_str.size();
    logical_stream.insert(logical_stream.end(), xml_str.begin(), xml_str.end());

    size_t rem_page = logical_stream.size() % E57_LOGICAL_PAGE_SIZE;
    if (rem_page != 0) {
        logical_stream.resize(logical_stream.size() + (E57_LOGICAL_PAGE_SIZE - rem_page), 0);
    }

    size_t num_pages = logical_stream.size() / E57_LOGICAL_PAGE_SIZE;
    uint64_t file_physical_len = static_cast<uint64_t>(num_pages) * E57_PHYSICAL_PAGE_SIZE;

    // Populate 48-byte E57FileHeader at logical_stream[0..47]
    std::vector<uint8_t> hdr;
    hdr.insert(hdr.end(), {'A', 'S', 'T', 'M', '-', 'E', '5', '7'});
    appendLe32(hdr, 1);
    appendLe32(hdr, 0);
    appendLe64(hdr, file_physical_len);
    appendLe64(hdr, xml_physical_offset);
    appendLe64(hdr, xml_logical_length);
    appendLe32(hdr, static_cast<uint32_t>(E57_PHYSICAL_PAGE_SIZE));
    appendLe32(hdr, 0);
    std::memcpy(logical_stream.data(), hdr.data(), 48);

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) return false;

    uint8_t page_buf[E57_PHYSICAL_PAGE_SIZE];
    for (size_t p = 0; p < num_pages; ++p) {
        const uint8_t* src = logical_stream.data() + p * E57_LOGICAL_PAGE_SIZE;
        std::memcpy(page_buf, src, E57_LOGICAL_PAGE_SIZE);
        uint32_t crc = StructuredWalkthroughExporter::computeCrc32c(src, E57_LOGICAL_PAGE_SIZE);
        page_buf[1020] = static_cast<uint8_t>((crc >> 24) & 0xFFu);
        page_buf[1021] = static_cast<uint8_t>((crc >> 16) & 0xFFu);
        page_buf[1022] = static_cast<uint8_t>((crc >> 8) & 0xFFu);
        page_buf[1023] = static_cast<uint8_t>(crc & 0xFFu);
        ofs.write(reinterpret_cast<const char*>(page_buf), E57_PHYSICAL_PAGE_SIZE);
    }
    ofs.close();
    out_file_bytes = file_physical_len;
    return true;
}

Eigen::Matrix4d arrayToMatrix4d(const std::array<double, 16>& a) {
    Eigen::Matrix4d M;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            M(r, c) = a[static_cast<size_t>(r * 4 + c)];
    return M;
}

std::array<double, 16> matrix4dToArray(const Eigen::Matrix4d& M) {
    std::array<double, 16> a{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            a[static_cast<size_t>(r * 4 + c)] = M(r, c);
    return a;
}

void transformPointCloudInPlace(std::vector<schemas::PointXYZI>& pts, const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    for (auto& p : pts) {
        Eigen::Vector3d v = R * Eigen::Vector3d(p.x, p.y, p.z) + t;
        p.x = static_cast<float>(v.x());
        p.y = static_cast<float>(v.y());
        p.z = static_cast<float>(v.z());
    }
}

struct VoxelCoord {
    int64_t x, y, z;
    bool operator==(const VoxelCoord& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelCoordHash {
    size_t operator()(const VoxelCoord& k) const noexcept {
        size_t h1 = std::hash<int64_t>{}(k.x);
        size_t h2 = std::hash<int64_t>{}(k.y);
        size_t h3 = std::hash<int64_t>{}(k.z);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2)) ^ (h3 * 0xc4ceb9fe1a85ec53ULL);
    }
};

std::vector<Eigen::Vector3d> voxelSubsampleEigen(const std::vector<schemas::PointXYZI>& pts, double voxel_m) {
    std::unordered_set<VoxelCoord, VoxelCoordHash> seen;
    std::vector<Eigen::Vector3d> out;
    out.reserve(std::min<size_t>(pts.size(), 80000));
    double inv = 1.0 / std::max(0.01, voxel_m);
    for (const auto& p : pts) {
        VoxelCoord vc{
            static_cast<int64_t>(std::floor(static_cast<double>(p.x) * inv)),
            static_cast<int64_t>(std::floor(static_cast<double>(p.y) * inv)),
            static_cast<int64_t>(std::floor(static_cast<double>(p.z) * inv))
        };
        if (seen.insert(vc).second) {
            out.emplace_back(p.x, p.y, p.z);
        }
    }
    return out;
}

nlohmann::json renderFloorKeyplanPng(
    const std::vector<schemas::PointXYZI>& points,
    double floor_station_mean_z,
    const std::string& out_png_path,
    const std::vector<Eigen::Vector2d>& station_xys,
    double meters_per_pixel = 0.02,
    int max_dim_px = 2048) {
    if (points.empty() && station_xys.empty()) return nlohmann::json();

    std::vector<double> slab_cand_z;
    slab_cand_z.reserve(points.size() / 4 + 16);
    for (const auto& p : points) {
        if (p.z >= floor_station_mean_z - 2.35 && p.z <= floor_station_mean_z - 0.45) {
            slab_cand_z.push_back(p.z);
        }
    }
    double slab_z = floor_station_mean_z - 1.55;
    if (slab_cand_z.size() >= 30) {
        std::sort(slab_cand_z.begin(), slab_cand_z.end());
        slab_z = slab_cand_z[slab_cand_z.size() * 15 / 100];
    }

    double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
    bool any_xy = false;
    for (const auto& p : points) {
        if (p.z >= slab_z - 0.25 && p.z <= slab_z + 2.40) {
            min_x = std::min<double>(min_x, p.x);
            max_x = std::max<double>(max_x, p.x);
            min_y = std::min<double>(min_y, p.y);
            max_y = std::max<double>(max_y, p.y);
            any_xy = true;
        }
    }
    for (const auto& xy : station_xys) {
        min_x = std::min(min_x, xy.x());
        max_x = std::max(max_x, xy.x());
        min_y = std::min(min_y, xy.y());
        max_y = std::max(max_y, xy.y());
        any_xy = true;
    }
    if (!any_xy) return nlohmann::json();

    min_x -= 0.80;
    max_x += 0.80;
    min_y -= 0.80;
    max_y += 0.80;

    double span_x = std::max(2.0, max_x - min_x);
    double span_y = std::max(2.0, max_y - min_y);
    double mpp = std::max(meters_per_pixel, std::max(span_x, span_y) / static_cast<double>(max_dim_px));

    int w = std::max(64, static_cast<int>(std::ceil(span_x / mpp)));
    int h = std::max(64, static_cast<int>(std::ceil(span_y / mpp)));
    max_x = min_x + w * mpp;
    min_y = max_y - h * mpp;

    // Charcoal #171717 background (RGB: 23, 23, 23)
    std::vector<uint8_t> canvas(static_cast<size_t>(w) * static_cast<size_t>(h) * 3, 23);
    std::vector<uint8_t> slab_mask(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    std::vector<float> wall_hist(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);

    for (const auto& p : points) {
        if (p.z < slab_z - 0.25 || p.z > slab_z + 2.40) continue;
        int col = std::clamp(static_cast<int>((p.x - min_x) / mpp), 0, w - 1);
        int row = std::clamp(static_cast<int>((max_y - p.y) / mpp), 0, h - 1); // +Y North is UP (row 0)
        size_t idx = static_cast<size_t>(row) * static_cast<size_t>(w) + static_cast<size_t>(col);
        if (p.z >= slab_z - 0.22 && p.z <= slab_z + 0.25) {
            slab_mask[idx] = 1;
        }
        if (p.z >= slab_z + 0.35 && p.z <= slab_z + 2.10) {
            wall_hist[idx] += 1.0f;
        }
    }

    // Simple 3x3 dilation on floor slab interior -> #22262e (RGB: 34, 38, 46)
    for (int r = 1; r < h - 1; ++r) {
        for (int c = 1; c < w - 1; ++c) {
            bool hit = false;
            for (int dy = -1; dy <= 1 && !hit; ++dy)
                for (int dx = -1; dx <= 1 && !hit; ++dx)
                    if (slab_mask[static_cast<size_t>(r + dy) * static_cast<size_t>(w) + static_cast<size_t>(c + dx)]) hit = true;
            if (hit) {
                size_t p_i = (static_cast<size_t>(r) * static_cast<size_t>(w) + static_cast<size_t>(c)) * 3;
                canvas[p_i + 0] = 34;
                canvas[p_i + 1] = 38;
                canvas[p_i + 2] = 46;
            }
        }
    }

    std::vector<float> nonzero;
    nonzero.reserve(4096);
    for (float v : wall_hist) if (v > 0.0f) nonzero.push_back(v);
    float p90 = 1.0f;
    if (!nonzero.empty()) {
        std::sort(nonzero.begin(), nonzero.end());
        p90 = std::max(1.0f, nonzero[nonzero.size() * 9 / 10]);
    }

    // 2x2 crisp wall slice rendering -> #e5e7eb (RGB: 229, 231, 235)
    for (int r = 0; r < h - 1; ++r) {
        for (int c = 0; c < w - 1; ++c) {
            float mv = std::max({
                wall_hist[static_cast<size_t>(r) * static_cast<size_t>(w) + static_cast<size_t>(c)],
                wall_hist[static_cast<size_t>(r + 1) * static_cast<size_t>(w) + static_cast<size_t>(c)],
                wall_hist[static_cast<size_t>(r) * static_cast<size_t>(w) + static_cast<size_t>(c + 1)],
                wall_hist[static_cast<size_t>(r + 1) * static_cast<size_t>(w) + static_cast<size_t>(c + 1)]
            });
            float norm_w = std::clamp(mv / p90, 0.0f, 1.0f);
            if (norm_w > 0.08f) {
                size_t p_i = (static_cast<size_t>(r) * static_cast<size_t>(w) + static_cast<size_t>(c)) * 3;
                float alpha = std::clamp(0.45f + 0.55f * norm_w, 0.0f, 1.0f);
                canvas[p_i + 0] = static_cast<uint8_t>(canvas[p_i + 0] * (1.0f - alpha) + 229.0f * alpha);
                canvas[p_i + 1] = static_cast<uint8_t>(canvas[p_i + 1] * (1.0f - alpha) + 231.0f * alpha);
                canvas[p_i + 2] = static_cast<uint8_t>(canvas[p_i + 2] * (1.0f - alpha) + 235.0f * alpha);
            }
        }
    }

    fs::create_directories(fs::path(out_png_path).parent_path());
    writePngRgb8(out_png_path, canvas.data(), w, h);

    return {
        {"min_x", std::round(min_x * 10000.0) / 10000.0},
        {"max_x", std::round(max_x * 10000.0) / 10000.0},
        {"min_y", std::round(min_y * 10000.0) / 10000.0},
        {"max_y", std::round(max_y * 10000.0) / 10000.0},
        {"meters_per_pixel", std::round(mpp * 1000000.0) / 1000000.0},
        {"width_px", w},
        {"height_px", h},
        {"slab_z_m", std::round(slab_z * 10000.0) / 10000.0}
    };
}

} // namespace

uint32_t StructuredWalkthroughExporter::computeCrc32c(const uint8_t* data, size_t length) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        constexpr uint32_t poly = 0x82F63B78u;
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int b = 0; b < 8; ++b) {
                crc = (crc & 1u) ? ((crc >> 1) ^ poly) : (crc >> 1);
            }
            table[i] = crc;
        }
        initialized = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFFu];
    }
    return crc ^ 0xFFFFFFFFu;
}

bool StructuredWalkthroughExporter::decodeJpegFileRgb(
    const std::string& jpeg_path,
    std::vector<uint8_t>& out_rgb,
    int& out_width,
    int& out_height) {
    out_rgb.clear();
    out_width = 0;
    out_height = 0;

    std::FILE* infile = std::fopen(jpeg_path.c_str(), "rb");
    if (!infile) return false;

    jpeg_decompress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    out_width = static_cast<int>(cinfo.output_width);
    out_height = static_cast<int>(cinfo.output_height);
    const int channels = static_cast<int>(cinfo.output_components);
    if (out_width <= 0 || out_height <= 0 || channels != 3) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);
        return false;
    }

    const size_t row_stride = static_cast<size_t>(out_width) * 3;
    out_rgb.resize(static_cast<size_t>(out_height) * row_stride);

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_ptr[1];
        row_ptr[0] = reinterpret_cast<JSAMPROW>(
            out_rgb.data() + static_cast<size_t>(cinfo.output_scanline) * row_stride);
        jpeg_read_scanlines(&cinfo, row_ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(infile);
    return true;
}

bool StructuredWalkthroughExporter::populateMonochromeGrayscaleForUncoloredPoints(
    std::vector<schemas::PointXYZI>& points,
    std::vector<bool>* out_point_has_true_rgb) {
    if (out_point_has_true_rgb) {
        out_point_has_true_rgb->assign(points.size(), false);
    }
    if (points.empty()) return true;

    // Detect if the cloud has true RGB colors vs uncolored/placeholder uniform values
    size_t colored_flag_cnt = 0;
    size_t non_mono_cnt = 0;
    uint8_t first_r = points[0].r, first_g = points[0].g, first_b = points[0].b;
    bool all_identical_rgb = true;

    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    float min_inten = std::numeric_limits<float>::max();
    float max_inten = std::numeric_limits<float>::lowest();

    for (const auto& p : points) {
        if (p.has_color) ++colored_flag_cnt;
        if (p.r != p.g || p.g != p.b) ++non_mono_cnt;
        if (p.r != first_r || p.g != first_g || p.b != first_b) all_identical_rgb = false;
        if (std::isfinite(p.z)) {
            min_z = std::min(min_z, p.z);
            max_z = std::max(max_z, p.z);
        }
        if (std::isfinite(p.intensity)) {
            min_inten = std::min(min_inten, p.intensity);
            max_inten = std::max(max_inten, p.intensity);
        }
    }

    const double z_span = std::max(0.25, static_cast<double>(max_z - min_z));
    const double i_span = static_cast<double>(max_inten - min_inten);
    const bool has_varying_inten = (i_span > 1e-3);

    // If all points have identical default gray/white/black RGB (e.g. 255,255,255 or 175,175,175 or 0,0,0),
    // treat as uncolored geometry even if has_color was set by a generic loader.
    const bool uniform_placeholder = all_identical_rgb && (first_r == first_g && first_g == first_b);
    size_t true_rgb_count = 0;

    for (size_t idx = 0; idx < points.size(); ++idx) {
        auto& p = points[idx];
        const bool pt_has_rgb = p.has_color && !uniform_placeholder &&
                                !(non_mono_cnt == 0 && colored_flag_cnt == 0);
        if (out_point_has_true_rgb) {
            (*out_point_has_true_rgb)[idx] = pt_has_rgb;
        }
        if (pt_has_rgb) {
            ++true_rgb_count;
            continue;
        }

        // Compute true Black-and-White (R = G = B) grayscale from LiDAR intensity + elevation shading
        double norm_z = std::clamp((static_cast<double>(p.z) - static_cast<double>(min_z)) / z_span, 0.0, 1.0);
        double norm_i = 0.65;
        if (has_varying_inten && std::isfinite(p.intensity)) {
            norm_i = std::clamp((static_cast<double>(p.intensity) - static_cast<double>(min_inten)) / i_span, 0.0, 1.0);
        } else if (std::isfinite(p.intensity) && p.intensity > 1.0f) {
            norm_i = std::clamp(static_cast<double>(p.intensity) / 255.0, 0.15, 1.0);
        } else if (std::isfinite(p.intensity) && p.intensity > 0.0f) {
            norm_i = std::clamp(static_cast<double>(p.intensity), 0.15, 1.0);
        }

        double lum_f = has_varying_inten
            ? (36.0 + 145.0 * norm_i + 58.0 * norm_z)
            : (42.0 + 135.0 * norm_z + 45.0 * norm_i);
        uint8_t gray = static_cast<uint8_t>(std::clamp(std::round(lum_f), 32.0, 245.0));
        p.r = gray;
        p.g = gray;
        p.b = gray;
    }

    return (true_rgb_count == 0);
}

E57ValidationSummary StructuredWalkthroughExporter::validateStructuredE57(const std::string& e57_path) {
    E57ValidationSummary rep;
    std::ifstream ifs(e57_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        rep.error_message = "Cannot open E57 file: " + e57_path;
        return rep;
    }
    std::streamsize fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (fsize <= 0 || (fsize % static_cast<std::streamsize>(E57_PHYSICAL_PAGE_SIZE)) != 0) {
        rep.error_message = "Invalid E57 physical file size (not a multiple of 1024 bytes).";
        return rep;
    }
    rep.physical_size_bytes = static_cast<uint64_t>(fsize);
    rep.total_pages = static_cast<size_t>(fsize) / E57_PHYSICAL_PAGE_SIZE;

    std::vector<uint8_t> logical_stream;
    logical_stream.reserve(rep.total_pages * E57_LOGICAL_PAGE_SIZE);
    uint8_t page[E57_PHYSICAL_PAGE_SIZE];

    for (size_t p = 0; p < rep.total_pages; ++p) {
        ifs.read(reinterpret_cast<char*>(page), E57_PHYSICAL_PAGE_SIZE);
        uint32_t expected = (static_cast<uint32_t>(page[1020]) << 24) |
                            (static_cast<uint32_t>(page[1021]) << 16) |
                            (static_cast<uint32_t>(page[1022]) << 8) |
                            static_cast<uint32_t>(page[1023]);
        uint32_t actual = computeCrc32c(page, E57_LOGICAL_PAGE_SIZE);
        if (expected != actual) {
            rep.error_message = "CRC-32C mismatch at page " + std::to_string(p);
            return rep;
        }
        ++rep.crc_verified_pages;
        logical_stream.insert(logical_stream.end(), page, page + E57_LOGICAL_PAGE_SIZE);
    }

    if (std::memcmp(logical_stream.data(), "ASTM-E57", 8) != 0) {
        rep.error_message = "Missing ASTM-E57 file signature.";
        return rep;
    }
    uint64_t xml_phys = 0, xml_len = 0;
    std::memcpy(&xml_phys, logical_stream.data() + 24, sizeof(uint64_t));
    std::memcpy(&xml_len, logical_stream.data() + 32, sizeof(uint64_t));
    uint64_t xml_log = physicalToLogicalOffset(xml_phys);
    if (xml_log + xml_len > logical_stream.size()) {
        rep.error_message = "XML section out of logical stream bounds.";
        return rep;
    }

    std::string xml(reinterpret_cast<const char*>(logical_stream.data() + xml_log), static_cast<size_t>(xml_len));
    size_t d3d_pos = xml.find("<data3D");
    size_t i2d_pos = xml.find("<images2D");
    if (d3d_pos == std::string::npos || i2d_pos == std::string::npos) {
        rep.error_message = "Missing data3D or images2D section in E57 XML.";
        return rep;
    }

    std::string d3d_sub = xml.substr(d3d_pos, i2d_pos - d3d_pos);
    std::string i2d_sub = xml.substr(i2d_pos);

    std::set<std::string> scan_guids;
    size_t pos = 0;
    while ((pos = d3d_sub.find("<guid type=\"String\"><![CDATA[", pos)) != std::string::npos) {
        pos += 29;
        size_t end = d3d_sub.find("]]></guid>", pos);
        if (end != std::string::npos) {
            scan_guids.insert(d3d_sub.substr(pos, end - pos));
            ++rep.station_count;
            pos = end;
        }
    }

    bool all_matched = !scan_guids.empty();
    pos = 0;
    while ((pos = i2d_sub.find("<associatedData3DGuid type=\"String\"><![CDATA[", pos)) != std::string::npos) {
        pos += 45;
        size_t end = i2d_sub.find("]]></associatedData3DGuid>", pos);
        if (end != std::string::npos) {
            std::string assoc = i2d_sub.substr(pos, end - pos);
            if (scan_guids.find(assoc) == scan_guids.end()) all_matched = false;
            ++rep.image_count;
            pos = end;
        }
    }
    rep.associated_guids_matched = all_matched && (rep.image_count == rep.station_count);
    rep.valid = rep.associated_guids_matched;
    return rep;
}

size_t StructuredWalkthroughExporter::writeLod0Bin(
    const std::string& bin_path,
    const std::vector<schemas::PointXYZI>& points,
    size_t max_points) {
    fs::create_directories(fs::path(bin_path).parent_path());
    std::ofstream ofs(bin_path, std::ios::binary);
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
        const auto& p = points[idx];
        std::memcpy(rec + 0, &p.x, 4);
        std::memcpy(rec + 4, &p.y, 4);
        std::memcpy(rec + 8, &p.z, 4);
        rec[12] = p.r;
        rec[13] = p.g;
        rec[14] = p.b;
        ofs.write(reinterpret_cast<const char*>(rec), 15);
    }
    return indices.size();
}

bool StructuredWalkthroughExporter::readLod0Bin(
    const std::string& bin_path,
    std::vector<schemas::PointXYZI>& out_points) {
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
            break;
        }
        auto& p = out_points[i];
        std::memcpy(&p.x, rec + 0, 4);
        std::memcpy(&p.y, rec + 4, 4);
        std::memcpy(&p.z, rec + 8, 4);
        p.r = rec[12];
        p.g = rec[13];
        p.b = rec[14];
        p.intensity = 1.0f;
        p.has_color = true;
    }
    return true;
}

std::array<double, 16> StructuredWalkthroughExporter::computeTransformToProjectIcp(
    const std::vector<schemas::PointXYZI>& source_points,
    const std::vector<schemas::PointXYZI>& reference_points,
    double& out_rmse_m,
    double& out_overlap_ratio) {
    out_rmse_m = 0.0;
    out_overlap_ratio = 1.0;
    Eigen::Matrix4d T_cur = Eigen::Matrix4d::Identity();
    if (source_points.size() < 20 || reference_points.size() < 20) {
        return matrix4dToArray(T_cur);
    }

    auto buildT = [](double yaw, const Eigen::Vector3d& t) {
        Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
        double c = std::cos(yaw), s = std::sin(yaw);
        M(0, 0) = c;  M(0, 1) = -s;
        M(1, 0) = s;  M(1, 1) = c;
        M.block<3, 1>(0, 3) = t;
        return M;
    };

    auto makeKdTree = [](const std::vector<Eigen::Vector3d>& pts) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->points.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            cloud->points[i].x = static_cast<float>(pts[i].x());
            cloud->points[i].y = static_cast<float>(pts[i].y());
            cloud->points[i].z = static_cast<float>(pts[i].z());
        }
        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(cloud);
        return std::make_pair(cloud, tree);
    };

    auto src_coarse = voxelSubsampleEigen(source_points, 0.25);
    auto ref_coarse = voxelSubsampleEigen(reference_points, 0.25);
    if (src_coarse.size() < 10 || ref_coarse.size() < 10) {
        return matrix4dToArray(T_cur);
    }

    std::vector<double> zs_s, zs_r;
    Eigen::Vector2d c_src(0.0, 0.0), c_ref(0.0, 0.0);
    for (const auto& v : src_coarse) { zs_s.push_back(v.z()); c_src += v.head<2>(); }
    for (const auto& v : ref_coarse) { zs_r.push_back(v.z()); c_ref += v.head<2>(); }
    c_src /= static_cast<double>(src_coarse.size());
    c_ref /= static_cast<double>(ref_coarse.size());
    std::sort(zs_s.begin(), zs_s.end());
    std::sort(zs_r.begin(), zs_r.end());
    double dz0 = zs_r[zs_r.size() / 10] - zs_s[zs_s.size() / 10];

    auto [ref_cloud_c, tree_c] = makeKdTree(ref_coarse);
    std::vector<int> nn_idx(1);
    std::vector<float> nn_sq(1);

    // Evaluate identity first
    std::vector<double> id_dists;
    id_dists.reserve(src_coarse.size());
    for (const auto& v : src_coarse) {
        pcl::PointXYZ q(static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z()));
        if (tree_c.nearestKSearch(q, 1, nn_idx, nn_sq) > 0) id_dists.push_back(std::sqrt(nn_sq[0]));
    }
    std::sort(id_dists.begin(), id_dists.end());
    double id_med = id_dists.empty() ? 999.0 : id_dists[id_dists.size() / 2];

    if (id_med >= 0.12) {
        double best_score = 1e18;
        double best_yaw = 0.0;
        Eigen::Vector3d best_t(c_ref.x() - c_src.x(), c_ref.y() - c_src.y(), dz0);

        for (int k = 0; k < 16; ++k) {
            double yaw = (2.0 * kPi * k) / 16.0;
            Eigen::Matrix2d R2;
            R2 << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
            std::vector<double> d_list;
            d_list.reserve(src_coarse.size());
            for (const auto& v : src_coarse) {
                Eigen::Vector2d xy = R2 * (v.head<2>() - c_src) + c_ref;
                pcl::PointXYZ q(static_cast<float>(xy.x()), static_cast<float>(xy.y()), static_cast<float>(v.z() + dz0));
                if (tree_c.nearestKSearch(q, 1, nn_idx, nn_sq) > 0) d_list.push_back(std::sqrt(nn_sq[0]));
            }
            if (d_list.empty()) continue;
            std::sort(d_list.begin(), d_list.end());
            double score = d_list[d_list.size() * 65 / 100];
            if (score < best_score) {
                best_score = score;
                best_yaw = yaw;
                Eigen::Vector2d txy = c_ref - R2 * c_src;
                best_t = Eigen::Vector3d(txy.x(), txy.y(), dz0);
            }
        }
        T_cur = buildT(best_yaw, best_t);
    }

    struct Stage { double vs; double max_corr; int iters; };
    const Stage stages[2] = {{0.25, 1.20, 20}, {0.08, 0.35, 35}};
    for (const auto& stg : stages) {
        auto src_pts = voxelSubsampleEigen(source_points, stg.vs);
        auto ref_pts = voxelSubsampleEigen(reference_points, stg.vs);
        if (src_pts.size() < 10 || ref_pts.size() < 10) continue;
        auto [ref_cl, tree] = makeKdTree(ref_pts);

        for (int it = 0; it < stg.iters; ++it) {
            std::vector<Eigen::Vector3d> ps, pr;
            ps.reserve(src_pts.size());
            pr.reserve(src_pts.size());
            Eigen::Matrix3d R_c = T_cur.block<3, 3>(0, 0);
            Eigen::Vector3d t_c = T_cur.block<3, 1>(0, 3);

            for (const auto& sv : src_pts) {
                Eigen::Vector3d tv = R_c * sv + t_c;
                pcl::PointXYZ q(static_cast<float>(tv.x()), static_cast<float>(tv.y()), static_cast<float>(tv.z()));
                if (tree.nearestKSearch(q, 1, nn_idx, nn_sq) > 0) {
                    double d = std::sqrt(static_cast<double>(nn_sq[0]));
                    if (d <= stg.max_corr) {
                        ps.push_back(tv);
                        pr.push_back(ref_pts[static_cast<size_t>(nn_idx[0])]);
                    }
                }
            }
            if (ps.size() < 10) break;

            Eigen::Vector2d mu_s(0.0, 0.0), mu_r(0.0, 0.0);
            std::vector<double> dzs;
            dzs.reserve(ps.size());
            for (size_t i = 0; i < ps.size(); ++i) {
                mu_s += ps[i].head<2>();
                mu_r += pr[i].head<2>();
                dzs.push_back(pr[i].z() - ps[i].z());
            }
            mu_s /= static_cast<double>(ps.size());
            mu_r /= static_cast<double>(pr.size());

            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            for (size_t i = 0; i < ps.size(); ++i) {
                H += (ps[i].head<2>() - mu_s) * (pr[i].head<2>() - mu_r).transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d R2_d = svd.matrixV() * svd.matrixU().transpose();
            if (R2_d.determinant() < 0.0) {
                Eigen::Matrix2d V = svd.matrixV();
                V.col(1) *= -1.0;
                R2_d = V * svd.matrixU().transpose();
            }
            double dyaw = std::atan2(R2_d(1, 0), R2_d(0, 0));
            Eigen::Vector2d dt_xy = mu_r - R2_d * mu_s;
            std::sort(dzs.begin(), dzs.end());
            double dt_z = dzs[dzs.size() / 2];

            T_cur = buildT(dyaw, Eigen::Vector3d(dt_xy.x(), dt_xy.y(), dt_z)) * T_cur;
            if (std::abs(dyaw) < 1e-4 && dt_xy.norm() < 5e-4 && std::abs(dt_z) < 5e-4) break;
        }
    }

    auto src_eval = voxelSubsampleEigen(source_points, 0.10);
    auto ref_eval = voxelSubsampleEigen(reference_points, 0.10);
    auto [ref_cl_e, tree_e] = makeKdTree(ref_eval);
    size_t inl = 0;
    double sq_sum = 0.0;
    Eigen::Matrix3d R_f = T_cur.block<3, 3>(0, 0);
    Eigen::Vector3d t_f = T_cur.block<3, 1>(0, 3);
    for (const auto& sv : src_eval) {
        Eigen::Vector3d tv = R_f * sv + t_f;
        pcl::PointXYZ q(static_cast<float>(tv.x()), static_cast<float>(tv.y()), static_cast<float>(tv.z()));
        if (tree_e.nearestKSearch(q, 1, nn_idx, nn_sq) > 0 && nn_sq[0] <= 0.25f * 0.25f) {
            ++inl;
            sq_sum += static_cast<double>(nn_sq[0]);
        }
    }
    out_overlap_ratio = static_cast<double>(inl) / static_cast<double>(std::max<size_t>(1, src_eval.size()));
    out_rmse_m = inl > 0 ? std::sqrt(sq_sum / static_cast<double>(inl)) : 0.0;
    return matrix4dToArray(T_cur);
}

WalkthroughBundleResult StructuredWalkthroughExporter::exportWalkthroughBundle(
    const std::string& output_dir,
    const std::vector<schemas::PointXYZI>& input_points,
    const std::vector<schemas::Pose3D>& trajectory,
    const std::vector<FlashStationAnchor>& flash_anchors,
    const std::vector<PosedRgbKeyframe>& keyframes,
    const WalkthroughExportConfig& config,
    const std::vector<schemas::PointXYZI>* reference_baseline_points) {
    auto t0 = std::chrono::steady_clock::now();
    WalkthroughBundleResult result;
    result.bundle_dir = output_dir;

    if (input_points.empty()) {
        result.error_message = "Cannot generate walkthrough bundle from empty point cloud.";
        return result;
    }

    fs::create_directories(fs::path(output_dir) / "panos");
    fs::create_directories(fs::path(output_dir) / "floors");
    fs::create_directories(fs::path(output_dir) / "cache");
    fs::create_directories(fs::path(output_dir) / "stations");

    std::vector<schemas::PointXYZI> cloud = input_points;
    Eigen::Matrix4d T_ground = Eigen::Matrix4d::Identity();
    if (config.enforce_gravity_z_up) {
        auto gp_res = PointCloudProcessor::detectAndAlignGroundPlane(cloud, 0.04, 300, 35.0, true, false);
        if (gp_res.found) {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    T_ground(r, c) = gp_res.alignment_transform[r][c];
                }
            }
        }
    }

    // Optional 4D historical registration into Project Baseline Frame
    Eigen::Matrix4d T_proj = Eigen::Matrix4d::Identity();
    if (reference_baseline_points && reference_baseline_points->size() >= 20) {
        auto arr_tf = computeTransformToProjectIcp(
            cloud, *reference_baseline_points, result.registration_rmse_m, result.registration_overlap);
        T_proj = arrayToMatrix4d(arr_tf);
    }
    const bool apply_tf = !(T_proj - Eigen::Matrix4d::Identity()).isZero(1e-6);
    if (apply_tf) {
        transformPointCloudInPlace(cloud, T_proj);
    }
    const Eigen::Matrix4d T_pose = T_proj * T_ground;
    const bool apply_pose_tf = !(T_pose - Eigen::Matrix4d::Identity()).isZero(1e-6);
    double tf_yaw = std::atan2(T_pose(1, 0), T_pose(0, 0));

    // Populate true Black-and-White (R = G = B) grayscale for any uncolored / LiDAR-only points
    std::vector<bool> pt_has_true_rgb;
    const bool cloud_is_monochrome = populateMonochromeGrayscaleForUncoloredPoints(cloud, &pt_has_true_rgb);
    result.is_monochrome_lidar_only = cloud_is_monochrome && keyframes.empty();

    float cloud_min_i = std::numeric_limits<float>::max();
    float cloud_max_i = std::numeric_limits<float>::lowest();
    for (const auto& p : cloud) {
        if (std::isfinite(p.intensity)) {
            cloud_min_i = std::min(cloud_min_i, p.intensity);
            cloud_max_i = std::max(cloud_max_i, p.intensity);
        }
    }
    const double cloud_i_span = static_cast<double>(cloud_max_i - cloud_min_i);
    const bool cloud_has_varying_i = (cloud_i_span > 1e-3);

    // Optional keyframes export (keyframes/kf_XXXX.jpg + keyframes/keyframes_manifest.json)
    if (!keyframes.empty()) {
        fs::path kf_dir = fs::path(output_dir) / "keyframes";
        fs::create_directories(kf_dir);
        nlohmann::json kf_list = nlohmann::json::array();
        double fx_val = keyframes[0].fx;
        double fy_val = keyframes[0].fy;
        double cx_val = keyframes[0].cx;
        double cy_val = keyframes[0].cy;
        int w_val = keyframes[0].width;
        int h_val = keyframes[0].height;

        for (size_t k_i = 0; k_i < keyframes.size(); ++k_i) {
            const auto& kf = keyframes[k_i];
            if (kf.width <= 0 || kf.height <= 0 ||
                kf.rgb_pixels.size() < static_cast<size_t>(kf.width * kf.height * 3)) {
                continue;
            }
            std::ostringstream fn_ss;
            fn_ss << "kf_" << std::setw(4) << std::setfill('0') << k_i << ".jpg";
            std::string fname = fn_ss.str();
            auto jpg_bytes = encodeJpegRgb(kf.rgb_pixels.data(), kf.width, kf.height, config.jpeg_quality);
            std::ofstream kofs((kf_dir / fname).string(), std::ios::binary);
            kofs.write(reinterpret_cast<const char*>(jpg_bytes.data()), static_cast<std::streamsize>(jpg_bytes.size()));
            kofs.close();

            Eigen::Matrix4d T_wc = arrayToMatrix4d(kf.T_world_cam);
            if (apply_pose_tf) T_wc = T_pose * T_wc;
            Eigen::Vector3d cam_t = T_wc.block<3, 1>(0, 3);
            Eigen::Quaterniond cam_q(T_wc.block<3, 3>(0, 0));
            cam_q.normalize();
            double cam_yaw = std::atan2(T_wc(1, 0), T_wc(0, 0));

            nlohmann::json T_rows = nlohmann::json::array();
            for (int r = 0; r < 4; ++r) {
                T_rows.push_back({T_wc(r, 0), T_wc(r, 1), T_wc(r, 2), T_wc(r, 3)});
            }

            kf_list.push_back({
                {"id", k_i},
                {"file", fname},
                {"timestamp", static_cast<double>(kf.timestamp_ns) * 1e-9},
                {"sharpness_laplacian_var", kf.sharpness_score},
                {"position_world_m", {cam_t.x(), cam_t.y(), cam_t.z()}},
                {"quaternion_xyzw_world", {cam_q.x(), cam_q.y(), cam_q.z(), cam_q.w()}},
                {"yaw_rad", cam_yaw},
                {"T_world_cam", T_rows}
            });
        }

        nlohmann::json kf_manifest = {
            {"schema", "ArtificialVision.KeyframesManifest.v1"},
            {"camera_intrinsics", {
                {"fx", fx_val},
                {"fy", fy_val},
                {"cx", cx_val},
                {"cy", cy_val},
                {"width", w_val},
                {"height", h_val}
            }},
            {"keyframes", kf_list}
        };
        std::ofstream kmf((kf_dir / "keyframes_manifest.json").string());
        kmf << kf_manifest.dump(2);
        kmf.close();
    }

    // Estimate local floor Z helper
    auto estimateLocalSlabZ = [&](double x, double y, double z_hint) -> double {
        std::vector<double> zs;
        for (size_t i = 0; i < cloud.size(); i += std::max<size_t>(1, cloud.size() / 60000)) {
            const auto& p = cloud[i];
            double dx = p.x - x, dy = p.y - y;
            if (dx * dx + dy * dy <= 1.44 && p.z <= z_hint + 0.5 && p.z >= z_hint - 2.8) {
                zs.push_back(p.z);
            }
        }
        if (zs.empty()) return z_hint;
        std::sort(zs.begin(), zs.end());
        return zs[zs.size() / 10];
    };

    // 1. Select Virtual Scan Stations (Flash anchors -> Trajectory spacing -> Cloud fallback)
    struct RawStation {
        std::string id;
        Eigen::Vector3d pos;
        double yaw_rad;
        std::string mode;
    };
    std::vector<RawStation> stations;

    for (size_t i = 0; i < flash_anchors.size(); ++i) {
        const auto& fa = flash_anchors[i];
        Eigen::Vector3d p(fa.position.x, fa.position.y, fa.position.z);
        if (apply_pose_tf) p = T_pose.block<3, 3>(0, 0) * p + T_pose.block<3, 1>(0, 3);
        if (!fa.z_is_sensor_height) {
            p.z() = estimateLocalSlabZ(p.x(), p.y(), p.z()) + config.sensor_eye_height_m;
        }
        stations.push_back({
            fa.label.empty() ? ("flash_" + std::to_string(i)) : fa.label,
            p,
            fa.yaw_rad + tf_yaw,
            "flash_anchor"
        });
    }

    auto tooCloseToExisting = [&](const Eigen::Vector3d& cand, double min_d) {
        for (const auto& s : stations) {
            if ((s.pos.head<2>() - cand.head<2>()).norm() < min_d && std::abs(s.pos.z() - cand.z()) < 1.8) {
                return true;
            }
        }
        return false;
    };

    for (const auto& tp : trajectory) {
        Eigen::Vector3d p(tp.position.x, tp.position.y, tp.position.z);
        if (apply_pose_tf) p = T_pose.block<3, 3>(0, 0) * p + T_pose.block<3, 1>(0, 3);
        double qx = tp.orientation.x, qy = tp.orientation.y, qz = tp.orientation.z, qw = tp.orientation.w;
        double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)) + tf_yaw;
        if (!tooCloseToExisting(p, std::max(0.8, config.station_spacing_m))) {
            stations.push_back({"traj_" + std::to_string(stations.size()), p, yaw, "auto_trajectory"});
        }
    }

    if (stations.empty()) {
        // Fallback for raw clouds without trajectory: place viewpoint(s) over navigable floor centroids
        Eigen::Vector3d mean_p(0.0, 0.0, 0.0);
        std::vector<double> all_z;
        all_z.reserve(std::min<size_t>(cloud.size(), 20000));
        size_t stride = std::max<size_t>(1, cloud.size() / 20000);
        size_t cnt = 0;
        for (size_t i = 0; i < cloud.size(); i += stride) {
            mean_p += Eigen::Vector3d(cloud[i].x, cloud[i].y, cloud[i].z);
            all_z.push_back(cloud[i].z);
            ++cnt;
        }
        mean_p /= static_cast<double>(std::max<size_t>(1, cnt));
        std::sort(all_z.begin(), all_z.end());
        double floor_z = all_z.empty() ? 0.0 : all_z[all_z.size() / 10];
        stations.push_back({
            "auto_center_000",
            Eigen::Vector3d(mean_p.x(), mean_p.y(), floor_z + config.sensor_eye_height_m),
            0.0,
            "auto_geometry"
        });
    }

    // 2. Synthesize Structured Scans & 360° ASTM E2807-11 Panoramas
    const int p_w = std::max(256, config.pano_width);
    const int p_h = std::max(128, config.pano_height);
    const int s_rows = std::max(32, config.scan_rows);
    const int s_cols = std::max(64, config.scan_cols);

    std::vector<E57StationData> e57_stations;
    e57_stations.reserve(stations.size());

    for (size_t s_i = 0; s_i < stations.size(); ++s_i) {
        const auto& raw_st = stations[s_i];
        E57StationData st;
        st.station_id = raw_st.id;
        st.selection_mode = raw_st.mode;
        std::ostringstream pid_ss;
        pid_ss << "pano_" << std::setw(3) << std::setfill('0') << s_i;
        st.pano_id = pid_ss.str();
        st.scan_guid = makeDeterministicUuid("av-scan-3d-" + output_dir + "-" + st.pano_id, true);
        st.image_guid = makeDeterministicUuid("av-scan-2d-" + output_dir + "-" + st.pano_id, true);
        st.pos_world = raw_st.pos;
        st.yaw_rad = raw_st.yaw_rad;

        double half_y = 0.5 * raw_st.yaw_rad;
        double qw = std::cos(half_y), qz = std::sin(half_y);
        st.quat_wxyz = {qw, 0.0, 0.0, qz};
        st.quat_xyzw = {0.0, 0.0, qz, qw};

        Eigen::Matrix3d R_ws;
        R_ws << std::cos(raw_st.yaw_rad), -std::sin(raw_st.yaw_rad), 0.0,
                std::sin(raw_st.yaw_rad),  std::cos(raw_st.yaw_rad), 0.0,
                0.0,                       0.0,                      1.0;

        st.scan_rows = s_rows;
        st.scan_cols = s_cols;
        const size_t scan_cells = static_cast<size_t>(s_rows) * static_cast<size_t>(s_cols);
        st.scan_x.assign(scan_cells, 0.0f);
        st.scan_y.assign(scan_cells, 0.0f);
        st.scan_z.assign(scan_cells, 0.0f);
        st.scan_invalid.assign(scan_cells, 2); // 2 = unobserved/missing
        st.scan_r.assign(scan_cells, 0);
        st.scan_g.assign(scan_cells, 0);
        st.scan_b.assign(scan_cells, 0);
        std::vector<float> scan_best_r(scan_cells, 1e9f);

        st.pano_w = p_w;
        st.pano_h = p_h;
        const size_t pano_px = static_cast<size_t>(p_w) * static_cast<size_t>(p_h);
        st.pano_rgb.assign(pano_px * 3, 0); // Pure #000000 black for unobserved regions!
        st.pano_depth_m.assign(pano_px, 1e9f);
        // Provenance codes: 0=UNOBSERVED, 1=PHOTO_RGB, 2=POINTCLOUD_RGB, 3=DEPTH_GUIDED_FILL, 4=GEOMETRY_UNCOLORED
        st.pano_prov_code.assign(pano_px, 0);

        for (size_t pt_idx = 0; pt_idx < cloud.size(); ++pt_idx) {
            const auto& pt = cloud[pt_idx];
            Eigen::Vector3d dp(pt.x - st.pos_world.x(), pt.y - st.pos_world.y(), pt.z - st.pos_world.z());
            Eigen::Vector3d pl = R_ws.transpose() * dp;
            double r = pl.norm();
            if (r < config.min_range_m || r > config.max_range_m) continue;

            double az = std::atan2(pl.y(), pl.x()); // [-pi, pi]
            double el = std::asin(std::clamp(pl.z() / r, -1.0, 1.0));
            double u_norm = std::fmod(0.5 - az / (2.0 * kPi) + 1.0, 1.0);
            double v_norm = std::clamp(0.5 - el / kPi, 0.0, 1.0 - 1e-7);

            const bool has_true_rgb = (pt_idx < pt_has_true_rgb.size()) && pt_has_true_rgb[pt_idx];
            uint8_t cr = pt.r, cg = pt.g, cb = pt.b;
            uint8_t prov = has_true_rgb ? 2 : 4; // 2=POINTCLOUD_RGB, 4=GEOMETRY_UNCOLORED
            if (!has_true_rgb) {
                // True monochrome Black-and-White (R = G = B) shading from LiDAR intensity + range + elevation
                double norm_i = 0.65;
                if (cloud_has_varying_i && std::isfinite(pt.intensity)) {
                    norm_i = std::clamp((static_cast<double>(pt.intensity) - static_cast<double>(cloud_min_i)) / cloud_i_span, 0.0, 1.0);
                } else {
                    norm_i = static_cast<double>(pt.r) / 255.0;
                }
                double range_atten = std::clamp(1.0 - (r / std::max(10.0, config.max_range_m)) * 0.45, 0.35, 1.0);
                double elev_cue = std::clamp(0.60 + 0.30 * (pl.z() / r), 0.30, 1.0);
                double shade = std::clamp(range_atten * (0.55 * norm_i + 0.45 * elev_cue), 0.14, 0.96);
                uint8_t lum = static_cast<uint8_t>(std::clamp(std::round(shade * 245.0), 28.0, 245.0));
                cr = lum;
                cg = lum;
                cb = lum;
            }

            // Structured spherical scan cell (z-buffered closest surface)
            int sc_c = static_cast<int>(u_norm * s_cols) % s_cols;
            int sc_r = std::clamp(static_cast<int>(v_norm * s_rows), 0, s_rows - 1);
            size_t sc_i = static_cast<size_t>(sc_r) * static_cast<size_t>(s_cols) + static_cast<size_t>(sc_c);
            if (static_cast<float>(r) < scan_best_r[sc_i]) {
                scan_best_r[sc_i] = static_cast<float>(r);
                st.scan_x[sc_i] = static_cast<float>(pl.x());
                st.scan_y[sc_i] = static_cast<float>(pl.y());
                st.scan_z[sc_i] = static_cast<float>(pl.z());
                st.scan_invalid[sc_i] = 0;
                st.scan_r[sc_i] = cr;
                st.scan_g[sc_i] = cg;
                st.scan_b[sc_i] = cb;
            }

            // Range-adaptive surfel splat into panorama
            int pc = static_cast<int>(u_norm * p_w) % p_w;
            int pr = std::clamp(static_cast<int>(v_norm * p_h), 0, p_h - 1);
            int rad = (r < 2.2) ? 3 : ((r < 5.5) ? 2 : 1);
            for (int dy = -rad; dy <= rad; ++dy) {
                int rr = pr + dy;
                if (rr < 0 || rr >= p_h) continue;
                for (int dx = -rad; dx <= rad; ++dx) {
                    if (dx * dx + dy * dy > rad * rad) continue;
                    int cc = (pc + dx + p_w) % p_w;
                    size_t pi = static_cast<size_t>(rr) * static_cast<size_t>(p_w) + static_cast<size_t>(cc);
                    float eff_r = static_cast<float>(r) + (dx == 0 && dy == 0 ? 0.0f : 0.03f);
                    if (eff_r < st.pano_depth_m[pi]) {
                        st.pano_depth_m[pi] = eff_r;
                        st.pano_rgb[pi * 3 + 0] = cr;
                        st.pano_rgb[pi * 3 + 1] = cg;
                        st.pano_rgb[pi * 3 + 2] = cb;
                        st.pano_prov_code[pi] = prov;
                    }
                }
            }
        }

        // Optional Posed RGB Keyframe Reprojection onto observed geometry
        bool used_kf = false;
        for (const auto& kf : keyframes) {
            if (kf.width <= 0 || kf.height <= 0 || kf.rgb_pixels.size() < static_cast<size_t>(kf.width * kf.height * 3)) continue;
            Eigen::Matrix4d T_wc = arrayToMatrix4d(kf.T_world_cam);
            if (apply_pose_tf) T_wc = T_pose * T_wc;
            Eigen::Vector3d cam_pos = T_wc.block<3, 1>(0, 3);
            if ((cam_pos - st.pos_world).norm() > 2.5) continue;
            Eigen::Matrix3d R_cw = T_wc.block<3, 3>(0, 0).transpose();
            Eigen::Vector3d t_cw = -R_cw * cam_pos;

            for (int rr = 0; rr < p_h; ++rr) {
                double el = (0.5 - (rr + 0.5) / p_h) * kPi;
                double cos_el = std::cos(el), sin_el = std::sin(el);
                for (int cc = 0; cc < p_w; ++cc) {
                    size_t pi = static_cast<size_t>(rr) * static_cast<size_t>(p_w) + static_cast<size_t>(cc);
                    float d = st.pano_depth_m[pi];
                    if (d >= 1e8f) continue;
                    double az = (0.5 - (cc + 0.5) / p_w) * 2.0 * kPi;
                    Eigen::Vector3d p_loc(d * cos_el * std::cos(az), d * cos_el * std::sin(az), d * sin_el);
                    Eigen::Vector3d p_w_pt = R_ws * p_loc + st.pos_world;
                    Eigen::Vector3d p_cam = R_cw * p_w_pt + t_cw;
                    if (p_cam.z() <= 0.25 || p_cam.z() > 12.0) continue;
                    int u = static_cast<int>(std::round(kf.fx * (p_cam.x() / p_cam.z()) + kf.cx));
                    int v = static_cast<int>(std::round(kf.fy * (p_cam.y() / p_cam.z()) + kf.cy));
                    if (u >= 2 && u < kf.width - 2 && v >= 2 && v < kf.height - 2) {
                        size_t ki = (static_cast<size_t>(v) * static_cast<size_t>(kf.width) + static_cast<size_t>(u)) * 3;
                        st.pano_rgb[pi * 3 + 0] = kf.rgb_pixels[ki + 0];
                        st.pano_rgb[pi * 3 + 1] = kf.rgb_pixels[ki + 1];
                        st.pano_rgb[pi * 3 + 2] = kf.rgb_pixels[ki + 2];
                        st.pano_prov_code[pi] = 1; // 1 = PHOTO_RGB
                        used_kf = true;
                    }
                }
            }
        }

        // Strictly gated 1-px micro-pinhole fill (< 8 cm depth discontinuity)
        if (config.enable_small_gap_fill) {
            std::vector<uint8_t> rgb_copy = st.pano_rgb;
            std::vector<float> d_copy = st.pano_depth_m;
            for (int rr = 1; rr < p_h - 1; ++rr) {
                for (int cc = 0; cc < p_w; ++cc) {
                    size_t pi = static_cast<size_t>(rr) * static_cast<size_t>(p_w) + static_cast<size_t>(cc);
                    if (d_copy[pi] < 1e8f) continue;
                    size_t nbs[4] = {
                        static_cast<size_t>(rr - 1) * static_cast<size_t>(p_w) + static_cast<size_t>(cc),
                        static_cast<size_t>(rr + 1) * static_cast<size_t>(p_w) + static_cast<size_t>(cc),
                        static_cast<size_t>(rr) * static_cast<size_t>(p_w) + static_cast<size_t>((cc + p_w - 1) % p_w),
                        static_cast<size_t>(rr) * static_cast<size_t>(p_w) + static_cast<size_t>((cc + 1) % p_w)
                    };
                    int vcnt = 0;
                    float dmin = 1e9f, dmax = 0.0f, dsum = 0.0f;
                    int rsum = 0, gsum = 0, bsum = 0;
                    for (size_t ni : nbs) {
                        if (d_copy[ni] < 1e8f) {
                            ++vcnt;
                            dmin = std::min(dmin, d_copy[ni]);
                            dmax = std::max(dmax, d_copy[ni]);
                            dsum += d_copy[ni];
                            rsum += rgb_copy[ni * 3 + 0];
                            gsum += rgb_copy[ni * 3 + 1];
                            bsum += rgb_copy[ni * 3 + 2];
                        }
                    }
                    if (vcnt >= 3 && (dmax - dmin) <= 0.08f) {
                        st.pano_depth_m[pi] = dsum / static_cast<float>(vcnt);
                        uint8_t fr = static_cast<uint8_t>(rsum / vcnt);
                        uint8_t fg = static_cast<uint8_t>(gsum / vcnt);
                        uint8_t fb = static_cast<uint8_t>(bsum / vcnt);
                        if (result.is_monochrome_lidar_only) {
                            fg = fr;
                            fb = fr;
                        }
                        st.pano_rgb[pi * 3 + 0] = fr;
                        st.pano_rgb[pi * 3 + 1] = fg;
                        st.pano_rgb[pi * 3 + 2] = fb;
                        st.pano_prov_code[pi] = 3; // 3 = DEPTH_GUIDED_FILL
                    }
                }
            }
        }

        size_t obs_px = 0;
        size_t photo_px = 0;
        size_t pc_rgb_px = 0;
        size_t fill_px = 0;
        size_t geom_uncol_px = 0;
        std::vector<uint16_t> depth_mm(pano_px, 0);
        std::vector<uint8_t> cov_mask(pano_px, 0);
        for (size_t pi = 0; pi < pano_px; ++pi) {
            if (st.pano_depth_m[pi] < 1e8f) {
                ++obs_px;
                cov_mask[pi] = 255;
                depth_mm[pi] = static_cast<uint16_t>(std::clamp(std::round(st.pano_depth_m[pi] * 1000.0f), 0.0f, 65535.0f));
                uint8_t pc_code = st.pano_prov_code[pi];
                if (pc_code == 1) ++photo_px;
                else if (pc_code == 2) ++pc_rgb_px;
                else if (pc_code == 3) ++fill_px;
                else if (pc_code == 4) ++geom_uncol_px;
            } else {
                st.pano_depth_m[pi] = 0.0f;
                st.pano_prov_code[pi] = 0;
            }
        }
        const double inv_total_px = 1.0 / static_cast<double>(std::max<size_t>(1, pano_px));
        st.valid_pixel_ratio = static_cast<double>(obs_px) * inv_total_px;
        st.photo_rgb_ratio = static_cast<double>(photo_px) * inv_total_px;
        st.pointcloud_rgb_ratio = static_cast<double>(pc_rgb_px) * inv_total_px;
        st.depth_fill_ratio = static_cast<double>(fill_px) * inv_total_px;
        st.geometry_uncolored_ratio = static_cast<double>(geom_uncol_px) * inv_total_px;
        st.unobserved_ratio = std::max(0.0, 1.0 - st.valid_pixel_ratio);

        if (used_kf || photo_px > 0) {
            st.provenance = "HYBRID_KEYFRAME_AND_CLOUD";
        } else if (pc_rgb_px > 0) {
            st.provenance = "SYNTHETIC_FROM_CLOUD";
        } else {
            st.provenance = "GEOMETRY_UNCOLORED";
        }

        for (uint8_t inv : st.scan_invalid) if (inv == 0) ++st.valid_scan_points;

        st.pano_jpeg = encodeJpegRgb(st.pano_rgb.data(), p_w, p_h, config.jpeg_quality);
        std::ofstream jf((fs::path(output_dir) / "panos" / (st.pano_id + ".jpg")).string(), std::ios::binary);
        jf.write(reinterpret_cast<const char*>(st.pano_jpeg.data()), static_cast<std::streamsize>(st.pano_jpeg.size()));
        jf.close();

        writePngGray16((fs::path(output_dir) / "panos" / (st.pano_id + "_depth.png")).string(), depth_mm.data(), p_w, p_h);

        // Write AV Structure per-station directory: stations/<station_id>/...
        fs::path st_dir = fs::path(output_dir) / "stations" / st.station_id;
        fs::create_directories(st_dir);
        {
            std::ofstream sjf((st_dir / "panorama_rgb.jpg").string(), std::ios::binary);
            sjf.write(reinterpret_cast<const char*>(st.pano_jpeg.data()), static_cast<std::streamsize>(st.pano_jpeg.size()));
        }
        writePngGray16((st_dir / "depth_map_mm.png").string(), depth_mm.data(), p_w, p_h);
        writePngGray8((st_dir / "coverage_mask.png").string(), cov_mask.data(), p_w, p_h);
        writePngGray8((st_dir / "provenance_code.png").string(), st.pano_prov_code.data(), p_w, p_h);
        writeBinaryPlyStructuredScan((st_dir / "structured_scan.ply").string(), st);

        nlohmann::json st_meta = {
            {"station_id", st.station_id},
            {"selection_mode", st.selection_mode},
            {"scan_guid", st.scan_guid},
            {"image_guid", st.image_guid},
            {"position_world_m", {
                std::round(st.pos_world.x() * 1000000.0) / 1000000.0,
                std::round(st.pos_world.y() * 1000000.0) / 1000000.0,
                std::round(st.pos_world.z() * 1000000.0) / 1000000.0
            }},
            {"yaw_rad", std::round(st.yaw_rad * 1000000.0) / 1000000.0},
            {"quaternion_wxyz_world", {st.quat_wxyz[0], st.quat_wxyz[1], st.quat_wxyz[2], st.quat_wxyz[3]}},
            {"panorama_width", p_w},
            {"panorama_height", p_h},
            {"structured_scan_rows", s_rows},
            {"structured_scan_cols", s_cols},
            {"structured_scan_valid_points", st.valid_scan_points},
            {"provenance_fractions", {
                {"photo_rgb_fraction", std::round(st.photo_rgb_ratio * 10000.0) / 10000.0},
                {"pointcloud_rgb_fraction", std::round(st.pointcloud_rgb_ratio * 10000.0) / 10000.0},
                {"depth_fill_fraction", std::round(st.depth_fill_ratio * 10000.0) / 10000.0},
                {"geometry_uncolored_fraction", std::round(st.geometry_uncolored_ratio * 10000.0) / 10000.0},
                {"unobserved_fraction", std::round(st.unobserved_ratio * 10000.0) / 10000.0}
            }},
            {"files", {
                {"panorama_rgb", "panorama_rgb.jpg"},
                {"depth_map_mm", "depth_map_mm.png"},
                {"coverage_mask", "coverage_mask.png"},
                {"provenance_code", "provenance_code.png"},
                {"structured_scan_ply", "structured_scan.ply"}
            }}
        };
        {
            std::ofstream smf((st_dir / "station_metadata.json").string());
            smf << st_meta.dump(2);
        }

        st.description = "AV Scan Virtual Station (" + raw_st.mode + ")";
        e57_stations.push_back(std::move(st));
    }

    // 3. Multi-story floor plateau detection + midpoint stair split
    std::vector<double> z_vals;
    for (const auto& s : e57_stations) z_vals.push_back(s.pos_world.z());
    std::vector<double> sorted_z = z_vals;
    std::sort(sorted_z.begin(), sorted_z.end());
    sorted_z.erase(std::unique(sorted_z.begin(), sorted_z.end(), [](double a, double b) { return std::abs(a - b) < 1e-4; }), sorted_z.end());

    struct FloorCluster { std::vector<double> all_z; double mean_z{0.0}; double plateau_z{0.0}; };
    std::vector<FloorCluster> clusters;
    std::vector<double> cur_u = {sorted_z[0]};
    for (size_t i = 1; i < sorted_z.size(); ++i) {
        if (sorted_z[i] - sorted_z[i - 1] > 0.25) {
            FloorCluster fc;
            for (double z : z_vals) for (double uz : cur_u) if (std::abs(z - uz) < 1e-4) fc.all_z.push_back(z);
            fc.mean_z = std::accumulate(fc.all_z.begin(), fc.all_z.end(), 0.0) / static_cast<double>(fc.all_z.size());
            clusters.push_back(fc);
            cur_u = {sorted_z[i]};
        } else {
            cur_u.push_back(sorted_z[i]);
        }
    }
    {
        FloorCluster fc;
        for (double z : z_vals) for (double uz : cur_u) if (std::abs(z - uz) < 1e-4) fc.all_z.push_back(z);
        fc.mean_z = std::accumulate(fc.all_z.begin(), fc.all_z.end(), 0.0) / static_cast<double>(fc.all_z.size());
        clusters.push_back(fc);
    }

    size_t max_cl_cnt = 1;
    for (const auto& c : clusters) max_cl_cnt = std::max(max_cl_cnt, c.all_z.size());

    std::vector<FloorCluster> major_floors, stair_clusters;
    for (size_t i = 0; i < clusters.size(); ++i) {
        double lgap = (i > 0) ? (clusters[i].mean_z - clusters[i - 1].mean_z) : 999.0;
        double rgap = (i + 1 < clusters.size()) ? (clusters[i + 1].mean_z - clusters[i].mean_z) : 999.0;
        bool is_major = (clusters[i].all_z.size() >= std::max<size_t>(2, max_cl_cnt / 5)) || (std::min(lgap, rgap) > 0.60);
        if (is_major && (major_floors.empty() || std::abs(clusters[i].mean_z - major_floors.back().plateau_z) > 1.80)) {
            clusters[i].plateau_z = clusters[i].mean_z;
            major_floors.push_back(clusters[i]);
        } else {
            stair_clusters.push_back(clusters[i]);
        }
    }
    if (major_floors.empty()) {
        for (auto& c : clusters) c.plateau_z = c.mean_z;
        major_floors = clusters;
        stair_clusters.clear();
    }
    for (const auto& sc : stair_clusters) {
        int below_i = -1, above_i = -1;
        for (int i = 0; i < static_cast<int>(major_floors.size()); ++i) {
            if (major_floors[static_cast<size_t>(i)].plateau_z <= sc.mean_z) below_i = i;
            if (major_floors[static_cast<size_t>(i)].plateau_z > sc.mean_z && above_i < 0) above_i = i;
        }
        if (below_i >= 0 && above_i >= 0) {
            double mid_z = 0.5 * (major_floors[static_cast<size_t>(below_i)].plateau_z + major_floors[static_cast<size_t>(above_i)].plateau_z);
            for (double z : sc.all_z) {
                if (z < mid_z) major_floors[static_cast<size_t>(below_i)].all_z.push_back(z);
                else major_floors[static_cast<size_t>(above_i)].all_z.push_back(z);
            }
        } else if (below_i >= 0) {
            major_floors[static_cast<size_t>(below_i)].all_z.insert(
                major_floors[static_cast<size_t>(below_i)].all_z.end(), sc.all_z.begin(), sc.all_z.end());
        } else if (above_i >= 0) {
            major_floors[static_cast<size_t>(above_i)].all_z.insert(
                major_floors[static_cast<size_t>(above_i)].all_z.end(), sc.all_z.begin(), sc.all_z.end());
        }
    }

    std::vector<int> station_floor_ids(e57_stations.size(), 1);
    nlohmann::json floors_json = nlohmann::json::array();

    for (size_t f_i = 0; f_i < major_floors.size(); ++f_i) {
        int fid = static_cast<int>(f_i + 1);
        const auto& mf = major_floors[f_i];
        double fmin = *std::min_element(mf.all_z.begin(), mf.all_z.end());
        double fmax = *std::max_element(mf.all_z.begin(), mf.all_z.end());
        std::vector<Eigen::Vector2d> st_xys;
        for (size_t s_i = 0; s_i < e57_stations.size(); ++s_i) {
            for (double z : mf.all_z) {
                if (std::abs(e57_stations[s_i].pos_world.z() - z) < 1e-4) {
                    station_floor_ids[s_i] = fid;
                    st_xys.emplace_back(e57_stations[s_i].pos_world.x(), e57_stations[s_i].pos_world.y());
                    break;
                }
            }
        }
        std::string kp_rel = "floors/floor_" + std::to_string(fid) + "_keyplan.png";
        nlohmann::json wb = renderFloorKeyplanPng(cloud, mf.plateau_z, (fs::path(output_dir) / kp_rel).string(), st_xys);

        double floor_elev = (!wb.is_null() && wb.contains("slab_z_m"))
            ? wb["slab_z_m"].get<double>()
            : (mf.plateau_z - config.sensor_eye_height_m);

        nlohmann::json fl_obj = {
            {"id", fid},
            {"name", "Level " + std::to_string(fid)},
            {"min_z", std::round(fmin * 10000.0) / 10000.0},
            {"max_z", std::round(fmax * 10000.0) / 10000.0},
            {"mean_z", std::round(mf.plateau_z * 10000.0) / 10000.0},
            {"floor_elevation", std::round(floor_elev * 10000.0) / 10000.0},
            {"keyplan_path", kp_rel}
        };
        if (!wb.is_null() && !wb.empty()) fl_obj["world_bounds"] = wb;
        floors_json.push_back(fl_obj);
    }

    // 4. Wall-occlusion-checked 8-sector + staircase symmetric navigation graph
    std::unordered_map<VoxelCoord, int, VoxelCoordHash> vox_counts;
    constexpr double nav_vs = 0.14;
    size_t occ_stride = std::max<size_t>(1, cloud.size() / 250000);
    for (size_t i = 0; i < cloud.size(); i += occ_stride) {
        VoxelCoord vc{
            static_cast<int64_t>(std::floor(cloud[i].x / nav_vs)),
            static_cast<int64_t>(std::floor(cloud[i].y / nav_vs)),
            static_cast<int64_t>(std::floor(cloud[i].z / nav_vs))
        };
        vox_counts[vc] += 1;
    }
    int min_occ_pts = (cloud.size() / occ_stride > 25000) ? 2 : 1;

    auto hasLineOfSight = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> bool {
        Eigen::Vector3d vec = b - a;
        double dist = vec.norm();
        if (dist <= 0.90) return true;
        int steps = std::max(4, static_cast<int>(dist / (nav_vs * 0.75)));
        int hits = 0;
        for (int k = 0; k < steps; ++k) {
            double t = (0.40 / dist) + (1.0 - 0.80 / dist) * (static_cast<double>(k) / std::max(1, steps - 1));
            Eigen::Vector3d p = a + t * vec;
            VoxelCoord vc{
                static_cast<int64_t>(std::floor(p.x() / nav_vs)),
                static_cast<int64_t>(std::floor(p.y() / nav_vs)),
                static_cast<int64_t>(std::floor(p.z() / nav_vs))
            };
            auto it = vox_counts.find(vc);
            if (it != vox_counts.end() && it->second >= min_occ_pts) {
                if (++hits >= 2) return false;
            }
        }
        return true;
    };

    std::vector<std::set<std::string>> nav_sets(e57_stations.size());
    for (size_t i = 0; i < e57_stations.size(); ++i) {
        for (int sec = 0; sec < 8; ++sec) {
            double best_d = 1e18;
            int best_j = -1;
            for (size_t j = 0; j < e57_stations.size(); ++j) {
                if (i == j || station_floor_ids[i] != station_floor_ids[j]) continue;
                Eigen::Vector2d dxy = e57_stations[j].pos_world.head<2>() - e57_stations[i].pos_world.head<2>();
                double d = dxy.norm();
                if (d < 0.05 || d > 12.0) continue;
                double ang = std::atan2(dxy.y(), dxy.x());
                int s_id = static_cast<int>(std::floor((ang + kPi + (kPi / 8.0)) / (kPi / 4.0))) % 8;
                if (s_id < 0) s_id += 8;
                if (s_id == sec && d < best_d && hasLineOfSight(e57_stations[i].pos_world, e57_stations[j].pos_world)) {
                    best_d = d;
                    best_j = static_cast<int>(j);
                }
            }
            if (best_j >= 0) {
                nav_sets[i].insert(e57_stations[static_cast<size_t>(best_j)].pano_id);
                nav_sets[static_cast<size_t>(best_j)].insert(e57_stations[i].pano_id);
            }
        }
    }

    // Cross-floor staircase transitions
    std::vector<bool> has_stair_link(e57_stations.size(), false);
    for (size_t i = 0; i < e57_stations.size(); ++i) {
        for (size_t j = i + 1; j < e57_stations.size(); ++j) {
            if (station_floor_ids[i] == station_floor_ids[j]) continue;
            double dz = std::abs(e57_stations[i].pos_world.z() - e57_stations[j].pos_world.z());
            double dxy = (e57_stations[i].pos_world.head<2>() - e57_stations[j].pos_world.head<2>()).norm();
            if (dz >= 0.10 && dz <= 2.20 && dxy <= 5.50) {
                nav_sets[i].insert(e57_stations[j].pano_id);
                nav_sets[j].insert(e57_stations[i].pano_id);
                has_stair_link[i] = true;
                has_stair_link[j] = true;
            }
        }
    }
    // Fallback for consecutive floors without an intermediate mid-stair landing station:
    // link the single closest horizontal pair (<= 6.5m XY, <= 5.0m Z) matching AV Showcase v2.0
    for (size_t f_i = 0; f_i + 1 < major_floors.size(); ++f_i) {
        const int fid_a = static_cast<int>(f_i + 1);
        const int fid_b = static_cast<int>(f_i + 2);
        bool already_linked = false;
        size_t best_a = 0, best_b = 0;
        double best_dxy = 1e18;
        bool found_pair = false;
        for (size_t i = 0; i < e57_stations.size(); ++i) {
            if (station_floor_ids[i] != fid_a) continue;
            for (size_t j = 0; j < e57_stations.size(); ++j) {
                if (station_floor_ids[j] != fid_b) continue;
                if (nav_sets[i].count(e57_stations[j].pano_id) > 0) {
                    already_linked = true;
                    break;
                }
                const double dz = std::abs(e57_stations[i].pos_world.z() - e57_stations[j].pos_world.z());
                const double dxy = (e57_stations[i].pos_world.head<2>() - e57_stations[j].pos_world.head<2>()).norm();
                if (dz <= 5.00 && dxy <= 6.50 && dxy < best_dxy) {
                    best_dxy = dxy;
                    best_a = i;
                    best_b = j;
                    found_pair = true;
                }
            }
            if (already_linked) break;
        }
        if (!already_linked && found_pair) {
            nav_sets[best_a].insert(e57_stations[best_b].pano_id);
            nav_sets[best_b].insert(e57_stations[best_a].pano_id);
            has_stair_link[best_a] = true;
            has_stair_link[best_b] = true;
        }
    }

    // 5. Write Structured E57 & lod0.bin & stations_manifest.json & av_walkthrough_bundle.json
    if (config.export_e57) {
        result.structured_e57_path = (fs::path(output_dir) / "structured_stations.e57").string();
        if (!writeStructuredE57File(result.structured_e57_path, e57_stations, result.e57_size_bytes)) {
            result.error_message = "Failed to write structured_stations.e57.";
            return result;
        }
    }

    // Write stations_manifest.json matching av-scanar / AV Structure v1
    {
        nlohmann::json st_manifest_list = nlohmann::json::array();
        for (const auto& s : e57_stations) {
            st_manifest_list.push_back({
                {"station_id", s.station_id},
                {"selection_mode", s.selection_mode},
                {"scan_guid", s.scan_guid},
                {"image_guid", s.image_guid},
                {"position_world_m", {
                    std::round(s.pos_world.x() * 1000000.0) / 1000000.0,
                    std::round(s.pos_world.y() * 1000000.0) / 1000000.0,
                    std::round(s.pos_world.z() * 1000000.0) / 1000000.0
                }},
                {"yaw_rad", std::round(s.yaw_rad * 1000000.0) / 1000000.0},
                {"quaternion_wxyz_world", {s.quat_wxyz[0], s.quat_wxyz[1], s.quat_wxyz[2], s.quat_wxyz[3]}},
                {"panorama_width", s.pano_w},
                {"panorama_height", s.pano_h},
                {"structured_scan_rows", s.scan_rows},
                {"structured_scan_cols", s.scan_cols},
                {"structured_scan_valid_points", s.valid_scan_points},
                {"provenance_fractions", {
                    {"photo_rgb_fraction", std::round(s.photo_rgb_ratio * 10000.0) / 10000.0},
                    {"pointcloud_rgb_fraction", std::round(s.pointcloud_rgb_ratio * 10000.0) / 10000.0},
                    {"depth_fill_fraction", std::round(s.depth_fill_ratio * 10000.0) / 10000.0},
                    {"geometry_uncolored_fraction", std::round(s.geometry_uncolored_ratio * 10000.0) / 10000.0},
                    {"unobserved_fraction", std::round(s.unobserved_ratio * 10000.0) / 10000.0}
                }},
                {"files", {
                    {"panorama_rgb", "stations/" + s.station_id + "/panorama_rgb.jpg"},
                    {"depth_map_mm", "stations/" + s.station_id + "/depth_map_mm.png"},
                    {"coverage_mask", "stations/" + s.station_id + "/coverage_mask.png"},
                    {"provenance_code", "stations/" + s.station_id + "/provenance_code.png"},
                    {"structured_scan_ply", "stations/" + s.station_id + "/structured_scan.ply"}
                }}
            });
        }
        nlohmann::json stations_manifest = {
            {"schema", "ArtificialVision.AVStructure.StationsManifest.v1"},
            {"generated_utc", currentUtcIsoString()},
            {"station_count", e57_stations.size()},
            {"structured_e57_path", config.export_e57 ? nlohmann::json("structured_stations.e57") : nlohmann::json(nullptr)},
            {"e57_size_bytes", result.e57_size_bytes},
            {"non_fabrication_policy",
             "Unobserved spherical angles remain black (0,0,0) with depth=0 and provenance=0 (UNOBSERVED). "
             "Only <=1px pinholes on continuous surfaces (<8cm depth delta) are filled."},
            {"stations", st_manifest_list}
        };
        result.stations_manifest_path = (fs::path(output_dir) / "stations_manifest.json").string();
        std::ofstream smf(result.stations_manifest_path);
        smf << stations_manifest.dump(2);
        smf.close();
    }

    result.lod0_bin_path = (fs::path(output_dir) / "cache" / "lod0.bin").string();
    result.lod0_points_written = writeLod0Bin(result.lod0_bin_path, cloud);

    nlohmann::json panos_json = nlohmann::json::array();
    for (size_t i = 0; i < e57_stations.size(); ++i) {
        const auto& s = e57_stations[i];
        nlohmann::json nbrs = nlohmann::json::array();
        for (const auto& nid : nav_sets[i]) nbrs.push_back(nid);
        panos_json.push_back({
            {"id", s.pano_id},
            {"station_id", s.station_id},
            {"station_guid", s.scan_guid},
            {"position", {
                std::round(s.pos_world.x() * 1000000.0) / 1000000.0,
                std::round(s.pos_world.y() * 1000000.0) / 1000000.0,
                std::round(s.pos_world.z() * 1000000.0) / 1000000.0
            }},
            {"rotation", {s.quat_xyzw[0], s.quat_xyzw[1], s.quat_xyzw[2], s.quat_xyzw[3]}},
            {"is_level", true},
            {"heading_yaw_rad", std::round(s.yaw_rad * 1000000.0) / 1000000.0},
            {"floor_id", station_floor_ids[i]},
            {"image_path", "panos/" + s.pano_id + ".jpg"},
            {"depth_path", "panos/" + s.pano_id + "_depth.png"},
            {"provenance", s.provenance},
            {"valid_pixel_ratio", std::round(s.valid_pixel_ratio * 10000.0) / 10000.0},
            {"stair_link", static_cast<bool>(has_stair_link[i])},
            {"neighbors", nbrs}
        });
    }

    nlohmann::json tf_flat = nlohmann::json::array();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            tf_flat.push_back(std::round(T_proj(r, c) * 100000000.0) / 100000000.0);
        }
    }

    std::string scan_date = config.scan_date.empty() ? currentLocalDateString(false) : config.scan_date;
    std::string scan_id = config.scan_id.empty() ? fs::path(output_dir).filename().string() : config.scan_id;
    if (scan_id.empty()) scan_id = currentLocalDateString(true);
    std::string proj_id = config.project_id.empty() ? makeDeterministicUuid("av-proj-" + config.project_name, false) : config.project_id;

    nlohmann::json bundle_json = {
        {"format", "av_spatial_bundle"},
        {"schema", "av_walkthrough_bundle"},
        {"schema_version", "1.1"},
        {"producer", {{"app", "AV Scan"}, {"version", "1.1.0"}}},
        {"generator", "AV Scan (C++17)"},
        {"generated_utc", currentUtcIsoString()},
        {"project_id", proj_id},
        {"project_name", config.project_name},
        {"scan_id", scan_id},
        {"scan_date", scan_date},
        {"coordinate_system", {
            {"handedness", "right"},
            {"up", "+Z"},
            {"units", "meters"},
            {"pano_projection", "ASTM_E2807_equirectangular"}
        }},
        {"panorama_projection", "astm_e2807_equirectangular_u05_pos_x"},
        {"transform_to_project", tf_flat},
        {"registration_metrics", {
            {"rmse_m", std::round(result.registration_rmse_m * 10000.0) / 10000.0},
            {"overlap_ratio", std::round(result.registration_overlap * 10000.0) / 10000.0}
        }},
        {"structured_e57_path", config.export_e57 ? nlohmann::json("structured_stations.e57") : nlohmann::json(nullptr)},
        {"point_cloud", {{"lod0", "cache/lod0.bin"}, {"format", "av_lod0_v1"}}},
        {"point_cloud_cache", "cache/lod0.bin"},
        {"point_cloud_lod0_points", result.lod0_points_written},
        {"nav_graph", "los_v1"},
        {"floors", floors_json},
        {"panos", panos_json}
    };

    result.bundle_json_path = (fs::path(output_dir) / "av_walkthrough_bundle.json").string();
    std::ofstream bf(result.bundle_json_path);
    bf << bundle_json.dump(2);
    bf.close();

    if (!config.av_showcase_project_dir.empty()) {
        double align_rmse = result.registration_rmse_m;
        double align_ov = result.registration_overlap;
        result.avproj_path = exportOrAppendAvShowcaseProject(
            config.av_showcase_project_dir,
            output_dir,
            reference_baseline_points == nullptr,
            &align_rmse,
            &align_ov);
        if (reference_baseline_points == nullptr) {
            result.registration_rmse_m = align_rmse;
            result.registration_overlap = align_ov;
        }
    }

    // 6. Export standardized av-scanar point clouds, trajectories, flash anchors, and scan_metadata.json
    if (config.export_standardized_clouds) {
        // Trajectory CSV & TUM
        result.trajectory_csv_path = (fs::path(output_dir) / "trajectory.csv").string();
        result.trajectory_tum_path = (fs::path(output_dir) / "trajectory_tum.txt").string();
        {
            std::ofstream csv_f(result.trajectory_csv_path);
            std::ofstream tum_f(result.trajectory_tum_path);
            csv_f << "timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw,yaw_rad\n";
            csv_f << std::fixed << std::setprecision(6);
            tum_f << std::fixed << std::setprecision(6);

            for (size_t ti = 0; ti < trajectory.size(); ++ti) {
                const auto& tp = trajectory[ti];
                Eigen::Vector3d p(tp.position.x, tp.position.y, tp.position.z);
                if (apply_pose_tf) p = T_pose.block<3, 3>(0, 0) * p + T_pose.block<3, 1>(0, 3);
                double qx = tp.orientation.x, qy = tp.orientation.y, qz = tp.orientation.z, qw = tp.orientation.w;
                double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)) + tf_yaw;
                double t_sec = (tp.timestamp_ns > 0)
                    ? (static_cast<double>(tp.timestamp_ns) * 1e-9)
                    : (static_cast<double>(ti) * 0.1);
                csv_f << t_sec << "," << p.x() << "," << p.y() << "," << p.z() << ","
                      << qx << "," << qy << "," << qz << "," << qw << "," << yaw << "\n";
                tum_f << t_sec << " " << p.x() << " " << p.y() << " " << p.z() << " "
                      << qx << " " << qy << " " << qz << " " << qw << "\n";
            }
        }

        // Flash anchors JSON
        {
            nlohmann::json fa_list = nlohmann::json::array();
            for (size_t i = 0; i < flash_anchors.size(); ++i) {
                const auto& fa = flash_anchors[i];
                Eigen::Vector3d p(fa.position.x, fa.position.y, fa.position.z);
                if (apply_pose_tf) p = T_pose.block<3, 3>(0, 0) * p + T_pose.block<3, 1>(0, 3);
                fa_list.push_back({
                    {"id", static_cast<int>(i + 1)},
                    {"label", fa.label.empty() ? ("flash_" + std::to_string(i)) : fa.label},
                    {"position", {p.x(), p.y(), p.z()}},
                    {"yaw_rad", fa.yaw_rad + tf_yaw}
                });
            }
            nlohmann::json fa_doc = {{"flash_anchors", fa_list}};
            std::ofstream faf((fs::path(output_dir) / "flash_anchors.json").string());
            faf << fa_doc.dump(2);
        }

        // Standardized Point Clouds:
        //   - pointcloud_airy.ply / .pcd (Pure XYZ geometry)
        //   - pointcloud_stereo.ply / .pcd (Stereo/Camera colored XYZ+RGB; empty 0-point cloud if LiDAR-only)
        //   - pointcloud_fused_rgb.ply / .pcd (Unified XYZ+RGB; true RGB or B&W grayscale when LiDAR-only)
        const std::vector<schemas::PointXYZI>* airy_src = config.airy_points ? config.airy_points : &cloud;
        std::vector<schemas::PointXYZI> empty_stereo;
        std::vector<schemas::PointXYZI> derived_stereo;
        const std::vector<schemas::PointXYZI>* stereo_src = &empty_stereo;
        if (config.stereo_points) {
            stereo_src = config.stereo_points;
        } else if (!result.is_monochrome_lidar_only) {
            derived_stereo.reserve(cloud.size());
            for (size_t idx = 0; idx < cloud.size(); ++idx) {
                if (idx < pt_has_true_rgb.size() && pt_has_true_rgb[idx]) {
                    derived_stereo.push_back(cloud[idx]);
                }
            }
            stereo_src = &derived_stereo;
        }

        result.pointcloud_airy_ply_path = (fs::path(output_dir) / "pointcloud_airy.ply").string();
        result.pointcloud_airy_pcd_path = (fs::path(output_dir) / "pointcloud_airy.pcd").string();
        writeStandardizedPlyAscii(result.pointcloud_airy_ply_path, *airy_src, false);
        writeStandardizedPcdAscii(result.pointcloud_airy_pcd_path, *airy_src, false);

        result.pointcloud_stereo_ply_path = (fs::path(output_dir) / "pointcloud_stereo.ply").string();
        result.pointcloud_stereo_pcd_path = (fs::path(output_dir) / "pointcloud_stereo.pcd").string();
        writeStandardizedPlyAscii(result.pointcloud_stereo_ply_path, *stereo_src, true);
        writeStandardizedPcdAscii(result.pointcloud_stereo_pcd_path, *stereo_src, true);

        result.pointcloud_fused_rgb_ply_path = (fs::path(output_dir) / "pointcloud_fused_rgb.ply").string();
        result.pointcloud_fused_rgb_pcd_path = (fs::path(output_dir) / "pointcloud_fused_rgb.pcd").string();
        writeStandardizedPlyAscii(result.pointcloud_fused_rgb_ply_path, cloud, true);
        writeStandardizedPcdAscii(result.pointcloud_fused_rgb_pcd_path, cloud, true);

        size_t cslam_pts_cnt = 0;
        if (config.cslam_peers_count > 0 && config.cslam_merged_points && !config.cslam_merged_points->empty()) {
            cslam_pts_cnt = config.cslam_merged_points->size();
            writeStandardizedPlyAscii(
                (fs::path(output_dir) / "pointcloud_cslam_merged.ply").string(),
                *config.cslam_merged_points,
                true);
        }

        schemas::Vector3D bbox_min{}, bbox_max{};
        PointCloudProcessor::computeBoundingBox(cloud, bbox_min, bbox_max);
        double dx = std::max(0.0, bbox_max.x - bbox_min.x);
        double dy = std::max(0.0, bbox_max.y - bbox_min.y);
        double dz = std::max(0.0, bbox_max.z - bbox_min.z);

        nlohmann::json scan_meta = {
            {"session_id", scan_id},
            {"export_timestamp", currentUtcIsoString()},
            {"capture_duration_seconds", std::round(config.capture_duration_seconds * 100.0) / 100.0},
            {"airy_points", airy_src->size()},
            {"stereo_points", stereo_src->size()},
            {"fused_points", cloud.size()},
            {"is_monochrome_lidar_only", result.is_monochrome_lidar_only},
            {"cslam_peers", config.cslam_peers_count},
            {"cslam_merged_points", cslam_pts_cnt},
            {"rgb_keyframes_saved", keyframes.size()},
            {"virtual_stations_count", e57_stations.size()},
            {"floor_count", floors_json.size()},
            {"structured_e57_file", config.export_e57 ? nlohmann::json("structured_stations.e57") : nlohmann::json(nullptr)},
            {"av_walkthrough_bundle_file", "av_walkthrough_bundle.json"},
            {"av_showcase_avproj_path", result.avproj_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(result.avproj_path)},
            {"depth_range_gate_m", {config.min_range_m, config.max_range_m}},
            {"surface_agreement_tolerance_m", 0.15},
            {"spatial_bounding_box", {
                {"min_m", {bbox_min.x, bbox_min.y, bbox_min.z}},
                {"max_m", {bbox_max.x, bbox_max.y, bbox_max.z}},
                {"dimensions_m", {dx, dy, dz}},
                {"volume_m3", std::round(dx * dy * dz * 100.0) / 100.0}
            }}
        };
        result.scan_metadata_path = (fs::path(output_dir) / "scan_metadata.json").string();
        std::ofstream sm_ofs(result.scan_metadata_path);
        sm_ofs << scan_meta.dump(2);
        sm_ofs.close();
    }

    result.station_count = e57_stations.size();
    result.floor_count = floors_json.size();
    result.success = true;
    result.export_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

std::string StructuredWalkthroughExporter::exportOrAppendAvShowcaseProject(
    const std::string& av_showcase_project_dir,
    const std::string& bundle_dir,
    bool auto_align_to_baseline,
    double* out_rmse_m,
    double* out_overlap_ratio) {
    fs::create_directories(fs::path(av_showcase_project_dir) / "logs");
    fs::path bundle_file = fs::path(bundle_dir) / "av_walkthrough_bundle.json";
    if (!fs::exists(bundle_file)) return "";

    nlohmann::json bundle;
    {
        std::ifstream ifs(bundle_file);
        ifs >> bundle;
    }

    fs::path avproj_path = fs::path(av_showcase_project_dir) / "project.avproj";
    nlohmann::json proj_data;
    if (fs::exists(avproj_path)) {
        std::ifstream ifs(avproj_path);
        ifs >> proj_data;
        if (!proj_data.contains("schema_version")) proj_data["schema_version"] = "1.1";
        if (!proj_data.contains("application_version")) proj_data["application_version"] = "2.0.0";
        if (!proj_data.contains("project_id")) {
            proj_data["project_id"] = proj_data.value("id", bundle.value("project_id", "av_project"));
        }
    } else {
        std::string pid = bundle.value("project_id", "av_project");
        proj_data = {
            {"schema_version", "1.1"},
            {"application_version", "2.0.0"},
            {"project_id", pid},
            {"id", pid},
            {"name", bundle.value("project_name", "AV Scan Project")},
            {"description", ""},
            {"settings", {{"max_autostack_distance", 10.0}, {"default_unit", "metric"}}},
            {"bookmarks", nlohmann::json::array()},
            {"scans", nlohmann::json::array()}
        };
    }

    std::string base_date = bundle.value("scan_date", currentLocalDateString(false));
    std::string scan_id = bundle.value("scan_id", base_date);
    std::set<std::string> existing_keys;
    for (const auto& s : proj_data["scans"]) {
        if (s.contains("scan_date")) existing_keys.insert(s["scan_date"].get<std::string>());
    }
    std::string scan_key = (existing_keys.find(base_date) == existing_keys.end()) ? base_date : scan_id;

    fs::path dest_scan_dir = fs::path(av_showcase_project_dir) / "Scans" / scan_key;
    fs::create_directories(dest_scan_dir / "panos");
    fs::create_directories(dest_scan_dir / "floors");
    fs::create_directories(dest_scan_dir / "cache");

    std::string lod0_rel_src = bundle.value("point_cloud_cache", "cache/lod0.bin");
    if (bundle.contains("point_cloud") && bundle["point_cloud"].is_object() && bundle["point_cloud"].contains("lod0")) {
        lod0_rel_src = bundle["point_cloud"]["lod0"].get<std::string>();
    }
    std::vector<schemas::PointXYZI> src_pts;
    readLod0Bin((fs::path(bundle_dir) / lod0_rel_src).string(), src_pts);

    nlohmann::json panos_copy = bundle["panos"];
    nlohmann::json floors_copy = bundle["floors"];
    Eigen::Matrix4d T_align = Eigen::Matrix4d::Identity();

    const nlohmann::json* base_scan_ptr = nullptr;
    if (!proj_data["scans"].empty()) {
        std::string baseline_id = proj_data.value("baseline_scan_id", "");
        if (!baseline_id.empty()) {
            for (const auto& s : proj_data["scans"]) {
                if (s.value("scan_id", "") == baseline_id || s.value("scan_date", "") == baseline_id) {
                    base_scan_ptr = &s;
                    break;
                }
            }
        }
        if (!base_scan_ptr) {
            for (const auto& s : proj_data["scans"]) {
                if (!base_scan_ptr || s.value("scan_date", "") < base_scan_ptr->value("scan_date", "")) {
                    base_scan_ptr = &s;
                }
            }
        }
    }

    if (auto_align_to_baseline && base_scan_ptr != nullptr &&
        base_scan_ptr->value("scan_date", "") != scan_key) {
        std::string ref_rel = base_scan_ptr->value("point_cloud_cache", "");
        fs::path ref_bin = fs::path(av_showcase_project_dir) / ref_rel;
        std::vector<schemas::PointXYZI> ref_pts;
        if (!ref_rel.empty() && readLod0Bin(ref_bin.string(), ref_pts) && ref_pts.size() >= 20 && src_pts.size() >= 20) {
            double rmse = 0.0, overlap = 0.0;
            auto tf_arr = computeTransformToProjectIcp(src_pts, ref_pts, rmse, overlap);
            if (out_rmse_m) *out_rmse_m = rmse;
            if (out_overlap_ratio) *out_overlap_ratio = overlap;
            T_align = arrayToMatrix4d(tf_arr);
            if (overlap >= 0.15 && !(T_align - Eigen::Matrix4d::Identity()).isZero(1e-5)) {
                transformPointCloudInPlace(src_pts, T_align);
                Eigen::Quaterniond q_tf(T_align.block<3, 3>(0, 0));
                for (auto& p : panos_copy) {
                    Eigen::Vector3d pos(p["position"][0], p["position"][1], p["position"][2]);
                    Eigen::Vector3d pos_new = T_align.block<3, 3>(0, 0) * pos + T_align.block<3, 1>(0, 3);
                    Eigen::Quaterniond q_old(p["rotation"][3], p["rotation"][0], p["rotation"][1], p["rotation"][2]);
                    Eigen::Quaterniond q_new = (q_tf * q_old).normalized();
                    p["position"] = {pos_new.x(), pos_new.y(), pos_new.z()};
                    p["rotation"] = {q_new.x(), q_new.y(), q_new.z(), q_new.w()};
                }
                const double tz_shift = T_align(2, 3);
                for (auto& fl : floors_copy) {
                    for (const char* zk : {"min_z", "max_z", "mean_z", "floor_elevation"}) {
                        if (fl.contains(zk) && fl[zk].is_number()) {
                            fl[zk] = std::round((fl[zk].get<double>() + tz_shift) * 10000.0) / 10000.0;
                        }
                    }
                }
            }
        }
    }

    std::error_code ec;
    for (const auto& p : panos_copy) {
        std::string img_rel = p.value("image_path", "");
        if (!img_rel.empty() && fs::exists(fs::path(bundle_dir) / img_rel)) {
            fs::copy_file(fs::path(bundle_dir) / img_rel, dest_scan_dir / img_rel, fs::copy_options::overwrite_existing, ec);
        }
        std::string dep_rel = p.value("depth_path", "");
        if (!dep_rel.empty() && fs::exists(fs::path(bundle_dir) / dep_rel)) {
            fs::copy_file(fs::path(bundle_dir) / dep_rel, dest_scan_dir / dep_rel, fs::copy_options::overwrite_existing, ec);
        }
    }

    bool re_render_kp = !(T_align - Eigen::Matrix4d::Identity()).isZero(1e-5) && !src_pts.empty();
    for (auto& fl : floors_copy) {
        int fid = fl.value("id", 1);
        std::string kp_rel = fl.value("keyplan_path", "floors/floor_" + std::to_string(fid) + "_keyplan.png");
        fs::path dst_kp = dest_scan_dir / kp_rel;
        if (re_render_kp) {
            std::vector<Eigen::Vector2d> st_xys;
            for (const auto& p : panos_copy) {
                if (p.value("floor_id", 1) == fid) st_xys.emplace_back(p["position"][0], p["position"][1]);
            }
            nlohmann::json wb = renderFloorKeyplanPng(src_pts, fl.value("mean_z", 1.55), dst_kp.string(), st_xys);
            if (!wb.is_null() && !wb.empty()) {
                fl["world_bounds"] = wb;
                if (wb.contains("slab_z_m")) {
                    fl["floor_elevation"] = std::round(wb["slab_z_m"].get<double>() * 10000.0) / 10000.0;
                }
            }
        } else if (fs::exists(fs::path(bundle_dir) / kp_rel)) {
            fs::copy_file(fs::path(bundle_dir) / kp_rel, dst_kp, fs::copy_options::overwrite_existing, ec);
        }
        fl["keyplan_path"] = kp_rel;
        fl["floorplan_path"] = "Scans/" + scan_key + "/" + kp_rel;
    }

    writeLod0Bin((dest_scan_dir / "cache" / "lod0.bin").string(), src_pts);

    nlohmann::json tf_flat = nlohmann::json::array();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            tf_flat.push_back(std::round(T_align(r, c) * 100000000.0) / 100000000.0);
        }
    }

    nlohmann::json scan_entry = {
        {"scan_id", scan_id},
        {"scan_date", scan_key},
        {"calendar_date", base_date},
        {"scan_dir", "Scans/" + scan_key},
        {"description", "AV Scan Capture (" + scan_id + ")"},
        {"import_mode", "managed"},
        {"source_path", ""},
        {"point_cloud_cache", "Scans/" + scan_key + "/cache/lod0.bin"},
        {"transform_to_project", tf_flat},
        {"nav_graph", "los_v1"},
        {"floors", floors_copy},
        {"panos", panos_copy}
    };

    // Write companion av_walkthrough_bundle.json inside Scans/<scan_key>/ matching AV Showcase v2.0
    nlohmann::json scan_bundle = {
        {"format", "av_spatial_bundle"},
        {"schema_version", "1.1"},
        {"producer", {{"app", "AV Scan"}, {"version", "1.1.0"}}},
        {"project_id", proj_data.value("project_id", "av_project")},
        {"scan_id", scan_id},
        {"scan_date", scan_key},
        {"coordinate_system", {
            {"handedness", "right"},
            {"up", "+Z"},
            {"units", "meters"},
            {"pano_projection", "ASTM_E2807_equirectangular"}
        }},
        {"transform_to_project", tf_flat},
        {"point_cloud", {{"lod0", "cache/lod0.bin"}, {"format", "av_lod0_v1"}}},
        {"point_cloud_cache", "cache/lod0.bin"},
        {"nav_graph", "los_v1"},
        {"floors", floors_copy},
        {"panos", panos_copy}
    };
    {
        std::ofstream sbf(dest_scan_dir / "av_walkthrough_bundle.json");
        sbf << scan_bundle.dump(2);
    }

    nlohmann::json new_scans = nlohmann::json::array();
    for (const auto& s : proj_data["scans"]) {
        if (s.value("scan_date", "") != scan_key) new_scans.push_back(s);
    }
    new_scans.push_back(scan_entry);
    std::sort(new_scans.begin(), new_scans.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
        return a.value("scan_date", "") < b.value("scan_date", "");
    });
    proj_data["scans"] = new_scans;
    if (!proj_data.contains("baseline_scan_id") || proj_data["baseline_scan_id"].get<std::string>().empty()) {
        proj_data["baseline_scan_id"] = scan_id;
    }

    std::ofstream ofs(avproj_path);
    ofs << proj_data.dump(2);
    ofs.close();
    return avproj_path.string();
}

} // namespace av::core::processing

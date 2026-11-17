#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace av::core::terrestrial {

namespace math {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

struct Vec2d {
    double x{0.0};
    double y{0.0};

    constexpr Vec2d() = default;
    constexpr Vec2d(double x_, double y_) : x(x_), y(y_) {}
};

struct Vec3d {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vec3d() = default;
    constexpr Vec3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double operator[](int idx) const { return idx == 0 ? x : (idx == 1 ? y : z); }
    double& operator[](int idx) { return idx == 0 ? x : (idx == 1 ? y : z); }

    Vec3d operator+(const Vec3d& o) const { return Vec3d(x + o.x, y + o.y, z + o.z); }
    Vec3d operator-(const Vec3d& o) const { return Vec3d(x - o.x, y - o.y, z - o.z); }
    Vec3d operator-() const { return Vec3d(-x, -y, -z); }
    Vec3d operator*(double s) const { return Vec3d(x * s, y * s, z * s); }
    Vec3d operator/(double s) const { return Vec3d(x / s, y / s, z / s); }

    Vec3d& operator+=(const Vec3d& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3d& operator-=(const Vec3d& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    double dot(const Vec3d& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3d cross(const Vec3d& o) const {
        return Vec3d(y * o.z - z * o.y,
                     z * o.x - x * o.z,
                     x * o.y - y * o.x);
    }
    double squaredNorm() const { return x * x + y * y + z * z; }
    double norm() const { return std::sqrt(squaredNorm()); }
    Vec3d normalized() const {
        double n = norm();
        return (n > 1e-12) ? (*this / n) : Vec3d(0.0, 0.0, 0.0);
    }
};

struct Mat3d {
    double m[3][3]{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    static Mat3d Identity() {
        return Mat3d{};
    }

    static Mat3d Zero() {
        Mat3d z;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                z.m[r][c] = 0.0;
        return z;
    }

    double& operator()(int r, int c) { return m[r][c]; }
    double operator()(int r, int c) const { return m[r][c]; }

    Vec3d operator*(const Vec3d& v) const {
        return Vec3d(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }

    Mat3d operator*(const Mat3d& o) const {
        Mat3d res = Zero();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                res.m[r][c] = m[r][0] * o.m[0][c] +
                              m[r][1] * o.m[1][c] +
                              m[r][2] * o.m[2][c];
            }
        }
        return res;
    }

    Mat3d transpose() const {
        Mat3d t;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                t.m[r][c] = m[c][r];
        return t;
    }

    double determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
};

struct Quatd {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Quatd() = default;
    constexpr Quatd(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

    static Quatd Identity() { return Quatd(1.0, 0.0, 0.0, 0.0); }

    static Quatd FromEuler(double roll_rad, double pitch_rad, double yaw_rad) {
        double cr = std::cos(roll_rad * 0.5);
        double sr = std::sin(roll_rad * 0.5);
        double cp = std::cos(pitch_rad * 0.5);
        double sp = std::sin(pitch_rad * 0.5);
        double cy = std::cos(yaw_rad * 0.5);
        double sy = std::sin(yaw_rad * 0.5);

        Quatd q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;
        return q.normalized();
    }

    Quatd normalized() const {
        double n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n <= 1e-12) return Identity();
        return Quatd(w / n, x / n, y / n, z / n);
    }

    Mat3d toRotationMatrix() const {
        Quatd q = normalized();
        Mat3d R;
        double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        R(0, 0) = 1.0 - 2.0 * (yy + zz);
        R(0, 1) = 2.0 * (xy - wz);
        R(0, 2) = 2.0 * (xz + wy);

        R(1, 0) = 2.0 * (xy + wz);
        R(1, 1) = 1.0 - 2.0 * (xx + zz);
        R(1, 2) = 2.0 * (yz - wx);

        R(2, 0) = 2.0 * (xz - wy);
        R(2, 1) = 2.0 * (yz + wx);
        R(2, 2) = 1.0 - 2.0 * (xx + yy);
        return R;
    }
};

struct SO3d {
    Mat3d R{Mat3d::Identity()};

    SO3d() = default;
    explicit SO3d(const Mat3d& R_) : R(R_) {}
    explicit SO3d(const Quatd& q) : R(q.toRotationMatrix()) {}

    static SO3d Identity() { return SO3d(Mat3d::Identity()); }

    const Mat3d& matrix() const { return R; }
    Vec3d operator*(const Vec3d& v) const { return R * v; }
    SO3d operator*(const SO3d& o) const { return SO3d(R * o.R); }
    SO3d inverse() const { return SO3d(R.transpose()); }
};

struct SE3d {
    SO3d so3{SO3d::Identity()};
    Vec3d translation{0.0, 0.0, 0.0};

    SE3d() = default;
    SE3d(const SO3d& r, const Vec3d& t) : so3(r), translation(t) {}
    SE3d(const Mat3d& R, const Vec3d& t) : so3(R), translation(t) {}

    static SE3d Identity() { return SE3d(SO3d::Identity(), Vec3d(0.0, 0.0, 0.0)); }

    static SE3d FromRowMajor4x4(const std::array<double, 16>& m) {
        Mat3d R;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R(r, c) = m[static_cast<size_t>(r * 4 + c)];
        Vec3d t(m[3], m[7], m[11]);
        return SE3d(R, t);
    }

    std::array<double, 16> toRowMajor4x4() const {
        const Mat3d& R = so3.matrix();
        return {
            R(0, 0), R(0, 1), R(0, 2), translation.x,
            R(1, 0), R(1, 1), R(1, 2), translation.y,
            R(2, 0), R(2, 1), R(2, 2), translation.z,
            0.0,     0.0,     0.0,     1.0
        };
    }

    const Mat3d& rotationMatrix() const { return so3.matrix(); }

    Vec3d operator*(const Vec3d& p) const {
        return (so3 * p) + translation;
    }

    SE3d operator*(const SE3d& o) const {
        return SE3d(so3 * o.so3, (so3 * o.translation) + translation);
    }

    SE3d inverse() const {
        SO3d r_inv = so3.inverse();
        return SE3d(r_inv, -(r_inv * translation));
    }
};

} // namespace math

struct TripodAxisCalibration {
    // Mechanical offset from Tripod Rotation Axis center to Scanner Body mount (in Station/Body frame)
    // Station frame: +X Forward at theta=0, +Y Left (CCW 90 deg), +Z Up (along tripod vertical axis)
    math::Vec3d T_axis_body_offset_m{0.0, 0.0, 0.05}; // Default [0.0, 0.0, 0.05] m (5cm above tripod mount)
    // Dual-axis bubble tilt leveling (roll, pitch in rad; [0, 0] when leveled)
    math::Vec2d axis_tilt_leveling_rad{0.0, 0.0};
};

class RotationGeometry {
public:
    /**
     * Canonical Mapping from Camera Optical Frame (+X Right, +Y Down, +Z Forward)
     * to Station Reference Frame at theta = 0 (+X Forward, +Y Left, +Z Up):
     *   X_station (Forward) = +Z_cam
     *   Y_station (Left)    = -X_cam
     *   Z_station (Up)      = -Y_cam
     *
     * R_body_cam = [  0   0   1 ]
     *              [ -1   0   0 ]
     *              [  0  -1   0 ]
     */
    static math::SE3d cameraOpticalToStationZero() {
        math::Mat3d R_sc = math::Mat3d::Zero();
        R_sc(0, 2) =  1.0; // X_station = +Z_cam (Forward)
        R_sc(1, 0) = -1.0; // Y_station = -X_cam (Left)
        R_sc(2, 1) = -1.0; // Z_station = -Y_cam (Up)
        return math::SE3d(math::SO3d(R_sc), math::Vec3d(0.0, 0.0, 0.0));
    }

    /**
     * Pure horizontal rotation R_z(theta) around the vertical tripod rotation axis (+Z_station).
     */
    static math::SE3d tripodZRotation(double rotation_angle_deg) {
        double theta_rad = rotation_angle_deg * math::kDegToRad;
        math::Quatd q_z = math::Quatd::FromEuler(0.0, 0.0, theta_rad);
        return math::SE3d(math::SO3d(q_z), math::Vec3d(0.0, 0.0, 0.0));
    }

    /**
     * Computes T_station_lidar for stop angle theta_i around the vertical tripod +Z_station axis
     * when sensor extrinsics T_cam_lidar map points from LiDAR frame into Camera Optical Frame:
     *
     *   T_station_lidar(theta_i) = T_station_axis * R_z(theta_i) * T_axis_body * T_body_cam * T_cam_lidar
     *
     * Where:
     *   - T_cam_lidar: Calibrated extrinsic matrix (handling 45 deg pitch on av-scanar, 90 deg pitch on av-ios,
     *                  or active rig profile on av-scan).
     *   - T_body_cam:  cameraOpticalToStationZero() converting Camera Optical Frame to Station Reference Frame.
     *   - T_axis_body: Translation offset from tripod rotation center to scanner body mount (default [0, 0, 0.05] m).
     *   - R_z(theta):  Pure rotation of theta_i around +Z_station.
     *   - T_station_axis: Tripod leveling transform (identity when leveled).
     */
    static math::SE3d computeStationLidarTransform(
        double rotation_angle_deg,
        const math::SE3d& T_cam_lidar,
        const TripodAxisCalibration& axis_calib = TripodAxisCalibration(),
        bool extrinsic_is_camera_optical = true) {

        // 1. Rotation R_z(theta) around the vertical tripod rotation axis (+Z_station)
        math::SE3d R_axis = tripodZRotation(rotation_angle_deg);

        // 2. Translation offset of device body above the tripod mount center
        math::SE3d T_axis_body(math::SO3d::Identity(), axis_calib.T_axis_body_offset_m);

        // 3. Station leveling transform (identity when tripod is leveled)
        math::Quatd q_tilt = math::Quatd::FromEuler(axis_calib.axis_tilt_leveling_rad.x,
                                                    axis_calib.axis_tilt_leveling_rad.y,
                                                    0.0);
        math::SE3d T_station_axis(math::SO3d(q_tilt), math::Vec3d(0.0, 0.0, 0.0));

        if (extrinsic_is_camera_optical) {
            // 4a. Transform from Camera Optical Frame to Station Reference Frame at theta=0
            math::SE3d T_body_cam = cameraOpticalToStationZero();
            return T_station_axis * R_axis * T_axis_body * T_body_cam * T_cam_lidar;
        } else {
            // 4b. Extrinsic is already T_body_lidar in (+X Forward, +Y Left, +Z Up) body frame
            return T_station_axis * R_axis * T_axis_body * T_cam_lidar;
        }
    }

    /**
     * Explicit helper when extrinsics are stored directly as T_body_lidar in an
     * X-Forward, Y-Left, Z-Up body frame:
     *   T_station_lidar(theta_i) = T_station_axis * R_z(theta_i) * T_axis_body * T_body_lidar
     */
    static math::SE3d computeStationLidarTransformFromBodyExtrinsic(
        double rotation_angle_deg,
        const math::SE3d& T_body_lidar,
        const TripodAxisCalibration& axis_calib = TripodAxisCalibration()) {
        return computeStationLidarTransform(rotation_angle_deg, T_body_lidar, axis_calib, false);
    }

    /**
     * Transforms any 3D point (with .x, .y, .z members) from LiDAR sensor frame into Station frame.
     */
    template <typename PointT>
    static PointT transformPointToStation(const PointT& p_lidar,
                                          const math::SE3d& T_station_lidar) {
        PointT p_stat = p_lidar;
        math::Vec3d v_in(static_cast<double>(p_lidar.x),
                         static_cast<double>(p_lidar.y),
                         static_cast<double>(p_lidar.z));
        math::Vec3d v_out = T_station_lidar * v_in;
        p_stat.x = static_cast<decltype(p_stat.x)>(v_out.x);
        p_stat.y = static_cast<decltype(p_stat.y)>(v_out.y);
        p_stat.z = static_cast<decltype(p_stat.z)>(v_out.z);
        return p_stat;
    }
};

} // namespace av::core::terrestrial

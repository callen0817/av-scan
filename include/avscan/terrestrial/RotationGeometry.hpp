#pragma once

#include "core/terrestrial/rotation_geometry.hpp"

namespace avscan {
namespace math {
using av::core::terrestrial::math::kPi;
using av::core::terrestrial::math::kDegToRad;
using av::core::terrestrial::math::kRadToDeg;
using av::core::terrestrial::math::Vec2d;
using av::core::terrestrial::math::Vec3d;
using av::core::terrestrial::math::Mat3d;
using av::core::terrestrial::math::Quatd;
using av::core::terrestrial::math::SO3d;
using av::core::terrestrial::math::SE3d;
} // namespace math

namespace terrestrial {
using av::core::terrestrial::TripodAxisCalibration;
using av::core::terrestrial::RotationGeometry;
} // namespace terrestrial
} // namespace avscan

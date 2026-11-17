#pragma once

#include "core/terrestrial/station_data_model.hpp"
#include "include/avscan/terrestrial/RotationGeometry.hpp"

namespace avscan {
namespace core {
using av::core::terrestrial::LidarPointXYZIRT;
using av::core::terrestrial::LidarFrame;
using av::core::terrestrial::ImuMeasurement;
using av::core::terrestrial::DeviceMountProfile;
} // namespace core

namespace terrestrial {
using av::core::terrestrial::RotationMode;
using av::core::terrestrial::MotionState;
using av::core::terrestrial::MotorVerificationStatus;
using av::core::terrestrial::AcquisitionMode;
using av::core::terrestrial::LidarPointXYZIRT;
using av::core::terrestrial::LidarFrame;
using av::core::terrestrial::ImuMeasurement;
using av::core::terrestrial::DeviceMountProfile;
using av::core::terrestrial::StopObservation;
using av::core::terrestrial::TerrestrialStationConfig;
using av::core::terrestrial::TerrestrialStationMetadata;
} // namespace terrestrial
} // namespace avscan

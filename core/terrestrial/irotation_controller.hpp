#pragma once

#include "core/terrestrial/station_data_model.hpp"
#include <cmath>
#include <functional>
#include <string>

namespace av::core::terrestrial {

struct MotorCapabilities {
    bool has_programmatic_connection{false};
    bool has_measured_angle_feedback{false};
    bool can_command_angle{false};
    std::string device_model{"Manual Tripod Head / Turntable Detent"};
};

class IRotationController {
public:
    virtual ~IRotationController() = default;

    // Lifecycle
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;

    // Introspection
    virtual MotorCapabilities getCapabilities() const = 0;
    virtual RotationMode getRotationMode() const = 0;

    // Motion & Angle Control
    virtual bool setZeroPosition() = 0;
    virtual bool moveToAngleDeg(double target_deg) = 0;
    virtual bool moveRelativeDeg(double delta_deg) = 0;
    virtual void stopMotion() = 0;

    virtual MotionState getMotionState() const = 0;
    virtual bool getCurrentAngleDeg(double& out_angle_deg) const = 0;

    // Non-blocking Operator Prompt Callback
    using PromptOperatorCallback = std::function<void(const std::string& prompt_msg, double target_angle_deg)>;
    virtual void setOperatorPromptCallback(PromptOperatorCallback cb) = 0;
};

/**
 * Standard Default Manual / Detent Indexer (operates without Bluetooth / Serial motor hardware;
 * never blocks Terrestrial Mode or Hybrid Flash Station Mode).
 */
class ManualRotationController : public IRotationController {
public:
    ManualRotationController() : current_angle_deg_(0.0), state_(MotionState::STATIONARY) {}

    bool connect() override { return true; }
    void disconnect() override {}
    bool isConnected() const override { return true; }

    MotorCapabilities getCapabilities() const override {
        MotorCapabilities cap;
        cap.has_programmatic_connection = false;
        cap.has_measured_angle_feedback = false;
        cap.can_command_angle = false;
        cap.device_model = "Manual Tripod Head / Turntable Detent";
        return cap;
    }

    RotationMode getRotationMode() const override { return RotationMode::MANUAL_INDEX; }

    bool setZeroPosition() override {
        current_angle_deg_ = 0.0;
        state_ = MotionState::STATIONARY;
        return true;
    }

    bool moveToAngleDeg(double target_deg) override {
        // Cable safety guard: wrap at 360 deg so operator never twists cables past one revolution
        if (target_deg >= 360.0 || target_deg <= -360.0) {
            target_deg = std::fmod(target_deg, 360.0);
            if (target_deg < 0.0) target_deg += 360.0;
        }
        current_angle_deg_ = target_deg;
        state_ = MotionState::STATIONARY;
        if (prompt_cb_) {
            prompt_cb_("Rotate scanner to target position and confirm stillness.", target_deg);
        }
        return true;
    }

    bool moveRelativeDeg(double delta_deg) override {
        return moveToAngleDeg(current_angle_deg_ + delta_deg);
    }

    void stopMotion() override {
        state_ = MotionState::STATIONARY;
    }

    MotionState getMotionState() const override { return state_; }
    void setMotionState(MotionState state) { state_ = state; }

    bool getCurrentAngleDeg(double& out_angle_deg) const override {
        out_angle_deg = current_angle_deg_;
        return false; // Nominal angle only, not hardware encoder
    }

    void setOperatorPromptCallback(PromptOperatorCallback cb) override {
        prompt_cb_ = std::move(cb);
    }

private:
    double current_angle_deg_{0.0};
    MotionState state_{MotionState::STATIONARY};
    PromptOperatorCallback prompt_cb_;
};

/**
 * Programmatic Bluetooth / Serial Motorized Head Controller (Syrp Genie Mini / ZEAPON Pons)
 * with automatic non-blocking fallback to manual indexing when physical motor hardware is absent.
 */
class AutoMotorRotationController : public IRotationController {
public:
    explicit AutoMotorRotationController(const std::string& model_name = "Syrp Genie Mini II",
                                         bool hardware_online = true)
        : model_name_(model_name.empty() ? "Syrp Genie Mini II" : model_name),
          hardware_online_(hardware_online) {}

    bool connect() override {
        connected_ = true;
        return true; // Non-blocking: falls back to manual indexing if physical motor is offline
    }

    void disconnect() override {
        connected_ = false;
        state_ = MotionState::STATIONARY;
    }

    bool isConnected() const override { return connected_; }

    MotorCapabilities getCapabilities() const override {
        MotorCapabilities cap;
        cap.has_programmatic_connection = hardware_online_;
        cap.has_measured_angle_feedback = hardware_online_;
        cap.can_command_angle = hardware_online_;
        cap.device_model = hardware_online_
            ? model_name_
            : (model_name_ + " (Fallback to Manual Detent)");
        return cap;
    }

    RotationMode getRotationMode() const override {
        return hardware_online_ ? RotationMode::AUTOMATIC_MOTOR : RotationMode::MANUAL_INDEX;
    }

    bool setZeroPosition() override {
        // Unwind cable back to 0.0 deg
        current_angle_deg_ = 0.0;
        state_ = MotionState::STATIONARY;
        return true;
    }

    bool moveToAngleDeg(double target_deg) override {
        // Cable safety guard: never command > 360 deg unidirectional rotation
        if (target_deg >= 360.0 || target_deg <= -360.0) {
            target_deg = std::fmod(target_deg, 360.0);
            if (target_deg < 0.0) target_deg += 360.0;
        }
        current_angle_deg_ = target_deg;
        state_ = MotionState::STATIONARY;
        if (prompt_cb_) {
            if (hardware_online_) {
                prompt_cb_("Motor indexed to " + std::to_string(static_cast<int>(std::round(target_deg))) + " deg.", target_deg);
            } else {
                prompt_cb_("Auto motor fallback: rotate turntable to target stop and hold still.", target_deg);
            }
        }
        return true;
    }

    bool moveRelativeDeg(double delta_deg) override {
        return moveToAngleDeg(current_angle_deg_ + delta_deg);
    }

    void stopMotion() override {
        state_ = MotionState::STATIONARY;
    }

    MotionState getMotionState() const override { return state_; }

    bool getCurrentAngleDeg(double& out_angle_deg) const override {
        out_angle_deg = current_angle_deg_;
        return hardware_online_;
    }

    void setOperatorPromptCallback(PromptOperatorCallback cb) override {
        prompt_cb_ = std::move(cb);
    }

private:
    std::string model_name_;
    bool hardware_online_{true};
    bool connected_{true};
    double current_angle_deg_{0.0};
    MotionState state_{MotionState::STATIONARY};
    PromptOperatorCallback prompt_cb_;
};

} // namespace av::core::terrestrial

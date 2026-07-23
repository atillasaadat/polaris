#include "actuators/reaction_wheel.hpp"

#include <algorithm>

namespace polaris::sim::actuators {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kRpmToRadPerS = kTwoPi / 60.0;

double sign(double x) {
  return (x > 0.0) - (x < 0.0);
}

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}
}  // namespace

ReactionWheelSpec ReactionWheelSpec::fromParams(const std::map<std::string, double>& p) {
  ReactionWheelSpec s;
  s.max_torque_nm = get(p, "max_torque_nm");
  s.max_momentum_nms = get(p, "max_momentum_nms");
  s.max_speed_rad_s = get(p, "max_speed_rpm") * kRpmToRadPerS;
  s.rotor_inertia_kg_m2 = get(p, "rotor_inertia_kg_m2");
  s.motor_kt_nm_a = get(p, "motor_kt_nm_a");
  s.motor_resistance_ohm = get(p, "motor_resistance_ohm");
  s.dry_friction_nm = get(p, "dry_friction_nm");
  s.viscous_friction_nm_s = get(p, "viscous_friction_nm_s");
  s.aero_friction_nm_s2 = get(p, "aero_friction_nm_s2");
  s.torque_quantization_nm = get(p, "torque_quantization_nm");
  s.static_imbalance_kg_m = get(p, "static_imbalance_kg_m");
  s.dynamic_imbalance_kg_m2 = get(p, "dynamic_imbalance_kg_m2");
  s.idle_power_w = get(p, "idle_power_w");
  s.speed_loop_gain_nm_per_rad_s = get(p, "speed_loop_gain_nm_per_rad_s");
  return s;
}

double ReactionWheel::frictionTorque() const {
  // Rundown model (RW-0.4 ICD): dry + wet·|ω| + aero·ω², opposing the spin.
  const double mag = spec_.dry_friction_nm + spec_.viscous_friction_nm_s * std::abs(speed_) +
                     spec_.aero_friction_nm_s2 * speed_ * speed_;
  return -sign(speed_) * mag;
}

ReactionWheelOutput ReactionWheel::step(double dt) {
  ReactionWheelOutput out;
  if (!(dt > 0.0)) {
    out.speed_rad_s = speed_;
    out.momentum_nms = inertia_ * speed_;
    return out;
  }

  // Bearing friction (pre-step speed), needed both by the dynamics and by the
  // speed-loop feedforward below.
  const double friction = frictionTorque();

  // 1. Resolve the motor torque request, honouring faults, the command mode, and
  //    the torque box.
  double motor_torque = commanded_torque_;
  if (fault_stuck_) {
    motor_torque = 0.0;  // drive off; only friction acts
  } else if (fault_runaway_) {
    motor_torque = runaway_sign_ * spec_.max_torque_nm;
  } else if (mode_ == Mode::kSpeed && inertia_ > 0.0) {
    // Onboard speed loop. With a configured gain it is a finite-bandwidth P loop;
    // otherwise the ideal inner loop asks for the torque that reaches the target
    // this step (net I·Δω/dt), with a friction feedforward so it holds at target.
    // Either way the torque box below is the real limit on authority.
    const double speed_error = commanded_speed_ - speed_;
    motor_torque = (spec_.speed_loop_gain_nm_per_rad_s > 0.0)
                       ? spec_.speed_loop_gain_nm_per_rad_s * speed_error
                       : inertia_ * speed_error / dt - friction;
  }
  if (spec_.max_torque_nm > 0.0) {
    motor_torque = std::clamp(motor_torque, -spec_.max_torque_nm, spec_.max_torque_nm);
  }
  if (spec_.torque_quantization_nm > 0.0) {
    motor_torque =
        std::round(motor_torque / spec_.torque_quantization_nm) * spec_.torque_quantization_nm;
  }

  // 2. Rotor dynamics: I·ω̇ = motor torque − bearing friction.
  double omega_dot = (inertia_ > 0.0) ? (motor_torque + friction) / inertia_ : 0.0;
  double new_speed = speed_ + omega_dot * dt;

  // Momentum/speed ceiling: the torque box cannot push past max speed. Clamp and
  // back out the acceleration that was actually delivered, so the reaction torque
  // reported to the body matches the motion (a saturated wheel delivers less).
  if (spec_.max_speed_rad_s > 0.0 && std::abs(new_speed) > spec_.max_speed_rad_s) {
    new_speed = sign(new_speed) * spec_.max_speed_rad_s;
    omega_dot = (new_speed - speed_) / dt;
    // The motor torque implied by the clamped motion (friction still acts).
    motor_torque = inertia_ * omega_dot - friction;
  }

  // 3. Reaction on the body is the negative rate of change of rotor momentum.
  out.reaction_torque_nm = -inertia_ * omega_dot;

  // 4. Electrical power: copper loss (I²R, I = τ/Kt) + mechanical (τ·ω, signed so
  //    braking regenerates), plus housekeeping.
  double copper = 0.0;
  if (spec_.motor_kt_nm_a > 0.0) {
    const double current = motor_torque / spec_.motor_kt_nm_a;
    copper = current * current * spec_.motor_resistance_ohm;
  }
  const double mechanical = motor_torque * speed_;  // uses pre-step speed
  out.bus_power_w = spec_.idle_power_w + copper + mechanical;

  // 5. Advance state and phase.
  speed_ = new_speed;
  angle_ = std::fmod(angle_ + speed_ * dt, kTwoPi);

  out.speed_rad_s = speed_;
  out.momentum_nms = inertia_ * speed_;

  // 6. Imbalance disturbances, rotating with the rotor (wheel-frame radial).
  const double w2 = speed_ * speed_;
  const double c = std::cos(angle_);
  const double s = std::sin(angle_);
  const double force_mag = spec_.static_imbalance_kg_m * w2;
  const double torque_mag = spec_.dynamic_imbalance_kg_m2 * w2;
  out.jitter_force_n = Eigen::Vector3d(force_mag * c, force_mag * s, 0.0);
  out.jitter_torque_nm = Eigen::Vector3d(torque_mag * c, torque_mag * s, 0.0);
  return out;
}

}  // namespace polaris::sim::actuators

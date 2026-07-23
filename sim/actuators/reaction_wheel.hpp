#ifndef POLARIS_SIM_ACTUATORS_REACTION_WHEEL_HPP
#define POLARIS_SIM_ACTUATORS_REACTION_WHEEL_HPP

/// @file
/// @brief Reaction-wheel truth model (design doc §7), datasheet-driven.
///
/// Torque-in → reaction-torque-out. A commanded motor torque is limited by the
/// wheel's torque box (peak torque, and the momentum/speed ceiling), reduced by
/// bearing friction, quantized by the drive resolution, and integrated to a rotor
/// speed and stored momentum. The reaction on the spacecraft body is the negative
/// rate of change of rotor angular momentum (Newton's third law), so friction and
/// saturation both show up in the delivered control torque — the effects an
/// attitude controller must cope with.
///
/// **Friction** follows the RW-0.4 rundown model (its ICD): a Coulomb (dry) term,
/// a viscous (wet) term ∝ speed, and an aerodynamic term ∝ speed² (only relevant
/// on the ground / in residual atmosphere). The Coulomb term opposes the spin
/// (−sign(ω)·mag); at exactly ω = 0 it is zero — this model does not include
/// static friction / stiction holding a stopped rotor against a sub-threshold
/// torque, which is immaterial at the µN·m friction levels here. **Power** is the sum of copper
/// loss (I²R, I = τ/Kt) and mechanical power (τ·ω), the latter signed so that braking returns
/// energy to the bus — the regenerative behaviour the ICD calls out.
///
/// **Static and dynamic imbalance are first-class**, because the near-term reason
/// to model this wheel is jitter. A residual static imbalance Us (a mass·radius
/// offset) throws a radial force |F| = Us·ω² that rotates with the rotor; a
/// dynamic imbalance Ud (a product-of-inertia) throws a radial torque
/// |T| = Ud·ω². Both are emitted per step in the wheel frame (spin = +z) at the
/// rotor phase, so a later jitter / waterfall analysis (disturbance spectrum vs
/// wheel speed over a spin-up) can integrate them without changing this model. See
/// the sim/CLAUDE.md "Reaction-wheel jitter" note and design doc §7.
///
/// Fault injection is first-class (§9): a stuck wheel drops all motor torque and
/// coasts on friction; a runaway wheel drives peak torque regardless of command.
///
/// The imbalance magnitudes and the friction/Kt constants are unit-specific
/// calibration data (a balance report), not published datasheet values, so they
/// default to zero / representative and are meant to be filled per unit.
///
/// Implements REQ-SIM-003 (actuator truth models with full error stacks) and
/// REQ-SIM-005 (scriptable fault injection: stuck/runaway).

#include <cmath>
#include <Eigen/Core>
#include <map>
#include <string>

namespace polaris::sim::actuators {

/// Reaction-wheel parameters, SI. Torque-box and momentum come from the datasheet;
/// friction, Kt, and imbalance are per-unit calibration (default 0 = ideal).
struct ReactionWheelSpec {
  double max_torque_nm = 0.0;            ///< peak motor torque
  double max_momentum_nms = 0.0;         ///< stored-momentum ceiling
  double max_speed_rad_s = 0.0;          ///< rotor speed ceiling
  double rotor_inertia_kg_m2 = 0.0;      ///< I; if 0, derived from max_momentum/max_speed
  double motor_kt_nm_a = 0.0;            ///< torque constant (I = τ/Kt) for power
  double motor_resistance_ohm = 0.0;     ///< winding resistance for copper loss
  double dry_friction_nm = 0.0;          ///< Coulomb friction (opposes spin)
  double viscous_friction_nm_s = 0.0;    ///< viscous friction ∝ speed [N·m/(rad/s)]
  double aero_friction_nm_s2 = 0.0;      ///< aero friction ∝ speed² [N·m/(rad/s)²]
  double torque_quantization_nm = 0.0;   ///< drive torque LSB (0 = none)
  double static_imbalance_kg_m = 0.0;    ///< Us: radial force = Us·ω²
  double dynamic_imbalance_kg_m2 = 0.0;  ///< Ud: radial torque = Ud·ω²
  double idle_power_w = 0.0;             ///< housekeeping power draw
  /// Proportional gain of the wheel's onboard speed loop [N·m/(rad/s)], used only
  /// in speed-command mode. 0 selects the **ideal inner loop** (the drive commands
  /// whatever torque the box allows to reach the target this step) — the right
  /// default when the wheel's kHz loop is far faster than the ACS step. A positive
  /// gain models a finite-bandwidth loop, which tracks with a steady-state droop.
  double speed_loop_gain_nm_per_rad_s = 0.0;

  /// Rotor inertia, using max_momentum/max_speed when not set explicitly.
  double inertia() const {
    if (rotor_inertia_kg_m2 > 0.0) {
      return rotor_inertia_kg_m2;
    }
    return (max_speed_rad_s > 0.0) ? max_momentum_nms / max_speed_rad_s : 0.0;
  }

  /// Build a spec from hardware-library params (the keys used by the
  /// `config/hardware/reaction_wheel/*.yaml` entries); `max_speed_rpm` converts to
  /// rad/s. Missing keys default to 0 (that term disabled). See reaction_wheel.cpp.
  static ReactionWheelSpec fromParams(const std::map<std::string, double>& params);
};

/// One reaction-wheel step, in the wheel frame (spin axis = +z).
struct ReactionWheelOutput {
  double reaction_torque_nm = 0.0;  ///< control torque on the body about +z (= -I·ω̇)
  double speed_rad_s = 0.0;         ///< rotor speed
  double momentum_nms = 0.0;        ///< stored momentum (I·ω)
  double bus_power_w = 0.0;         ///< electrical power (negative = regenerating)
  Eigen::Vector3d jitter_force_n = Eigen::Vector3d::Zero();    ///< static-imbalance force
  Eigen::Vector3d jitter_torque_nm = Eigen::Vector3d::Zero();  ///< dynamic-imbalance torque
};

/// A single reaction wheel spinning about its own +z. The caller applies the
/// mounting rotation to place the outputs in the body frame.
class ReactionWheel {
 public:
  explicit ReactionWheel(const ReactionWheelSpec& spec) : spec_(spec), inertia_(spec.inertia()) {}

  /// Command a motor torque [N·m] (torque mode). This is the interface a
  /// torque-authority ACS uses, and what the allocation layer (§8.5) drives.
  void commandTorque(double torque_nm) {
    mode_ = Mode::kTorque;
    commanded_torque_ = torque_nm;
  }

  /// Command a rotor speed [rad/s] (speed mode). Real wheels expose this as a
  /// selectable onboard mode: the drive's local loop supplies whatever torque —
  /// within the torque box — reaches and holds the target, rejecting friction.
  /// The plant behaves differently than in torque mode (torque becomes an
  /// internal variable), which is why it is modelled here rather than upstream.
  void commandSpeed(double speed_rad_s) {
    mode_ = Mode::kSpeed;
    commanded_speed_ = speed_rad_s;
  }

  /// Advance the rotor by @p dt seconds (dt > 0) and return the delivered reaction
  /// torque, telemetry, and imbalance disturbances at the new rotor phase.
  ReactionWheelOutput step(double dt);

  double speed() const { return speed_; }

  double momentum() const { return inertia_ * speed_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Stuck: motor drive off, rotor coasts on friction until cleared. Takes
  /// precedence over a runaway fault if both are set.
  void setStuck(bool stuck) { fault_stuck_ = stuck; }

  /// Runaway: drive peak torque in @p sign direction regardless of command.
  void setRunaway(bool runaway, double sign = 1.0) {
    fault_runaway_ = runaway;
    runaway_sign_ = (sign >= 0.0) ? 1.0 : -1.0;
  }

  void clearFaults() {
    fault_stuck_ = false;
    fault_runaway_ = false;
  }

 private:
  /// Bearing friction torque opposing the current spin (magnitude·−sign(ω)).
  double frictionTorque() const;

  enum class Mode { kTorque, kSpeed };

  ReactionWheelSpec spec_;
  double inertia_ = 0.0;
  double speed_ = 0.0;  ///< rotor speed [rad/s]
  double angle_ = 0.0;  ///< rotor phase [rad], for imbalance
  Mode mode_ = Mode::kTorque;
  double commanded_torque_ = 0.0;
  double commanded_speed_ = 0.0;
  bool fault_stuck_ = false;
  bool fault_runaway_ = false;
  double runaway_sign_ = 1.0;
};

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_REACTION_WHEEL_HPP

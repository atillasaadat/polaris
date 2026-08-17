#ifndef POLARIS_SIM_ACTUATORS_THRUSTER_HPP
#define POLARIS_SIM_ACTUATORS_THRUSTER_HPP

/// @file
/// @brief Thruster truth model (design doc §7, §17), datasheet-driven.
///
/// Throttle-in → force-and-torque-out. A commanded throttle (0..1) is realised
/// as a delivered thrust through the valve/chamber lag, the unit's fixed
/// thrust-scale error, white thrust noise and a fixed misalignment of the
/// thrust vector off its nominal axis; the mass flow follows from the delivered
/// thrust and Isp. Every burn is a **finite burn** (§17): there is no impulsive
/// path here, the executor holds a throttle for a duration and the plant
/// integrates what the thruster actually delivered.
///
/// **Delivered thrust** follows a first-order lag toward the commanded level,
/// with separate rise and fall time constants (valve opening and chamber
/// pressurisation are slower than the tail-off on most small units):
/// \f[
///   \dot F = \frac{F_{cmd} - F}{\tau}, \qquad
///   \tau = \begin{cases}\tau_{rise} & F_{cmd} > F \\ \tau_{fall} & \text{otherwise}\end{cases},
///   \qquad F_{cmd} = u\,F_{max}\,(1+\epsilon_s),
/// \f]
/// integrated exactly over the step (\f$F' = F_{cmd} + (F - F_{cmd})e^{-\Delta t/\tau}\f$)
/// so a step shorter than the constant is not overshot. \f$\epsilon_s\f$ is the
/// **scale error**, a per-unit constant a hot-fire calibration would trim; the
/// **noise** term multiplies the delivered thrust by \f$(1+\epsilon_n\,\nu)\f$
/// per step with \f$\nu\sim N(0,1)\f$. A zero time constant delivers the command
/// this step. **Minimum impulse bit** gates a pulse: a command that would
/// deliver less than \f$I_{min}\f$ over the step is dropped to zero — the valve
/// does not resolve it — which is what makes small trims quantised.
///
/// **Direction and torque.** The nominal thrust axis is the unit's +z; the
/// delivered vector is rotated off it by the misalignment angle at a fixed
/// azimuth (a manufacturing/alignment error, not jitter). The body-frame force
/// is the caller's `mounting_dcm` applied to that; the torque is
/// \f$\mathbf r_{mount}\times\mathbf F\f$ about the body origin — the
/// thrust-misalignment torque §17 says a burn's attitude control must fight,
/// which is why the position on the vehicle is part of the mounting and not
/// ignored. (The CM offset from the body origin is the plant's business.)
///
/// **Mass flow** \f$\dot m = F/(I_{sp}\,g_0)\f$ from the delivered thrust, so a
/// lagged or noisy thrust spends propellant as it should. The plant depletes
/// the vehicle mass with it (`sim/io/closed_loop.cpp`).
///
/// Fault injection is first-class (§9): stuck-off (a valve that never opens —
/// the burn is silently missed) and stuck-on (a valve that fires regardless —
/// the runaway the momentum-anomaly monitor exists for).
///
/// Representative catalog values live in `config/hardware/thruster/*.yaml`;
/// they are datasheet class figures, and the error terms are per-unit
/// calibration (default 0 = ideal).
///
/// Implements REQ-SIM-003 (actuator truth models with full error stacks) and
/// REQ-SIM-005 (scriptable fault injection).

#include <cmath>
#include <Eigen/Core>
#include <map>
#include <string>

#include "random/rng.hpp"

namespace polaris::sim::actuators {

/// Thruster parameters, SI. Thrust, Isp and impulse bit from the datasheet; the
/// error terms are per-unit calibration (default 0 = ideal).
struct ThrusterSpec {
  double thrust_n = 0.0;                  ///< rated thrust at full throttle
  double isp_s = 0.0;                     ///< specific impulse (mass flow = F/(Isp g0))
  double min_impulse_bit_ns = 0.0;        ///< smallest impulse the valve resolves (0 = none)
  double rise_time_s = 0.0;               ///< lag toward a higher command (0 = instant)
  double fall_time_s = 0.0;               ///< lag toward a lower command (0 = instant)
  double thrust_scale_error = 0.0;        ///< fixed fractional bias on delivered thrust
  double thrust_noise_frac = 0.0;         ///< white fractional noise per step (1σ)
  double misalignment_rad = 0.0;          ///< fixed angle off the nominal axis
  double misalignment_azimuth_rad = 0.0;  ///< where in the plane the tilt points

  /// Build from hardware-library params (`config/hardware/thruster/*.yaml`).
  static ThrusterSpec fromParams(const std::map<std::string, double>& params);
};

/// One step's product, in the **thruster frame** (nominal thrust = +z). The
/// caller applies the mounting DCM and position.
struct ThrusterOutput {
  Eigen::Vector3d force_n{Eigen::Vector3d::Zero()};  ///< delivered force, thruster frame
  double delivered_thrust_n = 0.0;                   ///< |force|
  double mass_flow_kg_s = 0.0;                       ///< propellant rate at that thrust
  double throttle = 0.0;                             ///< the command that was applied
};

/// A single thruster firing along its own +z.
class Thruster {
 public:
  Thruster(const ThrusterSpec& spec, std::uint64_t master_seed, std::uint64_t stream_id)
      : spec_(spec), rng_(random::streamRng(master_seed, stream_id)) {}

  explicit Thruster(const ThrusterSpec& spec) : Thruster(spec, 0, 0) {}

  /// Command a throttle in [0, 1] (clamped), held until the next command.
  void commandThrottle(double throttle) {
    commanded_ = std::isfinite(throttle) ? std::min(1.0, std::max(0.0, throttle)) : 0.0;
  }

  double commandedThrottle() const { return commanded_; }

  /// Advance the valve/chamber by @p dt seconds and return what was delivered.
  ThrusterOutput step(double dt);

  double deliveredThrust() const { return thrust_; }

  // --- Fault injection (§9) --------------------------------------------------
  void setStuckOff(bool stuck) { fault_off_ = stuck; }

  void setStuckOn(bool stuck) { fault_on_ = stuck; }

  void clearFaults() {
    fault_off_ = false;
    fault_on_ = false;
  }

 private:
  ThrusterSpec spec_;
  random::SplitMix64 rng_;
  double commanded_ = 0.0;
  double thrust_ = 0.0;  ///< delivered thrust after the lag [N]
  bool fault_off_ = false;
  bool fault_on_ = false;
};

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_THRUSTER_HPP

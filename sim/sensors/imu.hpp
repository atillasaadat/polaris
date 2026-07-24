#ifndef POLARIS_SIM_SENSORS_IMU_HPP
#define POLARIS_SIM_SENSORS_IMU_HPP

/// @file
/// @brief Inertial measurement unit (gyro + accelerometer) truth model, and a
/// datasheet-driven spec so COTS units drop straight in (design doc §6.2, §19.2).
///
/// The IMU is the most-used ADCS sensor: the gyro rate feeds the MEKF's
/// propagation and its bias state, the accelerometer feeds OD. It carries the
/// richest error stack of any sensor here, so its parameters are captured in an
/// `ImuSpec` populated directly from a manufacturer datasheet — the same numbers
/// an engineer reads off the product brief, in the datasheet's own engineering
/// units — and converted to SI in one place. Adding a new COTS IMU is then just
/// writing a `config/hardware/imu/*.yaml` entry with those keys
/// (`stim377h.yaml` is the worked example): the config compiler resolves it and
/// `ImuSpec::fromParams` builds the spec. There is no in-code IMU catalog —
/// hardware values live only in the YAML library (design doc §19.4).
///
/// The truth model turns the true body rate and specific force into what the unit
/// would report, through:
///  - a per-unit **scale-factor + misalignment** matrix (fixed miscalibration
///    drawn once within the datasheet bound, so each seeded unit is a distinct but
///    in-spec device),
///  - a **turn-on bias** (bias repeatability, fixed per power-up),
///  - an in-run **bias drift** — a stateful first-order Gauss-Markov process whose
///    steady-state σ is the datasheet bias instability (this is the stateful term
///    Push 19's memoryless `VectorErrorModel` deferred),
///  - gyro **g-sensitivity** (rate bias proportional to specific force),
///  - **white noise** from the angular/velocity random walk, whose per-sample σ is
///    `random_walk / √dt` so it scales correctly with the sample interval,
///  - **quantization** and **range saturation**,
///
/// and integrates the result over the sample interval to the **delta-angle /
/// delta-velocity** the FSW consumes (§2.4). Coning/sculling compensation is an
/// FSW-side algorithm and the read-time buffer is the §2.4 macro-step's job, so
/// neither is here. Fault injection (bias jump, dropout) is first-class (§9).
///
/// Each unit owns one seed-derived stream (§3.5): reproducible from {config,seed}
/// and independent of every other sensor.
///
/// Implements REQ-SIM-003 (sensor truth models with full error stacks) and
/// REQ-SIM-005 (scriptable fault injection).

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/sensor_error.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// Datasheet parameters for one 3-axis sensor triad (gyro or accelerometer), in
/// SI: rate in rad/s and rad/√s, or specific force in m/s² and (m/s²)/√s.
struct TriadSpec {
  double range = 0.0;               ///< output saturation, symmetric [SI] (0 = none)
  double random_walk = 0.0;         ///< ARW / VRW [SI·√s], gives white noise σ = rw/√dt
  double bias_instability = 0.0;    ///< in-run bias 1σ (Gauss-Markov steady state) [SI]
  double bias_correlation_s = 0.0;  ///< Gauss-Markov correlation time τ [s] (0 = constant bias)
  double bias_repeatability = 0.0;  ///< turn-on / over-temperature bias 1σ [SI]
  double scale_factor = 0.0;        ///< scale-factor accuracy, 1σ fraction (500 ppm = 5e-4)
  double misalignment = 0.0;        ///< axis-misalignment 1σ [rad]
  double resolution = 0.0;          ///< quantization LSB [SI] (0 = none)
};

/// A full IMU spec: two triads plus the gyro's linear-acceleration sensitivity.
struct ImuSpec {
  TriadSpec gyro;
  TriadSpec accel;
  /// Gyro rate bias per unit specific force [(rad/s)/(m/s²)] — the datasheet's
  /// "°/h/g" linear-acceleration effect.
  double gyro_g_sensitivity = 0.0;
  /// Native output data rate [Hz]; informational (the actual noise scaling uses
  /// the dt passed to `sample`, not this).
  double sample_rate_hz = 0.0;

  /// Build a spec from datasheet-native library params (the keys used by the
  /// `config/hardware/*.yaml` IMU entries), converting each to SI. Missing keys
  /// default to 0 (that term disabled). See imu.cpp for the key list and units.
  static ImuSpec fromParams(const std::map<std::string, double>& params);
};

/// One IMU sample over an interval of @p dt seconds.
struct ImuSample {
  math::Vec3<math::frames::Body> delta_angle_rad{};      ///< ∫ measured rate dt
  math::Vec3<math::frames::Body> delta_velocity_mps{};   ///< ∫ measured specific force dt
  math::Vec3<math::frames::Body> angular_rate_rads{};    ///< measured rate (convenience)
  math::Vec3<math::frames::Body> specific_force_mps2{};  ///< measured specific force
  bool valid{true};
  time::Tai time_tag{};
};

/// A three-axis IMU. Construct with a spec and a per-source stream id under the
/// run's master seed; the fixed miscalibration and turn-on biases are realised at
/// construction, so the same {spec, seed, stream_id} always builds the same unit.
///
/// **Measurement model.** Each triad is the shared §6.1 stack
/// (`VectorErrorModel`) wrapped around a stateful bias. For the gyro, given a true
/// body rate \f$\omega\f$ and specific force \f$f\f$ over an interval \f$\Delta t\f$:
/// \f[
///   \tilde{\omega} = \operatorname{sat}_{r_g}\!\Big( Q_{\delta_g}\big( M_g\,\omega + b_g + n_g
///   \big) \Big), \qquad b_g = b_g^{\mathrm{on}} + d_g + k_g\,f + b_g^{\mathrm{flt}},
/// \f]
/// and identically for the accelerometer with \f$f\f$ in place of \f$\omega\f$ and
/// no g-sensitivity term:
/// \f[
///   \tilde{f} = \operatorname{sat}_{r_a}\!\Big( Q_{\delta_a}\big( M_a\,f + b_a + n_a \big) \Big),
///   \qquad
///   b_a = b_a^{\mathrm{on}} + d_a + b_a^{\mathrm{flt}}.
/// \f]
/// The terms:
///  - **Scale + misalignment** \f$M = I + \operatorname{diag}(s) + [a]_\times\f$,
///    drawn once at construction with \f$s_i \sim \mathcal{N}(0,\ \sigma_{sf}^2)\f$
///    (`scale_factor`) and small misalignment angles
///    \f$a_i \sim \mathcal{N}(0,\ \sigma_{\mathrm{mis}}^2)\f$ (`misalignment`),
///    \f$[a]_\times\f$ the skew-symmetric cross-product matrix.
///  - **Turn-on bias** \f$b^{\mathrm{on}} \sim \mathcal{N}(0,\ \sigma_{\mathrm{rep}}^2 I)\f$
///    (`bias_repeatability`), fixed per power-up.
///  - **In-run bias drift** \f$d_k\f$ — a first-order Gauss-Markov process advanced
///    each sample:
///    \f[
///      d_{k+1} = \phi\, d_k + w_k, \qquad
///      \phi = e^{-\Delta t/\tau}, \qquad
///      w_k \sim \mathcal{N}\!\big(0,\ \sigma_{bi}^2\,(1-\phi^2)\, I\big),
///    \f]
///    with correlation time \f$\tau\f$ (`bias_correlation_s`) and steady-state
///    \f$\sigma_{bi}\f$ (`bias_instability`). It is seeded at its stationary
///    distribution \f$d_0 \sim \mathcal{N}(0,\ \sigma_{bi}^2 I)\f$. When
///    \f$\tau \le 0\f$ there is no drift: \f$\phi = 1,\ w = 0,\ d_0 = 0\f$.
///  - **g-sensitivity** \f$k_g\,f\f$ — a gyro rate bias proportional to the specific
///    force, \f$k_g\f$ = `gyro_g_sensitivity`.
///  - **White noise** \f$n \sim \mathcal{N}(0,\ \sigma^2 I)\f$ with
///    \f$\sigma = \mathrm{rw}/\sqrt{\Delta t}\f$ from the angular/velocity random
///    walk (`random_walk`), so it scales correctly with the sample interval.
///  - **Quantization** \f$Q_{\delta}\f$ (`resolution`) and **saturation**
///    \f$\operatorname{sat}_r\f$ (`range`), as in `VectorErrorModel`.
///  - \f$b^{\mathrm{flt}}\f$ is the injected bias-jump fault (§9), applied with the
///    bias so it saturates and quantizes like a genuine one.
///
/// The reported sample carries both the rates and their integrals over the step,
/// the **delta-angle / delta-velocity** the FSW consumes:
/// \f[
///   \Delta\theta = \tilde{\omega}\,\Delta t, \qquad \Delta v = \tilde{f}\,\Delta t.
/// \f]
/// An **ideal** IMU (`noise_enabled = false`) reports \f$\tilde{\omega} = \omega\f$,
/// \f$\tilde{f} = f\f$: every stochastic term is drawn (to keep the stream aligned)
/// but none is applied.
class Imu {
 public:
  /// The spec this unit was built from (sensor rates drive the §2.4 loop).
  const ImuSpec& spec() const { return spec_; }

  /// @param spec The datasheet-derived error/rate specification.
  /// @param master_seed The run's master RNG seed (§3.5).
  /// @param stream_id This unit's per-source stream id.
  /// @param noise_enabled false builds an **ideal** IMU (measurement = truth, no
  ///        bias/scale/misalignment/noise), for noise-free baseline runs (§6.2).
  Imu(const ImuSpec& spec, std::uint64_t master_seed, std::uint64_t stream_id,
      bool noise_enabled = true);

  /// Measure over @p dt seconds (dt > 0) given the true body rate and the true
  /// specific force (non-gravitational acceleration) in the body frame at truth
  /// time @p epoch. Advances the internal bias-drift state by @p dt.
  ImuSample sample(const time::Tai& epoch, double dt,
                   const math::Vec3<math::frames::Body>& true_rate_body,
                   const math::Vec3<math::frames::Body>& true_specific_force_body);

  // --- Fault injection (§9) --------------------------------------------------

  /// Set a persistent gyro rate-bias jump [rad/s] until cleared (replaces, not
  /// accumulates). Applied with the bias, so it saturates/quantizes like a real one.
  void injectGyroBiasJump(const math::Vec3<math::frames::Body>& delta_rate) {
    gyro_fault_bias_ = delta_rate.eigen();
  }

  /// Set a persistent accelerometer bias jump [m/s²] until cleared.
  void injectAccelBiasJump(const math::Vec3<math::frames::Body>& delta_sf) {
    accel_fault_bias_ = delta_sf.eigen();
  }

  /// Force every subsequent sample invalid (dropout) until cleared.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  /// Clear all injected faults.
  void clearFaults() {
    gyro_fault_bias_.setZero();
    accel_fault_bias_.setZero();
    fault_dropout_ = false;
  }

 private:
  /// One triad's measurement, advancing its Gauss-Markov bias by @p dt.
  Eigen::Vector3d measureTriad(const TriadSpec& spec, VectorErrorModel& err,
                               const Eigen::Vector3d& turn_on_bias, Eigen::Vector3d& drift_state,
                               const Eigen::Vector3d& extra_bias, const Eigen::Vector3d& truth,
                               double dt);

  ImuSpec spec_;
  random::SplitMix64 rng_;

  // Fixed per-unit miscalibration (scale + misalignment) and turn-on bias, drawn
  // once at construction so the modelled unit is a specific in-spec device.
  VectorErrorModel gyro_err_;
  VectorErrorModel accel_err_;
  Eigen::Vector3d gyro_turn_on_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_turn_on_ = Eigen::Vector3d::Zero();

  // Evolving in-run bias-drift state (Gauss-Markov).
  Eigen::Vector3d gyro_drift_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_drift_ = Eigen::Vector3d::Zero();

  Eigen::Vector3d gyro_fault_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_fault_bias_ = Eigen::Vector3d::Zero();
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_IMU_HPP

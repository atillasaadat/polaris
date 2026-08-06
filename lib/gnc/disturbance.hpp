#ifndef POLARIS_GNC_DISTURBANCE_HPP
#define POLARIS_GNC_DISTURBANCE_HPP

/// @file
/// @brief Onboard disturbance-torque feedforward, tiers 1 and 2 (design doc
/// §8.5, §9; REQ-ACTL-011).
///
/// The §8.5 disturbance-estimation ladder, cheapest first. Tier 3 (physical
/// parameter estimation from long-arc data) waits on the §8.3 orbit filter and is
/// not here.
///
/// **Tier 1 — model-based, free.** The two secular torques the vehicle can
/// compute from states it already has:
///
///  * **Gravity gradient** (Hughes §5.2 [hughes1986]; Wertz §17.2 [wertz1978]),
///    \f[
///      \boldsymbol\tau_{gg} = 3n^2\,\hat{\mathbf n}\times(J\hat{\mathbf n}),
///    \f]
///    with \f$\hat{\mathbf n}\f$ the **nadir** direction in body axes and \f$n\f$
///    the orbital mean motion. It is the same formula the truth side evaluates
///    (`sim/world/gravity_gradient`, §5.3) — on *estimated* rather than true
///    states, which is the whole content of "model-based feedforward".
///  * **Residual dipole**, \f$\boldsymbol\tau_{m} = \mathbf m_{res}\times\mathbf
///    B\f$, from a configured body-fixed residual moment and the onboard field.
///
/// Both are exact functions of their inputs, so they are free functions rather
/// than an estimator: nothing is fitted, nothing is stored, and a cycle missing an
/// input simply does not get that term.
///
/// **Tier 2 — the momentum-based residual-torque observer.** The vehicle's total
/// angular momentum in body axes is \f$\mathbf H = J\boldsymbol\omega +
/// \bar{\mathbf h}_w\f$ (Wie §7 [wie2008]), and Euler's equation for the whole
/// system is \f$\dot{\mathbf H} + \boldsymbol\omega\times\mathbf H =
/// \boldsymbol\tau_{ext}\f$. Everything on the left is *measured* — the gyro
/// gives \f$\boldsymbol\omega\f$, the wheel tachometers give
/// \f$\bar{\mathbf h}_w\f$ — so differencing it gives the external torque
/// directly, and subtracting the tier-1 model leaves the **unmodelled** part:
/// \f[
///   \boldsymbol\tau_{res} = \frac{\mathbf H_k - \mathbf H_{k-1}}{\Delta t}
///     + \boldsymbol\omega\times\mathbf H \;-\; \boldsymbol\tau_{model}.
/// \f]
/// That difference quotient is dominated by tachometer and gyro noise at a single
/// 10 Hz step, which is exactly why the estimate is the **low-pass** of it: a
/// first-order filter of time constant \f$\tau_{lp}\f$ (chosen far below the
/// orbital period and far above the control bandwidth) whose steady state is the
/// secular torque and whose response to noise is \f$\sqrt{\Delta t/2\tau_{lp}}\f$
/// of it. Internal torques cancel identically in \f$\mathbf H\f$ — a wheel that
/// spins up takes its momentum from the body, and the sum does not move — so what
/// the observer sees is external by construction, which is the reason to observe
/// momentum rather than wheel speed alone.
///
/// **One estimator, two consumers (§8.5, §9).** The same
/// \f$\hat{\boldsymbol\tau}\f$ is fed forward into the control demand *and* gated
/// as the §9 **momentum-anomaly monitor**: a secular external torque far outside
/// the modelled disturbance budget is the anomaly signature — a stuck thruster, a
/// deployment that did not latch, a residual dipole an order of magnitude past the
/// magnetic-cleanliness allocation. Building a second estimator for the monitor
/// would be two answers to one question.
///
/// **What the monitor must *not* fire on** is the modelled environment itself.
/// The gravity-gradient torque on an Earth-pointing vehicle is a large signal at
/// twice the orbital frequency; it is subtracted as tier 1 before the filter, so
/// it appears in \f$\hat{\boldsymbol\tau}\f$ only to the extent the onboard model
/// is wrong. Feeding the observer an unsubtracted model would make the monitor
/// alarm once per half orbit on a perfectly healthy vehicle — the caller is
/// therefore required to pass the modelled torque it is *also* feeding forward,
/// and the two cannot disagree because they are the same vector.
///
/// **The MEKF's gyro-bias states are not this** (§8.5). A gyro bias is a *sensor*
/// error: it moves the measured rate and moves nothing else. An external torque
/// moves stored momentum. The two are distinguishable precisely because the wheels
/// are in this estimator and not in that one.
///
/// **Frames, units, conventions.** Torques `Vec3<Body>` in N·m, momentum
/// `Vec3<Body>` in N·m·s, rates rad/s, inertia kg·m², field tesla, dipole A·m²;
/// time tags TAI nanoseconds (§3.2). SI throughout.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Hughes, *Spacecraft Attitude Dynamics*, 1986, §5.2 (gravity-gradient
///    torque) [hughes1986].
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, 1978, §17.2
///    (environmental torques) [wertz1978].
///  - Wie, *Space Vehicle Dynamics and Control*, 2nd ed., 2008, §7 (system
///    momentum with wheels) [wie2008].
///  - Design doc §8.5 (feedforward tiers), §9 (the momentum-anomaly monitor),
///    §5.3 (the truth-side torques this is the onboard twin of).

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Gravity-gradient torque on a rigid body [N·m], body axes.
///
/// @param inertia_kgm2       body-frame inertia tensor.
/// @param nadir_unit_body    unit vector from the spacecraft toward the Earth's
///                           centre, body axes. Not normalised here: a caller
///                           that cannot produce a unit vector has no geometry,
///                           and normalising a null vector silently would invent
///                           one.
/// @param orbit_rate_rad_s   mean motion \f$n\f$ [rad/s].
/// @param out                receives the torque; untouched on refusal.
/// @return false on a non-finite input, a non-unit nadir (outside 1e-6) or a
///         non-positive rate — the cycle then gets no gravity-gradient term
///         rather than a guessed one.
bool gravityGradientTorque(const Eigen::Matrix3d& inertia_kgm2,
                           const math::Vec3<math::frames::Body>& nadir_unit_body,
                           double orbit_rate_rad_s, math::Vec3<math::frames::Body>& out);

/// Residual-dipole torque \f$\mathbf m_{res}\times\mathbf B\f$ [N·m], body axes.
///
/// @param residual_dipole_am2 the vehicle's body-fixed residual moment [A·m²]
///        (§19.1 `residual_dipole_am2`; a magnetic-cleanliness allocation, not a
///        fitted state — fitting it is tier 3).
/// @param field_tesla the onboard field estimate [T], body axes.
/// @param out receives the torque; untouched on refusal.
/// @return false on a non-finite input.
bool residualDipoleTorque(const math::Vec3<math::frames::Body>& residual_dipole_am2,
                          const math::Vec3<math::frames::Body>& field_tesla,
                          math::Vec3<math::frames::Body>& out);

/// Why a @ref DisturbanceObserver::update produced no estimate.
enum class DisturbanceRefusal : std::uint8_t {
  kNone = 0,          ///< an estimate was produced
  kUnconfigured,      ///< the config failed @ref DisturbanceObserverConfig::isValid
  kBadInput,          ///< non-finite momentum, rate or modelled torque
  kNoPreviousSample,  ///< first cycle: nothing to difference against
  kNonMonotonicTime,  ///< the time tag did not advance (stuck or backwards clock)
  kStepTooLong,       ///< gap past `max_dt_s`; the anchor is re-taken
  kNonFiniteOutput,   ///< the filtered estimate failed its finiteness guard
};

/// Observer tuning. No in-code defaults (§19.3).
struct DisturbanceObserverConfig {
  /// Low-pass time constant \f$\tau_{lp}\f$ [s]. Far above the control period
  /// (so the difference quotient's noise is averaged down) and far below the
  /// orbital period (so a real secular torque is still tracked).
  double tau_s = 0.0;

  /// Longest step [s] the difference quotient is formed over. A larger gap is a
  /// dropout: the anchor is re-taken and no estimate is produced, rather than a
  /// momentum change being divided by a step the vehicle did not run.
  double max_dt_s = 0.0;

  /// \f$\|\hat{\boldsymbol\tau}\|\f$ [N·m] above which the §9 momentum anomaly is
  /// declared. A **budget** value: the largest unmodelled secular torque the
  /// design allocates, with margin — not a number fitted to a measurement.
  double anomaly_torque_nm = 0.0;

  /// \f$\|\hat{\boldsymbol\tau}\|\f$ [N·m] below which, held for
  /// @ref anomaly_cycles consecutive updates, the anomaly clears. Must be at or
  /// below @ref anomaly_torque_nm: the deadband is what stops an estimate parked
  /// at the budget — which is what a real fault at the margin looks like through
  /// the low-pass — from cycling the anomaly once per confirmation count, the
  /// same argument @ref MomentumConfig makes for its enter/exit pair. Between
  /// the two thresholds the latch holds its state.
  double anomaly_clear_nm = 0.0;

  /// Consecutive updates over @ref anomaly_torque_nm that latch the anomaly, and
  /// consecutive updates under @ref anomaly_clear_nm that clear it. The same
  /// count both ways: an exclusion whose release needs more evidence than its
  /// trigger is a life sentence.
  std::uint32_t anomaly_cycles = 0;

  /// All values present and positive, the clear threshold at or below the latch
  /// threshold, and `anomaly_cycles` non-zero.
  bool isValid() const;
};

/// One observer update's product.
struct DisturbanceResult {
  /// Filtered external-torque estimate \f$\hat{\boldsymbol\tau}\f$ [N·m], body
  /// axes — the *unmodelled* part, since the caller's modelled torque is
  /// subtracted before the filter.
  math::Vec3<math::frames::Body> torque_nm{};

  /// This cycle's unfiltered residual [N·m]. Diagnostic: it is what the filter
  /// consumed, and the two together say whether an excursion is real or a step.
  math::Vec3<math::frames::Body> raw_torque_nm{};

  /// Interval the difference quotient used [s].
  double dt_s = 0.0;

  /// The §9 momentum anomaly is latched.
  bool anomaly = false;

  /// @ref torque_nm is usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  DisturbanceRefusal refusal = DisturbanceRefusal::kUnconfigured;
};

/// Momentum-based residual-torque observer (§8.5 tier 2) and the §9
/// momentum-anomaly monitor it doubles as.
class DisturbanceObserver {
 public:
  DisturbanceObserver() = default;

  /// Build with @p config. An invalid config leaves the observer **inert**.
  explicit DisturbanceObserver(const DisturbanceObserverConfig& config);

  bool isConfigured() const { return configured_; }

  /// Fold in one cycle.
  ///
  /// @param total_momentum_nms \f$\mathbf H = J\boldsymbol\omega + \bar{\mathbf
  ///        h}_w\f$, body axes [N·m·s]. The caller forms it, because the caller is
  ///        the one holding the inertia tensor and the wheel array.
  /// @param body_rate_radps the estimated body rate [rad/s], for the
  ///        \f$\boldsymbol\omega\times\mathbf H\f$ term.
  /// @param modelled_torque_nm the tier-1 torque being fed forward this cycle
  ///        [N·m]. Pass zero to observe the *total* external torque.
  /// @param time_tag_tai_ns TAI ns of this cycle.
  /// @param out receives the estimate and the diagnostics.
  /// @return true when @p out carries a usable estimate. False on the first
  ///         cycle and on every refusal; the running estimate is **held**, not
  ///         zeroed, because a missed cycle is not evidence the disturbance
  ///         stopped.
  bool update(const math::Vec3<math::frames::Body>& total_momentum_nms,
              const math::Vec3<math::frames::Body>& body_rate_radps,
              const math::Vec3<math::frames::Body>& modelled_torque_nm,
              std::int64_t time_tag_tai_ns, DisturbanceResult& out);

  /// The running estimate [N·m], body axes. Zero until the first accepted update.
  const math::Vec3<math::frames::Body>& estimate() const { return estimate_; }

  /// The §9 anomaly latch.
  bool anomaly() const { return anomaly_; }

  /// An estimate has converged enough to be worth feeding forward — i.e. at least
  /// one update has been accepted. Before that the estimate is a zero that means
  /// "unknown", and feeding it forward is a no-op either way; the flag exists so
  /// telemetry does not report an unknown as a measurement.
  bool hasEstimate() const { return have_estimate_; }

  /// Drop the estimate, the anchor and the anomaly latch (mode entry, commanded
  /// reset, or a tuning change that invalidates the filter's history).
  void reset();

 private:
  DisturbanceObserverConfig config_{};
  bool configured_ = false;
  bool have_previous_ = false;
  bool have_estimate_ = false;
  bool anomaly_ = false;
  math::Vec3<math::frames::Body> estimate_{Eigen::Vector3d::Zero()};
  math::Vec3<math::frames::Body> previous_momentum_{Eigen::Vector3d::Zero()};
  std::int64_t previous_time_ns_ = 0;
  std::uint32_t above_streak_ = 0;
  std::uint32_t below_streak_ = 0;
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_DISTURBANCE_HPP

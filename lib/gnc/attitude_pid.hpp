#ifndef POLARIS_GNC_ATTITUDE_PID_HPP
#define POLARIS_GNC_ATTITUDE_PID_HPP

/// @file
/// @brief Quaternion-error PID attitude controller (design doc §8.5;
/// REQ-ACTL-002).
///
/// Body-torque command from an attitude estimate and a reference attitude, the
/// standard quaternion-feedback regulator (Wie, Weiss & Arapostathis
/// [wie1989]; Markley & Crassidis §7.2 [markley2014]):
/// \f[
///   \boldsymbol\tau \;=\; K_p\,\delta\boldsymbol\theta
///        \;+\; K_i \!\int\! \delta\boldsymbol\theta\,dt
///        \;+\; K_d\,(\boldsymbol\omega_\mathrm{ref} - \hat{\boldsymbol\omega}).
/// \f]
///
/// **The error rotation.** With \f$\hat{\bar q}\f$ the estimate (Body←ECI) and
/// \f$\bar q_\mathrm{ref}\f$ the reference (Body_ref←ECI), the error quaternion is
/// \f$\delta\bar q = \bar q_\mathrm{ref}\otimes\hat{\bar q}^{-1}\f$ — the rotation
/// that carries the *current* body frame onto the *reference* one — and the error
/// rotation vector is \f$\delta\boldsymbol\theta = 2\,\mathrm{sgn}(\delta q_0)\,
/// \delta\mathbf q_v\f$, expressed in body axes to first order. The sign factor is
/// what makes the law take the **short way round**: \f$\delta\bar q\f$ and
/// \f$-\delta\bar q\f$ are the same rotation, and without it a 181° error is
/// driven the long way, which is the classic quaternion-feedback unwinding bug
/// [wie1989 §III]. The reported error *angle* is the `atan2` form
/// \f$2\,\mathrm{atan2}(\|\delta\mathbf q_v\|, |\delta q_0|)\f$, never
/// \f$2\arccos\f$ — the house rule (lib/README.md, §3.3), and it matters here
/// because the interesting regime is exactly where `acos` is worst.
///
/// **Sign.** \f$\delta\boldsymbol\theta\f$ is the rotation the body must still
/// perform, so the proportional term is **positive**: a torque along
/// \f$+\delta\boldsymbol\theta\f$ accelerates the body in the sense that closes the
/// error. With \f$\boldsymbol\omega_\mathrm{ref} = 0\f$ (inertial hold) the
/// derivative term reduces to \f$-K_d\hat{\boldsymbol\omega}\f$, i.e. rate damping.
///
/// **Anti-windup.** The integrator is clamped per axis to a configured bound and
/// is **frozen while the torque command is saturated** (conditional integration).
/// The clamp alone is not enough: against a constant disturbance larger than the
/// available torque the integrator would sit pinned at the clamp and then take as
/// long to unwind as it took to fill, which is the overshoot every windup story
/// ends with. Freezing on saturation is the cheap standard fix (Åström & Murray
/// §11.4 [astrom2008]); the clamp bounds the state even when saturation is never
/// reached.
///
/// **Slew-rate limit: saturate the commanded rate, not only the torque.** With
/// \f$K_p/K_d\f$ set for a small-angle bandwidth, a large error asks for a rate
/// the vehicle cannot carry: on the reference vehicle \f$K_p\,\delta\theta\f$ at
/// 100° is ~85× the torque limit, so a POINT entered from a tumble or a large
/// slew is a bang-bang manoeuvre with the wheels pinned, momentum leaving its
/// envelope and the estimator's fine mode demoted a dozen times on the way in
/// (measured, Push 67). Wie & Lu's rate-limited eigenaxis form [wie1995] fixes
/// it: the proportional term is written as a commanded rate
/// \f$\boldsymbol\omega_c = \mathrm{sat}_{\omega_{max}}\big((K_p/K_d)\,
/// \delta\boldsymbol\theta\big)\f$, saturated by *norm* so the eigenaxis is kept,
/// and the torque is \f$K_d(\boldsymbol\omega_\mathrm{ref}+\boldsymbol\omega_c-
/// \hat{\boldsymbol\omega})\f$ — identical to the PID inside the limit, a
/// constant-rate eigenaxis slew with pure rate damping outside it. A limited
/// cycle freezes the integrator like a saturated one: the error is large by
/// construction and would only wind it.
///
/// **Torque saturation preserves direction.** Unlike the B-dot law (see
/// `gnc/bdot.hpp`), the commanded torque is a *pointing* command: clipping one
/// axis rotates the commanded torque away from the direction the error asked for,
/// which can produce a slew about the wrong axis. The whole vector is therefore
/// scaled by one factor, so an over-demand becomes a slower correction along the
/// right axis rather than a faster one along the wrong axis.
///
/// **Gains are inputs.** \f$K_p, K_i, K_d\f$, the integrator clamp and the torque
/// limit are mission configuration (§19.3); a default-constructed config fails
/// validation and leaves the controller inert.
///
/// **Frames, units, conventions.** Quaternions are JPL scalar-first with
/// \f$q_0\ge0\f$ (§3.3); rates are `Vec3<Body>` in rad/s; torque is `Vec3<Body>` in
/// N·m; the step is seconds. SI throughout.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Wie, Weiss & Arapostathis, "Quaternion Feedback Regulator for Spacecraft
///    Eigenaxis Rotations," J. Guidance, Control & Dynamics 12(3):375–380, 1989
///    [wie1989].
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §7.2 (attitude control laws) [markley2014].
///  - Åström & Murray, *Feedback Systems*, 2008, §11.4 (integrator windup)
///    [astrom2008].
///  - Wie & Lu, "Feedback Control Logic for Spacecraft Eigenaxis Rotations
///    Under Slew Rate and Control Constraints," J. Guidance, Control & Dynamics
///    18(6):1372–1379, 1995 [wie1995].

#include <cstdint>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Why an `AttitudePid::update` produced no torque.
enum class AttitudePidRefusal : std::uint8_t {
  kNone = 0,         ///< a torque was produced
  kUnconfigured,     ///< built with an invalid config
  kBadInput,         ///< non-finite/non-unit quaternion or non-finite rate/dt (an out-of-range
                     ///< but finite dt only suppresses integration; see `dt_s`)
  kNonFiniteOutput,  ///< the computed torque failed its finiteness guard
};

/// PID tuning. No in-code defaults (§19.3).
struct AttitudePidConfig {
  /// Proportional gain on the error rotation vector [N·m/rad].
  double kp_nm_per_rad = 0.0;

  /// Integral gain on the accumulated error rotation [N·m/(rad·s)].
  double ki_nm_per_rad_s = 0.0;

  /// Derivative gain on the rate error [N·m/(rad/s)].
  double kd_nm_per_radps = 0.0;

  /// Per-axis clamp on the integrator state [rad·s]. Bounds the integral term at
  /// `ki * max_integral_rad_s`.
  double max_integral_rad_s = 0.0;

  /// Magnitude limit on the commanded body torque [N·m]. The vehicle's usable
  /// three-axis authority, not one wheel's rating — the allocation layer
  /// (`gnc/rw_allocation.hpp`) applies the per-wheel limits.
  double max_torque_nm = 0.0;

  /// Longest integration step accepted [s]. A gap larger than this is a dropout,
  /// not a control cycle: the proportional/derivative terms are still valid but
  /// the integral is not advanced across it.
  double max_dt_s = 0.0;

  /// Slew-rate limit [rad/s]: the norm of the commanded rate `(kp/kd)·δθ` is
  /// saturated here (file header). Sized against the torque budget — the rate
  /// damping `kd · max_slew_rate_radps` must leave torque to track with — and
  /// against what the estimator's fine mode holds through.
  double max_slew_rate_radps = 0.0;

  /// Gains non-negative and finite, `kp`/`kd` positive, limits, `max_dt_s` and
  /// `max_slew_rate_radps` positive. `ki` may be zero (a PD controller), and
  /// then `max_integral_rad_s` is unused but must still be non-negative.
  bool isValid() const;
};

/// One control cycle's product.
struct AttitudePidResult {
  /// Commanded body torque [N·m]. Meaningful only when @ref valid.
  math::Vec3<math::frames::Body> torque_nm{};

  /// Error rotation vector \f$\delta\boldsymbol\theta\f$ [rad], body axes.
  math::Vec3<math::frames::Body> attitude_error_rad{};

  /// Rate error \f$\boldsymbol\omega_\mathrm{ref}-\hat{\boldsymbol\omega}\f$ [rad/s].
  math::Vec3<math::frames::Body> rate_error_radps{};

  /// Pointing error angle [rad] — the `atan2` form, exact near zero.
  double error_angle_rad = 0.0;

  /// The unsaturated demand exceeded `max_torque_nm` and was scaled down.
  bool saturated = false;

  /// Magnitude of the demand **before** saturation [N·m].
  ///
  /// Carried because @ref torque_nm is the scaled-down command, so on a
  /// saturated cycle its norm is `max_torque_nm` by construction — reporting it
  /// tells an operator that the limit was reached and nothing about *how far
  /// over* the vehicle was asked to go, which is the whole diagnostic content.
  /// Equal to `torque_nm.norm()` when not saturated.
  double demand_nm = 0.0;

  /// The commanded rate `(kp/kd)·δθ` exceeded `max_slew_rate_radps` and was
  /// saturated: this cycle is a rate-limited eigenaxis slew, not a PID cycle.
  bool rate_limited = false;

  /// @ref torque_nm is usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  AttitudePidRefusal refusal = AttitudePidRefusal::kUnconfigured;
};

/// Quaternion-error PID. Holds the integrator across cycles.
///
/// Usage: construct with the tuning, call @ref update once per control cycle with
/// the estimate, the reference and the elapsed step, and command
/// @ref AttitudePidResult::torque_nm. Call @ref reset on mode entry and whenever
/// the reference jumps, so the integrator does not carry an error accumulated
/// against a target that no longer exists.
class AttitudePid {
 public:
  AttitudePid() = default;

  /// Build with @p config. An invalid config leaves the controller **inert**.
  explicit AttitudePid(const AttitudePidConfig& config);

  bool isConfigured() const { return configured_; }

  const AttitudePidConfig& config() const { return config_; }

  /// Compute this cycle's torque command.
  ///
  /// @param q_est      attitude estimate, Body←ECI.
  /// @param rate_est   estimated body rate [rad/s].
  /// @param q_ref      reference attitude, Body_ref←ECI. Typed as Body←ECI
  ///                   because the reference frame *is* the commanded body frame;
  ///                   the error quaternion is what distinguishes them.
  /// @param rate_ref   reference body rate [rad/s], body axes. Zero for an
  ///                   inertial hold.
  /// @param dt_s       time since the last accepted cycle [s], for the integrator.
  ///                   Non-positive or above `max_dt_s` suppresses integration
  ///                   for this cycle but still produces a PD command.
  /// @param out        receives the command and the diagnostics.
  /// @return true when @p out carries a usable torque; false leaves a zero torque.
  bool update(const math::Quat<math::frames::Body, math::frames::ECI>& q_est,
              const math::Vec3<math::frames::Body>& rate_est,
              const math::Quat<math::frames::Body, math::frames::ECI>& q_ref,
              const math::Vec3<math::frames::Body>& rate_ref, double dt_s, AttitudePidResult& out);

  /// As above, with a **feedforward** body torque added to the demand.
  ///
  /// @param feedforward_nm torque [N·m] added to the PID terms *before* the
  ///        saturation test and therefore before the anti-windup decision. That
  ///        ordering is the point of putting it here rather than at the caller:
  ///        feedforward that is added after saturation can command more torque
  ///        than the vehicle has, and feedforward the integrator cannot see makes
  ///        the integrator fill against a disturbance the feedforward is already
  ///        cancelling — the two would fight, and the integrator would win slowly.
  ///        The §8.5 disturbance feedforward passes **minus** the estimated
  ///        disturbance (`gnc/disturbance.hpp`), since the demand is the torque
  ///        the actuators must produce.
  ///
  /// The five-argument overload above is this one with zero feedforward.
  bool update(const math::Quat<math::frames::Body, math::frames::ECI>& q_est,
              const math::Vec3<math::frames::Body>& rate_est,
              const math::Quat<math::frames::Body, math::frames::ECI>& q_ref,
              const math::Vec3<math::frames::Body>& rate_ref,
              const math::Vec3<math::frames::Body>& feedforward_nm, double dt_s,
              AttitudePidResult& out);

  /// Zero the integrator.
  void reset();

  /// Current integrator state [rad·s], body axes.
  const math::Vec3<math::frames::Body>& integral() const { return integral_; }

 private:
  AttitudePidConfig config_{};
  bool configured_ = false;
  math::Vec3<math::frames::Body> integral_{Eigen::Vector3d::Zero()};
};

/// The error rotation vector \f$\delta\boldsymbol\theta\f$ [rad] carrying @p q_est
/// onto @p q_ref, expressed in body axes, sign-canonical (short way round).
/// Exposed because it is the piece worth testing directly and because guidance
/// (§8.4) reports the same quantity.
///
/// @param q_est the attitude estimate, Body<-ECI.
/// @param q_ref the reference attitude, Body_ref<-ECI.
/// @param out receives \f$2\,\mathrm{sgn}(\delta q_0)\,\delta\mathbf q_v\f$.
/// @param angle_rad receives the exact error angle \f$2\,\mathrm{atan2}(\|\delta
///        \mathbf q_v\|,|\delta q_0|)\f$ [rad], in [0, π].
/// @return false (leaving the outputs untouched) if either quaternion is
///         non-finite or not of unit norm within 1e-6.
bool attitudeError(const math::Quat<math::frames::Body, math::frames::ECI>& q_est,
                   const math::Quat<math::frames::Body, math::frames::ECI>& q_ref,
                   math::Vec3<math::frames::Body>& out, double& angle_rad);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_ATTITUDE_PID_HPP

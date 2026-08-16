#ifndef POLARIS_GNC_BDOT_HPP
#define POLARIS_GNC_BDOT_HPP

/// @file
/// @brief B-dot magnetic detumble law (design doc §8.5, §7; REQ-ACTL-001).
///
/// Commanded magnetic dipole from the *measured rate of change* of the
/// geomagnetic field in body axes. The control law implemented here is the
/// normalised B-dot of Avanzini & Giulietti [avanzini2012],
/// \f[
///   \mathbf{m} \;=\; -\,\frac{k}{\|\mathbf{B}\|^2}\,\dot{\mathbf{B}},
/// \f]
/// with \f$\mathbf{B}\f$ the body-frame field [T], \f$k\f$ a gain in
/// \f$\mathrm{N\,m\,s}\f$ and \f$\mathbf{m}\f$ the dipole [A·m²]. Normalising by
/// \f$\|\mathbf{B}\|^2\f$ is what makes the closed-loop decay rate independent of
/// field strength — the un-normalised \f$\mathbf{m} = -k\dot{\mathbf{B}}\f$ varies
/// its authority by a factor of ~4 over a polar orbit, which is the difference
/// between saturating at the poles and doing nothing at the equator. Avanzini &
/// Giulietti prove exponential convergence for this form and give the gain floor
/// \f$k \ge 2\,\omega_o\,(1 + \sin\xi)\,J_{\min}\f$ (their Eq. 20), with
/// \f$\omega_o\f$ the orbital rate, \f$\xi\f$ the inclination of the geomagnetic
/// field with respect to the orbit plane and \f$J_{\min}\f$ the smallest principal
/// inertia. The gain is **mission configuration**, not a constant here (§19.3).
///
/// **The derivative comes from successive magnetometer samples, never from the
/// control period.** In body axes \f$\dot{\mathbf{B}}\approx-\boldsymbol\omega
/// \times\mathbf{B}\f$, so the law is a rate feedback that never needs a gyro —
/// but only if the difference quotient is taken over the interval the two samples
/// were actually *tagged* with. Dividing by the nominal 10 Hz period instead
/// scales the whole command by whatever the real interval was, and under the §7
/// MTQ/MAG duty-cycle interlock the usable samples are the quiet-window ones,
/// which do not arrive every cycle. `update` therefore takes the sample's TAI
/// time tag and forms \f$\dot{\mathbf B} = (\mathbf B_k - \mathbf
/// B_{k-1})/(t_k-t_{k-1})\f$, refusing intervals outside a configured band: too
/// short amplifies magnetometer noise without adding information, too long means
/// the field rotated far enough that the secant is no longer the tangent.
///
/// **Duty factor.** The rods are energised only during the on-window of each
/// control period (§7), so the *average* dipole is `duty × commanded`. The law
/// therefore divides the demanded dipole by the duty factor before saturation, so
/// what the vehicle averages over a period is the dipole the gain asked for. This
/// is the §8.5 statement "average dipole authority scaled by the duty factor",
/// enacted where the command is formed rather than left to the tuning.
///
/// **Saturation is the caller's, and deliberately not this law's.** The rods have
/// rated moments and the demand has to be clamped to them, but a clamp is only
/// meaningful in the basis the rods are in — and this law works in body axes,
/// which coincide with the rod axes only when the triad happens to be
/// body-aligned. Clamping here *and* per rod downstream would, on any skewed
/// triad, throw away authority the rods have: the body-axis clamp cuts a
/// component the rods could still have delivered along their own directions. So
/// the law returns the unclamped demand and the component clamps once, per rod
/// (`flight::AttitudeController::runDetumble`).
///
/// That clamp is **componentwise per rod**, not a direction-preserving scale, and
/// that is the part worth keeping: componentwise clamping preserves the *sign* of
/// every component, so the energy rate
/// \f$\dot T = \boldsymbol\omega\cdot(\mathbf m\times\mathbf B) \propto -\sum_i
/// m_i\dot B_i\f$ keeps every term non-positive and a saturated B-dot stays
/// monotonically dissipative. Direction preservation is the right rule for a
/// *pointing* torque (see `gnc/rw_allocation.hpp`) and the wrong one here.
///
/// **Frames, units, conventions.** Field and dipole are `Vec3<Body>` in tesla and
/// A·m²; time tags are TAI nanoseconds (§3.2); SI throughout.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O — the
/// `AttitudeController` component wraps this.
///
/// References:
///  - Avanzini & Giulietti, "Magnetic Detumbling of a Rigid Spacecraft,"
///    J. Guidance, Control & Dynamics 35(4):1326–1334, 2012 [avanzini2012] —
///    the normalised law, its Lyapunov argument and the gain floor.
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §7.5 (magnetic control) [markley2014].
///  - Design doc §8.5 (control), §7 (MTQ/MAG duty-cycle interlock).

#include <cstdint>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Why a `BdotController::update` produced no command. Every refusal is a normal
/// flight condition, not a fault of the call.
enum class BdotRefusal : std::uint8_t {
  kNone = 0,          ///< a dipole was produced
  kUnconfigured,      ///< the controller was built with an invalid config
  kBadInput,          ///< non-finite field, or a non-positive/zero field magnitude
  kNoPreviousSample,  ///< first usable sample: nothing to difference against
  kIntervalTooShort,  ///< sample spacing below `min_sample_dt_s` (noise amplifier)
  kIntervalTooLong,   ///< spacing above `max_sample_dt_s` (secant is not the tangent)
  kNonMonotonicTime,  ///< the sample tag did not advance (stuck or backwards clock)
  kNonFiniteOutput,   ///< the computed dipole failed its finiteness guard
};

/// B-dot tuning. No in-code defaults: a default-constructed config fails
/// @ref isValid and leaves the controller inert (§19.3 — there are no flight
/// defaults, and a detumble law on an invented gain is worse than none).
struct BdotConfig {
  /// Control gain \f$k\f$ [N·m·s]. Set it from the Avanzini & Giulietti floor
  /// \f$2\,\omega_o(1+\sin\xi)J_{\min}\f$ and raise it toward the rods' saturation
  /// limit for a faster decay; the closed-loop rate time constant is \f$J/k\f$
  /// while unsaturated — the commanded dipole is divided by the duty factor
  /// (see `bdot.cpp`), which is exactly what keeps duty out of the average.
  double gain_nms = 0.0;

  /// On-window fraction of the control period, in (0, 1]. The demanded dipole is
  /// divided by it so the *average* over a period is what the gain asked for.
  double duty_factor = 0.0;

  /// Shortest usable spacing between two field samples [s]. Below it the
  /// difference quotient is mostly magnetometer noise.
  double min_sample_dt_s = 0.0;

  /// Longest usable spacing [s]. Above it the two samples no longer bracket a
  /// small rotation and the secant slope is not the field derivative; the stored
  /// sample is dropped and the law re-acquires from the next pair.
  double max_sample_dt_s = 0.0;

  /// All four values present and in range. Checked at construction.
  bool isValid() const;
};

/// One B-dot cycle's product.
struct BdotResult {
  /// Commanded dipole [A·m²] — the value the rods are driven at **during the
  /// on-window**, already divided by the duty factor. **Unclamped**: the caller
  /// clamps to the rods' rated moments in the rod basis (see the file comment).
  /// Meaningful only when @ref valid.
  math::Vec3<math::frames::Body> dipole_am2{};

  /// The measured field derivative [T/s] the command was formed from.
  math::Vec3<math::frames::Body> field_rate_tps{};

  /// Interval [s] between the two samples the derivative used.
  double sample_dt_s = 0.0;

  /// @ref dipole_am2 is usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  BdotRefusal refusal = BdotRefusal::kUnconfigured;
};

/// The B-dot detumble law. Holds one previous field sample across cycles, so it
/// is an object rather than a free function.
///
/// Usage: construct with the tuning, then call @ref update once per **accepted
/// quiet-window magnetometer sample** — not once per control cycle. A cycle whose
/// sample was rejected by the interlock must not call `update` at all, because a
/// corrupted sample differenced against a clean one produces a large, confident,
/// wrong derivative; call @ref reset instead when the gap grows past
/// `max_sample_dt_s` or the mode is left.
class BdotController {
 public:
  BdotController() = default;

  /// Build with @p config. An invalid config leaves the controller **inert**:
  /// every @ref update refuses with @ref BdotRefusal::kUnconfigured.
  explicit BdotController(const BdotConfig& config);

  bool isConfigured() const { return configured_; }

  const BdotConfig& config() const { return config_; }

  /// Fold in one magnetometer sample and produce this interval's dipole command.
  ///
  /// @param field_tesla    body-frame field [T] from the accepted (quiet-window)
  ///                       magnetometer sample.
  /// @param time_tag_tai_ns TAI nanoseconds the sample was taken at.
  /// @param out            receives the command and the diagnostics.
  /// @return true when @p out carries a usable dipole. False on the first sample
  ///         and on every refusal in @ref BdotRefusal; @p out then carries a zero
  ///         dipole, which is the safe command.
  bool update(const math::Vec3<math::frames::Body>& field_tesla, std::int64_t time_tag_tai_ns,
              BdotResult& out);

  /// Drop the stored sample. Call on mode exit, on an interlock outage, and
  /// whenever the caller knows the next sample is not comparable with the last.
  void reset();

  /// A previous sample is stored (the next @ref update can form a derivative).
  bool hasPreviousSample() const { return have_previous_; }

 private:
  BdotConfig config_{};
  bool configured_ = false;
  bool have_previous_ = false;
  math::Vec3<math::frames::Body> previous_field_{};
  std::int64_t previous_time_ns_ = 0;
};

/// Entry/exit hysteresis on body-rate magnitude: the "is the vehicle tumbling"
/// question, answered without chattering at the threshold.
///
/// Two thresholds and a confirmation count, because one threshold plus noise is a
/// mode oscillator. Detumble is **complete** when the rate norm has stayed below
/// @ref RateHysteresisConfig::exit_radps for @ref
/// RateHysteresisConfig::confirm_cycles consecutive cycles; it is **needed** when
/// the rate norm exceeds @ref RateHysteresisConfig::enter_radps once. Between the
/// two the previous verdict stands.
///
/// This push uses it only to *report* completion — autonomous entry into detumble
/// on rate is the Phase-7 mode manager's decision (§10), and this is the predicate
/// it will read.
struct RateHysteresisConfig {
  double enter_radps = 0.0;          ///< above this, the vehicle is tumbling [rad/s]
  double exit_radps = 0.0;           ///< below this (held), detumble is complete [rad/s]
  std::uint32_t confirm_cycles = 0;  ///< consecutive cycles under `exit_radps`

  /// Present, positive, and `exit_radps < enter_radps` (a non-empty deadband).
  bool isValid() const;
};

/// Stateful rate-threshold hysteresis. See @ref RateHysteresisConfig.
class RateHysteresis {
 public:
  RateHysteresis() = default;

  explicit RateHysteresis(const RateHysteresisConfig& config);

  bool isConfigured() const { return configured_; }

  /// Fold in one body-rate magnitude [rad/s].
  ///
  /// @param rate_norm_radps the measured rate magnitude. A non-finite value is
  ///        treated as *no information*: the verdict and the streak are held,
  ///        because a dropped rate is not evidence the vehicle stopped tumbling.
  /// @return true when the vehicle is considered to be tumbling.
  bool update(double rate_norm_radps);

  /// The rate has been under the exit threshold for the full confirmation count.
  bool complete() const { return configured_ && !tumbling_; }

  /// Consecutive cycles currently under the exit threshold.
  std::uint32_t belowStreak() const { return below_streak_; }

  /// Return to the "tumbling, unconfirmed" state (mode entry).
  void reset();

  /// Return to the **confirmed-below** state: the verdict reads false and one
  /// sample above `enter_radps` is enough to flip it. `reset` assumes the unsafe
  /// answer, which is right for detumble — a vehicle of unknown rate is treated
  /// as tumbling — and wrong for a predicate whose "yes" *starts* an actuation,
  /// such as the momentum-desaturation demand (`gnc/momentum.hpp`): there, an
  /// unknown state must not begin by driving the rods.
  void clear();

 private:
  RateHysteresisConfig config_{};
  bool configured_ = false;
  bool tumbling_ = true;
  std::uint32_t below_streak_ = 0;
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_BDOT_HPP

#ifndef POLARIS_GNC_J2_PROPAGATOR_HPP
#define POLARIS_GNC_J2_PROPAGATOR_HPP

/// @file
/// @brief Two-body + J2 propagation of an uploaded state vector (design doc
/// §8.3, §21.4; REQ-ODP-002).
///
/// The second of the two ways a target object's position is known onboard. A
/// **TLE** slot carries mean elements and is propagated by `gnc::Sgp4`, because
/// a TLE is only meaningful under the theory it was fitted with. A **state
/// vector** slot carries an osculating Cartesian state — from a ground OD
/// solution, an operator upload, or another vehicle's own downlink — and is
/// propagated here.
///
/// ## Why J2 and nothing else
///
/// Not an accuracy ceiling reached by omission; a deliberate stopping point.
///
///  - **Drag is excluded on purpose.** It needs a density model, a ballistic
///    coefficient and space-weather inputs *for the target*, none of which an
///    uploaded state vector carries. A drag term with a guessed `B` is not more
///    accurate than no drag term — it is wrong by an amount nobody can bound,
///    which is worse for a propagator whose error must stay predictable.
///  - **J2 is where the return stops being free.** J2 is ~1e-3 of the two-body
///    term; J3 and J2² are ~1e-6. Over the hour-scale horizons a pointing
///    target is propagated across, dropping J2 costs kilometres (it drives nodal
///    regression and apsidal rotation, both secular) while adding J3 saves
///    metres.
///  - **It needs no Earth-orientation data**, which the higher-order field does.
///    A zonal harmonic is axisymmetric, so only the *axis* matters and the
///    evaluation is legitimate in ECI about the true pole — the reasoning
///    `geopotential.hpp` sets out, and the reason that file refuses the same
///    shortcut for tesserals. That keeps this propagator usable through a GNSS
///    and EOP outage, matching `frames::teme_eci` and `gnc::Sgp4`, so a target's
///    position never becomes unavailable for a reason unrelated to the target.
///
/// The honest statement of accuracy is therefore: **exact for the model, and the
/// model omits drag, third bodies and SRP**. A caller that needs better should
/// uplink a fresher state, not ask this for more — and `sigmaGrowth` exists so
/// the growing uncertainty is legible rather than implied.
///
/// ## Integration
///
/// Classical RK4 on fixed sub-steps, the same integrator and the same
/// bounded-loop discipline `OrbitOd` uses, for the same reason: a fixed step
/// makes the cost of a propagation a function of the span alone, which is what
/// a control cycle needs. Propagation is **`const` and order-independent** —
/// every call integrates from the slot's own epoch rather than from wherever the
/// last call left off, so a state never depends on what was asked before it.
/// That is the property `Sgp4` also holds and for the same reason: a pointing
/// request and a telemetry query must not interfere.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen, bounded loops.

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Why a J2 propagation produced no state.
enum class J2Status : unsigned char {
  kOk = 0,
  kNotInitialised,  ///< propagate() before a valid state was set
  kBadState,        ///< non-finite, zero-radius, or sub-surface epoch state
  kSpanTooLong,     ///< the requested span needs more sub-steps than the bound
  kDiverged,        ///< the integration left the domain (impact or escape)
};

const char* toString(J2Status status);

/// Two-body + J2 gravitational acceleration in ECI [m/s²].
///
/// Exposed separately because it is the whole physical content of this file and
/// is worth testing directly against a closed form rather than only through the
/// integrator.
math::Vec3<math::frames::ECI> j2Acceleration(const math::Vec3<math::frames::ECI>& r_eci_m);

/// An uploaded osculating state and the epoch it is valid at.
struct StateVectorSlot {
  time::Tai epoch;
  math::Vec3<math::frames::ECI> position_m;
  math::Vec3<math::frames::ECI> velocity_m_s;
  /// 1-sigma position uncertainty at @ref epoch [m], as uplinked. Carried
  /// rather than assumed because it is the only thing that makes the propagated
  /// uncertainty meaningful, and a ground solution always knows it.
  double sigma_at_epoch_m = 0.0;
  bool valid = false;
};

/// Propagator for one uploaded state vector.
///
/// Holds no integration state between calls — see the header on
/// order-independence — so one instance may serve any number of queries.
class J2Propagator {
 public:
  /// The largest span a single propagate() call will integrate [s].
  ///
  /// Not a tuning knob: it is the bound that makes the sub-step loop finite at
  /// compile time (§3.6). Twelve hours is several revolutions at any altitude a
  /// pointing target occupies, and a request beyond it is refused rather than
  /// silently truncated, because a target position quietly frozen at a bound is
  /// the failure that looks like successful tracking of the wrong thing.
  static constexpr double kMaxSpanSec = 43200.0;

  /// Integration sub-step [s]. RK4 at 10 s holds a LEO orbit to well under a
  /// metre over kMaxSpanSec against a 1 s reference, which is far inside the
  /// model error the omitted forces already carry.
  static constexpr double kStepSec = 10.0;

  static constexpr int kMaxSteps = static_cast<int>(kMaxSpanSec / kStepSec) + 2;

  /// Accept an uploaded state. Returns false (and changes nothing) if the state
  /// is not finite, or sits at or below the Earth's surface.
  bool setState(const StateVectorSlot& slot);

  bool isInitialised() const { return slot_.valid; }

  const StateVectorSlot& state() const { return slot_; }

  /// Propagate to @p t. Integrates from the slot epoch every call.
  ///
  /// Propagates **backwards** as readily as forwards: an uploaded state's epoch
  /// is often in the recent past but need not be, and a target whose epoch is
  /// slightly ahead of the current time is a routine consequence of a
  /// look-ahead upload rather than an error.
  J2Status propagate(const time::Tai& t, math::Vec3<math::frames::ECI>& position_m,
                     math::Vec3<math::frames::ECI>& velocity_m_s) const;

  /// 1-sigma position uncertainty at @p t [m], or a negative value if unknown.
  ///
  /// The uplinked sigma grown by the model error this propagator *knows it is
  /// not modelling* — dominated by drag on a low target — at a documented
  /// per-hour rate. Deliberately crude: the point is not to predict the error
  /// but to make its growth visible, so a pointing solution built on a
  /// twelve-hour-old state vector is legibly worse than one built on a fresh
  /// one instead of looking identical.
  double sigmaAt(const time::Tai& t) const;

 private:
  StateVectorSlot slot_;
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_J2_PROPAGATOR_HPP

#ifndef POLARIS_GNC_TARGET_PROPAGATOR_HPP
#define POLARIS_GNC_TARGET_PROPAGATOR_HPP

/// @file
/// @brief Geopotential propagation of an uploaded state vector (design doc
/// §8.3, §21.4; REQ-ODP-002).
///
/// The second of the two ways a target object's position is known onboard. A
/// **TLE** slot carries mean elements and is propagated by `gnc::Sgp4`, because
/// a TLE is only meaningful under the theory it was fitted with. A **state
/// vector** slot carries an osculating Cartesian state — from a ground OD
/// solution, an operator upload, or another vehicle's own downlink — and is
/// propagated here.
///
/// ## The field is a setting, and it defaults to 8x8
///
/// This propagated two-body + J2 only until Push 82 measured what that cost.
/// Against the same integrator at the same step with only the force model
/// changed, J2 differs from the onboard 8x8 EGM2008 truncation by ~13 m over
/// 900 s for an 8 000 km target and by ~1.3 km over this file's own 12 h
/// horizon; for a 500 km target the 12 h figure is ~2.3 km. The old default was
/// therefore contributing more error than the uncertainty model claimed for
/// *everything* it omits (`kSigmaGrowthMPerHour`, 40 m/h, sized from drag), and
/// the field the vehicle already carries for its own orbit filter costs nothing
/// to reuse — `gnc::geopotential`, the same ~1 kB table `OrbitOd` evaluates at
/// degree 8.
///
/// A zonal-only degree 8 was measured as a middle option and rejected: the
/// tesseral remainder is comparable to or larger than the zonal gain at every
/// span tried, so order 0 is not a cheap way to most of the benefit. It is
/// kept for the reason below, not as an accuracy compromise.
///
///  - **Drag is still excluded on purpose.** It needs a density model, a
///    ballistic coefficient and space-weather inputs *for the target*, none of
///    which an uploaded state vector carries. A drag term with a guessed `B` is
///    not more accurate than no drag term — it is wrong by an amount nobody can
///    bound, which is worse for a propagator whose error must stay predictable.
///  - **Order 0 needs no Earth-orientation data; any order above it does.** A
///    zonal harmonic is axisymmetric, so only the *axis* matters and the
///    evaluation is legitimate about the pole without knowing the Earth's
///    rotation angle — the reasoning `geopotential.hpp` sets out, and the reason
///    that file refuses the same shortcut for tesserals. So the configured
///    order is also the propagator's EOP dependency, and `propagate` takes the
///    Earth orientation as an **optional** argument: absent, it clamps to order
///    0 and evaluates about the ECI Z axis, which is what this file did
///    unconditionally before. That keeps a target's position available through
///    a GNSS and EOP outage — matching `frames::teme_eci` and `gnc::Sgp4` — so
///    it never becomes unavailable for a reason unrelated to the target, and
///    the degradation is now a named, measured step rather than the permanent
///    state of affairs.
///
/// The honest statement of accuracy is therefore: **exact for the model, and the
/// model omits drag, third bodies and SRP**. A caller that needs better should
/// uplink a fresher state, not ask this for more — and `sigmaGrowth` exists so
/// the growing uncertainty is legible rather than implied. With 8x8 flying, that
/// 40 m/h figure is once again sized for what actually dominates: at J2 it
/// understated the gravity truncation alone by roughly ten times.
///
/// ## Integration
///
/// Classical RK4 on fixed sub-steps, the same integrator and the same
/// bounded-loop discipline `OrbitOd` uses, for the same reason: a fixed step
/// makes the cost of a propagation a function of the span alone, which is what
/// a control cycle needs.
///
/// Propagation is **`const` and order-independent**: a state never depends on
/// what was asked before it, which is the property `Sgp4` also holds and for the
/// same reason — a pointing request and a telemetry query must not interfere.
/// That invariant is preserved, but it is no longer implemented by re-deriving
/// everything from the slot epoch on every call, because at 8x8 that stopped
/// being affordable: a 12 h-old state at the 10 Hz GNC rate is 4 320 sub-steps
/// and ~17 000 field evaluations **per cycle**, and each stage above order 0
/// additionally needs an Earth-orientation reduction.
///
/// Instead the integration is carried forward on a **grid anchored at the slot
/// epoch**: retained state only ever sits at `epoch + n * kStepSec`, and a query
/// is answered by advancing that grid to the last whole step at or before @p t
/// and then taking one partial step which is *not* retained. Because the grid
/// points do not depend on which times were asked for, two callers asking in any
/// order get bit-identical answers — pinned by test, not asserted here. A query
/// before the cursor re-seeds from the slot. Steady-state cost falls from the
/// whole span to a single step per cycle.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen, bounded loops.

#include <Eigen/Dense>

#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Why a propagation produced no state.
enum class PropagationStatus : unsigned char {
  kOk = 0,
  kNotInitialised,  ///< propagate() before a valid state was set
  kBadState,        ///< non-finite, zero-radius, or sub-surface epoch state
  kSpanTooLong,     ///< the requested span needs more sub-steps than the bound
  kDiverged,        ///< the integration left the domain (impact or escape)
};

const char* toString(PropagationStatus status);

/// Which geopotential truncation a target slot is propagated with.
///
/// `degree 2, order 0` reproduces two-body + J2 exactly (`geopotential.hpp`
/// says so where the recursion is defined) and is the setting that needs no
/// Earth orientation — it is what this file did unconditionally before Push 82,
/// kept as the deliberate low-fidelity / no-EOP mode rather than as a second
/// code path. Anything with `order > 0` is refused the shortcut and needs the
/// Earth's rotation angle.
struct TargetForceModel {
  int degree = 8;
  int order = 8;

  /// Clamp into the range the compiled coefficient table supports. Clamping
  /// rather than refusing because an out-of-range uplink should fly the nearest
  /// supported field and say so in telemetry, not leave the slot unpropagatable.
  void clampToSupported();

  bool needsEarthOrientation() const { return order > 0; }
};

/// Gravitational acceleration in ECI [m/s²] for @p model.
///
/// Exposed separately because it is the whole physical content of this file and
/// is worth testing directly against a closed form rather than only through the
/// integrator. @p eci_from_ecef is the rotation at the evaluation epoch; pass
/// `nullptr` to evaluate about the ECI Z axis, which is only legitimate at
/// order 0.
math::Vec3<math::frames::ECI> targetAcceleration(const math::Vec3<math::frames::ECI>& r_eci_m,
                                                 const TargetForceModel& model,
                                                 const Eigen::Matrix3d* eci_from_ecef);

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
/// Carries a grid-anchored integration cursor (see the header): the answers do
/// not depend on call order, but the cost of a forward sequence does not repeat
/// the whole span.
class TargetPropagator {
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
  /// is not finite, or sits at or below the Earth's surface. Resets the cursor.
  bool setState(const StateVectorSlot& slot);

  /// Choose the geopotential truncation. Clamped to what the compiled table
  /// supports; resets the cursor, because the retained grid state was
  /// integrated under the old model and mixing the two would produce a
  /// trajectory that is the history of a setting rather than a model.
  void setForceModel(const TargetForceModel& model);

  const TargetForceModel& forceModel() const { return model_; }

  bool isInitialised() const { return slot_.valid; }

  const StateVectorSlot& state() const { return slot_; }

  /// Propagate to @p t.
  ///
  /// @p eop is the Earth-orientation data the configured order needs. Passing
  /// `nullptr` states that none is available, which clamps the evaluation to
  /// order 0 about the ECI Z axis for this call only — the configured model is
  /// not modified, so the slot recovers full fidelity as soon as the tables do.
  ///
  /// Propagates **backwards** as readily as forwards: an uploaded state's epoch
  /// is often in the recent past but need not be, and a target whose epoch is
  /// slightly ahead of the current time is a routine consequence of a
  /// look-ahead upload rather than an error.
  PropagationStatus propagate(const time::Tai& t, const frames::EopValue* eop,
                              math::Vec3<math::frames::ECI>& position_m,
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
  /// Advance the grid cursor to the last whole sub-step at or before @p t,
  /// re-seeding first if @p t precedes it. Leaves the cursor on the grid.
  PropagationStatus advanceGridTo(const time::Tai& t, const frames::EopValue* eop) const;

  void resetCursor() const;

  StateVectorSlot slot_;
  TargetForceModel model_;

  // The integration cursor. Mutable because advancing it is a cache: propagate()
  // is logically const, and the values it can return are fixed by the slot and
  // the model alone — the cursor only decides how much work getting there costs.
  // Grid-anchored, so `cursor_step_` whole steps from the slot epoch is the only
  // state that is ever retained.
  mutable bool cursor_valid_ = false;
  /// Whether the retained state was integrated *with* the Earth orientation
  /// applied. The effective model is the configured one only when EOP was
  /// available, so this is part of the cursor's identity: carrying state
  /// forward across a change would make the trajectory a history of table
  /// availability rather than the result of a model — the same defect
  /// `setForceModel` resets the cursor to avoid, arriving by a different door.
  mutable bool cursor_used_eop_ = false;
  mutable int cursor_step_ = 0;
  mutable Eigen::Vector3d cursor_position_m_{Eigen::Vector3d::Zero()};
  mutable Eigen::Vector3d cursor_velocity_m_s_{Eigen::Vector3d::Zero()};
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_TARGET_PROPAGATOR_HPP

#include "gnc/target_propagator.hpp"

#include <cmath>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "gnc/geopotential.hpp"

namespace polaris::gnc {
namespace {

namespace c = polaris::constants;

/// The J2 coefficients and the GM they were solved with travel together — see
/// `constants.hpp` on what pairing them with the WGS-84 GM instead costs.
constexpr double kMu = c::gravity::kGM;
constexpr double kRe = c::gravity::kReferenceRadius;
constexpr double kJ2 = c::gravity::kJ2;

/// Refuse a state at or below the surface. Bounded by the *equatorial* radius
/// (the largest), so a legitimately low polar pass is not rejected by the
/// ellipsoid's own shape.
constexpr double kMinRadiusM = c::wgs84::kSemiMajorAxis;

/// The uncertainty growth rate the omitted forces represent [m per hour].
///
/// Dominated by drag, which is the largest force this propagator does not
/// model. Sized from a LEO target: a 10 % density error on a typical ballistic
/// coefficient leaves ~1 km of along-track error after a day, so ~40 m/hour, and
/// the along-track direction is where essentially all of it lands. Deliberately
/// one scalar and deliberately pessimistic for a high target — the number's job
/// is to make staleness visible to an operator, not to be a covariance.
constexpr double kSigmaGrowthMPerHour = 40.0;
constexpr double kSecondsPerHour = 3600.0;

/// The rotation taking ECEF to ECI at @p t, or std::nullopt when the reduction
/// refuses the epoch. Computed per RK4 stage rather than once per span: it is
/// the Earth's rotation angle that the tesseral terms are a function of, and at
/// 10 s per sub-step it turns 0.04 deg between stages. Affordable only because
/// the grid cursor means a steady-state cycle takes one step, not the span.
bool eciFromEcefAt(const time::Tai& t, const frames::EopValue& eop, Eigen::Matrix3d& out) {
  math::Quat<math::frames::ECI, math::frames::ECEF> q;
  if (!frames::eciFromEcef(t, eop, q)) {
    return false;
  }
  out = q.core().toRotationMatrix();
  return true;
}

}  // namespace

const char* toString(PropagationStatus status) {
  switch (status) {
    case PropagationStatus::kOk:
      return "OK";
    case PropagationStatus::kNotInitialised:
      return "NOT_INITIALISED";
    case PropagationStatus::kBadState:
      return "BAD_STATE";
    case PropagationStatus::kSpanTooLong:
      return "SPAN_TOO_LONG";
    case PropagationStatus::kDiverged:
      return "DIVERGED";
  }
  return "UNKNOWN";
}

void TargetForceModel::clampToSupported() {
  if (degree < 0)
    degree = 0;
  if (degree > kGeopotentialMaxDegree)
    degree = kGeopotentialMaxDegree;
  if (order < 0)
    order = 0;
  if (order > degree)
    order = degree;
}

math::Vec3<math::frames::ECI> targetAcceleration(const math::Vec3<math::frames::ECI>& r_eci_m,
                                                 const TargetForceModel& model,
                                                 const Eigen::Matrix3d* eci_from_ecef) {
  TargetForceModel m = model;
  m.clampToSupported();
  // Without a rotation the field can only be evaluated about the ECI Z axis,
  // which is legitimate for a zonal (axisymmetric: only the axis matters) and
  // wrong in longitude by the whole Earth-rotation angle for anything else. So
  // the absence of the rotation *is* the clamp to order 0, stated here rather
  // than trusted to the caller.
  if (eci_from_ecef == nullptr) {
    m.order = 0;
    return math::Vec3<math::frames::ECI>(
        geopotentialAcceleration(r_eci_m.eigen(), m.degree, m.order, kMu, kRe));
  }
  const Eigen::Vector3d r_ecef = eci_from_ecef->transpose() * r_eci_m.eigen();
  const Eigen::Vector3d a_ecef = geopotentialAcceleration(r_ecef, m.degree, m.order, kMu, kRe);
  return math::Vec3<math::frames::ECI>(*eci_from_ecef * a_ecef);
}

void TargetPropagator::resetCursor() const {
  cursor_valid_ = false;
  cursor_used_eop_ = false;
  cursor_step_ = 0;
}

void TargetPropagator::setForceModel(const TargetForceModel& model) {
  model_ = model;
  model_.clampToSupported();
  resetCursor();
}

bool TargetPropagator::setState(const StateVectorSlot& slot) {
  if (!slot.position_m.isFinite() || !slot.velocity_m_s.isFinite()) {
    return false;
  }
  if (!std::isfinite(slot.sigma_at_epoch_m) || slot.sigma_at_epoch_m < 0.0) {
    return false;
  }
  if (!(slot.position_m.norm() > kMinRadiusM)) {
    return false;
  }
  slot_ = slot;
  slot_.valid = true;
  resetCursor();
  return true;
}

namespace {

/// One RK4 sub-step of @p h seconds from (r, v) at @p t0. Returns false when the
/// Earth-orientation reduction the model needs refuses one of the stage epochs.
bool rk4Step(const time::Tai& t0, double h, const TargetForceModel& model,
             const frames::EopValue* eop, Eigen::Vector3d& r, Eigen::Vector3d& v) {
  const auto at = [&t0](double dt) {
    return time::Tai::fromNanosecondsSinceEpoch(t0.nanosecondsSinceEpoch() +
                                                static_cast<std::int64_t>(dt * 1.0e9));
  };
  // One reduction per sub-step, at its midpoint, shared by all four stages —
  // not one per stage. The reduction is the expensive part (measured: 420 ms to
  // catch up a 12 h-old state with three per step, 140 ms with one), and what
  // it buys per stage is a rotation-angle correction of at most half a sub-step,
  // 0.02 deg of longitude. That phase error lands only on the *tesseral* terms,
  // which are ~1e-6 of the acceleration, so it is nine orders below the model
  // error the omitted forces already carry. Order 0 needs no reduction at all,
  // which is the whole EOP-free mode.
  Eigen::Matrix3d mid;
  const Eigen::Matrix3d* rot = nullptr;
  if (model.needsEarthOrientation() && eop != nullptr) {
    if (!eciFromEcefAt(at(0.5 * h), *eop, mid)) {
      return false;
    }
    rot = &mid;
  }
  const Eigen::Matrix3d* p0 = rot;
  const Eigen::Matrix3d* ph = rot;
  const Eigen::Matrix3d* p1 = rot;
  const auto acc = [&model](const Eigen::Vector3d& x, const Eigen::Matrix3d* frame) {
    return targetAcceleration(math::Vec3<math::frames::ECI>(x), model, frame).eigen();
  };
  const Eigen::Vector3d k1v = acc(r, p0);
  const Eigen::Vector3d k1r = v;
  const Eigen::Vector3d k2v = acc(r + 0.5 * h * k1r, ph);
  const Eigen::Vector3d k2r = v + 0.5 * h * k1v;
  const Eigen::Vector3d k3v = acc(r + 0.5 * h * k2r, ph);
  const Eigen::Vector3d k3r = v + 0.5 * h * k2v;
  const Eigen::Vector3d k4v = acc(r + h * k3r, p1);
  const Eigen::Vector3d k4r = v + h * k3v;
  r += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
  v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
  return true;
}

}  // namespace

PropagationStatus TargetPropagator::advanceGridTo(const time::Tai& t,
                                                  const frames::EopValue* eop) const {
  const double span_s = (t - slot_.epoch).seconds();
  // Grid index of the last whole sub-step at or before t. Negative spans floor
  // downwards, so a backwards propagation walks the same grid the other way and
  // is not a special case.
  const int target_step = static_cast<int>(std::floor(span_s / kStepSec));

  // The effective model for this call: the configured one, degraded to order 0
  // when no Earth orientation is available. Retained state integrated under a
  // different effective model must not be continued — measured before this
  // guard existed, a single 10 s cycle without EOP moved every later answer by
  // 0.4 m and never recovered, because the pollution rides the cursor forward.
  const bool using_eop = model_.needsEarthOrientation() && eop != nullptr;

  // The retained state at step N means "N steps from the seed, taken in the
  // direction of sign(N)". Continuing is therefore only legitimate when the
  // request is *further from the seed in the same direction*; anything else
  // re-seeds. A plain `target_step < cursor_step_` got this wrong for negative
  // spans, which a look-ahead upload makes routine: a walk to step -10 followed
  // by a request at -5 satisfied -5 > -10 and integrated forward from -10
  // instead of re-seeding and walking back 5. RK4 is not time-reversible, so
  // the two paths disagreed — measured at 8.9e-7 m, against a file that
  // promises bit-identical. A change of effective model re-seeds for the same
  // reason: the cursor's identity is (steps, direction, model).
  const bool continues = cursor_valid_ && cursor_used_eop_ == using_eop &&
                         (cursor_step_ == 0 || (target_step >= cursor_step_ && cursor_step_ > 0) ||
                          (target_step <= cursor_step_ && cursor_step_ < 0));
  if (!continues) {
    cursor_step_ = 0;
    cursor_position_m_ = slot_.position_m.eigen();
    cursor_velocity_m_s_ = slot_.velocity_m_s.eigen();
    cursor_valid_ = true;
    cursor_used_eop_ = using_eop;
  }

  // Bounded at the loop, not merely by the caller's span check two frames up
  // (§3.6). `kMaxSteps` is the constant that expresses the bound and was
  // previously defined and never used; a `!=` condition with no counter spins
  // forever rather than overshooting once if the cursor ever leaves range.
  int guard = 0;
  while (cursor_step_ != target_step) {
    if (++guard > kMaxSteps) {
      resetCursor();
      return PropagationStatus::kSpanTooLong;
    }
    const double h = target_step > cursor_step_ ? kStepSec : -kStepSec;
    const time::Tai from = time::Tai::fromNanosecondsSinceEpoch(
        slot_.epoch.nanosecondsSinceEpoch() +
        static_cast<std::int64_t>(static_cast<double>(cursor_step_) * kStepSec * 1.0e9));
    if (!rk4Step(from, h, model_, eop, cursor_position_m_, cursor_velocity_m_s_)) {
      resetCursor();
      return PropagationStatus::kBadState;
    }
    cursor_step_ += target_step > cursor_step_ ? 1 : -1;
    if (!cursor_position_m_.allFinite() || !cursor_velocity_m_s_.allFinite() ||
        !(cursor_position_m_.norm() > kMinRadiusM)) {
      resetCursor();
      return PropagationStatus::kDiverged;
    }
  }
  return PropagationStatus::kOk;
}

PropagationStatus TargetPropagator::propagate(const time::Tai& t, const frames::EopValue* eop,
                                              math::Vec3<math::frames::ECI>& position_m,
                                              math::Vec3<math::frames::ECI>& velocity_m_s) const {
  if (!slot_.valid) {
    return PropagationStatus::kNotInitialised;
  }
  const double span_s = (t - slot_.epoch).seconds();
  if (!std::isfinite(span_s)) {
    return PropagationStatus::kBadState;
  }
  if (std::fabs(span_s) > kMaxSpanSec) {
    return PropagationStatus::kSpanTooLong;
  }

  const PropagationStatus grid = advanceGridTo(t, eop);
  if (grid != PropagationStatus::kOk) {
    return grid;
  }

  // The remainder, taken from the grid state and deliberately **not** retained:
  // what is kept must depend only on the slot and the step count, never on which
  // instants happened to be asked for.
  Eigen::Vector3d r = cursor_position_m_;
  Eigen::Vector3d v = cursor_velocity_m_s_;
  const double grid_time_s = static_cast<double>(cursor_step_) * kStepSec;
  const double tail = span_s - grid_time_s;
  if (tail != 0.0) {
    const time::Tai from = time::Tai::fromNanosecondsSinceEpoch(
        slot_.epoch.nanosecondsSinceEpoch() + static_cast<std::int64_t>(grid_time_s * 1.0e9));
    if (!rk4Step(from, tail, model_, eop, r, v)) {
      return PropagationStatus::kBadState;
    }
  }
  if (!r.allFinite() || !v.allFinite() || !(r.norm() > kMinRadiusM)) {
    // Impact or numerical escape. Reported rather than clamped: a target that
    // has re-entered has no position to point at, and returning the last good
    // one would be indistinguishable from tracking it.
    return PropagationStatus::kDiverged;
  }

  position_m = math::Vec3<math::frames::ECI>(r);
  velocity_m_s = math::Vec3<math::frames::ECI>(v);
  return PropagationStatus::kOk;
}

double TargetPropagator::sigmaAt(const time::Tai& t) const {
  if (!slot_.valid) {
    return -1.0;
  }
  const double span_s = (t - slot_.epoch).seconds();
  if (!std::isfinite(span_s)) {
    return -1.0;
  }
  // Linear in |age|, and symmetric: a state propagated an hour backwards is as
  // uncertain as one propagated an hour forwards.
  return slot_.sigma_at_epoch_m + kSigmaGrowthMPerHour * std::fabs(span_s) / kSecondsPerHour;
}

}  // namespace polaris::gnc

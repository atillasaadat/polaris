#include "gnc/j2_propagator.hpp"

#include <cmath>

#include "constants/constants.hpp"

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

Eigen::Vector3d accelerationEci(const Eigen::Vector3d& r) {
  const double r2 = r.squaredNorm();
  const double rn = std::sqrt(r2);
  const double r3 = r2 * rn;

  // Two-body.
  Eigen::Vector3d a = -kMu / r3 * r;

  // J2. The zonal form in ECI about the true pole: a zonal harmonic is
  // axisymmetric, so only the axis matters and no Earth-rotation angle enters —
  // which is why this needs no EOP and why `geopotential.hpp` refuses the same
  // shortcut for tesseral terms.
  const double zr = r.z() / rn;
  const double k = 1.5 * kJ2 * kMu * kRe * kRe / (r3 * r2);
  const double five_z2 = 5.0 * zr * zr;
  a.x() += k * r.x() * (five_z2 - 1.0);
  a.y() += k * r.y() * (five_z2 - 1.0);
  a.z() += k * r.z() * (five_z2 - 3.0);
  return a;
}

}  // namespace

const char* toString(J2Status status) {
  switch (status) {
    case J2Status::kOk:
      return "OK";
    case J2Status::kNotInitialised:
      return "NOT_INITIALISED";
    case J2Status::kBadState:
      return "BAD_STATE";
    case J2Status::kSpanTooLong:
      return "SPAN_TOO_LONG";
    case J2Status::kDiverged:
      return "DIVERGED";
  }
  return "UNKNOWN";
}

math::Vec3<math::frames::ECI> j2Acceleration(const math::Vec3<math::frames::ECI>& r_eci_m) {
  return math::Vec3<math::frames::ECI>(accelerationEci(r_eci_m.eigen()));
}

bool J2Propagator::setState(const StateVectorSlot& slot) {
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
  return true;
}

J2Status J2Propagator::propagate(const time::Tai& t, math::Vec3<math::frames::ECI>& position_m,
                                 math::Vec3<math::frames::ECI>& velocity_m_s) const {
  if (!slot_.valid) {
    return J2Status::kNotInitialised;
  }
  const double span_s = (t - slot_.epoch).seconds();
  if (!std::isfinite(span_s)) {
    return J2Status::kBadState;
  }
  if (std::fabs(span_s) > kMaxSpanSec) {
    return J2Status::kSpanTooLong;
  }

  Eigen::Vector3d r = slot_.position_m.eigen();
  Eigen::Vector3d v = slot_.velocity_m_s.eigen();

  // Signed sub-steps: the direction of time is the sign of the span, so a
  // backwards propagation is the same code rather than a special case.
  const double direction = span_s < 0.0 ? -1.0 : 1.0;
  const double remaining_total = std::fabs(span_s);
  const int whole = static_cast<int>(remaining_total / kStepSec);
  const double tail = remaining_total - static_cast<double>(whole) * kStepSec;

  for (int i = 0; i <= whole; ++i) {
    const double h = direction * (i < whole ? kStepSec : tail);
    if (h == 0.0) {
      continue;
    }
    // Classical RK4 on (r, v) — the same integrator OrbitOd uses, so a reader
    // comparing the two propagators is comparing models and not methods.
    const Eigen::Vector3d k1v = accelerationEci(r);
    const Eigen::Vector3d k1r = v;
    const Eigen::Vector3d k2v = accelerationEci(r + 0.5 * h * k1r);
    const Eigen::Vector3d k2r = v + 0.5 * h * k1v;
    const Eigen::Vector3d k3v = accelerationEci(r + 0.5 * h * k2r);
    const Eigen::Vector3d k3r = v + 0.5 * h * k2v;
    const Eigen::Vector3d k4v = accelerationEci(r + h * k3r);
    const Eigen::Vector3d k4r = v + h * k3v;

    r += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
    v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

    if (!r.allFinite() || !v.allFinite() || !(r.norm() > kMinRadiusM)) {
      // Impact or numerical escape. Reported rather than clamped: a target that
      // has re-entered has no position to point at, and returning the last good
      // one would be indistinguishable from tracking it.
      return J2Status::kDiverged;
    }
  }

  position_m = math::Vec3<math::frames::ECI>(r);
  velocity_m_s = math::Vec3<math::frames::ECI>(v);
  return J2Status::kOk;
}

double J2Propagator::sigmaAt(const time::Tai& t) const {
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

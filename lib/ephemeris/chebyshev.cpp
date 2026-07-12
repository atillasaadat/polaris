/// @file
/// @brief Onboard Chebyshev ephemeris evaluation (design doc §11.3). See
/// chebyshev.hpp.

#include "ephemeris/chebyshev.hpp"

#include <cmath>

namespace polaris::ephemeris {

namespace {

/// Absolute slack on the normalized argument so a query exactly on an interval
/// boundary (|τ| == 1) is accepted despite floating-point rounding.
constexpr double kTauSlack = 1e-9;

/// Normalized time τ = (t − mid)/radius for a TDB instant, plus the query epoch
/// in seconds. Returns false if the segment is malformed or τ falls outside
/// [−1, 1] (with a small boundary slack).
bool normalizedTime(const ChebyshevSegment& seg, const time::Tdb& t, double& tau) {
  if (!(seg.radius_seconds > 0.0) || seg.degree < 0 || seg.degree > kMaxChebyshevDegree) {
    return false;
  }
  if (!std::isfinite(seg.radius_seconds)) {
    return false;
  }
  // Difference the epoch in int64 ns first, then cast the small, bounded offset
  // to double — keeps full nanosecond precision instead of collapsing the
  // absolute epoch to double and losing ~0.2 µs to cancellation (REQ-CONV-005).
  const double offset_seconds = static_cast<double>(t.nanosecondsSinceEpoch() - seg.mid_ns) /
                                static_cast<double>(time::Duration::kNsPerSecond);
  const double x = offset_seconds / seg.radius_seconds;
  if (!std::isfinite(x) || x < -(1.0 + kTauSlack) || x > (1.0 + kTauSlack)) {
    return false;
  }
  tau = x;
  return true;
}

/// Evaluate Σ c_k T_k(τ) for one component via the forward Chebyshev recurrence.
double series(const double* c, int degree, double tau) {
  // T_0 = 1, T_1 = τ, T_k = 2τ T_{k-1} − T_{k-2}.
  double t_prev = 1.0;
  double t_curr = tau;
  double sum = c[0];
  if (degree >= 1) {
    sum += c[1] * tau;
  }
  for (int k = 2; k <= degree; ++k) {
    const double t_next = 2.0 * tau * t_curr - t_prev;
    sum += c[k] * t_next;
    t_prev = t_curr;
    t_curr = t_next;
  }
  return sum;
}

/// Evaluate the component value and its dτ-derivative Σ c_k T'_k(τ) together.
void seriesWithDerivative(const double* c, int degree, double tau, double& value, double& deriv) {
  // T_0=1, T_1=τ; T'_0=0, T'_1=1; recurrences run in lock-step.
  double t_prev = 1.0;
  double t_curr = tau;
  double d_prev = 0.0;
  double d_curr = 1.0;
  value = c[0];
  deriv = 0.0;
  if (degree >= 1) {
    value += c[1] * tau;
    deriv += c[1];  // c_1 * T'_1 = c_1
  }
  for (int k = 2; k <= degree; ++k) {
    const double t_next = 2.0 * tau * t_curr - t_prev;
    const double d_next = 2.0 * t_curr + 2.0 * tau * d_curr - d_prev;
    value += c[k] * t_next;
    deriv += c[k] * d_next;
    t_prev = t_curr;
    t_curr = t_next;
    d_prev = d_curr;
    d_curr = d_next;
  }
}

}  // namespace

bool ChebyshevSegment::covers(const time::Tdb& t) const {
  double tau;
  return normalizedTime(*this, t, tau);
}

bool evaluate(const ChebyshevSegment& seg, const time::Tdb& t,
              math::Vec3<math::frames::ECI>& pos_out) {
  double tau;
  if (!normalizedTime(seg, t, tau)) {
    return false;
  }
  const math::Vec3<math::frames::ECI> pos(series(seg.cx, seg.degree, tau),
                                          series(seg.cy, seg.degree, tau),
                                          series(seg.cz, seg.degree, tau));
  if (!pos.isFinite()) {  // finite inputs could still overflow a pathological fit
    return false;
  }
  pos_out = pos;
  return true;
}

bool evaluate(const ChebyshevSegment& seg, const time::Tdb& t,
              math::Vec3<math::frames::ECI>& pos_out, math::Vec3<math::frames::ECI>& vel_out) {
  double tau;
  if (!normalizedTime(seg, t, tau)) {
    return false;
  }
  // dτ/dt = 1/radius, so velocity = (dP/dτ) / radius [m/s].
  const double inv_radius = 1.0 / seg.radius_seconds;
  double px, py, pz, dx, dy, dz;
  seriesWithDerivative(seg.cx, seg.degree, tau, px, dx);
  seriesWithDerivative(seg.cy, seg.degree, tau, py, dy);
  seriesWithDerivative(seg.cz, seg.degree, tau, pz, dz);
  const math::Vec3<math::frames::ECI> pos(px, py, pz);
  const math::Vec3<math::frames::ECI> vel(dx * inv_radius, dy * inv_radius, dz * inv_radius);
  if (!pos.isFinite() || !vel.isFinite()) {
    return false;
  }
  pos_out = pos;
  vel_out = vel;
  return true;
}

}  // namespace polaris::ephemeris

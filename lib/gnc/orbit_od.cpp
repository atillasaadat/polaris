/// @file
/// @brief Onboard orbit determination implementation (design doc §8.3). See
/// orbit_od.hpp for the force-model fidelity argument, the Φ/Q derivation, the
/// coast-horizon policy, and the references.

#include "gnc/orbit_od.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "constants/constants.hpp"

namespace polaris::gnc {

namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;

/// Radius below which the force model is not evaluated [m]: a real orbit never
/// reaches the geocentre, and the `1/r²` terms would overflow long before it.
constexpr double kMinRadiusM = 1.0e5;

/// Perturbation floors for the central-difference Jacobian. Each is scaled to
/// the argument (`kJacRelStep` of it) with these as the floor, so the step never
/// collapses to zero for a near-zero velocity component and never loses the
/// argument's exponent for a large one. `1e-6` relative sits near the optimum
/// for a central difference of a smooth double-precision function (the balance
/// of `O(h²)` truncation against `O(ε/h)` round-off is `h ~ ε^{1/3} ≈ 6e-6`).
constexpr double kJacRelStep = 1.0e-6;
constexpr double kJacMinPosStepM = 1.0e-2;
constexpr double kJacMinVelStepMps = 1.0e-6;

/// Force exact symmetry. The algebra is symmetric but the products that build it
/// are not bitwise symmetric, and an asymmetric covariance breaks the Cholesky
/// factorisations downstream consumers do.
void symmetrise(OrbitOd::Covariance& p) {
  p = 0.5 * (p + p.transpose().eval());
}

/// Local geodetic east/north/up basis at ECEF position @p r, columns of the
/// returned matrix in that order.
///
/// Up is **geocentric** (`r̂`) rather than geodetic. The two differ by < 0.2° on
/// an oblate Earth, which is far inside the point of splitting horizontal from
/// vertical error at all — and it is the same approximation
/// `sim/sensors/gnss.cpp` draws the error with, so the filter reconstructs the
/// covariance in the basis the noise was actually realised in rather than in a
/// slightly different one. At the poles east is degenerate; fall back to X.
Eigen::Matrix3d enuBasis(const Eigen::Vector3d& r) {
  Eigen::Vector3d up = r.normalized();
  Eigen::Vector3d east = Eigen::Vector3d::UnitZ().cross(up);
  if (east.norm() < 1.0e-9) {
    east = Eigen::Vector3d::UnitX();
  }
  east.normalize();
  const Eigen::Vector3d north = up.cross(east);
  Eigen::Matrix3d e;
  e.col(0) = east;
  e.col(1) = north;
  e.col(2) = up;
  return e;
}

}  // namespace

bool OrbitOdConfig::isValid() const {
  const bool finite =
      std::isfinite(mu_m3_per_s2) && std::isfinite(zonal_j2) && std::isfinite(reference_radius_m) &&
      std::isfinite(drag_ballistic_coeff_m2_per_kg) && std::isfinite(drag_ref_density_kg_m3) &&
      std::isfinite(drag_ref_altitude_m) && std::isfinite(drag_scale_height_m) &&
      std::isfinite(accel_psd_m2_per_s3) && std::isfinite(position_nis_gate) &&
      std::isfinite(velocity_nis_gate) && std::isfinite(max_coast_s) && std::isfinite(max_dt_s) &&
      std::isfinite(max_step_s) && std::isfinite(min_radius_m) && std::isfinite(max_radius_m);
  if (!finite) {
    return false;
  }
  if (!(mu_m3_per_s2 > 0.0) || zonal_j2 < 0.0 || drag_ballistic_coeff_m2_per_kg < 0.0) {
    return false;
  }
  // A coefficient is only meaningful with the scale it is expressed against, so
  // the reference radius is required whenever a term that uses it is enabled
  // rather than defaulted to something plausible: a zero reference radius would
  // silently kill J2 and put the drag model at an altitude measured from the
  // geocentre. This is the "a field whose zero-default disables a physical
  // effect must be required" rule (P57) applied at the config boundary.
  const bool needs_radius = zonal_j2 > 0.0 || drag_ballistic_coeff_m2_per_kg > 0.0;
  if (needs_radius && !(reference_radius_m > 0.0)) {
    return false;
  }
  if (drag_ballistic_coeff_m2_per_kg > 0.0 &&
      (!(drag_ref_density_kg_m3 > 0.0) || !(drag_scale_height_m > 0.0))) {
    return false;
  }
  if (!(accel_psd_m2_per_s3 > 0.0) || !(position_nis_gate > 0.0) || !(velocity_nis_gate > 0.0)) {
    return false;
  }
  if (!(max_coast_s > 0.0) || !(max_dt_s > 0.0) || !(max_step_s > 0.0)) {
    return false;
  }
  // The sub-step loop is bounded at compile time (§3.6). Rejecting a config the
  // bound would truncate keeps that from ever being a silent short propagation:
  // the failure is at construction, not four orbits into a run.
  if (max_dt_s > static_cast<double>(OrbitOd::kMaxSubsteps) * max_step_s) {
    return false;
  }
  return min_radius_m > 0.0 && max_radius_m > min_radius_m;
}

math::Vec3<math::frames::ECI> onboardAcceleration(const OrbitOdConfig& cfg,
                                                  const math::Vec3<math::frames::ECI>& position,
                                                  const math::Vec3<math::frames::ECI>& velocity,
                                                  const math::Vec3<math::frames::ECI>& pole_eci) {
  const Eigen::Vector3d r = position.eigen();
  const Eigen::Vector3d v = velocity.eigen();
  const Eigen::Vector3d pole = pole_eci.eigen();
  if (!r.allFinite() || !v.allFinite() || !pole.allFinite()) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  const double r_mag = r.norm();
  if (!(r_mag > kMinRadiusM)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }

  // --- Point mass ----------------------------------------------------------
  const double r2 = r_mag * r_mag;
  const double r3 = r2 * r_mag;
  Eigen::Vector3d a = -(cfg.mu_m3_per_s2 / r3) * r;

  // --- J2, about the true pole (see the file header) -----------------------
  // a_J2 = -(3/2) J2 μ Re²/r⁴ [ (1 - 5s²) r̂ + 2s p̂ ], s = r̂·p̂. With p̂ = ẑ this
  // is the textbook component form (Montenbruck & Gill §3.2, Eq. 3.30; Vallado
  // §8.7): the z component reduces to s(3 - 5s²), which is the standard result.
  if (cfg.zonal_j2 > 0.0) {
    const Eigen::Vector3d r_hat = r / r_mag;
    const double s = r_hat.dot(pole);
    const double re = cfg.reference_radius_m;
    const double k = -1.5 * cfg.zonal_j2 * cfg.mu_m3_per_s2 * re * re / (r2 * r2);
    a += k * ((1.0 - 5.0 * s * s) * r_hat + 2.0 * s * pole);
  }

  // --- Exponential-density drag --------------------------------------------
  // The atmosphere co-rotates with the Earth, so the aerodynamically relevant
  // velocity is v - ω⊕ × r; at LEO the transport term is ~465 m/s against a
  // 7.7 km/s orbital velocity, i.e. a ~12% error in the drag magnitude and a
  // real cross-track component if it is dropped. ω⊕ is about the *true* pole,
  // the same axis the zonal field is referenced to.
  if (cfg.drag_ballistic_coeff_m2_per_kg > 0.0) {
    const double altitude = r_mag - cfg.reference_radius_m;
    const double x = (altitude - cfg.drag_ref_altitude_m) / cfg.drag_scale_height_m;
    // Guard the exponential rather than the altitude: a fix below the reference
    // altitude is physically a re-entry, and letting exp() overflow to inf would
    // put a NaN into the state where a large-but-finite density puts a large-
    // but-finite drag the finiteness guard can still reason about.
    const double density = (x > -700.0) ? cfg.drag_ref_density_kg_m3 * std::exp(-x) : 0.0;
    if (std::isfinite(density) && density > 0.0) {
      const Eigen::Vector3d omega = constants::wgs84::kEarthRate * pole;
      const Eigen::Vector3d v_rel = v - omega.cross(r);
      const double v_rel_mag = v_rel.norm();
      a += (-0.5 * cfg.drag_ballistic_coeff_m2_per_kg * density * v_rel_mag) * v_rel;
    }
  }

  if (!a.allFinite()) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  return math::Vec3<math::frames::ECI>(a);
}

bool polarAxisEci(const time::Tai& t, const frames::EopValue& eop,
                  math::Vec3<math::frames::ECI>& out) {
  math::Quat<math::frames::ECI, math::frames::ECEF> q_eci_ecef;
  if (!frames::eciFromEcef(t, eop, q_eci_ecef)) {
    return false;
  }
  const math::Vec3<math::frames::ECI> pole =
      q_eci_ecef.rotate(math::Vec3<math::frames::ECEF>(Eigen::Vector3d::UnitZ()));
  if (!pole.isFinite()) {
    return false;
  }
  out = pole;
  return true;
}

namespace {

/// Derivative of the 6-state `[r; v]` under the onboard force model.
Vec6 stateDerivative(const OrbitOdConfig& cfg, const Vec6& x,
                     const math::Vec3<math::frames::ECI>& pole) {
  Vec6 dx;
  dx.head<3>() = x.tail<3>();
  dx.tail<3>() = onboardAcceleration(cfg, math::Vec3<math::frames::ECI>(x.head<3>()),
                                     math::Vec3<math::frames::ECI>(x.tail<3>()), pole)
                     .eigen();
  return dx;
}

/// Dynamics Jacobian `F = ∂ẋ/∂x` at @p x. The upper blocks are exact; the lower
/// two are central differences of the acceleration (see the file header for why
/// the analytic form was not written by hand).
Eigen::Matrix<double, 6, 6> dynamicsJacobian(const OrbitOdConfig& cfg, const Vec6& x,
                                             const math::Vec3<math::frames::ECI>& pole) {
  Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Zero();
  f.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

  const double r_step = std::max(kJacMinPosStepM, kJacRelStep * x.head<3>().norm());
  const double v_step = std::max(kJacMinVelStepMps, kJacRelStep * x.tail<3>().norm());

  for (int i = 0; i < 3; ++i) {
    Vec6 plus = x;
    Vec6 minus = x;
    plus(i) += r_step;
    minus(i) -= r_step;
    f.block<3, 1>(3, i) =
        (stateDerivative(cfg, plus, pole).tail<3>() - stateDerivative(cfg, minus, pole).tail<3>()) /
        (2.0 * r_step);

    plus = x;
    minus = x;
    plus(3 + i) += v_step;
    minus(3 + i) -= v_step;
    f.block<3, 1>(3, 3 + i) =
        (stateDerivative(cfg, plus, pole).tail<3>() - stateDerivative(cfg, minus, pole).tail<3>()) /
        (2.0 * v_step);
  }
  return f;
}

}  // namespace

OrbitOd::OrbitOd(const OrbitOdConfig& config) : cfg_(config), configured_(config.isValid()) {}

void OrbitOd::dropSolution() {
  position_.setZero();
  velocity_.setZero();
  p_.setZero();
  last_epoch_ = time::Tai{};
  age_s_ = 0.0;
  initialised_ = false;
}

void OrbitOd::reset() {
  dropSolution();
  // Only a *commanded* reset clears the count and the fix-epoch memory. A coast
  // expiry or internal fault drops the solution through dropSolution() and
  // leaves both standing: a filter that just diverged is precisely when FDIR
  // needs to see how many fixes it had been rejecting, and a monotonicity guard
  // that a fault could clear would let a replayed fix back in.
  rejected_ = 0;
  have_fix_ = false;
  last_fix_epoch_ = time::Tai{};
}

OrbitOdRefusal OrbitOd::initialize(const time::Tai& epoch,
                                   const math::Vec3<math::frames::ECI>& position,
                                   const math::Vec3<math::frames::ECI>& velocity,
                                   const Covariance& cov) {
  if (!configured_) {
    return OrbitOdRefusal::kUnconfigured;
  }
  if (!position.isFinite() || !velocity.isFinite() || !cov.allFinite()) {
    return OrbitOdRefusal::kFixNotFinite;
  }
  const double r_mag = position.eigen().norm();
  if (!(r_mag >= cfg_.min_radius_m && r_mag <= cfg_.max_radius_m)) {
    return OrbitOdRefusal::kFixImplausible;
  }
  // The seed covariance is a trust boundary: an indefinite P makes S indefinite,
  // which makes the NIS gate meaningless (a negative NIS would sail through a
  // "too large?" test). Cholesky is the cheap, fixed-size, stack-only check that
  // it is a covariance at all.
  if (cov.llt().info() != Eigen::Success) {
    return OrbitOdRefusal::kFixSigmaInvalid;
  }

  position_ = position.eigen();
  velocity_ = velocity.eigen();
  p_ = cov;
  symmetrise(p_);
  last_epoch_ = epoch;
  age_s_ = 0.0;
  initialised_ = true;
  return OrbitOdRefusal::kNone;
}

OrbitOdRefusal OrbitOd::propagate(const time::Tai& epoch, const frames::EopValue& eop) {
  if (!configured_) {
    return OrbitOdRefusal::kUnconfigured;
  }
  if (!initialised_) {
    return OrbitOdRefusal::kUninitialised;
  }

  // Strictly increasing epochs only. Backwards is obvious; a *stuck* clock is
  // the dangerous one, because a zero-length step followed by an update would
  // fold the same measurement in twice. Both are refused without destroying the
  // solution.
  const double dt_s = (epoch - last_epoch_).seconds();
  if (!(dt_s > 0.0)) {
    return OrbitOdRefusal::kNonMonotonicEpoch;
  }
  if (dt_s > cfg_.max_dt_s) {
    // A gap this long is a clock glitch or a scheduler stall, not a coast. It is
    // refused rather than integrated: the age keeps growing on the *next*
    // accepted step, so a genuine long outage still expires the solution through
    // the horizon below rather than through a step this filter pretended to take.
    return OrbitOdRefusal::kStepTooLong;
  }

  math::Vec3<math::frames::ECI> pole;
  if (!polarAxisEci(epoch, eop, pole)) {
    return OrbitOdRefusal::kFrameConversion;
  }

  const int substeps =
      std::min(kMaxSubsteps, std::max(1, static_cast<int>(std::ceil(dt_s / cfg_.max_step_s))));
  const double h = dt_s / static_cast<double>(substeps);

  Vec6 x;
  x.head<3>() = position_;
  x.tail<3>() = velocity_;
  Covariance p = p_;

  // --- Discrete process noise, CWNA (see the file header) ------------------
  // Built once: it depends only on the sub-step length, which is constant across
  // the loop. The off-diagonal blocks are the point — the position and velocity
  // error from one unmodelled acceleration are the same error seen twice.
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
  const double q_a = cfg_.accel_psd_m2_per_s3;
  Covariance q_d = Covariance::Zero();
  q_d.block<3, 3>(kPosition, kPosition) = (q_a * h * h * h / 3.0) * id;
  const Eigen::Matrix3d q_cross = (q_a * h * h / 2.0) * id;
  q_d.block<3, 3>(kPosition, kVelocity) = q_cross;
  q_d.block<3, 3>(kVelocity, kPosition) = q_cross;
  q_d.block<3, 3>(kVelocity, kVelocity) = (q_a * h) * id;

  for (int step = 0; step < substeps; ++step) {
    // Φ from the Jacobian at the sub-step start. The covariance does not need
    // the state's integration order: over h ≤ max_step_s the Jacobian's own
    // variation is O(nh) of a term that is already a small correction to I.
    const Eigen::Matrix<double, 6, 6> f = dynamicsJacobian(cfg_, x, pole);
    const Covariance phi = Covariance::Identity() + f * h + 0.5 * (f * f) * (h * h);

    // Classical RK4 on the state.
    const Vec6 k1 = stateDerivative(cfg_, x, pole);
    const Vec6 k2 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k1), pole);
    const Vec6 k3 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k2), pole);
    const Vec6 k4 = stateDerivative(cfg_, Vec6(x + h * k3), pole);
    x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    p = phi * p * phi.transpose() + q_d;
  }
  symmetrise(p);

  if (!x.allFinite() || !p.allFinite()) {
    dropSolution();  // a NaN can never recover on its own; drop back to cold start
    return OrbitOdRefusal::kFilterFault;
  }

  position_ = x.head<3>();
  velocity_ = x.tail<3>();
  p_ = p;
  last_epoch_ = epoch;
  age_s_ += dt_s;

  if (age_s_ > cfg_.max_coast_s) {
    // Past the horizon the coarse force model's error is systematic and the
    // covariance has stopped covering it, so the solution is declared invalid
    // and *dropped* — the next fix re-acquires whole rather than blending
    // against a prior that no longer means anything. See the file header for why
    // this differs from the attitude MEKF's retention.
    dropSolution();
    return OrbitOdRefusal::kCoastExpired;
  }
  return OrbitOdRefusal::kNone;
}

OrbitOdRefusal OrbitOd::seedFrom(const time::Tai& epoch, const Eigen::Vector3d& r_eci,
                                 const Eigen::Vector3d& v_eci, const Eigen::Matrix3d& r_pos_cov,
                                 double velocity_sigma) {
  Covariance cov = Covariance::Zero();
  cov.block<3, 3>(kPosition, kPosition) = r_pos_cov;
  cov.block<3, 3>(kVelocity, kVelocity) =
      (velocity_sigma * velocity_sigma) * Eigen::Matrix3d::Identity();
  // No position/velocity cross-covariance: the receiver's PVT solution reports
  // the two with correlated *errors* in reality (they come out of one internal
  // filter), but it does not publish that correlation, and claiming a value it
  // did not report would be this filter inventing information. Zero is the
  // honest statement, and it is conservative for the gate that follows.
  return initialize(epoch, math::Vec3<math::frames::ECI>(r_eci),
                    math::Vec3<math::frames::ECI>(v_eci), cov);
}

bool OrbitOd::applyUpdate(int offset, const Eigen::Vector3d& measured, const Eigen::Matrix3d& r_cov,
                          double gate, OrbitOdUpdate& out) {
  out = OrbitOdUpdate{};

  // H = [I 0] (position) or [0 I] (velocity), so HPHᵀ is the corresponding 3×3
  // diagonal block and PHᵀ the corresponding 6×3 column block — written that way
  // rather than as a matrix product because a sparse H multiplied out is the
  // same arithmetic with more places to get an index wrong.
  const Eigen::Vector3d predicted = (offset == kPosition) ? position_ : velocity_;
  const Eigen::Vector3d y = measured - predicted;
  const Eigen::Matrix3d s = p_.block<3, 3>(offset, offset) + r_cov;
  const Eigen::Matrix3d s_inv = s.inverse();
  if (!s_inv.allFinite() || !y.allFinite()) {
    return false;
  }

  const double nis = y.dot(s_inv * y);
  if (!std::isfinite(nis)) {
    return false;
  }
  out.innovation = y;
  out.innovation_cov = s;
  out.nis = nis;

  // Divergence guard. An outlier folded in at full gain drags the trajectory off
  // and the filter never comes back — a spoofed GNSS fix is exactly that, and
  // §6.2 makes it deliberately *valid* so that catching it is this gate's job
  // and not a flag's. One rejection is normal (a χ² tail, a multipath fix); a
  // *stream* is what FDIR watches rejectedCount for.
  //
  // Written as an accept range rather than `nis > gate`: if P ever went
  // indefinite, S would follow and the NIS could come back **negative**, which a
  // one-sided "too large?" test waves straight through.
  if (!(nis >= 0.0 && nis <= gate)) {
    ++rejected_;
    return false;
  }

  Eigen::Matrix<double, kDim, 3> h_t = Eigen::Matrix<double, kDim, 3>::Zero();
  h_t.block<3, 3>(offset, 0) = Eigen::Matrix3d::Identity();
  const Eigen::Matrix<double, kDim, 3> k_gain = p_ * h_t * s_inv;
  const Vec6 dx = k_gain * y;
  if (!k_gain.allFinite() || !dx.allFinite()) {
    return false;
  }

  // Joseph form rather than (I−KH)P: it stays symmetric positive-definite under
  // round-off and under a gain that is not exactly optimal, which is the case
  // whenever R is inflated for a systematic budget.
  const Covariance ikh = Covariance::Identity() - k_gain * h_t.transpose();
  Covariance p_new = ikh * p_ * ikh.transpose() + k_gain * r_cov * k_gain.transpose();
  symmetrise(p_new);

  const Eigen::Vector3d new_position = position_ + dx.head<3>();
  const Eigen::Vector3d new_velocity = velocity_ + dx.tail<3>();
  if (!new_position.allFinite() || !new_velocity.allFinite() || !p_new.allFinite()) {
    dropSolution();
    out = OrbitOdUpdate{};
    return false;
  }
  position_ = new_position;
  velocity_ = new_velocity;
  p_ = p_new;
  out.accepted = true;
  return true;
}

bool OrbitOd::ingest(const GnssFix& fix, const frames::EopValue& eop, OrbitOdResult& out) {
  out = OrbitOdResult{};
  if (!configured_) {
    out.refusal = OrbitOdRefusal::kUnconfigured;
    return false;
  }

  // --- Trust boundary: the fix is wire data (§9.1) --------------------------
  if (!fix.position_m.isFinite() || !fix.velocity_m_s.isFinite()) {
    out.refusal = OrbitOdRefusal::kFixNotFinite;
    return false;
  }
  if (!(fix.position_sigma_h_m > 0.0) || !(fix.position_sigma_v_m > 0.0) ||
      !std::isfinite(fix.position_sigma_h_m) || !std::isfinite(fix.position_sigma_v_m)) {
    out.refusal = OrbitOdRefusal::kFixSigmaInvalid;
    return false;
  }
  if (fix.velocity_valid &&
      (!(fix.velocity_sigma_m_s > 0.0) || !std::isfinite(fix.velocity_sigma_m_s))) {
    out.refusal = OrbitOdRefusal::kFixSigmaInvalid;
    return false;
  }
  const double fix_radius = fix.position_m.eigen().norm();
  if (!(fix_radius >= cfg_.min_radius_m && fix_radius <= cfg_.max_radius_m)) {
    out.refusal = OrbitOdRefusal::kFixImplausible;
    return false;
  }

  // --- Ingest conversion: GPS → TAI, ECEF → ECI (REQ-CONV-001) -------------
  const time::Tai epoch = time::toTai(fix.time_tag);

  // Fix epochs must be strictly increasing, checked here rather than left to the
  // propagation guard: a re-presented fix at the epoch the filter already sits
  // at would skip propagation entirely and fold the same measurement in twice,
  // shrinking the covariance on information already used. This is the guard that
  // makes a *stuck* receiver clock safe.
  if (have_fix_ && !(epoch > last_fix_epoch_)) {
    out.refusal = OrbitOdRefusal::kNonMonotonicEpoch;
    return false;
  }

  math::Vec3<math::frames::ECI> r_eci;
  math::Vec3<math::frames::ECI> v_eci;
  if (!frames::eciStateFromEcef(epoch, eop, fix.position_m, fix.velocity_m_s, r_eci, v_eci)) {
    out.refusal = OrbitOdRefusal::kFrameConversion;
    return false;
  }
  math::Quat<math::frames::ECI, math::frames::ECEF> q_eci_ecef;
  if (!frames::eciFromEcef(epoch, eop, q_eci_ecef)) {
    out.refusal = OrbitOdRefusal::kFrameConversion;
    return false;
  }

  // The receiver's own anisotropic position covariance, rotated into ECI: the
  // horizontal/vertical split is realised in the local geodetic frame at the fix
  // (§6.2), so flattening it to a scalar would throw away accuracy the receiver
  // reported and mis-weight the vertical direction by the VDOP/HDOP ratio.
  const Eigen::Matrix3d enu = enuBasis(fix.position_m.eigen());
  Eigen::Vector3d sig2(fix.position_sigma_h_m * fix.position_sigma_h_m,
                       fix.position_sigma_h_m * fix.position_sigma_h_m,
                       fix.position_sigma_v_m * fix.position_sigma_v_m);
  const Eigen::Matrix3d r_pos_ecef = enu * sig2.asDiagonal() * enu.transpose();
  const Eigen::Matrix3d a_eci_ecef = q_eci_ecef.core().toRotationMatrix();
  Eigen::Matrix3d r_pos_eci = a_eci_ecef * r_pos_ecef * a_eci_ecef.transpose();
  r_pos_eci = 0.5 * (r_pos_eci + r_pos_eci.transpose().eval());
  if (!r_pos_eci.allFinite()) {
    out.refusal = OrbitOdRefusal::kFrameConversion;
    return false;
  }
  // The velocity error is per-axis white in ECEF, and a rotation leaves σ²I
  // alone — so it needs no basis change. (The ω⊕ × r transport term does add
  // the position error's rotation into the ECI velocity, ~1.5e-4 m/s on a 2 m
  // fix, three orders under a receiver's 0.03 m/s velocity σ; it is neglected,
  // and it is neglected in the direction that makes R slightly optimistic.)

  auto seed = [&](void) -> bool {
    if (!fix.velocity_valid) {
      out.refusal = OrbitOdRefusal::kNoVelocityForSeed;
      return false;
    }
    const OrbitOdRefusal r =
        seedFrom(epoch, r_eci.eigen(), v_eci.eigen(), r_pos_eci, fix.velocity_sigma_m_s);
    out.refusal = r;
    if (r != OrbitOdRefusal::kNone) {
      return false;
    }
    out.seeded = true;
    have_fix_ = true;
    last_fix_epoch_ = epoch;
    return true;
  };

  // --- Cold start / re-acquisition -----------------------------------------
  if (!initialised_) {
    return seed();
  }

  // --- Propagate to the fix epoch ------------------------------------------
  if (epoch > last_epoch_) {
    const OrbitOdRefusal r = propagate(epoch, eop);
    if (r == OrbitOdRefusal::kCoastExpired || r == OrbitOdRefusal::kFilterFault) {
      // Both dropped the solution. Re-acquire whole from this fix rather than
      // reporting a refusal and leaving the vehicle with no position: the fix in
      // hand determines the state outright and better than the prior that was
      // just discarded.
      return seed();
    }
    if (r != OrbitOdRefusal::kNone) {
      out.refusal = r;
      return false;
    }
  } else if (epoch < last_epoch_) {
    // The filter has been propagated past this fix by the GNC cycle. Folding it
    // in against a later state would apply the measurement at the wrong epoch,
    // which is a position error of v·Δt — metres per millisecond in LEO.
    out.refusal = OrbitOdRefusal::kNonMonotonicEpoch;
    return false;
  }

  have_fix_ = true;
  last_fix_epoch_ = epoch;

  // --- Sequential 3-row updates --------------------------------------------
  const bool pos_ok =
      applyUpdate(kPosition, r_eci.eigen(), r_pos_eci, cfg_.position_nis_gate, out.position);
  if (!pos_ok) {
    // A failed position update either tripped the gate or dropped the filter on
    // a non-finite result; either way the velocity is not folded in against a
    // state the position half just refused.
    out.refusal =
        initialised_ ? OrbitOdRefusal::kMeasurementRejected : OrbitOdRefusal::kFilterFault;
    return false;
  }
  age_s_ = 0.0;

  if (fix.velocity_valid) {
    const Eigen::Matrix3d r_vel =
        (fix.velocity_sigma_m_s * fix.velocity_sigma_m_s) * Eigen::Matrix3d::Identity();
    if (!applyUpdate(kVelocity, v_eci.eigen(), r_vel, cfg_.velocity_nis_gate, out.velocity)) {
      // The position was accepted, so the solution stands and stays valid; the
      // velocity half is reported rejected. A receiver whose velocity degrades
      // while its position is fine is a real mode, and it must not cost the
      // position fix.
      out.refusal =
          initialised_ ? OrbitOdRefusal::kMeasurementRejected : OrbitOdRefusal::kFilterFault;
      return initialised_;
    }
  }
  return true;
}

bool OrbitOd::nees(const math::Vec3<math::frames::ECI>& position_true,
                   const math::Vec3<math::frames::ECI>& velocity_true, double& out) const {
  if (!initialised_ || !position_true.isFinite() || !velocity_true.isFinite()) {
    return false;
  }
  Vec6 e;
  e.head<3>() = position_true.eigen() - position_;
  e.tail<3>() = velocity_true.eigen() - velocity_;

  const Covariance p_inv = p_.inverse();
  if (!p_inv.allFinite() || !e.allFinite()) {
    return false;
  }
  const double value = e.dot(p_inv * e);
  if (!std::isfinite(value)) {
    return false;
  }
  out = value;
  return true;
}

void writeToEstimatedState(const OrbitOd& filter, const time::Tai& epoch,
                           state::EstimatedState& state) {
  const bool valid = filter.solutionValid();
  state.epoch = epoch;
  state.position = filter.position();
  state.velocity = filter.velocity();
  state.valid.position = valid;
  state.valid.velocity = valid;
  if (valid) {
    const OrbitOd::Covariance& p = filter.covariance();
    state.covariance.block<3, 3>(state::ErrorState::kPosition, state::ErrorState::kPosition) =
        p.block<3, 3>(OrbitOd::kPosition, OrbitOd::kPosition);
    state.covariance.block<3, 3>(state::ErrorState::kVelocity, state::ErrorState::kVelocity) =
        p.block<3, 3>(OrbitOd::kVelocity, OrbitOd::kVelocity);
    state.covariance.block<3, 3>(state::ErrorState::kPosition, state::ErrorState::kVelocity) =
        p.block<3, 3>(OrbitOd::kPosition, OrbitOd::kVelocity);
    state.covariance.block<3, 3>(state::ErrorState::kVelocity, state::ErrorState::kPosition) =
        p.block<3, 3>(OrbitOd::kVelocity, OrbitOd::kPosition);
    state.valid.covariance = true;
  }
}

}  // namespace polaris::gnc

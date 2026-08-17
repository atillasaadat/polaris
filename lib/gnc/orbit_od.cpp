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
  // --- Harmonic field ------------------------------------------------------
  if (geopotential_degree < 0 || geopotential_degree > kGeopotentialMaxDegree) {
    return false;
  }
  if (geopotential_order < 0 || geopotential_order > geopotential_degree) {
    return false;
  }
  if (geopotential_degree > 0) {
    // The harmonic field already contains the degree-2 zonal, so a config
    // setting both is describing the same physics twice and getting one of them
    // silently ignored. Refuse rather than pick.
    if (zonal_j2 > 0.0) {
      return false;
    }
    // The field carries its own GM and reference radius — the values the
    // coefficients were *solved* with — and uses them regardless of what the
    // config says. But `mu_m3_per_s2` and `reference_radius_m` still drive the
    // drag altitude and every μ-dependent consumer, so a config that disagrees
    // with the table is describing two different Earths. Requiring agreement
    // here turns that into a construction-time failure rather than a
    // sub-millimetre inconsistency nobody ever looks for.
    //
    // The band is loose enough to accept WGS84 against EGM2008's own constants
    // (they differ by 7.5e-10 in GM, 0.3 m in radius) — which is a legitimate
    // pairing — and tight enough to catch the mistakes that matter: a
    // kilometre-vs-metre radius, or another body's μ.
    constexpr double kConstantMatchTolerance = 1.0e-6;
    const double mu_mismatch = std::abs(mu_m3_per_s2 - egm2008::kGm) / egm2008::kGm;
    const double re_mismatch =
        std::abs(reference_radius_m - egm2008::kReferenceRadius) / egm2008::kReferenceRadius;
    if (!(mu_mismatch < kConstantMatchTolerance) || !(re_mismatch < kConstantMatchTolerance)) {
      return false;
    }
  }
  // A fix arriving behind the filter's own epoch is a *latency*, not a fault, up
  // to this bound (see the header). Negative is meaningless; zero restores the
  // strictly-forward behaviour.
  if (!std::isfinite(max_fix_latency_s) || max_fix_latency_s < 0.0) {
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
  if (!std::isfinite(max_degraded_coast_s) || max_degraded_coast_s < max_coast_s) {
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
                                                  const EarthOrientation& earth) {
  const Eigen::Vector3d r = position.eigen();
  const Eigen::Vector3d v = velocity.eigen();
  const Eigen::Vector3d pole = earth.pole();
  if (!r.allFinite() || !v.allFinite() || !earth.eci_from_ecef.allFinite()) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  const double r_mag = r.norm();
  if (!(r_mag > kMinRadiusM)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }

  const double r2 = r_mag * r_mag;
  const double r3 = r2 * r_mag;
  Eigen::Vector3d a;

  if (cfg.geopotential_degree > 0) {
    // --- Truncated EGM2008 field, evaluated in ECEF ------------------------
    // The harmonic sum carries its own point-mass term (C_00 = 1), so this
    // *replaces* the two-body and J2 branches below rather than perturbing
    // them. Position rotates into ECEF and the acceleration rotates back: a
    // tesseral field is not axisymmetric, so unlike the zonal branch there is no
    // shortcut through the pole alone (see geopotential.hpp).
    const Eigen::Vector3d r_ecef = earth.eci_from_ecef.transpose() * r;
    a = earth.eci_from_ecef * geopotentialAcceleration(r_ecef, cfg.geopotential_degree,
                                                       cfg.geopotential_order, egm2008::kGm,
                                                       egm2008::kReferenceRadius);
  } else {
    // --- Point mass --------------------------------------------------------
    a = -(cfg.mu_m3_per_s2 / r3) * r;

    // --- J2, about the true pole (see the file header) ---------------------
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

bool earthOrientationAt(const time::Tai& t, const frames::EopValue& eop, EarthOrientation& out) {
  math::Quat<math::frames::ECI, math::frames::ECEF> q_eci_ecef;
  if (!frames::eciFromEcef(t, eop, q_eci_ecef)) {
    return false;
  }
  const Eigen::Matrix3d a = q_eci_ecef.core().toRotationMatrix();
  if (!a.allFinite()) {
    return false;
  }
  out.eci_from_ecef = a;
  return true;
}

namespace {

/// Derivative of the 6-state `[r; v]` under the onboard force model, plus a
/// known non-gravitational acceleration @p a_ng (zero when none).
Vec6 stateDerivative(const OrbitOdConfig& cfg, const Vec6& x, const EarthOrientation& earth,
                     const Eigen::Vector3d& a_ng = Eigen::Vector3d::Zero()) {
  Vec6 dx;
  dx.head<3>() = x.tail<3>();
  dx.tail<3>() = onboardAcceleration(cfg, math::Vec3<math::frames::ECI>(x.head<3>()),
                                     math::Vec3<math::frames::ECI>(x.tail<3>()), earth)
                     .eigen() +
                 a_ng;
  return dx;
}

/// Dynamics Jacobian `F = ∂ẋ/∂x` at @p x. The upper blocks are exact; the lower
/// two are central differences of the acceleration (see the file header for why
/// the analytic form was not written by hand).
Eigen::Matrix<double, 6, 6> dynamicsJacobian(const OrbitOdConfig& cfg, const Vec6& x,
                                             const EarthOrientation& earth) {
  Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Zero();
  f.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

  const double r_step = std::max(kJacMinPosStepM, kJacRelStep * x.head<3>().norm());
  const double v_step = std::max(kJacMinVelStepMps, kJacRelStep * x.tail<3>().norm());

  for (int i = 0; i < 3; ++i) {
    Vec6 plus = x;
    Vec6 minus = x;
    plus(i) += r_step;
    minus(i) -= r_step;
    f.block<3, 1>(3, i) = (stateDerivative(cfg, plus, earth).tail<3>() -
                           stateDerivative(cfg, minus, earth).tail<3>()) /
                          (2.0 * r_step);

    plus = x;
    minus = x;
    plus(3 + i) += v_step;
    minus(3 + i) -= v_step;
    f.block<3, 1>(3, 3 + i) = (stateDerivative(cfg, plus, earth).tail<3>() -
                               stateDerivative(cfg, minus, earth).tail<3>()) /
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
  forced_ = 0;
  have_fix_ = false;
  last_fix_epoch_ = time::Tai{};
}

OrbitOdRefusal OrbitOd::retune(const OrbitOdConfig& config) {
  // TB 20-03 item (g) / TP §9.3: the tuning changes, the navigation data does
  // not. Only the configuration is swapped; a bad upload leaves the running
  // filter — configuration included — exactly as it was.
  if (!config.isValid()) {
    return OrbitOdRefusal::kUnconfigured;
  }
  cfg_ = config;
  configured_ = true;
  return OrbitOdRefusal::kNone;
}

OrbitOdRefusal OrbitOd::reinitializeCovariance(double sigma_pos_m, double sigma_vel_m_s) {
  // TB 20-03 item (f) / TP §9.2: the covariance is re-opened, the state is kept.
  if (!configured_) {
    return OrbitOdRefusal::kUnconfigured;
  }
  if (!initialised_) {
    return OrbitOdRefusal::kUninitialised;
  }
  if (!std::isfinite(sigma_pos_m) || !std::isfinite(sigma_vel_m_s) || !(sigma_pos_m > 0.0) ||
      !(sigma_vel_m_s > 0.0)) {
    return OrbitOdRefusal::kFixSigmaInvalid;
  }
  p_.setZero();
  p_.block<3, 3>(kPosition, kPosition) = sigma_pos_m * sigma_pos_m * Eigen::Matrix3d::Identity();
  p_.block<3, 3>(kVelocity, kVelocity) =
      sigma_vel_m_s * sigma_vel_m_s * Eigen::Matrix3d::Identity();
  return OrbitOdRefusal::kNone;
}

bool OrbitOd::covarianceHealthy() const {
  // TP Ch. 7: a UDU filter reads definiteness off D for free; a full-P filter
  // has to ask. LDLᵀ is the pivoted, stack-only, fixed-size way to ask a 6×6.
  if (!initialised_) {
    return true;
  }
  if (!p_.allFinite()) {
    return false;
  }
  const Eigen::LDLT<Covariance> ldlt(p_);
  return ldlt.info() == Eigen::Success && ldlt.isPositive();
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

OrbitOdRefusal OrbitOd::seed(const time::Tai& epoch, const math::Vec3<math::frames::ECI>& position,
                             const math::Vec3<math::frames::ECI>& velocity, double sigma_pos_m,
                             double sigma_vel_m_s) {
  if (!std::isfinite(sigma_pos_m) || !std::isfinite(sigma_vel_m_s) || !(sigma_pos_m > 0.0) ||
      !(sigma_vel_m_s > 0.0)) {
    return OrbitOdRefusal::kFixSigmaInvalid;
  }
  Covariance cov = Covariance::Zero();
  cov.block<3, 3>(kPosition, kPosition) = sigma_pos_m * sigma_pos_m * Eigen::Matrix3d::Identity();
  cov.block<3, 3>(kVelocity, kVelocity) =
      sigma_vel_m_s * sigma_vel_m_s * Eigen::Matrix3d::Identity();
  return initialize(epoch, position, velocity, cov);
}

OrbitOdRefusal OrbitOd::propagate(const time::Tai& epoch, const frames::EopValue& eop,
                                  const NonGravAccelInput* accel) {
  if (!configured_) {
    return OrbitOdRefusal::kUnconfigured;
  }
  if (!initialised_) {
    return OrbitOdRefusal::kUninitialised;
  }
  // A known non-gravitational acceleration over the step (file header). Not
  // finite or a negative sigma is a caller bug, refused like a bad fix rather
  // than propagated into the state.
  Eigen::Vector3d a_ng = Eigen::Vector3d::Zero();
  double q_ng = 0.0;
  if (accel != nullptr) {
    if (!accel->accel_m_s2.isFinite() || !std::isfinite(accel->sigma_m_s2) ||
        accel->sigma_m_s2 < 0.0) {
      return OrbitOdRefusal::kFixNotFinite;
    }
    a_ng = accel->accel_m_s2.eigen();
    q_ng = accel->sigma_m_s2 * accel->sigma_m_s2;
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
  // The thrust-knowledge term joins it as a second white acceleration over the
  // step: σ_a² [m²/s⁴] acting for h seconds is a PSD of σ_a²·h [m²/s³] over the
  // sub-step, which puts it in the same discretisation as q_a.
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
  const double q_a = cfg_.accel_psd_m2_per_s3 + q_ng * h;
  Covariance q_d = Covariance::Zero();
  q_d.block<3, 3>(kPosition, kPosition) = (q_a * h * h * h / 3.0) * id;
  const Eigen::Matrix3d q_cross = (q_a * h * h / 2.0) * id;
  q_d.block<3, 3>(kPosition, kVelocity) = q_cross;
  q_d.block<3, 3>(kVelocity, kPosition) = q_cross;
  q_d.block<3, 3>(kVelocity, kVelocity) = (q_a * h) * id;

  for (int step = 0; step < substeps; ++step) {
    // The Earth orientation is resolved **per sub-step**, not once per call. A
    // zonal field only needed the pole, which moves ~50 arcsec/yr and could be
    // held across any propagation; a tesseral field is fixed to the rotating
    // Earth, and holding one rotation across a max_dt_s = 60 s step would smear
    // it by 0.25° of longitude. Held *within* the sub-step across the RK4 stages,
    // though: over h ≤ 1 s the Earth turns 6e-5 rad, which mis-orients a ~1e-5
    // m/s² tesseral term by ~6e-10 m/s² — orders below the truncation itself.
    const time::Tai sub_epoch =
        last_epoch_ + time::Duration::fromSecondsF(static_cast<double>(step) * h);
    EarthOrientation earth;
    if (!earthOrientationAt(sub_epoch, eop, earth)) {
      return OrbitOdRefusal::kFrameConversion;
    }

    // Φ from the Jacobian at the sub-step start. The covariance does not need
    // the state's integration order: over h ≤ max_step_s the Jacobian's own
    // variation is O(nh) of a term that is already a small correction to I.
    const Eigen::Matrix<double, 6, 6> f = dynamicsJacobian(cfg_, x, earth);
    const Covariance phi = Covariance::Identity() + f * h + 0.5 * (f * f) * (h * h);

    // Classical RK4 on the state.
    const Vec6 k1 = stateDerivative(cfg_, x, earth, a_ng);
    const Vec6 k2 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k1), earth, a_ng);
    const Vec6 k3 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k2), earth, a_ng);
    const Vec6 k4 = stateDerivative(cfg_, Vec6(x + h * k3), earth, a_ng);
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

  if (age_s_ > cfg_.max_degraded_coast_s) {
    // Past the degraded horizon the coasted prediction is dropped: the next fix
    // re-acquires whole rather than blending against a prior whose covariance
    // has stopped covering the model's systematic error (file header). Between
    // the two horizons the solution stands and quality() says degraded.
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
                          double gate, bool force, OrbitOdUpdate& out) {
  out = OrbitOdUpdate{};

  // H = [I 0] (position) or [0 I] (velocity), so HPHᵀ is the corresponding 3×3
  // diagonal block and PHᵀ the corresponding 6×3 column block — written that way
  // rather than as a matrix product because a sparse H multiplied out is the
  // same arithmetic with more places to get an index wrong.
  const Eigen::Vector3d predicted = (offset == kPosition) ? position_ : velocity_;
  const Eigen::Vector3d y = measured - predicted;
  const Eigen::Matrix3d s = p_.block<3, 3>(offset, offset) + r_cov;
  // Published before the inversion is checked, so a numeric fault still
  // telemeters the innovation it faulted on rather than a zero that reads as a
  // perfect fit.
  out.innovation = y;
  out.innovation_cov = s;
  const Eigen::Matrix3d s_inv = s.inverse();
  if (!s_inv.allFinite() || !y.allFinite()) {
    out.numeric_fault = true;
    return false;
  }

  const double nis = y.dot(s_inv * y);
  if (!std::isfinite(nis)) {
    out.numeric_fault = true;
    return false;
  }
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
  //
  // Under `force` (TP §9.1's "force" flag) the gate is overridden — but only the
  // *gate*: a negative NIS is the covariance gone indefinite, and no operator
  // flag makes an update against that meaningful, so it stays a numeric fault.
  if (!(nis >= 0.0)) {
    ++rejected_;
    return false;
  }
  if (nis > gate) {
    if (!force) {
      ++rejected_;
      return false;
    }
    ++forced_;
    out.forced = true;
  }

  Eigen::Matrix<double, kDim, 3> h_t = Eigen::Matrix<double, kDim, 3>::Zero();
  h_t.block<3, 3>(offset, 0) = Eigen::Matrix3d::Identity();
  const Eigen::Matrix<double, kDim, 3> k_gain = p_ * h_t * s_inv;
  const Vec6 dx = k_gain * y;
  if (!k_gain.allFinite() || !dx.allFinite()) {
    out.numeric_fault = true;
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
    out.numeric_fault = true;
    return false;
  }
  position_ = new_position;
  velocity_ = new_velocity;
  p_ = p_new;
  out.accepted = true;
  return true;
}

bool OrbitOd::ingest(const GnssFix& fix, const frames::EopValue& eop, OrbitOdResult& out,
                     const GnssMeasurementPolicy& policy) {
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

  // TP §9.1 editing policy. An inhibited position is the whole fix withheld: it
  // is the measurement the solution stands on. A seed needs both halves, so
  // either inhibited refuses the seed too — an operator who has inhibited
  // velocity on a cold filter starts it with OD_SEED_STATE instead.
  const bool position_inhibited = policy.position == MeasurementMode::kInhibit;
  const bool velocity_inhibited = policy.velocity == MeasurementMode::kInhibit;
  if (position_inhibited) {
    out.refusal = OrbitOdRefusal::kMeasurementInhibited;
    return false;
  }

  auto seed = [&](void) -> bool {
    if (velocity_inhibited) {
      out.refusal = OrbitOdRefusal::kMeasurementInhibited;
      return false;
    }
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

  // --- Reconcile the fix epoch with the filter's ---------------------------
  // Forward, the filter moves to the fix. Backward — a latent fix — the fix
  // moves to the filter. Either way the two are at one epoch before the update.
  double latency_s = 0.0;
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
    // --- Latent fix: forward-propagate the measurement, not the filter -------
    // The GNC cycle has propagated past this fix, because the receiver's
    // solution is valid at its measurement epoch but reaches the FSW a fix
    // latency later. Folding it in against the later state unchanged would
    // apply the measurement at the wrong epoch — a position error of v·Δt,
    // ~7.6 m per millisecond in LEO, which is the single largest error this
    // filter can carry and is why it is corrected rather than tolerated.
    //
    // The correction is on the *measurement*: advance the reported PVT to the
    // filter's epoch on the receiver's own reported velocity plus the onboard
    // acceleration, rather than retrodicting the filter (Bar-Shalom §5.6's
    // out-of-sequence-measurement problem, in the benign case where the
    // measurement determines the whole state). Two properties make the simple
    // form exact enough: the fix carries velocity, so the linear term is
    // measured rather than modelled; and the residual is O(τ³·jerk), which at
    // τ ≤ 0.2 s is micrometres. This is the solution-domain analogue of the
    // signal-transit-time correction Kim et al. (2025) apply to
    // navigation-solution measurements [kim2025].
    latency_s = (last_epoch_ - epoch).seconds();
    if (!(latency_s <= cfg_.max_fix_latency_s)) {
      out.refusal = OrbitOdRefusal::kNonMonotonicEpoch;
      return false;
    }
    if (!fix.velocity_valid) {
      // Without a velocity there is nothing to propagate the position on. The
      // filter's own velocity is not a substitute: using it would fold the
      // filter's current error into a measurement that is supposed to be
      // independent of it, which is exactly how a consistent filter is made
      // overconfident. Named as the missing-velocity refusal, not as a clock
      // fault: an FDIR rule keyed on kNonMonotonicEpoch means "the receiver's
      // time tags went wrong", and an honest, on-time fix that merely lacks a
      // velocity field is a different failure with a different response.
      out.refusal = OrbitOdRefusal::kNoVelocityForSeed;
      return false;
    }

    EarthOrientation earth;
    if (!earthOrientationAt(epoch, eop, earth)) {
      out.refusal = OrbitOdRefusal::kFrameConversion;
      return false;
    }
    const math::Vec3<math::frames::ECI> accel = onboardAcceleration(cfg_, r_eci, v_eci, earth);
    const Eigen::Vector3d r_advanced =
        r_eci.eigen() + v_eci.eigen() * latency_s + 0.5 * accel.eigen() * latency_s * latency_s;
    const Eigen::Vector3d v_advanced = v_eci.eigen() + accel.eigen() * latency_s;
    if (!r_advanced.allFinite() || !v_advanced.allFinite()) {
      out.refusal = OrbitOdRefusal::kFrameConversion;
      return false;
    }
    r_eci = math::Vec3<math::frames::ECI>(r_advanced);
    v_eci = math::Vec3<math::frames::ECI>(v_advanced);

    // The advance is not free of uncertainty: the reported velocity's own error
    // integrates into position over the latency. Inflating R by (σ_v·τ)² keeps
    // the covariance honest about a measurement that is now partly propagated.
    // It is a small term — 3 mm at σ_v = 0.03 m/s and τ = 0.1 s against a ~1 m
    // fix — and it is added rather than neglected because the alternative is a
    // filter that gets quietly more confident the later its data arrives.
    const double sigma_advance = fix.velocity_sigma_m_s * latency_s;
    r_pos_eci += (sigma_advance * sigma_advance) * Eigen::Matrix3d::Identity();
  }

  have_fix_ = true;
  last_fix_epoch_ = epoch;
  out.fix_latency_s = latency_s;

  // --- Sequential 3-row updates --------------------------------------------
  const bool pos_ok = applyUpdate(kPosition, r_eci.eigen(), r_pos_eci, cfg_.position_nis_gate,
                                  policy.position == MeasurementMode::kForce, out.position);
  if (!pos_ok) {
    // A failed position update either tripped the gate or faulted on a
    // non-finite intermediate; either way the velocity is not folded in against
    // a state the position half just refused. The two are named apart: a gate
    // rejection increments rejectedCount and is the FDIR stream signal, while a
    // numeric fault is filter health and must not hide inside that stream.
    out.refusal = (out.position.numeric_fault || !initialised_)
                      ? OrbitOdRefusal::kFilterFault
                      : OrbitOdRefusal::kMeasurementRejected;
    return false;
  }
  age_s_ = 0.0;

  if (fix.velocity_valid && !velocity_inhibited) {
    const Eigen::Matrix3d r_vel =
        (fix.velocity_sigma_m_s * fix.velocity_sigma_m_s) * Eigen::Matrix3d::Identity();
    if (!applyUpdate(kVelocity, v_eci.eigen(), r_vel, cfg_.velocity_nis_gate,
                     policy.velocity == MeasurementMode::kForce, out.velocity)) {
      // The position was accepted, so the solution stands and stays valid; the
      // velocity half is reported rejected. A receiver whose velocity degrades
      // while its position is fine is a real mode, and it must not cost the
      // position fix.
      out.refusal = (out.velocity.numeric_fault || !initialised_)
                        ? OrbitOdRefusal::kFilterFault
                        : OrbitOdRefusal::kMeasurementRejected;
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

  // Solve rather than invert: LDLT on an SPD 6x6 is cheaper and better
  // conditioned than an explicit inverse for the same quadratic form, and it is
  // the SPD discipline the rest of the file already uses (llt() gates, Joseph
  // form).
  const Eigen::LDLT<Covariance> ldlt = p_.ldlt();
  if (ldlt.info() != Eigen::Success || !e.allFinite()) {
    return false;
  }
  const Vec6 solved = ldlt.solve(e);
  if (!solved.allFinite()) {
    return false;
  }
  const double value = e.dot(solved);
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

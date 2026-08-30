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
template <typename M>
void symmetrise(M& p) {
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
      std::isfinite(accel_psd_m2_per_s3) && accel_psd_rtn_m2_per_s3.allFinite() &&
      std::isfinite(dmc_tau_s) && dmc_psd_rtn_m2_per_s5.allFinite() &&
      std::isfinite(gnss_corr_sigma_h_m) && std::isfinite(gnss_corr_sigma_v_m) &&
      std::isfinite(drag_scale_tau_s) && std::isfinite(drag_scale_psd_per_s) &&
      std::isfinite(drag_scale_seed_sigma) && std::isfinite(drag_scale_max_deviation) &&
      std::isfinite(position_nis_gate) && std::isfinite(velocity_nis_gate) &&
      std::isfinite(max_coast_s) && std::isfinite(max_dt_s) && std::isfinite(max_step_s) &&
      std::isfinite(min_radius_m) && std::isfinite(max_radius_m);
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
  // Some white acceleration noise must drive the filter — isotropic, RTN, or
  // both (TP §2.2.3.1); a filter with none stops opening its covariance through
  // a coast. The DMC states are optional (τ = 0 off), never negative.
  const bool snc_present = accel_psd_m2_per_s3 > 0.0 || accel_psd_rtn_m2_per_s3.maxCoeff() > 0.0;
  if (accel_psd_m2_per_s3 < 0.0 || accel_psd_rtn_m2_per_s3.minCoeff() < 0.0 || !snc_present ||
      dmc_tau_s < 0.0 || dmc_psd_rtn_m2_per_s5.minCoeff() < 0.0) {
    return false;
  }
  // The DMC kernel is evaluated per sub-step by a fixed-node quadrature that is
  // exact for h ≪ τ; a correlation time under ten sub-steps is refused rather
  // than integrated coarsely (and would be a strange model anyway).
  if (dmc_tau_s > 0.0 && dmc_tau_s < 10.0 * max_step_s) {
    return false;
  }
  // The correlated-GNSS R inflation (Push 77). Zero is off; negative is not a
  // smaller error, it is a covariance being *shrunk* by configuration.
  if (gnss_corr_sigma_h_m < 0.0 || gnss_corr_sigma_v_m < 0.0) {
    return false;
  }
  // The drag scale factor (Push 76). Zero PSD is off; positive demands the rest
  // of its tuning be present and sane, so a half-configured scale factor is a
  // construction-time refusal rather than a state with a nonsense prior. Its
  // kernel is the DMC's, hence the same ten-sub-step quadrature limit.
  if (drag_scale_psd_per_s < 0.0 || drag_scale_tau_s < 0.0) {
    return false;
  }
  if (drag_scale_psd_per_s > 0.0) {
    if (!(drag_scale_tau_s >= 10.0 * max_step_s) || !(drag_scale_seed_sigma > 0.0) ||
        !(drag_scale_max_deviation > 0.0)) {
      return false;
    }
    // A band wider than 1 admits a *negative* scale — drag pushing the vehicle
    // along its own velocity — which is not a large error, it is a different
    // sign of physics, and no measurement should be able to talk the filter
    // into it.
    if (!(drag_scale_max_deviation < 1.0)) {
      return false;
    }
  }
  if (!(position_nis_gate > 0.0) || !(velocity_nis_gate > 0.0)) {
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

math::Vec3<math::frames::ECI> dragAcceleration(const OrbitOdConfig& cfg,
                                               const math::Vec3<math::frames::ECI>& position,
                                               const math::Vec3<math::frames::ECI>& velocity,
                                               const EarthOrientation& earth) {
  if (!(cfg.drag_ballistic_coeff_m2_per_kg > 0.0)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
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

  const double altitude = r_mag - cfg.reference_radius_m;
  const double x = (altitude - cfg.drag_ref_altitude_m) / cfg.drag_scale_height_m;
  // Guard the exponential rather than the altitude: a fix below the reference
  // altitude is physically a re-entry, and letting exp() overflow to inf would
  // put a NaN into the state where a large-but-finite density puts a large-
  // but-finite drag the finiteness guard can still reason about.
  const double density = (x > -700.0) ? cfg.drag_ref_density_kg_m3 * std::exp(-x) : 0.0;
  if (!std::isfinite(density) || !(density > 0.0)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  // The atmosphere co-rotates with the Earth, so the aerodynamically relevant
  // velocity is v - ω⊕ × r; at LEO the transport term is ~465 m/s against a
  // 7.7 km/s orbital velocity, i.e. a ~12% error in the drag magnitude and a
  // real cross-track component if it is dropped. ω⊕ is about the *true* pole,
  // the same axis the zonal field is referenced to.
  const Eigen::Vector3d omega = constants::wgs84::kEarthRate * pole;
  const Eigen::Vector3d v_rel = v - omega.cross(r);
  const Eigen::Vector3d a =
      (-0.5 * cfg.drag_ballistic_coeff_m2_per_kg * density * v_rel.norm()) * v_rel;
  if (!a.allFinite()) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  return math::Vec3<math::frames::ECI>(a);
}

math::Vec3<math::frames::ECI> onboardAcceleration(const OrbitOdConfig& cfg,
                                                  const math::Vec3<math::frames::ECI>& position,
                                                  const math::Vec3<math::frames::ECI>& velocity,
                                                  const EarthOrientation& earth,
                                                  double drag_scale) {
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

  // --- Exponential-density drag, scaled ------------------------------------
  // Linear in the scale factor by construction, which is what makes the drag
  // scale factor's Jacobian column exact (see dragAcceleration).
  a += drag_scale * dragAcceleration(cfg, position, velocity, earth).eigen();

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
/// known non-gravitational acceleration @p a_ng (zero when none) — which the
/// caller has already folded the DMC estimate into, in ECI.
Vec6 stateDerivative(const OrbitOdConfig& cfg, const Vec6& x, const EarthOrientation& earth,
                     const Eigen::Vector3d& a_ng = Eigen::Vector3d::Zero(),
                     double drag_scale = 1.0) {
  Vec6 dx;
  dx.head<3>() = x.tail<3>();
  dx.tail<3>() = onboardAcceleration(cfg, math::Vec3<math::frames::ECI>(x.head<3>()),
                                     math::Vec3<math::frames::ECI>(x.tail<3>()), earth, drag_scale)
                     .eigen() +
                 a_ng;
  return dx;
}

/// Dynamics Jacobian `F = ∂ẋ/∂x` at @p x. The upper blocks are exact; the lower
/// two are central differences of the acceleration (see the file header for why
/// the analytic form was not written by hand).
Eigen::Matrix<double, 6, 6> dynamicsJacobian(const OrbitOdConfig& cfg, const Vec6& x,
                                             const EarthOrientation& earth,
                                             double drag_scale = 1.0) {
  Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Zero();
  f.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

  const double r_step = std::max(kJacMinPosStepM, kJacRelStep * x.head<3>().norm());
  const double v_step = std::max(kJacMinVelStepMps, kJacRelStep * x.tail<3>().norm());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  for (int i = 0; i < 3; ++i) {
    Vec6 plus = x;
    Vec6 minus = x;
    plus(i) += r_step;
    minus(i) -= r_step;
    f.block<3, 1>(3, i) = (stateDerivative(cfg, plus, earth, zero, drag_scale).tail<3>() -
                           stateDerivative(cfg, minus, earth, zero, drag_scale).tail<3>()) /
                          (2.0 * r_step);

    plus = x;
    minus = x;
    plus(3 + i) += v_step;
    minus(3 + i) -= v_step;
    f.block<3, 1>(3, 3 + i) = (stateDerivative(cfg, plus, earth, zero, drag_scale).tail<3>() -
                               stateDerivative(cfg, minus, earth, zero, drag_scale).tail<3>()) /
                              (2.0 * v_step);
  }
  return f;
}

}  // namespace

OrbitOd::OrbitOd(const OrbitOdConfig& config) : cfg_(config), configured_(config.isValid()) {}

void OrbitOd::dropSolution() {
  position_.setZero();
  velocity_.setZero();
  dmc_.setZero();
  // Back to the nominal 1, not to zero: a dropped solution knows nothing about
  // the atmosphere, and "nothing" for a multiplier is unity. The scale the old
  // arc had learned belonged to that arc.
  drag_scale_ = 1.0;
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
  drag_scale_refused_ = 0;
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
  const bool was_on = dragScaleEnabled();
  cfg_ = config;
  configured_ = true;
  // One exception to "the tuning changes, the navigation data does not": a drag
  // scale factor that was **off** and is now **on** has no covariance to carry
  // forward — its block is identically zero, which is a state the Kalman gain
  // can never reach and the process noise would take months of random walk to
  // open. So the newly-live state takes its configured seed prior, exactly as a
  // cold start would give it. Everything else, the pos/vel solution included, is
  // untouched, and turning the state *off* needs nothing: it stops being read.
  if (initialised_ && !was_on && dragScaleEnabled()) {
    seedDragScaleBlock();
  }
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
  // The DMC states re-open to their stationary variance too, keeping their
  // estimate: the operator re-opened the covariance, not the state.
  const Eigen::Vector3d dmc_kept = dmc_;
  seedDmcBlock();
  dmc_ = dmc_kept;
  // Same for the drag scale factor, and for the same reason.
  const double drag_scale_kept = drag_scale_;
  seedDragScaleBlock();
  drag_scale_ = drag_scale_kept;
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
  const Eigen::LDLT<FullCovariance> ldlt(p_);
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
  p_.setZero();
  p_.topLeftCorner<kPosVelDim, kPosVelDim>() = cov;
  seedDmcBlock();
  seedDragScaleBlock();
  symmetrise(p_);
  last_epoch_ = epoch;
  age_s_ = 0.0;
  initialised_ = true;
  return OrbitOdRefusal::kNone;
}

void OrbitOd::seedDmcBlock() {
  // The DMC states start at zero with their steady-state variance qτ/2 (TP
  // §5.2.4): a fresh seed knows nothing about the unmodelled acceleration, and
  // that is what the FOGM's stationary distribution says. Off (τ = 0), the
  // block is zero and the states are inert.
  dmc_.setZero();
  p_.block<3, 3>(kDmc, kDmc).setZero();
  p_.block<3, kPosVelDim>(kDmc, 0).setZero();
  p_.block<kPosVelDim, 3>(0, kDmc).setZero();
  if (cfg_.dmc_tau_s > 0.0) {
    p_.block<3, 3>(kDmc, kDmc) = (0.5 * cfg_.dmc_tau_s * cfg_.dmc_psd_rtn_m2_per_s5).asDiagonal();
  }
}

bool OrbitOd::dragScaleEnabled() const {
  // Both halves are required. A PSD with drag switched off would carry a state
  // with no path to the measurement at all: its variance would grow every step
  // and never be reduced, which is a covariance that only ever gets worse.
  return cfg_.drag_scale_psd_per_s > 0.0 && cfg_.drag_ballistic_coeff_m2_per_kg > 0.0;
}

void OrbitOd::seedDragScaleBlock() {
  // A fresh seed's prior on the scale is the configured seed sigma about the
  // nominal 1 — deliberately *not* the FOGM stationary variance q·τ/2 the DMC
  // states use. The two answer different questions: the DMC's stationary spread
  // is the process's own, while "how wrong can a static exponential atmosphere
  // be against the real one" is a fact about the model, established on the
  // ground, and much larger than what the slow random walk would imply.
  drag_scale_ = 1.0;
  p_.row(kDragScale).setZero();
  p_.col(kDragScale).setZero();
  if (dragScaleEnabled()) {
    p_(kDragScale, kDragScale) = cfg_.drag_scale_seed_sigma * cfg_.drag_scale_seed_sigma;
  }
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
  FullCovariance p = p_;

  // --- Discrete process noise, CWNA (see the file header) ------------------
  // Built once: it depends only on the sub-step length, which is constant across
  // the loop. The off-diagonal blocks are the point — the position and velocity
  // error from one unmodelled acceleration are the same error seen twice.
  // The thrust-knowledge term joins it as a second white acceleration over the
  // step: σ_a² [m²/s⁴] acting for h seconds is a PSD of σ_a²·h [m²/s³] over the
  // sub-step, which puts it in the same discretisation as q_a.
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
  const double q_iso = cfg_.accel_psd_m2_per_s3 + q_ng * h;
  // The RTN part is rotated at each sub-step (the basis turns with the orbit);
  // only the isotropic part and the DMC kernel are step-invariant.
  const bool dmc = cfg_.dmc_tau_s > 0.0;
  const Eigen::Matrix3d psi = dmc ? dmcNoiseKernel(h, cfg_.dmc_tau_s) : Eigen::Matrix3d::Zero();
  const double phi_dmc = dmc ? std::exp(-h / cfg_.dmc_tau_s) : 1.0;
  Eigen::Vector3d dmc_state = dmc_;
  // The drag scale factor (Push 76) is the same first-order Gauss-Markov shape
  // as the DMC states, so it reuses their discrete-noise kernel — one scalar
  // channel instead of three, with the state-dependent map ∂a/∂s = a_drag in
  // place of the DMC's constant M_rtn. That map is held across the sub-step
  // exactly as the Jacobian and the RTN basis already are.
  const bool drag_scale_on = dragScaleEnabled();
  const Eigen::Matrix3d psi_s =
      drag_scale_on ? dmcNoiseKernel(h, cfg_.drag_scale_tau_s) : Eigen::Matrix3d::Zero();
  const double phi_s = drag_scale_on ? std::exp(-h / cfg_.drag_scale_tau_s) : 1.0;
  double drag_scale_state = drag_scale_;

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

    // The orbit-fixed basis at the sub-step start, held across it (TP §2.2.3.1's
    // "approximately constant over the interval"). Both the RTN noise and the
    // DMC acceleration are expressed in it.
    const Eigen::Matrix3d m_rtn = rtnBasis(x.head<3>(), x.tail<3>());
    const Eigen::Matrix3d q_rtn_eci =
        q_iso * id + m_rtn * cfg_.accel_psd_rtn_m2_per_s3.asDiagonal() * m_rtn.transpose();
    // --- Discrete process noise ------------------------------------------
    // SNC (TP Eq. 2.49): the CWNA blocks on Q̃ = q_iso I + M Q_rtn Mᵀ. The
    // off-diagonal blocks are the point — the position and velocity error from
    // one unmodelled acceleration are the same error seen twice. The thrust-
    // knowledge term is in q_iso as a second white acceleration over the step:
    // σ_a² [m²/s⁴] acting for h seconds is a PSD of σ_a²·h [m²/s³].
    FullCovariance q_d = FullCovariance::Zero();
    q_d.block<3, 3>(kPosition, kPosition) = (h * h * h / 3.0) * q_rtn_eci;
    q_d.block<3, 3>(kPosition, kVelocity) = (h * h / 2.0) * q_rtn_eci;
    q_d.block<3, 3>(kVelocity, kPosition) = (h * h / 2.0) * q_rtn_eci;
    q_d.block<3, 3>(kVelocity, kVelocity) = h * q_rtn_eci;
    if (dmc) {
      // DMC (TP Eq. 2.55): Q = kron(Ψ, Q̃_dmc), Q̃_dmc = M diag(q_dmc) Mᵀ — the
      // position/velocity blocks in ECI, the acceleration block in RTN.
      const Eigen::Matrix3d q_dmc_eci =
          m_rtn * cfg_.dmc_psd_rtn_m2_per_s5.asDiagonal() * m_rtn.transpose();
      const Eigen::Matrix3d q_dmc_rtn = cfg_.dmc_psd_rtn_m2_per_s5.asDiagonal();
      q_d.block<3, 3>(kPosition, kPosition) += psi(0, 0) * q_dmc_eci;
      q_d.block<3, 3>(kPosition, kVelocity) += psi(0, 1) * q_dmc_eci;
      q_d.block<3, 3>(kVelocity, kPosition) += psi(1, 0) * q_dmc_eci;
      q_d.block<3, 3>(kVelocity, kVelocity) += psi(1, 1) * q_dmc_eci;
      // Cross terms between an ECI position/velocity error and an RTN
      // acceleration error: (M diag(q))·ψ.
      const Eigen::Matrix3d mq = m_rtn * q_dmc_rtn;
      q_d.block<3, 3>(kPosition, kDmc) = psi(0, 2) * mq;
      q_d.block<3, 3>(kDmc, kPosition) = psi(2, 0) * mq.transpose();
      q_d.block<3, 3>(kVelocity, kDmc) = psi(1, 2) * mq;
      q_d.block<3, 3>(kDmc, kVelocity) = psi(2, 1) * mq.transpose();
      q_d.block<3, 3>(kDmc, kDmc) = psi(2, 2) * q_dmc_rtn;
    }
    // The drag scale factor's own noise, mapped into position/velocity through
    // b = ∂a/∂s = a_drag at unit scale. Same kron(Ψ, Q̃) structure as the DMC
    // block above with the 3x1 map b in place of M: Q̃ = q_s b bᵀ on the
    // position/velocity blocks, q_s b on the cross terms, q_s on the state
    // itself. With drag off b is zero and only the (harmless) state block
    // survives — which is why dragScaleEnabled() also requires drag.
    if (drag_scale_on) {
      const Eigen::Vector3d b = dragAcceleration(cfg_, math::Vec3<math::frames::ECI>(x.head<3>()),
                                                 math::Vec3<math::frames::ECI>(x.tail<3>()), earth)
                                    .eigen();
      const double q_s = cfg_.drag_scale_psd_per_s;
      const Eigen::Matrix3d bbt = q_s * b * b.transpose();
      q_d.block<3, 3>(kPosition, kPosition) += psi_s(0, 0) * bbt;
      q_d.block<3, 3>(kPosition, kVelocity) += psi_s(0, 1) * bbt;
      q_d.block<3, 3>(kVelocity, kPosition) += psi_s(1, 0) * bbt;
      q_d.block<3, 3>(kVelocity, kVelocity) += psi_s(1, 1) * bbt;
      const Eigen::Vector3d qb = q_s * b;
      q_d.block<3, 1>(kPosition, kDragScale) = psi_s(0, 2) * qb;
      q_d.block<1, 3>(kDragScale, kPosition) = psi_s(2, 0) * qb.transpose();
      q_d.block<3, 1>(kVelocity, kDragScale) = psi_s(1, 2) * qb;
      q_d.block<1, 3>(kDragScale, kVelocity) = psi_s(2, 1) * qb.transpose();
      q_d(kDragScale, kDragScale) = psi_s(2, 2) * q_s;
    }

    // Φ from the Jacobian at the sub-step start. The covariance does not need
    // the state's integration order: over h ≤ max_step_s the Jacobian's own
    // variation is O(nh) of a term that is already a small correction to I.
    // With the DMC states: ∂v̇/∂a = M and ∂ȧ/∂a = −I/τ (TP Eq. 2.54); ∂(Ma)/∂r
    // is dropped, as the TP does.
    Eigen::Matrix<double, kDim, kDim> f = Eigen::Matrix<double, kDim, kDim>::Zero();
    f.topLeftCorner<6, 6>() = dynamicsJacobian(cfg_, x, earth, drag_scale_state);
    if (dmc) {
      f.block<3, 3>(kVelocity, kDmc) = m_rtn;
      f.block<3, 3>(kDmc, kDmc) = -(1.0 / cfg_.dmc_tau_s) * id;
    }
    if (drag_scale_on) {
      // ∂v̇/∂s is the drag acceleration itself — exact, not a difference, because
      // the drag term is linear in s. ∂ṡ/∂s = −1/τ_s is the FOGM decay.
      f.block<3, 1>(kVelocity, kDragScale) =
          dragAcceleration(cfg_, math::Vec3<math::frames::ECI>(x.head<3>()),
                           math::Vec3<math::frames::ECI>(x.tail<3>()), earth)
              .eigen();
      f(kDragScale, kDragScale) = -(1.0 / cfg_.drag_scale_tau_s);
    }
    const FullCovariance phi = FullCovariance::Identity() + f * h + 0.5 * (f * f) * (h * h);

    // Classical RK4 on the state, the DMC acceleration (constant over the
    // sub-step, in ECI) riding with the known non-gravitational one.
    const Eigen::Vector3d a_known =
        a_ng + (dmc ? Eigen::Vector3d(m_rtn * dmc_state) : Eigen::Vector3d::Zero());
    const Vec6 k1 = stateDerivative(cfg_, x, earth, a_known, drag_scale_state);
    const Vec6 k2 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k1), earth, a_known, drag_scale_state);
    const Vec6 k3 = stateDerivative(cfg_, Vec6(x + 0.5 * h * k2), earth, a_known, drag_scale_state);
    const Vec6 k4 = stateDerivative(cfg_, Vec6(x + h * k3), earth, a_known, drag_scale_state);
    x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    dmc_state *= phi_dmc;  // the FOGM mean decays (TP §5.2.4)
    // The scale decays toward its nominal 1, not toward zero: it is a
    // multiplier on a term that exists, so "no information" means 1.
    drag_scale_state = 1.0 + (drag_scale_state - 1.0) * phi_s;

    p = phi * p * phi.transpose() + q_d;
  }
  symmetrise(p);

  if (!x.allFinite() || !p.allFinite() || !dmc_state.allFinite() ||
      !std::isfinite(drag_scale_state)) {
    dropSolution();  // a NaN can never recover on its own; drop back to cold start
    return OrbitOdRefusal::kFilterFault;
  }

  position_ = x.head<3>();
  velocity_ = x.tail<3>();
  dmc_ = dmc_state;
  drag_scale_ = drag_scale_state;
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
  const Eigen::Matrix<double, kDim, 1> dx = k_gain * y;
  if (!k_gain.allFinite() || !dx.allFinite()) {
    out.numeric_fault = true;
    return false;
  }

  // Joseph form rather than (I−KH)P: it stays symmetric positive-definite under
  // round-off and under a gain that is not exactly optimal, which is the case
  // whenever R is inflated for a systematic budget.
  const FullCovariance ikh = FullCovariance::Identity() - k_gain * h_t.transpose();
  FullCovariance p_new = ikh * p_ * ikh.transpose() + k_gain * r_cov * k_gain.transpose();
  symmetrise(p_new);

  const Eigen::Vector3d new_position = position_ + dx.segment<3>(kPosition);
  const Eigen::Vector3d new_velocity = velocity_ + dx.segment<3>(kVelocity);
  const Eigen::Vector3d new_dmc = dmc_ + dx.segment<3>(kDmc);
  const double new_drag_scale = drag_scale_ + dx(kDragScale);
  if (!new_position.allFinite() || !new_velocity.allFinite() || !new_dmc.allFinite() ||
      !std::isfinite(new_drag_scale) || !p_new.allFinite()) {
    dropSolution();
    out = OrbitOdUpdate{};
    out.numeric_fault = true;
    return false;
  }
  position_ = new_position;
  velocity_ = new_velocity;
  dmc_ = new_dmc;  // zero-variance when off, so a gain of exactly zero reaches it
  // The drag scale factor is **refused rather than clamped** outside its
  // configured band (§8.5, and the same rule the tier-3 dipole estimator flies):
  // an exponential atmosphere is not wrong by the factor an out-of-band estimate
  // claims, so such an estimate has been driven by something that is not drag,
  // and clamping it would fly a magnitude the policy chose rather than one the
  // data supported. The refusal holds the last accepted scale; the position and
  // velocity update — which is what the vehicle actually navigates on — stands.
  // With the state off the gain into it is exactly zero, so this never fires.
  if (!dragScaleEnabled() || std::abs(new_drag_scale - 1.0) <= cfg_.drag_scale_max_deviation) {
    drag_scale_ = new_drag_scale;
  } else {
    ++drag_scale_refused_;
  }
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
  // The reported sigmas describe the receiver's *white* error only — its formal
  // covariance is blind to an error common to every satellite it tracks — so the
  // configured correlated sigmas are added in quadrature here, in the same local
  // basis the reported ones are split in (Push 77; see the config field docs for
  // why this is inflation and not a bias state). Zero leaves R exactly as
  // reported, which is the pre-Push-77 filter.
  const double sig_h2 = fix.position_sigma_h_m * fix.position_sigma_h_m +
                        cfg_.gnss_corr_sigma_h_m * cfg_.gnss_corr_sigma_h_m;
  const double sig_v2 = fix.position_sigma_v_m * fix.position_sigma_v_m +
                        cfg_.gnss_corr_sigma_v_m * cfg_.gnss_corr_sigma_v_m;
  Eigen::Vector3d sig2(sig_h2, sig_h2, sig_v2);
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
  const Covariance p6 = covariance();
  const Eigen::LDLT<Covariance> ldlt = p6.ldlt();
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
    const OrbitOd::Covariance p = filter.covariance();
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

Eigen::Matrix3d rtnBasis(const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
  const double r_norm = r.norm();
  const Eigen::Vector3d h = r.cross(v);
  const double h_norm = h.norm();
  if (!(r_norm > 0.0) || !(h_norm > 0.0) || !r.allFinite() || !v.allFinite()) {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::Vector3d r_hat = r / r_norm;
  const Eigen::Vector3d n_hat = h / h_norm;
  const Eigen::Vector3d t_hat = n_hat.cross(r_hat);
  Eigen::Matrix3d m;
  m.col(0) = r_hat;
  m.col(1) = t_hat;
  m.col(2) = n_hat;
  return m;
}

double smaSigma(const Eigen::Vector3d& r_m, const Eigen::Vector3d& v_m_s,
                const OrbitOd::Covariance& p, double mu_m3_per_s2) {
  if (!r_m.allFinite() || !v_m_s.allFinite() || !p.allFinite() || !(mu_m3_per_s2 > 0.0)) {
    return -1.0;
  }
  const double r = r_m.norm();
  if (!(r > 0.0)) {
    return -1.0;
  }
  // Vis-viva: 1/a = 2/r − v²/μ; TP Eq. 2.21–2.23.
  const double inv_a = 2.0 / r - v_m_s.squaredNorm() / mu_m3_per_s2;
  if (!(inv_a > 0.0)) {
    return -1.0;  // unbound: no semi-major axis to have an error in
  }
  const double a = 1.0 / inv_a;
  Eigen::Matrix<double, 1, 6> f;
  f.head<3>() = (2.0 * a * a / (r * r * r)) * r_m.transpose();
  f.tail<3>() = (2.0 * a * a / mu_m3_per_s2) * v_m_s.transpose();
  const double var = (f * p * f.transpose())(0, 0);
  return var >= 0.0 && std::isfinite(var) ? std::sqrt(var) : -1.0;
}

double flightPathAngleSigma(const Eigen::Vector3d& r_m, const Eigen::Vector3d& v_m_s,
                            const OrbitOd::Covariance& p) {
  if (!r_m.allFinite() || !v_m_s.allFinite() || !p.allFinite()) {
    return -1.0;
  }
  const double r = r_m.norm();
  const double v = v_m_s.norm();
  if (!(r > 0.0) || !(v > 0.0)) {
    return -1.0;
  }
  const Eigen::Vector3d u_r = r_m / r;
  const Eigen::Vector3d u_v = v_m_s / v;
  const double sin_g = u_r.dot(u_v);
  const double cos2 = 1.0 - sin_g * sin_g;
  if (!(cos2 > 1e-12)) {
    return -1.0;  // radial motion: the flight-path angle is ±90° and undefined in derivative
  }
  // TP Eq. 2.25.
  Eigen::Matrix<double, 1, 6> f;
  f.head<3>() = ((u_v - sin_g * u_r) / r).transpose();
  f.tail<3>() = ((u_r - sin_g * u_v) / v).transpose();
  f /= std::sqrt(cos2);
  const double var = (f * p * f.transpose())(0, 0);
  return var >= 0.0 && std::isfinite(var) ? std::sqrt(var) : -1.0;
}

double alongTrackPsdFromOneOrbitError(double sigma_along_track_m, double period_s) {
  if (!(period_s > 0.0) || !std::isfinite(sigma_along_track_m)) {
    return 0.0;
  }
  return sigma_along_track_m * sigma_along_track_m / (3.0 * period_s * period_s * period_s);
}

Eigen::Matrix3d dmcNoiseKernel(double h_s, double tau_s) {
  Eigen::Matrix3d psi = Eigen::Matrix3d::Zero();
  if (!(h_s > 0.0) || !(tau_s > 0.0)) {
    return psi;
  }
  // Composite 5-point Gauss–Legendre on 4 panels: exact for polynomials to
  // degree 9 per panel, which covers Γ_r Γ_rᵀ ~ s⁴ exactly and leaves the
  // exponentials' residual at ~1e-12 relative for h ≤ τ/10 (the config bound).
  // Simpson at a fixed node count is *not* exact for the quartic and was 1e-5
  // relative off in the position block — measured, and why this is here.
  // Bounded: 20 evaluations.
  constexpr int kPanels = 4;
  constexpr double kNodes[5] = {-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683,
                                0.906179845938664};
  constexpr double kWeights[5] = {0.236926885056189, 0.478628670499366, 0.568888888888889,
                                  0.478628670499366, 0.236926885056189};
  const double panel = h_s / kPanels;
  for (int p = 0; p < kPanels; ++p) {
    const double s0 = panel * (static_cast<double>(p) + 0.5);
    for (int i = 0; i < 5; ++i) {
      const double s = s0 + 0.5 * panel * kNodes[i];
      const double e = std::exp(-s / tau_s);
      const double one_minus_e = -std::expm1(-s / tau_s);
      Eigen::Vector3d g;
      g << tau_s * (s - tau_s * one_minus_e), tau_s * one_minus_e, e;
      psi += (0.5 * panel * kWeights[i]) * (g * g.transpose());
    }
  }
  return psi;
}

}  // namespace polaris::gnc

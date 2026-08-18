/// @file
/// @brief The orbit filter's Push 73 items from NASA/TP-2018-219822 Ch. 2:
/// RTN state-noise compensation (§2.2.3.1), the DMC acceleration states
/// (§2.2.3.3), and the covariance metrics of §2.1 (REQ-ODP-011).
///
/// The truth for the DMC cases is the filter's own force model plus a
/// **constant unmodelled acceleration in RTN** — the shape of the error a
/// truncated field or a mis-modelled drag leaves, and exactly what the
/// exponentially-correlated acceleration states exist to absorb.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "gnc/orbit_od.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace {

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using pg::GnssFix;
using pg::NonGravAccelInput;
using pg::OrbitOd;
using pg::OrbitOdConfig;
using pg::OrbitOdRefusal;
using pg::OrbitOdResult;

constexpr double kAltitudeM = 400'000.0;
constexpr double kInclinationRad = 0.9006;
constexpr double kSigmaPosM = 1.0;
constexpr double kSigmaVelMps = 0.03;

pt::Tai testEpoch() {
  pt::UtcDateTime utc;
  utc.year = 2026;
  utc.month = 1;
  utc.day = 1;
  return pt::taiFromUtc(utc, pt::LeapSecondTable::historical());
}

pt::Tai advance(const pt::Tai& base, double seconds) {
  return base + pt::Duration::fromSecondsF(seconds);
}

pf::EopValue zeroEop(const pt::Tai& t) {
  pf::EopTable<8> table;
  const double mjd0 = std::floor(static_cast<double>(t.nanosecondsSinceEpoch()) / 1.0e9 /
                                 pc::time::kSecondsPerDay) +
                      pf::kMjd1970 - 1.0;
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(table.addEntry({mjd0 + static_cast<double>(i), 0.0, 0.0, 0.0}));
  }
  pf::EopValue v;
  EXPECT_TRUE(table.lookup(t, pt::LeapSecondTable::historical(), v));
  return v;
}

/// The reference filter configuration: J2 closed form, drag off, the Push 65
/// isotropic q_a — the flown SNC — with the DMC states off unless a case turns
/// them on.
OrbitOdConfig baseConfig() {
  OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.zonal_j2 = pc::gravity::kJ2;
  cfg.accel_psd_m2_per_s3 = 1.8e-7;
  cfg.position_nis_gate = 16.27;
  cfg.velocity_nis_gate = 16.27;
  cfg.max_coast_s = 1.0e9;
  cfg.max_degraded_coast_s = 1.0e9;
  cfg.max_dt_s = 10.0;
  cfg.max_step_s = 1.0;
  cfg.max_fix_latency_s = 0.2;
  cfg.min_radius_m = 6.5e6;
  cfg.max_radius_m = 8.0e6;
  return cfg;
}

void circularState(Eigen::Vector3d& r, Eigen::Vector3d& v) {
  const double radius = pc::gravity::kReferenceRadius + kAltitudeM;
  r = Eigen::Vector3d(radius, 0.0, 0.0);
  const double speed = std::sqrt(pc::gravity::kGM / radius);
  v = Eigen::Vector3d(0.0, speed * std::cos(kInclinationRad), speed * std::sin(kInclinationRad));
}

GnssFix fixFrom(const pt::Tai& epoch, const Eigen::Vector3d& r_eci, const Eigen::Vector3d& v_eci,
                const pf::EopValue& eop) {
  pm::Vec3<pmf::ECEF> r_ecef;
  pm::Vec3<pmf::ECEF> v_ecef;
  EXPECT_TRUE(pf::ecefStateFromEci(epoch, eop, pm::Vec3<pmf::ECI>(r_eci), pm::Vec3<pmf::ECI>(v_eci),
                                   r_ecef, v_ecef));
  GnssFix fix;
  fix.time_tag = pt::toGps(epoch);
  fix.position_m = r_ecef;
  fix.velocity_m_s = v_ecef;
  fix.position_sigma_h_m = kSigmaPosM;
  fix.position_sigma_v_m = kSigmaPosM;
  fix.velocity_sigma_m_s = kSigmaVelMps;
  fix.velocity_valid = true;
  return fix;
}

/// A truth propagator: the filter's own model driven, every second, by a
/// constant acceleration @p a_rtn expressed in the truth's own RTN basis. The
/// filter is not told about it (that is the point); it reaches it only through
/// the fixes.
class Truth {
 public:
  Truth(const OrbitOdConfig& cfg, const pt::Tai& epoch0, const Eigen::Vector3d& r0,
        const Eigen::Vector3d& v0, const Eigen::Vector3d& a_rtn, const pf::EopValue& eop)
      : od_(cfg), a_rtn_(a_rtn), eop_(eop) {
    OrbitOd::Covariance cov = OrbitOd::Covariance::Identity();
    EXPECT_EQ(od_.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0), cov),
              OrbitOdRefusal::kNone);
  }

  void stepTo(const pt::Tai& epoch) {
    while (od_.epoch() < epoch) {
      const double remaining = (epoch - od_.epoch()).seconds();
      const pt::Tai next = remaining <= 1.0 ? epoch : advance(od_.epoch(), 1.0);
      NonGravAccelInput in;
      in.accel_m_s2 =
          pm::Vec3<pmf::ECI>(pg::rtnBasis(od_.position().eigen(), od_.velocity().eigen()) * a_rtn_);
      in.sigma_m_s2 = 0.0;
      ASSERT_EQ(od_.propagate(next, eop_, &in), OrbitOdRefusal::kNone);
    }
  }

  Eigen::Vector3d r() const { return od_.position().eigen(); }

  Eigen::Vector3d v() const { return od_.velocity().eigen(); }

 private:
  OrbitOd od_;
  Eigen::Vector3d a_rtn_;
  pf::EopValue eop_;
};

}  // namespace

// ===========================================================================
// The DMC discrete-noise kernel (TP Eq. 2.55) and the RTN basis
// ===========================================================================

TEST(OrbitOdDmc, NoiseKernelMatchesHighOrderQuadratureAndItsSmallStepLimit) {
  RecordProperty("verifies", "REQ-ODP-011");
  const double tau = 600.0;
  // h ≤ τ/10, which `OrbitOdConfig::isValid` guarantees for the sub-step (the
  // fixed Gauss–Legendre rule is exact to round-off there; it is not asked to
  // cover h ~ τ, where the exponentials vary across the step).
  for (const double h : {0.1, 1.0, 10.0, 60.0}) {
    const Eigen::Matrix3d psi = pg::dmcNoiseKernel(h, tau);
    // Reference: composite Simpson at 20000 nodes on the same integrand (the
    // trapezoid at that count is only 1e-6 relative at h = 60 s — measured).
    Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
    const int n = 20000;
    const double ds = h / n;
    for (int i = 0; i <= n; ++i) {
      const double s = ds * i;
      const double e = std::exp(-s / tau);
      Eigen::Vector3d g;
      g << tau * (s - tau * (1.0 - e)), tau * (1.0 - e), e;
      const double w = (i == 0 || i == n) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
      ref += (w * ds / 3.0) * (g * g.transpose());
    }
    EXPECT_LT((psi - ref).norm(), 1e-9 * ref.norm() + 1e-30) << "h = " << h;
    // Symmetric, positive semi-definite.
    EXPECT_LT((psi - psi.transpose()).norm(), 1e-15 * psi.norm());
    EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(psi).eigenvalues().minCoeff(),
              -1e-12 * psi.norm());
  }
  // h ≪ τ: the random-walk-acceleration limit [h⁵/20, h⁴/8, h³/6; ., h³/3, h²/2; ., ., h].
  const double h = 0.1;
  const Eigen::Matrix3d psi = pg::dmcNoiseKernel(h, tau);
  EXPECT_NEAR(psi(0, 0), std::pow(h, 5) / 20.0, 1e-3 * std::pow(h, 5) / 20.0);
  EXPECT_NEAR(psi(0, 1), std::pow(h, 4) / 8.0, 1e-3 * std::pow(h, 4) / 8.0);
  EXPECT_NEAR(psi(0, 2), std::pow(h, 3) / 6.0, 1e-3 * std::pow(h, 3) / 6.0);
  EXPECT_NEAR(psi(1, 1), std::pow(h, 3) / 3.0, 1e-3 * std::pow(h, 3) / 3.0);
  EXPECT_NEAR(psi(1, 2), std::pow(h, 2) / 2.0, 1e-3 * std::pow(h, 2) / 2.0);
  EXPECT_NEAR(psi(2, 2), h, 1e-3 * h);
  // The exact acceleration-block value at any h: τ/2 (1 − e^{−2h/τ}) (TP Eq. 5.27).
  EXPECT_NEAR(pg::dmcNoiseKernel(60.0, tau)(2, 2), 0.5 * tau * (1.0 - std::exp(-120.0 / tau)),
              1e-9 * tau);
  // Degenerate arguments: zero.
  EXPECT_EQ(pg::dmcNoiseKernel(0.0, tau).norm(), 0.0);
  EXPECT_EQ(pg::dmcNoiseKernel(1.0, 0.0).norm(), 0.0);
}

TEST(OrbitOdDmc, RtnBasisIsOrthonormalAndOriented) {
  RecordProperty("verifies", "REQ-ODP-011");
  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  const Eigen::Matrix3d m = pg::rtnBasis(r, v);
  EXPECT_LT((m * m.transpose() - Eigen::Matrix3d::Identity()).norm(), 1e-14);
  EXPECT_NEAR(m.determinant(), 1.0, 1e-14);
  EXPECT_LT((m.col(0) - r.normalized()).norm(), 1e-14) << "R̂ along the radius";
  EXPECT_LT((m.col(2) - r.cross(v).normalized()).norm(), 1e-14) << "N̂ along h";
  EXPECT_GT(m.col(1).dot(v), 0.0) << "T̂ along the motion (circular: along v)";
  EXPECT_LT((pg::rtnBasis(Eigen::Vector3d::Zero(), v) - Eigen::Matrix3d::Identity()).norm(), 1e-15)
      << "degenerate: identity";
}

// ===========================================================================
// RTN state-noise compensation (TP §2.2.3.1) and Eq. 2.88
// ===========================================================================

TEST(OrbitOdDmc, RtnProcessNoiseGrowsTheVelocityAlongTheAxisItNames) {
  RecordProperty("verifies", "REQ-ODP-011");
  OrbitOdConfig cfg = baseConfig();
  cfg.accel_psd_m2_per_s3 = 0.0;
  cfg.accel_psd_rtn_m2_per_s3 = Eigen::Vector3d(0.0, 1.0e-6, 0.0);  // along-track only
  ASSERT_TRUE(cfg.isValid());
  OrbitOdConfig none = cfg;
  none.accel_psd_rtn_m2_per_s3.setZero();
  EXPECT_FALSE(none.isValid()) << "no white acceleration at all is refused";
  OrbitOdConfig neg = cfg;
  neg.accel_psd_rtn_m2_per_s3(0) = -1.0;
  EXPECT_FALSE(neg.isValid());

  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              1e-12 * OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);
  ASSERT_EQ(filter.propagate(advance(epoch0, 10.0), eop), OrbitOdRefusal::kNone);
  const Eigen::Matrix3d m = pg::rtnBasis(filter.position().eigen(), filter.velocity().eigen());
  const Eigen::Matrix3d p_vv = filter.covariance().block<3, 3>(3, 3);
  const Eigen::Matrix3d p_vv_rtn = m.transpose() * p_vv * m;
  // Velocity variance along T̂ is q_T·t; radial and normal see only the
  // dynamical coupling over 10 s, orders below.
  EXPECT_NEAR(p_vv_rtn(1, 1), 1.0e-6 * 10.0, 0.05 * 1.0e-6 * 10.0);
  EXPECT_LT(p_vv_rtn(0, 0), 1e-3 * p_vv_rtn(1, 1));
  EXPECT_LT(p_vv_rtn(2, 2), 1e-3 * p_vv_rtn(1, 1));

  // Eq. 2.88: a 100 m one-orbit along-track error on a 5554 s period.
  const double q_t = pg::alongTrackPsdFromOneOrbitError(100.0, 5554.0);
  EXPECT_NEAR(q_t, 100.0 * 100.0 / (3.0 * std::pow(5554.0, 3)), 1e-25);
  EXPECT_EQ(pg::alongTrackPsdFromOneOrbitError(100.0, 0.0), 0.0);
}

// ===========================================================================
// DMC: an unmodelled constant acceleration is estimated and carried into a coast
// ===========================================================================

TEST(OrbitOdDmc, UnmodelledAccelerationIsEstimatedAndClosesTheCoast) {
  RecordProperty("verifies", "REQ-ODP-011");
  // 2e-5 m/s² along-track — the size of the reference vehicle's field
  // truncation (1.28 m over 300 s ≈ 2·δr/T² = 2.8e-5), two orders above drag.
  const Eigen::Vector3d a_rtn(0.0, 2.0e-5, 0.0);
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOdConfig blind = baseConfig();
  OrbitOdConfig dmc = baseConfig();
  dmc.dmc_tau_s = 600.0;
  // Steady-state σ_a = 5e-5 m/s²: q = 2σ²/τ.
  dmc.dmc_psd_rtn_m2_per_s5 = Eigen::Vector3d::Constant(2.0 * 2.5e-9 / 600.0);
  ASSERT_TRUE(dmc.isValid());
  OrbitOdConfig too_short = dmc;
  too_short.dmc_tau_s = 5.0;  // under 10 sub-steps: refused (the kernel's own limit)
  EXPECT_FALSE(too_short.isValid());
  OrbitOd f_blind(blind);
  OrbitOd f_dmc(dmc);
  Truth truth(baseConfig(), epoch0, r0, v0, a_rtn, eop);

  // 900 s of 1 Hz fixes on the perturbed truth. Both filters take them; only
  // the DMC one can explain the drift as an acceleration.
  OrbitOdResult out;
  for (int i = 0; i <= 900; ++i) {
    const pt::Tai t = advance(epoch0, static_cast<double>(i));
    truth.stepTo(t);
    ASSERT_TRUE(f_blind.ingest(fixFrom(t, truth.r(), truth.v(), eop), eop, out))
        << "blind fix " << i;
    ASSERT_TRUE(f_dmc.ingest(fixFrom(t, truth.r(), truth.v(), eop), eop, out)) << "dmc fix " << i;
  }
  const Eigen::Vector3d a_est = f_dmc.dmcAcceleration();
  EXPECT_NEAR(a_est(1), 2.0e-5, 1.0e-5)
      << "along-track acceleration estimated: " << a_est.transpose();
  EXPECT_LT(std::abs(a_est(0)), 1.0e-5);
  EXPECT_LT(std::abs(a_est(2)), 1.0e-5);
  EXPECT_EQ(f_blind.dmcAcceleration().norm(), 0.0) << "off: inert";
  EXPECT_TRUE(f_dmc.covarianceHealthy());

  // A 300 s coast: the blind filter drifts by ~½at² = 0.9 m against the
  // acceleration it does not know; the DMC filter carries its estimate (decaying
  // at τ) and closes most of it.
  const pt::Tai t_end = advance(epoch0, 1200.0);
  for (int i = 901; i <= 1200; ++i) {
    const pt::Tai t = advance(epoch0, static_cast<double>(i));
    ASSERT_EQ(f_blind.propagate(t, eop), OrbitOdRefusal::kNone);
    ASSERT_EQ(f_dmc.propagate(t, eop), OrbitOdRefusal::kNone);
  }
  truth.stepTo(t_end);
  const double err_blind = (f_blind.position().eigen() - truth.r()).norm();
  const double err_dmc = (f_dmc.position().eigen() - truth.r()).norm();
  RecordProperty("coast_error_blind_m", std::to_string(err_blind));
  RecordProperty("coast_error_dmc_m", std::to_string(err_dmc));
  EXPECT_GT(err_blind, 0.5);
  EXPECT_LT(err_dmc, 0.5 * err_blind) << "blind " << err_blind << " m vs DMC " << err_dmc << " m";
  // And the DMC covariance covers its own coast error (single-run sanity, not
  // the campaign): NEES on the position/velocity marginal inside chi2_6(0.999).
  double nees = 0.0;
  ASSERT_TRUE(f_dmc.nees(pm::Vec3<pmf::ECI>(truth.r()), pm::Vec3<pmf::ECI>(truth.v()), nees));
  EXPECT_LT(nees, 22.5);

  // The 9-state covariance carries the states; the marginal is what consumers
  // see; reinitialisation keeps the estimate and re-opens the DMC block.
  EXPECT_GT(f_dmc.fullCovariance()(OrbitOd::kDmc + 1, OrbitOd::kDmc + 1), 0.0);
  EXPECT_EQ(f_dmc.covariance().rows(), 6);
  ASSERT_EQ(f_dmc.reinitializeCovariance(10.0, 0.1), OrbitOdRefusal::kNone);
  EXPECT_NEAR(f_dmc.fullCovariance()(OrbitOd::kDmc, OrbitOd::kDmc), 2.5e-9, 1e-15);
  EXPECT_NEAR(f_dmc.dmcAcceleration()(1), a_est(1) * std::exp(-300.0 / 600.0), 1e-9)
      << "the estimate decayed through the coast at τ and survived the re-init";
}

// ===========================================================================
// Covariance metrics (TP §2.1)
// ===========================================================================

TEST(OrbitOdDmc, SmaAndFlightPathAngleSigmasMatchTheClosedFormsOnACircularOrbit) {
  RecordProperty("verifies", "REQ-ODP-011");
  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  const double a = r.norm();
  const double mu = pc::gravity::kGM;
  const double sr = 2.0;
  const double sv = 0.05;
  OrbitOd::Covariance p = OrbitOd::Covariance::Zero();
  p.block<3, 3>(0, 0) = sr * sr * Eigen::Matrix3d::Identity();
  p.block<3, 3>(3, 3) = sv * sv * Eigen::Matrix3d::Identity();
  // Circular, isotropic: σ_a² = 4a⁴ (σ_r²/r⁴ + v² σ_v²/μ²) (TP Eq. 2.18 with ρ=0).
  const double expected_sa =
      2.0 * a * a * std::sqrt(sr * sr / std::pow(a, 4) + v.squaredNorm() * sv * sv / (mu * mu));
  EXPECT_NEAR(pg::smaSigma(r, v, p, mu), expected_sa, 1e-9 * expected_sa);
  // The TP's near-circular form Eq. 2.20: σ_a = 2 sqrt(σ_r² + (T/2π)² σ_v²).
  const double tp = 2.0 * M_PI * std::sqrt(a * a * a / mu);
  EXPECT_NEAR(pg::smaSigma(r, v, p, mu),
              2.0 * std::sqrt(sr * sr + std::pow(tp / (2.0 * M_PI), 2) * sv * sv),
              1e-6 * expected_sa);
  // Circular γ = 0: σ_γ² = σ_r²/r² + σ_v²/v² (Eq. 2.33 with in-track / bi-normal).
  const double expected_fpa = std::sqrt(sr * sr / (a * a) + sv * sv / v.squaredNorm());
  EXPECT_NEAR(pg::flightPathAngleSigma(r, v, p), expected_fpa, 1e-9 * expected_fpa);
  // Correlation and balance (TP §2.1.1): a negative r–v correlation along the
  // radial/velocity pair reduces σ_a below the uncorrelated value.
  OrbitOd::Covariance pc_ = p;
  const Eigen::Vector3d rh = r.normalized();
  const Eigen::Vector3d vh = v.normalized();
  pc_.block<3, 3>(0, 3) = -0.9 * sr * sv * rh * vh.transpose();
  pc_.block<3, 3>(3, 0) = pc_.block<3, 3>(0, 3).transpose();
  EXPECT_LT(pg::smaSigma(r, v, pc_, mu), pg::smaSigma(r, v, p, mu));
  // Refusals.
  EXPECT_LT(pg::smaSigma(r, v * 3.0, p, mu), 0.0) << "hyperbolic: no SMA";
  EXPECT_LT(pg::smaSigma(r, v, p, 0.0), 0.0);
  EXPECT_LT(pg::flightPathAngleSigma(r, r * 1e-3, p), 0.0) << "radial motion: undefined";
}

/// @file
/// @brief The orbit filter's drag scale factor — §8.5 tier-3 disturbance
/// estimation, orbit half (Push 76; REQ-ODP-013).
///
/// The onboard force model flies a *static* exponential atmosphere with no
/// solar or geomagnetic activity in it, so its density is wrong by a factor of
/// order one against the real one. The scale factor is the standard remedy for
/// a force-model coefficient known worse than the measurements (TP §2.2.3.4;
/// Tapley, Schutz & Born §4.16 [tapley2004] for the augmented-state form).
///
/// Truth here is the filter's *own* model with a different reference density —
/// which is precisely the error being estimated, and nothing else, so a test
/// that fails indicts the scale factor rather than a second modelling
/// difference standing in for it. The campaign (`tests/mc/orbit_od_mc.cpp`)
/// is where the real NRLMSIS-against-exponential mismatch is measured; this
/// file establishes that the mechanism is correct first.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
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
using pg::OrbitOd;
using pg::OrbitOdConfig;
using pg::OrbitOdRefusal;
using pg::OrbitOdResult;

constexpr double kAltitudeM = 400'000.0;
constexpr double kInclinationRad = 0.9006;
constexpr double kSigmaPosM = 1.0;
constexpr double kSigmaVelMps = 0.03;

/// The flown reference density (`config/spacecraft/leo_smallsat.yaml`:
/// Vallado Table 8-4, 400 km band) and ballistic coefficient (2.2·0.06/12).
constexpr double kRefDensityKgM3 = 3.725e-12;
constexpr double kBallisticCoeff = 0.011;
constexpr double kScaleHeightM = 58'515.0;

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

/// The reference configuration: J2 closed form, drag **on** at the flown
/// coefficients, the flown isotropic q_a, drag scale factor off unless a case
/// turns it on.
OrbitOdConfig baseConfig() {
  OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.zonal_j2 = pc::gravity::kJ2;
  cfg.drag_ballistic_coeff_m2_per_kg = kBallisticCoeff;
  cfg.drag_ref_density_kg_m3 = kRefDensityKgM3;
  cfg.drag_ref_altitude_m = kAltitudeM;
  cfg.drag_scale_height_m = kScaleHeightM;
  cfg.accel_psd_m2_per_s3 = 1.8e-7;
  cfg.position_nis_gate = 16.27;
  cfg.velocity_nis_gate = 16.27;
  cfg.max_coast_s = 1.0e9;
  cfg.max_degraded_coast_s = 1.0e9;
  cfg.max_dt_s = 60.0;
  cfg.max_step_s = 1.0;
  cfg.max_fix_latency_s = 0.2;
  cfg.min_radius_m = 6.5e6;
  cfg.max_radius_m = 8.0e6;
  return cfg;
}

/// @p cfg with the drag scale factor enabled. τ is long against the arcs flown
/// here — the scale is a slowly-varying model error, not a fast disturbance —
/// and the seed σ says "the static atmosphere could be 50 % wrong", which for a
/// model with no solar activity in it is the honest prior.
OrbitOdConfig withDragScale(OrbitOdConfig cfg, double psd = 1.0e-8) {
  cfg.drag_scale_tau_s = 86'400.0;
  cfg.drag_scale_psd_per_s = psd;
  cfg.drag_scale_seed_sigma = 0.5;
  cfg.drag_scale_max_deviation = 0.9;
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

/// Truth: the filter's own model with the reference density scaled by
/// @p true_scale. A denser atmosphere than the filter believes in, and nothing
/// else different.
class Truth {
 public:
  Truth(const OrbitOdConfig& cfg, double true_scale, const pt::Tai& epoch0,
        const Eigen::Vector3d& r0, const Eigen::Vector3d& v0, const pf::EopValue& eop)
      : od_(scaled(cfg, true_scale)), eop_(eop) {
    EXPECT_EQ(od_.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                             OrbitOd::Covariance::Identity()),
              OrbitOdRefusal::kNone);
  }

  void stepTo(const pt::Tai& epoch) {
    while (od_.epoch() < epoch) {
      const double remaining = (epoch - od_.epoch()).seconds();
      const pt::Tai next = remaining <= 10.0 ? epoch : advance(od_.epoch(), 10.0);
      ASSERT_EQ(od_.propagate(next, eop_), OrbitOdRefusal::kNone);
    }
  }

  Eigen::Vector3d r() const { return od_.position().eigen(); }

  Eigen::Vector3d v() const { return od_.velocity().eigen(); }

 private:
  static OrbitOdConfig scaled(OrbitOdConfig cfg, double s) {
    cfg.drag_ref_density_kg_m3 *= s;
    // Truth carries no scale-factor state of its own — it *is* the truth.
    cfg.drag_scale_psd_per_s = 0.0;
    cfg.drag_scale_tau_s = 0.0;
    cfg.drag_scale_seed_sigma = 0.0;
    cfg.drag_scale_max_deviation = 0.0;
    return cfg;
  }

  OrbitOd od_;
  pf::EopValue eop_;
};

/// Fly @p filter against a truth at @p true_scale for @p duration_s, feeding a
/// noiseless fix every @p fix_period_s. Returns the filter by reference.
void flyAgainstTruth(OrbitOd& filter, double true_scale, double duration_s, double fix_period_s,
                     const pt::Tai& epoch0, const pf::EopValue& eop) {
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(r0, v0);
  Truth truth(baseConfig(), true_scale, epoch0, r0, v0, eop);

  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);

  for (double t = fix_period_s; t <= duration_s; t += fix_period_s) {
    const pt::Tai now = advance(epoch0, t);
    truth.stepTo(now);
    OrbitOdResult out;
    ASSERT_TRUE(filter.ingest(fixFrom(now, truth.r(), truth.v(), eop), eop, out))
        << "fix refused at t = " << t << " s";
  }
}

}  // namespace

// ===========================================================================
// The Jacobian column, which is the whole reason the drag term was factored out
// ===========================================================================

/// `∂a/∂s` is the drag acceleration at unit scale — exact, not a difference.
///
/// This is the one partial in the filter that is written analytically rather
/// than central-differenced, so it is the one that can be analytically wrong.
/// Differencing `onboardAcceleration` in the scale is the independent check.
TEST(OrbitOdDragScale, JacobianColumnIsTheUnitScaleDragAcceleration) {
  RecordProperty("verifies", "REQ-ODP-013");
  const OrbitOdConfig cfg = baseConfig();
  const pt::Tai epoch = testEpoch();
  const pf::EopValue eop = zeroEop(epoch);
  pg::EarthOrientation earth;
  ASSERT_TRUE(pg::earthOrientationAt(epoch, eop, earth));

  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  const pm::Vec3<pmf::ECI> rv(r);
  const pm::Vec3<pmf::ECI> vv(v);

  const Eigen::Vector3d analytic = pg::dragAcceleration(cfg, rv, vv, earth).eigen();
  ASSERT_GT(analytic.norm(), 0.0) << "drag is off — this test would pass vacuously";

  const double ds = 1.0e-6;
  const Eigen::Vector3d numeric = (pg::onboardAcceleration(cfg, rv, vv, earth, 1.0 + ds).eigen() -
                                   pg::onboardAcceleration(cfg, rv, vv, earth, 1.0 - ds).eigen()) /
                                  (2.0 * ds);

  // 1e-6 relative, which is the central difference's own floor here and not the
  // partial's: at ds = 1e-6 on a ~1e-6 m/s^2 term the difference cancels twelve
  // significant figures before dividing, so ~1e-8 relative is all it can
  // resolve. Measured 3.7e-9. The exact check is the linearity one below.
  EXPECT_LT((analytic - numeric).norm(), 1.0e-6 * analytic.norm())
      << "analytic " << analytic.transpose() << " vs differenced " << numeric.transpose();

  // And the term really is linear in the scale, which is what makes one column
  // enough — a quadratic term would make the exact partial state-dependent.
  // This one *is* machine-exact: no differencing, no cancellation.
  const Eigen::Vector3d at_two = pg::onboardAcceleration(cfg, rv, vv, earth, 2.0).eigen();
  const Eigen::Vector3d at_one = pg::onboardAcceleration(cfg, rv, vv, earth, 1.0).eigen();
  EXPECT_LT(((at_two - at_one) - analytic).norm(), 1.0e-12 * analytic.norm());
}

// ===========================================================================
// Off means off
// ===========================================================================

/// With the PSD at zero the filter is the pre-Push-76 filter, exactly.
///
/// Asserted bit-for-bit rather than to a tolerance: the disabled path adds no
/// arithmetic to the state at all, so anything but an identical trajectory
/// means the tenth state is leaking into a configuration that never asked for
/// it. This is what lets the flown tuning stay unchanged while the state ships.
TEST(OrbitOdDragScale, DisabledStateLeavesTheTrajectoryBitForBit) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOd off(baseConfig());
  OrbitOd also_off(withDragScale(baseConfig(), /*psd=*/0.0));
  ASSERT_TRUE(also_off.isConfigured());

  flyAgainstTruth(off, 1.6, 6000.0, 60.0, epoch0, eop);
  flyAgainstTruth(also_off, 1.6, 6000.0, 60.0, epoch0, eop);

  EXPECT_EQ(off.position().eigen(), also_off.position().eigen());
  EXPECT_EQ(off.velocity().eigen(), also_off.velocity().eigen());
  EXPECT_EQ(off.dragScale(), 1.0);
  EXPECT_EQ(also_off.dragScale(), 1.0);
  EXPECT_EQ(also_off.dragScaleSigma(), 0.0);
}

/// A PSD with drag switched off does not create a state.
///
/// It would be a state with no path to any measurement: its variance would grow
/// on every step and never be reduced by anything, which is a covariance that
/// only ever gets worse. Refusing to carry it is cheaper than explaining it.
TEST(OrbitOdDragScale, NoStateWithoutDrag) {
  RecordProperty("verifies", "REQ-ODP-013");
  OrbitOdConfig cfg = withDragScale(baseConfig());
  cfg.drag_ballistic_coeff_m2_per_kg = 0.0;
  OrbitOd filter(cfg);
  ASSERT_TRUE(filter.isConfigured());

  const pt::Tai epoch0 = testEpoch();
  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r), pm::Vec3<pmf::ECI>(v),
                              OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);
  EXPECT_EQ(filter.dragScaleSigma(), 0.0);
  EXPECT_EQ(filter.dragScale(), 1.0);
}

// ===========================================================================
// What it is for
// ===========================================================================

/// The mechanism, on an arc where the geopotential truncation does not drown it.
///
/// 1.6 is the size of error a static exponential atmosphere really makes against
/// a real one over the solar cycle, not a token perturbation. `q_a` is reduced
/// by 10^3 from the flown value — **not** a tuning proposal (see the test below
/// for why it cannot be flown), but the isolation that shows the estimator
/// itself is sound: with the truncation budget out of the way, one orbit
/// recovers the atmosphere.
///
/// Measured: s = 1.516 against a true 1.6, sigma 0.5 -> 0.158.
TEST(OrbitOdDragScale, RecoversADensityBiasWhenTheTruncationBudgetIsOutOfTheWay) {
  RecordProperty("verifies", "REQ-ODP-013");
  const double true_scale = 1.6;
  const double seed_sigma = 0.5;
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOdConfig cfg = withDragScale(baseConfig());
  cfg.accel_psd_m2_per_s3 /= 1000.0;
  OrbitOd filter(cfg);
  ASSERT_TRUE(filter.isConfigured());

  flyAgainstTruth(filter, true_scale, 5500.0, 30.0, epoch0, eop);

  const double estimated = filter.dragScale();
  const double sigma = filter.dragScaleSigma();

  // Both halves matter: an estimate that moved with an unchanged sigma is a
  // filter being pushed around rather than one measuring.
  EXPECT_GT(estimated, 1.4) << "scale reached only " << estimated;
  EXPECT_LT(std::abs(estimated - true_scale), 0.2)
      << "estimated " << estimated << " against a true " << true_scale;
  EXPECT_LT(sigma, 0.5 * seed_sigma) << "sigma " << sigma << " has not come down from its prior";
  // The truth is inside the published band — the honesty check a
  // converged-looking estimate has to pass to mean anything.
  EXPECT_LT(std::abs(estimated - true_scale), 3.0 * sigma)
      << "truth outside the published 3-sigma: " << estimated << " +/- " << sigma;

  RecordProperty("estimated_scale", std::to_string(estimated));
  RecordProperty("scale_sigma", std::to_string(sigma));
}

/// **The flown tuning cannot resolve the atmosphere, and this is what says so.**
///
/// The isotropic `q_a` is sized from the 8x8 geopotential truncation — 1.28 m
/// over the 300 s coast horizon, an equivalent constant acceleration of
/// 2.8e-5 m/s^2. The whole drag-scale signal being estimated is
/// (s-1)*a_drag = 0.6 * 1.2e-6 = 7.2e-7 m/s^2, **39x smaller**. The filter is
/// therefore right to attribute the along-track signature to process noise
/// rather than to the atmosphere, and the scale factor barely moves.
///
/// Measured over six orbits at the flown tuning: s = 1.023 of a true 1.6 — 4 %
/// of the error recovered — with sigma still 0.33 of its 0.5 prior. Arc length
/// is not the lever (5500 s gives 1.006, 33000 s gives 1.023); `q_a` is, and
/// Push 74 measured that every reduced-`q_a` variant fails the NEES consistency
/// gate precisely *because* `q_a` is covering that truncation. The two are
/// locked together on this vehicle, so what unblocks the drag scale factor is a
/// higher-degree onboard field, not a longer pass. See the file header.
///
/// This test is an **upper bound that is expected to fail** the day the onboard
/// field improves — which is the point. It is asserted rather than merely
/// recorded so that the day the ratio changes, this says so.
TEST(OrbitOdDragScale, FlownTuningLeavesTheAtmosphereUnresolved) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOd filter(withDragScale(baseConfig()));  // the flown q_a
  ASSERT_TRUE(filter.isConfigured());

  flyAgainstTruth(filter, 1.6, 33'000.0, 30.0, epoch0, eop);

  const double estimated = filter.dragScale();
  const double sigma = filter.dragScaleSigma();

  // It moves the right way — the estimator is working, not inert.
  EXPECT_GT(estimated, 1.0);
  // But nowhere near the truth, and the sigma says so rather than pretending.
  EXPECT_LT(estimated, 1.10) << "the truncation budget stopped dominating: " << estimated
                             << " — re-derive the 39x ratio in this test's comment";
  EXPECT_GT(sigma, 0.25) << "sigma " << sigma << " claims a resolution the signal cannot support";

  RecordProperty("flown_estimated_scale", std::to_string(estimated));
  RecordProperty("flown_scale_sigma", std::to_string(sigma));
}

/// On a short arc the sigma says the estimate is still the prior.
///
/// This is the property that keeps the number honest on a vehicle or an arc
/// where drag is not resolvable — the same role `DipoleEstimator`'s gates play
/// in the attitude half of tier 3 (§8.5). A scale factor that reported a
/// confident value after four minutes would be reporting its own seed.
TEST(OrbitOdDragScale, ShortArcLeavesTheSigmaAtItsPrior) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOd filter(withDragScale(baseConfig()));
  flyAgainstTruth(filter, 1.6, 240.0, 30.0, epoch0, eop);

  // Barely reduced from the 0.5 prior: four minutes of a 400 km orbit carries
  // essentially no secular along-track drag signature.
  EXPECT_GT(filter.dragScaleSigma(), 0.9 * 0.5)
      << "sigma " << filter.dragScaleSigma() << " claims a four-minute arc resolved the atmosphere";
  RecordProperty("short_arc_sigma", std::to_string(filter.dragScaleSigma()));
}

// ===========================================================================
// The band: refused, not clamped
// ===========================================================================

/// An out-of-band scale is refused, the last accepted value stands, and the
/// position/velocity solution the same fix carried is applied anyway.
///
/// The mechanism is exercised by shrinking the band to something the recovery
/// case above is known to cross, which is a cleaner provocation than inventing
/// a pathological measurement: the same fixes, the same filter, one number
/// different.
TEST(OrbitOdDragScale, OutOfBandScaleIsRefusedNotClamped) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOdConfig cfg = withDragScale(baseConfig());
  cfg.accel_psd_m2_per_s3 /= 1000.0;    // so the estimate actually travels
  cfg.drag_scale_max_deviation = 0.05;  // the 1.6 truth is far outside this
  OrbitOd filter(cfg);
  ASSERT_TRUE(filter.isConfigured());

  flyAgainstTruth(filter, 1.6, 5500.0, 30.0, epoch0, eop);

  EXPECT_GT(filter.dragScaleRefusedCount(), 0u) << "the band was never crossed — test is vacuous";
  // Held inside the band, and *not* sitting exactly on its edge, which is what
  // a clamp would produce.
  EXPECT_LE(std::abs(filter.dragScale() - 1.0), 0.05);
  EXPECT_NE(filter.dragScale(), 1.05);
  // The vehicle still has a navigation solution: refusing a scale factor is not
  // refusing the fix.
  EXPECT_TRUE(filter.isInitialised());
  EXPECT_EQ(filter.quality(), pg::OrbitOdQuality::kFine);

  RecordProperty("refused_updates", std::to_string(filter.dragScaleRefusedCount()));
}

/// Enabling the state by uplink on a running filter opens it to its prior.
///
/// The trap this closes: while the state is off its covariance block is
/// identically zero, and zero variance is a state the Kalman gain can never
/// reach — enabling it would leave a permanently frozen scale that only the
/// process noise could open, over months. `retune` therefore re-seeds that one
/// block on the off->on transition, and **only** that block: the position and
/// velocity solution the vehicle is navigating on is untouched, which is the
/// whole point of re-tuning in place (NESC TB 20-03 item g; TP §9.3).
TEST(OrbitOdDragScale, EnablingByRetuneOpensTheStateToItsPrior) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOd filter(baseConfig());  // drag scale off
  flyAgainstTruth(filter, 1.6, 3000.0, 30.0, epoch0, eop);
  ASSERT_EQ(filter.dragScaleSigma(), 0.0);
  const Eigen::Vector3d r_before = filter.position().eigen();
  const Eigen::Vector3d v_before = filter.velocity().eigen();

  OrbitOdConfig on = withDragScale(baseConfig());
  ASSERT_EQ(filter.retune(on), OrbitOdRefusal::kNone);

  EXPECT_DOUBLE_EQ(filter.dragScaleSigma(), on.drag_scale_seed_sigma);
  EXPECT_EQ(filter.dragScale(), 1.0);
  // The navigation solution is exactly what it was.
  EXPECT_EQ(filter.position().eigen(), r_before);
  EXPECT_EQ(filter.velocity().eigen(), v_before);

  // And re-tuning again while already on does *not* re-seed: that would throw
  // away a converged scale on every unrelated parameter upload.
  OrbitOdConfig also_on = on;
  also_on.max_coast_s = 400.0;
  ASSERT_EQ(filter.retune(also_on), OrbitOdRefusal::kNone);
  EXPECT_DOUBLE_EQ(filter.dragScaleSigma(), on.drag_scale_seed_sigma);
}

/// A commanded reset returns the scale to its nominal and clears the count.
TEST(OrbitOdDragScale, ResetReturnsTheScaleToNominal) {
  RecordProperty("verifies", "REQ-ODP-013");
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOdConfig cfg = withDragScale(baseConfig());
  cfg.accel_psd_m2_per_s3 /= 1000.0;
  OrbitOd filter(cfg);
  flyAgainstTruth(filter, 1.6, 5500.0, 30.0, epoch0, eop);
  ASSERT_NE(filter.dragScale(), 1.0);

  filter.reset();
  EXPECT_EQ(filter.dragScale(), 1.0);
  EXPECT_EQ(filter.dragScaleRefusedCount(), 0u);
  EXPECT_FALSE(filter.isInitialised());
}

// ===========================================================================
// Configuration is a trust boundary
// ===========================================================================

/// A half-configured scale factor is refused at construction, not flown.
TEST(OrbitOdDragScale, PartialConfigurationIsRefused) {
  RecordProperty("verifies", "REQ-ODP-013");
  const OrbitOdConfig good = withDragScale(baseConfig());
  ASSERT_TRUE(good.isValid());

  // A PSD with no correlation time, seed sigma or band.
  OrbitOdConfig no_tau = good;
  no_tau.drag_scale_tau_s = 0.0;
  EXPECT_FALSE(no_tau.isValid());

  OrbitOdConfig no_seed = good;
  no_seed.drag_scale_seed_sigma = 0.0;
  EXPECT_FALSE(no_seed.isValid());

  OrbitOdConfig no_band = good;
  no_band.drag_scale_max_deviation = 0.0;
  EXPECT_FALSE(no_band.isValid());

  // A band at or past 1 admits a negative scale — drag pushing the vehicle
  // along its own velocity, which is a different sign of physics rather than a
  // large error.
  OrbitOdConfig wide_band = good;
  wide_band.drag_scale_max_deviation = 1.0;
  EXPECT_FALSE(wide_band.isValid());

  // τ under ten sub-steps: the same quadrature limit the DMC kernel carries,
  // since it is the same kernel.
  OrbitOdConfig fast_tau = good;
  fast_tau.drag_scale_tau_s = 5.0 * good.max_step_s;
  EXPECT_FALSE(fast_tau.isValid());

  OrbitOdConfig negative = good;
  negative.drag_scale_psd_per_s = -1.0;
  EXPECT_FALSE(negative.isValid());
}

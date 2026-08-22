/// @file Unit tests for the onboard orbit determination filter
/// (`lib/gnc/orbit_od`, design doc §8.3).
///
/// Four groups, deliberately separated because they answer different questions
/// and a pass in one says nothing about the others:
///
///  1. **Force model against closed form** — a circular orbit closes, the
///     two-body part conserves energy, drag dissipates it, and the J2 term
///     produces the textbook nodal regression rate. These catch sign errors,
///     unit slips and a mis-axed field on quantities with an analytic answer.
///     The *absolute* correctness of the propagator against an independent tool
///     is `tests/golden/orbit_od_golden_test.cpp`'s job (GMAT), not this file's.
///  2. **The numerical Jacobian against the analytic gravity gradient** — the
///     one part of the filter where a closed form exists and was deliberately
///     not written by hand, so the choice has to be paid for with a test.
///  3. **Refusals** — every path that declines to act, by name. A refusal that
///     is never exercised is a refusal that will be wrong when it fires.
///  4. **Consistency** — the truncation measurement that sizes the process
///     noise, and the NEES/NIS Monte Carlo that says the covariance means what
///     it claims. A filter with small errors and an optimistic covariance passes
///     an accuracy test and fails this one, and that failure shows up later as
///     innovation gating rejecting good measurements.

#include "gnc/orbit_od.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "state/estimated_state.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"
#include "world/eop_file.hpp"

namespace {

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace ps = polaris::state;
namespace scenario = polaris::sim::scenario;

using pg::GnssFix;
using pg::GnssMeasurementPolicy;
using pg::MeasurementMode;
using pg::NonGravAccelInput;
using pg::OrbitOd;
using pg::OrbitOdConfig;
using pg::OrbitOdQuality;
using pg::OrbitOdRefusal;
using pg::OrbitOdResult;

// ---------------------------------------------------------------------------
// The reference vehicle and orbit these tests fly
// ---------------------------------------------------------------------------

/// ISS-class circular LEO: 400 km altitude, 51.6° inclination. Chosen because it
/// is where the drag term matters, where the geopotential truncation is largest,
/// and because the golden fixture's `iss_leo` case is the same regime — so the
/// truncation measured here and the one measured against GMAT are comparable.
constexpr double kAltitudeM = 400'000.0;
constexpr double kInclinationRad = 51.6 * 3.14159265358979323846 / 180.0;

/// The **coast horizon** [s] (design doc §8.3).
///
/// Sized from the outage modes the receiver model actually has (§6.2), not
/// picked round: the reference NovAtel OEM7600 quotes a 34 s cold-start
/// time-to-first-fix and a 0.5 s signal reacquisition, and `config/hardware/gnss`
/// carries both. 300 s is ~9x the dominant one, which covers a cold start plus a
/// missed reacquisition plus a run of dropped fixes, and is short enough that the
/// coarse force model's error over it is still comparable to one fix's noise —
/// which is the condition that makes the process noise below a fair model of it.
constexpr double kCoastHorizonS = 300.0;

/// Force-model truncation over the coast horizon [m] — the **design input** for
/// the process noise, and nothing else may set it.
///
/// Measured at **1.28 m** by `CoarseModelTruncationOverTheCoastHorizon` below,
/// against the GMAT-validated truth sim at 32x32 gravity, and carried **at the
/// measurement**. It used to be carried at 1.8 m, ~40% above it, and that
/// margin was the process noise being over-budgeted: the OD Monte Carlo
/// (`analysis/od`) measured the campaign NEES at 4.22 against a 95% floor of
/// 4.83, all of it velocity, and re-flying the nominal scenario across the
/// margin put NEES at 4.68 / 5.28 / 6.81 for 1.5 / 1.28 / 1.0 m — consistent
/// only at the measurement. The margin was there so the CI fence below does not
/// flap with the epoch, and it belongs on the fence, not on the flight tuning;
/// hence `kTruncationFenceM` beside this, which is the only place headroom is
/// added.
///
/// **This value dropped from 5.0 m when the onboard force model gained the 8x8
/// EGM2008 field**, and the drop is smaller than the field's own contribution
/// suggests because the reference got harder at the same time. On one arc, both
/// models against the same 32x32 truth: the closed-form J2 model diverges
/// **3.17 m** and the 8x8 field **1.28 m**, a 2.5x reduction. The older 3.59 m
/// figure was measured against an *8x8* truth, which the onboard model now
/// matches — an 8x8-vs-8x8 comparison measures two implementations of one field
/// and reports 0.045 m, which is why the truth degree was raised rather than the
/// number simply banked. `CoarseModelTruncationOverTheCoastHorizon` records both
/// models every run and fails if the field ever stops winning.
///
/// The measured growth is still very close to **quadratic** — 0.0138 m at 30 s
/// against 1.280 m at 300 s, a factor of 93 for a factor of 10 in time — which
/// is the direct evidence that what remains is a *systematic acceleration* and
/// not noise, and therefore why the CWNA fit below is documented as an
/// approximation matched at the horizon rather than as a description. Improving
/// the field did not change that character; it lowered the coefficient.
constexpr double kTruncationAtHorizonM = 1.28;

/// The CI fence on that measurement [m]: `CoarseModelTruncationOverTheCoastHorizon`
/// fails above it. Sits ~17% over the measurement so a shift in epoch (the
/// geopotential residual depends on where in the orbit the arc starts) or a
/// legitimate truth-model improvement does not fail the build, and low enough
/// that a force-model regression the process noise no longer covers is caught
/// long before it reaches the ensemble check's ~13% noise floor at 30 runs. The
/// epoch sensitivity itself is unmeasured; 17% is a judgement, not a number, and
/// is not fed into anything the filter flies with.
constexpr double kTruncationFenceM = 1.5;

/// Process-noise acceleration PSD [m²/s³], from the CWNA position spread
/// `σ_r(T) = √(q_a T³/3)` matched to the truncation at the horizon:
/// `q_a = 3 δr(T)² / T³`. Written as the formula, not as the evaluated number,
/// so the derivation cannot be quietly severed from its input.
constexpr double kAccelPsd = 3.0 * kTruncationAtHorizonM * kTruncationAtHorizonM /
                             (kCoastHorizonS * kCoastHorizonS * kCoastHorizonS);

/// χ² 99.9% quantiles for 3 degrees of freedom — the NIS gate for a position or
/// velocity update, neither of which has a degenerate direction.
constexpr double kChi2_3_999 = 16.266;

/// NovAtel OEM7600 single-point figures, per `config/hardware/gnss`. The
/// horizontal per-axis σ is the datasheet's 1.2 m 2D-RMS over √2; vertical is
/// the model's 1.5x default.
constexpr double kSigmaHM = 1.2 / 1.4142135623730951;
constexpr double kSigmaVM = 1.5 * kSigmaHM;
constexpr double kSigmaVelMps = 0.03;

/// Largest fix latency the filter absorbs [s].
///
/// `config/hardware/gnss/novatel_oem7600.yaml` carries `fix_latency_s: 0.05` —
/// the datasheet's 20 ms measurement latency plus bus transport and a scheduler
/// slot. 0.2 s is 4x that, which covers a missed GNC cycle or two without
/// becoming a licence to accept a fix from a stopped clock: at 7.6 km/s the bound
/// itself is 1.5 km of forward propagation, and the point of having a bound at
/// all is that the extrapolation stays short enough for the fix's own velocity to
/// carry it.
constexpr double kMaxFixLatencyS = 0.2;

/// Reference vehicle ballistic properties: 12 kg, 0.06 m² ram area, C_d = 2.2.
constexpr double kBallisticCoeff = 2.2 * 0.06 / 12.0;

/// The onboard filter configuration under test.
///
/// The geopotential constants are taken as a **consistent set** from
/// `constants::gravity` — GM, reference radius and J2 all from the same
/// geopotential solution — for the reason that namespace documents and the
/// golden test measured: mixing WGS84's GM or radius into an EGM-solved J2 costs
/// centimetres to decimetres per revolution.
///
/// The drag band is one band of the **same** Vallado Table 8-4 piecewise
/// exponential fit the truth sim's `exponentialDensity` interpolates
/// (`sim/world/atmosphere.cpp`), selected at the operating altitude. That is
/// deliberate: it makes the onboard/truth drag difference exactly "one band
/// versus the whole table", a stated and bounded approximation, rather than two
/// unrelated atmospheres whose disagreement is unattributable.
OrbitOdConfig referenceConfig() {
  OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  // The flown gravity model: the compiled-in 8x8 EGM2008 truncation, not the
  // closed-form J2 the filter originally carried. `zonal_j2` stays zero — the
  // harmonic field already contains the degree-2 zonal, and `isValid` refuses a
  // config that sets both.
  cfg.geopotential_degree = 8;
  cfg.geopotential_order = 8;
  cfg.drag_ballistic_coeff_m2_per_kg = kBallisticCoeff;
  cfg.drag_ref_density_kg_m3 = 3.725e-12;  // Vallado Table 8-4, 400 km band
  cfg.drag_ref_altitude_m = 400'000.0;
  cfg.drag_scale_height_m = 58'515.0;
  cfg.accel_psd_m2_per_s3 = kAccelPsd;
  cfg.position_nis_gate = kChi2_3_999;
  cfg.velocity_nis_gate = kChi2_3_999;
  cfg.max_coast_s = kCoastHorizonS;
  cfg.max_degraded_coast_s = kCoastHorizonS;  // the pre-Push-70 policy; the horizon
                                              // tests set their own
  cfg.max_dt_s = 10.0;
  cfg.max_step_s = 1.0;
  cfg.max_fix_latency_s = kMaxFixLatencyS;
  // The band is sized to *this vehicle's* orbit, not to Earth orbits in general.
  // It is the only check on the seed path, where there is no prior and so no NIS
  // gate; see `OrbitOdConfig::min_radius_m`. 120 km to 1600 km altitude.
  cfg.min_radius_m = 6.5e6;
  cfg.max_radius_m = 8.0e6;
  return cfg;
}

/// Lift the coast horizon out of the way, for the tests that exercise the
/// **propagator** over arcs of orbits.
///
/// Not a way around the design: the horizon is a validity *policy* about how
/// long an unaided solution may be trusted, and it is tested on its own in
/// `CoastHorizonExpiryDropsTheSolutionAndTheNextFixReacquiresWhole`. Leaving it
/// at 300 s here would drop the solution mid-arc and every force-model test
/// would fail for a reason that has nothing to do with the force model.
OrbitOdConfig propagationOnly(OrbitOdConfig cfg) {
  cfg.max_coast_s = 1.0e9;
  cfg.max_degraded_coast_s = 1.0e9;
  return cfg;
}

/// Gravity-only variant, for the analytic checks that need an isolated term.
/// Gravity-only variant, for the analytic checks that need an isolated term.
///
/// These drop back to the **closed-form** two-body and J2 paths rather than
/// asking the harmonic field for degree 0 or degree 2 order 0. The point of an
/// analytic check is to compare the flown code against an expression written
/// independently of it, and the closed forms are the ones the textbooks print;
/// `geopotential_test.cpp` separately pins the harmonic evaluator's degree-0 and
/// degree-2/order-0 cases onto exactly these same closed forms, so the chain is
/// closed without either test standing in for the other.
OrbitOdConfig twoBodyConfig() {
  OrbitOdConfig cfg = propagationOnly(referenceConfig());
  cfg.geopotential_degree = 0;
  cfg.geopotential_order = 0;
  cfg.zonal_j2 = 0.0;
  cfg.drag_ballistic_coeff_m2_per_kg = 0.0;
  return cfg;
}

OrbitOdConfig j2OnlyConfig() {
  OrbitOdConfig cfg = twoBodyConfig();
  cfg.zonal_j2 = pc::gravity::kJ2;
  return cfg;
}

// ---------------------------------------------------------------------------
// Fixtures: epoch, EOP, initial state
// ---------------------------------------------------------------------------

pt::Tai testEpoch() {
  pt::UtcDateTime utc;
  utc.year = 2026;
  utc.month = 1;
  utc.day = 1;
  utc.hour = 0;
  utc.minute = 0;
  utc.second = 0;
  return pt::taiFromUtc(utc, pt::LeapSecondTable::historical());
}

pt::Tai advance(const pt::Tai& base, double seconds) {
  return base + pt::Duration::fromSecondsF(seconds);
}

/// A small synthetic EOP table spanning three days around @p around.
///
/// UT1−UTC and polar motion are zero. That is not a claim they are negligible —
/// the reduction's numeric accuracy is pinned against ERFA in
/// `eci_ecef_test.cpp` and against the real committed series in
/// `eop_golden_test.cpp` — it is that these tests are about the *filter*, and a
/// synthetic table keeps them from re-parsing a 3.6 MB fixture per case. The one
/// test that must match the truth sim's Earth orientation loads the real file.
template <std::size_t Capacity = 8>
std::unique_ptr<pf::EopTable<Capacity>> makeEop(const pt::Tai& around, int days = 3) {
  auto table = std::make_unique<pf::EopTable<Capacity>>();
  const double mjd0 = std::floor(static_cast<double>(around.nanosecondsSinceEpoch()) / 1.0e9 /
                                 pc::time::kSecondsPerDay) +
                      pf::kMjd1970 - 1.0;
  for (int i = 0; i < days; ++i) {
    EXPECT_TRUE(table->addEntry({mjd0 + static_cast<double>(i), 0.0, 0.0, 0.0}));
  }
  return table;
}

pf::EopValue eopAt(const pt::Tai& t) {
  const auto table = makeEop(t);
  pf::EopValue v;
  EXPECT_TRUE(table->lookup(t, pt::LeapSecondTable::historical(), v));
  return v;
}

/// Circular orbit state at @p altitude and @p inclination, with the ascending
/// node on the ECI X axis and the vehicle at the node.
void circularState(double altitude, double inclination, double mu, Eigen::Vector3d& r,
                   Eigen::Vector3d& v) {
  const double radius = pc::gravity::kReferenceRadius + altitude;
  r = Eigen::Vector3d(radius, 0.0, 0.0);
  const double speed = std::sqrt(mu / radius);
  v = Eigen::Vector3d(0.0, speed * std::cos(inclination), speed * std::sin(inclination));
}

/// Seed covariance from the receiver's own figures — the covariance the filter
/// would build for itself out of a fix.
OrbitOd::Covariance receiverSeedCovariance() {
  OrbitOd::Covariance cov = OrbitOd::Covariance::Zero();
  cov.block<3, 3>(OrbitOd::kPosition, OrbitOd::kPosition) =
      Eigen::Matrix3d::Identity() * (kSigmaVM * kSigmaVM);
  cov.block<3, 3>(OrbitOd::kVelocity, OrbitOd::kVelocity) =
      Eigen::Matrix3d::Identity() * (kSigmaVelMps * kSigmaVelMps);
  return cov;
}

/// Propagate @p filter to `epoch0 + t_s` in steps the config accepts.
[[nodiscard]] bool walkTo(OrbitOd& filter, const pt::Tai& epoch0, double t_s, double step_s,
                          const pf::EopValue& eop) {
  const pt::Tai target = advance(epoch0, t_s);
  while (filter.epoch() < target) {
    const double remaining = (target - filter.epoch()).seconds();
    const pt::Tai next = (remaining <= step_s) ? target : advance(filter.epoch(), step_s);
    if (filter.propagate(next, eop) != OrbitOdRefusal::kNone) {
      return false;
    }
  }
  return true;
}

/// Build the GNSS fix a perfect receiver would report for an ECI truth state:
/// rotate to ECEF through the one reduction, stamp GPS time. Noise is the
/// caller's to add — a test that wants a clean fix gets a clean one, and the
/// σ fields are still the receiver's, since a receiver reports its accuracy
/// whether or not this particular draw happened to be zero.
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
  fix.position_sigma_h_m = kSigmaHM;
  fix.position_sigma_v_m = kSigmaVM;
  fix.velocity_sigma_m_s = kSigmaVelMps;
  fix.velocity_valid = true;
  return fix;
}

/// Truth state at `epoch0 + t_s`, propagated on the same force model by an
/// independent filter instance used purely as a propagator.
///
/// Fixes built from the *initial* state instead would sit ~7.6 km from where the
/// filter has propagated to after a single second, which the NIS gate correctly
/// rejects — so a test meaning to exercise something else would end up
/// exercising the gate.
void truthAt(const OrbitOdConfig& cfg, const pt::Tai& epoch0, const Eigen::Vector3d& r0,
             const Eigen::Vector3d& v0, double t_s, const pf::EopValue& eop, Eigen::Vector3d& r,
             Eigen::Vector3d& v) {
  OrbitOd propagator(propagationOnly(cfg));
  ASSERT_EQ(propagator.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                                  receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  ASSERT_TRUE(walkTo(propagator, epoch0, t_s, 1.0, eop));
  r = propagator.position().eigen();
  v = propagator.velocity().eigen();
}

/// Specific orbital energy [J/kg] of a two-body state.
double specificEnergy(const Eigen::Vector3d& r, const Eigen::Vector3d& v, double mu) {
  return 0.5 * v.squaredNorm() - mu / r.norm();
}

/// Right ascension of the ascending node [rad] from a state vector.
double raan(const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
  const Eigen::Vector3d h = r.cross(v);
  // Node vector n = ẑ × h; its right ascension is the RAAN. atan2, per the house
  // rule — the components carry the quadrant and an acos of a ratio would not.
  const Eigen::Vector3d n = Eigen::Vector3d::UnitZ().cross(h);
  return std::atan2(n.y(), n.x());
}

}  // namespace

// ===========================================================================
// 1. Force model against closed form
// ===========================================================================

/// A circular two-body orbit returns to its starting state after exactly one
/// period. Catches a wrong μ, a sign error in the central term, and any
/// integrator defect that is not a pure phase error, all in one number.
TEST(OrbitOdForceModel, CircularTwoBodyOrbitClosesAfterOnePeriod) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = twoBodyConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const double radius = r0.norm();
  const double period = 2.0 * M_PI * std::sqrt(radius * radius * radius / cfg.mu_m3_per_s2);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  ASSERT_TRUE(walkTo(filter, epoch0, period, 10.0, eop));

  // 1 mm over a 5550 s orbit is the RK4 truncation at h = 1 s; the closure
  // itself is exact in the model.
  EXPECT_LT((filter.position().eigen() - r0).norm(), 1.0e-3);
  EXPECT_LT((filter.velocity().eigen() - v0).norm(), 1.0e-6);
}

/// The two-body part is conservative: specific orbital energy is invariant.
///
/// This is a weak test on its own — a smoothly wrong field conserves energy too,
/// which is exactly why the GMAT golden comparison exists — but it is the check
/// that separates "the force model is wrong" from "the integrator is leaking",
/// and it is the baseline the drag test below is a departure from.
TEST(OrbitOdForceModel, TwoBodyEnergyIsConserved) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = twoBodyConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);

  const double e0 = specificEnergy(r0, v0, cfg.mu_m3_per_s2);
  ASSERT_TRUE(walkTo(filter, epoch0, 3000.0, 10.0, eop));
  const double e1 =
      specificEnergy(filter.position().eigen(), filter.velocity().eigen(), cfg.mu_m3_per_s2);
  EXPECT_LT(std::abs((e1 - e0) / e0), 1.0e-12);
}

/// Drag removes energy, and it removes about as much as the closed form says.
///
/// The rate of energy loss for a circular orbit is `Ė = a_drag · v = −½ρB v³`,
/// which is a one-line prediction from the model's own inputs. Checking the
/// *magnitude* rather than only the sign is what catches a ballistic coefficient
/// off by a factor (a mass in the wrong place, a missing ½) — a sign test would
/// pass all of those.
TEST(OrbitOdForceModel, DragDissipatesEnergyAtTheClosedFormRate) {
  RecordProperty("verifies", "REQ-ODP-001");
  // Two-body gravity plus drag: `specificEnergy` below is the *two-body*
  // integral, which only a two-body field conserves. Under the flown 8x8 field
  // the geopotential's own short-period energy exchange is three orders above
  // the drag loss being measured, so isolating drag means isolating the whole
  // harmonic field, not just its degree-2 zonal.
  OrbitOdConfig cfg = twoBodyConfig();
  cfg.drag_ballistic_coeff_m2_per_kg = kBallisticCoeff;
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);

  constexpr double kArcS = 2000.0;
  const double e0 = specificEnergy(r0, v0, cfg.mu_m3_per_s2);
  ASSERT_TRUE(walkTo(filter, epoch0, kArcS, 10.0, eop));
  const double e1 =
      specificEnergy(filter.position().eigen(), filter.velocity().eigen(), cfg.mu_m3_per_s2);
  EXPECT_LT(e1, e0) << "drag must remove energy";

  // Predicted loss, using the atmosphere-relative speed the model actually flies
  // (co-rotating atmosphere) rather than the inertial one — at 51.6° that is a
  // ~5% difference in v_rel and ~15% in v_rel³, which is more than the tolerance
  // below, so getting it wrong here would look like a model defect.
  const Eigen::Vector3d omega = pc::wgs84::kEarthRate * Eigen::Vector3d::UnitZ();
  const double v_rel = (v0 - omega.cross(r0)).norm();
  const double density = cfg.drag_ref_density_kg_m3;  // at exactly the reference altitude
  const double predicted =
      -0.5 * cfg.drag_ballistic_coeff_m2_per_kg * density * v_rel * v_rel * v_rel * kArcS;
  // 15%: the orbit decays slightly over the arc so ρ, v and the geometry all
  // move, and the prediction holds them fixed at t = 0.
  EXPECT_NEAR(e1 - e0, predicted, 0.15 * std::abs(predicted));
}

/// The J2 term produces the textbook **nodal regression** rate
/// `Ω̇ = −(3/2) n J₂ (Rₑ/p)² cos i` (Vallado §9.6; Montenbruck & Gill §3.2).
///
/// This is the sharpest analytic check available on a zonal field, because the
/// secular node drift depends on the coefficient, the reference radius, the
/// inclination *and* the axis the field is referenced to, all at once. A field
/// evaluated about the wrong pole, a J2 with the wrong sign, or a reference
/// radius off by a percent each move this number out of band; none of them
/// touches the energy check above.
TEST(OrbitOdForceModel, J2ProducesTheTextbookNodalRegression) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = j2OnlyConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const double radius = r0.norm();
  const double n = std::sqrt(cfg.mu_m3_per_s2 / (radius * radius * radius));
  const double re_over_p = cfg.reference_radius_m / radius;  // circular: p = a = r
  const double expected_rate =
      -1.5 * n * cfg.zonal_j2 * re_over_p * re_over_p * std::cos(kInclinationRad);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);

  // Three orbits: long enough that the secular drift dominates the short-period
  // oscillation, short enough to stay in the linear regime the formula is for.
  const double period = 2.0 * M_PI / n;
  const double arc = 3.0 * period;
  const double raan0 = raan(r0, v0);
  ASSERT_TRUE(walkTo(filter, epoch0, arc, 10.0, eop));
  const double raan1 = raan(filter.position().eigen(), filter.velocity().eigen());

  double drift = raan1 - raan0;
  while (drift > M_PI) {
    drift -= 2.0 * M_PI;
  }
  while (drift < -M_PI) {
    drift += 2.0 * M_PI;
  }
  const double measured_rate = drift / arc;

  // 2%: the mean-element formula is first order in J2 and the state-vector RAAN
  // carries the short-period term the formula averages away, which is O(J2)
  // relative — so sub-percent agreement would be luck, not accuracy.
  EXPECT_NEAR(measured_rate, expected_rate, 0.02 * std::abs(expected_rate));
  // And the sign is worth asserting on its own: a prograde orbit regresses.
  EXPECT_LT(measured_rate, 0.0);
}

/// A **polar** orbit has no nodal regression — `cos i = 0`. The complementary
/// check to the one above: it fails if the J2 term has any inclination-blind
/// component, which is what an isotropic or mis-derived implementation produces.
TEST(OrbitOdForceModel, PolarOrbitHasNoNodalRegression) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = j2OnlyConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, M_PI / 2.0, cfg.mu_m3_per_s2, r0, v0);
  const double radius = r0.norm();
  const double period = 2.0 * M_PI * std::sqrt(radius * radius * radius / cfg.mu_m3_per_s2);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);

  const double raan0 = raan(r0, v0);
  ASSERT_TRUE(walkTo(filter, epoch0, 3.0 * period, 10.0, eop));
  const double raan1 = raan(filter.position().eigen(), filter.velocity().eigen());

  // The reference for "small": what an equal-and-opposite 51.6° orbit drifts in
  // the same time. A polar orbit must be two orders under it, not merely
  // "close to zero" against an absolute number nobody can interpret.
  const double n = 2.0 * M_PI / period;
  const double re_over_p = cfg.reference_radius_m / radius;
  const double inclined_drift = std::abs(-1.5 * n * cfg.zonal_j2 * re_over_p * re_over_p *
                                         std::cos(kInclinationRad) * 3.0 * period);
  EXPECT_LT(std::abs(raan1 - raan0), 0.01 * inclined_drift);
}

/// With the pole on ẑ the closed-form J2 acceleration must reduce to the
/// textbook component form. Pins the vector identity used in `orbit_od.cpp`
/// against the expression every reference prints.
TEST(OrbitOdForceModel, J2AccelerationMatchesTheTextbookComponentForm) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = j2OnlyConfig();
  const Eigen::Vector3d r(4.0e6, 3.0e6, 4.5e6);
  const Eigen::Vector3d v(0.0, 7.0e3, 0.0);
  // Identity orientation puts the pole on ẑ, which is what makes the closed form
  // below the textbook one.
  const pg::EarthOrientation earth;

  const Eigen::Vector3d a =
      pg::onboardAcceleration(cfg, pm::Vec3<pmf::ECI>(r), pm::Vec3<pmf::ECI>(v), earth).eigen();

  // a = a_pointmass + a_J2, with (Montenbruck & Gill §3.2, Eq. 3.30)
  //   a_J2,{x,y} = -(3/2) J2 (μ/r²)(Re/r)² (·/r)(1 - 5z²/r²)
  //   a_J2,z     = -(3/2) J2 (μ/r²)(Re/r)² (z/r)(3 - 5z²/r²)
  const double rm = r.norm();
  const double zr2 = (r.z() / rm) * (r.z() / rm);
  const double scale = -1.5 * cfg.zonal_j2 * (cfg.mu_m3_per_s2 / (rm * rm)) *
                       (cfg.reference_radius_m / rm) * (cfg.reference_radius_m / rm);
  Eigen::Vector3d expected = -(cfg.mu_m3_per_s2 / (rm * rm * rm)) * r;
  expected.x() += scale * (r.x() / rm) * (1.0 - 5.0 * zr2);
  expected.y() += scale * (r.y() / rm) * (1.0 - 5.0 * zr2);
  expected.z() += scale * (r.z() / rm) * (3.0 - 5.0 * zr2);

  EXPECT_LT((a - expected).norm(), 1.0e-15 * expected.norm());
}

// ===========================================================================
// 2. The numerical Jacobian against the analytic gravity gradient
// ===========================================================================

/// The central-difference Jacobian the covariance propagation runs on must
/// reproduce the closed-form two-body gravity gradient `−μ/r³(I − 3r̂r̂ᵀ)`.
///
/// The header argues that a numerical Jacobian is preferable to a hand-written
/// analytic one because it cannot drift from the force model it differentiates.
/// This is the price of that argument: the accuracy claim it raises is settled
/// by measurement, on the one term where the answer is known exactly.
///
/// The Jacobian is a private detail, so it is exercised the way the filter uses
/// it — through a covariance propagation. Over a short step with zero process
/// noise, `P⁺ = ΦPΦᵀ` with `Φ ≈ I + Fh`, so seeding `P = I` and differencing
/// recovers `F` to first order, and the position/velocity block is `∂a/∂r`.
TEST(OrbitOdJacobian, MatchesTheClosedFormTwoBodyGravityGradient) {
  RecordProperty("verifies", "REQ-ODP-001");
  OrbitOdConfig cfg = twoBodyConfig();
  cfg.accel_psd_m2_per_s3 = 1.0e-30;  // isolate ΦPΦᵀ from Q
  cfg.max_step_s = 1.0e-3;
  cfg.max_dt_s = 1.0e-3;

  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOd::Covariance seed = OrbitOd::Covariance::Identity();
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0), seed),
            OrbitOdRefusal::kNone);

  constexpr double h = 1.0e-3;
  ASSERT_EQ(filter.propagate(advance(epoch0, h), eop), OrbitOdRefusal::kNone);

  // With P = I: ΦΦᵀ = (I + Fh + …)(I + Fᵀh + …) = I + (F + Fᵀ)h + O(h²), so
  // `(P⁺ − I)/h = F + Fᵀ`. Its (velocity, position) block is
  // `F_vp + (F_pv)ᵀ = ∂a/∂r + Iᵀ`, because `F_pv = ∂ṙ/∂v` is exactly the
  // identity — so `∂a/∂r` is recovered by subtracting that known term.
  const OrbitOd::Covariance dp = (filter.covariance() - seed) / h;
  const Eigen::Matrix3d measured =
      dp.block<3, 3>(OrbitOd::kVelocity, OrbitOd::kPosition) - Eigen::Matrix3d::Identity();

  const double rm = r0.norm();
  const Eigen::Vector3d r_hat = r0 / rm;
  const Eigen::Matrix3d expected = -(cfg.mu_m3_per_s2 / (rm * rm * rm)) *
                                   (Eigen::Matrix3d::Identity() - 3.0 * r_hat * r_hat.transpose());

  // Relative to the gradient's own scale (μ/r³ ≈ 1.3e-6 s⁻²). The bound is set by
  // the O(h) truncation of Φ ≈ I + Fh at h = 1 ms, not by the difference stencil.
  const double scale = cfg.mu_m3_per_s2 / (rm * rm * rm);
  EXPECT_LT((measured - expected).norm(), 1.0e-5 * scale) << "measured:\n"
                                                          << measured << "\nexpected:\n"
                                                          << expected;
}

// ===========================================================================
// 3. Refusals
// ===========================================================================

TEST(OrbitOdConfigValidation, RejectsEveryOutOfRangeField) {
  RecordProperty("verifies", "REQ-ODP-001");
  EXPECT_TRUE(referenceConfig().isValid());

  // A default-constructed config is deliberately invalid: there are no flight
  // defaults worth trusting (§19.3).
  EXPECT_FALSE(OrbitOdConfig{}.isValid());

  auto broken = [](void (*mutate)(OrbitOdConfig&)) {
    OrbitOdConfig cfg = referenceConfig();
    mutate(cfg);
    return cfg.isValid();
  };
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.mu_m3_per_s2 = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.zonal_j2 = -1.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.accel_psd_m2_per_s3 = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.position_nis_gate = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.velocity_nis_gate = -1.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.max_coast_s = 0.0; }));
  // The degraded horizon must be at least the fine one, and finite.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.max_degraded_coast_s = c.max_coast_s - 1.0; }));
  EXPECT_FALSE(broken(
      [](OrbitOdConfig& c) { c.max_degraded_coast_s = std::numeric_limits<double>::infinity(); }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.max_step_s = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.min_radius_m = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.max_radius_m = 1.0; }));
  EXPECT_FALSE(
      broken([](OrbitOdConfig& c) { c.mu_m3_per_s2 = std::numeric_limits<double>::quiet_NaN(); }));

  // A zonal or drag term with no reference radius is a silently disabled force,
  // not a defaulted one (P57) — rejected at construction.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.reference_radius_m = 0.0; }));
  // Drag with no atmosphere is the same class.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.drag_ref_density_kg_m3 = 0.0; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.drag_scale_height_m = 0.0; }));
  // ...but a *zero ballistic coefficient* is the documented way to disable drag,
  // and then the atmosphere parameters are legitimately unset.
  {
    OrbitOdConfig cfg = referenceConfig();
    cfg.drag_ballistic_coeff_m2_per_kg = 0.0;
    cfg.drag_ref_density_kg_m3 = 0.0;
    cfg.drag_scale_height_m = 0.0;
    EXPECT_TRUE(cfg.isValid());
  }

  // The sub-step loop's compile-time bound must not be able to truncate a
  // legitimate propagation, so a config it would truncate is refused up front.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) {
    c.max_step_s = 1.0;
    c.max_dt_s = static_cast<double>(OrbitOd::kMaxSubsteps) + 1.0;
  }));

  // --- Harmonic field ------------------------------------------------------
  // Degree must be inside the compiled-in table; a request the evaluator would
  // silently clamp is refused here instead, so a config asking for a fidelity
  // the build does not have fails loudly rather than flying a lower one.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.geopotential_degree = -1; }));
  EXPECT_FALSE(broken(
      [](OrbitOdConfig& c) { c.geopotential_degree = polaris::gnc::kGeopotentialMaxDegree + 1; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.geopotential_order = c.geopotential_degree + 1; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.geopotential_order = -1; }));

  // Both gravity models at once is two descriptions of the same physics with
  // one of them silently ignored.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.zonal_j2 = pc::gravity::kJ2; }));

  // The field uses the table's own GM and reference radius, so a config
  // disagreeing with them describes two different Earths — and the drag altitude
  // is computed from the config's radius, so the disagreement is not academic.
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.mu_m3_per_s2 *= 1.001; }));
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.reference_radius_m *= 1.001; }));
  // ...but WGS84's constants against EGM2008's coefficients is a legitimate
  // pairing (7.5e-10 in GM), and must stay legal.
  {
    OrbitOdConfig cfg = referenceConfig();
    cfg.mu_m3_per_s2 = pc::wgs84::kGM;
    EXPECT_TRUE(cfg.isValid());
  }
  // With the field off, `zonal_j2` and a mismatched mu are legal again — the
  // closed-form path carries its own coefficient and scale.
  {
    OrbitOdConfig cfg = referenceConfig();
    cfg.geopotential_degree = 0;
    cfg.geopotential_order = 0;
    cfg.zonal_j2 = pc::gravity::kJ2;
    cfg.mu_m3_per_s2 *= 1.001;
    EXPECT_TRUE(cfg.isValid());
  }

  // --- Fix latency ---------------------------------------------------------
  EXPECT_FALSE(broken([](OrbitOdConfig& c) { c.max_fix_latency_s = -0.1; }));
  EXPECT_FALSE(broken(
      [](OrbitOdConfig& c) { c.max_fix_latency_s = std::numeric_limits<double>::quiet_NaN(); }));
  // Zero is the documented way to demand strictly-forward fixes.
  {
    OrbitOdConfig cfg = referenceConfig();
    cfg.max_fix_latency_s = 0.0;
    EXPECT_TRUE(cfg.isValid());
  }
}

TEST(OrbitOdRefusals, UnconfiguredFilterIsInertForever) {
  RecordProperty("verifies", "REQ-ODP-001");
  OrbitOd filter{OrbitOdConfig{}};
  EXPECT_FALSE(filter.isConfigured());

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  EXPECT_EQ(filter.propagate(advance(epoch0, 1.0), eop), OrbitOdRefusal::kUnconfigured);
  EXPECT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>::Zero(), pm::Vec3<pmf::ECI>::Zero(),
                              OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kUnconfigured);

  OrbitOdResult out;
  EXPECT_FALSE(filter.ingest(GnssFix{}, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kUnconfigured);
}

TEST(OrbitOdRefusals, UninitialisedPropagationRefusesRatherThanGuessing) {
  RecordProperty("verifies", "REQ-ODP-001");
  OrbitOd filter(referenceConfig());
  const pt::Tai epoch0 = testEpoch();
  EXPECT_EQ(filter.propagate(advance(epoch0, 1.0), eopAt(epoch0)), OrbitOdRefusal::kUninitialised);
  EXPECT_FALSE(filter.isInitialised());
  EXPECT_FALSE(filter.solutionValid());
}

TEST(OrbitOdRefusals, AGeoRadiusFixCannotSeedTheFilterOnALeoVehicle) {
  RecordProperty("verifies", "REQ-ODP-001");
  // A cold filter has no prior, so it has no innovation and therefore no NIS
  // gate: the plausibility band in `OrbitOdConfig` is the *only* thing standing
  // between a wire value and the state the vehicle then flies on. That makes
  // the band's width a real decision rather than a formality — it must be sized
  // to the orbit this vehicle is on, not to every orbit that exists. Sized to
  // admit GEO on a 400 km vehicle, this fix would be accepted whole.
  //
  // The update path is checked as well, because the two refuse for different
  // reasons and only one of them is the trust boundary: an initialised filter
  // would also reject this on the innovation, which is defence in depth and not
  // a substitute — it is unavailable at exactly the moment the band matters.
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  // Geostationary radius, the `bad_data` campaign scenario's injected value.
  constexpr double kGeoRadiusM = 4.2164e7;
  const Eigen::Vector3d r_geo = r0.normalized() * kGeoRadiusM;

  OrbitOd cold(cfg);
  OrbitOdResult out;
  EXPECT_FALSE(cold.ingest(fixFrom(epoch0, r_geo, v0, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixImplausible);
  EXPECT_FALSE(cold.isInitialised());
  EXPECT_FALSE(out.seeded);

  OrbitOd warm(cfg);
  ASSERT_TRUE([&] {
    OrbitOdResult seed;
    return warm.ingest(fixFrom(epoch0, r0, v0, eop), eop, seed) && seed.seeded;
  }());
  EXPECT_FALSE(warm.ingest(fixFrom(advance(epoch0, 10.0), r_geo, v0, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixImplausible);
  // Still on its own orbit rather than anywhere near the fix it refused. Not
  // asserted as "unchanged": `ingest` propagates to the fix epoch before it
  // judges the measurement, so the state has legitimately moved ten seconds
  // along the trajectory. What must not have happened is the 35000 km jump.
  EXPECT_TRUE(warm.solutionValid());
  EXPECT_NEAR(warm.position().eigen().norm(), r0.norm(), 1.0e3);
}

TEST(OrbitOdRefusals, NonFiniteFixIsRefusedWithoutTouchingTheSolution) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_TRUE([&] {
    OrbitOdResult out;
    return filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out) && out.seeded;
  }());
  const Eigen::Vector3d before = filter.position().eigen();
  const OrbitOd::Covariance p_before = filter.covariance();

  GnssFix bad = fixFrom(advance(epoch0, 1.0), r0, v0, eop);
  bad.position_m.eigen().x() = std::numeric_limits<double>::quiet_NaN();
  OrbitOdResult out;
  EXPECT_FALSE(filter.ingest(bad, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixNotFinite);
  EXPECT_EQ(filter.position().eigen(), before);
  EXPECT_EQ(filter.covariance(), p_before);
}

/// A **stuck** receiver clock is the dangerous one: re-presenting the same fix
/// at the same epoch would fold the same measurement in twice and shrink the
/// covariance on information already used. The covariance is the assertion here,
/// not the return value — a filter that refused but had already updated would
/// pass a return-code-only test.
TEST(OrbitOdRefusals, StuckClockCannotFoldTheSameFixInTwice) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  ASSERT_TRUE(out.seeded);

  // One good fix a second later, accepted.
  const pt::Tai t1 = advance(epoch0, 1.0);
  Eigen::Vector3d r1;
  Eigen::Vector3d v1;
  truthAt(cfg, epoch0, r0, v0, 1.0, eop, r1, v1);
  ASSERT_TRUE(filter.ingest(fixFrom(t1, r1, v1, eop), eop, out));
  ASSERT_TRUE(out.position.accepted);
  const OrbitOd::Covariance p_after_one = filter.covariance();

  // The same time tag again — refused, and the covariance is untouched. The
  // covariance is the assertion that matters: a filter that returned false but
  // had already folded the measurement in would pass a return-code-only check
  // while having shrunk its uncertainty on information it already used.
  EXPECT_FALSE(filter.ingest(fixFrom(t1, r1, v1, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNonMonotonicEpoch);
  EXPECT_EQ(filter.covariance(), p_after_one);

  // And backwards is refused too.
  EXPECT_FALSE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNonMonotonicEpoch);
  EXPECT_EQ(filter.covariance(), p_after_one);
}

TEST(OrbitOdRefusals, PropagationRefusesAStuckOrBackwardsClock) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);

  EXPECT_EQ(filter.propagate(epoch0, eop), OrbitOdRefusal::kNonMonotonicEpoch);
  EXPECT_EQ(filter.propagate(advance(epoch0, -1.0), eop), OrbitOdRefusal::kNonMonotonicEpoch);
  EXPECT_EQ(filter.position().eigen(), r0);

  // A gap longer than max_dt_s is a clock glitch, not a coast: refused rather
  // than integrated on a step the model was never validated over.
  EXPECT_EQ(filter.propagate(advance(epoch0, cfg.max_dt_s * 2.0), eop),
            OrbitOdRefusal::kStepTooLong);
  EXPECT_EQ(filter.position().eigen(), r0);
}

/// An epoch the uploaded EOP table does not cover cannot be brought into ECI at
/// all. The table does not extrapolate, so the fix is refused rather than
/// converted on a guessed Earth orientation.
TEST(OrbitOdRefusals, FixOutsideEopCoverageIsRefusedNotExtrapolated) {
  RecordProperty("verifies", "REQ-ODP-001;REQ-CONV-002");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const auto table = makeEop(epoch0);
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const pf::EopValue eop = eopAt(epoch0);

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), *table, leap, out));
  ASSERT_TRUE(out.seeded);

  // Ten days on, well past the three-day synthetic table.
  const pt::Tai far = advance(epoch0, 10.0 * 86400.0);
  GnssFix fix = fixFrom(epoch0, r0, v0, eop);
  fix.time_tag = pt::toGps(far);
  EXPECT_FALSE(filter.ingest(fix, *table, leap, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFrameConversion);

  // The propagation path refuses on the same grounds.
  EXPECT_EQ(filter.propagate(far, *table, leap), OrbitOdRefusal::kFrameConversion);
}

TEST(OrbitOdRefusals, ImplausibleRadiusAndBadSigmasAreRefused) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  OrbitOd filter(cfg);
  OrbitOdResult out;

  // A fix inside the Earth. Finite, plausible-looking to a finiteness check, and
  // it would poison the propagation while every validity flag still read true.
  GnssFix sub_surface = fixFrom(epoch0, r0, v0, eop);
  sub_surface.position_m = pm::Vec3<pmf::ECEF>(Eigen::Vector3d(1.0e6, 0.0, 0.0));
  EXPECT_FALSE(filter.ingest(sub_surface, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixImplausible);

  GnssFix zero_sigma = fixFrom(epoch0, r0, v0, eop);
  zero_sigma.position_sigma_h_m = 0.0;
  EXPECT_FALSE(filter.ingest(zero_sigma, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixSigmaInvalid);

  GnssFix bad_vel_sigma = fixFrom(epoch0, r0, v0, eop);
  bad_vel_sigma.velocity_sigma_m_s = -1.0;
  EXPECT_FALSE(filter.ingest(bad_vel_sigma, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kFixSigmaInvalid);

  // Position alone cannot start a 6-state, so a first fix with no velocity is
  // refused by name rather than seeded with an invented zero velocity.
  GnssFix no_velocity = fixFrom(epoch0, r0, v0, eop);
  no_velocity.velocity_valid = false;
  EXPECT_FALSE(filter.ingest(no_velocity, eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNoVelocityForSeed);
  EXPECT_FALSE(filter.isInitialised());
}

/// Past the coast horizon the solution is declared invalid **and dropped**, and
/// the next fix re-acquires whole rather than blending against a stale prior.
TEST(OrbitOdRefusals, CoastHorizonExpiryDropsTheSolutionAndTheNextFixReacquiresWhole) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));

  // Coast right up to the horizon: still valid, and the covariance has grown.
  const double p0 = filter.covariance()(OrbitOd::kPosition, OrbitOd::kPosition);
  ASSERT_TRUE(walkTo(filter, epoch0, kCoastHorizonS, cfg.max_dt_s, eop));
  EXPECT_TRUE(filter.solutionValid());
  EXPECT_NEAR(filter.ageSeconds(), kCoastHorizonS, 1.0e-6);
  EXPECT_GT(filter.covariance()(OrbitOd::kPosition, OrbitOd::kPosition), p0)
      << "an unaided coast must widen the position covariance";

  // One step past it: dropped.
  EXPECT_EQ(filter.propagate(advance(epoch0, kCoastHorizonS + cfg.max_dt_s), eop),
            OrbitOdRefusal::kCoastExpired);
  EXPECT_FALSE(filter.isInitialised());
  EXPECT_FALSE(filter.solutionValid());

  // The next fix re-acquires whole — `seeded`, and the covariance is the fix's
  // own rather than anything inherited from the discarded prior.
  const pt::Tai t_return = advance(epoch0, kCoastHorizonS + 2.0 * cfg.max_dt_s);
  Eigen::Vector3d r_ret;
  Eigen::Vector3d v_ret;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r_ret, v_ret);
  ASSERT_TRUE(filter.ingest(fixFrom(t_return, r_ret, v_ret, eop), eop, out));
  EXPECT_TRUE(out.seeded);
  EXPECT_TRUE(filter.solutionValid());
  EXPECT_LT((filter.position().eigen() - r_ret).norm(), 1.0e-6)
      << "a re-acquisition must take the fix whole, not blend it with the stale prior";
}

/// **A known thrust is propagated and budgeted (Push 70).** A constant
/// acceleration over a step moves the state by ½aΔt² relative to the unthrusted
/// propagation, and its uncertainty grows the covariance like a second white
/// acceleration over the step.
TEST(OrbitOdLatency, NonGravitationalAccelerationIsPropagatedAndBudgeted) {
  RecordProperty("verifies", "REQ-ODP-008");
  const OrbitOdConfig cfg = propagationOnly(referenceConfig());
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  OrbitOd coast(cfg);
  OrbitOd burn(cfg);
  ASSERT_EQ(coast.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                             receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  ASSERT_EQ(burn.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                            receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  const Eigen::Vector3d a = 0.03 * v0.normalized();  // along-track, a 30 mm/s² burn
  NonGravAccelInput input;
  input.accel_m_s2 = pm::Vec3<pmf::ECI>(a);
  input.sigma_m_s2 = 0.05 * 0.03;  // 5 % thrust knowledge
  const double dt = 1.0;
  ASSERT_EQ(coast.propagate(advance(epoch0, dt), eop), OrbitOdRefusal::kNone);
  ASSERT_EQ(burn.propagate(advance(epoch0, dt), eop, &input), OrbitOdRefusal::kNone);

  // Δr = ½aΔt², Δv = aΔt, to 1e-9 relative: over one second the gravity
  // gradient acting on the ~15 mm displacement is 1e-8 m — below the tolerance
  // only because the step is short, which is what makes the check analytic.
  const Eigen::Vector3d dr = burn.position().eigen() - coast.position().eigen();
  const Eigen::Vector3d dv = burn.velocity().eigen() - coast.velocity().eigen();
  EXPECT_LT((dr - 0.5 * a * dt * dt).norm(), 1.0e-9 * (0.5 * a * dt * dt).norm() + 1.0e-7);
  // The velocity picks up the gravity gradient across the 15 mm displacement
  // over the second (~6e-9 m/s), so the tolerance is 1e-8 rather than 1e-9.
  EXPECT_LT((dv - a * dt).norm(), 1.0e-9 * (a * dt).norm() + 1.0e-8);

  // Covariance: the extra term is σ_a²·h over the sub-step in the same CWNA
  // structure as q_a. With max_step_s = 1 the whole step is one sub-step, so
  // the velocity block grew by σ_a²·h·h = σ_a²·dt² relative to the coast.
  const double dpvv = burn.covariance()(OrbitOd::kVelocity, OrbitOd::kVelocity) -
                      coast.covariance()(OrbitOd::kVelocity, OrbitOd::kVelocity);
  EXPECT_NEAR(dpvv, input.sigma_m_s2 * input.sigma_m_s2 * dt * dt, 1.0e-3 * dpvv + 1.0e-18);
  EXPECT_GT(burn.covariance()(OrbitOd::kPosition, OrbitOd::kPosition),
            coast.covariance()(OrbitOd::kPosition, OrbitOd::kPosition));

  // Refused, state untouched, on a non-finite or negative-sigma input.
  NonGravAccelInput bad = input;
  bad.sigma_m_s2 = -1.0;
  const Eigen::Vector3d before = burn.position().eigen();
  EXPECT_EQ(burn.propagate(advance(epoch0, 2.0 * dt), eop, &bad), OrbitOdRefusal::kFixNotFinite);
  EXPECT_EQ(burn.position().eigen(), before);
}

/// **A ground seed (Push 70).** `seed()` builds an isotropic covariance and takes
/// the same plausibility gates as a fix; a bad σ or an implausible radius leaves
/// the filter untouched.
TEST(OrbitOdIngest, GroundSeedIsAcceptedAndItsRefusalsLeaveTheStateAlone) {
  RecordProperty("verifies", "REQ-ODP-008");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  OrbitOd filter(cfg);
  EXPECT_EQ(filter.seed(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0), 0.0, 1.0),
            OrbitOdRefusal::kFixSigmaInvalid);
  EXPECT_EQ(filter.seed(epoch0, pm::Vec3<pmf::ECI>(10.0 * r0), pm::Vec3<pmf::ECI>(v0), 1000.0, 1.0),
            OrbitOdRefusal::kFixImplausible);
  EXPECT_FALSE(filter.isInitialised());
  ASSERT_EQ(filter.seed(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0), 1000.0, 1.0),
            OrbitOdRefusal::kNone);
  EXPECT_TRUE(filter.solutionValid());
  EXPECT_EQ(filter.quality(), OrbitOdQuality::kFine);
  EXPECT_NEAR(filter.covariance()(OrbitOd::kPosition, OrbitOd::kPosition), 1.0e6, 1e-9);
  EXPECT_NEAR(filter.covariance()(OrbitOd::kVelocity, OrbitOd::kVelocity), 1.0, 1e-12);
  EXPECT_NEAR(filter.covariance()(OrbitOd::kPosition, OrbitOd::kVelocity), 0.0, 1e-12);
  // And it propagates from there.
  EXPECT_EQ(filter.propagate(advance(epoch0, 1.0), eopAt(epoch0)), OrbitOdRefusal::kNone);
}

/// A gross outlier — a spoofed fix, which §6.2 makes deliberately *valid* so
/// that catching it is the innovation gate's job and not a flag's — is rejected,
/// counted, and leaves the state alone.
TEST(OrbitOdRefusals, NisGateRejectsAnOutlierAndCountsIt) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  // A few clean fixes so the covariance tightens and the gate has teeth.
  Eigen::Vector3d r_t;
  Eigen::Vector3d v_t;
  for (int i = 1; i <= 5; ++i) {
    truthAt(cfg, epoch0, r0, v0, static_cast<double>(i), eop, r_t, v_t);
    ASSERT_TRUE(
        filter.ingest(fixFrom(advance(epoch0, static_cast<double>(i)), r_t, v_t, eop), eop, out))
        << "clean fix " << i;
  }
  ASSERT_EQ(filter.rejectedCount(), 0u);

  // 1 km of spoof offset against a ~1 m σ. §6.2 keeps a spoofed fix `valid` on
  // purpose, so nothing upstream of this gate is going to catch it.
  truthAt(cfg, epoch0, r0, v0, 6.0, eop, r_t, v_t);
  const Eigen::Vector3d spoofed = r_t + Eigen::Vector3d(1000.0, 0.0, 0.0);
  EXPECT_FALSE(filter.ingest(fixFrom(advance(epoch0, 6.0), spoofed, v_t, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementRejected);
  EXPECT_GT(out.position.nis, cfg.position_nis_gate);
  EXPECT_FALSE(out.position.accepted);
  EXPECT_EQ(filter.rejectedCount(), 1u);
  // The filter propagated to the fix epoch — that is legitimate — but the
  // *measurement* did not move it: the 1 km offset never entered the state.
  EXPECT_LT((filter.position().eigen() - r_t).norm(), 5.0);

  // A commanded reset clears the count; a fault would not.
  filter.reset();
  EXPECT_EQ(filter.rejectedCount(), 0u);
  EXPECT_FALSE(filter.isInitialised());
}

// ===========================================================================
// Ingest conversion, and the canonical state
// ===========================================================================

/// The GPS→TAI and ECEF→ECI conversion on ingest (REQ-CONV-001), end to end: a
/// fix built by rotating a known ECI state out must seed the filter back onto
/// that same ECI state.
///
/// The velocity is the load-bearing half. A rotation alone cannot transform a
/// state — ECEF is rotating, so the velocity carries the `ω⊕ × r` transport term
/// and skipping it is a ~465 m/s error at the equator. A test that only checked
/// position would pass with that bug in place.
TEST(OrbitOdIngest, ConvertsGpsTimeAndEcefStateOntoTheInertialFrame) {
  RecordProperty("verifies", "REQ-ODP-001;REQ-CONV-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  const GnssFix fix = fixFrom(epoch0, r0, v0, eop);

  // The time tag really is on the GPS scale, 19 s behind TAI.
  EXPECT_EQ(pt::toTai(fix.time_tag).nanosecondsSinceEpoch(), epoch0.nanosecondsSinceEpoch());
  EXPECT_EQ(fix.time_tag.nanosecondsSinceEpoch(), epoch0.nanosecondsSinceEpoch() - 19'000'000'000);
  // And the ECEF velocity is genuinely different from the ECI one, so the
  // round-trip below is testing the transport term rather than an identity.
  EXPECT_GT((fix.velocity_m_s.eigen() - v0).norm(), 100.0);

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fix, eop, out));
  ASSERT_TRUE(out.seeded);
  EXPECT_EQ(filter.epoch().nanosecondsSinceEpoch(), epoch0.nanosecondsSinceEpoch());
  EXPECT_LT((filter.position().eigen() - r0).norm(), 1.0e-6);
  EXPECT_LT((filter.velocity().eigen() - v0).norm(), 1.0e-9);
}

/// The seeded covariance is the receiver's own anisotropic accuracy rotated into
/// ECI, not a scalar. Checked by its invariants — the trace is the sum of the
/// three reported variances whatever frame it is expressed in, and the largest
/// eigenvalue is the vertical one — because the ECI orientation of the ellipsoid
/// depends on the epoch and asserting on its components would be asserting on
/// the reduction rather than on this filter.
TEST(OrbitOdIngest, SeedCovarianceCarriesTheReceiversAnisotropy) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));

  const Eigen::Matrix3d p_pos =
      filter.covariance().block<3, 3>(OrbitOd::kPosition, OrbitOd::kPosition);
  EXPECT_NEAR(p_pos.trace(), 2.0 * kSigmaHM * kSigmaHM + kSigmaVM * kSigmaVM, 1.0e-9);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(p_pos);
  ASSERT_EQ(es.info(), Eigen::Success);
  EXPECT_NEAR(es.eigenvalues()(2), kSigmaVM * kSigmaVM, 1.0e-9);
  EXPECT_NEAR(es.eigenvalues()(0), kSigmaHM * kSigmaHM, 1.0e-9);
  EXPECT_GT(es.eigenvalues()(2) / es.eigenvalues()(0), 2.0)
      << "an isotropic covariance would mean the receiver's vertical/horizontal "
         "split was thrown away";

  // Velocity is per-axis white in ECEF, and a rotation leaves σ²I alone.
  const Eigen::Matrix3d p_vel =
      filter.covariance().block<3, 3>(OrbitOd::kVelocity, OrbitOd::kVelocity);
  EXPECT_TRUE(p_vel.isApprox(Eigen::Matrix3d::Identity() * (kSigmaVelMps * kSigmaVelMps), 1e-12));
}

TEST(OrbitOdState, WritesTheOrbitBlocksOnlyAndOnlyWhenValid) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  // An attitude estimator got there first: its blocks and flags must survive.
  ps::EstimatedState state;
  state.valid.attitude = true;
  state.valid.gyro_bias = true;
  state.valid.covariance = true;
  state.mode = ps::EstimationMode::Fine;
  state.covariance.block<3, 3>(ps::ErrorState::kAttitude, ps::ErrorState::kAttitude) =
      Eigen::Matrix3d::Identity() * 1.5;

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  pg::writeToEstimatedState(filter, epoch0, state);

  EXPECT_TRUE(state.valid.position);
  EXPECT_TRUE(state.valid.velocity);
  EXPECT_LT((state.position.eigen() - r0).norm(), 1.0e-6);
  const Eigen::Matrix3d state_pos_block =
      state.covariance.block<3, 3>(ps::ErrorState::kPosition, ps::ErrorState::kPosition);
  const Eigen::Matrix3d filter_pos_block =
      filter.covariance().block<3, 3>(OrbitOd::kPosition, OrbitOd::kPosition);
  EXPECT_TRUE(state_pos_block.isApprox(filter_pos_block));
  // The attitude filter's work is untouched — including the mode, which names
  // the *attitude* mode and is none of this estimator's business.
  EXPECT_TRUE(state.valid.attitude);
  EXPECT_EQ(state.mode, ps::EstimationMode::Fine);
  EXPECT_NEAR(state.covariance(ps::ErrorState::kAttitude, ps::ErrorState::kAttitude), 1.5, 1e-12);

  // An invalid solution clears the validity flags but must not stamp a
  // meaningless covariance over the last good one.
  const ps::Covariance cov_before = state.covariance;
  filter.reset();
  pg::writeToEstimatedState(filter, advance(epoch0, 1.0), state);
  EXPECT_FALSE(state.valid.position);
  EXPECT_FALSE(state.valid.velocity);
  EXPECT_EQ(state.covariance, cov_before);
}

// ===========================================================================
// 4. Consistency: the truncation that sizes Q, and NEES/NIS
// ===========================================================================

namespace {

/// The truth-sim configuration the coarse onboard model is characterised
/// against: the full §5.2 stack — 8×8 EGM2008, the whole piecewise-exponential
/// atmosphere, Sun and Moon third bodies, SRP.
scenario::SimConfig truthConfig(const pt::Tai& epoch, const Eigen::Vector3d& r0,
                                const Eigen::Vector3d& v0, double duration_s) {
  scenario::SimConfig config;
  config.scenario_name = "orbit-od-truncation";
  config.spacecraft.name = "od-reference";
  config.spacecraft.mass_kg = 12.0;
  config.spacecraft.inertia_kgm2 = Eigen::Matrix3d::Identity() * 0.2;
  config.spacecraft.drag_area_m2 = 0.06;
  config.spacecraft.drag_cd = 2.2;
  config.spacecraft.srp_area_m2 = 0.06;
  config.spacecraft.srp_cr = 1.3;

  // Truth gravity must sit **above** the onboard field, or this measures
  // nothing. It used to be 8x8, which was comfortably above a closed-form J2
  // onboard model; now that the filter itself flies 8x8, an 8x8 truth would
  // reduce this test to a comparison of two implementations of the same field
  // and report a truncation of approximately zero — passing, and meaningless.
  // 32x32 is the next real step: at 400 km the degree-33+ residual is ~1e-9
  // m/s², two orders below the degree-9..32 band this now measures, so the
  // number is a fair stand-in for "everything the onboard model drops".
  config.environment.gravity_degree = 32;
  config.environment.gravity_order = 32;
  config.environment.drag_enabled = true;
  config.environment.srp_enabled = true;
  config.environment.sun_third_body = true;
  config.environment.moon_third_body = true;
  config.environment.magnetic_field = scenario::MagneticModel::kNone;

  config.initial_state.epoch = epoch;
  config.initial_state.position = pm::Vec3<pmf::ECI>(r0);
  config.initial_state.velocity = pm::Vec3<pmf::ECI>(v0);
  config.initial_state.attitude = pm::Quat<pmf::Body, pmf::ECI>::Identity();
  config.initial_state.body_rate = pm::Vec3<pmf::Body>(Eigen::Vector3d::Zero());

  config.propagation.abs_tol = 1.0e-12;
  config.propagation.rel_tol = 1.0e-12;
  config.propagation.max_step_s = 60.0;
  config.propagation.duration_s = duration_s;
  config.propagation.output_step_s = duration_s;
  return config;
}

}  // namespace

/// **The measurement that sizes the process noise.**
///
/// GMAT validates the truth sim; the truth sim, being densely samplable, then
/// supplies the reference for the divergence of the coarse onboard model over
/// the coast horizon. That divergence is the design input for `q_a`, and this
/// test asserts it stays under the value `q_a` was derived from — so a
/// force-model regression fails CI rather than quietly invalidating the tuning.
///
/// It is deliberately an upper bound and not a two-sided band: the number is a
/// characterisation of a modelling choice, not a requirement, and pinning it
/// tightly would make every legitimate force-model improvement a test failure.
TEST(OrbitOdConsistency, CoarseModelTruncationOverTheCoastHorizon) {
  RecordProperty("verifies", "REQ-ODP-001");
  RecordProperty("verifies", "REQ-ODP-005");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();

  // The truth sim's own Earth orientation, so the two propagators reference the
  // zonal field and the co-rotating atmosphere to the same pole. A synthetic EOP
  // table here would show up as a spurious ~0.36°-of-pole truncation.
  std::vector<polaris::sim::world::FinalsRow> rows;
  std::string error;
  ASSERT_TRUE(polaris::sim::world::parseFinals(
      std::string(POLARIS_GOLDEN_DIR) + "/finals.all.iau2000.txt", 1.0, 0.0, 0.0, rows, &error))
      << error;
  auto eop_table = std::make_unique<pf::EopTable<24576>>();
  for (const polaris::sim::world::FinalsRow& r : rows) {
    ASSERT_TRUE(eop_table->addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec}));
  }
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();

  scenario::SimRunner runner;
  ASSERT_TRUE(runner.build(truthConfig(epoch0, r0, v0, kCoastHorizonS),
                           scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error))
      << error;

  std::vector<double> times;
  for (double t = 30.0; t <= kCoastHorizonS + 1.0e-9; t += 30.0) {
    times.push_back(t);
  }
  std::vector<scenario::TrajectorySample> truth;
  ASSERT_TRUE(runner.runAt(times, truth, &error)) << error;
  ASSERT_EQ(truth.size(), times.size());

  // Coast one configuration against the truth samples, recording the divergence
  // at each and returning the one at the horizon. Factored out because the same
  // walk is run twice — once for the flown field and once for the J2-only model
  // it replaced — against the *same* truth arc, which is the only way the
  // improvement is a measurement rather than two numbers from different runs.
  const auto coast = [&](const OrbitOdConfig& model, const std::string& tag) -> double {
    OrbitOd filter(model);
    EXPECT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                                receiverSeedCovariance()),
              OrbitOdRefusal::kNone);
    double divergence = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
      const pt::Tai target = advance(epoch0, times[i]);
      while (filter.epoch() < target) {
        const double remaining = (target - filter.epoch()).seconds();
        const pt::Tai next =
            (remaining <= model.max_dt_s) ? target : advance(filter.epoch(), model.max_dt_s);
        EXPECT_EQ(filter.propagate(next, *eop_table, leap), OrbitOdRefusal::kNone);
      }
      divergence = (filter.position().eigen() - truth[i].state.position.eigen()).norm();
      RecordProperty(tag + "_m_at_" + std::to_string(static_cast<int>(times[i])) + "s",
                     std::to_string(divergence));
    }
    return divergence;
  };

  const double divergence_at_horizon = coast(cfg, "truncation");

  // The model this one replaced, flown over the identical arc. Recorded and
  // asserted rather than merely described: "the 8x8 field is better than J2"
  // is the entire justification for carrying a coefficient table on the flight
  // side, and a claim that load-bearing should fail CI if it stops being true.
  OrbitOdConfig j2_model = propagationOnly(referenceConfig());
  j2_model.geopotential_degree = 0;
  j2_model.geopotential_order = 0;
  j2_model.zonal_j2 = pc::gravity::kJ2;
  const double j2_divergence = coast(j2_model, "j2_only_truncation");

  RecordProperty("q_a_m2_per_s3", std::to_string(kAccelPsd));
  EXPECT_LT(divergence_at_horizon, j2_divergence)
      << "the 8x8 field (" << divergence_at_horizon << " m) does not beat the closed-form J2 model "
      << "(" << j2_divergence << " m) it replaced — the flight-side coefficient table is not "
      << "earning its keep";
  EXPECT_GT(divergence_at_horizon, 0.0) << "a coarse model that matched truth exactly would mean "
                                           "the two propagators are not actually different";
  EXPECT_LT(divergence_at_horizon, kTruncationFenceM)
      << "the coarse force model diverges from truth by more over the coast horizon than the "
         "fence above the value q_a was sized from ("
      << kTruncationAtHorizonM
      << " m) — §8.3's process noise no longer covers its own truncation and must be re-derived";
}

namespace {

/// One Monte-Carlo trial's worth of accumulated consistency statistics.
struct ConsistencyStats {
  double nees_sum{0.0};
  double nis_sum{0.0};
  std::size_t nees_count{0};
  std::size_t nis_count{0};
};

}  // namespace

/// **Two horizons (Push 70).** Past `max_coast_s` the solution stands, degraded,
/// with its grown covariance; past `max_degraded_coast_s` it is dropped and the
/// next fix re-acquires whole. Inside the degraded band a returning fix is
/// *accepted* on the grown covariance, not re-seeded — checked against a truth
/// arc from the sim's own 32×32 field over a 20 min coast: the NEES at the end
/// of the coast sits inside χ²₆ and the fix passes the NIS gate.
TEST(OrbitOdRefusals, DegradedHorizonKeepsTheSolutionAndReacquiresOnTheGrownCovariance) {
  RecordProperty("verifies", "REQ-ODP-007");
  OrbitOdConfig cfg = referenceConfig();
  cfg.max_degraded_coast_s = 1800.0;
  cfg.max_dt_s = 60.0;
  cfg.max_step_s = 10.0;
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();

  std::vector<polaris::sim::world::FinalsRow> rows;
  std::string error;
  ASSERT_TRUE(polaris::sim::world::parseFinals(
      std::string(POLARIS_GOLDEN_DIR) + "/finals.all.iau2000.txt", 1.0, 0.0, 0.0, rows, &error))
      << error;
  auto eop_table = std::make_unique<pf::EopTable<24576>>();
  for (const polaris::sim::world::FinalsRow& r : rows) {
    ASSERT_TRUE(eop_table->addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec}));
  }
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  constexpr double kCoastS = 1200.0;  // 20 min: the paper's longest outage
  scenario::SimRunner runner;
  ASSERT_TRUE(runner.build(truthConfig(epoch0, r0, v0, kCoastS),
                           scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error))
      << error;
  std::vector<double> times;
  for (double t = 60.0; t <= kCoastS + 1.0e-9; t += 60.0) {
    times.push_back(t);
  }
  times.push_back(kCoastS + 1.0);  // the returning fix's epoch, from the truth plant
  std::vector<scenario::TrajectorySample> truth;
  ASSERT_TRUE(runner.runAt(times, truth, &error)) << error;
  const std::size_t n_coast = times.size() - 1;

  OrbitOd filter(cfg);
  ASSERT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  bool saw_degraded = false;
  for (std::size_t i = 0; i < n_coast; ++i) {
    ASSERT_EQ(filter.propagate(advance(epoch0, times[i]), *eop_table, leap), OrbitOdRefusal::kNone)
        << "t = " << times[i];
    if (times[i] > cfg.max_coast_s) {
      EXPECT_EQ(filter.quality(), OrbitOdQuality::kDegraded) << "t = " << times[i];
      saw_degraded = true;
    } else {
      EXPECT_EQ(filter.quality(), OrbitOdQuality::kFine) << "t = " << times[i];
    }
    EXPECT_TRUE(filter.solutionValid());
  }
  EXPECT_TRUE(saw_degraded);

  // Consistency at the end of the coast: the grown covariance covers the truth.
  const auto& end = truth[n_coast - 1].state;
  double nees = 0.0;
  ASSERT_TRUE(filter.nees(end.position, end.velocity, nees));
  const double err_m = (filter.position().eigen() - end.position.eigen()).norm();
  RecordProperty("coast_1200s_position_error_m", std::to_string(err_m));
  RecordProperty("coast_1200s_nees", std::to_string(nees));
  EXPECT_LT(nees, 22.46) << "chi2_6(0.999): the degraded covariance does not cover the coast";
  EXPECT_LT(err_m, 100.0) << "an 8x8 model coasts 20 min to tens of metres, not " << err_m;

  // The returning fix is accepted on that covariance — no re-seed.
  pf::EopValue eop_ret;
  ASSERT_TRUE(eop_table->lookup(advance(epoch0, kCoastS + 1.0), leap, eop_ret));
  OrbitOdResult out;
  const pt::Tai t_ret = advance(epoch0, kCoastS + 1.0);
  ASSERT_EQ(filter.propagate(t_ret, eop_ret), OrbitOdRefusal::kNone);
  // Truth at t_ret from the plant itself (a kinematic extrapolation over one
  // second is 4 m and 9 m/s off — enough to trip the velocity gate).
  const auto& ret = truth.back().state;
  ASSERT_TRUE(filter.ingest(fixFrom(t_ret, ret.position.eigen(), ret.velocity.eigen(), eop_ret),
                            eop_ret, out));
  EXPECT_FALSE(out.seeded) << "a degraded solution re-acquires by update, not by seed";
  EXPECT_EQ(filter.quality(), OrbitOdQuality::kFine);
  EXPECT_EQ(filter.rejectedCount(), 0u);

  // And past the degraded horizon: dropped, and the next fix re-seeds.
  OrbitOd second(cfg);
  ASSERT_EQ(second.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                              receiverSeedCovariance()),
            OrbitOdRefusal::kNone);
  const pf::EopValue eop = eopAt(epoch0);
  ASSERT_TRUE(walkTo(second, epoch0, cfg.max_degraded_coast_s, cfg.max_dt_s, eop));
  EXPECT_EQ(second.quality(), OrbitOdQuality::kDegraded);
  EXPECT_EQ(second.propagate(advance(epoch0, cfg.max_degraded_coast_s + cfg.max_dt_s), eop),
            OrbitOdRefusal::kCoastExpired);
  EXPECT_EQ(second.quality(), OrbitOdQuality::kNone);
  EXPECT_FALSE(second.solutionValid());
}

/// **NEES/NIS consistency against a truth the filter's own model describes.**
///
/// Truth is propagated with the same force model and perturbed by a white
/// acceleration drawn from exactly the discretised `Q_d(h)` the filter adds, and
/// measured with Gaussian GNSS noise at exactly the σ the filter is told. Under
/// those conditions the filter is optimal by construction, so the sample NEES
/// must sit at the state dimension (6) and the NIS at the measurement dimension
/// (3) — and if it does not, the defect is in the algebra: the Joseph update,
/// the `Q` cross-terms, the anisotropic `R` rotation, or `Φ`.
///
/// This is the half that says **the covariance means what it claims**. An
/// accuracy test — "the position error is small" — passes just as happily with a
/// covariance two times too small, and that failure mode does not stay quiet: it
/// surfaces later as the innovation gate rejecting good measurements, which is
/// the exact FDIR pathology the review catalogue records for the attitude side.
///
/// Truth-versus-*sim* is the other half, and it is a different question with a
/// different answer — the sim's extra forces are a systematic the white `Q` can
/// only cover in the aggregate. That one is `CoarseModelTruncationOverTheCoastHorizon`
/// above, which measures the systematic directly rather than laundering it
/// through a NEES that would report "inconsistent" without saying why.
TEST(OrbitOdConsistency, MonteCarloNeesAndNisSitAtTheirStateAndMeasurementDimensions) {
  RecordProperty("verifies", "REQ-ODP-001");
  const OrbitOdConfig cfg = referenceConfig();
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  constexpr int kRuns = 120;
  constexpr double kStepS = 1.0;     // one fix per second, and one filter sub-step
  constexpr int kStepsPerRun = 200;  // 200 s, comfortably inside the coast horizon
  constexpr int kBurnIn = 60;        // samples discarded while the seed transient decays

  // Q_d(h) for the truth perturbation — the same expression the filter uses, so
  // the process noise the truth actually experiences is the process noise the
  // filter budgets for. Its Cholesky factor turns a standard normal 6-vector
  // into a correctly-correlated increment; the position/velocity cross-terms
  // are what make that increment a *coherent* acceleration rather than two
  // independent jolts, which is the thing a diagonal shortcut would get wrong.
  OrbitOd::Covariance q_d = OrbitOd::Covariance::Zero();
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
  q_d.block<3, 3>(0, 0) = (kAccelPsd * kStepS * kStepS * kStepS / 3.0) * id;
  q_d.block<3, 3>(0, 3) = (kAccelPsd * kStepS * kStepS / 2.0) * id;
  q_d.block<3, 3>(3, 0) = q_d.block<3, 3>(0, 3);
  q_d.block<3, 3>(3, 3) = (kAccelPsd * kStepS) * id;
  const Eigen::LLT<OrbitOd::Covariance> q_llt(q_d);
  ASSERT_EQ(q_llt.info(), Eigen::Success);
  const OrbitOd::Covariance q_chol = q_llt.matrixL();

  Eigen::Vector3d r_nom;
  Eigen::Vector3d v_nom;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r_nom, v_nom);

  ConsistencyStats stats;
  std::size_t rejected_total = 0;

  for (int run = 0; run < kRuns; ++run) {
    // Per-run substream: adding a run must not perturb the others, and the whole
    // campaign is reproducible from this one seed (§3.5).
    auto rng = polaris::random::streamRng(0xC0FFEEULL, static_cast<std::uint64_t>(run));
    auto draw = [&rng](void) { return rng.gaussian(); };

    Eigen::Vector3d r_true = r_nom;
    Eigen::Vector3d v_true = v_nom;

    OrbitOd filter(cfg);
    OrbitOdResult out;
    // Seed from a noisy first fix — the flight path, and it puts the initial
    // error where the covariance says it should be rather than at zero, which is
    // what makes the early NEES meaningful instead of trivially small.
    {
      GnssFix fix = fixFrom(epoch0, r_true, v_true, eop);
      fix.position_m.eigen() += kSigmaHM * Eigen::Vector3d(draw(), draw(), draw());
      fix.velocity_m_s.eigen() += kSigmaVelMps * Eigen::Vector3d(draw(), draw(), draw());
      ASSERT_TRUE(filter.ingest(fix, eop, out)) << "run " << run;
      ASSERT_TRUE(out.seeded);
    }

    for (int step = 1; step <= kStepsPerRun; ++step) {
      const pt::Tai t = advance(epoch0, static_cast<double>(step) * kStepS);

      // --- Truth: RK4 on the same force model, then the process-noise kick ---
      pg::EarthOrientation earth;
      ASSERT_TRUE(pg::earthOrientationAt(t, eop, earth));
      auto deriv = [&](const Eigen::Vector3d& r, const Eigen::Vector3d& v, Eigen::Vector3d& dr,
                       Eigen::Vector3d& dv) {
        dr = v;
        dv = pg::onboardAcceleration(cfg, pm::Vec3<pmf::ECI>(r), pm::Vec3<pmf::ECI>(v), earth)
                 .eigen();
      };
      Eigen::Vector3d k1r;
      Eigen::Vector3d k1v;
      Eigen::Vector3d k2r;
      Eigen::Vector3d k2v;
      Eigen::Vector3d k3r;
      Eigen::Vector3d k3v;
      Eigen::Vector3d k4r;
      Eigen::Vector3d k4v;
      deriv(r_true, v_true, k1r, k1v);
      deriv(r_true + 0.5 * kStepS * k1r, v_true + 0.5 * kStepS * k1v, k2r, k2v);
      deriv(r_true + 0.5 * kStepS * k2r, v_true + 0.5 * kStepS * k2v, k3r, k3v);
      deriv(r_true + kStepS * k3r, v_true + kStepS * k3v, k4r, k4v);
      r_true += (kStepS / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
      v_true += (kStepS / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

      Eigen::Matrix<double, 6, 1> w;
      for (int i = 0; i < 6; ++i) {
        w(i) = draw();
      }
      const Eigen::Matrix<double, 6, 1> kick = q_chol * w;
      r_true += kick.head<3>();
      v_true += kick.tail<3>();

      // --- Measurement -----------------------------------------------------
      GnssFix fix = fixFrom(t, r_true, v_true, eop);
      // Position noise is realised in the local geodetic frame, exactly as
      // `sim/sensors/gnss` draws it, so the filter's rotated anisotropic R is
      // being scored against the anisotropy it was actually given. Drawing it
      // isotropically here would make the R rotation untestable — the very thing
      // most likely to be wrong.
      {
        const Eigen::Vector3d up = fix.position_m.eigen().normalized();
        Eigen::Vector3d east = Eigen::Vector3d::UnitZ().cross(up).normalized();
        const Eigen::Vector3d north = up.cross(east);
        fix.position_m.eigen() +=
            kSigmaHM * draw() * east + kSigmaHM * draw() * north + kSigmaVM * draw() * up;
      }
      fix.velocity_m_s.eigen() += kSigmaVelMps * Eigen::Vector3d(draw(), draw(), draw());

      const bool ok = filter.ingest(fix, eop, out);
      if (!ok) {
        rejected_total += 1;
        continue;
      }

      if (step > kBurnIn) {
        double nees = 0.0;
        ASSERT_TRUE(filter.nees(pm::Vec3<pmf::ECI>(r_true), pm::Vec3<pmf::ECI>(v_true), nees));
        stats.nees_sum += nees;
        stats.nees_count += 1;
        stats.nis_sum += out.position.nis;
        stats.nis_count += 1;
      }
    }
  }

  ASSERT_GT(stats.nees_count, 0u);
  const double mean_nees = stats.nees_sum / static_cast<double>(stats.nees_count);
  const double mean_nis = stats.nis_sum / static_cast<double>(stats.nis_count);
  RecordProperty("mean_nees", std::to_string(mean_nees));
  RecordProperty("mean_nis", std::to_string(mean_nis));
  RecordProperty("nees_samples", std::to_string(stats.nees_count));
  RecordProperty("gate_rejections", std::to_string(rejected_total));

  // Bands are ±10% of the theoretical values (6 and 3), not fitted to what the
  // filter happens to produce. A strict χ² interval would be much tighter, and
  // would also be wrong: samples within a run are serially correlated, so the
  // effective sample count is well below `nees_count` and the nominal interval
  // does not apply. 10% is loose enough to be honest about that and tight enough
  // that the failure modes it exists for — an overconfident covariance, a
  // dropped Q cross-term, an un-rotated R — move it clean out of band.
  EXPECT_NEAR(mean_nees, 6.0, 0.6) << "sample NEES " << mean_nees
                                   << " against a 6-state filter: above means the covariance is "
                                      "optimistic, below means it is conservative";
  EXPECT_NEAR(mean_nis, 3.0, 0.3) << "sample position NIS " << mean_nis
                                  << " against a 3-row update";

  // A consistent filter at a 99.9% gate should reject about one measurement in a
  // thousand. Asserting the gate is *quiet* is the other half of asserting it
  // fires: a gate rejecting good data is the pathology the review catalogue
  // records, and it would otherwise hide behind a passing NEES.
  const double reject_fraction =
      static_cast<double>(rejected_total) / static_cast<double>(kRuns * kStepsPerRun);
  EXPECT_LT(reject_fraction, 0.01) << "the NIS gate is rejecting consistent measurements";
}

// ===========================================================================
// 5. Fix latency (§6.2, §8.3)
// ===========================================================================

/// The correction, measured against the error it removes.
///
/// A fix tagged one latency behind the filter describes where the vehicle was,
/// not where it is. Applied unchanged it drags the state backwards by ~v·τ; this
/// asserts the corrected path lands on truth-at-the-filter's-epoch instead, and
/// — the half that matters — asserts the *uncorrected* error is large, so a
/// regression that quietly dropped the correction could not pass by being
/// "close enough".
TEST(OrbitOdLatency, LatentFixIsAdvancedToTheFilterEpochRatherThanApplied) {
  RecordProperty("verifies", "REQ-ODP-006");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  constexpr double kLatencyS = 0.05;
  constexpr double kCycleS = 1.0;

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  ASSERT_TRUE(out.seeded);

  // The GNC cycle runs the filter forward to `t_now`. The receiver's next
  // solution was measured at `t_now - latency` and arrives now.
  const pt::Tai t_now = advance(epoch0, kCycleS);
  ASSERT_EQ(filter.propagate(t_now, eop), OrbitOdRefusal::kNone);

  const double t_measured = kCycleS - kLatencyS;
  Eigen::Vector3d r_measured;
  Eigen::Vector3d v_measured;
  truthAt(cfg, epoch0, r0, v0, t_measured, eop, r_measured, v_measured);
  Eigen::Vector3d r_now;
  Eigen::Vector3d v_now;
  truthAt(cfg, epoch0, r0, v0, kCycleS, eop, r_now, v_now);

  // The error the correction exists to remove, stated before it is removed.
  const double uncorrected_error = (r_measured - r_now).norm();
  EXPECT_GT(uncorrected_error, 300.0)
      << "a 50 ms latency should be hundreds of metres of along-track position; if it is not, "
         "this test is not exercising the thing it is named for";

  ASSERT_TRUE(
      filter.ingest(fixFrom(advance(epoch0, t_measured), r_measured, v_measured, eop), eop, out));
  EXPECT_TRUE(out.position.accepted);
  EXPECT_NEAR(out.fix_latency_s, kLatencyS, 1.0e-6);

  // The measurement, advanced, must agree with truth at the filter's epoch to
  // far better than the fix's own noise — the residual is O(tau^3 * jerk), not
  // O(tau^2). Compared against the *measurement* the filter now holds, reached
  // through the innovation: a converged filter sits between prior and
  // measurement, so asserting on the state would be asserting on the gain.
  EXPECT_LT(out.position.innovation.norm(), 1.0e-3)
      << "the advanced measurement disagrees with the filter's own propagated state by more than "
         "a millimetre — the advance is not tracking the propagation";

  // And the state itself must have stayed on truth, not been dragged back
  // toward where the vehicle was one latency ago.
  EXPECT_LT((filter.position().eigen() - r_now).norm(), 0.1 * uncorrected_error);
}

/// The bound is real: a fix from far enough in the past is a clock fault, not a
/// latency, and is refused. Checked either side of the boundary so the test
/// pins the threshold rather than merely observing that some refusal happens.
TEST(OrbitOdLatency, FixOlderThanTheBoundIsRefusedAndTheSolutionUntouched) {
  RecordProperty("verifies", "REQ-ODP-006");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));

  const double kCycleS = 2.0;
  ASSERT_EQ(filter.propagate(advance(epoch0, kCycleS), eop), OrbitOdRefusal::kNone);

  // Just inside the bound: accepted.
  const double inside = kCycleS - (kMaxFixLatencyS - 0.01);
  Eigen::Vector3d r_in;
  Eigen::Vector3d v_in;
  truthAt(cfg, epoch0, r0, v0, inside, eop, r_in, v_in);
  EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, inside), r_in, v_in, eop), eop, out));
  EXPECT_TRUE(out.position.accepted);

  // Just outside it: refused, and nothing moves. The filter's epoch is still
  // t = kCycleS (an accepted latent fix does not rewind it), so a fix at
  // kCycleS - kMaxFixLatencyS - 0.01 is over the bound.
  const Eigen::Vector3d before = filter.position().eigen();
  const OrbitOd::Covariance p_before = filter.covariance();
  const double outside = kCycleS - (kMaxFixLatencyS + 0.01);
  Eigen::Vector3d r_out;
  Eigen::Vector3d v_out;
  truthAt(cfg, epoch0, r0, v0, outside, eop, r_out, v_out);
  EXPECT_FALSE(filter.ingest(fixFrom(advance(epoch0, outside), r_out, v_out, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNonMonotonicEpoch);
  EXPECT_EQ(filter.position().eigen(), before);
  EXPECT_EQ(filter.covariance(), p_before);
}

/// A latent fix without a velocity cannot be advanced, and the filter's own
/// velocity must not be borrowed to do it — that would fold the filter's error
/// into a measurement required to be independent of it, which is how a
/// consistent filter is made overconfident.
TEST(OrbitOdLatency, LatentFixWithoutVelocityIsRefusedRatherThanAdvancedOnThePrior) {
  RecordProperty("verifies", "REQ-ODP-006");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);

  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  ASSERT_EQ(filter.propagate(advance(epoch0, 1.0), eop), OrbitOdRefusal::kNone);

  Eigen::Vector3d r_m;
  Eigen::Vector3d v_m;
  truthAt(cfg, epoch0, r0, v0, 0.95, eop, r_m, v_m);
  GnssFix fix = fixFrom(advance(epoch0, 0.95), r_m, v_m, eop);
  fix.velocity_valid = false;

  const OrbitOd::Covariance p_before = filter.covariance();
  EXPECT_FALSE(filter.ingest(fix, eop, out));
  // The missing-velocity name, not the clock-fault one: FDIR keyed on
  // kNonMonotonicEpoch must mean time tags went wrong, nothing else.
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNoVelocityForSeed);
  EXPECT_EQ(filter.covariance(), p_before);

  // The same fix at the filter's own epoch needs no advance and is accepted
  // position-only, so the refusal above is about the *advance*, not about
  // velocity-less fixes in general.
  Eigen::Vector3d r_f;
  Eigen::Vector3d v_f;
  truthAt(cfg, epoch0, r0, v0, 2.0, eop, r_f, v_f);
  GnssFix forward = fixFrom(advance(epoch0, 2.0), r_f, v_f, eop);
  forward.velocity_valid = false;
  EXPECT_TRUE(filter.ingest(forward, eop, out));
  EXPECT_TRUE(out.position.accepted);
}

/// Zero latency restores the strict behaviour: any fix behind the filter is
/// refused. The feature must be opt-in per config, not a relaxation every
/// existing scenario silently inherits.
TEST(OrbitOdLatency, ZeroBoundRefusesAnyFixBehindTheFilter) {
  RecordProperty("verifies", "REQ-ODP-006");
  OrbitOdConfig cfg = referenceConfig();
  cfg.max_fix_latency_s = 0.0;
  ASSERT_TRUE(cfg.isValid());

  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  OrbitOd filter(cfg);
  OrbitOdResult out;
  ASSERT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  ASSERT_EQ(filter.propagate(advance(epoch0, 1.0), eop), OrbitOdRefusal::kNone);

  Eigen::Vector3d r_m;
  Eigen::Vector3d v_m;
  truthAt(cfg, epoch0, r0, v0, 0.99, eop, r_m, v_m);
  EXPECT_FALSE(filter.ingest(fixFrom(advance(epoch0, 0.99), r_m, v_m, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kNonMonotonicEpoch);
}

// ===========================================================================
// NESC navigation-filter usability practices (NASA/TP-2018-219822 Ch. 7, §9;
// NESC TB 20-03 items d, f, g) and the leap-second exposure (TP §6.3) — Push 71
// ===========================================================================

namespace {

/// A filter that has seeded and taken @p n clean fixes at 1 Hz. Returns the truth
/// state at the last fix through @p r_t / @p v_t.
OrbitOd trackingFilter(const OrbitOdConfig& cfg, const pt::Tai& epoch0, const Eigen::Vector3d& r0,
                       const Eigen::Vector3d& v0, const pf::EopValue& eop, int n,
                       Eigen::Vector3d& r_t, Eigen::Vector3d& v_t) {
  OrbitOd filter(cfg);
  OrbitOdResult out;
  EXPECT_TRUE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out));
  for (int i = 1; i <= n; ++i) {
    truthAt(cfg, epoch0, r0, v0, static_cast<double>(i), eop, r_t, v_t);
    EXPECT_TRUE(
        filter.ingest(fixFrom(advance(epoch0, static_cast<double>(i)), r_t, v_t, eop), eop, out))
        << "clean fix " << i;
  }
  return filter;
}

}  // namespace

/// TB 20-03 item (g) / TP §9.3: a tuning change under a running solution keeps
/// the solution — state, covariance, epoch, age, counters — and a bad upload
/// leaves the configuration in force untouched.
TEST(OrbitOdUsability, RetuneKeepsTheSolutionAndRefusesABadConfig) {
  RecordProperty("verifies", "REQ-ODP-009");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  Eigen::Vector3d r_t;
  Eigen::Vector3d v_t;
  OrbitOd filter = trackingFilter(cfg, epoch0, r0, v0, eop, 5, r_t, v_t);
  const Eigen::Vector3d r_before = filter.position().eigen();
  const OrbitOd::Covariance p_before = filter.covariance();
  const double age_before = filter.ageSeconds();
  const pt::Tai epoch_before = filter.epoch();

  OrbitOdConfig tighter = cfg;
  tighter.position_nis_gate = 7.81;  // chi-square(3) at 95%: a real change
  tighter.max_coast_s = 120.0;
  ASSERT_EQ(filter.retune(tighter), OrbitOdRefusal::kNone);
  EXPECT_TRUE(filter.isInitialised());
  EXPECT_TRUE(filter.position().eigen() == r_before);
  EXPECT_TRUE(filter.covariance() == p_before);
  EXPECT_EQ(filter.ageSeconds(), age_before);
  EXPECT_TRUE(filter.epoch() == epoch_before);

  // The new tuning governs: a 20 m offset (NIS ~ 100 against a ~2 m sigma... but
  // well under 16.27 would not be) — use an offset that the 95% gate refuses.
  truthAt(cfg, epoch0, r0, v0, 6.0, eop, r_t, v_t);
  OrbitOdResult out;
  const Eigen::Vector3d off = r_t + Eigen::Vector3d(20.0, 0.0, 0.0);
  EXPECT_FALSE(filter.ingest(fixFrom(advance(epoch0, 6.0), off, v_t, eop), eop, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementRejected);
  EXPECT_EQ(filter.rejectedCount(), 1u);

  OrbitOdConfig bad = tighter;
  bad.accel_psd_m2_per_s3 = -1.0;
  EXPECT_EQ(filter.retune(bad), OrbitOdRefusal::kUnconfigured);
  EXPECT_TRUE(filter.isConfigured());
  EXPECT_TRUE(filter.isInitialised());
  truthAt(cfg, epoch0, r0, v0, 7.0, eop, r_t, v_t);
  EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, 7.0), r_t, v_t, eop), eop, out))
      << "still running on the last valid tuning";
}

/// TB 20-03 item (f) / TP §9.2: the covariance is re-opened, the state kept,
/// and a measurement the settled covariance would have gated is then taken —
/// which is the remedy for an over-confident filter editing good fixes.
TEST(OrbitOdUsability, CovarianceReinitialisationKeepsTheState) {
  RecordProperty("verifies", "REQ-ODP-009");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);
  Eigen::Vector3d r_t;
  Eigen::Vector3d v_t;
  OrbitOd filter = trackingFilter(cfg, epoch0, r0, v0, eop, 5, r_t, v_t);
  const Eigen::Vector3d r_before = filter.position().eigen();
  const Eigen::Vector3d v_before = filter.velocity().eigen();

  EXPECT_EQ(filter.reinitializeCovariance(0.0, 1.0), OrbitOdRefusal::kFixSigmaInvalid);
  EXPECT_EQ(filter.reinitializeCovariance(10.0, -1.0), OrbitOdRefusal::kFixSigmaInvalid);
  ASSERT_EQ(filter.reinitializeCovariance(100.0, 1.0), OrbitOdRefusal::kNone);
  EXPECT_TRUE(filter.position().eigen() == r_before);
  EXPECT_TRUE(filter.velocity().eigen() == v_before);
  const OrbitOd::Covariance p = filter.covariance();
  EXPECT_DOUBLE_EQ(p(0, 0), 1.0e4);
  EXPECT_DOUBLE_EQ(p(3, 3), 1.0);
  EXPECT_EQ(p(0, 3), 0.0);
  EXPECT_TRUE(filter.covarianceHealthy());

  // A 20 m offset that the settled P rejected above is now inside the gate.
  truthAt(cfg, epoch0, r0, v0, 6.0, eop, r_t, v_t);
  OrbitOdResult out;
  const Eigen::Vector3d off = r_t + Eigen::Vector3d(20.0, 0.0, 0.0);
  EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, 6.0), off, v_t, eop), eop, out));
  EXPECT_TRUE(out.position.accepted);
  EXPECT_FALSE(out.position.forced);
  EXPECT_GT((filter.position().eigen() - r_t).norm(), 15.0)
      << "taken almost whole against a 100 m P";

  OrbitOd cold(cfg);
  EXPECT_EQ(cold.reinitializeCovariance(10.0, 1.0), OrbitOdRefusal::kUninitialised);
  EXPECT_TRUE(cold.covarianceHealthy());
  OrbitOd inert(OrbitOdConfig{});
  EXPECT_EQ(inert.reinitializeCovariance(10.0, 1.0), OrbitOdRefusal::kUnconfigured);
  EXPECT_EQ(inert.retune(cfg), OrbitOdRefusal::kNone) << "retune can bring an inert filter up";
  EXPECT_TRUE(inert.isConfigured());
}

/// TB 20-03 item (d) / TP §9.1: the three-way editing flag per measurement type.
TEST(OrbitOdUsability, MeasurementPolicyInhibitsAndForces) {
  RecordProperty("verifies", "REQ-ODP-009");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = eopAt(epoch0);

  // Inhibited position: nothing seeds, nothing updates.
  {
    OrbitOd filter(cfg);
    OrbitOdResult out;
    GnssMeasurementPolicy policy;
    policy.position = MeasurementMode::kInhibit;
    EXPECT_FALSE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out, policy));
    EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementInhibited);
    EXPECT_FALSE(filter.isInitialised());
    // Inhibited velocity on a cold filter: a seed needs both halves.
    policy.position = MeasurementMode::kAccept;
    policy.velocity = MeasurementMode::kInhibit;
    EXPECT_FALSE(filter.ingest(fixFrom(epoch0, r0, v0, eop), eop, out, policy));
    EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementInhibited);
    EXPECT_FALSE(filter.isInitialised());
  }

  Eigen::Vector3d r_t;
  Eigen::Vector3d v_t;
  OrbitOd filter = trackingFilter(cfg, epoch0, r0, v0, eop, 5, r_t, v_t);

  // Inhibited velocity on a running filter: the position half is processed,
  // the velocity half is not touched at all.
  {
    GnssMeasurementPolicy policy;
    policy.velocity = MeasurementMode::kInhibit;
    truthAt(cfg, epoch0, r0, v0, 6.0, eop, r_t, v_t);
    OrbitOdResult out;
    const Eigen::Vector3d bad_v = v_t + Eigen::Vector3d(50.0, 0.0, 0.0);
    EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, 6.0), r_t, bad_v, eop), eop, out, policy));
    EXPECT_TRUE(out.position.accepted);
    EXPECT_FALSE(out.velocity.accepted);
    EXPECT_EQ(out.velocity.nis, 0.0) << "the velocity update never ran";
    EXPECT_EQ(filter.rejectedCount(), 0u);
    EXPECT_LT((filter.velocity().eigen() - v_t).norm(), 0.5) << "the 50 m/s never entered";
  }

  // Inhibited position on a running filter: the whole fix is withheld.
  {
    GnssMeasurementPolicy policy;
    policy.position = MeasurementMode::kInhibit;
    const Eigen::Vector3d r_before = filter.position().eigen();
    truthAt(cfg, epoch0, r0, v0, 7.0, eop, r_t, v_t);
    OrbitOdResult out;
    EXPECT_FALSE(filter.ingest(fixFrom(advance(epoch0, 7.0), r_t, v_t, eop), eop, out, policy));
    EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementInhibited);
    EXPECT_TRUE(filter.isInitialised());
    EXPECT_TRUE(filter.position().eigen() == r_before) << "not even propagated to the fix";
  }

  // Force: a 1 km spoof the gate would refuse is applied, and reported as forced.
  {
    GnssMeasurementPolicy policy;
    policy.position = MeasurementMode::kForce;
    truthAt(cfg, epoch0, r0, v0, 8.0, eop, r_t, v_t);
    OrbitOdResult out;
    const Eigen::Vector3d spoofed = r_t + Eigen::Vector3d(1000.0, 0.0, 0.0);
    EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, 8.0), spoofed, v_t, eop), eop, out, policy));
    EXPECT_TRUE(out.position.accepted);
    EXPECT_TRUE(out.position.forced);
    EXPECT_GT(out.position.nis, cfg.position_nis_gate);
    EXPECT_FALSE(out.velocity.forced);
    EXPECT_EQ(filter.forcedCount(), 1u);
    // The velocity half ran in accept mode against a state the forced position
    // just dragged a kilometre (through the r-v cross-covariance): whether the
    // gate takes it is its own business, and if it did not, that is a counted
    // rejection, not a forced one.
    EXPECT_EQ(filter.rejectedCount(), out.velocity.accepted ? 0u : 1u);
    // Applied at the settled gain: a fraction of the kilometre, not the ~metres a
    // rejection would have left.
    EXPECT_GT((filter.position().eigen() - r_t).norm(), 50.0) << "the forced fix moved the state";
    filter.reset();
    EXPECT_EQ(filter.forcedCount(), 0u);
  }
}

/// TP §6.3: the filter must be designed so a misapplied leap second cannot
/// affect it. This filter is TAI-clean internally, but the ECEF→ECI conversion
/// of a GNSS fix goes through UT1, and UT1 is reached from TAI through ΔAT — a
/// leap-second table stale by one leap rotates the Earth by 15 arcsec, ~500 m
/// at LEO. Two things must hold: a **running** filter refuses such fixes on the
/// NIS gate rather than absorbing a 500 m step; and a **cold** filter seeds on
/// them (there is no prior to disagree with), giving a solution that is
/// self-consistent but rotated — an exposure recorded here so it is a known
/// limit and not a surprise. `frozen(36)` is exactly "the table before the
/// 2017-01-01 leap"; the 2026 epoch needs 37.
TEST(OrbitOdTimeScales, AStaleLeapSecondTableIsRefusedByATrackingFilterAndOnlyRotatesASeed) {
  RecordProperty("verifies", "REQ-ODP-010");
  const OrbitOdConfig cfg = referenceConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  const pt::Tai epoch0 = testEpoch();
  const auto table = makeEop(epoch0);
  const pt::LeapSecondTable good = pt::LeapSecondTable::historical();
  const pt::LeapSecondTable stale = pt::LeapSecondTable::frozen(36);
  ASSERT_EQ(good.deltaAtForTaiSeconds(epoch0.nanosecondsSinceEpoch() / 1000000000LL), 37);
  const pf::EopValue eop = eopAt(epoch0);

  Eigen::Vector3d r_t;
  Eigen::Vector3d v_t;
  OrbitOd filter = trackingFilter(cfg, epoch0, r0, v0, eop, 5, r_t, v_t);
  truthAt(cfg, epoch0, r0, v0, 6.0, eop, r_t, v_t);
  const GnssFix fix = fixFrom(advance(epoch0, 6.0), r_t, v_t, eop);
  OrbitOdResult out;
  EXPECT_FALSE(filter.ingest(fix, *table, stale, out));
  EXPECT_EQ(out.refusal, OrbitOdRefusal::kMeasurementRejected);
  EXPECT_GT(out.position.nis, cfg.position_nis_gate);
  EXPECT_GT(out.position.innovation.norm(), 300.0) << "15 arcsec of Earth rotation at LEO";
  EXPECT_LT(out.position.innovation.norm(), 700.0);
  EXPECT_LT((filter.position().eigen() - r_t).norm(), 5.0) << "the rotation never entered";
  // The next fix through the right table is taken (the refused epoch itself is
  // spent — re-presenting it is the stuck-clock guard's case).
  truthAt(cfg, epoch0, r0, v0, 7.0, eop, r_t, v_t);
  EXPECT_TRUE(filter.ingest(fixFrom(advance(epoch0, 7.0), r_t, v_t, eop), *table, good, out));

  // The seed-path exposure, measured: a cold filter has no NIS gate.
  OrbitOd cold(cfg);
  EXPECT_TRUE(cold.ingest(fixFrom(epoch0, r0, v0, eop), *table, stale, out));
  EXPECT_TRUE(out.seeded);
  const double rotated_m = (cold.position().eigen() - r0).norm();
  EXPECT_GT(rotated_m, 300.0);
  EXPECT_LT(rotated_m, 700.0);
}

// The quality verdict travels to logs and campaign records as its own name, so
// the campaign never carries a copy of the enum (the same reason `refusalName`
// exists). A value added without a case here is what "unknown" is for; this
// pins the three that exist, so the omission surfaces as a failing name rather
// than as a column of "unknown" nobody reads.
TEST(OrbitOdQualityNames, EveryVerdictHasItsOwnName) {
  EXPECT_STREQ(qualityName(OrbitOdQuality::kNone), "none");
  EXPECT_STREQ(qualityName(OrbitOdQuality::kDegraded), "degraded");
  EXPECT_STREQ(qualityName(OrbitOdQuality::kFine), "fine");
}

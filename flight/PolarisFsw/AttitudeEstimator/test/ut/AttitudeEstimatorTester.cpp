// ======================================================================
// \title  AttitudeEstimatorTester.cpp
// \brief  Test harness for the AttitudeEstimator component (§8.1, §23.1)
// ======================================================================

#include "AttitudeEstimatorTester.hpp"

#include <cmath>
#include <Eigen/Geometry>
#include <limits>

#include "environment/igrf_iaga.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace flight {

namespace {

namespace pm = polaris::math;
using ECEF = pm::frames::ECEF;
using ECI = pm::frames::ECI;
using Body = pm::frames::Body;
using QuatBI = pm::Quat<Body, ECI>;

constexpr I64 kNsPerSecond = 1000000000LL;

//! A 2026 TAI epoch, inside the IGRF snapshot's validity and the same era the
//! committed reference data covers.
constexpr I64 kStartTaiNs = 1'770'000'000LL * kNsPerSecond;

//! Decimal year the onboard snapshot is taken at (2025 IAGA bracket).
constexpr double kIgrfEpochYear = 2026.1;

//! Orbit radius [m] the harness places the vehicle at: a 620 km altitude, a real
//! place where the field model is well defined. Most tests hold one point on it
//! (the geometry is all they need); the calibration tests fly it — see
//! AttitudeEstimatorTester::runTumbleCycles.
constexpr double kOrbitRadiusM = 7.0e6;

//! Tuning used by every test that expects the estimator to run. TriadGain = 1
//! snaps to the TRIAD solution, so an acquisition is exact and a wrong frame
//! cannot hide behind a partial blend.
constexpr F64 kSigmaSunWhite = 0.01;
constexpr F64 kSigmaSunSys = 0.005;
constexpr F64 kSigmaMagWhite = 0.02;
constexpr F64 kSigmaMagSys = 0.01;
constexpr F64 kGyroArw = 1.0e-4;
constexpr F64 kMinSinAngle = 0.1;
constexpr F64 kTriadGain = 1.0;
constexpr F64 kMaxCoastSec = 5.0;
constexpr F64 kMaxDtSec = 1.0;
constexpr F64 kMaxMeasAgeSec = 0.5;
constexpr F64 kMinPositionRadiusM = 6.4e6;
constexpr F64 kMaxPositionRadiusM = 5.0e7;

//! Fine-mode tuning. The horizons and streaks are far shorter than flight values
//! so a demotion path is a handful of cycles rather than minutes of them; the
//! noise terms are the reference vehicle's, since those are what the filter's
//! consistency depends on.
constexpr F64 kMekfRrw = 4.7e-7;
constexpr F64 kMekfNisGate = 13.82;
constexpr F64 kMekfMaxCoastSec = 2.0;
constexpr F64 kMekfBiasSigmaInit = 1.0e-3;
constexpr U32 kMekfRefusalStreak = 5;
constexpr U32 kMekfNisStreak = 3;
constexpr F64 kSeedMinObservability = 0.0076;

//! Magnetometer-calibration tuning. The band is wide enough for the ~30 uT field
//! at the test position; the sample count is the reference vehicle's 100 so the
//! fit is ten-times overdetermined, and the coverage and condition gates are the
//! flight values — the point of the harness is the component, and moving the
//! gates would stop it testing the ones that ship.
constexpr F64 kMagCalNominalFieldT = 30.0e-6;
constexpr F64 kMagCalMinFieldT = 5.0e-6;
constexpr F64 kMagCalMaxFieldT = 1.0e-4;
constexpr U32 kMagCalMinSamples = 100;
constexpr F64 kMagCalMinCoverage = 0.35;
constexpr F64 kMagCalMaxCondition = 1.0e6;
constexpr F64 kMagCalMinImprovement = 2.0;

//! Hard iron injected by the calibration tests [T, Body]: 1.37 uT total against
//! a ~30 uT field, the reference vehicle's MAG-GENERIC 1 uT class. Uncorrected
//! this is tens of milliradians of field-direction error, which is exactly the
//! 1.9 deg systematic §8.1 commits to removing.
const Eigen::Vector3d kInjectedHardIronT(1.0e-6, -0.5e-6, 0.8e-6);

//! Zero-EOP, matching what the stubbed getEopAt reports.
polaris::frames::EopValue stubEop() {
  return polaris::frames::EopValue{0.0, 0.0, 0.0};
}

//! ECEF -> ECI orientation at @p taiNs under the stubbed EOP.
pm::Quat<ECI, ECEF> rotationAt(I64 taiNs) {
  pm::Quat<ECI, ECEF> q;
  const bool ok = polaris::frames::eciFromEcef(polaris::time::Tai::fromNanosecondsSinceEpoch(taiNs),
                                               stubEop(), q);
  EXPECT_TRUE(ok);
  return q;
}

Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
  return out;
}

//! Some unit vector perpendicular to @p v — used to place the Sun so the two
//! reference directions are always 90 degrees apart and TRIAD is never near its
//! observability gate. (Degenerate geometry is pinned in tests/unit/triad_test.)
Eigen::Vector3d perpendicularTo(const Eigen::Vector3d& v) {
  const Eigen::Vector3d axis =
      (std::fabs(v.normalized().z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  return v.cross(axis).normalized();
}

}  // namespace

// ----------------------------------------------------------------------
// Construction and destruction
// ----------------------------------------------------------------------

AttitudeEstimatorTester ::AttitudeEstimatorTester()
    : AttitudeEstimatorGTestBase("Tester", MAX_HISTORY_SIZE), component("AttitudeEstimator") {
  this->initComponents();
  this->connectPorts();
}

AttitudeEstimatorTester ::~AttitudeEstimatorTester() {}

// ----------------------------------------------------------------------
// Stubbed query ports
// ----------------------------------------------------------------------

bool AttitudeEstimatorTester ::from_getBodyPosition_handler(FwIndexType portNum,
                                                            const OnboardBody& body, I64 taiNs,
                                                            PosEciMeters& posEciM) {
  EXPECT_EQ(body.e, OnboardBody::SUN);
  // One astronomical unit along a direction square to the modelled field, so the
  // pair geometry is unambiguous (see perpendicularTo).
  const Eigen::Vector3d dir = perpendicularTo(this->expectedMagRef(taiNs).eigen());
  const Eigen::Vector3d sun = 1.495978707e11 * dir;
  posEciM = PosEciMeters(sun.x(), sun.y(), sun.z(), this->stub_grade_);
  return true;
}

bool AttitudeEstimatorTester ::from_getEopAt_handler(FwIndexType portNum, I64 taiNs,
                                                     EopSample& sample) {
  const polaris::frames::EopValue eop = stubEop();
  sample = EopSample(eop.ut1_minus_tai, eop.xp_arcsec, eop.yp_arcsec, this->stub_grade_);
  return true;
}

void AttitudeEstimatorTester ::from_estimateOut_handler(FwIndexType portNum,
                                                        const AttitudeEstimate& estimate) {
  this->last_estimate_ = estimate;
  ++this->estimate_count_;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::setValidParameters(bool withFine) {
  this->paramSet_SigmaSunWhiteRad(kSigmaSunWhite, Fw::ParamValid::VALID);
  this->paramSet_SigmaSunSysRad(kSigmaSunSys, Fw::ParamValid::VALID);
  this->paramSet_SigmaMagWhiteRad(kSigmaMagWhite, Fw::ParamValid::VALID);
  this->paramSet_SigmaMagSysRad(kSigmaMagSys, Fw::ParamValid::VALID);
  this->paramSet_GyroArw(kGyroArw, Fw::ParamValid::VALID);
  this->paramSet_MinSinAngle(kMinSinAngle, Fw::ParamValid::VALID);
  this->paramSet_TriadGain(kTriadGain, Fw::ParamValid::VALID);
  this->paramSet_MaxCoastSec(kMaxCoastSec, Fw::ParamValid::VALID);
  this->paramSet_MaxDtSec(kMaxDtSec, Fw::ParamValid::VALID);
  this->paramSet_MaxMeasAgeSec(kMaxMeasAgeSec, Fw::ParamValid::VALID);
  this->paramSet_MinPositionRadiusM(kMinPositionRadiusM, Fw::ParamValid::VALID);
  this->paramSet_MaxPositionRadiusM(kMaxPositionRadiusM, Fw::ParamValid::VALID);
  if (withFine) {
    this->paramSet_MekfRrw(kMekfRrw, Fw::ParamValid::VALID);
    this->paramSet_MekfNisGate(kMekfNisGate, Fw::ParamValid::VALID);
    this->paramSet_MekfMaxCoastSec(kMekfMaxCoastSec, Fw::ParamValid::VALID);
    this->paramSet_MekfBiasSigmaInit(kMekfBiasSigmaInit, Fw::ParamValid::VALID);
    this->paramSet_MekfRefusalStreak(kMekfRefusalStreak, Fw::ParamValid::VALID);
    this->paramSet_MekfNisStreak(kMekfNisStreak, Fw::ParamValid::VALID);
    this->paramSet_SeedMinObservability(kSeedMinObservability, Fw::ParamValid::VALID);
  }
  // paramSet_* only stages values in the harness's table; the component caches
  // them at load, exactly as the topology does after ParameterDb is up.
  this->component.loadParameters();
}

void AttitudeEstimatorTester ::setMagCalParameters() {
  this->paramSet_MagCalNominalFieldT(kMagCalNominalFieldT, Fw::ParamValid::VALID);
  this->paramSet_MagCalMinFieldT(kMagCalMinFieldT, Fw::ParamValid::VALID);
  this->paramSet_MagCalMaxFieldT(kMagCalMaxFieldT, Fw::ParamValid::VALID);
  this->paramSet_MagCalMinSamples(kMagCalMinSamples, Fw::ParamValid::VALID);
  this->paramSet_MagCalMinCoverage(kMagCalMinCoverage, Fw::ParamValid::VALID);
  this->paramSet_MagCalMaxCondition(kMagCalMaxCondition, Fw::ParamValid::VALID);
  this->paramSet_MagCalMinImprovement(kMagCalMinImprovement, Fw::ParamValid::VALID);
  this->component.loadParameters();
}

QuatBI AttitudeEstimatorTester ::tumbleAt(int k) {
  // Two incommensurate rates about perpendicular axes. A single-axis tumble
  // traces the body field direction round a *cone*, whose coverage metric is
  // bounded well below 1 and can be zero — the fit needs directions off that
  // cone, which is precisely what the second axis buys. The rates are large
  // enough that a 100-sample window covers several revolutions of each: a slow
  // sweep clears the coverage gate on a sparse curve but leaves the normal
  // matrix badly conditioned, which the fit refuses (and rightly — those ten
  // parameters really are not separated by a curve's worth of directions).
  const double a = 0.7 * static_cast<double>(k);
  const double b = 0.43 * static_cast<double>(k);
  const polaris::math::Quaternion qa =
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), a);
  const polaris::math::Quaternion qb =
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), b);
  return QuatBI(qa * qb);
}

void AttitudeEstimatorTester ::runTumbleCycles(int count, I64& t) {
  for (int i = 0; i < count; ++i) {
    // Fly a near-polar orbit while tumbling. Orientation diversity alone is not
    // enough for the ellipsoid fit: with the vehicle parked at one point the
    // IGRF magnitude is constant, so |x|² = const makes the quadric's three
    // diagonal terms linearly dependent on its constant term and the normal
    // matrix is genuinely rank-deficient — the fit refuses on CONDITION, which
    // is the right answer to data that cannot separate those parameters. A real
    // window is flown over an orbit, where |B| runs ~25-50 uT between equator
    // and pole, and that variation is what makes the ten parameters observable.
    const double u = 0.06 * static_cast<double>(this->tumble_step_);   // latitude sweep
    const double w = 0.013 * static_cast<double>(this->tumble_step_);  // node drift
    this->position_ecef_ = kOrbitRadiusM * Eigen::Vector3d(std::cos(u) * std::cos(w),
                                                           std::cos(u) * std::sin(w), std::sin(u));
    // The gyro is fed zero rate: with TriadGain = 1 the coarse chain snaps to
    // each cycle's TRIAD, so the tumble does not have to be kinematically
    // consistent for the vector geometry under test to be.
    this->feedMeasurements(t, tumbleAt(this->tumble_step_), Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    ++this->tumble_step_;
    t += kNsPerSecond / 10;  // 10 Hz
  }
}

double AttitudeEstimatorTester ::maxPublishedErrorOverCycles(int count, I64& t) {
  double worst = 0.0;
  for (int i = 0; i < count; ++i) {
    this->runTumbleCycles(1, t);
    const double error = this->publishedErrorRad(tumbleAt(this->tumble_step_ - 1));
    if (error > worst) {
      worst = error;
    }
  }
  return worst;
}

double AttitudeEstimatorTester ::publishedErrorRad(const QuatBI& truth) const {
  const QuatF64 q = this->last_estimate_.get_qBodyEci();
  return polaris::math::Quaternion(q[0], q[1], q[2], q[3])
      .angularDistance(truth.core().canonical());
}

void AttitudeEstimatorTester ::loadIgrf(double decimalYear) {
  const double year = (decimalYear > 0.0) ? decimalYear : kIgrfEpochYear;
  ASSERT_TRUE(this->component.configureIgrf(POLARIS_IGRF_COEFFS, year));
}

pm::Vec3<ECI> AttitudeEstimatorTester ::expectedMagRef(I64 taiNs) const {
  // Loaded once for the whole binary: the fine-mode tests run hundreds of cycles
  // and re-reading the IAGA file per measurement dominated their runtime. The
  // coefficients are immutable, so one load is one load.
  static const polaris::environment::IgrfField field = [] {
    polaris::environment::IgrfCoefficients coefficients;
    EXPECT_TRUE(
        polaris::environment::loadIgrfIaga(POLARIS_IGRF_COEFFS, kIgrfEpochYear, coefficients));
    return polaris::environment::IgrfField(coefficients);
  }();

  double year = 0.0;
  EXPECT_TRUE(polaris::time::decimalYear(polaris::time::Tai::fromNanosecondsSinceEpoch(taiNs),
                                         polaris::time::LeapSecondTable::historical(), year));
  pm::Vec3<ECEF> b_ecef;
  EXPECT_TRUE(field.field(pm::Vec3<ECEF>(this->position_ecef_), year, b_ecef));
  return rotationAt(taiNs).rotate(b_ecef);
}

pm::Vec3<ECI> AttitudeEstimatorTester ::expectedSunRef(I64 taiNs) const {
  const Eigen::Vector3d dir = perpendicularTo(this->expectedMagRef(taiNs).eigen());
  const pm::Vec3<ECI> sun(1.495978707e11 * dir);
  // The component makes the reference spacecraft-centric before normalising.
  const pm::Vec3<ECI> r_eci = rotationAt(taiNs).rotate(pm::Vec3<ECEF>(this->position_ecef_));
  pm::Vec3<ECI> unit;
  EXPECT_TRUE((sun - r_eci).normalized(unit));
  return unit;
}

void AttitudeEstimatorTester ::runCycleAt(I64 taiNs) {
  const U32 seconds = static_cast<U32>(taiNs / kNsPerSecond);
  const U32 useconds = static_cast<U32>((taiNs % kNsPerSecond) / 1000);
  this->setTestTime(Fw::Time(seconds, useconds));
  this->invoke_to_run(0, 0);
}

void AttitudeEstimatorTester ::feedMeasurements(I64 taiNs, const QuatBI& q_bi,
                                                const Eigen::Vector3d& rate_body, bool sunInView) {
  const I64 tag = taiNs + this->meas_time_offset_ns_;

  ImuMeas imu;
  // The gyro reports rate + bias; the vector measurements below keep following
  // the true attitude, which is exactly the split the MEKF resolves.
  imu.set_deltaAngleRad(toVec3F64((rate_body + this->gyro_bias_) * 0.1));  // 10 Hz cycle
  imu.set_deltaVelMps(toVec3F64(Eigen::Vector3d::Zero()));
  imu.set_intervalSec(0.1);
  imu.set_timeTagNs(tag);
  imu.set_valid(true);
  this->invoke_to_imuIn(0, imu);

  SunSensorMeas sun;
  Eigen::Vector3d sun_body = q_bi.rotate(this->expectedSunRef(taiNs)).eigen();
  if (this->sun_body_error_rad_ != 0.0) {
    sun_body = Eigen::AngleAxisd(this->sun_body_error_rad_, perpendicularTo(sun_body)) * sun_body;
  }
  if (this->sun_body_degenerate_) {
    // Finite, so the component's own isFinite gate passes it through, but
    // unnormalisable — which the filter refuses rather than gate-rejects.
    sun_body.setZero();
  }
  sun.set_dirBody(toVec3F64(sun_body));
  sun.set_sigmaRad(kSigmaSunWhite);
  sun.set_timeTagNs(tag);
  sun.set_sunPresent(sunInView);
  sun.set_valid(true);
  this->invoke_to_sunSensorIn(0, sun);

  MagnetometerMeas mag;
  // The sensor model the ellipsoid fit inverts: m = S·B_body + b. With the
  // defaults (identity S, zero b) this is a perfect magnetometer, so every test
  // that predates the calibration sees exactly what it saw before.
  const Eigen::Vector3d b_body = q_bi.rotate(this->expectedMagRef(taiNs)).eigen();
  mag.set_fieldTesla(toVec3F64(this->mag_soft_iron_ * b_body + this->mag_hard_iron_t_));
  mag.set_timeTagNs(tag);
  mag.set_valid(true);
  this->invoke_to_magnetometerIn(0, mag);

  GnssMeas gnss;
  gnss.set_posEcefM(toVec3F64(this->position_override_.has_value() ? *this->position_override_
                                                                   : this->position_ecef_));
  gnss.set_velEcefMps(toVec3F64(Eigen::Vector3d::Zero()));
  // The receiver stamps GPS time; the component applies TAI = GPS + 19 s.
  gnss.set_timeTagGpsNs(polaris::time::toGps(polaris::time::Tai::fromNanosecondsSinceEpoch(tag))
                            .nanosecondsSinceEpoch());
  gnss.set_valid(this->gnss_valid_);
  this->invoke_to_gnssIn(0, gnss);
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testRefusesWithoutParameters() {
  this->loadIgrf();
  // No paramSet_* at all: ParameterDb has nothing for this component.
  const QuatBI q = QuatBI(polaris::math::Quaternion::Identity());
  this->feedMeasurements(kStartTaiNs, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_TLM_EstMode_SIZE(1);
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);
  // Nothing was published: a consumer must not see an estimate at all rather
  // than one with its flags cleared.
  ASSERT_EQ(this->estimate_count_, 0u);

  // Edge-gated: a second refused cycle does not repeat the alert.
  this->feedMeasurements(kStartTaiNs + kNsPerSecond, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
}

void AttitudeEstimatorTester ::testAcquiresFromSyntheticMeasurements() {
  this->loadIgrf();
  this->setValidParameters();

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 2.0, 3.0).normalized(), 0.7));

  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
  ASSERT_EVENTS_ConfigInvalid_SIZE(0);
  ASSERT_TLM_EstMode(0, EstimationMode::COARSE);
  ASSERT_TLM_GyroValid(0, true);
  ASSERT_TLM_SunValid(0, true);
  ASSERT_TLM_MagValid(0, true);
  ASSERT_TLM_PositionValid(0, true);
  ASSERT_TLM_TriadAccepted(0, 1);
  ASSERT_TLM_TriadRejected(0, 0);
  ASSERT_TLM_CyclesRefused(0, 0);
  ASSERT_TLM_RefGrade(0, TableGrade::PRECISE);

  // The published solution is the truth attitude: with TriadGain = 1 the fix is
  // taken whole, so any frame error in the reference assembly shows up here.
  ASSERT_EQ(this->estimate_count_, 1u);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  ASSERT_TRUE(this->last_estimate_.get_rateValid());
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::COARSE);
  const QuatF64 q = this->last_estimate_.get_qBodyEci();
  const polaris::math::Quaternion published(q[0], q[1], q[2], q[3]);
  EXPECT_LT(published.angularDistance(truth.core().canonical()), 1.0e-6);
  EXPECT_EQ(this->last_estimate_.get_epochTaiNs(), kStartTaiNs);
}

void AttitudeEstimatorTester ::testCoastsThroughEclipseAndReacquires() {
  this->loadIgrf();
  this->setValidParameters();

  const QuatBI truth(polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 0.3));

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);

  // Eclipse: the sun leaves the field of view, so no pair and no TRIAD. Coast
  // past MaxCoastSec (5 s) at 10 Hz.
  for (int i = 0; i < 70; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), false);
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_AttitudeLost_SIZE(1);  // one edge, not one per cycle
  ASSERT_FALSE(this->last_estimate_.get_attitudeValid());
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::INVALID);
  // The body rate stays published through the coast: Safe-mode rate damping
  // needs it before attitude is back.
  ASSERT_TRUE(this->last_estimate_.get_rateValid());

  // Sun returns: the next TRIAD re-acquires whole.
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(2);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  const QuatF64 q = this->last_estimate_.get_qBodyEci();
  const polaris::math::Quaternion published(q[0], q[1], q[2], q[3]);
  EXPECT_LT(published.angularDistance(truth.core().canonical()), 1.0e-6);
}

void AttitudeEstimatorTester ::testStaleMeasurementsAreExcluded() {
  this->loadIgrf();
  this->setValidParameters();

  const QuatBI truth(polaris::math::Quaternion::Identity());
  // Time-tag every measurement two seconds in the past — well past the 0.5 s
  // staleness gate, though the values themselves are perfectly consistent.
  this->meas_time_offset_ns_ = -2 * kNsPerSecond;
  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_TLM_GyroValid(0, false);
  ASSERT_TLM_SunValid(0, false);
  ASSERT_TLM_MagValid(0, false);
  ASSERT_TLM_PositionValid(0, false);
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(0);
}

void AttitudeEstimatorTester ::testReferenceGradeIsCarriedAndAlerted() {
  this->loadIgrf();
  this->setValidParameters();
  const QuatBI truth(polaris::math::Quaternion::Identity());

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_RefGrade(0, TableGrade::PRECISE);
  ASSERT_EVENTS_ReferenceDegraded_SIZE(0);

  // The tables fall back to their coarse sources (Push 38): still safe, but the
  // systematic budget is larger, so it is alerted — once.
  this->stub_grade_ = TableGrade::COARSE;
  for (int i = 0; i < 3; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }
  // One alert per domain — the operator needs to know *which* reference went
  // coarse, since the ephemeris and EOP tables are refreshed separately.
  ASSERT_EVENTS_ReferenceDegraded_SIZE(2);
  ASSERT_EVENTS_ReferenceDegraded(0, TableDomain::EPHEMERIS, TableGrade::COARSE);
  ASSERT_EVENTS_ReferenceDegraded(1, TableDomain::EOP, TableGrade::COARSE);
  ASSERT_TLM_RefGrade(3, TableGrade::COARSE);

  // Recovery re-arms the alerts.
  this->stub_grade_ = TableGrade::PRECISE;
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_ReferenceRecovered_SIZE(2);
}

void AttitudeEstimatorTester ::testPositionLossBlocksTheMagneticPair() {
  this->loadIgrf();
  this->setValidParameters();
  const QuatBI truth(polaris::math::Quaternion::Identity());

  this->gnss_valid_ = false;
  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_EVENTS_PositionUnavailable_SIZE(1);
  ASSERT_TLM_PositionValid(0, false);
  ASSERT_TLM_MagValid(0, false);   // no position, no field model
  ASSERT_TLM_SunValid(0, true);    // the sun pair is unaffected
  ASSERT_TLM_TriadAccepted(0, 0);  // but one pair is not an attitude
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);
  // The rate still goes out — that is what Safe-mode damping runs on.
  ASSERT_TRUE(this->last_estimate_.get_rateValid());

  // Edge-gated: the warning does not repeat every cycle of an outage.
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_EVENTS_PositionUnavailable_SIZE(1);
}

void AttitudeEstimatorTester ::testImplausiblePositionIsRejected() {
  this->loadIgrf();
  this->setValidParameters();
  const QuatBI truth(polaris::math::Quaternion::Identity());

  // A fix flagged valid and perfectly fresh, but at a radius no spacecraft
  // occupies. Left ungated it would reach the field model and the sun reference
  // while PositionValid still read true.
  this->position_override_ = Eigen::Vector3d(1.0e3, 0.0, 0.0);
  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_TLM_PositionValid(0, false);
  ASSERT_TLM_MagValid(0, false);
  ASSERT_EVENTS_PositionUnavailable_SIZE(1);
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);

  // Same for a non-finite position off the wire.
  this->position_override_ = Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_TLM_PositionValid(1, false);
  ASSERT_TLM_MagValid(1, false);
}

void AttitudeEstimatorTester ::testResetReArmsEveryAlert() {
  this->loadIgrf();
  this->setValidParameters();
  const QuatBI truth(polaris::math::Quaternion::Identity());

  // Get every edge-gated alert to fire once: coarse references and no position.
  this->stub_grade_ = TableGrade::COARSE;
  this->gnss_valid_ = false;
  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);
  ASSERT_EVENTS_PositionUnavailable_SIZE(1);
  ASSERT_EVENTS_ReferenceDegraded_SIZE(2);

  // Still faulted, so nothing repeats.
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_EVENTS_PositionUnavailable_SIZE(1);
  ASSERT_EVENTS_ReferenceDegraded_SIZE(2);

  // A reset means "tell me everything again": the same faults must re-report.
  this->sendCmd_RESET_ESTIMATOR(0, 0);
  ASSERT_EVENTS_EstimatorReset_SIZE(1);
  this->feedMeasurements(kStartTaiNs + 2 * kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + 2 * kNsPerSecond / 10);
  ASSERT_EVENTS_PositionUnavailable_SIZE(2);
  ASSERT_EVENTS_ReferenceDegraded_SIZE(4);
}

void AttitudeEstimatorTester ::testExpiredIgrfSnapshotRefusesTheMagneticReference() {
  // A snapshot taken from an old IAGA bracket: its published model runs out
  // before the epoch the vehicle is flying at, so the field must be refused
  // rather than extrapolated across the gap.
  this->loadIgrf(2005.0);
  this->setValidParameters();
  const QuatBI truth(polaris::math::Quaternion::Identity());

  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  ASSERT_EVENTS_MagneticReferenceStale_SIZE(1);
  ASSERT_TLM_MagValid(0, false);
  ASSERT_TLM_PositionValid(0, true);  // the position is fine; the model is not
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);

  // Edge-gated: an expired snapshot does not warn every cycle.
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_EVENTS_MagneticReferenceStale_SIZE(1);
}

// ----------------------------------------------------------------------
// Fine mode (REQ-ADET-004 arbitration)
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testPromotesToFineAndEstimatesGyroBias() {
  this->loadIgrf();
  this->setValidParameters(true);

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 2.0, 3.0).normalized(), 0.7));
  // ~0.06 deg/s of gyro bias: the error that makes an unaided gyro useless
  // within minutes, and the one thing the coarse chain structurally cannot see.
  const Eigen::Vector3d bias(1.0e-3, -5.0e-4, 8.0e-4);
  this->gyro_bias_ = bias;

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  // Both pairs valid on the first cycle, so coarse acquires and the Davenport
  // seed engages fine mode in the same cycle — but the *published* product is
  // still coarse for this one cycle. `Mekf::initialize` seeds an attitude and no
  // rate (nothing has been propagated yet), so publishing the filter here would
  // hand the GNC chain a one-cycle rate dropout while the coarse chain had a
  // perfectly good rate. This is the assertion that pins that.
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);
  ASSERT_EVENTS_FineConfigInvalid_SIZE(0);
  ASSERT_EVENTS_FineInitFailed_SIZE(0);
  ASSERT_TLM_EstMode(0, EstimationMode::COARSE);
  ASSERT_TRUE(this->last_estimate_.get_rateValid());
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  const F64 seed_trace = this->tlmHistory_FineAttCovTrace->at(0).arg;
  ASSERT_TRUE(std::isfinite(seed_trace));

  // The filter takes over on the next cycle, with a propagated rate in hand.
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_EstMode(1, EstimationMode::FINE);
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::FINE);
  ASSERT_TRUE(this->last_estimate_.get_rateValid());

  // 60 s at 10 Hz. The truth attitude is held fixed while the gyro reports a
  // bias, so every cycle the propagated attitude walks away from the vector
  // measurements by a consistent amount — which is what makes the bias
  // observable at all.
  for (int i = 0; i < 600; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    this->clearHistory();  // 600 cycles overrun the bounded history
  }
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  // Never gave the solution up: no demotion, and the mode is still FINE.
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_TLM_EstMode(0, EstimationMode::FINE);
  ASSERT_TLM_FineDemotions(0, 0);

  const Vec3F64 estimated = this->tlmHistory_GyroBias->at(0).arg;
  const Eigen::Vector3d b_est(estimated[0], estimated[1], estimated[2]);
  // Deliberately a loose bound: what is being pinned here is that the component
  // wired the filter up such that the bias converges *towards* the truth, not
  // the convergence rate, which tests/unit/mekf_test.cpp owns.
  EXPECT_LT((b_est - bias).norm(), 0.5 * bias.norm())
      << "estimated bias " << b_est.transpose() << " vs truth " << bias.transpose();

  // And the point of running a filter at all: the fine covariance sits well
  // below the single-frame solution that seeded it, which is the coarse chain's
  // floor. AttCovTrace, the published one, is the fine one while FINE is active.
  const F64 fine_trace = this->tlmHistory_FineAttCovTrace->at(0).arg;
  EXPECT_LT(fine_trace, 0.3 * seed_trace);
  EXPECT_DOUBLE_EQ(this->tlmHistory_AttCovTrace->at(0).arg, fine_trace);
  EXPECT_TRUE(std::isfinite(this->tlmHistory_BiasCovTrace->at(0).arg));
}

void AttitudeEstimatorTester ::testNisStreakDemotesToCoarse() {
  this->loadIgrf();
  this->setValidParameters(true);
  const QuatBI truth(polaris::math::Quaternion::Identity());

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);

  // A sun direction 60 deg from where the filter says it must be: the innovation
  // is orders of magnitude past the 35 mrad sigma, so the NIS gate rejects it
  // every cycle. The magnetic pair stays honest, so this is specifically a
  // rejection streak and not a total measurement loss.
  this->sun_body_error_rad_ = 1.05;
  for (U32 i = 0; i < kMekfNisStreak; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }

  ASSERT_EVENTS_FineModeDemoted_SIZE(1);
  ASSERT_EQ(this->eventHistory_FineModeDemoted->at(0).reason.e, FineDemotionReason::NIS_STREAK);
  ASSERT_TLM_FineDemotions(this->tlmHistory_FineDemotions->size() - 1, 1);
  // The published product falls back to the coarse chain, which never stopped
  // running — so the vehicle still has an attitude.
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::COARSE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  // The demotion dropped the filter, which zeroes its own rejection count — the
  // component's running total is what survives, and it is what FDIR trends. A
  // counter that resets itself at the moment the fault it counts occurs would be
  // worse than no counter.
  const FwSizeType last = this->tlmHistory_MekfRejectedTotal->size() - 1;
  ASSERT_GE(this->tlmHistory_MekfRejectedTotal->at(last).arg, kMekfNisStreak);
  ASSERT_EQ(this->tlmHistory_MekfRejected->at(last).arg, 0u);
}

void AttitudeEstimatorTester ::testRefusalStreakDemotesToCoarse() {
  this->loadIgrf();
  this->setValidParameters(true);
  const QuatBI truth(polaris::math::Quaternion::Identity());

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);

  // A zero-length sun vector: finite, so it clears the component's own gate and
  // reaches the filter, which refuses it as unnormalisable. That is a *refusal*,
  // not a gate rejection, and the two must not be confused — one says the
  // measurement is an outlier, the other that it is not a measurement.
  //
  // FILTER_FAULT is the one demotion path left unexercised here: it needs a
  // non-finite *internal* result, and every route to one from outside runs
  // through a finiteness gate in the component or in Mekf::update first. Pinning
  // it would mean reaching past both, which tests the harness rather than the
  // component. tests/unit/mekf_test.cpp owns the filter's own fault handling.
  this->sun_body_degenerate_ = true;
  for (U32 i = 0; i < kMekfRefusalStreak; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }

  ASSERT_EVENTS_FineModeDemoted_SIZE(1);
  ASSERT_EQ(this->eventHistory_FineModeDemoted->at(0).reason.e, FineDemotionReason::REFUSAL_STREAK);
  // Not one rejection: a refusal is not an outlier, and counting it as one would
  // have FDIR chasing the wrong fault.
  const FwSizeType last = this->tlmHistory_MekfRejectedTotal->size() - 1;
  ASSERT_EQ(this->tlmHistory_MekfRejectedTotal->at(last).arg, 0u);
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::COARSE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
}

void AttitudeEstimatorTester ::testCoastDemotesFineMode() {
  this->loadIgrf();
  this->setValidParameters(true);
  const QuatBI truth(polaris::math::Quaternion::Identity());

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);

  // Both vector sources gone: eclipse takes the sun pair, and without a GNSS fix
  // there is no position to evaluate the field model at, so the magnetic pair
  // goes with it. 30 cycles = 3 s, past the 2 s fine coast horizon but inside
  // the coarse chain's 5 s one.
  this->gnss_valid_ = false;
  for (int i = 0; i < 30; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), false);
    this->runCycleAt(t);
  }

  ASSERT_EVENTS_FineModeDemoted_SIZE(1);
  ASSERT_EQ(this->eventHistory_FineModeDemoted->at(0).reason.e, FineDemotionReason::COAST);
  // Nothing was *lost*: the coarse solution was live underneath the whole time,
  // so a demotion must not look to FDIR like an attitude loss.
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::COARSE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
}

void AttitudeEstimatorTester ::testMissingFineTuningLeavesCoarseRunning() {
  this->loadIgrf();
  this->setValidParameters(false);  // the twelve coarse values only
  const QuatBI truth(polaris::math::Quaternion::Identity());

  this->feedMeasurements(kStartTaiNs, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);

  // One warning, and a working vehicle: the §10 Safe-mode floor does not depend
  // on the fine mode being configured.
  ASSERT_EVENTS_FineConfigInvalid_SIZE(1);
  ASSERT_EVENTS_ConfigInvalid_SIZE(0);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
  ASSERT_EVENTS_FineModeEngaged_SIZE(0);
  ASSERT_TLM_EstMode(0, EstimationMode::COARSE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());

  // Edge-gated, like every other alert here.
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_EVENTS_FineConfigInvalid_SIZE(1);
}

void AttitudeEstimatorTester ::testResetDropsFineMode() {
  this->loadIgrf();
  this->setValidParameters(true);
  const QuatBI truth(polaris::math::Quaternion::Identity());

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);

  // A reset that left a converged filter publishing would be exactly the
  // solution the operator asked to be rid of.
  this->sendCmd_RESET_ESTIMATOR(0, 0);
  ASSERT_EVENTS_FineModeDemoted_SIZE(1);
  ASSERT_EQ(this->eventHistory_FineModeDemoted->at(0).reason.e, FineDemotionReason::COMMANDED);

  // Re-acquisition goes through a fresh Davenport seed, not a resumed filter:
  // the coarse chain re-acquires whole off the next TRIAD and re-seeds on the
  // same cycle, with the filter taking over publication the cycle after.
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(2);

  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EQ(this->last_estimate_.get_mode(), EstimationMode::FINE);
}

// ----------------------------------------------------------------------
// Commanded magnetometer calibration (§8.1)
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testMagCalCollectsFitsAndAppliesTheCorrection() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;
  // Symmetric soft iron: magnitude data constrains S only up to a left rotation,
  // so a symmetric injection is the part that is physically recoverable.
  this->mag_soft_iron_ << 1.012, 0.004, -0.003,  //
      0.004, 0.993, 0.005,                       //
      -0.003, 0.005, 1.006;

  I64 t = kStartTaiNs;
  // Baseline: with the iron uncorrected, TRIAD puts the whole magnetic-direction
  // error into the roll about the sun, which is what the calibration removes.
  const double uncalibrated_error = this->maxPublishedErrorOverCycles(10, t);
  EXPECT_GT(uncalibrated_error, 0.01) << "the injected iron should be plainly visible";

  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  ASSERT_EVENTS_MagCalStarted_SIZE(1);
  ASSERT_EVENTS_MagCalStarted(0, kMagCalMinSamples);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_MAG_CAL_START, 0, Fw::CmdResponse::OK);

  // One cycle in, the window is visibly open and counting.
  this->runTumbleCycles(1, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::COLLECTING);
  ASSERT_TLM_MagCalSamples(this->tlmHistory_MagCalSamples->size() - 1, 1);

  // The rest of the window. The fit fires on the cycle the target is reached.
  this->runTumbleCycles(static_cast<int>(kMagCalMinSamples) - 1, t);
  ASSERT_EVENTS_MagCalRejected_SIZE(0);
  ASSERT_EVENTS_MagCalComplete_SIZE(1);
  const F64 residual = this->eventHistory_MagCalComplete->at(0).residualAngleRad;
  const F64 coverage = this->eventHistory_MagCalComplete->at(0).coverage;
  EXPECT_EQ(this->eventHistory_MagCalComplete->at(0).samples, kMagCalMinSamples);
  EXPECT_GE(coverage, kMagCalMinCoverage);
  // Noiseless synthetic data, so the fit is exact to round-off; the assertion is
  // only that the reported residual is a real, small number rather than zero by
  // construction on a rank-deficient solve.
  EXPECT_LT(residual, 1.0e-4);
  EXPECT_TRUE(std::isfinite(residual));

  // The state machine moved to APPLIED and the residual channel is no longer NaN.
  this->runTumbleCycles(1, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::APPLIED);
  EXPECT_LT(
      this->tlmHistory_MagCalResidualAngle->at(this->tlmHistory_MagCalResidualAngle->size() - 1)
          .arg,
      1.0e-4);

  // **The assertion that matters.** The published attitude error collapses,
  // which can only happen if the correction is applied to the vector the
  // estimator consumes — a calibration merely stored would leave this unchanged.
  const double calibrated_error = this->maxPublishedErrorOverCycles(10, t);
  EXPECT_LT(calibrated_error, uncalibrated_error / 10.0)
      << "uncalibrated " << uncalibrated_error << " rad, calibrated " << calibrated_error
      << " rad — the correction is not reaching the estimator's magnetic pair";
}

void AttitudeEstimatorTester ::testMagCalAbortDiscardsTheWindow() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  I64 t = kStartTaiNs;
  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  this->runTumbleCycles(20, t);
  ASSERT_TLM_MagCalSamples(this->tlmHistory_MagCalSamples->size() - 1, 20);

  this->sendCmd_MAG_CAL_ABORT(0, 0);
  ASSERT_CMD_RESPONSE(1, AttitudeEstimator::OPCODE_MAG_CAL_ABORT, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_MagCalAborted_SIZE(1);
  ASSERT_EVENTS_MagCalAborted(0, 20);

  // Nothing was fitted and nothing applied: back to IDLE with no samples.
  this->runTumbleCycles(1, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
  ASSERT_TLM_MagCalSamples(this->tlmHistory_MagCalSamples->size() - 1, 0);
  ASSERT_EVENTS_MagCalComplete_SIZE(0);
  ASSERT_EVENTS_MagCalRejected_SIZE(0);

  // Running well past the old target must not resurrect the window.
  this->runTumbleCycles(static_cast<int>(kMagCalMinSamples) + 10, t);
  ASSERT_EVENTS_MagCalComplete_SIZE(0);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);

  // A second abort with nothing open is a no-op the vehicle accepts.
  this->sendCmd_MAG_CAL_ABORT(0, 0);
  ASSERT_CMD_RESPONSE(2, AttitudeEstimator::OPCODE_MAG_CAL_ABORT, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_MagCalAborted_SIZE(1);
}

void AttitudeEstimatorTester ::testMagCalClearRevertsToRaw() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  I64 t = kStartTaiNs;
  const double uncalibrated_error = this->maxPublishedErrorOverCycles(10, t);

  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  this->runTumbleCycles(static_cast<int>(kMagCalMinSamples), t);
  ASSERT_EVENTS_MagCalComplete_SIZE(1);
  const double calibrated_error = this->maxPublishedErrorOverCycles(10, t);
  EXPECT_LT(calibrated_error, uncalibrated_error / 10.0);

  this->sendCmd_MAG_CAL_CLEAR(0, 0);
  ASSERT_EVENTS_MagCalCleared_SIZE(1);

  // Reverted: every consumer is back on the raw field, so the error returns.
  const double reverted_error = this->maxPublishedErrorOverCycles(10, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
  EXPECT_TRUE(std::isnan(
      this->tlmHistory_MagCalResidualAngle->at(this->tlmHistory_MagCalResidualAngle->size() - 1)
          .arg));
  // Against the *calibrated* run rather than the uncalibrated baseline: the two
  // windows cover different orbit geometries, and how much an uncorrected hard
  // iron tilts the field depends on where the field points, so comparing two
  // uncorrected windows compares geometries. Calibrated-vs-reverted over the
  // same geometry is the clean statement, and it is an order of magnitude.
  EXPECT_GT(reverted_error, 10.0 * calibrated_error);

  // Idempotent: clearing nothing is accepted and emits nothing.
  this->sendCmd_MAG_CAL_CLEAR(0, 0);
  ASSERT_EVENTS_MagCalCleared_SIZE(1);
}

void AttitudeEstimatorTester ::testMagCalRejectsNarrowCoverage() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  I64 t = kStartTaiNs;
  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  // A nearly-fixed attitude: the body-frame field direction barely moves, so the
  // samples sit inside a pinhole cone and the ellipsoid would be extrapolated
  // over directions it never saw.
  const QuatBI held(polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 0.2));
  for (U32 i = 0; i < kMagCalMinSamples; ++i) {
    this->feedMeasurements(t, held, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  ASSERT_EVENTS_MagCalComplete_SIZE(0);
  ASSERT_EVENTS_MagCalRejected_SIZE(1);
  ASSERT_EQ(this->eventHistory_MagCalRejected->at(0).reason, MagCalRejectReason::COVERAGE);
  EXPECT_EQ(this->eventHistory_MagCalRejected->at(0).samples, kMagCalMinSamples);
  EXPECT_LT(this->eventHistory_MagCalRejected->at(0).coverage, kMagCalMinCoverage);

  // Nothing applied, and the window is closed rather than left half-open.
  this->feedMeasurements(t, held, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
  EXPECT_TRUE(std::isnan(
      this->tlmHistory_MagCalResidualAngle->at(this->tlmHistory_MagCalResidualAngle->size() - 1)
          .arg));
}

void AttitudeEstimatorTester ::testMagCalResetAbortsAndClears() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  I64 t = kStartTaiNs;

  // First: a calibration applied, then a second window opened over it.
  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  this->runTumbleCycles(static_cast<int>(kMagCalMinSamples), t);
  ASSERT_EVENTS_MagCalComplete_SIZE(1);
  const double calibrated_error = this->maxPublishedErrorOverCycles(10, t);
  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  this->runTumbleCycles(10, t);
  // COLLECTING outranks APPLIED: the open window is the condition worth seeing.
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::COLLECTING);
  // And the second window started from zero rather than inheriting the first
  // window's accumulator, which would fit the two runs together.
  ASSERT_TLM_MagCalSamples(this->tlmHistory_MagCalSamples->size() - 1, 10);

  this->sendCmd_RESET_ESTIMATOR(0, 0);
  ASSERT_EVENTS_MagCalCleared_SIZE(1);

  // Both halves of the full reset: the window is gone (no second Complete even
  // well past its target) and the applied correction is gone with it.
  this->runTumbleCycles(static_cast<int>(kMagCalMinSamples) + 5, t);
  const double reverted_error = this->maxPublishedErrorOverCycles(10, t);
  ASSERT_EVENTS_MagCalComplete_SIZE(1);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
  EXPECT_GT(reverted_error, 10.0 * calibrated_error);
}

void AttitudeEstimatorTester ::testMagCalStartRefusedWithoutParameters() {
  this->loadIgrf();
  this->setValidParameters();  // flight tuning present, MagCal* deliberately not
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  this->sendCmd_MAG_CAL_START(0, 0, kMagCalMinSamples);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_MAG_CAL_START, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_MagCalRejected_SIZE(1);
  ASSERT_EQ(this->eventHistory_MagCalRejected->at(0).reason, MagCalRejectReason::CONFIG);
  ASSERT_EVENTS_MagCalStarted_SIZE(0);

  // No window opened, and — the point of putting this gate on the command — the
  // estimator is entirely unaffected: it still acquires, on a raw magnetometer.
  I64 t = kStartTaiNs;
  this->runTumbleCycles(3, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
  ASSERT_EVENTS_ConfigInvalid_SIZE(0);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
}

void AttitudeEstimatorTester ::testMagCalStartRejectsOutOfRangeCounts() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();

  // Below MagCalMinSamples: the fit would be refused on SAMPLES at the end of
  // the window, so saying so now costs the operator the command, not the window.
  this->sendCmd_MAG_CAL_START(0, 0, 1);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_MAG_CAL_START, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_MagCalRejected_SIZE(1);
  ASSERT_EQ(this->eventHistory_MagCalRejected->at(0).reason, MagCalRejectReason::SAMPLES);

  // Above the ceiling: a window the ground would never see close.
  this->sendCmd_MAG_CAL_START(0, 1, AttitudeEstimator::kMaxCalSamples + 1);
  ASSERT_CMD_RESPONSE(1, AttitudeEstimator::OPCODE_MAG_CAL_START, 1,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_MagCalRejected_SIZE(2);
  ASSERT_EQ(this->eventHistory_MagCalRejected->at(1).reason, MagCalRejectReason::SAMPLES);

  // Neither opened a window.
  ASSERT_EVENTS_MagCalStarted_SIZE(0);
  I64 t = kStartTaiNs;
  this->runTumbleCycles(3, t);
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::IDLE);
}

void AttitudeEstimatorTester ::testEstimatorUndisturbedDuringCollection() {
  this->loadIgrf();
  this->setValidParameters();
  this->setMagCalParameters();
  this->mag_hard_iron_t_ = kInjectedHardIronT;

  // A target the run never reaches, so the comparison covers collection alone —
  // applying a fit is *supposed* to change the estimator, and does so in
  // testMagCalCollectsFitsAndAppliesTheCorrection.
  const U32 unreachable = kMagCalMinSamples * 10;
  I64 t = kStartTaiNs;
  this->sendCmd_MAG_CAL_START(0, 0, unreachable);
  this->runTumbleCycles(30, t);

  // The same measurements through a second component with no window open.
  AttitudeEstimatorTester quiet;
  quiet.loadIgrf();
  quiet.setValidParameters();
  quiet.mag_hard_iron_t_ = kInjectedHardIronT;
  I64 t_quiet = kStartTaiNs;
  quiet.runTumbleCycles(30, t_quiet);

  ASSERT_EQ(this->tlmHistory_EstMode->size(), quiet.tlmHistory_EstMode->size());
  for (U32 i = 0; i < this->tlmHistory_EstMode->size(); ++i) {
    ASSERT_EQ(this->tlmHistory_EstMode->at(i).arg, quiet.tlmHistory_EstMode->at(i).arg)
        << "mode diverged at sample " << i << " purely from having a window open";
  }
  ASSERT_EQ(this->estimate_count_, quiet.estimate_count_);
  // Bit-identical, not merely close: collection touches nothing the estimator
  // reads, so any difference at all would be a defect.
  const QuatF64 mine = this->last_estimate_.get_qBodyEci();
  const QuatF64 theirs = quiet.last_estimate_.get_qBodyEci();
  for (U32 i = 0; i < 4; ++i) {
    EXPECT_EQ(mine[i], theirs[i]) << "attitude component " << i << " changed during collection";
  }
  // And the window really was running, so the comparison meant something.
  ASSERT_TLM_MagCalState(this->tlmHistory_MagCalState->size() - 1, MagCalState::COLLECTING);
  ASSERT_TLM_MagCalSamples(this->tlmHistory_MagCalSamples->size() - 1, 30);
}

}  // namespace flight

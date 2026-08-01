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

//! Fixed spacecraft position, ECEF [m]: a 620 km equatorial point. The test does
//! not fly an orbit — the geometry only has to be a real place where the field
//! model is well defined.
const pm::Vec3<ECEF> kPositionEcef(7.0e6, 0.0, 0.0);

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
  EXPECT_TRUE(field.field(kPositionEcef, year, b_ecef));
  return rotationAt(taiNs).rotate(b_ecef);
}

pm::Vec3<ECI> AttitudeEstimatorTester ::expectedSunRef(I64 taiNs) const {
  const Eigen::Vector3d dir = perpendicularTo(this->expectedMagRef(taiNs).eigen());
  const pm::Vec3<ECI> sun(1.495978707e11 * dir);
  // The component makes the reference spacecraft-centric before normalising.
  const pm::Vec3<ECI> r_eci = rotationAt(taiNs).rotate(kPositionEcef);
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
  mag.set_fieldTesla(toVec3F64(q_bi.rotate(this->expectedMagRef(taiNs)).eigen()));
  mag.set_timeTagNs(tag);
  mag.set_valid(true);
  this->invoke_to_magnetometerIn(0, mag);

  GnssMeas gnss;
  gnss.set_posEcefM(toVec3F64(this->position_override_.has_value() ? *this->position_override_
                                                                   : kPositionEcef.eigen()));
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

}  // namespace flight

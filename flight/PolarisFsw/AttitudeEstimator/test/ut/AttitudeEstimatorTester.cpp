// ======================================================================
// \title  AttitudeEstimatorTester.cpp
// \brief  Test harness for the AttitudeEstimator component (§8.1, §23.1)
// ======================================================================

#include "AttitudeEstimatorTester.hpp"

#include <cmath>
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

void AttitudeEstimatorTester ::setValidParameters() {
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
  // paramSet_* only stages values in the harness's table; the component caches
  // them at load, exactly as the topology does after ParameterDb is up.
  this->component.loadParameters();
}

void AttitudeEstimatorTester ::loadIgrf(double decimalYear) {
  const double year = (decimalYear > 0.0) ? decimalYear : kIgrfEpochYear;
  ASSERT_TRUE(this->component.configureIgrf(POLARIS_IGRF_COEFFS, year));
}

pm::Vec3<ECI> AttitudeEstimatorTester ::expectedMagRef(I64 taiNs) const {
  polaris::environment::IgrfCoefficients coefficients;
  EXPECT_TRUE(
      polaris::environment::loadIgrfIaga(POLARIS_IGRF_COEFFS, kIgrfEpochYear, coefficients));
  const polaris::environment::IgrfField field(coefficients);

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
  imu.set_deltaAngleRad(toVec3F64(rate_body * 0.1));  // 10 Hz cycle
  imu.set_deltaVelMps(toVec3F64(Eigen::Vector3d::Zero()));
  imu.set_intervalSec(0.1);
  imu.set_timeTagNs(tag);
  imu.set_valid(true);
  this->invoke_to_imuIn(0, imu);

  SunSensorMeas sun;
  sun.set_dirBody(toVec3F64(q_bi.rotate(this->expectedSunRef(taiNs)).eigen()));
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

}  // namespace flight

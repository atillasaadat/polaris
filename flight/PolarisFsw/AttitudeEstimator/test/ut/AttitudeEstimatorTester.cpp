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
constexpr F64 kSigmaSunAlbedoCorr = 0.005;
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

//! Multi-IMU voting gates (§8.2). The rate limit is 30 deg/s, the reference
//! vehicle's; the disagreement gate is 0.5 deg/s. Re-admission is deliberately
//! **3** rather than the flight 10, so a harness case can walk the recovery edge
//! in three cycles instead of ten — the policy is what is under test here, not
//! the number, which lib/gnc's own tests pin at its own value.
constexpr F64 kImuMaxRateRadps = 0.5236;
constexpr F64 kImuDisagreementRadps = 0.0087;
constexpr U32 kImuReadmitCycles = 3;
//! Two cycles rather than the flight five, for the same reason: the *policy* is
//! under test here, not the number.
constexpr U32 kImuIdentifyConfirmCycles = 2;
//! Ambiguity escalation horizon, in cycles. Short enough that a harness case can
//! walk past it twice and see the bounded repeat.
constexpr U32 kImuAmbiguityEscalateCycles = 5;

//! Earth-albedo correction tuning. The peak and field of view are the reference
//! vehicle's GomSpace FSS; the boresight is body +Z, its mounting there. The
//! uncorrected sun systematic is deliberately much wider than the corrected one
//! (a factor of 5, the flight ratio is ~2.8) so a test can tell which of the two
//! a cycle was weighted with by looking at the reported covariance.
constexpr F64 kSunAlbedoPeakRad = 0.20944;     // 12 deg
constexpr F64 kSunAlbedoHalfFovRad = 1.04720;  // 60 deg
constexpr F64 kSigmaSunAlbedoUncorr = 5.0 * kSigmaSunAlbedoCorr;

//! Ephemeris terms. The harness's stubbed getBodyPosition answers at
//! `stub_grade_`, so a test can put the reference on either side of the split;
//! the two are two orders of magnitude apart, as they are in flight, so a test
//! can tell which was used from the reported covariance alone.
constexpr F64 kSigmaSunEphem = 7.0e-3;
constexpr F64 kSigmaSunEphemPrecise = 5.0e-5;

//! Fine-mode tuning. The horizons and streaks are far shorter than flight values
//! so a demotion path is a handful of cycles rather than minutes of them; the
//! noise terms are the reference vehicle's, since those are what the filter's
//! consistency depends on.
constexpr F64 kMekfRrw = 4.7e-7;
constexpr F64 kMekfNisGate = 13.82;
constexpr F64 kMekfAttNisGate = 16.27;  // chi-square(3) at 99.9%
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

//! Multi-magnetometer voting gates (§8.2). The band is the reference vehicle's;
//! the disagreement gate is 5 uT, likewise. Re-admission and identification
//! confirmation are shortened to 3 and 2 for the same reason the IMU pair are:
//! the *policy* is under test here, not the number.
constexpr F64 kMagMinFieldRatio = 0.5;
constexpr F64 kMagMaxFieldRatio = 1.6;
constexpr F64 kMagDisagreementT = 5.0e-6;
constexpr F64 kMagMaxAttSigmaRad = 0.02;
constexpr U32 kMagReadmitCycles = 3;
constexpr U32 kMagIdentifyConfirmCycles = 2;

//! Star-tracker fusion tuning (§8.2). The two sigmas are the reference vehicle's
//! AURIGA figures; the king is unit 0. The monitor thresholds are the flight
//! values, because what is under test is whether a drifted source is caught, and
//! moving the gate would stop the harness testing the one that ships. The
//! persistence is **3** rather than the flight 50 so a case can walk the alert
//! edge in three cycles.
constexpr U32 kStKingUnit = 0;
constexpr F64 kStSigmaXyRad = 1.042e-4;
constexpr F64 kStSigmaZRad = 1.832e-4;
constexpr F64 kMonitorSunResidualRad = 0.15;
constexpr F64 kMonitorMagResidualRad = 0.2;
constexpr F64 kMonitorSunCrossUnitRad = 0.15;
constexpr U32 kMonitorAlertCycles = 3;

//! Inter-tracker alignment gates. The sample floor is 20 rather than the flight
//! 100 so a window closes inside a short harness run; the residual and eigen-gap
//! gates are the flight values.
constexpr U32 kStAlignMinSamples = 20;

//! `kCoarseAgreementGate` in AttitudeEstimatorStarTracker.cpp: χ²₃ at 0.999.
//! Mirrored rather than exported, so the tests read the contract at the same
//! number the flight code decides on and a change to either fails loudly.
constexpr double kCoarseAgreementGate = 16.266;
constexpr F64 kStAlignMaxResidualRad = 5.0e-4;
constexpr F64 kStAlignMinEigenGap = 0.9;

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
  for (FwIndexType i = 0; i < AttitudeEstimator::NUM_STARTRACKERIN_INPUT_PORTS; ++i) {
    this->star_error_[i].setZero();
  }
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
  const Eigen::Vector3d sun = 1.495978707e11 * this->sunDirectionEci(taiNs);
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

void AttitudeEstimatorTester ::setValidParameters(bool withFine, bool withAlbedo,
                                                  bool withStarTracker) {
  this->paramSet_SigmaSunWhiteRad(kSigmaSunWhite, Fw::ParamValid::VALID);
  this->paramSet_SigmaSunAlbedoRad(kSigmaSunAlbedoCorr, Fw::ParamValid::VALID);
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
  this->paramSet_SigmaSunAlbedoUncorrRad(kSigmaSunAlbedoUncorr, Fw::ParamValid::VALID);
  this->paramSet_SigmaSunEphemRad(kSigmaSunEphem, Fw::ParamValid::VALID);
  this->paramSet_SigmaSunEphemPreciseRad(kSigmaSunEphemPrecise, Fw::ParamValid::VALID);
  // The §8.2 IMU-voting gates ride in the coarse set: without a voted rate there
  // is no estimator at all, so their absence costs what a missing sigma costs.
  this->paramSet_ImuMaxRateRadps(kImuMaxRateRadps, Fw::ParamValid::VALID);
  this->paramSet_ImuDisagreementRadps(kImuDisagreementRadps, Fw::ParamValid::VALID);
  this->paramSet_ImuReadmitCycles(kImuReadmitCycles, Fw::ParamValid::VALID);
  this->paramSet_ImuIdentifyConfirmCycles(kImuIdentifyConfirmCycles, Fw::ParamValid::VALID);
  this->paramSet_ImuAmbiguityEscalateCycles(kImuAmbiguityEscalateCycles, Fw::ParamValid::VALID);
  // The §8.2 magnetometer-voting gates ride in the same set and for the same
  // reason: without a voted field there is no TRIAD and hence no coarse attitude.
  this->paramSet_MagMinFieldRatio(kMagMinFieldRatio, Fw::ParamValid::VALID);
  this->paramSet_MagMaxFieldRatio(kMagMaxFieldRatio, Fw::ParamValid::VALID);
  this->paramSet_MagDisagreementT(kMagDisagreementT, Fw::ParamValid::VALID);
  this->paramSet_MagMaxAttSigmaRad(kMagMaxAttSigmaRad, Fw::ParamValid::VALID);
  this->paramSet_MagReadmitCycles(kMagReadmitCycles, Fw::ParamValid::VALID);
  this->paramSet_MagIdentifyConfirmCycles(kMagIdentifyConfirmCycles, Fw::ParamValid::VALID);
  if (withStarTracker) {
    this->paramSet_StKingUnit(kStKingUnit, Fw::ParamValid::VALID);
    this->paramSet_StSigmaXyRad(kStSigmaXyRad, Fw::ParamValid::VALID);
    this->paramSet_StSigmaZRad(kStSigmaZRad, Fw::ParamValid::VALID);
    this->paramSet_MonitorSunResidualRad(kMonitorSunResidualRad, Fw::ParamValid::VALID);
    this->paramSet_MonitorMagResidualRad(kMonitorMagResidualRad, Fw::ParamValid::VALID);
    this->paramSet_MonitorSunCrossUnitRad(kMonitorSunCrossUnitRad, Fw::ParamValid::VALID);
    this->paramSet_MonitorAlertCycles(kMonitorAlertCycles, Fw::ParamValid::VALID);
    Vec3F64PerUnit boresights;
    for (FwIndexType i = 0; i < static_cast<FwIndexType>(this->st_boresights_.size()); ++i) {
      boresights[i] = this->st_boresights_[static_cast<std::size_t>(i)];
    }
    this->paramSet_StBoresightsBody(boresights, Fw::ParamValid::VALID);
  }
  if (withAlbedo) {
    this->paramSet_SunAlbedoPeakRad(kSunAlbedoPeakRad, Fw::ParamValid::VALID);
    this->paramSet_SunAlbedoHalfFovRad(kSunAlbedoHalfFovRad, Fw::ParamValid::VALID);
    Vec3F64PerUnit boresights;
    for (FwIndexType i = 0; i < static_cast<FwIndexType>(this->sun_boresights_.size()); ++i) {
      boresights[i] = this->sun_boresights_[static_cast<std::size_t>(i)];
    }
    this->paramSet_SunAlbedoBoresightsBody(boresights, Fw::ParamValid::VALID);
  }
  if (withFine) {
    this->paramSet_MekfRrw(kMekfRrw, Fw::ParamValid::VALID);
    this->paramSet_MekfNisGate(kMekfNisGate, Fw::ParamValid::VALID);
    this->paramSet_MekfAttNisGate(kMekfAttNisGate, Fw::ParamValid::VALID);
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

void AttitudeEstimatorTester ::setStAlignParameters() {
  this->paramSet_StAlignMinSamples(kStAlignMinSamples, Fw::ParamValid::VALID);
  this->paramSet_StAlignMaxResidualRad(kStAlignMaxResidualRad, Fw::ParamValid::VALID);
  this->paramSet_StAlignMinEigenGap(kStAlignMinEigenGap, Fw::ParamValid::VALID);
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

Eigen::Vector3d AttitudeEstimatorTester ::sunDirectionEci(I64 taiNs) const {
  const Eigen::Vector3d mag_ref = this->expectedMagRef(taiNs).eigen();
  if (!this->sun_at_45_from_nadir_) {
    // The default: square to the modelled field, so the pair geometry is
    // unambiguous and every test that predates the albedo correction sees the
    // Sun exactly where it always did.
    return perpendicularTo(mag_ref);
  }
  // The albedo correction's working geometry. Its magnitude goes as
  // cos(theta) * sin(theta) in the spacecraft's angle theta from the sub-solar
  // point — the dayside factor times the Sun-to-Earth-centre separation — so 45
  // degrees is where it peaks. The out-of-plane axis keeps the Sun well away
  // from the field direction, so TRIAD stays far from its observability gate.
  const Eigen::Vector3d r_hat =
      rotationAt(taiNs).rotate(pm::Vec3<ECEF>(this->position_ecef_)).eigen().normalized();
  const Eigen::Vector3d out_of_plane = r_hat.cross(mag_ref).normalized();
  return (M_SQRT1_2 * (r_hat + out_of_plane)).normalized();
}

pm::Vec3<ECI> AttitudeEstimatorTester ::expectedSunRef(I64 taiNs) const {
  const pm::Vec3<ECI> sun(1.495978707e11 * this->sunDirectionEci(taiNs));
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

  // The IMU suite (§8.2). Every unit reports the same rate + bias — identical
  // readings median to themselves, so a healthy suite behaves exactly as the
  // single unit did — unless imu_fault_ names one to break.
  for (FwIndexType unit = 0; unit < this->imu_unit_count_; ++unit) {
    ImuMeas imu;
    Eigen::Vector3d delta_angle = (rate_body + this->gyro_bias_) * 0.1;  // 10 Hz cycle
    I64 unit_tag = tag;
    bool unit_valid = true;
    if (unit == this->imu_fault_.index) {
      switch (this->imu_fault_.kind) {
        case ImuFault::kRailed:
          // Reported as a delta-angle, so the rate the component reconstructs is
          // value / interval * interval = value.
          delta_angle = this->imu_fault_.value * 0.1;
          break;
        case ImuFault::kNotFinite:
          delta_angle = Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
          break;
        case ImuFault::kStale:
          // Aged past MaxMeasAgeSec: the unit is *absent*, not implausible, which
          // is the distinction the vote has to keep (a dropout must not latch an
          // exclusion).
          unit_tag = tag - static_cast<I64>(10.0 * kMaxMeasAgeSec * kNsPerSecond);
          break;
        case ImuFault::kOffset:
          delta_angle = (rate_body + this->gyro_bias_ + this->imu_fault_.value) * 0.1;
          break;
        case ImuFault::kNone:
          break;
      }
    }
    imu.set_deltaAngleRad(toVec3F64(delta_angle));
    imu.set_deltaVelMps(toVec3F64(Eigen::Vector3d::Zero()));
    imu.set_intervalSec(0.1);
    imu.set_timeTagNs(unit_tag);
    imu.set_valid(unit_valid);
    this->invoke_to_imuIn(unit, imu);
  }

  SunSensorMeas sun;
  const Eigen::Vector3d sun_body_truth = q_bi.rotate(this->expectedSunRef(taiNs)).eigen();
  Eigen::Vector3d sun_body = sun_body_truth;
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
  // Which port the unit arrives on. The albedo correction follows the *selected*
  // unit's boresight (§8.2), so moving this index — with or without writing that
  // slot's boresight — is how the per-unit path is tested.
  this->invoke_to_sunSensorIn(this->sun_port_index_, sun);
  if (this->sun_extra_.index >= 0) {
    // A second unit seeing the same Sun, differing only in its reported sigma:
    // the discriminator the component selects on.
    SunSensorMeas other = sun;
    other.set_sigmaRad(this->sun_extra_.sigma_rad);
    if (this->sun_extra_truthful_) {
      other.set_dirBody(toVec3F64(sun_body_truth));
    }
    this->invoke_to_sunSensorIn(this->sun_extra_.index, other);
  }

  // The magnetometer suite (§8.2). Every unit reports the same field — identical
  // readings combine to themselves, so a healthy pair behaves exactly as the
  // single unit did — unless mag_fault_index_ names one to offset.
  const Eigen::Vector3d b_body = q_bi.rotate(this->expectedMagRef(taiNs)).eigen();
  for (FwIndexType unit = 0; unit < this->mag_unit_count_; ++unit) {
    MagnetometerMeas mag;
    // The sensor model the ellipsoid fit inverts: m = S·B_body + b. With the
    // defaults (identity S, zero b) this is a perfect magnetometer, so every test
    // that predates the calibration sees exactly what it saw before.
    Eigen::Vector3d field = this->mag_soft_iron_ * b_body + this->mag_hard_iron_t_;
    if (unit == this->mag_fault_index_) {
      field = this->mag_fault_scale_ * field + this->mag_fault_offset_t_;
    }
    mag.set_fieldTesla(toVec3F64(field));
    mag.set_timeTagNs(tag);
    mag.set_valid(true);
    this->invoke_to_magnetometerIn(unit, mag);
  }

  // The star-tracker suite (§8.2). Each unit reports the truth attitude composed
  // with its own fixed small-angle error — slot 0 standing in for the king's own
  // bias, which nothing removes because it *is* the frame, and slot 1 for the
  // second unit's mounting misalignment, which is what ST_ALIGN_CAL estimates.
  for (FwIndexType unit = 0; unit < this->star_unit_count_; ++unit) {
    StarTrackerMeas st;
    polaris::math::Quaternion measured = q_bi.core();
    const Eigen::Vector3d& theta = this->star_error_[unit];
    const double angle = theta.norm();
    if (angle > 0.0) {
      // δq(θ) ⊗ q_true, the same composition the sim's tracker model uses, so a
      // harness error and a truth-model error are the same kind of thing.
      measured = polaris::math::Quaternion::FromAxisAngle(theta / angle, angle) * measured;
    }
    (void)measured.normalize();
    measured = measured.canonical();
    QuatF64 q;
    q[0] = measured.w();
    q[1] = measured.x();
    q[2] = measured.y();
    q[3] = measured.z();
    st.set_qBodyEci(q);
    st.set_timeTagNs(tag);
    st.set_valid(this->star_valid_[unit]);
    this->invoke_to_starTrackerIn(unit, st);
  }

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

// --- Earth-albedo correction (§8.1) -----------------------------------------

QuatBI AttitudeEstimatorTester ::earthInTheSunSensorField(I64 taiNs) const {
  // The correction only has work to do when the Earth is actually in the sun
  // sensor's field, and the reference vehicle's sensor looks along body +Z. So
  // hand back the attitude that puts nadir there — an Earth-pointing vehicle,
  // which is the geometry this correction exists for.
  const pm::Vec3<ECI> r_eci = rotationAt(taiNs).rotate(pm::Vec3<ECEF>(this->position_ecef_));
  const Eigen::Vector3d nadir_eci = -r_eci.eigen().normalized();
  const Eigen::Vector3d axis = nadir_eci.cross(Eigen::Vector3d::UnitZ());
  const double angle = std::atan2(axis.norm(), nadir_eci.dot(Eigen::Vector3d::UnitZ()));
  if (!(axis.norm() > 0.0)) {
    return QuatBI(polaris::math::Quaternion::Identity());
  }
  // Negative angle: Quaternion is the frame-rotation convention (§ quaternion.hpp
  // "v_rot = A(q) v_ref"), so rotating the *frame* by -theta about the axis is
  // what carries the nadir *vector* onto +Z.
  const QuatBI q(polaris::math::Quaternion::FromAxisAngle(axis.normalized(), -angle));
  EXPECT_LT((q.rotate(pm::Vec3<ECI>(nadir_eci)).eigen() - Eigen::Vector3d::UnitZ()).norm(), 1.0e-9)
      << "the Earth-pointing attitude does not put nadir on the boresight";
  return q;
}

double AttitudeEstimatorTester ::publishedCovTrace() const {
  const Vec3F64 diag = this->last_estimate_.get_attCovDiagRad2();
  return diag[0] + diag[1] + diag[2];
}

void AttitudeEstimatorTester ::testAlbedoCorrectionAppliesAndTightensTheCovariance() {
  this->sun_at_45_from_nadir_ = true;
  this->loadIgrf();
  this->setValidParameters(false, true);

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);

  // Cycle one: the component has no attitude yet, so it cannot place the Earth
  // in the sensor's field and must **not** correct. Acquisition happens on this
  // cycle, so this is not an edge case to be tolerated — it is every cold start.
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_AlbedoConfigInvalid_SIZE(0);
  EXPECT_TRUE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(0).arg))
      << "corrected on the acquisition cycle, before there was an attitude to correct with";
  const double uncorrected_trace = this->publishedCovTrace();

  // Cycle two: an attitude is published, the Earth is on the boresight and the
  // day side is lit, so the correction runs.
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  const double applied = this->tlmHistory_SunAlbedoCorrection->at(1).arg;
  EXPECT_FALSE(std::isnan(applied)) << "the correction did not run on a geometry that supports it";
  EXPECT_GT(applied, 0.0);
  EXPECT_LE(applied, kSunAlbedoPeakRad) << "a correction larger than the part's peak error";

  // And the sigma selection followed it: the reported covariance floor is the
  // *systematic* budget, so a cycle the correction ran on must report a tighter
  // one than a cycle it did not. That is the whole point of carrying two
  // parameters, and it is what stops the estimator being overconfident on the
  // cycles it could not correct.
  EXPECT_LT(this->publishedCovTrace(), uncorrected_trace)
      << "the corrected cycle was still weighted at the uncorrected sigma";
}

void AttitudeEstimatorTester ::testAlbedoCorrectionSkippedWithoutGeometry() {
  this->sun_at_45_from_nadir_ = true;
  this->loadIgrf();
  this->setValidParameters(false, true);

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);
  // Two cycles to acquire and have an attitude in hand, so the only thing
  // missing below is the geometry itself.
  for (int i = 0; i < 2; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_FALSE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(1).arg));

  // No position fix. The correction refuses rather than assuming a nominal
  // altitude — a guessed geometry would inject a bias the size of the one it
  // removes, pointed wherever the guess happened to point.
  this->gnss_valid_ = false;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  EXPECT_TRUE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(2).arg))
      << "corrected without a position fix";
  this->gnss_valid_ = true;

  // Earth behind the sensor: the vehicle turns until nadir is out of the field.
  // Normal, frequent, and the correct answer is zero rather than a small number.
  // Two cycles, because the correction places the Earth with the *published*
  // attitude: the first cycle is still working from the old one, and it is the
  // second — once the estimator has followed the vehicle round — that the
  // geometry is really gone in.
  const QuatBI away(polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), M_PI) *
                    truth.core());
  for (int i = 0; i < 2; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, away, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }
  EXPECT_TRUE(std::isnan(
      this->tlmHistory_SunAlbedoCorrection->at(this->tlmHistory_SunAlbedoCorrection->size() - 1)
          .arg))
      << "corrected with the Earth out of the sensor's field";
}

void AttitudeEstimatorTester ::testAlbedoSigmaInflatesWithTheAttitudeUncertainty() {
  // The correction's own error is dominated by the rotation-**axis** term
  // `A·ε` — set by the sensor's *peak* albedo scale, not by the applied pull —
  // so how well the measurement can be trusted after correcting depends on how
  // well the attitude that placed the Earth was known. Right after acquisition
  // that is degrees; converged it is a fraction of one.
  //
  // Gating on attitude *validity* alone (which this did before review) tells
  // both estimators the converged number on a cycle whose attitude is 10 deg
  // out — overconfidence in the unsafe direction on exactly the worst cycles.
  // What is asserted here is that the effective sigma **relaxes** as the
  // covariance converges, which is only observable because the coarse chain's
  // reported covariance floor is built from it.
  this->sun_at_45_from_nadir_ = true;
  this->loadIgrf();
  this->setValidParameters(false, true);

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);

  // Acquire, then take the first cycle the correction actually runs on: the
  // attitude is one TRIAD old and the covariance still carries the acquisition
  // uncertainty.
  for (int i = 0; i < 2; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_FALSE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(1).arg));
  const double first_corrected_trace = this->publishedCovTrace();

  // Now let it settle. The inflation feeds off the published covariance and the
  // published covariance floor is built from the inflation, so this is a fixed
  // point — it must converge *downward* rather than sit at the acquisition value.
  for (int i = 0; i < 30; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  const double settled_trace = this->publishedCovTrace();

  EXPECT_LT(settled_trace, first_corrected_trace)
      << "the attitude-driven sigma inflation never relaxed — is it reading the covariance?";
  // Still the corrected budget, not the uncorrected one: the inflation is a
  // quadrature addition to the corrected sigma, so it can never make a corrected
  // cycle worse than an uncorrected one would have been.
  const double kUncorrectedFloor = 3.0 * kSigmaSunAlbedoUncorr * kSigmaSunAlbedoUncorr;
  EXPECT_LT(settled_trace, kUncorrectedFloor)
      << "a corrected cycle reported a covariance at or above the uncorrected budget";
  // And the correction is genuinely running throughout, so the comparison is
  // between two corrected cycles rather than a corrected and a skipped one.
  for (U32 i = 1; i < this->tlmHistory_SunAlbedoCorrection->size(); ++i) {
    EXPECT_FALSE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(i).arg))
        << "cycle " << i << " stopped correcting mid-run";
  }
}

void AttitudeEstimatorTester ::testAlbedoSkippedForAnUncharacterisedSunSensor() {
  // The albedo boresight is **per unit** (§8.2) and a slot the configuration
  // left as the zero vector means "not installed, or mounting not
  // characterised". A wrong boresight *scales* the correction rather than
  // failing it, so applying unit 0's geometry to a unit on another face would be
  // a silent bias the size of the one being removed — which is why the zero slot
  // takes the uncorrected path instead of falling back to a neighbour's value.
  this->sun_at_45_from_nadir_ = true;
  this->sun_port_index_ = 1;  // default sun_boresights_ leaves slot 1 zero
  this->loadIgrf();
  this->setValidParameters(false, true);

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // The estimator is working — the unit at index 1 is a perfectly good sun
  // sensor and is selected — but it is never corrected, and the whole albedo
  // configuration stays valid (this is a per-unit condition, not a config fault).
  ASSERT_EVENTS_AlbedoConfigInvalid_SIZE(0);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
  ASSERT_TLM_SunValid(2, true);
  ASSERT_TLM_SunUnitSelected(2, 1);
  for (U32 i = 0; i < this->tlmHistory_SunAlbedoCorrection->size(); ++i) {
    EXPECT_TRUE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(i).arg))
        << "cycle " << i << " corrected a unit whose boresight is not configured";
  }
}

void AttitudeEstimatorTester ::testAlbedoFollowsTheSelectedUnitsBoresight() {
  // The other half of the per-unit path: a unit on a *different* port whose
  // boresight **is** configured must be corrected, with its own geometry. This
  // is what makes the §8.2 handoff sweep meaningful — without it, every attitude
  // that moves the Sun off the array normal would silently drop back to the
  // uncorrected sun budget and the accuracy campaign would be measuring a
  // suite the vehicle does not fly.
  this->sun_at_45_from_nadir_ = true;
  this->sun_port_index_ = 2;
  // Slot 2 gets the same body +Z the geometry helper points the Earth into, so
  // the correction has the same thing to remove as the slot-0 case does; the
  // point under test is that the *lookup* follows the selected index.
  this->sun_boresights_[6] = 0.0;
  this->sun_boresights_[7] = 0.0;
  this->sun_boresights_[8] = 1.0;
  this->loadIgrf();
  this->setValidParameters(false, true);

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  ASSERT_EVENTS_AlbedoConfigInvalid_SIZE(0);
  ASSERT_TLM_SunUnitSelected(2, 2);
  bool corrected = false;
  for (U32 i = 0; i < this->tlmHistory_SunAlbedoCorrection->size(); ++i) {
    const F64 applied = this->tlmHistory_SunAlbedoCorrection->at(i).arg;
    if (std::isfinite(applied) && applied > 0.0) {
      corrected = true;
    }
  }
  EXPECT_TRUE(corrected) << "the correction did not follow the selected unit's boresight";
}

void AttitudeEstimatorTester ::testSunSelectionTakesTheBestIlluminatedUnit() {
  // Selection is on the **realised sigma** the unit reports, which is the
  // incidence-cosine criterion in the quantity the estimator consumes (§8.2).
  // Two units see the same Sun; the one reporting the tighter sigma wins,
  // whichever port it arrives on — so the ordering is a property of the
  // measurement, not of the wiring.
  this->loadIgrf();
  this->setValidParameters();

  // Unit 0 at the default kSigmaSunWhite, unit 3 tighter: unit 3 must win even
  // though unit 0 is the lower index.
  this->sun_extra_ = ExtraSunUnit{3, 0.5 * kSigmaSunWhite};

  I64 t = kStartTaiNs;
  const QuatBI q(polaris::math::Quaternion::Identity());
  this->feedMeasurements(t, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_SunUnitSelected(0, 3);

  // Now make the extra unit the worse of the two: selection follows the sigma
  // back to unit 0. A tie would keep the lower index, which is the deterministic
  // half of the rule and is what the equality below pins.
  t += kNsPerSecond / 10;
  this->sun_extra_ = ExtraSunUnit{3, 2.0 * kSigmaSunWhite};
  this->feedMeasurements(t, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_SunUnitSelected(1, 0);

  t += kNsPerSecond / 10;
  this->sun_extra_ = ExtraSunUnit{3, kSigmaSunWhite};
  this->feedMeasurements(t, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_SunUnitSelected(2, 0);
}

void AttitudeEstimatorTester ::testSunSelectionRejectsAnUnusableSigma() {
  // A sigma is wire data the estimator weights with, so a non-positive or
  // non-finite one is gated with the rest (§9.1) rather than trusted because the
  // unit's own valid flag reads true — it would otherwise win every comparison
  // by being the smallest number in the array.
  this->loadIgrf();
  this->setValidParameters();
  this->sun_extra_ = ExtraSunUnit{3, -1.0};

  const QuatBI q(polaris::math::Quaternion::Identity());
  this->feedMeasurements(kStartTaiNs, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);
  ASSERT_TLM_SunUnitSelected(0, 0);
  ASSERT_TLM_SunValid(0, true);
}

void AttitudeEstimatorTester ::testRailedImuIsExcludedAndCostsNothing() {
  // The **median** branch of REQ-ADET-008, run through the component rather than
  // the library: a unit railed at 100 deg/s among three healthy ones must leave
  // the published solution **bit-identical** to the same run with three healthy
  // units. Bit-identical rather than "close": the vote either isolated the fault
  // completely or it did not, and a tolerance would hide the difference.
  //
  // Three units is deliberately *not* the reference vehicle, which flies two
  // (config/spacecraft/leo_smallsat.yaml). The median branch is still the
  // library's design point and stays covered here, so a vehicle that adds a
  // third IMU inherits a tested path rather than an untested one.
  const Eigen::Vector3d rate(0.004, -0.002, 0.003);

  auto flyCycles = [&](AttitudeEstimatorTester& tester) {
    tester.imu_unit_count_ = 3;
    tester.loadIgrf();
    tester.setValidParameters();
    I64 t = kStartTaiNs;
    for (int i = 0; i < 5; ++i) {
      const QuatBI truth(polaris::math::Quaternion::FromAxisAngle(
          rate.normalized(), rate.norm() * static_cast<double>(i) * 0.1));
      tester.feedMeasurements(t, truth, rate, true);
      tester.runCycleAt(t);
      t += kNsPerSecond / 10;
    }
  };

  AttitudeEstimatorTester healthy;
  flyCycles(healthy);
  const QuatF64 reference = healthy.last_estimate_.get_qBodyEci();

  this->imu_fault_ = ImuFaultInjection{2, ImuFault::kRailed, Eigen::Vector3d(1.745, 0.0, 0.0)};
  flyCycles(*this);

  ASSERT_EVENTS_ImuUnitExcluded_SIZE(1);
  ASSERT_EVENTS_ImuUnitExcluded(0, 2, ImuExclusionReason::RATE_LIMIT);
  ASSERT_TLM_ImuContributing(4, 2);
  ASSERT_TLM_ImuExclusionMask(4, 1u << 2);
  ASSERT_TLM_GyroValid(4, true);

  const QuatF64 published = this->last_estimate_.get_qBodyEci();
  for (U32 i = 0; i < 4; ++i) {
    EXPECT_EQ(reference[i], published[i])
        << "component " << i << ": the railed unit moved the published attitude";
  }
}

void AttitudeEstimatorTester ::testNonFiniteImuIsExcludedNotPropagated() {
  // A NaN rate does not fail loudly downstream: it propagates a NaN quaternion
  // behind a validity flag that still reads true. The gate is what stops it, and
  // the event names the data path rather than the vehicle's motion.
  this->imu_unit_count_ = 3;  // the median branch; the pair is covered below
  this->loadIgrf();
  this->setValidParameters();
  this->imu_fault_ = ImuFaultInjection{0, ImuFault::kNotFinite, Eigen::Vector3d::Zero()};

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  I64 t = kStartTaiNs;
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, QuatBI(polaris::math::Quaternion::Identity()), rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  ASSERT_EVENTS_ImuUnitExcluded_SIZE(1);
  ASSERT_EVENTS_ImuUnitExcluded(0, 0, ImuExclusionReason::NOT_FINITE);
  ASSERT_TLM_GyroValid(2, true);
  ASSERT_TLM_ImuContributing(2, 2);
  const Vec3F64 published = this->last_estimate_.get_bodyRateRadps();
  for (U32 i = 0; i < 3; ++i) {
    EXPECT_TRUE(std::isfinite(published[i]));
  }
}

void AttitudeEstimatorTester ::testStaleImuIsAbsentNotExcluded() {
  // Absence is not implausibility. A dropped or stale unit says nothing about
  // whether it is lying, so it must not latch an exclusion the ground then has
  // to reason about — and it must not accumulate re-admission credit either.
  this->imu_unit_count_ = 3;
  this->loadIgrf();
  this->setValidParameters();
  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kStale, Eigen::Vector3d::Zero()};

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  I64 t = kStartTaiNs;
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, QuatBI(polaris::math::Quaternion::Identity()), rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  ASSERT_EVENTS_ImuUnitExcluded_SIZE(0);
  ASSERT_TLM_ImuExclusionMask(2, 0);
  ASSERT_TLM_ImuContributing(2, 2);
  ASSERT_TLM_GyroValid(2, true);
}

void AttitudeEstimatorTester ::testExcludedImuIsReadmittedAfterRecovery() {
  // The automatic re-admission policy end to end (REQ-ADET-009): exclude, serve
  // out kImuReadmitCycles plausible cycles, come back — with the recovery edge
  // reported once, so the ground can pair it with the exclusion.
  this->imu_unit_count_ = 3;
  this->loadIgrf();
  this->setValidParameters();

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;

  this->imu_fault_ = ImuFaultInjection{2, ImuFault::kRailed, Eigen::Vector3d(1.745, 0.0, 0.0)};
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_EVENTS_ImuUnitExcluded_SIZE(1);

  this->imu_fault_ = ImuFaultInjection{};
  for (U32 i = 0; i < kImuReadmitCycles; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_EVENTS_ImuUnitReadmitted_SIZE(1);
  ASSERT_EVENTS_ImuUnitReadmitted(0, 2);
  ASSERT_TLM_ImuExclusionMask(kImuReadmitCycles, 0);
  ASSERT_TLM_ImuContributing(kImuReadmitCycles, 3);

  // And it stays in: the edge is an edge.
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  ASSERT_EVENTS_ImuUnitReadmitted_SIZE(1);
}

void AttitudeEstimatorTester ::testTwoImuDisagreementLeavesNoRate() {
  // Two units, individually plausible, disagreeing by more than the gate, with
  // no fine-mode solution to attribute it: **no body rate**. The deliberate
  // non-monotonicity against the single-unit case — one unit is no evidence of
  // a fault, two disagreeing units are evidence with no attribution.
  this->loadIgrf();  // imu_unit_count_ defaults to the reference vehicle's two
  this->setValidParameters();
  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kOffset, Eigen::Vector3d(0.05, 0.0, 0.0)};

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // Edge-gated: one event for the condition, not one per cycle in it.
  ASSERT_EVENTS_ImuVoteAmbiguous_SIZE(1);
  ASSERT_TLM_GyroValid(2, false);
  ASSERT_TLM_ImuContributing(2, 0);
  // Nothing is latched: latching the wrong unit is worse than carrying an
  // unattributed disagreement.
  ASSERT_TLM_ImuExclusionMask(2, 0);
  ASSERT_EVENTS_ImuUnitExcluded_SIZE(0);
}

void AttitudeEstimatorTester ::testTwoImuDisagreementIsIdentifiedByTheFilter() {
  // **The reference vehicle's branch** (REQ-ADET-008): two IMUs, one drifting by
  // an amount that is individually plausible — it clears the 30 deg/s rate limit
  // easily — so only the *pair* reveals the fault, and only a third information
  // source can say which of the two is lying. That source is the MEKF's
  // propagated body rate, so the test runs with fine mode engaged, which is the
  // condition the identification path depends on.
  this->loadIgrf();
  this->setValidParameters(true, false);

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;

  // Fly healthy until the filter is seeded and publishing a rate the vote can
  // use as its reference. Promotion takes one cycle beyond the coarse
  // acquisition, and the published rate has to be valid *before* the fault.
  for (int i = 0; i < 5; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);
  ASSERT_TLM_ImuContributing(4, 2);
  ASSERT_TLM_GyroValid(4, true);
  this->clearEvents();

  // 0.05 rad/s ~= 2.9 deg/s: about six times the pairwise gate, a tenth of the
  // plausibility limit. Exactly the fault a single-unit gate cannot see.
  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kOffset, Eigen::Vector3d(0.05, 0.0, 0.0)};
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;

  // One cycle is not enough: a latch is permanent until re-admission earns it
  // back, so the same verdict has to repeat before it is spent. The healthy unit
  // is published throughout, so nothing is lost while it confirms.
  ASSERT_EVENTS_ImuUnitExcluded_SIZE(0);
  ASSERT_TLM_GyroValid(this->tlmHistory_GyroValid->size() - 1, true);

  for (U32 i = 0; i < kImuIdentifyConfirmCycles; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // Identified, not merely detected: the offender is named and the gate says the
  // filter is what named it. The ambiguous fallback must NOT have fired — that
  // is the whole difference the fine solution buys.
  ASSERT_EVENTS_ImuUnitExcluded_SIZE(1);
  ASSERT_EVENTS_ImuUnitExcluded(0, 1, ImuExclusionReason::OUTVOTED);
  ASSERT_EVENTS_ImuVoteAmbiguous_SIZE(0);

  // Solution continuity: the healthy unit carries the rate straight through, so
  // there is no loss of attitude and no demotion of the filter that did the
  // identifying.
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  ASSERT_TLM_ImuContributing(this->tlmHistory_ImuContributing->size() - 1, 1);
  ASSERT_TLM_GyroValid(this->tlmHistory_GyroValid->size() - 1, true);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  EXPECT_TRUE(this->last_estimate_.get_rateValid());

  // And the surviving unit's reading is what is published — the faulted one
  // contributed nothing, rather than being averaged in at half weight.
  const Vec3F64 published = this->last_estimate_.get_bodyRateRadps();
  EXPECT_NEAR(published[0], rate.x(), 1.0e-3)
      << "the excluded unit's 0.05 rad/s offset leaked into the published rate";
}

void AttitudeEstimatorTester ::testOutvotedImuDoesNotFlap() {
  // C1, at component level. An outvoted unit is plausible **by construction** —
  // it passed every per-unit gate and lost a comparison — so a re-admission
  // policy that counts plausibility returns it unconditionally, and it loses the
  // same comparison on the next cycle. Over a sustained fault that is a
  // permanent exclude/re-admit flap, with an FDIR event on every lap, which is
  // exactly the "one event on the transition" contract it breaks.
  //
  // 50 cycles is 5 s at 10 Hz, more than sixteen re-admission windows at the
  // harness tuning: a flap could not hide in it.
  this->loadIgrf();
  this->setValidParameters(true, false);

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;
  for (int i = 0; i < 5; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);
  this->clearEvents();

  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kOffset, Eigen::Vector3d(0.05, 0.0, 0.0)};
  for (int i = 0; i < 50; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  ASSERT_EVENTS_ImuUnitExcluded_SIZE(1);
  ASSERT_EVENTS_ImuUnitReadmitted_SIZE(0);
  ASSERT_TLM_ImuExclusionMask(this->tlmHistory_ImuExclusionMask->size() - 1, 1u << 1);
  // And the vehicle kept flying on the surviving unit throughout.
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  ASSERT_TLM_GyroValid(this->tlmHistory_GyroValid->size() - 1, true);
}

void AttitudeEstimatorTester ::testPersistentAmbiguityEscalates() {
  // C4. The per-condition alert is edge-gated, which is right for a transient
  // and wrong for a permanent fault: without an escalation the vehicle would fly
  // rate-less indefinitely after a single warning. The escalation fires at the
  // configured horizon and repeats at that same period, bounded — not per cycle.
  this->loadIgrf();  // coarse only: no filter, so nothing can attribute the pair
  this->setValidParameters();
  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kOffset, Eigen::Vector3d(0.05, 0.0, 0.0)};

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;
  const int cycles = static_cast<int>(3 * kImuAmbiguityEscalateCycles);
  for (int i = 0; i < cycles; ++i) {
    this->feedMeasurements(t, q, rate, true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // The condition alert stays edge-gated...
  ASSERT_EVENTS_ImuVoteAmbiguous_SIZE(1);
  // ...and the escalation carries the persistence, three times over three
  // horizons rather than once per cycle.
  ASSERT_EVENTS_ImuVoteAmbiguousPersistent_SIZE(3);
  // The reported duration is measured from the clock, so it has to match the
  // cycles actually flown: the first escalation is one horizon in.
  EXPECT_NEAR(this->eventHistory_ImuVoteAmbiguousPersistent->at(0).durationSec,
              0.1 * static_cast<double>(kImuAmbiguityEscalateCycles - 1), 0.05);
  EXPECT_EQ(this->eventHistory_ImuVoteAmbiguousPersistent->at(0).cyclesWithoutRate,
            kImuAmbiguityEscalateCycles);

  // Throughout: no rate, but the vector pairs are untouched by a gyro fault, so
  // the vehicle still holds a TRIAD attitude. That is what makes withholding the
  // rate a safe response rather than a self-inflicted outage.
  ASSERT_TLM_GyroValid(this->tlmHistory_GyroValid->size() - 1, false);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  // Nothing latched: detection is not attribution.
  ASSERT_TLM_ImuExclusionMask(this->tlmHistory_ImuExclusionMask->size() - 1, 0);
}

void AttitudeEstimatorTester ::testResetClearsImuExclusions() {
  // RESET_ESTIMATOR is the commanded re-admission path: an operator saying
  // "start over" is different information from a unit having behaved for N
  // cycles, so it drops every latch outright. A really-failed unit re-excludes
  // on its next reading, which tells the ground the fault is persistent.
  this->imu_unit_count_ = 3;
  this->loadIgrf();
  this->setValidParameters();

  const Eigen::Vector3d rate(0.004, -0.002, 0.003);
  const QuatBI q(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;

  this->imu_fault_ = ImuFaultInjection{1, ImuFault::kRailed, Eigen::Vector3d(1.745, 0.0, 0.0)};
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_ImuExclusionMask(0, 1u << 1);

  this->imu_fault_ = ImuFaultInjection{};
  this->sendCmd_RESET_ESTIMATOR(0, 0);
  // A reset re-reads the tuning, which rebuilds the voter — so this pins the
  // *commanded* clear rather than the rebuild's, by asserting the vote is whole
  // again on the very next cycle with no re-admission wait.
  this->feedMeasurements(t, q, rate, true);
  this->runCycleAt(t);
  // Back to three straight away — no re-admission wait, and no recovery edge,
  // because the latch was dropped rather than served out.
  ASSERT_TLM_ImuExclusionMask(1, 0);
  ASSERT_TLM_ImuContributing(1, 3);
  ASSERT_EVENTS_ImuUnitReadmitted_SIZE(0);
}

void AttitudeEstimatorTester ::testNegativeSunSigmaIsRefused() {
  // The sun systematic is composed with a hypot, which squares its arguments —
  // so a sign typo on either good-side term would be absorbed silently, passing
  // both the ordering gate above it and every runtime finiteness check while
  // quietly meaning its own absolute value. There is nowhere downstream that
  // could catch it, so it is caught here.
  this->loadIgrf();
  this->setValidParameters();
  this->paramSet_SigmaSunAlbedoRad(-kSigmaSunAlbedoCorr, Fw::ParamValid::VALID);
  this->component.loadParameters();

  const QuatBI q(polaris::math::Quaternion::Identity());
  this->feedMeasurements(kStartTaiNs, q, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_TLM_EstMode(0, EstimationMode::INVALID);

  // Same for the ephemeris side, on a second component so the first one's
  // edge-gated alert cannot mask it.
  AttitudeEstimatorTester ephem;
  ephem.loadIgrf();
  ephem.setValidParameters();
  ephem.paramSet_SigmaSunEphemPreciseRad(-kSigmaSunEphemPrecise, Fw::ParamValid::VALID);
  ephem.component.loadParameters();
  ephem.feedMeasurements(kStartTaiNs, q, Eigen::Vector3d::Zero(), true);
  ephem.runCycleAt(kStartTaiNs);
  EXPECT_EQ(ephem.eventHistory_ConfigInvalid->size(), 1u)
      << "a negative SigmaSunEphemPreciseRad was accepted";
}

void AttitudeEstimatorTester ::testSunSigmaFollowsTheEphemerisGrade() {
  // The sun *reference* term is the ephemeris', and which value is in force is
  // decided by the grade the query answered at — a fact about the current epoch
  // and the current upload, not a configuration choice. With the DE440 tables
  // covering the epoch the sun direction is arcsecond-class and the term all but
  // vanishes; on the analytic fallback it is 7 mrad and, now that the albedo
  // correction has removed most of the sensor term, more than half the budget.
  //
  // The estimator's covariance floor is built from the composed systematic, so a
  // PRECISE-graded cycle must report a tighter one than a COARSE-graded cycle on
  // otherwise identical measurements. Two components rather than one run, so the
  // comparison is between two vehicles that differ *only* in the served grade.
  // Both components get the albedo tuning and the working albedo geometry, so
  // the *only* difference between them is the served grade.
  this->sun_at_45_from_nadir_ = true;
  this->stub_grade_ = TableGrade::PRECISE;
  this->loadIgrf();
  this->setValidParameters(false, true);

  // **Two** cycles each, not one. The first is the acquisition cycle, where the
  // albedo correction cannot run (no attitude yet to place the Earth with), so
  // the albedo term is the wide uncorrected one and it swamps the ephemeris
  // difference — the two grades would differ by under 4%. On the second cycle
  // the correction runs, the albedo term drops to its corrected value, and the
  // ephemeris term is a comparable share of what is left. That is the case the
  // vehicle actually flies in, and the one worth measuring.
  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(kStartTaiNs);
  AttitudeEstimatorTester degraded;
  degraded.sun_at_45_from_nadir_ = true;
  degraded.stub_grade_ = TableGrade::COARSE;
  degraded.loadIgrf();
  degraded.setValidParameters(false, true);
  I64 t_degraded = kStartTaiNs;
  for (int i = 0; i < 2; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    degraded.feedMeasurements(t_degraded, truth, Eigen::Vector3d::Zero(), true);
    degraded.runCycleAt(t_degraded);
    t_degraded += kNsPerSecond / 10;
  }
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  ASSERT_TRUE(degraded.last_estimate_.get_attitudeValid());
  const double precise_trace = this->publishedCovTrace();
  const double coarse_trace = degraded.publishedCovTrace();

  EXPECT_LT(precise_trace, coarse_trace)
      << "the sun sigma did not follow the ephemeris grade — tables-active and "
         "analytic-fallback cycles reported the same confidence";
  std::printf("[ephemeris grade] cov trace: tables %.6f vs analytic %.6f rad^2 (%.0f%% wider)\n",
              precise_trace, coarse_trace, 100.0 * (coarse_trace / precise_trace - 1.0));

  // And the grade really was the only difference. Asserted on the **domain
  // argument** of the degrade event rather than on RefGrade: the stubbed query
  // drives the EOP and ephemeris domains from one knob, and RefGrade is the
  // worse of the two, so it could not tell which domain moved — which is the
  // whole thing under test.
  ASSERT_EVENTS_ReferenceDegraded_SIZE(0);
  ASSERT_EQ(degraded.eventHistory_ReferenceDegraded->size(), 2u)
      << "the degraded component did not alert on both reference domains";
  bool saw_ephemeris = false;
  for (U32 i = 0; i < degraded.eventHistory_ReferenceDegraded->size(); ++i) {
    if (degraded.eventHistory_ReferenceDegraded->at(i).domain == TableDomain::EPHEMERIS) {
      saw_ephemeris = true;
      EXPECT_EQ(degraded.eventHistory_ReferenceDegraded->at(i).grade, TableGrade::COARSE);
    }
  }
  EXPECT_TRUE(saw_ephemeris) << "no EPHEMERIS-domain degrade — the sun reference term was "
                                "selected from something other than the ephemeris grade";
}

void AttitudeEstimatorTester ::testMissingAlbedoTuningLeavesTheEstimatorRunning() {
  this->sun_at_45_from_nadir_ = true;
  this->loadIgrf();
  this->setValidParameters();  // no albedo parameters at all

  I64 t = kStartTaiNs;
  const QuatBI truth = this->earthInTheSunSensorField(t);
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // Edge-gated: one alert for the condition, not one per cycle in it.
  ASSERT_EVENTS_AlbedoConfigInvalid_SIZE(1);
  // And the vehicle is flying, on the uncorrected budget — which is exactly how
  // it flew before this correction existed. A missing albedo parameter must cost
  // the correction and nothing else.
  ASSERT_EVENTS_ConfigInvalid_SIZE(0);
  ASSERT_EVENTS_AttitudeAcquired_SIZE(1);
  ASSERT_TLM_EstMode(2, EstimationMode::COARSE);
  for (U32 i = 0; i < this->tlmHistory_SunAlbedoCorrection->size(); ++i) {
    EXPECT_TRUE(std::isnan(this->tlmHistory_SunAlbedoCorrection->at(i).arg))
        << "cycle " << i << " corrected on tuning it does not have";
  }
}

// ----------------------------------------------------------------------
// §8.2 star-tracker fusion, the mode ladder, and the residual monitors
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testStarTrackerTakesTheLadderToItsTopRung() {
  this->loadIgrf();
  this->setValidParameters(/*withFine=*/true, /*withAlbedo=*/false, /*withStarTracker=*/true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.3, -0.4, 0.9).normalized(), 1.1));

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  // A tracker seeds the filter directly — no Davenport solve — and the engaging
  // cycle still publishes coarse, on the same one-cycle rule promotion has always
  // followed.
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);
  ASSERT_EVENTS_StConfigInvalid_SIZE(0);

  // Calibrate the second unit against the king: an uncalibrated non-king tracker
  // is deliberately *not* fused (its as-mounted reading carries the two units'
  // bias difference), so without this the vehicle would fly king-only — which is
  // what testUncalibratedSecondTrackerIsNotFused pins.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }

  // The promotion cycle publishes coarse and does not step the filter, so the
  // fine-mode channels are written from the *next* cycle on. Cleared here so the
  // assertions below read that cycle rather than counting history entries.
  this->clearHistory();
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  // The top rung: both trackers accepted, and the transition reported as a
  // **source change** rather than a demotion — the filter kept its state and the
  // published solution never went invalid.
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  ASSERT_TLM_StContributing(0, 2);
  ASSERT_TLM_StValidMask(0, 0x3u);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());

  // **The sun and magnetic pairs are not fused, they are monitored.** Both are
  // present and healthy this cycle, so the monitors run and report residuals —
  // which is the observable difference between "demoted" and "discarded".
  ASSERT_TLM_SunValid(0, true);
  ASSERT_TLM_MagValid(0, true);
  ASSERT_TRUE(std::isfinite(this->tlmHistory_SunResidualRad->at(0).arg));
  ASSERT_TRUE(std::isfinite(this->tlmHistory_MagResidualRad->at(0).arg));
  // Healthy sources must not alert.
  ASSERT_EVENTS_ResidualMonitorAlert_SIZE(0);

  // The covariance a tracker buys, against what the vector pairs buy. Both units
  // are arcsecond-class, so a few cycles of tracker updates must put the reported
  // attitude covariance orders of magnitude below the degree-class SS+MAG one.
  for (int i = 0; i < 20; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    this->clearHistory();
  }
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  const F64 star_trace = this->tlmHistory_FineAttCovTrace->at(0).arg;
  ASSERT_TRUE(std::isfinite(star_trace));
  // (1 mrad)^2 x 3 is already far looser than two AURIGAs justify; the point is
  // the order of magnitude against a sun/magnetometer solution, which sits at
  // ~1e-3 rad^2 (see SunSigmaFollowsTheEphemerisGrade).
  EXPECT_LT(star_trace, 3.0e-6)
      << "tracker fusion did not tighten the covariance — is R being built from "
         "the per-unit boresights?";
  // And the published attitude is genuinely on the trackers: the harness feeds
  // them at truth, so the error must be arcsecond-class rather than degree-class.
  EXPECT_LT(this->publishedErrorRad(truth), 1.0e-4);
}

void AttitudeEstimatorTester ::testStarTrackerLossFallsBackToSunAndMagnetometer() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 2;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 0.2, -0.3).normalized(), 0.4));

  I64 t = kStartTaiNs;
  for (int i = 0; i < 5; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  this->clearHistory();

  // Both trackers lose their solutions — an Earth or Sun keep-out, or a slew past
  // the tracking envelope. Geometry, not a fault.
  this->star_valid_[0] = false;
  this->star_valid_[1] = false;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  // Down one rung, and **nothing else changed**: the filter keeps its state and
  // its covariance, the published solution stays valid, and this is not a
  // demotion. A consumer with a knowledge requirement gates on FineSource; a
  // consumer that only needs *an* attitude sees no event at all.
  ASSERT_TLM_FineSource(0, FineSource::SUN_MAG);
  ASSERT_TLM_StContributing(0, 0);
  ASSERT_TLM_StValidMask(0, 0u);
  ASSERT_EVENTS_FineSourceChanged_SIZE(1);
  ASSERT_EVENTS_FineSourceChanged(0, FineSource::STAR_TRACKER, FineSource::SUN_MAG, 0);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  ASSERT_TLM_EstMode(0, EstimationMode::FINE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
  // The monitors stop reporting, because monitoring a source the filter is now
  // *using* would be circular — the residual would be small because the update
  // made it small.
  ASSERT_TRUE(std::isnan(this->tlmHistory_SunResidualRad->at(0).arg));

  // And it climbs back when a tracker returns.
  this->clearHistory();
  this->star_valid_[0] = true;
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  ASSERT_TLM_StContributing(0, 1);
  ASSERT_EVENTS_FineSourceChanged_SIZE(1);
}

void AttitudeEstimatorTester ::testDriftedSunSensorRaisesTheResidualMonitor() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 2;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.1, 1.0, 0.2).normalized(), 0.9));

  I64 t = kStartTaiNs;
  for (int i = 0; i < 4; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  this->clearHistory();

  // A sun sensor 20 degrees out — far past the 0.15 rad monitor threshold, and a
  // fault that in the SS+MAG mode would merely have been down-weighted into the
  // solution. With a tracker fused it is *observable*, which is the whole
  // argument for demoting the pair to a monitor rather than discarding it.
  this->sun_body_error_rad_ = 20.0 * M_PI / 180.0;
  for (U32 i = 0; i < kMonitorAlertCycles; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_EVENTS_ResidualMonitorAlert_SIZE(1);
  EXPECT_EQ(this->eventHistory_ResidualMonitorAlert->at(0).monitor, ResidualMonitor::SUN);
  // The residual is the injected offset. Compared with a tolerance rather than
  // for equality: it is a measured angle, not a copied constant.
  EXPECT_NEAR(this->eventHistory_ResidualMonitorAlert->at(0).residualRad, 20.0 * M_PI / 180.0,
              1.0e-6);
  EXPECT_EQ(this->eventHistory_ResidualMonitorAlert->at(0).thresholdRad, kMonitorSunResidualRad);
  // And the solution is untouched by it: the drifted sensor is not being fused,
  // so it cannot pull the attitude.
  EXPECT_LT(this->publishedErrorRad(truth), 1.0e-4);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);

  // Recovery closes the condition, so the ground does not have to notice that the
  // alerts stopped.
  this->clearHistory();
  this->sun_body_error_rad_ = 0.0;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_ResidualMonitorCleared_SIZE(1);
  ASSERT_EVENTS_ResidualMonitorCleared(0, ResidualMonitor::SUN);
}

void AttitudeEstimatorTester ::testMissingStarTrackerTuningCapsTheLadder() {
  this->loadIgrf();
  // Fine tuning present, tracker tuning absent — the fourth independent gate.
  this->setValidParameters(/*withFine=*/true, /*withAlbedo=*/false, /*withStarTracker=*/false);
  this->star_unit_count_ = 2;

  const QuatBI truth(polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), 0.3));

  I64 t = kStartTaiNs;
  for (int i = 0; i < 3; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  // One alert, edge-gated. Asserted before the history is cleared, since the
  // alert fired on the first cycle.
  ASSERT_EVENTS_StConfigInvalid_SIZE(1);
  ASSERT_EVENTS_ConfigInvalid_SIZE(0);
  ASSERT_EVENTS_FineConfigInvalid_SIZE(0);

  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  // The vehicle is still flying its SS+MAG fine mode — the accuracy it had before
  // trackers were fused, which is a working vehicle.
  ASSERT_TLM_FineSource(0, FineSource::SUN_MAG);
  ASSERT_TLM_StContributing(0, 0);
  // No tracker may be gathered without a configured boresight.
  ASSERT_TLM_StValidMask(0, 0u);
  ASSERT_TLM_EstMode(0, EstimationMode::FINE);
  ASSERT_TRUE(this->last_estimate_.get_attitudeValid());
}

void AttitudeEstimatorTester ::testStarTrackerSeedsFineModeWithoutTheVectorPairs() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 1;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.4, 0.4, 0.8).normalized(), 2.0));

  // Eclipse: no sun pair at all, so there is no Davenport seed to be had and no
  // coarse attitude to require. This is the case a vector seed structurally cannot
  // cover, and the reason a tracker promotes on its own.
  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), /*sunInView=*/false);
  this->runCycleAt(t);
  ASSERT_EVENTS_FineModeEngaged_SIZE(1);
  ASSERT_EVENTS_FineInitFailed_SIZE(0);

  this->clearHistory();
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), false);
  this->runCycleAt(t);
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  ASSERT_TLM_EstMode(0, EstimationMode::FINE);
  EXPECT_LT(this->publishedErrorRad(truth), 1.0e-4);
  // The coarse chain never acquired — there was never a sun pair — which is
  // exactly the state the tracker seed exists to fly out of.
  ASSERT_TLM_SunValid(0, false);
}

// ----------------------------------------------------------------------
// §8.2 commanded inter-tracker alignment
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testInterTrackerAlignmentCollectsFitsAndApplies() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;
  // The second unit is mounted ~0.06 degrees off the king — a plausible
  // integration tolerance, and tens of times the units' own noise. The king
  // carries a bias too, and nothing removes it: its mounting *is* the body frame.
  const Eigen::Vector3d misalignment(1.0e-3, -4.0e-4, 6.0e-4);
  this->star_error_[1] = misalignment;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.2, -0.9, 0.4).normalized(), 1.3));

  I64 t = kStartTaiNs;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StAlignState(0, StAlignState::IDLE);
  this->clearHistory();

  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  ASSERT_CMD_RESPONSE_SIZE(1);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_START, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_StAlignStarted_SIZE(1);
  this->clearHistory();

  // Collection is a **tap**: the estimator keeps running and the published
  // solution is unaffected until a fit is applied.
  for (U32 i = 0; i < kStAlignMinSamples; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
  }

  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_StAlignComplete_SIZE(1);
  ASSERT_EVENTS_StAlignRejected_SIZE(0);
  const F64 fitted = this->eventHistory_StAlignComplete->at(0).misalignRad;
  // What was found must be what was injected: the pairs are noise-free here, so
  // this is the algebra check — a composition-order error would recover the
  // inverse rotation, which is still a plausible-looking small angle.
  EXPECT_NEAR(fitted, misalignment.norm(), 1.0e-9);
  // Only the second unit carries a correction, and the king's slot is never
  // fitted — it has no alignment to estimate, its mounting *is* the frame.
  const FwSizeType last = this->tlmHistory_StAlignMask->size() - 1;
  EXPECT_EQ(this->tlmHistory_StAlignMask->at(last).arg, 0x2u);
  EXPECT_EQ(this->tlmHistory_StAlignState->at(last).arg, StAlignState::APPLIED);
  EXPECT_LT(this->tlmHistory_StAlignResidualRad->at(last).arg, 1.0e-7);

  // And the correction is actually applied: with both units now stated in the
  // king's frame, the solution follows the king rather than splitting the
  // difference with a unit 0.06 degrees away.
  this->clearHistory();
  for (int i = 0; i < 20; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    this->clearHistory();
  }
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  EXPECT_LT(this->publishedErrorRad(truth), 0.1 * misalignment.norm())
      << "the fitted alignment is not being applied to the second tracker";
}

void AttitudeEstimatorTester ::testInterTrackerAlignmentRefusesTheKingAndBadCommands() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 2;
  // One cycle first: the tracker tuning is read on the rate-group cycle (it is a
  // per-cycle gate), so a command sent before the estimator has ever run would see
  // an unconfigured tracker set and refuse with UNIT for the wrong reason. On the
  // vehicle the rate group has been running since boot.
  const QuatBI settle(polaris::math::Quaternion::Identity());
  this->feedMeasurements(kStartTaiNs, settle, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs);
  this->clearHistory();

  // (a) No StAlign* tuning yet: a command-time refusal that costs the vehicle
  //     nothing, because calibration is a commanded activity rather than a flight
  //     function.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_START, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_StAlignRejected_SIZE(1);
  ASSERT_EVENTS_StAlignRejected(0, 1, StAlignRejectReason::CONFIG, 0);

  this->setStAlignParameters();
  this->clearHistory();

  // (b) The **king**. Not an error of degree: it is a request to estimate a
  //     rotation that is zero by definition, because that unit's mounting is the
  //     body frame.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 0, kStAlignMinSamples);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_START, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_StAlignRejected_SIZE(1);
  ASSERT_EVENTS_StAlignRejected(0, 0, StAlignRejectReason::UNIT, 0);
  this->clearHistory();

  // (c) A unit with no configured boresight — "not installed" — which cannot be
  //     fused either, so calibrating it would fit a correction nothing applies.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 5, kStAlignMinSamples);
  ASSERT_EVENTS_StAlignRejected(0, 5, StAlignRejectReason::UNIT, 0);
  this->clearHistory();

  // (d) A sample count below the configured floor. Saying so now costs the
  //     operator the command rather than the whole collection window.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples - 1);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_START, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_StAlignRejected_SIZE(1);
  EXPECT_EQ(this->eventHistory_StAlignRejected->at(0).unit, 1);
  EXPECT_EQ(this->eventHistory_StAlignRejected->at(0).reason, StAlignRejectReason::SAMPLES);

  // Through all four, no window ever opened and nothing was applied.
  this->clearHistory();
  this->feedMeasurements(kStartTaiNs + kNsPerSecond / 10, settle, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(kStartTaiNs + kNsPerSecond / 10);
  ASSERT_TLM_StAlignState(0, StAlignState::IDLE);
  ASSERT_TLM_StAlignMask(0, 0u);
}

void AttitudeEstimatorTester ::testInterTrackerAlignmentAbortAndClear() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;
  this->star_error_[1] = Eigen::Vector3d(8.0e-4, 0.0, 0.0);

  const QuatBI truth(polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), 0.5));
  I64 t = kStartTaiNs;
  // One cycle first, so the tracker tuning has been read (see the refusal case).
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  this->clearHistory();

  // --- Abort discards the window without fitting --------------------------
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i < kStAlignMinSamples / 2; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    this->clearHistory();
  }
  this->sendCmd_ST_ALIGN_CAL_ABORT(0, 0);
  ASSERT_EVENTS_StAlignAborted_SIZE(1);
  ASSERT_EVENTS_StAlignComplete_SIZE(0);
  this->clearHistory();

  // Idempotent: aborting with nothing open is an operator making sure.
  this->sendCmd_ST_ALIGN_CAL_ABORT(0, 0);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_ABORT, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_StAlignAborted_SIZE(0);
  this->clearHistory();

  // --- Fit, then clear ----------------------------------------------------
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    this->clearHistory();
  }
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StAlignMask(0, 0x2u);
  this->clearHistory();

  this->sendCmd_ST_ALIGN_CAL_CLEAR(0, 0, 1);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_CLEAR, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_StAlignCleared_SIZE(1);
  this->clearHistory();
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StAlignMask(0, 0u);
  ASSERT_TLM_StAlignState(0, StAlignState::IDLE);

  // An out-of-range index is refused rather than silently ignored.
  this->clearHistory();
  this->sendCmd_ST_ALIGN_CAL_CLEAR(0, 0, 99);
  ASSERT_CMD_RESPONSE(0, AttitudeEstimator::OPCODE_ST_ALIGN_CAL_CLEAR, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);

  // --- RESET_ESTIMATOR does both, on the same "the solution is suspect" rule
  //     the magnetometer calibration follows.
  this->clearHistory();
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    this->clearHistory();
  }
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StAlignMask(0, 0x2u);

  this->clearHistory();
  this->sendCmd_RESET_ESTIMATOR(0, 0);
  ASSERT_EVENTS_StAlignCleared_SIZE(1);
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StAlignMask(0, 0u);
}

// ----------------------------------------------------------------------
// §8.2 multi-magnetometer voting and the sun cross-unit check
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::testImplausibleMagnetometerIsExcludedAndCostsNothing() {
  this->loadIgrf();
  this->setValidParameters();
  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 1.0, 0.0).normalized(), 0.6));

  // Settle on two healthy units. The truth attitude is fixed and there is no gyro
  // bias, so the coarse blend converges and the published error stops moving —
  // which is what makes the comparison below a statement about the vote rather
  // than about the transient.
  I64 t = kStartTaiNs;
  for (int i = 0; i < 40; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    this->clearHistory();
  }
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  const double healthy_error = this->publishedErrorRad(truth);
  ASSERT_TLM_MagContributing(0, 2);
  ASSERT_TLM_MagUnitSelected(0, 0);
  this->clearHistory();

  // Unit 1 dies: it reads near zero, which the IGRF-magnitude band catches before
  // any combination — the reading a mean would have split the difference with,
  // and the one a fixed full-scale check would have waved through.
  this->mag_fault_index_ = 1;
  this->mag_fault_scale_ = 0.05;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);

  ASSERT_EVENTS_MagUnitExcluded_SIZE(1);
  ASSERT_EVENTS_MagUnitExcluded(0, 1, MagExclusionReason::FIELD_MAGNITUDE);
  // The surviving unit carries the pair, and the attitude is untouched: a
  // single-unit fault must cost accuracy nothing at all, which is the whole
  // property the second magnetometer is carried for.
  ASSERT_TLM_MagValid(0, true);
  ASSERT_TLM_MagContributing(0, 1);
  ASSERT_TLM_MagExclusionMask(0, 0x2u);
  ASSERT_TLM_MagUnitSelected(0, 0);
  EXPECT_NEAR(this->publishedErrorRad(truth), healthy_error, 1.0e-9)
      << "an excluded magnetometer moved the published attitude";

  // Latched, not re-entering on its next plausible sample; then re-admitted after
  // the configured consecutive count.
  this->clearHistory();
  this->mag_fault_index_ = -1;
  this->mag_fault_scale_ = 1.0;
  for (U32 i = 0; i < kMagReadmitCycles - 1; ++i) {
    t += kNsPerSecond / 10;
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    // Latched: a plausible sample alone must not re-admit it.
    ASSERT_TLM_MagContributing(i, 1);
  }
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_MagUnitReadmitted_SIZE(1);
  ASSERT_EVENTS_MagUnitReadmitted(0, 1);
}

void AttitudeEstimatorTester ::testTwoMagnetometerDisagreementLeavesNoMagneticPair() {
  this->loadIgrf();
  // No fine tuning, so there is no filter solution and — more to the point — no
  // attitude good enough to rotate the modelled field with. Two plausible units
  // disagreeing then has nothing to attribute it, which is the honest outcome.
  this->setValidParameters();
  this->mag_fault_index_ = 1;
  // 10 uT: twice the disagreement gate, comfortably inside the magnitude band, so
  // **no per-unit gate can see it** and only the comparison can.
  this->mag_fault_offset_t_ = Eigen::Vector3d(0.0, 10.0e-6, 0.0);

  const QuatBI truth(polaris::math::Quaternion::Identity());
  I64 t = kStartTaiNs;
  for (int i = 0; i < 4; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // Reported once, on the edge, and nothing latched: detection is not
  // attribution, and excluding a unit on a guess would spend the remaining
  // redundancy.
  ASSERT_EVENTS_MagVoteAmbiguous_SIZE(1);
  ASSERT_EVENTS_MagUnitExcluded_SIZE(0);
  ASSERT_TLM_MagExclusionMask(3, 0u);
  ASSERT_TLM_MagContributing(3, 0);
  // The cost is bounded to the magnetic pair — cheaper than the IMU equivalent,
  // which loses the body rate. The gyro still propagates and the rate still flows.
  ASSERT_TLM_MagValid(3, false);
  ASSERT_TLM_GyroValid(3, true);
  ASSERT_TRUE(this->last_estimate_.get_rateValid());

  // Recovery clears the condition and the pair comes back.
  this->clearHistory();
  this->mag_fault_index_ = -1;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_MagValid(0, true);
  ASSERT_TLM_MagContributing(0, 2);
}

void AttitudeEstimatorTester ::testSunCrossUnitCheckAlertsAndOverrides() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 1;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.5, 0.5, 0.7).normalized(), 0.8));

  // Two units see the Sun. The **selected** one is the confidently-wrong one: it
  // reports the smaller sigma, so the σ-ordered selector takes it, and nothing
  // else on the vehicle would notice. This is the gap the cross-check closes.
  this->sun_extra_.index = 1;
  this->sun_extra_.sigma_rad = 10.0 * kSigmaSunWhite;  // the runner-up, honest but wider
  this->sun_body_error_rad_ = 20.0 * M_PI / 180.0;     // injected into the *selected* unit only
  this->sun_extra_truthful_ = true;

  I64 t = kStartTaiNs;
  for (int i = 0; i < 4; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  // Detection needs no attitude: it is a sensor-versus-sensor comparison, which
  // is what makes it the one cross-check available in Safe mode.
  ASSERT_TRUE(std::isfinite(this->tlmHistory_SunCrossUnitRad->at(0).arg));
  ASSERT_EVENTS_ResidualMonitorAlert_SIZE(1);
  EXPECT_EQ(this->eventHistory_ResidualMonitorAlert->at(0).monitor,
            ResidualMonitor::SUN_CROSS_UNIT);
  EXPECT_NEAR(this->eventHistory_ResidualMonitorAlert->at(0).residualRad, 20.0 * M_PI / 180.0,
              1.0e-6);
  EXPECT_EQ(this->eventHistory_ResidualMonitorAlert->at(0).thresholdRad, kMonitorSunCrossUnitRad);

  // Resolution needs one, and it has one: with the tracker fused, the runner-up
  // agrees with the solution and the selected unit does not, so the estimator
  // uses the runner-up. Nothing is latched — the override is re-decided every
  // cycle from the current evidence.
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_SunUnitOverridden_SIZE(1);
  EXPECT_EQ(this->eventHistory_SunUnitOverridden->at(0).selected, 0);
  EXPECT_EQ(this->eventHistory_SunUnitOverridden->at(0).used, 1);
  EXPECT_NEAR(this->eventHistory_SunUnitOverridden->at(0).angleRad, 20.0 * M_PI / 180.0, 1.0e-6);
  ASSERT_TLM_SunUnitSelected(0, 1);

  // The unit recovering simply stops it, with no command and no re-admission
  // policy to serve out.
  this->clearHistory();
  this->sun_body_error_rad_ = 0.0;
  t += kNsPerSecond / 10;
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_EVENTS_SunUnitOverridden_SIZE(0);
  ASSERT_TLM_SunUnitSelected(0, 0);
  ASSERT_EVENTS_ResidualMonitorCleared_SIZE(1);
}

void AttitudeEstimatorTester ::testBadStarTrackerIsIsolatedWithoutDemotingTheMode() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.6, 0.1, 0.8).normalized(), 1.0));
  I64 t = kStartTaiNs;

  // Calibrate unit 1 first, so it is *fused* and can then go bad — otherwise it
  // would simply never be gathered (the C2 rule) and this would test nothing.
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    this->clearHistory();
  }
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_StContributing(0, 2);
  this->clearHistory();

  // Unit 1 goes 5 degrees out — far past the chi-square(3) gate against an
  // arcsecond-class R, so every one of its updates is rejected while the king's
  // are accepted.
  this->star_error_[1] = Eigen::Vector3d(0.0, 5.0 * M_PI / 180.0, 0.0);
  for (U32 i = 0; i < kMekfNisStreak + 2; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // **The unit is isolated; the mode is not.** This is the whole point of a
  // per-unit streak: a cycle-global one would have demoted here, dropped the
  // filter, re-promoted off the same bad unit and flapped at the streak period.
  ASSERT_EVENTS_StUnitExcluded_SIZE(1);
  EXPECT_EQ(this->eventHistory_StUnitExcluded->at(0).unit, 1);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  this->clearHistory();

  // Still on the top rung, now on the king alone, and still accurate — the king
  // is at truth, so the solution must be too.
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  ASSERT_TLM_StContributing(0, 1);
  EXPECT_LT(this->publishedErrorRad(truth), 1.0e-3);
  // Edge-gated: a permanently bad unit costs one event, not one per cycle.
  ASSERT_EVENTS_StUnitExcluded_SIZE(0);

  // Recovery is automatic and judged on the criterion that excluded it —
  // agreement with the solution the *other* tracker built.
  this->clearHistory();
  this->star_error_[1].setZero();
  for (U32 i = 0; i <= kMonitorAlertCycles; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  ASSERT_EVENTS_StUnitReadmitted_SIZE(1);
  EXPECT_EQ(this->eventHistory_StUnitReadmitted->at(0).unit, 1);
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StContributing(0, 2);
}

// ----------------------------------------------------------------------
// The off-rung tracker arbitration (§8.2, §9.2; REQ-FDIR-013)
// ----------------------------------------------------------------------

void AttitudeEstimatorTester ::armOffRungArbitration(
    I64& t,
    const polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI>& truth,
    double trackerErrorRad) {
  // Promote on the vector pairs *first*, with no tracker in sight — the state the
  // whole arbitration exists for, and the one a sunlit boot always lands in
  // because the Davenport seed closes within a cycle while a tracker needs
  // seconds to acquire.
  // Long enough for the filter's covariance to converge well below the coarse
  // systematic floor — which is the whole precondition: the arbitration only
  // arms once the filter is confident enough to reject a tracker it should not.
  this->star_valid_[0] = false;
  for (int i = 0; i < 30; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  // The telemetry history indexes from the first cycle, so the state under test
  // is read on a cleared history rather than at index 0 of the whole run.
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_FineSource(0, FineSource::SUN_MAG);
  this->clearHistory();

  // Now the tracker arrives, disagreeing with the filter by more than an
  // arcsecond-class R allows, so every update is gate-rejected and the streak
  // arms the arbitration.
  this->star_error_[0] = Eigen::Vector3d(0.0, trackerErrorRad, 0.0);
  this->star_valid_[0] = true;
  for (U32 i = 0; i < kMekfNisStreak + 1; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
}

void AttitudeEstimatorTester ::testOffRungTrackerVerdictFollowsTheCoarseGate() {
  // **Just inside the boundary, and self-calibrating on it.** The event carries
  // the statistic it was decided on, so the test asserts the *contract* — adopted
  // iff d² is inside the gate — rather than an angle, which would move with the
  // coarse covariance and therefore with the tuning. At this harness's tuning the
  // boundary sits between 6° (d² = 15.9) and 9° (d² = 35.6), bracketing
  // χ²₃(0.999) = 16.266; the sibling test below takes the other side.
  //
  // The window exists at all because the filter's covariance converges *below*
  // the coarse chain's, which holds a systematic floor: under ~2° the filter
  // accepts the tracker and there is nothing to arbitrate.
  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.6, 0.1, 0.8).normalized(), 1.0));
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 1;
  I64 t = kStartTaiNs;
  this->armOffRungArbitration(t, truth, 6.0 * M_PI / 180.0);

  ASSERT_EVENTS_FineReseededFromStarTracker_SIZE(1);
  const auto& adopted = this->eventHistory_FineReseededFromStarTracker->at(0);
  EXPECT_EQ(adopted.unit, 0);
  EXPECT_LE(adopted.mahalanobis, kCoarseAgreementGate)
      << "adopted a tracker outside the gate it is supposed to be decided by";
  EXPECT_NEAR(adopted.separationRad, 6.0 * M_PI / 180.0, 2.0e-3);
  ASSERT_EVENTS_FineTrackerAdoptionRefused_SIZE(0);
  ASSERT_EVENTS_StUnitExcluded_SIZE(0);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);

  // The adoption took effect: the ladder is on its top rung, and the published
  // solution is the tracker's rather than the vector pairs'.
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  ASSERT_TLM_StContributing(0, 1);
}

void AttitudeEstimatorTester ::testOffRungTrackerOutsideTheGateIsRefusedNotLatched() {
  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.6, 0.1, 0.8).normalized(), 1.0));
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->star_unit_count_ = 1;
  I64 t = kStartTaiNs;
  // 9 degrees: **just** outside, d² = 35.6 against the 16.266 gate, so the pair of
  // tests brackets the boundary rather than sitting at opposite ends of the range.
  this->armOffRungArbitration(t, truth, 9.0 * M_PI / 180.0);

  ASSERT_EVENTS_FineTrackerAdoptionRefused_SIZE(1);
  const auto& refused = this->eventHistory_FineTrackerAdoptionRefused->at(0);
  EXPECT_EQ(refused.unit, 0);
  EXPECT_GT(refused.mahalanobis, kCoarseAgreementGate)
      << "refused a tracker the gate should have adopted";
  ASSERT_EVENTS_FineReseededFromStarTracker_SIZE(0);

  // **C1: nothing is latched.** An exclusion decided on coarse agreement whose
  // re-admission is decided on *fine* agreement (readmitStarTrackers) can never be
  // served, so the unit must stay a candidate — which is also what lets it be
  // adopted later from a better reference.
  ASSERT_EVENTS_StUnitExcluded_SIZE(0);

  // The mode is untouched: the vector pairs are healthy and they are not what
  // went wrong.
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_FineSource(0, FineSource::SUN_MAG);

  // Bounded cadence, not once and not per cycle: the condition is permanent, so
  // the report repeats every MonitorAlertCycles and no more often.
  this->clearHistory();
  const U32 cycles = 3 * kMonitorAlertCycles;
  for (U32 i = 0; i < cycles; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  const U32 refusals = this->eventHistory_FineTrackerAdoptionRefused->size();
  EXPECT_GE(refusals, 1u) << "a permanently unusable tracker went silent";
  EXPECT_LE(refusals, cycles / kMonitorAlertCycles + 1)
      << "the refusal is not on its bounded cadence: " << refusals << " in " << cycles << " cycles";
  ASSERT_EVENTS_StUnitExcluded_SIZE(0);
}

void AttitudeEstimatorTester ::testOffRungArbitrationPicksTheTrackerTheCoarseFixSupports() {
  // Two calibrated trackers, both rejected by the filter, disagreeing with each
  // other: one near the coarse solution and one far from it. The arbitration has
  // to take the one the coarse fix vouches for and leave the other alone — the
  // selection SITL cannot enumerate cheaply, because it needs two units to
  // disagree by a controlled amount.
  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.6, 0.1, 0.8).normalized(), 1.0));
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;

  // Calibrate unit 1 so it is a fusion candidate at all (the launch-state rule),
  // with both units still healthy so the fit is clean.
  I64 t = kStartTaiNs;
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  this->clearHistory();

  // Drop back to the vector pairs, then bring both units back disagreeing.
  this->sendCmd_RESET_ESTIMATOR(0, 0);
  this->star_valid_[0] = false;
  this->star_valid_[1] = false;
  for (int i = 0; i < 30; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_FineSource(0, FineSource::SUN_MAG);
  this->clearHistory();

  this->star_error_[0] = Eigen::Vector3d(0.0, 6.0 * M_PI / 180.0, 0.0);   // inside
  this->star_error_[1] = Eigen::Vector3d(0.0, 15.0 * M_PI / 180.0, 0.0);  // outside
  this->star_valid_[0] = true;
  this->star_valid_[1] = true;
  for (U32 i = 0; i < kMekfNisStreak + 1; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }

  // The king was adopted; the far unit was neither adopted nor latched out.
  ASSERT_EVENTS_FineReseededFromStarTracker_SIZE(1);
  EXPECT_EQ(this->eventHistory_FineReseededFromStarTracker->at(0).unit, 0);
  ASSERT_EVENTS_StUnitExcluded_SIZE(0);
  ASSERT_EVENTS_FineModeDemoted_SIZE(0);
  ASSERT_EVENTS_AttitudeLost_SIZE(0);

  // And the solution really is the good tracker's, not the bad one's.
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  // The solution is the *adopted* unit's, not the refused one's: about 6 deg from
  // truth (unit 0's error) and nowhere near 15 (unit 1's). Bracketed on both sides
  // deliberately — an upper bound alone would also pass if the arbitration had
  // quietly kept the vector solution.
  const double published_deg = this->publishedErrorRad(truth) * 180.0 / M_PI;
  EXPECT_GT(published_deg, 4.0) << "the solution is not the tracker's at all";
  EXPECT_LT(published_deg, 9.0) << "the arbitration adopted the far unit";
}

void AttitudeEstimatorTester ::testUncalibratedSecondTrackerIsNotFused() {
  this->loadIgrf();
  this->setValidParameters(true, false, true);
  this->setStAlignParameters();
  this->star_unit_count_ = 2;
  // A plausible integration tolerance on the second unit. Uncalibrated, this is a
  // *systematic* the configured 21.5 arcsec sigma does not cover.
  this->star_error_[1] = Eigen::Vector3d(1.0e-3, -4.0e-4, 6.0e-4);

  const QuatBI truth(
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d(0.2, 0.9, 0.3).normalized(), 0.8));
  I64 t = kStartTaiNs;
  for (int i = 0; i < 4; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
  }
  this->clearHistory();
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;

  // **King only.** Both units are delivering valid solutions — StValidMask says so
  // — but the uncalibrated one is not fused, so the vehicle flies the launch state
  // honestly rather than fusing a systematic at a white-noise sigma.
  // Both units *solved* — StValidMask says so — but only the king is fused. That
  // gap between the two channels is exactly what the ground reads to tell "the
  // unit stopped solving" from "the unit is not trusted yet".
  ASSERT_TLM_StValidMask(0, 0x3u);
  ASSERT_TLM_StContributing(0, 1);
  ASSERT_TLM_FineSource(0, FineSource::STAR_TRACKER);
  EXPECT_LT(this->publishedErrorRad(truth), 1.0e-4);

  // Calibrate it, and it joins.
  this->clearHistory();
  this->sendCmd_ST_ALIGN_CAL_START(0, 0, 1, kStAlignMinSamples);
  for (U32 i = 0; i <= kStAlignMinSamples; ++i) {
    this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
    this->runCycleAt(t);
    t += kNsPerSecond / 10;
    this->clearHistory();
  }
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  t += kNsPerSecond / 10;
  ASSERT_TLM_StValidMask(0, 0x3u);
  ASSERT_TLM_StContributing(0, 2);

  // And clearing the calibration drops it again, which is what makes
  // ST_ALIGN_CAL_CLEAR a safe command rather than one that silently degrades the
  // solution to an uncalibrated pair.
  this->clearHistory();
  this->sendCmd_ST_ALIGN_CAL_CLEAR(0, 0, 1);
  this->feedMeasurements(t, truth, Eigen::Vector3d::Zero(), true);
  this->runCycleAt(t);
  ASSERT_TLM_StContributing(0, 1);
  ASSERT_TLM_StValidMask(0, 0x3u);
}

}  // namespace flight

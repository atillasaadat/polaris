// ======================================================================
// \title  OrbitEstimatorTester.cpp
// \brief  Component unit tests for OrbitEstimator (design doc §23.1)
//
// The truth is the filter's own force model propagated noise-free from the
// circular reference orbit, presented back as receiver fixes: what is under
// test is the component's seam — seeding, the propagate-then-ingest cycle,
// the coast horizon, refusals, reset and the published product — and a truth
// the filter cannot disagree with keeps every gate out of those assertions.
// The filter's statistics are pinned in tests/unit and tests/mc.
// ======================================================================

#include "OrbitEstimatorTester.hpp"

#include <cmath>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "time/timescales.hpp"

namespace flight {

namespace {

namespace pg = polaris::gnc;
namespace pc = polaris::constants;
namespace pt = polaris::time;
namespace pm = polaris::math;

constexpr I64 kNsPerSecond = 1'000'000'000LL;
constexpr I64 kStartTaiNs = 1'770'000'000LL * kNsPerSecond;

// The campaign's reference tuning (tests/mc/orbit_od_mc.cpp filterConfig()).
constexpr U32 kDegree = 8;
constexpr F64 kBallisticCoeff = 2.2 * 0.06 / 12.0;
constexpr F64 kDragRefDensity = 3.725e-12;
constexpr F64 kDragRefAltitude = 400.0e3;
constexpr F64 kDragScaleHeight = 58'515.0;
constexpr F64 kAccelPsd = 1.8e-7;
constexpr F64 kNisGate = 16.266;
constexpr F64 kMaxCoastS = 300.0;
constexpr F64 kMaxDegradedCoastS = 900.0;  // short, so the drop is reachable at 1 Hz
constexpr F64 kMaxAccelAgeS = 0.5;
constexpr U32 kStatusPeriodCycles = 100;
constexpr F64 kMaxDtS = 60.0;
constexpr F64 kMaxStepS = 10.0;
constexpr F64 kMaxFixLatencyS = 0.2;
constexpr F64 kMinRadiusM = 6.5e6;
constexpr F64 kMaxRadiusM = 8.0e6;
constexpr F64 kBackupPeriodS = 60.0;

// The reference receiver's reported accuracies (NovAtel OEM7600 catalog entry).
constexpr F64 kPosSigmaH = 1.0;
constexpr F64 kPosSigmaV = 1.5;
constexpr F64 kVelSigma = 0.03;

constexpr double kAltitudeM = 400.0e3;
constexpr double kInclinationRad = 51.6 * M_PI / 180.0;

polaris::frames::EopValue stubEop() {
  return polaris::frames::EopValue{0.0, 0.0, 0.0};
}

pg::OrbitOdConfig referenceConfig() {
  pg::OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.geopotential_degree = kDegree;
  cfg.geopotential_order = kDegree;
  cfg.drag_ballistic_coeff_m2_per_kg = kBallisticCoeff;
  cfg.drag_ref_density_kg_m3 = kDragRefDensity;
  cfg.drag_ref_altitude_m = kDragRefAltitude;
  cfg.drag_scale_height_m = kDragScaleHeight;
  cfg.accel_psd_m2_per_s3 = kAccelPsd;
  cfg.position_nis_gate = kNisGate;
  cfg.velocity_nis_gate = kNisGate;
  cfg.max_coast_s = 1.0e9;  // the truth never expires
  cfg.max_degraded_coast_s = 1.0e9;
  cfg.max_dt_s = kMaxDtS;
  cfg.max_step_s = kMaxStepS;
  cfg.max_fix_latency_s = kMaxFixLatencyS;
  cfg.min_radius_m = kMinRadiusM;
  cfg.max_radius_m = kMaxRadiusM;
  return cfg;
}

Vec3F64 toVec3(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

Eigen::Vector3d toEigen(const Vec3F64& v) {
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

}  // namespace

// ----------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------

OrbitEstimatorTester ::OrbitEstimatorTester()
    : OrbitEstimatorGTestBase("Tester", MAX_HISTORY_SIZE),
      component("OrbitEstimator"),
      truth_(referenceConfig()) {
  this->initComponents();
  this->connectPorts();
}

OrbitEstimatorTester ::~OrbitEstimatorTester() {}

// ----------------------------------------------------------------------
// Port handlers
// ----------------------------------------------------------------------

bool OrbitEstimatorTester ::from_getEopAt_handler(FwIndexType portNum, I64 taiNs,
                                                  EopSample& sample) {
  if (!this->eop_available_) {
    return false;
  }
  const polaris::frames::EopValue eop = stubEop();
  sample = EopSample(eop.ut1_minus_tai, eop.xp_arcsec, eop.yp_arcsec, TableGrade::PRECISE);
  return true;
}

void OrbitEstimatorTester ::from_orbitStateOut_handler(FwIndexType portNum,
                                                       const OrbitEstimate& estimate) {
  this->last_estimate_ = estimate;
  ++this->estimate_count_;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void OrbitEstimatorTester ::setValidParameters() {
  this->paramSet_GeopotentialDegree(kDegree, Fw::ParamValid::VALID);
  this->paramSet_GeopotentialOrder(kDegree, Fw::ParamValid::VALID);
  this->paramSet_DragBallisticCoeffM2PerKg(kBallisticCoeff, Fw::ParamValid::VALID);
  this->paramSet_DragRefDensityKgM3(kDragRefDensity, Fw::ParamValid::VALID);
  this->paramSet_DragRefAltitudeM(kDragRefAltitude, Fw::ParamValid::VALID);
  this->paramSet_DragScaleHeightM(kDragScaleHeight, Fw::ParamValid::VALID);
  this->paramSet_AccelPsdM2PerS3(kAccelPsd, Fw::ParamValid::VALID);
  {
    Vec3F64 zero;
    zero[0] = 0.0;
    zero[1] = 0.0;
    zero[2] = 0.0;
    this->paramSet_AccelPsdRtnM2PerS3(zero, Fw::ParamValid::VALID);
    this->paramSet_DmcTauS(0.0, Fw::ParamValid::VALID);
    this->paramSet_DmcPsdRtnM2PerS5(zero, Fw::ParamValid::VALID);
  }
  this->paramSet_PositionNisGate(kNisGate, Fw::ParamValid::VALID);
  this->paramSet_VelocityNisGate(kNisGate, Fw::ParamValid::VALID);
  this->paramSet_MaxCoastS(kMaxCoastS, Fw::ParamValid::VALID);
  this->paramSet_MaxDegradedCoastS(kMaxDegradedCoastS, Fw::ParamValid::VALID);
  this->paramSet_MaxAccelAgeS(kMaxAccelAgeS, Fw::ParamValid::VALID);
  this->paramSet_StatusPeriodCycles(kStatusPeriodCycles, Fw::ParamValid::VALID);
  this->paramSet_MaxDtS(kMaxDtS, Fw::ParamValid::VALID);
  this->paramSet_MaxStepS(kMaxStepS, Fw::ParamValid::VALID);
  this->paramSet_MaxFixLatencyS(kMaxFixLatencyS, Fw::ParamValid::VALID);
  this->paramSet_MinRadiusM(kMinRadiusM, Fw::ParamValid::VALID);
  this->paramSet_MaxRadiusM(kMaxRadiusM, Fw::ParamValid::VALID);
  this->paramSet_PositionMeasMode(0, Fw::ParamValid::VALID);
  this->paramSet_VelocityMeasMode(0, Fw::ParamValid::VALID);
  this->paramSet_BackupPeriodS(kBackupPeriodS, Fw::ParamValid::VALID);
  // paramSet_* only stages values in the harness's table; the component's base
  // caches them at load, exactly as the topology does once ParameterDb is up.
  this->component.loadParameters();
}

void OrbitEstimatorTester ::startTruthAt(I64 taiNs) {
  const double radius = pc::gravity::kReferenceRadius + kAltitudeM;
  const double speed = std::sqrt(pc::gravity::kGM / radius);
  const Eigen::Vector3d r(radius, 0.0, 0.0);
  const Eigen::Vector3d v(0.0, speed * std::cos(kInclinationRad),
                          speed * std::sin(kInclinationRad));
  const pg::OrbitOd::Covariance cov = pg::OrbitOd::Covariance::Identity();
  ASSERT_EQ(
      this->truth_.initialize(pt::Tai::fromNanosecondsSinceEpoch(taiNs),
                              pm::Vec3<pm::frames::ECI>(r), pm::Vec3<pm::frames::ECI>(v), cov),
      pg::OrbitOdRefusal::kNone);
}

void OrbitEstimatorTester ::truthAt(I64 taiNs, Eigen::Vector3d& r_eci, Eigen::Vector3d& v_eci) {
  const pt::Tai target = pt::Tai::fromNanosecondsSinceEpoch(taiNs);
  // Walk forward in steps the truth's own max_dt admits.
  while (this->truth_.epoch() < target) {
    pt::Tai next = this->truth_.epoch() + pt::Duration::fromSecondsF(kMaxDtS);
    if (target < next) {
      next = target;
    }
    ASSERT_EQ(this->truth_.propagate(next, stubEop()), pg::OrbitOdRefusal::kNone);
  }
  r_eci = this->truth_.position().eigen();
  v_eci = this->truth_.velocity().eigen();
}

void OrbitEstimatorTester ::sendFixAt(I64 taiNs, double radiusScale, bool velocityValid) {
  Eigen::Vector3d r_eci;
  Eigen::Vector3d v_eci;
  this->truthAt(taiNs, r_eci, v_eci);
  const pt::Tai tai = pt::Tai::fromNanosecondsSinceEpoch(taiNs);
  pm::Vec3<pm::frames::ECEF> r_ecef;
  pm::Vec3<pm::frames::ECEF> v_ecef;
  ASSERT_TRUE(polaris::frames::ecefStateFromEci(tai, stubEop(), pm::Vec3<pm::frames::ECI>(r_eci),
                                                pm::Vec3<pm::frames::ECI>(v_eci), r_ecef, v_ecef));
  GnssMeas meas;
  meas.set_posEcefM(toVec3(radiusScale * r_ecef.eigen()));
  meas.set_velEcefMps(toVec3(v_ecef.eigen()));
  meas.set_timeTagGpsNs(pt::toGps(tai).nanosecondsSinceEpoch());
  meas.set_posSigmaHM(kPosSigmaH);
  meas.set_posSigmaVM(kPosSigmaV);
  meas.set_velSigmaMps(kVelSigma);
  meas.set_velValid(velocityValid);
  meas.set_valid(true);
  this->invoke_to_gnssIn(0, meas);
}

void OrbitEstimatorTester ::runCycleAt(I64 taiNs) {
  const U32 seconds = static_cast<U32>(taiNs / kNsPerSecond);
  const U32 useconds = static_cast<U32>((taiNs % kNsPerSecond) / 1000);
  this->setTestTime(Fw::Time(seconds, useconds));
  this->invoke_to_run(0, 0);
  ++this->cycles_run_;
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

void OrbitEstimatorTester ::testRefusesWithoutParameters() {
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_EVENTS_OrbitTuningInvalid_SIZE(1);
  ASSERT_EQ(this->estimate_count_, 1u);
  EXPECT_FALSE(this->last_estimate_.get_valid());
  ASSERT_TLM_SolutionValid_SIZE(1);
  ASSERT_TLM_SolutionValid(0, false);
  ASSERT_TLM_FixesAccepted(0, 0u);
  // Warned once, not per cycle.
  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_EVENTS_OrbitTuningInvalid_SIZE(1);
  ASSERT_EVENTS_OrbitSeeded_SIZE(0);
  // Tuning arrives: the fix that was waiting was dropped, not held over — the
  // next one seeds.
  this->setValidParameters();
  this->paramSend_MaxCoastS(0, 0);
  this->runCycleAt(kStartTaiNs + 2 * kNsPerSecond);
  ASSERT_EVENTS_OrbitSeeded_SIZE(0);
  EXPECT_FALSE(this->last_estimate_.get_valid());
}

void OrbitEstimatorTester ::testSeedsAndTracks() {
  this->setValidParameters();
  // Fixes stamped 50 ms before the cycle — the receiver's latency (§8.3). The
  // truth only walks forward, so it starts at the first fix's epoch.
  const I64 latencyNs = 50'000'000LL;
  this->startTruthAt(kStartTaiNs - latencyNs);
  I64 t = kStartTaiNs;
  this->sendFixAt(t - latencyNs);
  this->runCycleAt(t);
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 1u);

  // 60 s of 1 Hz fixes with 10 Hz cycles between them.
  for (int s = 1; s <= 60; ++s) {
    for (int k = 1; k <= 10; ++k) {
      t = kStartTaiNs + (10 * s + k - 10) * (kNsPerSecond / 10);
      if (k == 10) {
        this->sendFixAt(t - latencyNs);
      }
      this->runCycleAt(t);
      ASSERT_TRUE(this->last_estimate_.get_valid()) << "cycle " << s << "." << k;
    }
  }
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(t, r_true, v_true);
  const Eigen::Vector3d r_est = toEigen(this->last_estimate_.get_posEciM());
  const Eigen::Vector3d v_est = toEigen(this->last_estimate_.get_velEciMps());
  // Noise-free fixes on the filter's own model: the estimate sits on the truth
  // to well inside one receiver sigma, and the published epoch is this cycle's.
  EXPECT_LT((r_est - r_true).norm(), 0.1);
  EXPECT_LT((v_est - v_true).norm(), 0.01);
  EXPECT_EQ(this->last_estimate_.get_epochTaiNs(), t);
  EXPECT_LT(this->last_estimate_.get_ageSec(), 0.2);
  EXPECT_GT(this->last_estimate_.get_posSigmaM(), 0.0);
  EXPECT_LT(this->last_estimate_.get_posSigmaM(), 3.0);
  // Every fix was accepted; the latency it carried is reported.
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 61u);
  ASSERT_TLM_FixesRefused(this->tlmHistory_FixesRefused->size() - 1, 0u);
  const F64 latency =
      this->tlmHistory_FixLatencyS->at(this->tlmHistory_FixLatencyS->size() - 1).arg;
  EXPECT_NEAR(latency, 0.05, 1.0e-6);
  ASSERT_EVENTS_FixRefused_SIZE(0);
  ASSERT_EVENTS_OrbitSolutionDropped_SIZE(0);
}

void OrbitEstimatorTester ::testCoastsThenDropsAtHorizon() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::FINE);

  // Coast at 1 Hz cycles: FINE on the propagated state through the fine horizon.
  I64 t = kStartTaiNs;
  for (int s = 1; s <= 300; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->runCycleAt(t);
    ASSERT_TRUE(this->last_estimate_.get_valid()) << "coast second " << s;
    ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::FINE) << "coast second " << s;
    EXPECT_NEAR(this->last_estimate_.get_ageSec(), static_cast<F64>(s), 1.0e-6);
  }
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(t, r_true, v_true);
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 0.1);
  ASSERT_EVENTS_OrbitSolutionDegraded_SIZE(0);
  ASSERT_EVENTS_OrbitSolutionDropped_SIZE(0);

  // One second past the fine horizon: DEGRADED, still valid and published,
  // once-reported, and the position still tracks the truth on this model.
  t += kNsPerSecond;
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::DEGRADED);
  ASSERT_EVENTS_OrbitSolutionDegraded_SIZE(1);
  ASSERT_EVENTS_OrbitSolutionDropped_SIZE(0);
  for (int s = 302; s <= 900; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->runCycleAt(t);
    ASSERT_TRUE(this->last_estimate_.get_valid()) << "coast second " << s;
    ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::DEGRADED) << "second " << s;
  }
  ASSERT_EVENTS_OrbitSolutionDegraded_SIZE(1);
  this->truthAt(t, r_true, v_true);
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 0.5);
  EXPECT_GT(this->last_estimate_.get_posSigmaM(), 1.0)
      << "the degraded covariance must have grown past the receiver's own";

  // One second past the degraded horizon: dropped, once, with the age and the
  // horizon it happened at.
  t += kNsPerSecond;
  this->runCycleAt(t);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::NONE);
  ASSERT_EVENTS_OrbitSolutionDropped_SIZE(1);
  ASSERT_EVENTS_OrbitSolutionDropped(0, 901.0, kMaxDegradedCoastS);
  ASSERT_TLM_LastRefusal(this->tlmHistory_LastRefusal->size() - 1,
                         OrbitEstimator::OdRefusal::COAST_EXPIRED);
  // Stays dropped, quietly.
  t += kNsPerSecond;
  this->runCycleAt(t);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_OrbitSolutionDropped_SIZE(1);

  // The next fix re-seeds whole.
  t += kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::FINE);
  ASSERT_EVENTS_OrbitSeeded_SIZE(2);
  EXPECT_LT(this->last_estimate_.get_ageSec(), 1.0e-6);

  // The status cadence: one OrbitStatus per kStatusPeriodCycles cycles.
  const std::size_t cycles = static_cast<std::size_t>(this->cycles_run_);
  ASSERT_EVENTS_OrbitStatus_SIZE(cycles / kStatusPeriodCycles);
}

void OrbitEstimatorTester ::testDegradedReacquiresByUpdateNotSeed() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  I64 t = kStartTaiNs;
  for (int s = 1; s <= 600; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->runCycleAt(t);
  }
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::DEGRADED);
  // A returning fix inside the degraded band is an *update* on the grown
  // covariance: no second seed, quality back to FINE, nothing refused.
  t += kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::FINE);
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  ASSERT_EVENTS_FixRefused_SIZE(0);
  EXPECT_LT(this->last_estimate_.get_ageSec(), 1.0e-6);
}

void OrbitEstimatorTester ::testNonGravAccelIsAppliedOnlyWhenFreshAndValid() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_TRUE(this->last_estimate_.get_valid());

  // A 0.03 m/s^2 along-track thrust the executor reports at this cycle: the
  // filter propagates with it (the truth here does not thrust, so the estimate
  // departs the truth by 1/2 a t^2 — the point is that it *was* applied).
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(kStartTaiNs, r_true, v_true);
  const Eigen::Vector3d a = 0.03 * v_true.normalized();
  NonGravAccel accel;
  accel.set_epochTaiNs(kStartTaiNs + kNsPerSecond);
  accel.set_accelEciMps2(toVec3(a));
  accel.set_sigmaMps2(1.5e-3);
  accel.set_valid(true);
  this->invoke_to_accelIn(0, accel);
  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_EVENTS_NonGravAccelApplied_SIZE(1);
  ASSERT_TLM_NonGravAccelMps2(this->tlmHistory_NonGravAccelMps2->size() - 1, 0.03);
  this->truthAt(kStartTaiNs + kNsPerSecond, r_true, v_true);
  const double departure = (toEigen(this->last_estimate_.get_posEciM()) - r_true).norm();
  EXPECT_NEAR(departure, 0.5 * 0.03 * 1.0 * 1.0, 1.0e-3);

  // The same record two seconds later is stale (MaxAccelAgeS = 0.5): cleared,
  // and the propagate coasts.
  this->runCycleAt(kStartTaiNs + 3 * kNsPerSecond);
  ASSERT_EVENTS_NonGravAccelCleared_SIZE(1);
  ASSERT_TLM_NonGravAccelMps2(this->tlmHistory_NonGravAccelMps2->size() - 1, 0.0);

  // An invalid record is never applied, whatever its epoch.
  accel.set_epochTaiNs(kStartTaiNs + 4 * kNsPerSecond);
  accel.set_valid(false);
  this->invoke_to_accelIn(0, accel);
  this->runCycleAt(kStartTaiNs + 4 * kNsPerSecond);
  ASSERT_EVENTS_NonGravAccelApplied_SIZE(1);
  ASSERT_TLM_NonGravAccelMps2(this->tlmHistory_NonGravAccelMps2->size() - 1, 0.0);
}

void OrbitEstimatorTester ::testGroundSeedAcceptedAndRefused() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  // No fix at all: the ground seed is what starts the filter.
  this->runCycleAt(kStartTaiNs);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(kStartTaiNs, r_true, v_true);

  // Refused: an epoch further behind now than the degraded horizon.
  this->sendCmd_OD_SEED_STATE(0, 0, kStartTaiNs - 2000 * kNsPerSecond, r_true.x(), r_true.y(),
                              r_true.z(), v_true.x(), v_true.y(), v_true.z(), 100.0, 0.1);
  ASSERT_CMD_RESPONSE(0, OrbitEstimator::OPCODE_OD_SEED_STATE, 0,
                      Fw::CmdResponse::VALIDATION_ERROR);
  ASSERT_EVENTS_OrbitSeedRefused_SIZE(1);
  // Refused: an implausible radius.
  this->sendCmd_OD_SEED_STATE(0, 1, kStartTaiNs, 10.0 * r_true.x(), 10.0 * r_true.y(),
                              10.0 * r_true.z(), v_true.x(), v_true.y(), v_true.z(), 100.0, 0.1);
  ASSERT_CMD_RESPONSE(1, OrbitEstimator::OPCODE_OD_SEED_STATE, 1,
                      Fw::CmdResponse::VALIDATION_ERROR);
  ASSERT_EVENTS_OrbitSeedRefused_SIZE(2);
  ASSERT_EVENTS_OrbitSeedRefused(1, OrbitEstimator::OdRefusal::FIX_IMPLAUSIBLE);
  // Accepted: the truth at now, 100 m / 0.1 m/s.
  this->sendCmd_OD_SEED_STATE(0, 2, kStartTaiNs, r_true.x(), r_true.y(), r_true.z(), v_true.x(),
                              v_true.y(), v_true.z(), 100.0, 0.1);
  ASSERT_CMD_RESPONSE(2, OrbitEstimator::OPCODE_OD_SEED_STATE, 2, Fw::CmdResponse::OK);
  ASSERT_EVENTS_OrbitSeededFromGround_SIZE(1);
  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EQ(this->last_estimate_.get_quality(), OrbitQuality::FINE);
  EXPECT_NEAR(this->last_estimate_.get_posSigmaM(), 100.0 * std::sqrt(3.0), 1.0);
  this->truthAt(kStartTaiNs + kNsPerSecond, r_true, v_true);
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 0.01);
  // And a fix now updates it, no seed.
  this->sendFixAt(kStartTaiNs + 2 * kNsPerSecond);
  this->runCycleAt(kStartTaiNs + 2 * kNsPerSecond);
  ASSERT_EVENTS_OrbitSeeded_SIZE(0);
  ASSERT_EVENTS_FixRefused_SIZE(0);
}

void OrbitEstimatorTester ::testRefusesImplausibleFix() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  const Eigen::Vector3d r_before = toEigen(this->last_estimate_.get_posEciM());

  // A GEO-radius fix (§9.1): refused on the plausibility band, solution
  // untouched, one EVR for the run of them.
  for (int s = 1; s <= 3; ++s) {
    const I64 t = kStartTaiNs + s * kNsPerSecond;
    this->sendFixAt(t, 6.0);
    this->runCycleAt(t);
    ASSERT_TRUE(this->last_estimate_.get_valid());
  }
  ASSERT_EVENTS_FixRefused_SIZE(1);
  ASSERT_EVENTS_FixRefused(0, 0, OrbitEstimator::OdRefusal::FIX_IMPLAUSIBLE, 1u);
  ASSERT_TLM_FixesRefused(this->tlmHistory_FixesRefused->size() - 1, 3u);
  ASSERT_TLM_LastRefusal(this->tlmHistory_LastRefusal->size() - 1,
                         OrbitEstimator::OdRefusal::FIX_IMPLAUSIBLE);
  // The estimate moved by three seconds of orbit, not toward the bad fix.
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(kStartTaiNs + 3 * kNsPerSecond, r_true, v_true);
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 0.1);
  EXPECT_GT((toEigen(this->last_estimate_.get_posEciM()) - r_before).norm(), 1.0e4);

  // A different refusal logs again: a good fix at a stale epoch is the stuck
  // clock the filter refuses on the fix epoch.
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs + 4 * kNsPerSecond);
  ASSERT_EVENTS_FixRefused_SIZE(2);
  ASSERT_EVENTS_FixRefused(1, 0, OrbitEstimator::OdRefusal::NON_MONOTONIC_EPOCH, 4u);

  // A good fix clears the edge: the next refusal of the *first* kind logs.
  this->sendFixAt(kStartTaiNs + 5 * kNsPerSecond);
  this->runCycleAt(kStartTaiNs + 5 * kNsPerSecond);
  ASSERT_TLM_LastRefusal(this->tlmHistory_LastRefusal->size() - 1, OrbitEstimator::OdRefusal::NONE);
  this->sendFixAt(kStartTaiNs + 6 * kNsPerSecond, 6.0);
  this->runCycleAt(kStartTaiNs + 6 * kNsPerSecond);
  ASSERT_EVENTS_FixRefused_SIZE(3);
}

void OrbitEstimatorTester ::testEopUnavailable() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->eop_available_ = false;
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_EopUnavailable_SIZE(1);
  ASSERT_EVENTS_OrbitSeeded_SIZE(0);
  ASSERT_TLM_LastRefusal(this->tlmHistory_LastRefusal->size() - 1,
                         OrbitEstimator::OdRefusal::FRAME_CONVERSION);
  // Edge-gated across cycles.
  this->sendFixAt(kStartTaiNs + kNsPerSecond);
  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_EVENTS_EopUnavailable_SIZE(1);
  // Tables come back: seeds on the next fix, and a later loss logs again.
  this->eop_available_ = true;
  this->sendFixAt(kStartTaiNs + 2 * kNsPerSecond);
  this->runCycleAt(kStartTaiNs + 2 * kNsPerSecond);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  this->eop_available_ = false;
  this->runCycleAt(kStartTaiNs + 3 * kNsPerSecond);
  ASSERT_EVENTS_EopUnavailable_SIZE(2);
}

void OrbitEstimatorTester ::testResetDropsSolution() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  this->sendFixAt(kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_TRUE(this->last_estimate_.get_valid());

  this->sendCmd_OD_RESET(0, 0);
  ASSERT_CMD_RESPONSE_SIZE(1);
  ASSERT_CMD_RESPONSE(0, OrbitEstimator::OPCODE_OD_RESET, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_OrbitReset_SIZE(1);

  this->runCycleAt(kStartTaiNs + kNsPerSecond);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 0u);

  this->sendFixAt(kStartTaiNs + 2 * kNsPerSecond);
  this->runCycleAt(kStartTaiNs + 2 * kNsPerSecond);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_OrbitSeeded_SIZE(2);
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 1u);
}

// ----------------------------------------------------------------------
// NESC navigation-filter usability practices (NASA/TP-2018-219822 Ch. 7, §9;
// NESC TB 20-03 items d-g) — Push 71
// ----------------------------------------------------------------------

void OrbitEstimatorTester ::testTuningUploadKeepsTheSolution() {
  this->setValidParameters();
  this->startTruthAt(kStartTaiNs);
  I64 t = kStartTaiNs;
  for (int s = 0; s <= 5; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->sendFixAt(t);
    this->runCycleAt(t);
  }
  ASSERT_TRUE(this->last_estimate_.get_valid());
  const Eigen::Vector3d r_before = toEigen(this->last_estimate_.get_posEciM());
  ASSERT_EVENTS_OrbitTuningApplied_SIZE(0);

  // An upload: a tighter gate and a shorter fine horizon. The solution stays.
  this->paramSet_PositionNisGate(7.81, Fw::ParamValid::VALID);
  this->paramSend_PositionNisGate(0, 0);
  this->paramSet_MaxCoastS(120.0, Fw::ParamValid::VALID);
  this->paramSend_MaxCoastS(0, 0);
  ASSERT_EVENTS_OrbitTuningApplied_SIZE(2);
  ASSERT_EVENTS_OrbitTuningApplied(0, true);
  ASSERT_EVENTS_OrbitTuningApplied(1, true);
  ASSERT_EVENTS_OrbitReset_SIZE(0);
  t += kNsPerSecond / 10;
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid()) << "the solution survived the upload";
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_before).norm(), 800.0)
      << "one 100 ms step of the same trajectory, not a re-seed";
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  const U32 accepted_before =
      this->tlmHistory_FixesAccepted->at(this->tlmHistory_FixesAccepted->size() - 1).arg;
  EXPECT_EQ(accepted_before, 6u) << "the counters were kept, not rebuilt";

  // A bad upload: warned, and the last valid set stays in force — the filter
  // keeps running and keeps taking fixes.
  this->paramSet_MaxCoastS(-1.0, Fw::ParamValid::VALID);
  this->paramSend_MaxCoastS(0, 0);
  ASSERT_EVENTS_OrbitTuningInvalid_SIZE(1);
  ASSERT_EVENTS_OrbitTuningApplied_SIZE(2);
  t = kStartTaiNs + 7 * kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 7u);
  ASSERT_EVENTS_FixRefused_SIZE(0);
}

void OrbitEstimatorTester ::testCovarianceReinitAndMeasurementPolicy() {
  this->setValidParameters();

  // Before the first cycle the parameters have not been applied: OD_REINIT_COV
  // has no filter to act on, and says which.
  this->sendCmd_OD_REINIT_COV(0, 0, 100.0, 1.0);
  ASSERT_CMD_RESPONSE(0, OrbitEstimator::OPCODE_OD_REINIT_COV, 0,
                      Fw::CmdResponse::VALIDATION_ERROR);
  ASSERT_EVENTS_OrbitCovarianceReinitRefused_SIZE(1);
  ASSERT_EVENTS_OrbitCovarianceReinitRefused(0, OrbitEstimator::OdRefusal::UNCONFIGURED);

  this->startTruthAt(kStartTaiNs);
  I64 t = kStartTaiNs;
  for (int s = 0; s <= 5; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->sendFixAt(t);
    this->runCycleAt(t);
  }
  ASSERT_TRUE(this->last_estimate_.get_valid());
  const Eigen::Vector3d r_before = toEigen(this->last_estimate_.get_posEciM());
  const F64 sigma_before = this->last_estimate_.get_posSigmaM();
  ASSERT_LT(sigma_before, 3.0);
  const std::size_t healthy_idx = this->tlmHistory_CovarianceHealthy->size() - 1;
  ASSERT_TLM_CovarianceHealthy(healthy_idx, true);

  // Re-open the covariance: the state is untouched, the published sigma is the
  // commanded one, and a bad sigma is refused.
  this->sendCmd_OD_REINIT_COV(0, 0, 100.0, -1.0);
  ASSERT_EVENTS_OrbitCovarianceReinitRefused_SIZE(2);
  ASSERT_EVENTS_OrbitCovarianceReinitRefused(1, OrbitEstimator::OdRefusal::FIX_SIGMA_INVALID);
  this->sendCmd_OD_REINIT_COV(0, 0, 100.0, 1.0);
  ASSERT_CMD_RESPONSE(2, OrbitEstimator::OPCODE_OD_REINIT_COV, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_OrbitCovarianceReinitialised_SIZE(1);
  ASSERT_EVENTS_OrbitCovarianceReinitialised(0, 100.0, 1.0);
  t += kNsPerSecond / 10;
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_before).norm(), 800.0)
      << "state kept (one 100 ms step)";
  EXPECT_NEAR(this->last_estimate_.get_posSigmaM(), 100.0 * std::sqrt(3.0), 1.0)
      << "P = diag(100^2 I, ...): sqrt(trace) over three axes, plus 100 ms of growth";
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);

  // Policy: INHIBIT the position measurement. The fix is refused by name, not
  // gated, and the solution keeps coasting.
  this->paramSet_PositionMeasMode(1, Fw::ParamValid::VALID);
  this->paramSend_PositionMeasMode(0, 0);
  ASSERT_EVENTS_MeasurementPolicyChanged_SIZE(2);  // bring-up (0,0), then (1,0)
  ASSERT_EVENTS_MeasurementPolicyChanged(1, 1, 0);
  t = kStartTaiNs + 7 * kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_FixRefused_SIZE(1);
  ASSERT_EVENTS_FixRefused(0, 0, OrbitEstimator::OdRefusal::MEASUREMENT_INHIBITED, 1u);
  ASSERT_TLM_LastRefusal(
      this->tlmHistory_LastRefusal->size() - 1,
      OrbitEstimator::OdRefusal(OrbitEstimator::OdRefusal::MEASUREMENT_INHIBITED));

  // FORCE: a fix 2 km out in radius, which the gate would refuse against a
  // ~1 m sigma, is applied and reported as forced — counted apart from the
  // accepted and the refused.
  this->paramSet_PositionMeasMode(2, Fw::ParamValid::VALID);
  this->paramSend_PositionMeasMode(0, 0);
  ASSERT_EVENTS_MeasurementPolicyChanged_SIZE(3);
  t = kStartTaiNs + 8 * kNsPerSecond;
  const double radius = pc::gravity::kReferenceRadius + kAltitudeM;
  this->sendFixAt(t, 1.0 + 2000.0 / radius);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_TLM_FixesForced(this->tlmHistory_FixesForced->size() - 1, 1u);
  ASSERT_EVENTS_FixRefused_SIZE(1);
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(t, r_true, v_true);
  EXPECT_GT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 100.0)
      << "the forced fix moved the state off the truth";
}

void OrbitEstimatorTester ::testBackupEphemerisRestart() {
  this->setValidParameters();

  // No backup yet: the restart is refused by name.
  this->sendCmd_OD_RESTART_FROM_BACKUP(0, 0);
  ASSERT_CMD_RESPONSE(0, OrbitEstimator::OPCODE_OD_RESTART_FROM_BACKUP, 0,
                      Fw::CmdResponse::EXECUTION_ERROR);
  ASSERT_EVENTS_OrbitBackupRestartRefused_SIZE(1);

  this->startTruthAt(kStartTaiNs);
  I64 t = kStartTaiNs;
  for (int s = 0; s <= 5; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->sendFixAt(t);
    this->runCycleAt(t);
  }
  ASSERT_TRUE(this->last_estimate_.get_valid());
  // Seeded from the first FINE solution (the seed cycle itself), then
  // propagated alongside: age climbs, the divergence between two copies of the
  // same solution on the same model is at the round-off level.
  ASSERT_EVENTS_OrbitBackupSeeded_SIZE(1);
  const std::size_t last = this->tlmHistory_BackupAgeS->size() - 1;
  EXPECT_NEAR(this->tlmHistory_BackupAgeS->at(last).arg, 5.0, 1.0e-6);
  const F64 divergence = this->tlmHistory_BackupDivergenceM->at(last).arg;
  EXPECT_GE(divergence, 0.0);
  EXPECT_LT(divergence, 1.0) << "the fixes are on the model; the two copies agree";

  // The solution is lost (a commanded reset stands in for a fault or a halt).
  this->sendCmd_OD_RESET(0, 0);
  ASSERT_EVENTS_OrbitReset_SIZE(1);
  t += kNsPerSecond / 10;
  this->runCycleAt(t);
  ASSERT_FALSE(this->last_estimate_.get_valid());
  ASSERT_TLM_BackupDivergenceM(this->tlmHistory_BackupDivergenceM->size() - 1, -1.0);
  EXPECT_GE(this->tlmHistory_BackupAgeS->at(this->tlmHistory_BackupAgeS->size() - 1).arg, 5.0)
      << "the backup outlives the solution";

  // Restart from the backup: no uplink, no fix — the solution is back on the
  // truth to the accuracy of the model it was propagated on.
  this->sendCmd_OD_RESTART_FROM_BACKUP(0, 0);
  ASSERT_CMD_RESPONSE(2, OrbitEstimator::OPCODE_OD_RESTART_FROM_BACKUP, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_OrbitRestartedFromBackup_SIZE(1);
  t += kNsPerSecond / 10;
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());
  Eigen::Vector3d r_true;
  Eigen::Vector3d v_true;
  this->truthAt(t, r_true, v_true);
  EXPECT_LT((toEigen(this->last_estimate_.get_posEciM()) - r_true).norm(), 1.0);
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 0u);

  // Fixes resume against the restored solution — an update, not a seed.
  t = kStartTaiNs + 7 * kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_EVENTS_OrbitSeeded_SIZE(1);
  ASSERT_EVENTS_FixRefused_SIZE(0);
  ASSERT_TLM_FixesAccepted(this->tlmHistory_FixesAccepted->size() - 1, 1u);
}

void OrbitEstimatorTester ::testCovarianceMetricsAndDmcParameters() {
  this->setValidParameters();
  // Turn the DMC states on before the first cycle applies the set.
  Vec3F64 q_dmc;
  q_dmc[0] = 3.0e-12;
  q_dmc[1] = 3.0e-12;
  q_dmc[2] = 3.0e-12;
  this->paramSet_DmcTauS(600.0, Fw::ParamValid::VALID);
  this->paramSet_DmcPsdRtnM2PerS5(q_dmc, Fw::ParamValid::VALID);
  this->component.loadParameters();

  this->startTruthAt(kStartTaiNs);
  I64 t = kStartTaiNs;
  for (int s = 0; s <= 10; ++s) {
    t = kStartTaiNs + s * kNsPerSecond;
    this->sendFixAt(t);
    this->runCycleAt(t);
  }
  ASSERT_TRUE(this->last_estimate_.get_valid());
  ASSERT_EVENTS_OrbitTuningInvalid_SIZE(0);
  const std::size_t last = this->tlmHistory_SmaSigmaM->size() - 1;
  const F64 sma_sigma = this->tlmHistory_SmaSigmaM->at(last).arg;
  const F64 fpa_sigma = this->tlmHistory_FpaSigmaRad->at(last).arg;
  // TP Eq. 2.20 on the settled covariance: sigma_a ≈ 2 sqrt(sigma_r² + (T/2π)² sigma_v²)
  // — metres, since a 1 m position and 0.03 m/s velocity are ~10 m of SMA
  // (the velocity term dominates: T/2π ≈ 880 s).
  EXPECT_GT(sma_sigma, 0.0);
  EXPECT_LT(sma_sigma, 200.0);
  EXPECT_GT(fpa_sigma, 0.0);
  EXPECT_LT(fpa_sigma, 1.0e-3);
  // The DMC block is alive: its sigma is the stationary sqrt(3 q tau/2) at seed
  // and no larger after; its estimate is near zero on a truth with no
  // unmodelled acceleration.
  const F64 dmc_sigma =
      this->tlmHistory_DmcSigmaMps2->at(this->tlmHistory_DmcSigmaMps2->size() - 1).arg;
  EXPECT_GT(dmc_sigma, 0.0);
  EXPECT_LE(dmc_sigma, std::sqrt(3.0 * 3.0e-12 * 600.0 * 0.5) * 1.001);
  const Vec3F64 a =
      this->tlmHistory_DmcAccelRtnMps2->at(this->tlmHistory_DmcAccelRtnMps2->size() - 1).arg;
  EXPECT_LT(std::abs(a[1]), 1.0e-5);

  // A correlation time under ten sub-steps is refused as tuning; the running
  // filter keeps its last valid set (TB 20-03 item g).
  this->paramSet_DmcTauS(0.5, Fw::ParamValid::VALID);
  this->paramSend_DmcTauS(0, 0);
  ASSERT_EVENTS_OrbitTuningInvalid_SIZE(1);
  t += kNsPerSecond;
  this->sendFixAt(t);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_estimate_.get_valid());

  // No solution: the metrics read -1.
  this->sendCmd_OD_RESET(0, 0);
  t += kNsPerSecond;
  this->runCycleAt(t);
  ASSERT_TLM_SmaSigmaM(this->tlmHistory_SmaSigmaM->size() - 1, -1.0);
  ASSERT_TLM_FpaSigmaRad(this->tlmHistory_FpaSigmaRad->size() - 1, -1.0);
}

}  // namespace flight

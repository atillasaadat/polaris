// ======================================================================
// \title  AttitudeEstimator.cpp
// \brief  Attitude estimator component: coarse chain + MEKF fine mode with
//         arbitration (§8.1, §10; REQ-ADET-002, REQ-ADET-003, REQ-ADET-004)
// ======================================================================

#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "environment/igrf_iaga.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "Fw/Log/LogString.hpp"
#include "time/utc.hpp"

namespace flight {

namespace {

constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;

namespace pm = polaris::math;
using ECEF = pm::frames::ECEF;
using ECI = pm::frames::ECI;
using Body = pm::frames::Body;

//! Number of coarse-chain tuning parameters read from ParameterDb.
constexpr FwSizeType kParamCount = 12;

//! Number of fine-mode (MEKF + Davenport seed) tuning parameters. Validated
//! separately: a missing one costs the fine mode, not the whole estimator.
constexpr FwSizeType kFineF64ParamCount = 5;

//! Not a value: telemetry channels that have nothing to report this cycle. Zero
//! would draw as a perfect solution on a strip chart.
const double kNoValue = std::numeric_limits<double>::quiet_NaN();

//! Read a `Vec3F64` telemetry/port array member into a raw 3-vector.
Eigen::Vector3d toEigen(const Vec3F64& v) {
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

//! Pack a raw 3-vector into the port/telemetry array type.
Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
  return out;
}

//! True if @p timeTagNs is within @p maxAgeS of @p nowTaiNs. A measurement from
//! the future is stale too — a time tag ahead of the master clock is a fault,
//! not freshness (§9.1).
bool fresh(I64 nowTaiNs, I64 timeTagNs, F64 maxAgeS) {
  if (!(maxAgeS > 0.0)) {
    return false;
  }
  const I64 delta = nowTaiNs - timeTagNs;
  const I64 limit = static_cast<I64>(maxAgeS * static_cast<F64>(kNsPerSecond));
  return delta >= -limit && delta <= limit;
}

//! Map the lib estimation mode onto the telemetry/port enum.
EstimationMode::T toEstimationMode(polaris::state::EstimationMode m) {
  switch (m) {
    case polaris::state::EstimationMode::Coarse:
      return EstimationMode::COARSE;
    case polaris::state::EstimationMode::Fine:
      return EstimationMode::FINE;
    default:
      return EstimationMode::INVALID;
  }
}

//! The worse of two source grades — a reference chain is only as good as its
//! weakest link, so a coarse EOP degrades a precise ephemeris.
TableGrade::T worseGrade(TableGrade::T a, TableGrade::T b) {
  return (static_cast<U8>(a) <= static_cast<U8>(b)) ? a : b;
}

}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

AttitudeEstimator ::AttitudeEstimator(const char* const compName)
    : AttitudeEstimatorComponentBase(compName),
      // Both start inert: a default-constructed config fails isValid() and a
      // default coefficient set fails validIgrfCoefficients(), so nothing runs
      // on invented values before configuration arrives.
      estimator_(polaris::gnc::CoarseAttitudeConfig{}),
      mekf_(polaris::gnc::MekfConfig{}),
      igrf_(polaris::environment::IgrfCoefficients{}),
      leap_(polaris::time::LeapSecondTable::historical()) {}

AttitudeEstimator ::~AttitudeEstimator() {}

// ----------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------

bool AttitudeEstimator ::configureIgrf(const char* igrfPath, double decimalYear) {
  FW_ASSERT(igrfPath != nullptr);
  std::strncpy(this->igrf_path_, igrfPath, kMaxPathLength - 1);
  this->igrf_path_[kMaxPathLength - 1] = '\0';

  if (!std::isfinite(decimalYear)) {
    const Fw::LogStringArg reason("mission epoch for the IGRF snapshot is not a finite year");
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }

  polaris::environment::IgrfCoefficients coefficients;
  char why[polaris::environment::kIgrfMaxReasonLength] = {};
  if (!polaris::environment::loadIgrfIaga(this->igrf_path_, decimalYear, coefficients, why,
                                          sizeof(why))) {
    const Fw::LogStringArg reason(why);
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }

  this->igrf_ = polaris::environment::IgrfField(coefficients);
  if (!this->igrf_.good()) {
    const Fw::LogStringArg reason("IGRF evaluator rejected the loaded coefficients");
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }
  this->igrf_epoch_year_ = coefficients.epoch_year;
  this->igrf_valid_until_year_ = coefficients.valid_until_year;
  this->igrf_stale_flagged_ = false;
  this->log_ACTIVITY_HI_IgrfLoaded(coefficients.epoch_year, coefficients.valid_until_year,
                                   static_cast<U32>(coefficients.degree));
  return true;
}

bool AttitudeEstimator ::refreshCoarseConfig() {
  // Each get is its own statement: `valid` is an out-parameter, so reading it in
  // the same expression that calls the getter would be unsequenced.
  Fw::ParamValid valids[kParamCount];
  F64 values[kParamCount];
  values[0] = this->paramGet_SigmaSunWhiteRad(valids[0]);
  values[1] = this->paramGet_SigmaSunSysRad(valids[1]);
  values[2] = this->paramGet_SigmaMagWhiteRad(valids[2]);
  values[3] = this->paramGet_SigmaMagSysRad(valids[3]);
  values[4] = this->paramGet_GyroArw(valids[4]);
  values[5] = this->paramGet_MinSinAngle(valids[5]);
  values[6] = this->paramGet_TriadGain(valids[6]);
  values[7] = this->paramGet_MaxCoastSec(valids[7]);
  values[8] = this->paramGet_MaxDtSec(valids[8]);
  values[9] = this->paramGet_MaxMeasAgeSec(valids[9]);
  values[10] = this->paramGet_MinPositionRadiusM(valids[10]);
  values[11] = this->paramGet_MaxPositionRadiusM(valids[11]);

  static const char* const kNames[kParamCount] = {
      "SigmaSunWhiteRad", "SigmaSunSysRad", "SigmaMagWhiteRad",   "SigmaMagSysRad",
      "GyroArw",          "MinSinAngle",    "TriadGain",          "MaxCoastSec",
      "MaxDtSec",         "MaxMeasAgeSec",  "MinPositionRadiusM", "MaxPositionRadiusM"};

  for (FwSizeType i = 0; i < kParamCount; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i])) {
      char detail[80];
      (void)std::snprintf(detail, sizeof(detail), "%s missing or not finite in ParameterDb",
                          kNames[i]);
      this->failConfig(detail);
      return false;
    }
  }

  polaris::gnc::CoarseAttitudeConfig cfg;
  cfg.sigma_sun_white_rad = values[0];
  cfg.sigma_sun_sys_rad = values[1];
  cfg.sigma_mag_white_rad = values[2];
  cfg.sigma_mag_sys_rad = values[3];
  cfg.gyro_arw = values[4];
  cfg.min_sin_angle = values[5];
  cfg.triad_gain = values[6];
  cfg.max_coast_s = values[7];
  cfg.max_dt_s = values[8];

  if (!cfg.isValid()) {
    this->failConfig("tuning values are out of range (see CoarseAttitudeConfig::isValid)");
    return false;
  }
  // A staleness window wider than the coast horizon would keep handing the
  // estimator measurements it has already declared itself too old to trust.
  if (!(values[9] > 0.0) || values[9] > cfg.max_coast_s) {
    this->failConfig("MaxMeasAgeSec must be positive and <= MaxCoastSec");
    return false;
  }
  if (!(values[10] > 0.0) || !(values[11] > values[10])) {
    this->failConfig("position radius gate must satisfy 0 < Min < Max");
    return false;
  }

  // Rebuilding drops any solution in flight, which is the honest behaviour: the
  // covariance the old solution carries was computed under the old budget.
  this->noteAttitudeLost();  // the rebuild drops the solution; say so
  this->estimator_ = polaris::gnc::CoarseAttitudeEstimator(cfg);
  this->max_meas_age_s_ = values[9];
  this->min_position_radius_m_ = values[10];
  this->max_position_radius_m_ = values[11];
  // The MEKF and the Davenport seed take one sigma per source, and the filter
  // treats it as white. The systematic part is therefore root-sum-squared in
  // here rather than carried separately as the coarse chain carries it: an
  // uninflated sigma would let the filter average down an offset that does not
  // average down (gnc/mekf.hpp). It is derived from the coarse budget so the two
  // estimators cannot be told different things about the same sensor.
  this->sigma_sun_total_rad_ = std::hypot(cfg.sigma_sun_white_rad, cfg.sigma_sun_sys_rad);
  this->sigma_mag_total_rad_ = std::hypot(cfg.sigma_mag_white_rad, cfg.sigma_mag_sys_rad);
  this->config_invalid_flagged_ = false;
  return true;
}

bool AttitudeEstimator ::refreshFineConfig() {
  Fw::ParamValid valids[kFineF64ParamCount];
  F64 values[kFineF64ParamCount];
  values[0] = this->paramGet_MekfRrw(valids[0]);
  values[1] = this->paramGet_MekfNisGate(valids[1]);
  values[2] = this->paramGet_MekfMaxCoastSec(valids[2]);
  values[3] = this->paramGet_MekfBiasSigmaInit(valids[3]);
  values[4] = this->paramGet_SeedMinObservability(valids[4]);

  static const char* const kNames[kFineF64ParamCount] = {
      "MekfRrw", "MekfNisGate", "MekfMaxCoastSec", "MekfBiasSigmaInit", "SeedMinObservability"};

  for (FwSizeType i = 0; i < kFineF64ParamCount; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i])) {
      char detail[80];
      (void)std::snprintf(detail, sizeof(detail), "%s missing or not finite in ParameterDb",
                          kNames[i]);
      this->failFineConfig(detail);
      return false;
    }
  }

  Fw::ParamValid refusal_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid nis_valid = Fw::ParamValid::INVALID;
  const U32 refusal_streak = this->paramGet_MekfRefusalStreak(refusal_valid);
  const U32 nis_streak = this->paramGet_MekfNisStreak(nis_valid);
  if (refusal_valid != Fw::ParamValid::VALID || nis_valid != Fw::ParamValid::VALID) {
    this->failFineConfig("MekfRefusalStreak/MekfNisStreak missing from ParameterDb");
    return false;
  }
  if (refusal_streak == 0 || nis_streak == 0) {
    this->failFineConfig("demotion streak thresholds must be positive");
    return false;
  }

  // Two values are shared with the coarse chain rather than duplicated: the gyro
  // angle random walk (same gyro, same parametrisation) and the largest accepted
  // propagation step (same rate group, same dropout definition). Two parameters
  // for one physical quantity is two chances to disagree.
  Fw::ParamValid arw_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid dt_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid coarse_coast_valid = Fw::ParamValid::INVALID;
  const F64 arw = this->paramGet_GyroArw(arw_valid);
  const F64 max_dt = this->paramGet_MaxDtSec(dt_valid);
  const F64 coarse_coast_s = this->paramGet_MaxCoastSec(coarse_coast_valid);
  if (arw_valid != Fw::ParamValid::VALID || dt_valid != Fw::ParamValid::VALID ||
      coarse_coast_valid != Fw::ParamValid::VALID || !std::isfinite(arw) ||
      !std::isfinite(max_dt) || !std::isfinite(coarse_coast_s)) {
    this->failFineConfig("GyroArw/MaxDtSec/MaxCoastSec missing or not finite in ParameterDb");
    return false;
  }

  polaris::gnc::MekfConfig cfg;
  cfg.arw_rad_per_sqrt_s = arw;
  cfg.rrw_rad_per_s_per_sqrt_s = values[0];
  cfg.nis_gate = values[1];
  cfg.max_coast_s = values[2];
  cfg.max_dt_s = max_dt;

  if (!cfg.isValid()) {
    this->failFineConfig("MEKF tuning is out of range (see MekfConfig::isValid)");
    return false;
  }
  // A zero seed sigma would tell the filter the turn-on bias is known exactly,
  // and it would never learn one. The observability gate is a ratio in (0, 1).
  if (!(values[3] > 0.0)) {
    this->failFineConfig("MekfBiasSigmaInit must be positive");
    return false;
  }
  if (!(values[4] > 0.0) || !(values[4] < 1.0)) {
    this->failFineConfig("SeedMinObservability must be in (0, 1)");
    return false;
  }
  // The whole arbitration rests on the coarse chain outliving the fine one: a
  // fine mode that coasted longer than coarse would demote onto a fallback that
  // had already gone invalid, which is the one case where a demotion really does
  // lose the attitude.
  if (cfg.max_coast_s > coarse_coast_s) {
    this->failFineConfig("MekfMaxCoastSec must be <= MaxCoastSec (coarse must outlast fine)");
    return false;
  }

  // Rebuilding drops any fine solution in flight, for the same reason the coarse
  // rebuild does: its covariance was computed under the old budget.
  this->demoteFineMode(FineDemotionReason::COMMANDED);
  this->mekf_ = polaris::gnc::Mekf(cfg);
  this->bias_sigma_init_ = values[3];
  this->seed_min_observability_ = values[4];
  this->refusal_streak_limit_ = refusal_streak;
  this->nis_streak_limit_ = nis_streak;
  this->fine_config_invalid_flagged_ = false;
  return true;
}

void AttitudeEstimator ::failConfig(const char* detail) {
  // The fine mode goes first. The coarse chain is about to stop running, so
  // nothing would be published at all — a fine solution left engaged would sit
  // frozen behind that, un-stepped and un-published, until it eventually
  // demoted with a COAST reason that describes none of what happened.
  this->demoteFineMode(FineDemotionReason::COMMANDED);
  // Going inert loses the solution as surely as a coast expiry does, and a
  // consumer watching the validity flag deserves the same edge either way.
  this->noteAttitudeLost();
  // Edge-gated: one alert on entering the unconfigured state, not one per cycle.
  if (!this->config_invalid_flagged_) {
    const Fw::LogStringArg arg(detail);
    this->log_WARNING_HI_ConfigInvalid(arg);
    this->config_invalid_flagged_ = true;
  }
  this->estimator_ = polaris::gnc::CoarseAttitudeEstimator(polaris::gnc::CoarseAttitudeConfig{});
  this->max_meas_age_s_ = 0.0;
}

void AttitudeEstimator ::failFineConfig(const char* detail) {
  // Not fatal, unlike failConfig(): the vehicle keeps the §10 Safe-mode floor
  // and simply never promotes. An operator seeing this has a flyable vehicle.
  this->demoteFineMode(FineDemotionReason::COMMANDED);
  if (!this->fine_config_invalid_flagged_) {
    const Fw::LogStringArg arg(detail);
    this->log_WARNING_HI_FineConfigInvalid(arg);
    this->fine_config_invalid_flagged_ = true;
  }
  this->mekf_ = polaris::gnc::Mekf(polaris::gnc::MekfConfig{});
}

void AttitudeEstimator ::noteAttitudeLost() {
  if (!this->attitude_valid_) {
    return;
  }
  this->log_WARNING_HI_AttitudeLost(this->last_age_s_);
  this->attitude_valid_ = false;
}

void AttitudeEstimator ::parameterUpdated(FwPrmIdType id) {
  // Any change re-reads the whole set: the values are validated together
  // (CoarseAttitudeConfig::isValid), so one at a time means nothing.
  this->params_dirty_ = true;
}

// ----------------------------------------------------------------------
// Fine-mode arbitration (REQ-ADET-004)
// ----------------------------------------------------------------------

double AttitudeEstimator ::arbitrateFineMode(const polaris::time::Tai& epoch,
                                             const polaris::gnc::CoarseAttitudeInput& in,
                                             const polaris::gnc::CoarseAttitudeOutput& coarse) {
  if (!this->mekf_.isConfigured()) {
    return kNoValue;  // coarse-only operation; failFineConfig() has already said so
  }
  if (this->fine_active_) {
    return this->stepFineMode(epoch, in);
  }
  // Deliberately else-if and not a second chance in the same cycle: a demotion
  // costs at least one cycle on coarse, so a condition that oscillates shows up
  // in the event stream instead of hiding inside a single cycle.
  //
  // Promotion waits for a valid coarse attitude even though the Davenport solve
  // does not need one. It is the cheap statement that the vehicle has a working
  // floor before it starts trusting a filter on top of it.
  if (coarse.attitude_valid && in.sun_valid && in.mag_valid) {
    this->tryPromoteFineMode(epoch, in);
  } else {
    // Re-arm the seed alert whenever the preconditions are not met. Without this
    // a single refused solve latches the flag for the rest of the flight, since
    // the cycles that clear it are exactly the ones that promote — and an
    // eclipse, which stops promotion attempts entirely, would leave a later run
    // of genuine seed failures silent.
    this->fine_init_failed_flagged_ = false;
  }
  return kNoValue;
}

void AttitudeEstimator ::tryPromoteFineMode(const polaris::time::Tai& epoch,
                                            const polaris::gnc::CoarseAttitudeInput& in) {
  polaris::gnc::DavenportInput seed;
  seed.count = 2;
  seed.min_observability = this->seed_min_observability_;
  seed.observations[0].body = in.sun_body;
  seed.observations[0].reference = in.sun_ref;
  seed.observations[0].sigma_rad = this->sigma_sun_total_rad_;
  seed.observations[1].body = in.mag_body;
  seed.observations[1].reference = in.mag_ref;
  seed.observations[1].sigma_rad = this->sigma_mag_total_rad_;

  polaris::gnc::DavenportSolution solution;
  if (!polaris::gnc::davenport(seed, solution)) {
    // Near-parallel sun and field is a normal flight condition, so this is
    // edge-gated: one warning per run of failures, not one per cycle.
    if (!this->fine_init_failed_flagged_) {
      const Fw::LogStringArg arg("Davenport refused the seed (geometry or malformed pair)");
      this->log_WARNING_LO_FineInitFailed(arg);
      this->fine_init_failed_flagged_ = true;
    }
    return;
  }

  // The seed bias is zero at the configured turn-on repeatability: nothing
  // better is known at cold start, and inventing a bias is worse than admitting
  // to one this large.
  const Eigen::Matrix3d bias_cov =
      (this->bias_sigma_init_ * this->bias_sigma_init_) * Eigen::Matrix3d::Identity();
  if (!this->mekf_.initialize(epoch, solution.attitude, solution.covariance,
                              pm::Vec3<Body>(Eigen::Vector3d::Zero()), bias_cov)) {
    if (!this->fine_init_failed_flagged_) {
      const Fw::LogStringArg arg("MEKF rejected the seed (non-finite or indefinite covariance)");
      this->log_WARNING_LO_FineInitFailed(arg);
      this->fine_init_failed_flagged_ = true;
    }
    return;
  }

  this->fine_active_ = true;
  this->refusal_streak_ = 0;
  this->nis_streak_ = 0;
  this->fine_init_failed_flagged_ = false;
  this->log_ACTIVITY_HI_FineModeEngaged(solution.covariance.trace());
}

double AttitudeEstimator ::stepFineMode(const polaris::time::Tai& epoch,
                                        const polaris::gnc::CoarseAttitudeInput& in) {
  bool refused = !this->mekf_.propagate(epoch, in.gyro, in.gyro_valid);
  bool nis_rejected = false;
  // The **largest** NIS of the cycle, not the last: a rejected outlier followed
  // by a good update would otherwise be telemetered as if the cycle were
  // healthy, and the outlier is the whole reason the channel exists. A rejection
  // is above the gate and an accepted update below it, so the maximum is the
  // rejected one whenever there was one.
  double worst_nis = kNoValue;

  // One update per available pair, folded in one at a time — which is what makes
  // the §8.2 fusion layer more calls rather than an interface change.
  const polaris::gnc::VectorObservation pairs[2] = {
      {in.sun_body, in.sun_ref, this->sigma_sun_total_rad_},
      {in.mag_body, in.mag_ref, this->sigma_mag_total_rad_}};
  const bool pair_valid[2] = {in.sun_valid, in.mag_valid};

  for (int i = 0; i < 2; ++i) {
    if (!pair_valid[i] || !this->mekf_.isInitialised()) {
      continue;
    }
    const U32 rejected_before = this->mekf_.rejectedCount();
    polaris::gnc::MekfUpdate diagnostics;
    const bool applied =
        this->mekf_.update(pairs[i].body, pairs[i].reference, pairs[i].sigma_rad, diagnostics);
    // A gate rejection and a malformed measurement both return false, and they
    // mean opposite things: one is the divergence guard working, the other is
    // the filter refusing input it cannot use. The rejection counter is what
    // tells them apart, so the two demotion streaks stay meaningful — and a
    // malformed call leaves the diagnostics default-constructed, so its zero NIS
    // must not be telemetered as if it were measured.
    const bool gate_rejected = !applied && this->mekf_.rejectedCount() > rejected_before;
    if (!applied && !gate_rejected) {
      refused = true;
      continue;
    }
    nis_rejected = nis_rejected || gate_rejected;
    if (std::isnan(worst_nis) || diagnostics.nis > worst_nis) {
      worst_nis = diagnostics.nis;
    }
  }

  this->refusal_streak_ = refused ? (this->refusal_streak_ + 1) : 0;
  this->nis_streak_ = nis_rejected ? (this->nis_streak_ + 1) : 0;

  if (!this->mekf_.isInitialised()) {
    this->demoteFineMode(FineDemotionReason::FILTER_FAULT);
  } else if (this->nis_streak_ >= this->nis_streak_limit_) {
    this->demoteFineMode(FineDemotionReason::NIS_STREAK);
  } else if (this->refusal_streak_ >= this->refusal_streak_limit_) {
    this->demoteFineMode(FineDemotionReason::REFUSAL_STREAK);
  } else if (!this->mekf_.attitudeValid()) {
    this->demoteFineMode(FineDemotionReason::COAST);
  }
  return worst_nis;
}

void AttitudeEstimator ::demoteFineMode(FineDemotionReason::T reason) {
  if (!this->fine_active_) {
    return;
  }
  this->log_WARNING_HI_FineModeDemoted(reason, this->mekf_.ageSeconds());
  ++this->fine_demotions_;
  // Carry the rejection count across the reset. `Mekf::reset` clears it, by
  // contract, because a *commanded* reset is the operator saying "start over" —
  // but a demotion for NIS_STREAK or FILTER_FAULT would then erase the count at
  // precisely the moment FDIR needs it, so the component keeps its own running
  // total that only RESET_ESTIMATOR clears.
  this->fine_rejected_total_ += this->mekf_.rejectedCount();
  // The filter state goes with it. A demotion is the statement that this
  // solution is no longer trusted, and a solution not worth publishing is not
  // worth resuming — re-promotion runs a fresh Davenport seed.
  this->mekf_.reset();
  this->fine_active_ = false;
  this->refusal_streak_ = 0;
  this->nis_streak_ = 0;
}

// ----------------------------------------------------------------------
// Measurement latching
// ----------------------------------------------------------------------

void AttitudeEstimator ::imuIn_handler(FwIndexType portNum, const ImuMeas& meas) {
  this->imu_[portNum] = meas;
}

void AttitudeEstimator ::sunSensorIn_handler(FwIndexType portNum, const SunSensorMeas& meas) {
  this->sun_[portNum] = meas;
}

void AttitudeEstimator ::magnetometerIn_handler(FwIndexType portNum, const MagnetometerMeas& meas) {
  this->mag_[portNum] = meas;
}

void AttitudeEstimator ::gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) {
  this->gnss_[portNum] = meas;
}

void AttitudeEstimator ::starTrackerIn_handler(FwIndexType portNum, const StarTrackerMeas& meas) {
  // TODO(§8.2): the fusion layer feeds this to the MEKF as a third measurement
  // source. Counted here so the interface is exercised and a tracker delivering
  // solutions is visible in telemetry; the coarse solution stays
  // tracker-independent on purpose (§10).
  if (meas.get_valid()) {
    ++this->star_tracker_count_;
  }
}

const ImuMeas* AttitudeEstimator ::selectImu(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_IMUIN_INPUT_PORTS; ++i) {
    const ImuMeas& m = this->imu_[i];
    if (m.get_valid() && m.get_intervalSec() > 0.0 &&
        fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->imu_[i];
    }
  }
  return nullptr;
}

const SunSensorMeas* AttitudeEstimator ::selectSunSensor(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_SUNSENSORIN_INPUT_PORTS; ++i) {
    const SunSensorMeas& m = this->sun_[i];
    // No sun in view is a normal condition (eclipse, or the unit facing away),
    // not a fault — it simply excludes this unit from the pair this cycle.
    if (m.get_valid() && m.get_sunPresent() &&
        fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->sun_[i];
    }
  }
  return nullptr;
}

const MagnetometerMeas* AttitudeEstimator ::selectMagnetometer(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_MAGNETOMETERIN_INPUT_PORTS; ++i) {
    const MagnetometerMeas& m = this->mag_[i];
    if (m.get_valid() && fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->mag_[i];
    }
  }
  return nullptr;
}

const GnssMeas* AttitudeEstimator ::selectGnss(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_GNSSIN_INPUT_PORTS; ++i) {
    const GnssMeas& m = this->gnss_[i];
    if (!m.get_valid()) {
      continue;
    }
    // §9.1 range gate. A GNSS fix is wire data from outside the FSW, and a
    // non-finite or absurd position does not fail loudly downstream — it
    // poisons *both* references (a NaN radius through the field model, a NaN
    // sun direction through r_eci) while every validity flag still reads true.
    // So it is checked here, with the other per-source gates.
    const Eigen::Vector3d r = toEigen(m.get_posEcefM());
    if (!r.allFinite()) {
      continue;
    }
    const double radius = r.norm();
    if (radius < this->min_position_radius_m_ || radius > this->max_position_radius_m_) {
      continue;
    }
    // The receiver stamps GPS time; TAI = GPS + 19 s on ingest (§3.2).
    const polaris::time::Tai tag =
        polaris::time::toTai(polaris::time::Gps::fromNanosecondsSinceEpoch(m.get_timeTagGpsNs()));
    if (fresh(nowTaiNs, tag.nanosecondsSinceEpoch(), this->max_meas_age_s_)) {
      return &this->gnss_[i];
    }
  }
  return nullptr;
}

// ----------------------------------------------------------------------
// Estimation cycle
// ----------------------------------------------------------------------

void AttitudeEstimator ::run_handler(FwIndexType portNum, U32 context) {
  const I64 nowNs = this->currentTaiNs();

  if (this->params_dirty_) {
    this->params_dirty_ = false;
    // Two independent gates. The coarse set failing leaves the vehicle with no
    // attitude at all; the fine set failing leaves it with the Safe-mode floor,
    // which is a working vehicle — so the second is not allowed to take the
    // first down with it.
    (void)this->refreshCoarseConfig();  // emits ConfigInvalid and leaves us inert
    (void)this->refreshFineConfig();    // emits FineConfigInvalid; coarse-only
  }
  if (!this->estimator_.isConfigured()) {
    // No tuning, no estimate. Telemetering INVALID every cycle is what tells the
    // operator (and FDIR) that the vehicle has no attitude solution.
    this->state_ = polaris::state::EstimatedState{};
    this->state_.epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
    this->tlmWrite_EstMode(EstimationMode::INVALID);
    return;
  }

  // A stuck or backwards clock is refused here rather than inside the estimator,
  // so the refusal is counted and telemetered instead of looking like a quiet
  // no-op cycle.
  if (this->have_epoch_ && nowNs <= this->last_epoch_tai_ns_) {
    ++this->cycles_refused_;
    this->tlmWrite_CyclesRefused(this->cycles_refused_);
    return;
  }
  this->last_epoch_tai_ns_ = nowNs;
  this->have_epoch_ = true;

  const polaris::time::Tai epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
  polaris::gnc::CoarseAttitudeInput in;
  in.epoch = epoch;

  // --- Gyro -----------------------------------------------------------------
  const ImuMeas* const imu = this->selectImu(nowNs);
  if (imu != nullptr) {
    const Eigen::Vector3d rate = toEigen(imu->get_deltaAngleRad()) / imu->get_intervalSec();
    in.gyro = pm::Vec3<Body>(rate);
    in.gyro_valid = in.gyro.isFinite();
  }
  // gyro_bias stays zero: the coarse mode does not estimate bias (the MEKF
  // does), and substituting an unestimated bias would be inventing one.

  // --- Position and the ECEF->ECI rotation ----------------------------------
  // Both inertial references need the rotation; the magnetic one also needs the
  // position, which today comes straight off GNSS. There is no onboard orbit
  // propagator yet (§8.3), so a GNSS outage costs the magnetic pair — flagged
  // by PositionUnavailable rather than papered over with a guessed position.
  pm::Vec3<ECEF> r_ecef;
  const GnssMeas* const gnss = this->selectGnss(nowNs);
  const bool have_position = gnss != nullptr;
  if (have_position) {
    r_ecef = pm::Vec3<ECEF>(toEigen(gnss->get_posEcefM()));
  }

  TableGrade::T eop_grade = TableGrade::UNAVAILABLE;
  pm::Quat<ECI, ECEF> q_eci_ecef;
  bool have_rotation = false;
  if (this->isConnected_getEopAt_OutputPort(0)) {
    EopSample sample;
    if (this->getEopAt_out(0, nowNs, sample)) {
      const polaris::frames::EopValue eop{sample.get_ut1MinusTai(), sample.get_xpArcsec(),
                                          sample.get_ypArcsec()};
      have_rotation = polaris::frames::eciFromEcef(epoch, eop, q_eci_ecef);
      eop_grade = sample.get_grade();
    }
  }

  // --- Sun reference --------------------------------------------------------
  pm::Vec3<ECI> sun_ref;
  bool have_sun_ref = false;
  TableGrade::T ephem_grade = TableGrade::UNAVAILABLE;
  if (this->isConnected_getBodyPosition_OutputPort(0)) {
    PosEciMeters sunPos;
    if (this->getBodyPosition_out(0, OnboardBody::SUN, nowNs, sunPos)) {
      pm::Vec3<ECI> to_sun(sunPos.get_x(), sunPos.get_y(), sunPos.get_z());
      if (have_position && have_rotation) {
        // Geocentric to spacecraft-centric. Worth ~4e-5 rad at LEO — far below
        // the coarse budget, but it costs one subtraction and the rotation is
        // already in hand for the field.
        to_sun = to_sun - q_eci_ecef.rotate(r_ecef);
      }
      have_sun_ref = to_sun.normalized(sun_ref);
      ephem_grade = sunPos.get_grade();
    }
  }
  // A query that could not answer leaves its domain UNAVAILABLE, so the reported
  // grade never claims PRECISE on the strength of the *other* domain alone.
  const TableGrade::T grade = worseGrade(eop_grade, ephem_grade);

  // --- Magnetic reference ---------------------------------------------------
  pm::Vec3<ECI> mag_ref;
  bool have_mag_ref = false;
  if (have_position && have_rotation && this->igrf_.good()) {
    double year = 0.0;
    pm::Vec3<ECEF> b_ecef;
    if (polaris::time::decimalYear(epoch, this->leap_, year)) {
      // Refuse a field the loaded snapshot cannot honestly model. The horizon is
      // the one the IAGA file itself implies — the next tabulated epoch, or five
      // years past the last — not a fixed gap from the base epoch, which would
      // expire a snapshot taken near the end of a grid interval almost
      // immediately after launch.
      if (year > this->igrf_valid_until_year_) {
        if (!this->igrf_stale_flagged_) {
          this->log_WARNING_HI_MagneticReferenceStale(year, this->igrf_valid_until_year_);
          this->igrf_stale_flagged_ = true;
        }
      } else {
        this->igrf_stale_flagged_ = false;  // re-arm on the epoch check, not on a good eval
        if (this->igrf_.field(r_ecef, year, b_ecef)) {
          // B is a vector field: it rotates with the same rotation, no translation.
          mag_ref = q_eci_ecef.rotate(b_ecef);
          have_mag_ref = mag_ref.isFinite();
        }
      }
    }
  }

  // Edge-gated position alert: no position means no field model, hence no TRIAD.
  if (!have_position && !this->position_unavailable_flagged_) {
    this->log_WARNING_LO_PositionUnavailable();
    this->position_unavailable_flagged_ = true;
  } else if (have_position) {
    this->position_unavailable_flagged_ = false;
  }
  this->noteReferenceGrade(TableDomain::EPHEMERIS, ephem_grade, this->last_ephem_grade_,
                           this->have_ephem_grade_);
  this->noteReferenceGrade(TableDomain::EOP, eop_grade, this->last_eop_grade_,
                           this->have_eop_grade_);

  // --- Measurement pairs ----------------------------------------------------
  const SunSensorMeas* const sun = this->selectSunSensor(nowNs);
  if (sun != nullptr && have_sun_ref) {
    in.sun_body = pm::Vec3<Body>(toEigen(sun->get_dirBody()));
    in.sun_ref = sun_ref;
    in.sun_valid = in.sun_body.isFinite();
  }
  const MagnetometerMeas* const magnetometer = this->selectMagnetometer(nowNs);
  if (magnetometer != nullptr && have_mag_ref) {
    in.mag_body = pm::Vec3<Body>(toEigen(magnetometer->get_fieldTesla()));
    in.mag_ref = mag_ref;
    in.mag_valid = in.mag_body.isFinite();
  }

  // --- One estimation cycle -------------------------------------------------
  polaris::gnc::CoarseAttitudeOutput out;
  const bool valid_attitude = this->estimator_.update(in, out);

  // A refused cycle returns a default-constructed product. With a valid finite
  // gyro a cycle that actually ran always publishes a rate, so this identifies
  // the internal-refusal case (non-finite result) without guessing; a refusal
  // while the gyro is invalid is indistinguishable from a normal no-solution
  // cycle and is not counted.
  if (!valid_attitude && in.gyro_valid && !out.rate_valid) {
    ++this->cycles_refused_;
  }
  if (in.sun_valid && in.mag_valid) {
    if (out.triad_applied) {
      ++this->triad_accepted_;
    } else {
      ++this->triad_rejected_;
    }
  }

  // --- Fine mode ------------------------------------------------------------
  // Runs after the coarse cycle and on the same measurement set, so the coarse
  // solution is always a live fallback rather than one that has to re-acquire.
  const bool fine_was_active = this->fine_active_;
  const double cycle_nis = this->arbitrateFineMode(epoch, in, out);
  this->tlmWrite_MekfNis(cycle_nis);

  // The cycle that *engages* fine mode still publishes coarse. `Mekf::initialize`
  // seeds an attitude but no rate — nothing has been propagated yet — so
  // publishing the filter here would inject a one-cycle `rateValid == false` and
  // a zero body rate into the GNC product while the coarse chain had a perfectly
  // good rate in hand. Promotion therefore takes effect on the *next* cycle,
  // symmetrically with the one-cycle rule a demotion already follows. A demotion
  // still takes effect immediately: falling back is never worth delaying.
  const bool publish_fine = fine_was_active && this->fine_active_;

  // Acquisition/loss edges (REQ-ADET-004: mode transitions surfaced to FDIR).
  // Keyed on the *published* solution: a demotion with a live coarse solution
  // underneath loses nothing, and must not look like it did.
  const bool published_valid = publish_fine ? this->mekf_.attitudeValid() : out.attitude_valid;
  const double published_age_s = publish_fine ? this->mekf_.ageSeconds() : out.age_s;
  const double published_cov_trace =
      publish_fine ? this->mekf_.covariance()
                         .block<3, 3>(polaris::gnc::Mekf::kAttitude, polaris::gnc::Mekf::kAttitude)
                         .trace()
                   : out.covariance.trace();
  this->last_age_s_ = published_age_s;
  if (published_valid) {
    if (!this->attitude_valid_) {
      this->log_ACTIVITY_HI_AttitudeAcquired(published_age_s, published_cov_trace);
    }
    this->attitude_valid_ = true;
  } else {
    this->noteAttitudeLost();
  }

  if (publish_fine) {
    this->publishFine(epoch);
  } else {
    this->publishCoarse(out, epoch);
  }

  this->tlmWrite_GyroValid(in.gyro_valid);
  this->tlmWrite_SunValid(in.sun_valid);
  this->tlmWrite_MagValid(in.mag_valid);
  this->tlmWrite_PositionValid(have_position);
  this->tlmWrite_TriadAccepted(this->triad_accepted_);
  this->tlmWrite_TriadRejected(this->triad_rejected_);
  this->tlmWrite_CyclesRefused(this->cycles_refused_);
  this->tlmWrite_StarTrackerCount(this->star_tracker_count_);
  this->tlmWrite_RefGrade(grade);
  this->tlmWrite_MekfRejected(this->mekf_.rejectedCount());
  // The running total survives the demotions that clear the filter's own count.
  this->tlmWrite_MekfRejectedTotal(this->fine_rejected_total_ + this->mekf_.rejectedCount());
  this->tlmWrite_FineDemotions(this->fine_demotions_);
  // Zero bias while coarse-only: the coarse chain does not estimate a bias, and
  // the MEKF's is zero when it holds no solution.
  this->tlmWrite_GyroBias(toVec3F64(this->mekf_.gyroBias().eigen()));
  const bool fine = this->fine_active_;
  this->tlmWrite_FineAttCovTrace(
      fine ? this->mekf_.covariance()
                 .block<3, 3>(polaris::gnc::Mekf::kAttitude, polaris::gnc::Mekf::kAttitude)
                 .trace()
           : kNoValue);
  this->tlmWrite_BiasCovTrace(
      fine ? this->mekf_.covariance()
                 .block<3, 3>(polaris::gnc::Mekf::kGyroBias, polaris::gnc::Mekf::kGyroBias)
                 .trace()
           : kNoValue);
}

void AttitudeEstimator ::publishCoarse(const polaris::gnc::CoarseAttitudeOutput& out,
                                       const polaris::time::Tai& epoch) {
  // The canonical §8.0 mapping lives in lib/gnc, so the component cannot invent
  // its own idea of what "valid" or "Coarse" means.
  polaris::gnc::writeToEstimatedState(out, epoch, this->state_);
  this->emitEstimate(out.covariance, out.age_s);
}

void AttitudeEstimator ::publishFine(const polaris::time::Tai& epoch) {
  // Same rule for the fine product: the mapping — including which validity flags
  // a coasting filter is allowed to set — is the lib/gnc overload's, not this
  // component's.
  polaris::gnc::writeToEstimatedState(this->mekf_, epoch, this->state_);
  this->emitEstimate(this->mekf_.covariance().block<3, 3>(polaris::gnc::Mekf::kAttitude,
                                                          polaris::gnc::Mekf::kAttitude),
                     this->mekf_.ageSeconds());
}

void AttitudeEstimator ::emitEstimate(const Eigen::Matrix3d& cov, double age_s) {
  const polaris::math::Quaternion& q = this->state_.attitude.core();
  QuatF64 quat;
  quat[0] = q.w();
  quat[1] = q.x();
  quat[2] = q.y();
  quat[3] = q.z();
  const Vec3F64 rate = toVec3F64(this->state_.body_rate.eigen());
  const Vec3F64 cov_diag = toVec3F64(Eigen::Vector3d(cov.diagonal()));

  AttitudeEstimate estimate;
  estimate.set_epochTaiNs(this->state_.epoch.nanosecondsSinceEpoch());
  estimate.set_qBodyEci(quat);
  estimate.set_bodyRateRadps(rate);
  estimate.set_attCovDiagRad2(cov_diag);
  estimate.set_ageSec(age_s);
  estimate.set_mode(toEstimationMode(this->state_.mode));
  estimate.set_attitudeValid(this->state_.valid.attitude);
  estimate.set_rateValid(this->state_.valid.body_rate);
  if (this->isConnected_estimateOut_OutputPort(0)) {
    this->estimateOut_out(0, estimate);
  }

  this->tlmWrite_EstMode(toEstimationMode(this->state_.mode));
  this->tlmWrite_AttQuat(quat);
  this->tlmWrite_BodyRate(rate);
  // A refused or coasting-expired cycle has no covariance to report; zero would
  // draw as a perfect solution on a strip chart, so say "no value" instead.
  this->tlmWrite_AttCovTrace(this->state_.valid.attitude ? cov.trace() : kNoValue);
  this->tlmWrite_SolutionAge(age_s);
}

void AttitudeEstimator ::noteReferenceGrade(TableDomain::T domain, TableGrade::T grade,
                                            TableGrade::T& last, bool& have) {
  if (!have) {
    last = grade;
    have = true;
    // A first cycle already on the coarse fallback is worth one alert: the
    // operator has no earlier transition to have seen it from.
    if (grade != TableGrade::PRECISE) {
      this->log_WARNING_HI_ReferenceDegraded(domain, grade);
    }
    return;
  }
  if (grade == last) {
    return;
  }
  if (static_cast<U8>(grade) < static_cast<U8>(last)) {
    this->log_WARNING_HI_ReferenceDegraded(domain, grade);
  } else {
    this->log_ACTIVITY_HI_ReferenceRecovered(domain, grade);
  }
  last = grade;
}

// ----------------------------------------------------------------------
// Command handler implementations
// ----------------------------------------------------------------------

void AttitudeEstimator ::RESET_ESTIMATOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  // Both chains: a reset that left the fine filter converged would keep
  // publishing exactly the solution the operator asked to be rid of.
  this->demoteFineMode(FineDemotionReason::COMMANDED);
  this->mekf_.reset();
  // The one place the running rejection total is cleared: a command is the
  // operator saying "start over", which is exactly the contract Mekf::reset
  // keeps for its own count. A demotion carries the total forward instead.
  this->fine_rejected_total_ = 0;
  this->estimator_.reset();
  this->attitude_valid_ = false;
  this->last_age_s_ = 0.0;
  this->have_epoch_ = false;
  // A reset means "tell me everything again": every edge-gated alert is re-armed,
  // not just the configuration one, so a still-broken vehicle re-reports each
  // fault rather than staying quiet because it already mentioned it once.
  this->config_invalid_flagged_ = false;
  this->fine_config_invalid_flagged_ = false;
  this->fine_init_failed_flagged_ = false;
  this->position_unavailable_flagged_ = false;
  this->igrf_stale_flagged_ = false;
  this->have_ephem_grade_ = false;
  this->have_eop_grade_ = false;
  // Re-read the tuning too: a reset is also the recovery path after a config fix.
  this->params_dirty_ = true;
  this->log_ACTIVITY_HI_EstimatorReset();
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

I64 AttitudeEstimator ::currentTaiNs() {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

}  // namespace flight

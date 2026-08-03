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

//! Number of F64 coarse-chain tuning parameters read from ParameterDb. The two
//! §8.2 IMU-voting F64s ride in this set rather than in one of their own: without
//! a voted body rate there is no gyro propagation, so a missing value costs
//! exactly what a missing SigmaSunWhiteRad costs — the whole estimator — and a
//! separate gate would imply a degraded-but-flying state that does not exist.
//! (ImuReadmitCycles is U32 and is read alongside them.)
constexpr FwSizeType kParamCount = 17;

//! Number of fine-mode (MEKF + Davenport seed) tuning parameters. Validated
//! separately: a missing one costs the fine mode, not the whole estimator.
constexpr FwSizeType kFineF64ParamCount = 5;

//! Not a value: telemetry channels that have nothing to report this cycle. Zero
//! would draw as a perfect solution on a strip chart.
const double kNoValue = std::numeric_limits<double>::quiet_NaN();

//! Telemetered SunUnitSelected when no sun sensor was selectable this cycle.
//! 255 rather than 0, which is a perfectly good port index.
constexpr U8 kNoSunUnitIndex = 255;

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
      mag_cal_accumulator_(polaris::gnc::MagCalibrationConfig{}),
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
  values[1] = this->paramGet_SigmaSunAlbedoRad(valids[1]);
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
  values[12] = this->paramGet_SigmaSunAlbedoUncorrRad(valids[12]);
  values[13] = this->paramGet_SigmaSunEphemRad(valids[13]);
  values[14] = this->paramGet_SigmaSunEphemPreciseRad(valids[14]);
  values[15] = this->paramGet_ImuMaxRateRadps(valids[15]);
  values[16] = this->paramGet_ImuDisagreementRadps(valids[16]);
  Fw::ParamValid readmit_valid = Fw::ParamValid::INVALID;
  const U32 readmit_cycles = this->paramGet_ImuReadmitCycles(readmit_valid);
  Fw::ParamValid confirm_valid = Fw::ParamValid::INVALID;
  const U32 confirm_cycles = this->paramGet_ImuIdentifyConfirmCycles(confirm_valid);
  Fw::ParamValid escalate_valid = Fw::ParamValid::INVALID;
  const U32 escalate_cycles = this->paramGet_ImuAmbiguityEscalateCycles(escalate_valid);

  static const char* const kNames[kParamCount] = {"SigmaSunWhiteRad",
                                                  "SigmaSunAlbedoRad",
                                                  "SigmaMagWhiteRad",
                                                  "SigmaMagSysRad",
                                                  "GyroArw",
                                                  "MinSinAngle",
                                                  "TriadGain",
                                                  "MaxCoastSec",
                                                  "MaxDtSec",
                                                  "MaxMeasAgeSec",
                                                  "MinPositionRadiusM",
                                                  "MaxPositionRadiusM",
                                                  "SigmaSunAlbedoUncorrRad",
                                                  "SigmaSunEphemRad",
                                                  "SigmaSunEphemPreciseRad",
                                                  "ImuMaxRateRadps",
                                                  "ImuDisagreementRadps"};

  for (FwSizeType i = 0; i < kParamCount; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i])) {
      char detail[80];
      (void)std::snprintf(detail, sizeof(detail), "%s missing or not finite in ParameterDb",
                          kNames[i]);
      this->failConfig(detail);
      return false;
    }
  }
  if (readmit_valid != Fw::ParamValid::VALID || confirm_valid != Fw::ParamValid::VALID ||
      escalate_valid != Fw::ParamValid::VALID) {
    this->failConfig(
        "ImuReadmitCycles/ImuIdentifyConfirmCycles/ImuAmbiguityEscalateCycles missing from "
        "ParameterDb");
    return false;
  }
  if (escalate_cycles == 0) {
    this->failConfig("ImuAmbiguityEscalateCycles must be non-zero");
    return false;
  }

  polaris::gnc::ImuVoteConfig vote_cfg;
  vote_cfg.max_rate_radps = values[15];
  vote_cfg.disagreement_radps = values[16];
  vote_cfg.readmit_cycles = readmit_cycles;
  vote_cfg.identify_confirm_cycles = confirm_cycles;
  if (!vote_cfg.isValid()) {
    this->failConfig("IMU voting tuning is out of range (see ImuVoteConfig::isValid)");
    return false;
  }

  polaris::gnc::CoarseAttitudeConfig cfg;
  cfg.sigma_sun_white_rad = values[0];
  // The config's sun systematic is the **worst case** of the four-way per-cycle
  // composition below (no albedo correction, analytic ephemeris). Nothing reads
  // it — every cycle supplies its own through CoarseAttitudeInput — so what
  // matters is that it passes isValid() and that, if a future caller ever did
  // fall back to it, it would err wide rather than narrow.
  cfg.sigma_sun_sys_rad = std::hypot(values[12], values[13]);
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
  // Each pair must be ordered: the degraded member wider than the good one. A
  // pair the other way round says the albedo correction makes the measurement
  // worse, or that the analytic fallback beats the DE440 tables — configuration
  // errors rather than flight conditions, and flying either would have the
  // estimator report a covariance tighter than the truth on exactly the cycles
  // it should be least confident.
  // The non-negativity check on the *good* member of each pair rides here too.
  // It has nowhere else to live: the composition is a hypot, which squares its
  // arguments and would absorb a sign typo silently — a negative sigma would
  // pass both this validation and every runtime finiteness check while quietly
  // meaning its own absolute value.
  if (!(values[1] >= 0.0) || !(values[12] >= values[1])) {
    this->failConfig("need 0 <= SigmaSunAlbedoRad <= SigmaSunAlbedoUncorrRad");
    return false;
  }
  if (!(values[14] >= 0.0) || !(values[13] >= values[14])) {
    this->failConfig("need 0 <= SigmaSunEphemPreciseRad <= SigmaSunEphemRad");
    return false;
  }

  // Rebuilding drops any solution in flight, which is the honest behaviour: the
  // covariance the old solution carries was computed under the old budget.
  this->noteAttitudeLost();  // the rebuild drops the solution; say so
  this->estimator_ = polaris::gnc::CoarseAttitudeEstimator(cfg);
  // The voter is rebuilt with the rest, which also drops every exclusion latch:
  // a latch earned under one plausibility limit says nothing under another, and
  // the alternative — carrying it — would leave a unit excluded by a limit the
  // vehicle is no longer flying.
  this->imu_voter_ = polaris::gnc::ImuVoter(vote_cfg);
  this->imu_ambiguous_flagged_ = false;
  this->imu_ambiguous_cycles_ = 0;
  this->imu_ambiguity_escalate_cycles_ = escalate_cycles;
  this->max_meas_age_s_ = values[9];
  this->min_position_radius_m_ = values[10];
  this->max_position_radius_m_ = values[11];
  // The MEKF and the Davenport seed take one sigma per source, and the filter
  // treats it as white. The systematic part is therefore root-sum-squared in
  // here rather than carried separately as the coarse chain carries it: an
  // uninflated sigma would let the filter average down an offset that does not
  // average down (gnc/mekf.hpp). It is derived from the coarse budget so the two
  // estimators cannot be told different things about the same sensor.
  this->sigma_mag_total_rad_ = std::hypot(cfg.sigma_mag_white_rad, cfg.sigma_mag_sys_rad);
  // The same inflation for the cycles the albedo correction could not run on.
  // Precomputed here rather than per cycle: a hypot in the 10 Hz path to pick
  // between two constants is arithmetic the configuration already knows.
  this->sigma_sun_white_rad_ = cfg.sigma_sun_white_rad;
  this->sigma_sun_albedo_corr_rad_ = values[1];
  this->sigma_sun_albedo_uncorr_rad_ = values[12];
  this->sigma_sun_ephem_rad_ = values[13];
  this->sigma_sun_ephem_precise_rad_ = values[14];
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

void AttitudeEstimator ::setSunSigmaForCycle(double albedoSigmaRad, double ephemSigmaRad) {
  // The two terms are independent — one is how well the sensor measured the Sun,
  // the other how well the vehicle knows where the Sun is — so they compose in
  // quadrature. Both are per-cycle: the albedo term depends on whether the
  // correction ran and on how well the attitude is known, the ephemeris term on
  // whether the onboard tables cover this epoch.
  this->sigma_sun_sys_cycle_ = std::hypot(albedoSigmaRad, ephemSigmaRad);
  this->sigma_sun_total_cycle_ = std::hypot(this->sigma_sun_white_rad_, this->sigma_sun_sys_cycle_);
}

double AttitudeEstimator ::applyAlbedoCorrection(FwIndexType sunIndex, const pm::Vec3<ECEF>& r_ecef,
                                                 const pm::Quat<ECI, ECEF>& q_eci_ecef,
                                                 const pm::Vec3<ECI>& sun_geocentric,
                                                 bool havePositionAndRotation,
                                                 bool haveSunGeocentric, double ephemSigmaRad,
                                                 pm::Vec3<Body>& sunBody) {
  // **The one application point** for the Earth-albedo correction (§8.1), called
  // between unit selection and every consumer for the same reason the
  // magnetometer calibration is applied where it is: the coarse chain, the MEKF
  // update and the Davenport seed all read `in.sun_body`, so no consumer can
  // disagree with another about what the sun direction was.
  //
  // Unlike the magnetometer calibration this is a **model, not a commanded
  // calibration**: it needs geometry, not collected data, so it is always on and
  // there is no state to persist. What it does need is a position fix, the
  // ECEF->ECI rotation, and an attitude to place the Earth in the sensor's field
  // with — and that attitude is last cycle's published solution.
  //
  // When any of those is missing the correction is skipped and the *wider*
  // uncorrected sigma is used, rather than correcting on geometry the vehicle
  // does not have — which would inject a bias the size of the one removed,
  // pointed in an arbitrary direction.
  this->setSunSigmaForCycle(this->sigma_sun_albedo_uncorr_rad_, ephemSigmaRad);

  if (!this->albedo_configured_ || !havePositionAndRotation || !haveSunGeocentric ||
      !this->state_.valid.attitude) {
    return kNoValue;
  }

  // **Multi-unit (§8.2): the correction follows the selected unit's boresight.**
  // A wrong boresight scales the correction rather than failing it, so correcting
  // a unit on one face with another face's geometry would be a silent bias the
  // size of the one being removed. A slot the configuration left as the zero
  // vector — a unit that is not installed, or one whose mounting has not been
  // characterised — therefore takes the uncorrected path.
  polaris::gnc::AlbedoCorrectionConfig unit_config = this->albedo_config_;
  if (!this->sunBoresightFor(sunIndex, unit_config.boresight_body) || !unit_config.isValid()) {
    return kNoValue;
  }

  const pm::Vec3<ECI> r_eci = q_eci_ecef.rotate(r_ecef);
  pm::Vec3<ECI> r_hat;
  if (!r_eci.normalized(r_hat)) {
    return kNoValue;
  }

  polaris::gnc::AlbedoCorrectionInput ain;
  ain.sun_meas = sunBody;
  // Nadir is the anti-radial direction, rotated into body axes by the attitude
  // the vehicle currently believes it has.
  ain.nadir_body = this->state_.attitude.rotate(pm::Vec3<ECI>(-r_hat.eigen()));
  ain.radius_m = r_eci.eigen().norm();
  ain.dayside = std::max(0.0, r_hat.eigen().dot(sun_geocentric.eigen()));

  pm::Vec3<Body> corrected;
  double applied = 0.0;
  if (!polaris::gnc::albedoCorrection(unit_config, ain, corrected, applied)) {
    return kNoValue;
  }
  sunBody = corrected;

  // **The correction's own error, carried dynamically.** An attitude error ε
  // misplaces the Earth in the sensor's field, and the dominant consequence is
  // not the misjudged pull *magnitude* but the misjudged rotation **axis**:
  // â = (ŝ × d̂)/|ŝ × d̂| swings by ε/sin ψ, and the resulting error in the
  // correction vector is φ·(ε/sin ψ) = A·ε — set by the **peak** scale A, not by
  // the applied φ, and so present even where the correction itself is small.
  // Measured at ~0.1 deg of sun-vector error per degree of attitude error
  // (albedo_correction_test.cpp, AttitudeErrorGainIsBoundedByThePeakScale).
  //
  // That term is a function of how well the attitude is known *this cycle*, so
  // it belongs in the per-cycle sigma rather than in a configured constant. At a
  // converged fine-mode solution it is ~1 mrad and invisible; right after
  // acquisition, a slew, or a re-acquisition it is the dominant sun error, and
  // those are exactly the cycles where telling both estimators the converged
  // number would be overconfidence in the unsafe direction. Gating on attitude
  // *validity* alone — which is what this did before — could not see the
  // difference between a 0.5 deg solution and a 10 deg one.
  //
  // σ_att is the total 1σ attitude angle from the published covariance trace,
  // halved because the induced sun-vector error is transverse to the pull rather
  // than the full eigenaxis rotation.
  const double cov_trace =
      this->state_.covariance
          .block<3, 3>(polaris::state::ErrorState::kAttitude, polaris::state::ErrorState::kAttitude)
          .trace();
  const double sigma_att =
      (std::isfinite(cov_trace) && cov_trace > 0.0) ? std::sqrt(cov_trace) : 0.0;
  const double attitude_driven = this->albedo_config_.albedo_error_rad * sigma_att * 0.5;
  this->setSunSigmaForCycle(std::hypot(this->sigma_sun_albedo_corr_rad_, attitude_driven),
                            ephemSigmaRad);
  return applied;
}

bool AttitudeEstimator ::refreshAlbedoConfig() {
  Fw::ParamValid peak_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid fov_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid boresight_valid = Fw::ParamValid::INVALID;
  const F64 peak = this->paramGet_SunAlbedoPeakRad(peak_valid);
  const F64 half_fov = this->paramGet_SunAlbedoHalfFovRad(fov_valid);
  const Vec3F64PerUnit boresights = this->paramGet_SunAlbedoBoresightsBody(boresight_valid);

  if (peak_valid != Fw::ParamValid::VALID || fov_valid != Fw::ParamValid::VALID ||
      boresight_valid != Fw::ParamValid::VALID) {
    this->failAlbedoConfig(
        "SunAlbedoPeakRad/SunAlbedoHalfFovRad/SunAlbedoBoresightsBody missing from ParameterDb");
    return false;
  }

  // Every slot must at least be *finite*; a NaN boresight would otherwise sit in
  // the array until a handoff selected its unit, and then silently disable the
  // correction on a cycle nobody was watching. Non-finiteness is a configuration
  // error, so it is reported now rather than discovered later.
  for (FwIndexType i = 0; i < NUM_SUNSENSORIN_INPUT_PORTS * 3; ++i) {
    if (!std::isfinite(boresights[i])) {
      this->failAlbedoConfig("SunAlbedoBoresightsBody contains a non-finite component");
      return false;
    }
    this->sun_boresights_[i] = boresights[i];
  }

  polaris::gnc::AlbedoCorrectionConfig cfg;
  cfg.albedo_error_rad = peak;
  cfg.half_fov_rad = half_fov;
  // Slot 0's boresight stands in for the range check below. It is the
  // solar-array normal — the unit the vehicle nominally flies on — so a vehicle
  // whose slot 0 is unusable has no working correction at all and should say so;
  // the other slots are checked per cycle by `sunBoresightFor`, where a zero
  // vector is a legitimate "not installed" rather than an error.
  if (!this->sunBoresightFor(0, cfg.boresight_body)) {
    this->failAlbedoConfig("SunAlbedoBoresightsBody slot 0 is not a usable direction");
    return false;
  }
  // isValid() carries the range checks (units, field width, a boresight that
  // names an axis) so the library and the component cannot disagree about what
  // a usable configuration is.
  if (!cfg.isValid()) {
    this->failAlbedoConfig("albedo tuning is out of range (see AlbedoCorrectionConfig::isValid)");
    return false;
  }

  this->albedo_config_ = cfg;
  this->albedo_configured_ = true;
  this->albedo_config_invalid_flagged_ = false;
  return true;
}

void AttitudeEstimator ::failAlbedoConfig(const char* detail) {
  this->albedo_configured_ = false;
  this->albedo_config_ = polaris::gnc::AlbedoCorrectionConfig{};
  if (!this->albedo_config_invalid_flagged_) {
    const Fw::LogStringArg reason(detail);
    this->log_WARNING_HI_AlbedoConfigInvalid(reason);
    this->albedo_config_invalid_flagged_ = true;
  }
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
  seed.observations[0].sigma_rad = this->sigma_sun_total_cycle_;
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
      {in.sun_body, in.sun_ref, this->sigma_sun_total_cycle_},
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

// Per-type unit selection and the fault-tolerant IMU vote live in
// AttitudeEstimatorSensors.cpp (§8.2).

// ----------------------------------------------------------------------
// Estimation cycle
// ----------------------------------------------------------------------

void AttitudeEstimator ::run_handler(FwIndexType portNum, U32 context) {
  const I64 nowNs = this->currentTaiNs();

  // A collection window is counted in *accepted* samples, so an outage stalls it
  // rather than ending it. Age it here, at the top and before any of the cycle's
  // early returns, so a window cannot outlive its deadline just because the
  // estimator spent the outage refusing cycles — which is exactly the case that
  // would otherwise leave the vehicle telemetering COLLECTING forever.
  if (this->mag_cal_collecting_) {
    ++this->mag_cal_cycles_;
    if (this->mag_cal_cycles_ >= this->mag_cal_deadline_cycles_) {
      // Fit what it has. Enough samples and it succeeds; too few and the usual
      // gates refuse it with SAMPLES, which is the honest report of an outage.
      this->finishMagCal();
    }
  }

  if (this->params_dirty_) {
    this->params_dirty_ = false;
    // Three independent gates, in decreasing order of what their failure costs.
    // The coarse set failing leaves the vehicle with no attitude at all; the fine
    // set failing leaves it with the Safe-mode floor, which is a working vehicle;
    // the albedo set failing leaves both running on the wider uncorrected sun
    // budget, which is how the vehicle flew before the correction existed. None
    // is allowed to take the ones above it down.
    (void)this->refreshCoarseConfig();  // emits ConfigInvalid and leaves us inert
    (void)this->refreshFineConfig();    // emits FineConfigInvalid; coarse-only
    (void)this->refreshAlbedoConfig();  // emits AlbedoConfigInvalid; uncorrected
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

  // --- Gyro: the fault-tolerant vote across the IMU suite (§8.2) -------------
  // Not "the first valid unit": with redundant gyros the combination has to be
  // robust, because a railed unit reports a plausible-looking valid flag and an
  // impossible rate. voteImuRate gates each unit, takes the per-axis median of
  // the survivors, and raises the exclusion FDIR edges.
  pm::Vec3<Body> voted_rate;
  if (this->voteImuRate(nowNs, voted_rate)) {
    in.gyro = voted_rate;
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
  // The **geocentric** sun direction, kept alongside the spacecraft-centric one
  // above: the albedo correction's dayside factor is the angle between the
  // spacecraft and the sub-solar point, which is a geocentric quantity, and
  // reusing the parallax-corrected vector there would be the wrong geometry
  // (harmlessly so at LEO, but wrong is wrong in a model that is subtracted).
  pm::Vec3<ECI> sun_geocentric;
  bool have_sun_geocentric = false;
  TableGrade::T ephem_grade = TableGrade::UNAVAILABLE;
  if (this->isConnected_getBodyPosition_OutputPort(0)) {
    PosEciMeters sunPos;
    if (this->getBodyPosition_out(0, OnboardBody::SUN, nowNs, sunPos)) {
      pm::Vec3<ECI> to_sun(sunPos.get_x(), sunPos.get_y(), sunPos.get_z());
      have_sun_geocentric = to_sun.normalized(sun_geocentric);
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
  // How well this cycle's sun *reference* is known, from the grade the ephemeris
  // query answered at. PRECISE means the uploaded DE440 Chebyshev tables covered
  // the epoch and the direction is arcsecond-class; anything else means the
  // analytic fallback answered, two orders of magnitude wider. The grade is a
  // fact about the current epoch and the current upload, not a configuration
  // choice, so it is read per cycle rather than latched.
  const double ephem_sigma_rad = (ephem_grade == TableGrade::PRECISE)
                                     ? this->sigma_sun_ephem_precise_rad_
                                     : this->sigma_sun_ephem_rad_;

  // Best-illuminated unit of the suite (§8.2), and its port index — which is
  // what selects that unit's albedo boresight below, so a handoff to a sensor on
  // another face corrects with that face's geometry rather than the previous
  // unit's.
  FwIndexType sun_index = 0;
  const SunSensorMeas* const sun = this->selectSunSensor(nowNs, sun_index);
  this->tlmWrite_SunUnitSelected(sun != nullptr ? static_cast<U8>(sun_index) : kNoSunUnitIndex);
  // Default for a cycle with no sun measurement at all: the uncorrected albedo
  // budget, so nothing downstream can read a corrected sigma off a cycle that
  // had no measurement to correct.
  this->setSunSigmaForCycle(this->sigma_sun_albedo_uncorr_rad_, ephem_sigma_rad);
  double albedo_applied_rad = kNoValue;
  if (sun != nullptr && have_sun_ref) {
    pm::Vec3<Body> sun_body(toEigen(sun->get_dirBody()));
    albedo_applied_rad = this->applyAlbedoCorrection(
        sun_index, r_ecef, q_eci_ecef, sun_geocentric, have_position && have_rotation,
        have_sun_geocentric, ephem_sigma_rad, sun_body);

    in.sun_body = sun_body;
    in.sun_ref = sun_ref;
    in.sun_valid = in.sun_body.isFinite();
    // Per-cycle sigma. Both estimators are told the same thing about the same
    // measurement: the coarse chain through the input override (it needs the
    // systematic part alone, which is its covariance floor) and the MEKF through
    // the inflated total it takes per update.
    in.sun_sigma_sys_rad = this->sigma_sun_sys_cycle_;
  }
  this->tlmWrite_SunAlbedoCorrection(albedo_applied_rad);
  const MagnetometerMeas* const magnetometer = this->selectMagnetometer(nowNs);
  if (magnetometer != nullptr && have_mag_ref) {
    const pm::Vec3<Body> m_raw(toEigen(magnetometer->get_fieldTesla()));

    // §7 MTQ/MAG duty-cycle interlock gate point. An energised torque rod puts a
    // field on the sensor orders of magnitude above the ~30 uT ambient, and the
    // core's hysteresis outlives the drive, so such a sample is not a
    // measurement of the geomagnetic field and must never reach the calibration
    // fit. No magnetorquer actuation state reaches this component today — the
    // interlock lands with the §8.5 control push that first drives MTQs and this
    // magnetometer in the same loop — so every selected sample is clean by
    // construction. When that state arrives it is anded in here, and nothing
    // else on this path changes.
    const bool mag_sample_clean = true;
    if (mag_sample_clean) {
      // The fit is fed the **raw** reading against the modelled field magnitude:
      // an accumulator fed its own correction would refit the identity. A cycle
      // with no position has no modelled field and so contributes no sample
      // rather than a fabricated one — which is what have_mag_ref above gates.
      this->collectMagSample(m_raw, mag_ref.eigen().norm());
    }

    // **The one application point.** Everything downstream of unit selection —
    // the coarse chain, the MEKF update, and the future B-dot law — reads
    // `in.mag_body`, so the correction is applied exactly here and no consumer
    // can disagree with another about what the field was. Called
    // unconditionally: applyMagCalibration passes the raw vector through while
    // no calibration is applied, so an uncalibrated vehicle takes the same path.
    in.mag_body = polaris::gnc::applyMagCalibration(this->mag_cal_, m_raw);
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

  // Calibration health. The state is derived rather than stored — COLLECTING
  // outranks APPLIED because a window opened over an applied calibration is the
  // condition worth seeing — and coverage is written *live* so the ground can
  // tell a window that will succeed from one that will be refused, while there
  // is still time to abort and re-fly it over a larger tumble.
  this->tlmWrite_MagCalState(this->mag_cal_collecting_ ? MagCalState::COLLECTING
                             : this->mag_cal_.valid    ? MagCalState::APPLIED
                                                       : MagCalState::IDLE);
  this->tlmWrite_MagCalSamples(
      this->mag_cal_collecting_ ? static_cast<U32>(this->mag_cal_accumulator_.sampleCount()) : 0u);
  this->tlmWrite_MagCalCoverage(this->mag_cal_collecting_ ? this->mag_cal_accumulator_.coverage()
                                                          : kNoValue);
  this->tlmWrite_MagCalResidualAngle(this->mag_cal_.valid ? this->mag_cal_.residual_angle_rad
                                                          : kNoValue);
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
  // The commanded re-admission path for the IMU vote (§8.2): an operator saying
  // "start over" is different information from a unit having behaved for N
  // cycles, so it clears every exclusion latch outright. A unit that is really
  // failed re-excludes on its next reading, which costs one EVR and tells the
  // ground the fault is persistent rather than latched.
  this->imu_voter_.clearExclusions();
  this->imu_ambiguous_cycles_ = 0;
  this->attitude_valid_ = false;
  this->last_age_s_ = 0.0;
  this->have_epoch_ = false;
  // A reset means "tell me everything again": every edge-gated alert is re-armed,
  // not just the configuration one, so a still-broken vehicle re-reports each
  // fault rather than staying quiet because it already mentioned it once.
  this->config_invalid_flagged_ = false;
  this->fine_config_invalid_flagged_ = false;
  this->fine_init_failed_flagged_ = false;
  this->imu_ambiguous_flagged_ = false;
  this->position_unavailable_flagged_ = false;
  this->igrf_stale_flagged_ = false;
  this->have_ephem_grade_ = false;
  this->have_eop_grade_ = false;
  // Re-read the tuning too: a reset is also the recovery path after a config fix.
  this->params_dirty_ = true;
  // Full reset semantics reach the calibration too: a window in progress is
  // abandoned and an applied calibration is dropped. RESET_ESTIMATOR means "the
  // solution is suspect, start over", and a correction derived from data the
  // operator no longer trusts is part of what is suspect. MAG_CAL_CLEAR is the
  // narrower command for dropping the calibration alone.
  this->mag_cal_collecting_ = false;
  this->mag_cal_target_samples_ = 0;
  this->mag_cal_deadline_cycles_ = 0;
  this->mag_cal_accumulator_.reset();
  this->clearMagCal();
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

// ======================================================================
// \title  AttitudeEstimatorMagCal.cpp
// \brief  Commanded magnetometer hard/soft-iron calibration on the attitude
//         estimator (design doc §8.1; supports REQ-ADET-005/006)
//
// The same `flight::AttitudeEstimator` class as AttitudeEstimator.cpp, split
// into its own translation unit because the estimation cycle and the
// calibration lifecycle are read at different times by different people and
// one file carrying both had outgrown the 500-line guidance badly.
//
// What lives here: reading the MagCal* tuning, the collection window, the fit
// and its refusal reporting, and the three MAG_CAL_* command handlers. What
// does not: the sampling tap and the single application point, which are inside
// run_handler's magnetometer block in AttitudeEstimator.cpp, because their
// whole value is that they sit exactly once in the measurement path where a
// reader of that path will see them.
//
// Every command handler here is `guarded` (see the FPP model): they run on the
// command-dispatcher thread and write state the rate-group thread reads.
// ======================================================================

#include <cmath>

#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"

namespace flight {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;

//! Number of F64 magnetometer-calibration parameters (MagCalMinSamples is U32
//! and is read separately). Validated at MAG_CAL_START, not per cycle.
constexpr FwSizeType kMagCalF64ParamCount = 6;

//! Map a library refusal onto the telemetered reason. `None` cannot reach here
//! (the caller only asks after a refusal) but is mapped rather than asserted on:
//! an EVR that says NUMERICAL is a better failure than an abort in flight.
MagCalRejectReason::T toRejectReason(polaris::gnc::MagCalibrationRefusal why) {
  switch (why) {
    case polaris::gnc::MagCalibrationRefusal::NotConfigured:
      return MagCalRejectReason::CONFIG;
    case polaris::gnc::MagCalibrationRefusal::Samples:
      return MagCalRejectReason::SAMPLES;
    case polaris::gnc::MagCalibrationRefusal::Coverage:
      return MagCalRejectReason::COVERAGE;
    case polaris::gnc::MagCalibrationRefusal::Condition:
      return MagCalRejectReason::CONDITION;
    case polaris::gnc::MagCalibrationRefusal::NoImprovement:
      return MagCalRejectReason::NO_IMPROVEMENT;
    default:
      return MagCalRejectReason::NUMERICAL;
  }
}

}  // namespace

// ----------------------------------------------------------------------
// Configuration and the collection lifecycle
// ----------------------------------------------------------------------

void AttitudeEstimator ::commandMagCalAtStartup(U32 sampleCount) {
  if (sampleCount == 0) {
    return;
  }
  Fw::CmdArgBuffer args;
  if (args.serializeFrom(sampleCount) != Fw::FW_SERIALIZE_OK) {
    return;
  }
  // Through the command port rather than straight into the handler, so the
  // registration, argument deserialisation and command-response path are the
  // ones that fly. The opcode is this component's own, hence this wrapper: the
  // generated constant is protected and the topology has no business knowing it.
  this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_MAG_CAL_START, 0, args);
}

bool AttitudeEstimator ::refreshMagCalConfig() {
  Fw::ParamValid valids[kMagCalF64ParamCount];
  F64 values[kMagCalF64ParamCount];
  values[0] = this->paramGet_MagCalNominalFieldT(valids[0]);
  values[1] = this->paramGet_MagCalMinFieldT(valids[1]);
  values[2] = this->paramGet_MagCalMaxFieldT(valids[2]);
  values[3] = this->paramGet_MagCalMinCoverage(valids[3]);
  values[4] = this->paramGet_MagCalMaxCondition(valids[4]);
  values[5] = this->paramGet_MagCalMinImprovement(valids[5]);

  for (FwSizeType i = 0; i < kMagCalF64ParamCount; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i])) {
      return false;
    }
  }
  Fw::ParamValid samples_valid = Fw::ParamValid::INVALID;
  const U32 min_samples = this->paramGet_MagCalMinSamples(samples_valid);
  if (samples_valid != Fw::ParamValid::VALID) {
    return false;
  }

  polaris::gnc::MagCalibrationConfig cfg;
  cfg.nominal_field_t = values[0];
  cfg.min_field_t = values[1];
  cfg.max_field_t = values[2];
  cfg.min_samples = static_cast<std::int32_t>(min_samples);
  cfg.min_coverage = values[3];
  cfg.max_condition = values[4];
  cfg.min_residual_improvement = values[5];

  // The library owns the range rules (MagCalibrationConfig::isValid), so the
  // component cannot drift a second, disagreeing idea of what is in range. A
  // count past INT32_MAX would wrap the cast above into a negative minimum,
  // which isValid() would then wave through as "at least 10 samples" — so it is
  // caught here, where the narrowing happens.
  if (min_samples > static_cast<U32>(kMaxCalSamples) || !cfg.isValid()) {
    return false;
  }

  // Rebuilding discards whatever a window had collected under the old gates —
  // which is why this is only ever called by MAG_CAL_START, i.e. at the moment
  // the accumulator is about to be reset anyway.
  this->mag_cal_accumulator_ = polaris::gnc::MagCalibrationAccumulator(cfg);
  return true;
}

void AttitudeEstimator ::collectMagSample(const pm::Vec3<Body>& m_raw, double igrf_magnitude_t) {
  if (!this->mag_cal_collecting_) {
    return;
  }
  // A rejected sample (out of band, non-finite) simply does not count toward the
  // target: the window is defined in *accepted* samples, so a noisy stretch
  // costs time rather than fit quality. The return code is checked in the sense
  // that matters — sampleCount() below is the accumulator's own answer.
  (void)this->mag_cal_accumulator_.addSample(m_raw, igrf_magnitude_t);
  if (static_cast<U32>(this->mag_cal_accumulator_.sampleCount()) >= this->mag_cal_target_samples_) {
    this->finishMagCal();
  }
}

void AttitudeEstimator ::finishMagCal() {
  if (!this->mag_cal_collecting_) {
    return;
  }
  this->mag_cal_collecting_ = false;
  const U32 samples = static_cast<U32>(this->mag_cal_accumulator_.sampleCount());
  const F64 coverage = this->mag_cal_accumulator_.coverage();

  polaris::gnc::MagCalibrationResult fit;
  polaris::gnc::MagCalibrationRefusal why = polaris::gnc::MagCalibrationRefusal::None;
  if (!this->mag_cal_accumulator_.solve(fit, why)) {
    // Nothing is applied and any previous calibration is retained: a refused fit
    // is no information about the calibration already flying, and throwing a
    // good correction away because a later window failed would be a regression
    // the operator never asked for.
    this->log_WARNING_HI_MagCalRejected(toRejectReason(why), samples, coverage);
    return;
  }
  this->mag_cal_ = fit;
  this->log_ACTIVITY_HI_MagCalComplete(fit.residual_angle_rad, fit.coverage, samples);
}

void AttitudeEstimator ::clearMagCal() {
  if (!this->mag_cal_.valid) {
    return;
  }
  this->mag_cal_ = polaris::gnc::MagCalibrationResult{};
  this->log_ACTIVITY_HI_MagCalCleared();
}

// ----------------------------------------------------------------------
// Command handler implementations
// ----------------------------------------------------------------------

void AttitudeEstimator ::MAG_CAL_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq,
                                                  U32 sampleCount) {
  // Tuning is read here rather than per cycle: a missing calibration parameter
  // costs only the ability to start a calibration, so it is a refused command,
  // never a flight event on a vehicle that is otherwise entirely healthy.
  if (!this->refreshMagCalConfig()) {
    this->log_WARNING_HI_MagCalRejected(MagCalRejectReason::CONFIG, 0, 0.0);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }
  Fw::ParamValid min_valid = Fw::ParamValid::INVALID;
  const U32 min_samples = this->paramGet_MagCalMinSamples(min_valid);
  // refreshMagCalConfig() succeeded, so the parameter is present and in range;
  // the read is repeated rather than plumbed out of it because the alternative
  // is an out-parameter that exists solely for this bound check.
  if (min_valid != Fw::ParamValid::VALID || sampleCount < min_samples ||
      sampleCount > kMaxCalSamples) {
    // Below the configured minimum the fit would be refused on SAMPLES at the
    // end of the window; saying so now costs the operator nothing but the
    // command, rather than the whole collection.
    this->log_WARNING_HI_MagCalRejected(MagCalRejectReason::SAMPLES, sampleCount, 0.0);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }

  // Restarting over an open window is deliberate: the operator commanding a new
  // collection means the old one, not two overlapping fits.
  this->mag_cal_accumulator_.reset();
  this->mag_cal_target_samples_ = sampleCount;
  this->mag_cal_cycles_ = 0;
  // No overflow: the range check above caps sampleCount at kMaxCalSamples, so
  // this product is at most 1e6.
  this->mag_cal_deadline_cycles_ = sampleCount * kMaxCalStallFactor;
  this->mag_cal_collecting_ = true;
  // The applied calibration (if any) is untouched and keeps correcting the
  // estimator's input for the whole window. The fit is on raw samples, so the
  // two do not interact.
  this->log_ACTIVITY_HI_MagCalStarted(sampleCount);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeEstimator ::MAG_CAL_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  // Idempotent: an abort with no window open is an operator making sure, and
  // answering OK is the honest response to "there is no collection running".
  if (this->mag_cal_collecting_) {
    const U32 samples = static_cast<U32>(this->mag_cal_accumulator_.sampleCount());
    this->mag_cal_collecting_ = false;
    this->mag_cal_target_samples_ = 0;
    this->mag_cal_deadline_cycles_ = 0;
    this->mag_cal_accumulator_.reset();
    this->log_ACTIVITY_HI_MagCalAborted(samples);
  }
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeEstimator ::MAG_CAL_CLEAR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  // A window in progress is left alone on purpose: clearing the correction that
  // is flying and deciding to fit a new one are separate decisions, and the fit
  // runs on raw samples either way.
  this->clearMagCal();
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}
}  // namespace flight

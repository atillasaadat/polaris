// ======================================================================
// \title  AttitudeEstimatorStarTracker.cpp
// \brief  Star-tracker fusion, the mode ladder's residual monitors, and the
//         commanded inter-tracker alignment calibration (design doc §8.2;
//         REQ-ADET-007, REQ-ADET-012, REQ-ADET-013)
//
// The same `flight::AttitudeEstimator` class as AttitudeEstimator.cpp, split
// into its own translation unit for the same reason the magnetometer
// calibration and the multi-unit selection were: this is the top rung of the
// §8.2 ladder and it is read on its own.
//
// What lives here: the king-tracker architecture (which unit defines the body
// frame, and what that means for everything else), gathering and correcting the
// per-unit solutions, building each unit's anisotropic measurement covariance,
// the attitude updates into the MEKF, the residual monitors the demoted sun and
// magnetic pairs become, and the ST_ALIGN_CAL_* command flow.
//
// What does not: the arbitration that decides *whether* the filter runs at all
// (AttitudeEstimator.cpp), and the estimation cycle that calls into here — whose
// value is that the ladder is visible in one place to a reader of that cycle.
//
// Every function here runs either on the rate-group thread inside the `guarded`
// `run` port or in a `guarded` command handler, so the alignment state they
// share is serialised on the component mutex.
// ======================================================================

#include <cmath>
#include <cstdio>
#include <Eigen/Cholesky>
#include <limits>

#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"
#include "Fw/Log/LogString.hpp"

namespace flight {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;
using ECI = pm::frames::ECI;

constexpr I64 kNsPerSecond = 1000000000LL;

//! Not a value, for telemetry channels with nothing to report this cycle.
const double kNoValueSt = std::numeric_limits<double>::quiet_NaN();

//! True if @p timeTagNs is within @p maxAgeS of @p nowTaiNs (§9.1). Duplicated
//! from the other two translation units deliberately: it is four lines, and a
//! shared internal header for it would be more machinery than the rule it
//! encodes.
bool fresh(I64 nowTaiNs, I64 timeTagNs, F64 maxAgeS) {
  if (!(maxAgeS > 0.0)) {
    return false;
  }
  const I64 delta = nowTaiNs - timeTagNs;
  const I64 limit = static_cast<I64>(maxAgeS * static_cast<F64>(kNsPerSecond));
  return delta >= -limit && delta <= limit;
}

//! Angle between two directions, from the cross/dot form rather than an acos of
//! the dot product (§3.5, lib/README.md): `acos` near 0 or π returns an angle
//! wrong by ~1e-8 rad however exact its argument, and these residuals are read at
//! the milliradian level and gated at it.
double angleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

//! Map a library alignment refusal onto the telemetered reason.
StAlignRejectReason::T toStAlignReason(polaris::gnc::StAlignmentRejection why) {
  switch (why) {
    case polaris::gnc::StAlignmentRejection::kConfig:
      return StAlignRejectReason::CONFIG;
    case polaris::gnc::StAlignmentRejection::kSamples:
      return StAlignRejectReason::SAMPLES;
    case polaris::gnc::StAlignmentRejection::kDegenerate:
      return StAlignRejectReason::DEGENERATE;
    case polaris::gnc::StAlignmentRejection::kDispersion:
      return StAlignRejectReason::DISPERSION;
    default:
      // kNumerical, and anything a future library revision adds. Reporting a
      // numerical failure is the conservative default: it says "do not simply
      // re-fly this", which is right for every unknown case.
      return StAlignRejectReason::NUMERICAL;
  }
}

}  // namespace

// ----------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------

void AttitudeEstimator ::commandStAlignCalAtStartup(U8 unit, U32 sampleCount) {
  if (sampleCount == 0) {
    return;
  }
  Fw::CmdArgBuffer args;
  if (args.serializeFrom(unit) != Fw::FW_SERIALIZE_OK ||
      args.serializeFrom(sampleCount) != Fw::FW_SERIALIZE_OK) {
    return;
  }
  // Through the command port rather than straight into the handler, so the
  // registration, argument deserialisation and command-response path are the ones
  // that fly. The opcode is this component's own, hence this wrapper: the
  // generated constant is protected and the topology has no business knowing it.
  this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_ST_ALIGN_CAL_START, 0, args);
}

bool AttitudeEstimator ::refreshStConfig() {
  this->st_configured_ = false;

  Fw::ParamValid valids[5];
  F64 values[5];
  values[0] = this->paramGet_StSigmaXyRad(valids[0]);
  values[1] = this->paramGet_StSigmaZRad(valids[1]);
  values[2] = this->paramGet_MonitorSunResidualRad(valids[2]);
  values[3] = this->paramGet_MonitorMagResidualRad(valids[3]);
  values[4] = this->paramGet_MonitorSunCrossUnitRad(valids[4]);
  static const char* const kNames[5] = {"StSigmaXyRad", "StSigmaZRad", "MonitorSunResidualRad",
                                        "MonitorMagResidualRad", "MonitorSunCrossUnitRad"};
  for (FwSizeType i = 0; i < 5; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i]) || !(values[i] > 0.0)) {
      char detail[80];
      (void)std::snprintf(detail, sizeof(detail), "%s missing or not positive in ParameterDb",
                          kNames[i]);
      this->failStConfig(detail);
      return false;
    }
  }
  // A tracker tighter about its boresight than across it is a configuration
  // error, not an unusual unit: the identified stars barely move under a rotation
  // about the boresight, so that direction is always the loose one. Flying the
  // pair inverted would report a covariance tighter than the truth in exactly the
  // direction that is weakest — and the whole dual-tracker argument rests on
  // knowing which direction that is.
  if (!(values[1] >= values[0])) {
    this->failStConfig("need StSigmaZRad >= StSigmaXyRad (about-boresight is the loose axis)");
    return false;
  }

  Fw::ParamValid king_valid = Fw::ParamValid::INVALID;
  const U32 king = this->paramGet_StKingUnit(king_valid);
  if (king_valid != Fw::ParamValid::VALID ||
      king >= static_cast<U32>(NUM_STARTRACKERIN_INPUT_PORTS)) {
    this->failStConfig("StKingUnit missing or outside the starTrackerIn port array");
    return false;
  }

  Fw::ParamValid cycles_valid = Fw::ParamValid::INVALID;
  const U32 monitor_cycles = this->paramGet_MonitorAlertCycles(cycles_valid);
  if (cycles_valid != Fw::ParamValid::VALID || monitor_cycles == 0) {
    this->failStConfig("MonitorAlertCycles missing or zero");
    return false;
  }

  Fw::ParamValid gate_valid = Fw::ParamValid::INVALID;
  const F64 agreement_gate = this->paramGet_StCoarseAgreementGate(gate_valid);
  if (gate_valid != Fw::ParamValid::VALID || !std::isfinite(agreement_gate) ||
      !(agreement_gate > 0.0)) {
    this->failStConfig("StCoarseAgreementGate missing or not positive");
    return false;
  }

  Fw::ParamValid readmit_valid = Fw::ParamValid::INVALID;
  const U32 readmit_cycles = this->paramGet_StReadmitCycles(readmit_valid);
  if (readmit_valid != Fw::ParamValid::VALID || readmit_cycles == 0) {
    this->failStConfig("StReadmitCycles missing or zero");
    return false;
  }

  Fw::ParamValid boresights_valid = Fw::ParamValid::INVALID;
  const Vec3F64PerUnit boresights = this->paramGet_StBoresightsBody(boresights_valid);
  if (boresights_valid != Fw::ParamValid::VALID) {
    this->failStConfig("StBoresightsBody missing from ParameterDb");
    return false;
  }
  // Every slot must at least be *finite*; a NaN boresight would otherwise sit in
  // the array until that unit delivered a solution, and then silently drop it from
  // the fusion on a cycle nobody was watching.
  for (FwIndexType i = 0; i < NUM_STARTRACKERIN_INPUT_PORTS * 3; ++i) {
    const FwSizeType slot = static_cast<FwSizeType>(i);
    if (!std::isfinite(boresights[slot])) {
      this->failStConfig("StBoresightsBody contains a non-finite component");
      return false;
    }
    this->st_boresights_[slot] = boresights[slot];
  }

  this->st_king_unit_ = static_cast<FwIndexType>(king);
  this->sigma_st_xy_rad_ = values[0];
  this->sigma_st_z_rad_ = values[1];
  this->monitor_threshold_rad_[ResidualMonitor::SUN] = values[2];
  this->monitor_threshold_rad_[ResidualMonitor::MAGNETOMETER] = values[3];
  this->monitor_threshold_rad_[ResidualMonitor::SUN_CROSS_UNIT] = values[4];
  this->monitor_alert_cycles_ = monitor_cycles;
  this->st_coarse_agreement_gate_ = agreement_gate;
  this->st_readmit_cycles_ = readmit_cycles;
  this->st_configured_ = true;
  this->st_config_invalid_flagged_ = false;
  return true;
}

void AttitudeEstimator ::failStConfig(const char* detail) {
  if (!this->st_config_invalid_flagged_) {
    const Fw::LogStringArg arg(detail);
    this->log_WARNING_HI_StConfigInvalid(arg);
    this->st_config_invalid_flagged_ = true;
  }
  this->st_configured_ = false;
}

bool AttitudeEstimator ::stBoresightFor(FwIndexType index, pm::Vec3<Body>& out) const {
  if (index < 0 || index >= NUM_STARTRACKERIN_INPUT_PORTS) {
    return false;
  }
  const FwIndexType base = static_cast<FwIndexType>(index * 3);
  const Eigen::Vector3d v(this->st_boresights_[base], this->st_boresights_[base + 1],
                          this->st_boresights_[base + 2]);
  // The zero vector is the configured "no value for this index" — a unit that is
  // not installed, or one whose mounting has not been characterised. It cannot be
  // mistaken for a direction, which is why it was chosen over a sentinel, and
  // `Vec3::normalized` refuses it, so the two checks are one call.
  pm::Vec3<Body> unit;
  if (!v.allFinite() || !pm::Vec3<Body>(v).normalized(unit)) {
    return false;
  }
  out = unit;
  return true;
}

// ----------------------------------------------------------------------
// Gathering and fusing the tracker solutions (§8.2)
// ----------------------------------------------------------------------

int AttitudeEstimator ::collectStarTrackers(I64 nowTaiNs, StarTrackerSample* out,
                                            U32& validMask) const {
  validMask = 0;
  if (!this->st_configured_ || out == nullptr) {
    return 0;
  }

  int count = 0;
  // The king first, then the rest in port order. The Kalman update is
  // order-independent to first order, so this buys determinism rather than
  // accuracy — but a filter whose answer depends on port-scan order is a filter
  // whose regression tests are luck.
  for (int pass = 0; pass < 2; ++pass) {
    for (FwIndexType i = 0; i < NUM_STARTRACKERIN_INPUT_PORTS; ++i) {
      const bool is_king = (i == this->st_king_unit_);
      if ((pass == 0) != is_king) {
        continue;
      }
      const StarTrackerMeas& m = this->star_[i];
      if (!m.get_valid() || !fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
        continue;
      }
      pm::Vec3<Body> boresight;
      if (!this->stBoresightFor(i, boresight)) {
        // No characterised mounting, so no measurement covariance can be built.
        // Fusing it isotropically instead would be inventing the one number the
        // dual-tracker argument turns on, so the unit is simply not fused.
        continue;
      }
      const QuatF64& q = m.get_qBodyEci();
      pm::Quaternion core(q[0], q[1], q[2], q[3]);
      if (!core.isFinite() || !core.normalize()) {
        continue;
      }
      pm::Quat<Body, ECI> attitude(core.canonical());

      // The unit produced a usable solution, so it is reported on StValidMask
      // **whether or not it is fused**. The two eligibility gates below are
      // deliberately applied after this: reporting a healthy-but-not-fused unit as
      // if it had stopped solving would hide the exact gap the ground reads
      // StValidMask against StContributing to see.
      validMask |= (1u << static_cast<U32>(i));

      // **A non-king unit is not fused until it has been calibrated against the
      // king.** Its as-mounted reading carries the *difference* of the two units'
      // fixed biases — 45-110 arcsec on the reference vehicle's own campaign —
      // and fusing it at the ~21.5 arcsec sigma the configuration declares would
      // sell a systematic as white noise, which is precisely the overconfidence
      // the white-R model cannot represent. Flying king-only until ST_ALIGN_CAL
      // runs is the honest launch state, and it is what makes ST_ALIGN_CAL_CLEAR
      // a safe command rather than one that silently degrades the solution.
      if (i != this->st_king_unit_ && !this->st_align_[i].valid) {
        continue;
      }
      // Latched out by the per-unit NIS policy: a unit the filter has persistently
      // disbelieved is dropped from the fusion rather than allowed to demote the
      // mode (@ref readmitStarTrackers earns it back).
      if (this->st_excluded_[i]) {
        continue;
      }

      // **One application point for the inter-tracker alignment**, exactly as the
      // magnetometer calibration has one: everything downstream — the filter, the
      // residual monitors, any future consumer — reads the corrected solution, so
      // none of them can disagree about which frame a tracker was in. Called
      // unconditionally; `applyStAlignment` passes the reading through while
      // nothing is fitted, and the king's slot is never fitted at all.
      attitude = polaris::gnc::applyStAlignment(this->st_align_[i], attitude);

      // R = σ_xy²(I − b bᵀ) + σ_z² b bᵀ: isotropic across the boresight, loose
      // about it. This closed form is why only the boresight is configured and
      // not a full mounting DCM — the covariance is symmetric about the boresight,
      // so the roll of the sensor within its own mount cannot affect it.
      const Eigen::Vector3d b = boresight.eigen();
      const Eigen::Matrix3d bbt = b * b.transpose();
      out[count].attitude = attitude;
      out[count].noise_cov =
          (this->sigma_st_xy_rad_ * this->sigma_st_xy_rad_) * (Eigen::Matrix3d::Identity() - bbt) +
          (this->sigma_st_z_rad_ * this->sigma_st_z_rad_) * bbt;
      out[count].index = i;
      ++count;
    }
  }
  return count;
}

double AttitudeEstimator ::fuseStarTrackers(const StarTrackerSample* samples, int count,
                                            bool& accepted, bool& refused, bool& nisRejected) {
  accepted = false;
  double worst_nis = kNoValueSt;
  if (samples == nullptr) {
    return worst_nis;
  }
  for (int i = 0; i < count; ++i) {
    if (!this->mekf_.isInitialised()) {
      break;
    }
    const U32 rejected_before = this->mekf_.rejectedCount();
    polaris::gnc::MekfUpdate diagnostics;
    const bool force =
        this->st_meas_mode_ == static_cast<U8>(polaris::gnc::MeasurementMode::kForce);
    const bool applied = this->mekf_.updateAttitude(samples[i].attitude, samples[i].noise_cov,
                                                    diagnostics, force);  // TP §9.1
    // Same three-way distinction the vector path makes, and for the same reason:
    // a gate rejection and a malformed measurement both return false and mean
    // opposite things — one is the divergence guard working, the other is the
    // filter refusing input it cannot use — and only the rejection counter tells
    // them apart, which is what keeps the two demotion streaks meaningful.
    const bool gate_rejected = !applied && this->mekf_.rejectedCount() > rejected_before;
    if (!applied && !gate_rejected) {
      refused = true;
      continue;
    }

    // **The NIS streak is per unit, not per cycle.** A cycle-global streak ORs
    // every tracker's verdict together, so one persistently-bad unit demotes the
    // whole fine mode — which drops the filter, re-promotes off the *same* bad
    // unit, and flaps. The unit the filter disbelieves is the thing to isolate,
    // exactly as a disagreeing IMU or magnetometer is; the mode is not.
    const FwIndexType unit = samples[i].index;
    if (unit >= 0 && unit < NUM_STARTRACKERIN_INPUT_PORTS) {
      if (gate_rejected) {
        ++this->st_nis_streak_[unit];
        this->st_accepted_streak_[unit] = 0;
        // **And the streak only *convicts* while the solution is tracker-sourced**
        // (Push 53). Rejection against a SS+MAG solution is not evidence about the
        // tracker: that solution's error is the magnetometer's systematic, which
        // the white-`R` model has averaged the covariance down through, so the gate
        // rejects any tracker however good — and the exclusion is then permanent,
        // because @ref readmitStarTrackers judges against the same solution. The
        // streak still runs; what it *means* off the tracker rung is decided by
        // @ref arbitrateRejectedTrackers against the coarse covariance instead.
        if (this->fine_source_ == FineSource::STAR_TRACKER &&
            this->st_nis_streak_[unit] >= this->nis_streak_limit_ && !this->st_excluded_[unit]) {
          this->st_excluded_[unit] = true;
          this->log_WARNING_HI_StUnitExcluded(static_cast<U8>(unit), this->st_nis_streak_[unit]);
        }
      } else if (applied) {
        this->st_nis_streak_[unit] = 0;
      }
    }

    nisRejected = nisRejected || gate_rejected;
    accepted = accepted || applied;
    if (std::isnan(worst_nis) || diagnostics.nis > worst_nis) {
      worst_nis = diagnostics.nis;
    }
  }
  return worst_nis;
}

bool AttitudeEstimator ::arbitrateRejectedTrackers(const polaris::time::Tai& epoch,
                                                   const polaris::gnc::CoarseAttitudeOutput& coarse,
                                                   const StarTrackerSample* samples, int count) {
  if (samples == nullptr || !coarse.attitude_valid) {
    return false;  // no honest reference this cycle, so no verdict either way
  }
  // The coarse chain's covariance is the one attitude uncertainty on the vehicle
  // that is not optimistic: it is blended with a **systematic floor** and
  // converges to it rather than to zero (§8.1), which is exactly the property the
  // filter's own covariance lacks and the reason this comparison exists.
  //
  // Compared in the **Mahalanobis** metric rather than against an isotropic
  // multiple of the trace. The coarse covariance is not isotropic — the roll
  // about the sun line is the loose axis and grows as `1/sin²θ` with the sun/field
  // separation — so a trace-derived radius is simultaneously too tight across it
  // and too loose along it, i.e. wrong in both directions at once. It also makes
  // the gate self-correcting through a coast: the covariance grows, so the gate
  // widens with the honest uncertainty instead of holding a fixed radius.
  //
  // The factorisation is also the positive-definiteness check the metric needs
  // (an indefinite `P` makes `d²` meaningless, not merely large), so the two are
  // one call.
  const Eigen::LDLT<Eigen::Matrix3d> ldlt(coarse.covariance);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return false;
  }

  // A coarse fix this cycle, or one fresh enough to still be a measurement rather
  // than a propagation. **Only the refusal needs it.** Coast growth is a stated
  // *lower* bound on the true uncertainty, so a coasted covariance can still be
  // narrower than the truth — which would make a refusal (and the alert that goes
  // with it) an accusation built on a number known to be optimistic. Adoption is
  // allowed on any valid coarse solution because it errs the permissive way: the
  // worst case is moving to a source the ladder already calls better.
  const bool coarse_is_fresh = coarse.triad_applied || coarse.age_s <= this->max_meas_age_s_;

  for (int i = 0; i < count; ++i) {
    const FwIndexType unit = samples[i].index;
    if (unit < 0 || unit >= NUM_STARTRACKERIN_INPUT_PORTS ||
        this->st_nis_streak_[unit] < this->nis_streak_limit_) {
      continue;
    }
    // `canonical()` guarantees q0 >= 0, so the atan2 form needs no `fabs`.
    const pm::Quaternion error =
        (samples[i].attitude.core() * coarse.attitude.core().inverse()).canonical();
    const double separation = 2.0 * std::atan2(error.vec().norm(), error.scalar());
    // Small-angle attitude error the coarse covariance is expressed on:
    // `δθ = 2·δq_v` to first order, which is the same linearisation the filter's
    // own error state uses.
    const Eigen::Vector3d dtheta = 2.0 * error.vec();
    const double mahalanobis = dtheta.dot(ldlt.solve(dtheta));
    if (!std::isfinite(mahalanobis)) {
      continue;
    }

    if (mahalanobis > this->st_coarse_agreement_gate_) {
      // Outside what the vector data supports. **Nothing is latched.** An
      // exclusion here would be convicted on agreement with the *coarse* solution
      // and paroled on agreement with the *fine* one (readmitStarTrackers), and a
      // criterion mismatch like that is a life sentence — the catalog's first
      // FDIR rule, and the one this branch used to break. The unit simply is not
      // adopted this cycle and stays a candidate, so a later cycle with a better
      // reference (a coarse fix on cleaner geometry, or the other tracker seeding
      // the filter) can still take it.
      //
      // Reported at a bounded cadence rather than once, since a permanent
      // condition reported once is a warning and then silence for the flight.
      if (coarse_is_fresh && this->monitor_alert_cycles_ > 0 &&
          ((this->st_nis_streak_[unit] - this->nis_streak_limit_) % this->monitor_alert_cycles_) ==
              0) {
        this->log_WARNING_HI_FineTrackerAdoptionRefused(static_cast<U8>(unit), separation,
                                                        mahalanobis);
      }
      continue;
    }

    // Inside the gate: the tracker agrees with everything the vector data
    // supports, so the filter is the outlier. Adopt the better source — the same
    // seed path a cold promotion takes, bias zeroed at the turn-on sigma for the
    // same reason (nothing better is known, and inventing a bias is worse).
    const Eigen::Matrix3d seed_bias_cov =
        (this->bias_sigma_init_ * this->bias_sigma_init_) * Eigen::Matrix3d::Identity();
    if (!this->mekf_.initialize(epoch, samples[i].attitude, samples[i].noise_cov,
                                pm::Vec3<Body>(Eigen::Vector3d::Zero()), seed_bias_cov)) {
      continue;  // the filter refused the seed; leave the streak standing
    }
    // The NIS streaks go with it: nothing has been rejected by the *new* filter,
    // and leaving them set would let the very next rejection re-fire this at once.
    // `refusal_streak_` deliberately does **not** — it counts the filter refusing
    // calls it cannot use (a propagate failure, a malformed measurement), which a
    // re-seed neither fixes nor is evidence against, and clearing it would hide a
    // numerics fault behind an unrelated mode transition.
    this->st_nis_streak_[unit] = 0;
    this->st_accepted_streak_[unit] = 0;
    this->nis_streak_ = 0;
    this->log_WARNING_LO_FineReseededFromStarTracker(static_cast<U8>(unit), separation,
                                                     mahalanobis);
    return true;
  }
  return false;
}

// ----------------------------------------------------------------------
// Residual monitors on the demoted sources (§8.2, §9.2)
// ----------------------------------------------------------------------

void AttitudeEstimator ::readmitStarTrackers(I64 nowTaiNs) {
  if (!this->st_configured_ || !this->mekf_.attitudeValid()) {
    return;
  }
  // **Judged on the criterion that excluded it**, exactly as the vote judges an
  // outvoted unit: the tracker was excluded for disagreeing with the filter, so it
  // earns its way back by agreeing with it. It is *not* fused while on probation —
  // that is what "excluded" means — so this is a shadow comparison against the
  // solution the surviving trackers built, which is the only independent evidence
  // available.
  //
  // The threshold is the unit's **own 3σ about its weak axis**, derived from the
  // same configuration that builds its R, not one of the residual-monitor
  // thresholds. Those are sized for a sun sensor or a magnetometer — degrees — and
  // an arcsecond-class instrument that is degrees out would sail through one and
  // earn its way back while still grossly wrong, which is how a re-admission
  // policy quietly becomes a no-op.
  const double readmit_threshold = 3.0 * this->sigma_st_z_rad_;

  const pm::Quat<Body, ECI> solution = this->mekf_.attitude();
  for (FwIndexType i = 0; i < NUM_STARTRACKERIN_INPUT_PORTS; ++i) {
    if (!this->st_excluded_[i]) {
      continue;
    }
    const StarTrackerMeas& m = this->star_[i];
    if (!m.get_valid() || !fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      // Absence is not agreement. A unit cannot serve out its exclusion by going
      // quiet, which is the same rule the vote applies to a dropped-out unit.
      this->st_accepted_streak_[i] = 0;
      continue;
    }
    const QuatF64& q = m.get_qBodyEci();
    pm::Quaternion core(q[0], q[1], q[2], q[3]);
    if (!core.isFinite() || !core.normalize()) {
      this->st_accepted_streak_[i] = 0;
      continue;
    }
    pm::Quat<Body, ECI> attitude =
        polaris::gnc::applyStAlignment(this->st_align_[i], pm::Quat<Body, ECI>(core.canonical()));
    const pm::Quaternion error = (attitude.core() * solution.core().inverse()).canonical();
    const double angle = 2.0 * std::atan2(error.vec().norm(), std::fabs(error.scalar()));
    if (angle <= readmit_threshold) {
      ++this->st_accepted_streak_[i];
      if (this->st_accepted_streak_[i] >= this->st_readmit_cycles_) {
        this->st_excluded_[i] = false;
        this->st_nis_streak_[i] = 0;
        this->st_accepted_streak_[i] = 0;
        this->log_ACTIVITY_HI_StUnitReadmitted(static_cast<U8>(i));
      }
    } else {
      this->st_accepted_streak_[i] = 0;
    }
  }
}

void AttitudeEstimator ::noteMonitorResidual(ResidualMonitor::T monitor, double residualRad,
                                             double thresholdRad) {
  const int index = static_cast<int>(monitor);
  if (index < 0 || index >= kMonitorCount) {
    return;
  }
  if (!std::isfinite(residualRad) || !(thresholdRad > 0.0)) {
    // The monitor could not run this cycle (the source was absent, or the fine
    // solution was not available to compare against). The streak resets — a run
    // of exceedances broken by a gap is not a run — but an existing alert is
    // **not** cleared, because "we stopped looking" is not "it recovered", and
    // clearing on absence would let a faulted sensor close its own alert by
    // dropping out.
    this->monitor_streak_[index] = 0;
    return;
  }

  if (residualRad > thresholdRad) {
    ++this->monitor_streak_[index];
    if (this->monitor_alert_cycles_ > 0 &&
        (this->monitor_streak_[index] % this->monitor_alert_cycles_) == 0) {
      // Modular rather than latched, so the alert repeats at a bounded cadence
      // while the condition lasts — the same shape as the IMU ambiguity
      // escalation, and for the same reason: an edge-gated alert on a permanent
      // fault is one warning and then silence.
      this->log_WARNING_HI_ResidualMonitorAlert(monitor, residualRad, thresholdRad);
      this->monitor_alerted_[index] = true;
    }
    return;
  }

  this->monitor_streak_[index] = 0;
  if (this->monitor_alerted_[index]) {
    this->log_ACTIVITY_HI_ResidualMonitorCleared(monitor);
    this->monitor_alerted_[index] = false;
  }
}

void AttitudeEstimator ::updateResidualMonitors(const polaris::gnc::CoarseAttitudeInput& in,
                                                const pm::Quat<Body, ECI>& fine, bool active) {
  double sun_residual = kNoValueSt;
  double mag_residual = kNoValueSt;

  // Only meaningful while the pairs are *demoted*. On a cycle where the filter is
  // updating from them, the residual is small because the update made it small —
  // a circular check that would read healthy on a drifting sensor, which is the
  // exact failure this monitor exists to catch.
  if (active) {
    if (in.sun_valid) {
      sun_residual = angleBetween(in.sun_body.eigen(), fine.rotate(in.sun_ref).eigen());
    }
    if (in.mag_valid) {
      mag_residual = angleBetween(in.mag_body.eigen(), fine.rotate(in.mag_ref).eigen());
    }
  }

  this->noteMonitorResidual(ResidualMonitor::SUN, sun_residual,
                            this->monitor_threshold_rad_[ResidualMonitor::SUN]);
  this->noteMonitorResidual(ResidualMonitor::MAGNETOMETER, mag_residual,
                            this->monitor_threshold_rad_[ResidualMonitor::MAGNETOMETER]);
  this->tlmWrite_SunResidualRad(sun_residual);
  this->tlmWrite_MagResidualRad(mag_residual);
}

// ----------------------------------------------------------------------
// Commanded inter-tracker alignment calibration (§8.2)
// ----------------------------------------------------------------------

bool AttitudeEstimator ::refreshStAlignConfig() {
  Fw::ParamValid residual_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid gap_valid = Fw::ParamValid::INVALID;
  Fw::ParamValid samples_valid = Fw::ParamValid::INVALID;
  const F64 max_residual = this->paramGet_StAlignMaxResidualRad(residual_valid);
  const F64 min_gap = this->paramGet_StAlignMinEigenGap(gap_valid);
  const U32 min_samples = this->paramGet_StAlignMinSamples(samples_valid);
  if (residual_valid != Fw::ParamValid::VALID || gap_valid != Fw::ParamValid::VALID ||
      samples_valid != Fw::ParamValid::VALID || !std::isfinite(max_residual) ||
      !std::isfinite(min_gap)) {
    return false;
  }

  polaris::gnc::StAlignmentConfig cfg;
  cfg.min_samples = min_samples;
  cfg.max_residual_rad = max_residual;
  cfg.min_eigen_gap = min_gap;
  // The library owns the range rules, so the component cannot drift a second,
  // disagreeing idea of what is in range. The sample ceiling is checked here,
  // where the command's own bound lives.
  if (min_samples > kMaxCalSamples || !cfg.isValid()) {
    return false;
  }

  this->st_align_accumulator_ = polaris::gnc::StAlignmentAccumulator(cfg);
  return true;
}

void AttitudeEstimator ::collectStAlignSample(I64 nowTaiNs) {
  if (!this->st_align_collecting_ || !this->st_configured_) {
    return;
  }
  const FwIndexType king = this->st_king_unit_;
  const FwIndexType unit = this->st_align_unit_;
  if (king < 0 || king >= NUM_STARTRACKERIN_INPUT_PORTS || unit < 0 ||
      unit >= NUM_STARTRACKERIN_INPUT_PORTS) {
    return;
  }

  const StarTrackerMeas& king_meas = this->star_[king];
  const StarTrackerMeas& unit_meas = this->star_[unit];
  // **Simultaneity is the whole measurement.** Both units must have delivered a
  // fresh valid solution this cycle; a pair straddling a cycle boundary smears the
  // vehicle's own rotation into the estimate, and at 0.1 °/s one cycle is already
  // 36 arcsec — comparable to the misalignment being measured. A cycle where only
  // one unit solved costs the window a pair, not its correctness.
  if (!king_meas.get_valid() || !unit_meas.get_valid() ||
      !fresh(nowTaiNs, king_meas.get_timeTagNs(), this->max_meas_age_s_) ||
      !fresh(nowTaiNs, unit_meas.get_timeTagNs(), this->max_meas_age_s_)) {
    return;
  }

  const QuatF64& qk = king_meas.get_qBodyEci();
  const QuatF64& qu = unit_meas.get_qBodyEci();
  // **Uncorrected readings.** The fit must never see its own correction, on the
  // same rule as the magnetometer calibration: an accumulator fed a corrected
  // sample refits the identity and reports a beautiful residual for a correction
  // that has stopped tracking the hardware.
  const pm::Quat<Body, ECI> king_attitude(pm::Quaternion(qk[0], qk[1], qk[2], qk[3]));
  const pm::Quat<Body, ECI> unit_attitude(pm::Quaternion(qu[0], qu[1], qu[2], qu[3]));
  // The return code is checked in the sense that matters — sampleCount() below is
  // the accumulator's own answer, and a refused sample simply does not count
  // toward the target.
  (void)this->st_align_accumulator_.addSample(king_attitude, unit_attitude);
  if (this->st_align_accumulator_.sampleCount() >= this->st_align_target_samples_) {
    this->finishStAlign();
  }
}

void AttitudeEstimator ::finishStAlign() {
  if (!this->st_align_collecting_) {
    return;
  }
  this->st_align_collecting_ = false;
  const FwIndexType unit = this->st_align_unit_;
  const U32 samples = this->st_align_accumulator_.sampleCount();

  polaris::gnc::StAlignmentResult fit;
  const polaris::gnc::StAlignmentRejection why = this->st_align_accumulator_.fit(fit);
  if (why != polaris::gnc::StAlignmentRejection::kNone) {
    // Nothing is applied and any previous alignment for this unit is retained: a
    // refused fit is no information about the correction already flying, and
    // throwing a good one away because a later window failed would be a regression
    // the operator never asked for.
    this->log_WARNING_HI_StAlignRejected(static_cast<U8>(unit), toStAlignReason(why), samples);
    return;
  }
  if (unit >= 0 && unit < NUM_STARTRACKERIN_INPUT_PORTS) {
    this->st_align_[unit] = fit;
  }
  this->log_ACTIVITY_HI_StAlignComplete(static_cast<U8>(unit), fit.residual_angle_rad,
                                        fit.misalignment_angle_rad, samples);
}

void AttitudeEstimator ::clearStAlign(FwIndexType unit) {
  if (unit < 0 || unit >= NUM_STARTRACKERIN_INPUT_PORTS || !this->st_align_[unit].valid) {
    return;
  }
  this->st_align_[unit] = polaris::gnc::StAlignmentResult{};
  this->log_ACTIVITY_HI_StAlignCleared(static_cast<U8>(unit));
}

// ----------------------------------------------------------------------
// Command handlers
// ----------------------------------------------------------------------

void AttitudeEstimator ::ST_ALIGN_CAL_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 unit,
                                                       U32 sampleCount) {
  // Tuning is read here rather than per cycle: a missing alignment parameter
  // costs only the ability to start a calibration, so it is a refused command and
  // never a flight event on an otherwise healthy vehicle.
  if (!this->refreshStAlignConfig()) {
    this->log_WARNING_HI_StAlignRejected(unit, StAlignRejectReason::CONFIG, 0);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }

  // **The king index and the boresights are read here, not taken from the
  // per-cycle cache.** That cache is filled on the rate-group cycle, so using it
  // would make this command's verdict depend on whether the estimator had run yet
  // — which is invisible to an operator and wrong at topology setup, where the
  // SITL/bench hook issues it. Reading ParameterDb directly makes the command
  // answer the same way whenever it arrives.
  const FwIndexType index = static_cast<FwIndexType>(unit);
  Fw::ParamValid king_valid = Fw::ParamValid::INVALID;
  const U32 king = this->paramGet_StKingUnit(king_valid);
  Fw::ParamValid boresights_valid = Fw::ParamValid::INVALID;
  const Vec3F64PerUnit boresights = this->paramGet_StBoresightsBody(boresights_valid);
  bool boresight_usable = false;
  if (boresights_valid == Fw::ParamValid::VALID && index >= 0 &&
      index < NUM_STARTRACKERIN_INPUT_PORTS) {
    const FwSizeType base = static_cast<FwSizeType>(index) * 3u;
    const Eigen::Vector3d v(boresights[base], boresights[base + 1u], boresights[base + 2u]);
    pm::Vec3<Body> direction;
    boresight_usable = v.allFinite() && pm::Vec3<Body>(v).normalized(direction);
  }
  // Four ways the unit can be wrong, and they collapse to one report because the
  // operator's next action is the same: fix the command or the configuration.
  // Calibrating the *king* against itself is the one worth naming separately in
  // the docs — it is not an error of degree, it is a request to estimate a
  // rotation that is zero by definition (lib/gnc/st_alignment.hpp).
  if (king_valid != Fw::ParamValid::VALID || index < 0 || index >= NUM_STARTRACKERIN_INPUT_PORTS ||
      index == static_cast<FwIndexType>(king) || !boresight_usable) {
    this->log_WARNING_HI_StAlignRejected(unit, StAlignRejectReason::UNIT, 0);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }

  Fw::ParamValid min_valid = Fw::ParamValid::INVALID;
  const U32 min_samples = this->paramGet_StAlignMinSamples(min_valid);
  if (min_valid != Fw::ParamValid::VALID || sampleCount < min_samples ||
      sampleCount > kMaxCalSamples) {
    // Below the configured minimum the fit would be refused on SAMPLES at the end
    // of the window; saying so now costs the operator the command rather than the
    // whole collection.
    this->log_WARNING_HI_StAlignRejected(unit, StAlignRejectReason::SAMPLES, sampleCount);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }

  // Restarting over an open window is deliberate, and so is allowing the new
  // command to name a different unit: the operator commanding a new collection
  // means that one, not two overlapping fits.
  this->st_align_accumulator_.reset();
  this->st_align_unit_ = index;
  this->st_align_target_samples_ = sampleCount;
  this->st_align_cycles_ = 0;
  // No overflow: the range check above caps sampleCount at kMaxCalSamples.
  this->st_align_deadline_cycles_ = sampleCount * kMaxCalStallFactor;
  this->st_align_collecting_ = true;
  // Any applied alignment is untouched and keeps correcting that tracker for the
  // whole window. The fit is on uncorrected readings, so the two do not interact.
  this->log_ACTIVITY_HI_StAlignStarted(unit, sampleCount);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeEstimator ::ST_ALIGN_CAL_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  // Idempotent: an abort with no window open is an operator making sure, and
  // answering OK is the honest response to "there is no collection running".
  if (this->st_align_collecting_) {
    const U32 samples = this->st_align_accumulator_.sampleCount();
    const U8 unit = static_cast<U8>(this->st_align_unit_);
    this->st_align_collecting_ = false;
    this->st_align_target_samples_ = 0;
    this->st_align_deadline_cycles_ = 0;
    this->st_align_accumulator_.reset();
    this->log_ACTIVITY_HI_StAlignAborted(unit, samples);
  }
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeEstimator ::ST_ALIGN_CAL_CLEAR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 unit) {
  const FwIndexType index = static_cast<FwIndexType>(unit);
  if (index < 0 || index >= NUM_STARTRACKERIN_INPUT_PORTS) {
    this->log_WARNING_HI_StAlignRejected(unit, StAlignRejectReason::UNIT, 0);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }
  // A window in progress is left alone on purpose: clearing the correction that
  // is flying and deciding to fit a new one are separate decisions, and the fit
  // runs on uncorrected readings either way.
  this->clearStAlign(index);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

}  // namespace flight

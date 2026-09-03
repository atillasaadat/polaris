// ======================================================================
// \title  OrbitEstimator.cpp
// \brief  Onboard orbit determination component (§8.3, §9.2)
// ======================================================================

#include "flight/PolarisFsw/OrbitEstimator/OrbitEstimator.hpp"

#include <cmath>
#include <limits>

#include "constants/constants.hpp"
#include "frames/eop.hpp"
#include "time/timescales.hpp"

namespace flight {

namespace {

constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;
const F64 kNoValue = std::numeric_limits<F64>::quiet_NaN();

namespace pg = polaris::gnc;
namespace pc = polaris::constants;

Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

//! The FPP enum mirrors the library enum value for value; the static_asserts
//! are what pin the two lists together so a value inserted in one without the
//! other fails the build rather than mislabelling a refusal in telemetry.
OrbitEstimator::OdRefusal::T toFpp(pg::OrbitOdRefusal r) {
  using E = OrbitEstimator::OdRefusal;
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kNone) == E::NONE);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kUnconfigured) == E::UNCONFIGURED);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kUninitialised) == E::UNINITIALISED);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kNonMonotonicEpoch) == E::NON_MONOTONIC_EPOCH);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kStepTooLong) == E::STEP_TOO_LONG);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kCoastExpired) == E::COAST_EXPIRED);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kFixNotFinite) == E::FIX_NOT_FINITE);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kFixImplausible) == E::FIX_IMPLAUSIBLE);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kFixSigmaInvalid) == E::FIX_SIGMA_INVALID);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kFrameConversion) == E::FRAME_CONVERSION);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kNoVelocityForSeed) == E::NO_VELOCITY_FOR_SEED);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kMeasurementRejected) ==
                E::MEASUREMENT_REJECTED);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kFilterFault) == E::FILTER_FAULT);
  static_assert(static_cast<U8>(pg::OrbitOdRefusal::kMeasurementInhibited) ==
                E::MEASUREMENT_INHIBITED);
  // -Wswitch on refusalName() is what catches a *new* enumerator; the asserts
  // above catch a renumbering of an existing one.
  return static_cast<OrbitEstimator::OdRefusal::T>(static_cast<U8>(r));
}

}  // namespace

// ----------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------

OrbitEstimator ::OrbitEstimator(const char* compName)
    : OrbitEstimatorComponentBase(compName),
      od_(pg::OrbitOdConfig{}),
      backup_(pg::OrbitOdConfig{}) {}

// ----------------------------------------------------------------------
// Handlers
// ----------------------------------------------------------------------

void OrbitEstimator ::accelIn_handler(FwIndexType portNum, const NonGravAccel& accel) {
  static_cast<void>(portNum);
  this->accel_ = accel;
}

void OrbitEstimator ::gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) {
  if (portNum < 0 || portNum >= NUM_GNSSIN_INPUT_PORTS) {
    return;
  }
  this->gnss_[portNum] = meas;
  this->gnss_fresh_[portNum] = true;
}

I64 OrbitEstimator ::currentTaiNs() const {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

FwIndexType OrbitEstimator ::selectFix() const {
  // The freshest *valid* fix across the receivers, by the receiver's own time
  // tag. The plausibility and finiteness gates are the filter's (§9.1) and are
  // not duplicated here — a fix that fails them is refused with its own name.
  FwIndexType best = -1;
  I64 best_tag = 0;
  for (FwIndexType i = 0; i < NUM_GNSSIN_INPUT_PORTS; ++i) {
    if (!this->gnss_fresh_[i] || !this->gnss_[i].get_valid()) {
      continue;
    }
    const I64 tag = this->gnss_[i].get_timeTagGpsNs();
    if (best < 0 || tag > best_tag) {
      best = i;
      best_tag = tag;
    }
  }
  return best;
}

bool OrbitEstimator ::eopAt(I64 taiNs, polaris::frames::EopValue& out) {
  bool ok = false;
  if (this->isConnected_getEopAt_OutputPort(0)) {
    EopSample sample;
    if (this->getEopAt_out(0, taiNs, sample)) {
      out = polaris::frames::EopValue{sample.get_ut1MinusTai(), sample.get_xpArcsec(),
                                      sample.get_ypArcsec()};
      ok = true;
    }
  }
  if (!ok && !this->eop_alerted_) {
    this->log_WARNING_HI_EopUnavailable(taiNs);
  }
  this->eop_alerted_ = !ok;
  return ok;
}

void OrbitEstimator ::noteRefusal(U8 unit, pg::OrbitOdRefusal refusal) {
  ++this->fixes_refused_;
  this->last_refusal_ = refusal;
  if (refusal == pg::OrbitOdRefusal::kFilterFault) {
    this->log_WARNING_HI_OrbitFilterFault();
  }
  // Edge-gated on the reason: a receiver refused every second for the same
  // reason is one fact, a change of reason is a new one.
  if (refusal != this->last_alerted_refusal_) {
    this->log_WARNING_LO_FixRefused(unit, OdRefusal(toFpp(refusal)), this->fixes_refused_);
    this->last_alerted_refusal_ = refusal;
  }
}

void OrbitEstimator ::commandResetAtCycle(U32 cycle) {
  this->reset_at_cycle_ = cycle;
}

void OrbitEstimator ::run_handler(FwIndexType portNum, U32 context) {
  static_cast<void>(portNum);
  static_cast<void>(context);
  const I64 nowNs = this->currentTaiNs();
  if (++this->cycle_ == this->reset_at_cycle_) {
    // Not through the command port: this runs inside the guarded run handler,
    // and the command port is guarded by the same (non-recursive) mutex. What
    // runs is the command handler's own body, so the reset flown is the flight
    // reset — only the dispatch is skipped.
    this->resetFilter();
  }

  if (!this->configured_ && !this->applyParameters()) {
    // Inert: publish "no solution" so a consumer never reads a stale vector,
    // and drop any fix that arrived — it cannot be folded in later, because
    // the receiver's next fix would then be a re-presented epoch.
    for (FwIndexType i = 0; i < NUM_GNSSIN_INPUT_PORTS; ++i) {
      this->gnss_fresh_[i] = false;
    }
    this->publish(nowNs);
    return;
  }

  const polaris::time::Tai now = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
  polaris::frames::EopValue eop_now;
  const bool have_eop_now = this->eopAt(nowNs, eop_now);

  // --- 1. Bring the solution to this cycle -----------------------------------
  // Propagate first, then ingest: a fix in hand is at or behind now (its
  // latency), and the filter advances the *measurement* to its own epoch — so
  // the filter must already be at now for the correction to be the right one.
  // The burn executor's acceleration, if it is valid and fresh enough to be
  // this step's (§8.3, Push 70). Stale means "no thrust known", not "the last
  // thrust", so a dead executor coasts rather than burns forever.
  pg::NonGravAccelInput accel_input;
  const pg::NonGravAccelInput* accel = nullptr;
  {
    const double accel_age_s = static_cast<double>(nowNs - this->accel_.get_epochTaiNs()) / 1.0e9;
    const Vec3F64& a = this->accel_.get_accelEciMps2();
    const bool fresh = this->accel_input_enabled_ && this->accel_.get_valid() &&
                       accel_age_s >= 0.0 && accel_age_s <= this->max_accel_age_s_ &&
                       std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]) &&
                       std::isfinite(this->accel_.get_sigmaMps2()) &&
                       this->accel_.get_sigmaMps2() >= 0.0;
    if (fresh) {
      accel_input.accel_m_s2 = polaris::math::Vec3<polaris::math::frames::ECI>(a[0], a[1], a[2]);
      accel_input.sigma_m_s2 = this->accel_.get_sigmaMps2();
      accel = &accel_input;
    }
    if (fresh != this->accel_applied_) {
      if (fresh) {
        this->log_ACTIVITY_LO_NonGravAccelApplied(accel_input.accel_m_s2.eigen().norm(),
                                                  accel_input.sigma_m_s2);
      } else {
        this->log_ACTIVITY_LO_NonGravAccelCleared();
      }
      this->accel_applied_ = fresh;
    }
    this->tlmWrite_NonGravAccelMps2(fresh ? accel_input.accel_m_s2.eigen().norm() : 0.0);
  }

  if (this->od_.isInitialised() && have_eop_now) {
    const double age_before_s = this->od_.ageSeconds();
    const pg::OrbitOdRefusal r = this->od_.propagate(now, eop_now, accel);
    if (r == pg::OrbitOdRefusal::kCoastExpired) {
      const double dt_s = static_cast<double>(nowNs - this->last_run_ns_) / 1.0e9;
      this->log_WARNING_HI_OrbitSolutionDropped(age_before_s + dt_s, this->max_degraded_coast_s_);
      this->last_refusal_ = r;
    } else if (r == pg::OrbitOdRefusal::kFilterFault) {
      this->log_WARNING_HI_OrbitFilterFault();
      this->last_refusal_ = r;
    } else if (r != pg::OrbitOdRefusal::kNone) {
      // A stuck or over-long step: reported, not fatal — the next fix decides.
      this->last_refusal_ = r;
    }
  }
  this->last_run_ns_ = nowNs;

  // --- 2. Fold in the freshest fix ------------------------------------------
  const FwIndexType slot = this->selectFix();
  if (slot >= 0) {
    const GnssMeas& m = this->gnss_[slot];
    for (FwIndexType i = 0; i < NUM_GNSSIN_INPUT_PORTS; ++i) {
      this->gnss_fresh_[i] = false;  // every latched fix is this cycle's; consumed once
    }
    pg::GnssFix fix;
    fix.time_tag = polaris::time::Gps::fromNanosecondsSinceEpoch(m.get_timeTagGpsNs());
    fix.position_m = polaris::math::Vec3<polaris::math::frames::ECEF>(
        m.get_posEcefM()[0], m.get_posEcefM()[1], m.get_posEcefM()[2]);
    fix.velocity_m_s = polaris::math::Vec3<polaris::math::frames::ECEF>(
        m.get_velEcefMps()[0], m.get_velEcefMps()[1], m.get_velEcefMps()[2]);
    fix.position_sigma_h_m = m.get_posSigmaHM();
    fix.position_sigma_v_m = m.get_posSigmaVM();
    fix.velocity_sigma_m_s = m.get_velSigmaMps();
    fix.velocity_valid = m.get_velValid();

    const I64 fixTaiNs = polaris::time::toTai(fix.time_tag).nanosecondsSinceEpoch();
    polaris::frames::EopValue eop_fix;
    if (!this->eopAt(fixTaiNs, eop_fix)) {
      this->noteRefusal(static_cast<U8>(slot), pg::OrbitOdRefusal::kFrameConversion);
    } else {
      pg::OrbitOdResult result;
      const bool ok = this->od_.ingest(fix, eop_fix, result, this->policy_);
      this->last_result_ = result;
      if (ok) {
        ++this->fixes_accepted_;
        this->last_refusal_ = pg::OrbitOdRefusal::kNone;
        this->last_alerted_refusal_ = pg::OrbitOdRefusal::kNone;
        if (result.seeded) {
          this->log_ACTIVITY_HI_OrbitSeeded(
              static_cast<U8>(slot), std::sqrt(this->od_.covariance().block<3, 3>(0, 0).trace()));
        }
      } else {
        this->noteRefusal(static_cast<U8>(slot), result.refusal);
      }
    }
  }

  // --- 3. The backup ephemeris and the covariance's health -----------------
  this->serviceBackup(nowNs, have_eop_now ? &eop_now : nullptr, accel);
  {
    // TP Ch. 7: definiteness is what a UDU filter reads off D for free; a full-P
    // filter asks once a cycle. Edge-gated: indefinite is a state, not an event
    // stream. The remedy that keeps the state is OD_REINIT_COV.
    const bool healthy = this->od_.covarianceHealthy();
    if (!healthy && !this->cov_indefinite_alerted_) {
      this->log_WARNING_HI_OrbitCovarianceIndefinite();
    }
    this->cov_indefinite_alerted_ = !healthy;
    this->tlmWrite_CovarianceHealthy(healthy);
  }

  this->publish(nowNs);
}

void OrbitEstimator ::serviceBackup(I64 nowNs, const polaris::frames::EopValue* eopNow,
                                    const pg::NonGravAccelInput* accel) {
  // TP §9.2 / TB 20-03 item (e): "a backup ephemeris, unaltered by measurement
  // updates since initialization ... For extended operations, it will usually
  // be necessary to re-seed the backup with a current filter state at periodic
  // intervals." The backup is a value copy of the filter, propagated on the
  // same force model with the same thrust input, never given a fix.
  if (this->backup_period_s_ <= 0.0) {
    if (this->backup_valid_) {
      this->backup_valid_ = false;  // the operator switched it off
    }
  } else {
    if (this->backup_valid_ && eopNow != nullptr) {
      const polaris::time::Tai now = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
      if (this->backup_.epoch() < now) {
        const pg::OrbitOdRefusal r = this->backup_.propagate(now, *eopNow, accel);
        if (r == pg::OrbitOdRefusal::kCoastExpired || r == pg::OrbitOdRefusal::kFilterFault) {
          this->backup_valid_ = false;
          this->log_WARNING_LO_OrbitBackupLost();
        }
      }
    }
    // Re-seed from a FINE solution — never from a degraded one, which is itself
    // a coasted prediction and would make the comparator compare a coast to a
    // coast — every BackupPeriodS, or immediately when there is none.
    const bool fine = this->od_.isInitialised() &&
                      this->od_.quality() == pg::OrbitOdQuality::kFine &&
                      this->od_.covarianceHealthy();
    const double since_s = static_cast<double>(nowNs - this->backup_seeded_ns_) / 1.0e9;
    if (fine && (!this->backup_valid_ || since_s >= this->backup_period_s_)) {
      const bool first = !this->backup_valid_;
      this->backup_ = this->od_;
      this->backup_valid_ = true;
      this->backup_seeded_ns_ = nowNs;
      if (first) {
        this->log_ACTIVITY_LO_OrbitBackupSeeded();
      }
    }
  }
  const bool both = this->backup_valid_ && this->od_.isInitialised();
  this->tlmWrite_BackupAgeS(
      this->backup_valid_ ? static_cast<F64>(nowNs - this->backup_seeded_ns_) / 1.0e9 : -1.0);
  this->tlmWrite_BackupDivergenceM(
      both ? (this->od_.position().eigen() - this->backup_.position().eigen()).norm() : -1.0);
}

bool OrbitEstimator ::restartFromBackup() {
  if (!this->backup_valid_ || !this->backup_.isInitialised()) {
    this->log_WARNING_LO_OrbitBackupRestartRefused();
    return false;
  }
  const F64 age_s = static_cast<F64>(this->currentTaiNs() - this->backup_seeded_ns_) / 1.0e9;
  const F64 divergence_m =
      this->od_.isInitialised()
          ? (this->od_.position().eigen() - this->backup_.position().eigen()).norm()
          : -1.0;
  // The backup becomes the solution; the counters and the fix-epoch memory are
  // the backup's own copies from when it was seeded, which is the honest
  // history of the state now flown. The backup itself is kept: a second
  // restart from the same backup is legitimate.
  this->od_ = this->backup_;
  this->last_result_ = pg::OrbitOdResult{};
  this->last_refusal_ = pg::OrbitOdRefusal::kNone;
  this->last_alerted_refusal_ = pg::OrbitOdRefusal::kNone;
  this->log_ACTIVITY_HI_OrbitRestartedFromBackup(age_s, divergence_m);
  return true;
}

void OrbitEstimator ::publish(I64 nowNs) {
  const bool valid = this->configured_ && this->od_.solutionValid();
  const pg::OrbitOdQuality quality =
      this->configured_ ? this->od_.quality() : pg::OrbitOdQuality::kNone;
  OrbitEstimate est;
  est.set_epochTaiNs(valid ? this->od_.epoch().nanosecondsSinceEpoch() : nowNs);
  est.set_valid(false);
  est.set_quality(OrbitQuality::NONE);
  est.set_ageSec(kNoValue);
  est.set_posSigmaM(kNoValue);
  est.set_velSigmaMps(kNoValue);
  est.set_posEciM(toVec3F64(Eigen::Vector3d::Zero()));
  est.set_velEciMps(toVec3F64(Eigen::Vector3d::Zero()));
  if (valid) {
    const Eigen::Vector3d r = this->od_.position().eigen();
    const Eigen::Vector3d v = this->od_.velocity().eigen();
    const auto& p = this->od_.covariance();
    const double pos_sigma = std::sqrt(p.block<3, 3>(0, 0).trace());
    const double vel_sigma = std::sqrt(p.block<3, 3>(3, 3).trace());
    // Finiteness guard on the published product: the filter guards its own
    // state, but the boundary to every consumer is here.
    if (r.allFinite() && v.allFinite() && std::isfinite(pos_sigma) && std::isfinite(vel_sigma)) {
      est.set_posEciM(toVec3F64(r));
      est.set_velEciMps(toVec3F64(v));
      est.set_posSigmaM(pos_sigma);
      est.set_velSigmaMps(vel_sigma);
      est.set_ageSec(this->od_.ageSeconds());
      est.set_valid(true);
      est.set_quality(quality == pg::OrbitOdQuality::kFine ? OrbitQuality::FINE
                                                           : OrbitQuality::DEGRADED);
    }
  }
  // The fine -> degraded edge (§8.3). The drop edge is the propagate's own
  // refusal, reported there with the age it happened at.
  const pg::OrbitOdQuality published = est.get_valid() ? quality : pg::OrbitOdQuality::kNone;
  if (published == pg::OrbitOdQuality::kDegraded &&
      this->last_quality_ == pg::OrbitOdQuality::kFine) {
    this->log_WARNING_LO_OrbitSolutionDegraded(est.get_ageSec(), this->max_coast_s_,
                                               est.get_posSigmaM());
  }
  this->last_quality_ = published;
  if (this->status_period_cycles_ > 0 && (this->cycle_ % this->status_period_cycles_) == 0) {
    const Vec3F64& r = est.get_posEciM();
    this->log_ACTIVITY_LO_OrbitStatus(est.get_quality(), est.get_ageSec(), est.get_posSigmaM(),
                                      r[0], r[1], r[2]);
  }
  if (this->isConnected_orbitStateOut_OutputPort(0)) {
    // Fan out to every consumer: the attitude estimator and the §8.4 pointing
    // guidance. Looped rather than written twice so adding a consumer is a
    // topology change alone.
    for (FwIndexType i = 0; i < NUM_ORBITSTATEOUT_OUTPUT_PORTS; ++i) {
      if (this->isConnected_orbitStateOut_OutputPort(i)) {
        this->orbitStateOut_out(i, est);
      }
    }
  }
  this->tlmWrite_OrbitQualityTlm(est.get_quality());

  this->tlmWrite_PosEciM(est.get_posEciM());
  this->tlmWrite_VelEciMps(est.get_velEciMps());
  this->tlmWrite_PosSigmaM(est.get_posSigmaM());
  this->tlmWrite_VelSigmaMps(est.get_velSigmaMps());
  this->tlmWrite_SolutionAgeS(est.get_ageSec());
  this->tlmWrite_SolutionValid(est.get_valid());
  this->tlmWrite_PositionNis(this->last_result_.position.nis);
  this->tlmWrite_VelocityNis(this->last_result_.velocity.nis);
  this->tlmWrite_FixLatencyS(this->last_result_.fix_latency_s);
  this->tlmWrite_FixesAccepted(this->fixes_accepted_);
  this->tlmWrite_FixesRefused(this->fixes_refused_);
  this->tlmWrite_LastRefusal(OdRefusal(toFpp(this->last_refusal_)));
  this->tlmWrite_FixesForced(this->od_.forcedCount());
  // TP §2.1 covariance metrics and the DMC states (Push 73).
  if (valid) {
    const Eigen::Vector3d r = this->od_.position().eigen();
    const Eigen::Vector3d v = this->od_.velocity().eigen();
    const pg::OrbitOd::Covariance p6 = this->od_.covariance();
    this->tlmWrite_SmaSigmaM(pg::smaSigma(r, v, p6, pc::gravity::kGM));
    this->tlmWrite_FpaSigmaRad(pg::flightPathAngleSigma(r, v, p6));
    this->tlmWrite_DmcAccelRtnMps2(toVec3F64(this->od_.dmcAcceleration()));
    this->tlmWrite_DmcSigmaMps2(std::sqrt(
        this->od_.fullCovariance().block<3, 3>(pg::OrbitOd::kDmc, pg::OrbitOd::kDmc).trace()));
    this->tlmWrite_GnssBiasSigmaM(this->od_.gnssBiasSigma());
    this->tlmWrite_DragScale(this->od_.dragScale());
    this->tlmWrite_DragScaleSigma(this->od_.dragScaleSigma());
  } else {
    this->tlmWrite_SmaSigmaM(-1.0);
    this->tlmWrite_FpaSigmaRad(-1.0);
    this->tlmWrite_DmcAccelRtnMps2(toVec3F64(Eigen::Vector3d::Zero()));
    this->tlmWrite_DmcSigmaMps2(0.0);
    // The nominal, not a sentinel: with no solution the scale *is* 1, which is
    // what dropSolution() leaves it at, and a -1 here would read as a physical
    // sign flip rather than as "no data".
    this->tlmWrite_GnssBiasSigmaM(0.0);
    this->tlmWrite_DragScale(1.0);
    this->tlmWrite_DragScaleSigma(0.0);
  }
  // Outside the valid branch: a refusal count survives a dropped solution, and
  // is exactly what the ground wants to see after one.
  this->tlmWrite_DragScaleRefused(this->od_.dragScaleRefusedCount());
}

// ----------------------------------------------------------------------
// Configuration (§19.3 — no defaults)
// ----------------------------------------------------------------------

bool OrbitEstimator ::applyParameters() {
  auto fail = [this](const char* detail) {
    if (!this->tuning_alerted_) {
      Fw::LogStringArg arg(detail);
      this->log_WARNING_HI_OrbitTuningInvalid(arg);
      this->tuning_alerted_ = true;
    }
    // A running filter is never made inert by a bad upload (TB 20-03 item g):
    // the last valid set stays in force. Only a filter that never had one is
    // left inert.
    if (!this->od_.isConfigured()) {
      this->configured_ = false;
    }
    return false;
  };

#define POLARIS_GET(dest, getter, name)         \
  do {                                          \
    Fw::ParamValid v = Fw::ParamValid::INVALID; \
    (dest) = this->getter(v);                   \
    if (v != Fw::ParamValid::VALID) {           \
      return fail(name);                        \
    }                                           \
  } while (0)

  pg::OrbitOdConfig cfg;
  U32 degree = 0;
  U32 order = 0;
  POLARIS_GET(degree, paramGet_GeopotentialDegree, "GeopotentialDegree");
  POLARIS_GET(order, paramGet_GeopotentialOrder, "GeopotentialOrder");
  if (degree > static_cast<U32>(pg::kGeopotentialMaxDegree) || order > degree) {
    return fail("GeopotentialDegree/Order out of range");
  }
  cfg.geopotential_degree = static_cast<int>(degree);
  cfg.geopotential_order = static_cast<int>(order);
  // GM and the reference radius are paired with the coefficient table and with
  // J2 (§8.3): never a parameter. Degree 0 is the closed-form J2 path — the
  // flight does not fly pure two-body.
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.zonal_j2 = (degree == 0) ? pc::gravity::kJ2 : 0.0;
  POLARIS_GET(cfg.drag_ballistic_coeff_m2_per_kg, paramGet_DragBallisticCoeffM2PerKg,
              "DragBallisticCoeffM2PerKg");
  POLARIS_GET(cfg.drag_ref_density_kg_m3, paramGet_DragRefDensityKgM3, "DragRefDensityKgM3");
  POLARIS_GET(cfg.drag_ref_altitude_m, paramGet_DragRefAltitudeM, "DragRefAltitudeM");
  POLARIS_GET(cfg.drag_scale_height_m, paramGet_DragScaleHeightM, "DragScaleHeightM");
  POLARIS_GET(cfg.accel_psd_m2_per_s3, paramGet_AccelPsdM2PerS3, "AccelPsdM2PerS3");
  {
    // TP §2.2.3.1 / §2.2.3.3 (Push 73): the RTN SNC and the DMC states.
    Fw::ParamValid v1 = Fw::ParamValid::INVALID;
    Fw::ParamValid v2 = Fw::ParamValid::INVALID;
    Fw::ParamValid v3 = Fw::ParamValid::INVALID;
    const Vec3F64 rtn = this->paramGet_AccelPsdRtnM2PerS3(v1);
    const F64 tau = this->paramGet_DmcTauS(v2);
    const Vec3F64 dmc = this->paramGet_DmcPsdRtnM2PerS5(v3);
    if (v1 != Fw::ParamValid::VALID) {
      return fail("AccelPsdRtnM2PerS3");
    }
    if (v2 != Fw::ParamValid::VALID) {
      return fail("DmcTauS");
    }
    if (v3 != Fw::ParamValid::VALID) {
      return fail("DmcPsdRtnM2PerS5");
    }
    cfg.accel_psd_rtn_m2_per_s3 = Eigen::Vector3d(rtn[0], rtn[1], rtn[2]);
    cfg.dmc_tau_s = tau;
    cfg.dmc_psd_rtn_m2_per_s5 = Eigen::Vector3d(dmc[0], dmc[1], dmc[2]);
  }
  // The correlated-GNSS R repair (Push 77). Fetched unconditionally: a filter
  // flying a receiver with a correlated error and no inflation is exactly the
  // overconfident case this exists to prevent, so the value is never defaulted.
  POLARIS_GET(cfg.gnss_corr_fraction, paramGet_GnssCorrFraction, "GnssCorrFraction");
  POLARIS_GET(cfg.gnss_corr_tau_s, paramGet_GnssCorrTauS, "GnssCorrTauS");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const bool consider = this->paramGet_GnssBiasConsider(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("GnssBiasConsider");
    }
    cfg.gnss_bias_consider = consider;
  }
  // §8.5 tier 3, orbit half (Push 76). Fetched unconditionally rather than
  // only when the PSD is positive: a half-uploaded tuning has to reach
  // OrbitOdConfig::isValid, which refuses it, instead of being silently
  // completed from whatever the other three parameters happened to hold.
  POLARIS_GET(cfg.drag_scale_tau_s, paramGet_DragScaleTauS, "DragScaleTauS");
  POLARIS_GET(cfg.drag_scale_psd_per_s, paramGet_DragScalePsdPerS, "DragScalePsdPerS");
  POLARIS_GET(cfg.drag_scale_seed_sigma, paramGet_DragScaleSeedSigma, "DragScaleSeedSigma");
  POLARIS_GET(cfg.drag_scale_max_deviation, paramGet_DragScaleMaxDeviation,
              "DragScaleMaxDeviation");
  POLARIS_GET(cfg.position_nis_gate, paramGet_PositionNisGate, "PositionNisGate");
  POLARIS_GET(cfg.velocity_nis_gate, paramGet_VelocityNisGate, "VelocityNisGate");
  POLARIS_GET(cfg.max_coast_s, paramGet_MaxCoastS, "MaxCoastS");
  POLARIS_GET(cfg.max_degraded_coast_s, paramGet_MaxDegradedCoastS, "MaxDegradedCoastS");
  POLARIS_GET(this->max_accel_age_s_, paramGet_MaxAccelAgeS, "MaxAccelAgeS");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    this->status_period_cycles_ = this->paramGet_StatusPeriodCycles(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("StatusPeriodCycles");
    }
  }
  if (!std::isfinite(this->max_accel_age_s_) || this->max_accel_age_s_ < 0.0) {
    return fail("MaxAccelAgeS");
  }
  POLARIS_GET(cfg.max_dt_s, paramGet_MaxDtS, "MaxDtS");
  POLARIS_GET(cfg.max_step_s, paramGet_MaxStepS, "MaxStepS");
  POLARIS_GET(cfg.max_fix_latency_s, paramGet_MaxFixLatencyS, "MaxFixLatencyS");
  POLARIS_GET(cfg.min_radius_m, paramGet_MinRadiusM, "MinRadiusM");
  POLARIS_GET(cfg.max_radius_m, paramGet_MaxRadiusM, "MaxRadiusM");
  U8 pos_mode = 0;
  U8 vel_mode = 0;
  F64 backup_period_s = 0.0;
  POLARIS_GET(pos_mode, paramGet_PositionMeasMode, "PositionMeasMode");
  POLARIS_GET(vel_mode, paramGet_VelocityMeasMode, "VelocityMeasMode");
  POLARIS_GET(backup_period_s, paramGet_BackupPeriodS, "BackupPeriodS");
#undef POLARIS_GET

  if (!cfg.isValid()) {
    return fail("OrbitOdConfig::isValid failed");
  }
  // The U8 parameter mirrors the library enum value for value (TP §9.1).
  static_assert(static_cast<U8>(pg::MeasurementMode::kAccept) == 0);
  static_assert(static_cast<U8>(pg::MeasurementMode::kInhibit) == 1);
  static_assert(static_cast<U8>(pg::MeasurementMode::kForce) == 2);
  if (pos_mode > 2 || vel_mode > 2) {
    return fail("PositionMeasMode/VelocityMeasMode must be 0, 1 or 2");
  }
  if (!std::isfinite(backup_period_s) || backup_period_s < 0.0 ||
      backup_period_s >= cfg.max_degraded_coast_s) {
    return fail("BackupPeriodS must be in [0, MaxDegradedCoastS)");
  }

  // TB 20-03 item (g) / TP §9.3: a running filter is re-tuned in place; only a
  // filter that never had a valid set is constructed. The backup follows.
  const bool kept = this->od_.isConfigured();
  if (kept) {
    (void)this->od_.retune(cfg);  // cfg.isValid() held above
    (void)this->backup_.retune(cfg);
  } else {
    this->od_ = pg::OrbitOd(cfg);
    this->backup_ = pg::OrbitOd(cfg);
  }
  if (this->configured_) {
    // A re-read after a valid set: an upload. First-time configuration is not
    // "tuning applied", it is bring-up.
    this->log_ACTIVITY_LO_OrbitTuningApplied(kept && this->od_.isInitialised());
  }
  const pg::GnssMeasurementPolicy policy{static_cast<pg::MeasurementMode>(pos_mode),
                                         static_cast<pg::MeasurementMode>(vel_mode)};
  if (!this->policy_reported_ || policy.position != this->policy_.position ||
      policy.velocity != this->policy_.velocity) {
    this->log_ACTIVITY_HI_MeasurementPolicyChanged(pos_mode, vel_mode);
    this->policy_reported_ = true;
  }
  this->policy_ = policy;
  this->backup_period_s_ = backup_period_s;
  this->max_coast_s_ = cfg.max_coast_s;
  this->max_degraded_coast_s_ = cfg.max_degraded_coast_s;
  this->max_fix_latency_s_ = cfg.max_fix_latency_s;
  this->configured_ = true;
  this->tuning_alerted_ = false;
  return true;
}

void OrbitEstimator ::parameterUpdated(FwPrmIdType id) {
  static_cast<void>(id);
  // Re-tune in place (TB 20-03 item g; TP §9.3): a covariance built under the
  // old q_a is carried forward rather than the solution being thrown away —
  // the trade the TP recommends. A set that fails validation warns and leaves
  // the old one in force.
  (void)this->applyParameters();
}

// ----------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------

void OrbitEstimator ::resetFilter() {
  this->od_.reset();
  this->last_result_ = pg::OrbitOdResult{};
  this->fixes_accepted_ = 0;
  this->fixes_refused_ = 0;
  this->last_refusal_ = pg::OrbitOdRefusal::kNone;
  this->last_alerted_refusal_ = pg::OrbitOdRefusal::kNone;
  this->log_ACTIVITY_HI_OrbitReset();
}

void OrbitEstimator ::OD_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  this->resetFilter();
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void OrbitEstimator ::OD_SEED_STATE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, I64 epochTaiNs,
                                               F64 posEciX, F64 posEciY, F64 posEciZ, F64 velEciX,
                                               F64 velEciY, F64 velEciZ, F64 posSigmaM,
                                               F64 velSigmaMps) {
  if (!this->configured_) {
    this->log_WARNING_LO_OrbitSeedRefused(OdRefusal(OdRefusal::UNCONFIGURED));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }
  // The window the filter can then propagate from: no further behind now than
  // the degraded horizon, no further ahead than a fix may be latent.
  const double behind_s = static_cast<double>(this->currentTaiNs() - epochTaiNs) / 1.0e9;
  if (!(behind_s <= this->max_degraded_coast_s_) || !(behind_s >= -this->max_fix_latency_s_)) {
    this->log_WARNING_LO_OrbitSeedRefused(OdRefusal(OdRefusal::NON_MONOTONIC_EPOCH));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  using polaris::math::Vec3;
  using polaris::math::frames::ECI;
  const pg::OrbitOdRefusal r =
      this->od_.seed(polaris::time::Tai::fromNanosecondsSinceEpoch(epochTaiNs),
                     Vec3<ECI>(posEciX, posEciY, posEciZ), Vec3<ECI>(velEciX, velEciY, velEciZ),
                     posSigmaM, velSigmaMps);
  if (r != pg::OrbitOdRefusal::kNone) {
    this->log_WARNING_LO_OrbitSeedRefused(OdRefusal(toFpp(r)));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->last_refusal_ = pg::OrbitOdRefusal::kNone;
  this->last_alerted_refusal_ = pg::OrbitOdRefusal::kNone;
  this->log_ACTIVITY_HI_OrbitSeededFromGround(posSigmaM);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void OrbitEstimator ::OD_REINIT_COV_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 posSigmaM,
                                               F64 velSigmaMps) {
  const pg::OrbitOdRefusal r = this->od_.reinitializeCovariance(posSigmaM, velSigmaMps);
  if (r != pg::OrbitOdRefusal::kNone) {
    this->log_WARNING_LO_OrbitCovarianceReinitRefused(OdRefusal(toFpp(r)));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->cov_indefinite_alerted_ = false;
  this->log_ACTIVITY_HI_OrbitCovarianceReinitialised(posSigmaM, velSigmaMps);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void OrbitEstimator ::OD_RESTART_FROM_BACKUP_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  this->cmdResponse_out(
      opCode, cmdSeq,
      this->restartFromBackup() ? Fw::CmdResponse::OK : Fw::CmdResponse::EXECUTION_ERROR);
}

}  // namespace flight

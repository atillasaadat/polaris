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
  // -Wswitch on refusalName() is what catches a *new* enumerator; the asserts
  // above catch a renumbering of an existing one.
  return static_cast<OrbitEstimator::OdRefusal::T>(static_cast<U8>(r));
}

}  // namespace

// ----------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------

OrbitEstimator ::OrbitEstimator(const char* compName)
    : OrbitEstimatorComponentBase(compName), od_(pg::OrbitOdConfig{}) {}

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
      const bool ok = this->od_.ingest(fix, eop_fix, result);
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

  this->publish(nowNs);
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
    this->orbitStateOut_out(0, est);
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
    this->configured_ = false;
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
#undef POLARIS_GET

  if (!cfg.isValid()) {
    return fail("OrbitOdConfig::isValid failed");
  }
  this->od_ = pg::OrbitOd(cfg);
  this->max_coast_s_ = cfg.max_coast_s;
  this->max_degraded_coast_s_ = cfg.max_degraded_coast_s;
  this->max_fix_latency_s_ = cfg.max_fix_latency_s;
  this->configured_ = true;
  this->tuning_alerted_ = false;
  return true;
}

void OrbitEstimator ::parameterUpdated(FwPrmIdType id) {
  static_cast<void>(id);
  // Rebuilding the filter drops the solution: a covariance built under the old
  // tuning is a claim about a different filter. The next fix re-seeds.
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

}  // namespace flight

// ======================================================================
// \title  AttitudeEstimatorSensors.cpp
// \brief  Multi-unit sensor selection and fault-tolerant IMU voting on the
//         attitude estimator (design doc §8.2, §9.1, §9.2;
//         REQ-ADET-008, REQ-ADET-009)
//
// The same `flight::AttitudeEstimator` class as AttitudeEstimator.cpp, split
// into its own translation unit for the same reason the magnetometer
// calibration was: this is the layer between N measurement ports and the one
// measurement set the estimators consume, and it is read on its own.
//
// What lives here: the per-type combination rules — the IMU vote and its FDIR
// edges, best-illuminated sun-sensor selection, first-valid magnetometer and
// GNSS selection, and the per-unit albedo boresight lookup that follows the
// selected sun sensor.
//
// What does not: the estimation cycle that calls them (AttitudeEstimator.cpp)
// and the correction application points inside it, whose whole value is that
// they sit exactly once in the measurement path where a reader of that path
// will see them.
//
// Every function here runs on the rate-group thread inside the `guarded` `run`
// port, so the exclusion latch it mutates is serialised against the command
// handlers on the same component mutex.
// ======================================================================

#include <cmath>
#include <limits>

#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"

namespace flight {

namespace {

constexpr I64 kNsPerSecond = 1000000000LL;

namespace pm = polaris::math;
using Body = pm::frames::Body;

//! True if @p timeTagNs is within @p maxAgeS of @p nowTaiNs. A measurement from
//! the future is stale too — a time tag ahead of the master clock is a fault,
//! not freshness (§9.1). Duplicated from AttitudeEstimator.cpp's anonymous
//! namespace deliberately: it is four lines, and a shared internal header for it
//! would be more machinery than the rule it encodes.
bool fresh(I64 nowTaiNs, I64 timeTagNs, F64 maxAgeS) {
  if (!(maxAgeS > 0.0)) {
    return false;
  }
  const I64 delta = nowTaiNs - timeTagNs;
  const I64 limit = static_cast<I64>(maxAgeS * static_cast<F64>(kNsPerSecond));
  return delta >= -limit && delta <= limit;
}

//! Map a library exclusion reason onto the telemetered one. Only the three that
//! can reach an EVR are mapped; the rest are not exclusions at all (a
//! contributing or absent unit) and are filtered by the caller before this runs.
ImuExclusionReason::T toExclusionReason(polaris::gnc::ImuVoteReason why) {
  switch (why) {
    case polaris::gnc::ImuVoteReason::kOutOfRange:
      return ImuExclusionReason::RATE_LIMIT;
    case polaris::gnc::ImuVoteReason::kOutvoted:
      return ImuExclusionReason::OUTVOTED;
    default:
      // kNotFinite, and anything a future library revision adds. Reporting a
      // broken data path is the conservative default; refusing to emit the event
      // would lose the exclusion entirely.
      return ImuExclusionReason::NOT_FINITE;
  }
}

//! The same mapping for the magnetometer vote. A separate enumeration on the wire
//! because the middle gate is a different physical test — "RATE_LIMIT" on a
//! magnetometer would be a lie in telemetry the ground acts on.
MagExclusionReason::T toMagExclusionReason(polaris::gnc::MagVoteReason why) {
  switch (why) {
    case polaris::gnc::MagVoteReason::kOutOfRange:
      return MagExclusionReason::FIELD_MAGNITUDE;
    case polaris::gnc::MagVoteReason::kOutvoted:
      return MagExclusionReason::OUTVOTED;
    default:
      return MagExclusionReason::NOT_FINITE;
  }
}

//! Not a value, for telemetry channels with nothing to report this cycle.
const double kNoValueSensors = std::numeric_limits<double>::quiet_NaN();

//! Telemetered unit index when nothing was selected this cycle, for **any** unit
//! type. 255 rather than 0, which is a perfectly good port index.
constexpr U8 kNoUnitIndex = 255;

//! Angle between two directions, from the cross/dot form rather than an acos of
//! the dot product (§3.5, lib/README.md).
double angleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

}  // namespace

// ----------------------------------------------------------------------
// IMU: the fault-tolerant vote (§8.2)
// ----------------------------------------------------------------------

bool AttitudeEstimator ::voteImuRate(I64 nowTaiNs, pm::Vec3<Body>& rate) {
  // Stage the port array into the voter's input form. The two gates that need
  // the component's context — the unit's own validity flag with a positive
  // accumulation interval, and staleness against the master clock — are applied
  // here, and are reported to the voter as *absence* rather than as
  // implausibility: a dropout says nothing about whether the unit is lying, and
  // must not latch an exclusion the ground then has to reason about.
  polaris::gnc::ImuVoteInput units[NUM_IMUIN_INPUT_PORTS];
  static_assert(NUM_IMUIN_INPUT_PORTS <= polaris::gnc::kMaxImuUnits,
                "the IMU port array is wider than the voter can carry");

  for (FwIndexType i = 0; i < NUM_IMUIN_INPUT_PORTS; ++i) {
    const ImuMeas& m = this->imu_[i];
    if (!m.get_valid() || !(m.get_intervalSec() > 0.0) ||
        !fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      continue;
    }
    // Delta-angle over its own accumulation interval, which is the rate the unit
    // actually measured — not the GNC period, which may cover several samples.
    const Eigen::Vector3d unit_rate =
        Eigen::Vector3d(this->imu_[i].get_deltaAngleRad()[0], this->imu_[i].get_deltaAngleRad()[1],
                        this->imu_[i].get_deltaAngleRad()[2]) /
        m.get_intervalSec();
    units[i].rate = pm::Vec3<Body>(unit_rate);
    units[i].present = true;
  }

  // The tie-break for a two-unit disagreement: the filter's own propagated rate,
  // available only while fine mode is engaged with a valid rate. Passing nullptr
  // otherwise is what makes an unattributable disagreement report as ambiguous
  // instead of being resolved by a coin flip.
  pm::Vec3<Body> reference;
  const bool have_reference = this->fine_active_ && this->state_.valid.body_rate;
  if (have_reference) {
    reference = this->state_.body_rate;
  }

  polaris::gnc::ImuVoteResult result;
  const bool ok = this->imu_voter_.vote(units, NUM_IMUIN_INPUT_PORTS,
                                        have_reference ? &reference : nullptr, result);

  // FDIR edges. Both are reported on the transition, so a permanently dead unit
  // costs one event rather than one per 10 Hz cycle.
  for (FwIndexType i = 0; i < NUM_IMUIN_INPUT_PORTS; ++i) {
    if (result.newly_excluded[i]) {
      this->log_WARNING_HI_ImuUnitExcluded(static_cast<U8>(i), toExclusionReason(result.reason[i]));
    }
    if (result.newly_readmitted[i]) {
      this->log_ACTIVITY_HI_ImuUnitReadmitted(static_cast<U8>(i));
    }
  }

  // The one condition the vote can reach that is a genuine loss of knowledge
  // rather than an isolated fault: two plausible units that disagree, with
  // nothing to say which. Edge-gated, because it persists for as long as the
  // pair does — and *because* it is edge-gated it needs the persistence
  // escalation below, or a permanent condition would be one warning and then
  // silence for the rest of the flight.
  if (result.status == polaris::gnc::ImuVoteStatus::kAmbiguous) {
    if (this->imu_ambiguous_cycles_ == 0) {
      this->imu_ambiguous_start_ns_ = nowTaiNs;
    }
    ++this->imu_ambiguous_cycles_;
    // Escalation at the horizon, then re-reported at the same period. Modular
    // rather than latched so the cadence is bounded without a second counter,
    // and so a condition that clears and returns starts its horizon again.
    //
    // The duration is measured from the clock rather than inferred from the
    // cycle count and an assumed rate: the rate group is configuration (§2.4),
    // and cycles are also lost to refusals, so a count times a nominal period
    // would misreport the one number the ground acts on.
    if (this->imu_ambiguity_escalate_cycles_ > 0 &&
        (this->imu_ambiguous_cycles_ % this->imu_ambiguity_escalate_cycles_) == 0) {
      const double elapsed_s =
          static_cast<double>(nowTaiNs - this->imu_ambiguous_start_ns_) / 1.0e9;
      this->log_WARNING_HI_ImuVoteAmbiguousPersistent(elapsed_s, this->imu_ambiguous_cycles_);
    }
    if (!this->imu_ambiguous_flagged_) {
      // Report the disagreement that triggered it, which is what sizes the fault
      // for the ground; recomputing it here keeps the library's result struct
      // from carrying a field only telemetry wants.
      double worst = 0.0;
      for (FwIndexType i = 0; i < NUM_IMUIN_INPUT_PORTS; ++i) {
        for (FwIndexType j = static_cast<FwIndexType>(i + 1); j < NUM_IMUIN_INPUT_PORTS; ++j) {
          if (result.reason[i] != polaris::gnc::ImuVoteReason::kContributing ||
              result.reason[j] != polaris::gnc::ImuVoteReason::kContributing) {
            continue;
          }
          const double difference = (units[i].rate.eigen() - units[j].rate.eigen()).norm();
          worst = (difference > worst) ? difference : worst;
        }
      }
      this->log_WARNING_HI_ImuVoteAmbiguous(worst);
      this->imu_ambiguous_flagged_ = true;
    }
  } else {
    this->imu_ambiguous_flagged_ = false;
    this->imu_ambiguous_cycles_ = 0;
  }

  this->tlmWrite_ImuContributing(static_cast<U32>(result.contributing));
  this->tlmWrite_ImuExclusionMask(result.exclusion_mask);

  if (!ok) {
    return false;
  }
  rate = result.rate;
  return true;
}

// ----------------------------------------------------------------------
// Sun sensors: selection of the best-illuminated unit (§8.2)
// ----------------------------------------------------------------------

const SunSensorMeas* AttitudeEstimator ::selectSunSensor(I64 nowTaiNs, FwIndexType& index,
                                                         FwIndexType& runnerUp) const {
  // **Selection, not a weighted combination**, and the reason is the budget
  // rather than the algebra: the sun measurements' error is dominated by a
  // *shared* systematic — the albedo residual and the ephemeris term are common
  // to every unit on the vehicle — which combining cannot average down. A second
  // unit at a worse incidence therefore buys noise reduction on the small term
  // and nothing at all on the large one, while costing a correlation the
  // estimators' white-R model has no way to represent.
  //
  // The criterion is the **smallest realised sigma** (§6.2), which the unit
  // reports per sample from its own incidence angle. That is the incidence
  // cosine expressed in the quantity the estimator actually consumes, so it
  // stays right if a future suite mixes parts with different accuracy curves —
  // which raw incidence would not.
  const SunSensorMeas* best = nullptr;
  double best_sigma = 0.0;
  double runner_sigma = 0.0;
  runnerUp = -1;

  for (FwIndexType i = 0; i < NUM_SUNSENSORIN_INPUT_PORTS; ++i) {
    const SunSensorMeas& m = this->sun_[i];
    // No sun in view is a normal condition (eclipse, or the unit facing away),
    // not a fault — it simply excludes this unit from the pair this cycle.
    if (!m.get_valid() || !m.get_sunPresent() ||
        !fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      continue;
    }
    // A non-positive or non-finite sigma is wire data the estimator would divide
    // by, so it is gated here with the rest (§9.1) rather than trusted because
    // the unit's own flag reads true.
    const double sigma = m.get_sigmaRad();
    if (!std::isfinite(sigma) || !(sigma > 0.0)) {
      continue;
    }
    // Strictly-less, so a tie keeps the earlier index: the choice is
    // deterministic in vehicle build order, and index 0 is the solar-array
    // normal, the unit the accuracy budget was measured on.
    if (best == nullptr || sigma < best_sigma) {
      // The displaced leader becomes the runner-up, which keeps the second slot
      // ordered by the same rule as the first rather than by scan order.
      if (best != nullptr) {
        runnerUp = index;
        runner_sigma = best_sigma;
      }
      best = &this->sun_[i];
      best_sigma = sigma;
      index = i;
    } else if (runnerUp < 0 || sigma < runner_sigma) {
      runnerUp = i;
      runner_sigma = sigma;
    }
  }
  return best;
}

const SunSensorMeas* AttitudeEstimator ::crossCheckSunUnit(FwIndexType& index, FwIndexType runnerUp,
                                                           const pm::Vec3<Body>& reference,
                                                           bool haveReference) {
  if (index < 0 || index >= NUM_SUNSENSORIN_INPUT_PORTS) {
    return nullptr;
  }
  const SunSensorMeas* selected = &this->sun_[index];
  if (runnerUp < 0 || runnerUp >= NUM_SUNSENSORIN_INPUT_PORTS) {
    // One unit in view: nothing to cross-check against, which is a geometry fact
    // and not a fault. NaN and a streak reset, never a cleared alert — see
    // noteMonitorResidual.
    this->tlmWrite_SunCrossUnitRad(kNoValueSensors);
    this->noteMonitorResidual(ResidualMonitor::SUN_CROSS_UNIT, kNoValueSensors,
                              this->monitor_threshold_rad_[ResidualMonitor::SUN_CROSS_UNIT]);
    return selected;
  }

  const SunSensorMeas* runner = &this->sun_[runnerUp];
  const Eigen::Vector3d a(selected->get_dirBody()[0], selected->get_dirBody()[1],
                          selected->get_dirBody()[2]);
  const Eigen::Vector3d b(runner->get_dirBody()[0], runner->get_dirBody()[1],
                          runner->get_dirBody()[2]);
  // Wire data, so it is gated before it is compared: a non-finite or zero-length
  // reading would give a meaningless angle that would then latch a monitor.
  if (!a.allFinite() || !b.allFinite() || !(a.norm() > 0.0) || !(b.norm() > 0.0)) {
    this->tlmWrite_SunCrossUnitRad(kNoValueSensors);
    this->noteMonitorResidual(ResidualMonitor::SUN_CROSS_UNIT, kNoValueSensors,
                              this->monitor_threshold_rad_[ResidualMonitor::SUN_CROSS_UNIT]);
    return selected;
  }

  const double separation = angleBetween(a, b);
  this->tlmWrite_SunCrossUnitRad(separation);
  const double threshold = this->monitor_threshold_rad_[ResidualMonitor::SUN_CROSS_UNIT];
  this->noteMonitorResidual(ResidualMonitor::SUN_CROSS_UNIT, separation, threshold);

  // **Detection needs no attitude; resolution does.** Two units disagreeing is a
  // sensor-versus-sensor fact, which is what makes this the one cross-check
  // available in Safe mode — but it says only that one of them is wrong. Choosing
  // between them needs the third information source, exactly as the two-unit IMU
  // and magnetometer votes do, and here that is the fine solution's own predicted
  // sun direction.
  //
  // Gated on the monitor having *already alerted*, so a single noisy sample can
  // never move the selection: the disagreement has to have persisted for
  // MonitorAlertCycles first. Nothing is latched either way — the override is
  // re-decided every cycle from the current evidence, so a unit that recovers
  // simply stops being overridden, and there is no exclusion for the ground to
  // reason about on a condition the vehicle resolved on its own.
  if (!haveReference ||
      !this->monitor_alerted_[static_cast<int>(ResidualMonitor::SUN_CROSS_UNIT)]) {
    return selected;
  }
  const Eigen::Vector3d predicted = reference.eigen();
  if (!predicted.allFinite() || !(predicted.norm() > 0.0)) {
    return selected;
  }
  // **Decisive margin, not mere ordering** — the same rule the two-unit votes
  // apply, and for the same reason: ordering alone is a coin flip whenever the
  // two residuals are close, which is exactly the common-mode case. The reference
  // has to agree with the runner-up (inside the threshold) *and* disagree with the
  // incumbent (outside it) before the selection moves.
  const double selected_residual = angleBetween(a, predicted);
  const double runner_residual = angleBetween(b, predicted);
  const bool decisive = selected_residual > threshold && runner_residual <= threshold;
  if (decisive) {
    this->log_WARNING_LO_SunUnitOverridden(static_cast<U8>(index), static_cast<U8>(runnerUp),
                                           separation);
    index = runnerUp;
    return runner;
  }
  return selected;
}

bool AttitudeEstimator ::sunBoresightFor(FwIndexType index, pm::Vec3<Body>& out) const {
  if (index < 0 || index >= NUM_SUNSENSORIN_INPUT_PORTS) {
    return false;
  }
  const FwIndexType base = static_cast<FwIndexType>(index * 3);
  const Eigen::Vector3d v(this->sun_boresights_[base], this->sun_boresights_[base + 1],
                          this->sun_boresights_[base + 2]);
  // The zero vector is the configured "no value for this index" — a unit that is
  // not installed, or one whose mounting has not been characterised. It cannot be
  // mistaken for a direction, which is why it was chosen over a sentinel.
  // `Vec3::normalized` refuses it, so the two checks are one call.
  pm::Vec3<Body> unit;
  if (!v.allFinite() || !pm::Vec3<Body>(v).normalized(unit)) {
    return false;
  }
  out = unit;
  return true;
}

// ----------------------------------------------------------------------
// Magnetometer and GNSS: first valid, fresh unit
// ----------------------------------------------------------------------

bool AttitudeEstimator ::voteMagField(I64 nowTaiNs, const pm::Vec3<Body>& magRefBody,
                                      bool haveAttitude, double attitudeSigmaRad,
                                      double modelledMagnitudeT, pm::Vec3<Body>& field,
                                      FwIndexType& index) {
  // Stage the port array into the voter's input form. The two gates that need the
  // component's context — the unit's own validity flag and staleness against the
  // master clock — are applied here, and are reported to the voter as *absence*
  // rather than as implausibility: a dropout says nothing about whether the unit
  // is lying, and must not latch an exclusion the ground then has to reason about.
  polaris::gnc::MagVoteInput units[NUM_MAGNETOMETERIN_INPUT_PORTS];
  static_assert(NUM_MAGNETOMETERIN_INPUT_PORTS <= polaris::gnc::kMaxMagUnits,
                "the magnetometer port array is wider than the voter can carry");

  for (FwIndexType i = 0; i < NUM_MAGNETOMETERIN_INPUT_PORTS; ++i) {
    const MagnetometerMeas& m = this->mag_[i];
    if (!m.get_valid() || !fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      continue;
    }
    // §7 MTQ/MAG duty-cycle interlock, layer 2, applied at the **one** place
    // magnetometer data enters the vehicle. The gate is in two halves, and the
    // order of them is load-bearing.
    //
    // First the **timing** half: a sample taken while a rod was energised, or
    // before its field had settled, is not a measurement of the geomagnetic field
    // at all and there is nothing to be learned from it.
    if (!this->magSampleInQuietWindow(m.get_timeTagNs())) {
      ++this->mag_interlock_rejects_;
      continue;
    }

    // Raw magnitude, recorded here — **upstream of the §8.2 plausibility band,
    // upstream of the vote, and upstream of the interlock's own health verdict**
    // — for the §9 stuck-on monitor (see
    // GncPorts.AttitudeEstimate.magRawMagnitudeT). A rod stuck on puts hundreds
    // of microtesla on the sensor, which the magnitude gate below rejects as
    // implausible, so a monitor fed the voted field goes blind at exactly the
    // disturbance it names. The health verdict has to be upstream of it for the
    // same reason *inverted*: once the monitor latches, this is the only path
    // its clearing evidence can travel, and gating it on `interlockHealthy`
    // would make the latch unrevocable — an exclusion whose release test flows
    // through the gate the exclusion closes. The *largest* across units, because
    // one corrupted unit is enough evidence and an average would let a healthy
    // one hide it.
    const Eigen::Vector3d raw(m.get_fieldTesla()[0], m.get_fieldTesla()[1], m.get_fieldTesla()[2]);
    const double magnitude = raw.norm();
    if (std::isfinite(magnitude) && magnitude > this->pub_mag_raw_t_) {
      this->pub_mag_raw_t_ = magnitude;
      this->pub_mag_raw_valid_ = true;
    }

    // Then the **health** half, which is the consumption gate: with a rod
    // latched stuck-on no window is quiet whatever the clock says. Reported to
    // the voter as *absence* rather than as implausibility — exactly like the
    // staleness gate above, and for the same reason: a corrupted window says
    // nothing about whether the unit is lying, and must not latch an exclusion
    // the ground then has to reason about.
    if (!this->magSampleConsumable(m.get_timeTagNs())) {
      ++this->mag_interlock_rejects_;
      continue;
    }

    // **Raw**, before any applied hard/soft-iron correction: the vote decides
    // which unit's reading the vehicle believes, and a calibration fitted for one
    // unit must never be used to judge another. The correction is applied to the
    // vote's output, downstream.
    units[i].field_tesla = pm::Vec3<Body>(raw);
    units[i].present = true;
  }

  // The interlock is excluding otherwise-usable samples: an operator-visible
  // state, since it means the vehicle is flying without a magnetic pair. Edge
  // gated and then repeated at the shared alert cadence, so a latched rod costs
  // a bounded event stream rather than one per 10 Hz cycle.
  const bool excluding = this->have_mtq_schedule_ && !this->mtq_schedule_.get_interlockHealthy();
  if (excluding) {
    ++this->mag_interlock_excluding_cycles_;
    if (this->imu_ambiguity_escalate_cycles_ == 0 || this->mag_interlock_excluding_cycles_ == 1 ||
        (this->mag_interlock_excluding_cycles_ % this->imu_ambiguity_escalate_cycles_) == 0) {
      this->log_WARNING_HI_MagInterlockExcluding(this->mag_interlock_rejects_,
                                                 this->mag_interlock_excluding_cycles_);
    }
  } else if (this->mag_interlock_excluding_cycles_ != 0) {
    this->log_ACTIVITY_HI_MagInterlockRestored(this->mag_interlock_excluding_cycles_);
    this->mag_interlock_excluding_cycles_ = 0;
  }

  polaris::gnc::MagVoteReference reference;
  reference.field_tesla = magRefBody;
  reference.attitude_valid = haveAttitude;
  reference.attitude_sigma_rad = attitudeSigmaRad;

  polaris::gnc::MagVoteResult result;
  const bool ok = this->mag_voter_.vote(units, NUM_MAGNETOMETERIN_INPUT_PORTS, modelledMagnitudeT,
                                        &reference, result);

  // FDIR edges, reported on the transition so a permanently dead unit costs one
  // event rather than one per 10 Hz cycle.
  for (FwIndexType i = 0; i < NUM_MAGNETOMETERIN_INPUT_PORTS; ++i) {
    if (result.newly_excluded[i]) {
      this->log_WARNING_HI_MagUnitExcluded(static_cast<U8>(i),
                                           toMagExclusionReason(result.reason[i]));
    }
    if (result.newly_readmitted[i]) {
      this->log_ACTIVITY_HI_MagUnitReadmitted(static_cast<U8>(i));
    }
  }

  // Two plausible units disagreeing with nothing to attribute it. Edge-gated with
  // the same persistence escalation the IMU vote carries and on the same shared
  // horizon — but the *cost* is smaller and deliberately so: this loses one of two
  // vector pairs, where the IMU case loses the body rate. The vehicle keeps
  // propagating and, with the Sun in view, keeps acquiring.
  if (result.status == polaris::gnc::MagVoteStatus::kAmbiguous) {
    if (this->mag_ambiguous_cycles_ == 0) {
      this->mag_ambiguous_start_ns_ = nowTaiNs;
    }
    ++this->mag_ambiguous_cycles_;
    if (this->imu_ambiguity_escalate_cycles_ > 0 &&
        (this->mag_ambiguous_cycles_ % this->imu_ambiguity_escalate_cycles_) == 0) {
      const double elapsed_s =
          static_cast<double>(nowTaiNs - this->mag_ambiguous_start_ns_) / 1.0e9;
      this->log_WARNING_HI_MagVoteAmbiguousPersistent(elapsed_s, this->mag_ambiguous_cycles_);
    }
    if (!this->mag_ambiguous_flagged_) {
      double worst = 0.0;
      for (FwIndexType i = 0; i < NUM_MAGNETOMETERIN_INPUT_PORTS; ++i) {
        for (FwIndexType j = static_cast<FwIndexType>(i + 1); j < NUM_MAGNETOMETERIN_INPUT_PORTS;
             ++j) {
          if (result.reason[i] != polaris::gnc::MagVoteReason::kContributing ||
              result.reason[j] != polaris::gnc::MagVoteReason::kContributing) {
            continue;
          }
          const double difference =
              (units[i].field_tesla.eigen() - units[j].field_tesla.eigen()).norm();
          worst = (difference > worst) ? difference : worst;
        }
      }
      this->log_WARNING_HI_MagVoteAmbiguous(worst);
      this->mag_ambiguous_flagged_ = true;
    }
  } else {
    this->mag_ambiguous_flagged_ = false;
    this->mag_ambiguous_cycles_ = 0;
  }

  this->tlmWrite_MagContributing(static_cast<U32>(result.contributing));
  this->tlmWrite_MagExclusionMask(result.exclusion_mask);
  this->tlmWrite_MagInterlockRejects(this->mag_interlock_rejects_);
  this->tlmWrite_MagUnitSelected(
      result.published_index >= 0 ? static_cast<U8>(result.published_index) : kNoUnitIndex);

  if (!ok) {
    return false;
  }
  field = result.field_tesla;
  index = static_cast<FwIndexType>(result.published_index);
  return true;
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
    const Eigen::Vector3d r(m.get_posEcefM()[0], m.get_posEcefM()[1], m.get_posEcefM()[2]);
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

}  // namespace flight

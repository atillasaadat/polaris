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
    case polaris::gnc::ImuVoteReason::kRateLimit:
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

const SunSensorMeas* AttitudeEstimator ::selectSunSensor(I64 nowTaiNs, FwIndexType& index) const {
  // **Selection, not a weighted combination**, and the reason is the budget
  // rather than the algebra: the sun measurements' error is dominated by a
  // *shared* systematic — the albedo residual and the ephemeris term are common
  // to every unit on the vehicle — which combining cannot average down. A second
  // unit at a worse incidence therefore buys noise reduction on the small term
  // and nothing at all on the large one, while costing a correlation the
  // estimators' white-R model has no way to represent.
  //
  // The criterion is the **smallest realised sigma** (§6.4), which the unit
  // reports per sample from its own incidence angle. That is the incidence
  // cosine expressed in the quantity the estimator actually consumes, so it
  // stays right if a future suite mixes parts with different accuracy curves —
  // which raw incidence would not.
  const SunSensorMeas* best = nullptr;
  double best_sigma = 0.0;

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
      best = &this->sun_[i];
      best_sigma = sigma;
      index = i;
    }
  }
  return best;
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

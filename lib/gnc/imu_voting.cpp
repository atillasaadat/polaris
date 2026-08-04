/// @file
/// @brief Multi-IMU rate vote: the gyro-specific gates over the shared
/// redundancy-vote policy in gnc/unit_voting (design doc §8.2, §9.2;
/// REQ-ADET-008, REQ-ADET-009). See imu_voting.hpp for what is gyro-specific and
/// unit_voting.hpp for the policy itself.

#include "gnc/imu_voting.hpp"

#include <cmath>

namespace polaris::gnc {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;

}  // namespace

bool ImuVoteConfig::isValid() const {
  UnitVoteConfig core{};
  core.disagreement = disagreement_radps;
  core.readmit_cycles = readmit_cycles;
  core.identify_confirm_cycles = identify_confirm_cycles;
  return std::isfinite(max_rate_radps) && max_rate_radps > 0.0 && core.isValid();
}

bool medianRate(const pm::Vec3<Body>* rates, int count, pm::Vec3<Body>& out) {
  if (rates == nullptr || count < 1 || count > kMaxImuUnits) {
    return false;
  }
  Eigen::Vector3d values[kMaxImuUnits];
  for (int i = 0; i < count; ++i) {
    values[i] = rates[i].eigen();
  }
  Eigen::Vector3d median;
  if (!medianVector(values, count, median)) {
    return false;
  }
  out = pm::Vec3<Body>(median);
  return true;
}

ImuVoter::ImuVoter(const ImuVoteConfig& config) {
  if (config.isValid()) {
    config_ = config;
    UnitVoteConfig core{};
    core.disagreement = config.disagreement_radps;
    core.readmit_cycles = config.readmit_cycles;
    core.identify_confirm_cycles = config.identify_confirm_cycles;
    core_ = UnitVoter(core);
  }
}

bool ImuVoter::vote(const ImuVoteInput* units, int count, const pm::Vec3<Body>* reference_rate,
                    ImuVoteResult& out) {
  out = ImuVoteResult{};
  if (!core_.isConfigured() || units == nullptr || count < 0 || count > kMaxImuUnits) {
    return false;
  }

  // The gyro-specific plausibility gate: magnitude against the *vehicle's*
  // credible rate, not the gyro's measurement range. Finiteness is the core's, so
  // a non-finite reading must not be run through `norm()` here first — an
  // `in_range` computed from a NaN is meaningless either way, and the core reports
  // it as kNotFinite, which is the more actionable gate name.
  UnitVoteInput staged[kMaxVoteUnits];
  for (int i = 0; i < count; ++i) {
    staged[i].value = units[i].rate.eigen();
    staged[i].present = units[i].present;
    staged[i].in_range = units[i].rate.isFinite() && units[i].rate.norm() <= config_.max_rate_radps;
  }

  // The reference is wire-adjacent data like any other: it comes from the
  // estimator's own published state, which a fault upstream can corrupt. It
  // therefore passes the same plausibility gate the measurements do; failing it
  // means no reference, which is the honest ambiguous outcome (imu_voting.hpp).
  Eigen::Vector3d reference;
  const bool reference_usable = reference_rate != nullptr && reference_rate->isFinite() &&
                                reference_rate->norm() <= config_.max_rate_radps;
  if (reference_usable) {
    reference = reference_rate->eigen();
  }

  UnitVoteResult core_out;
  const bool ok = core_.vote(staged, count, reference_usable ? &reference : nullptr, core_out);

  out.rate = pm::Vec3<Body>(core_out.value);
  out.valid = core_out.valid;
  out.status = core_out.status;
  out.contributing = core_out.contributing;
  out.exclusion_mask = core_out.exclusion_mask;
  for (int i = 0; i < kMaxImuUnits; ++i) {
    out.reason[i] = core_out.reason[i];
    out.newly_excluded[i] = core_out.newly_excluded[i];
    out.newly_readmitted[i] = core_out.newly_readmitted[i];
  }
  return ok;
}

}  // namespace polaris::gnc

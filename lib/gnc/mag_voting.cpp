/// @file
/// @brief Multi-magnetometer field vote: the magnetometer-specific gates over the
/// shared redundancy-vote policy in gnc/unit_voting (design doc §8.2, §9.2;
/// REQ-ADET-011). See mag_voting.hpp for what is magnetometer-specific and
/// unit_voting.hpp for the policy itself.

#include "gnc/mag_voting.hpp"

#include <cmath>

namespace polaris::gnc {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;

}  // namespace

bool MagVoteConfig::isValid() const {
  UnitVoteConfig core{};
  core.disagreement = disagreement_tesla;
  core.readmit_cycles = readmit_cycles;
  core.identify_confirm_cycles = identify_confirm_cycles;
  // min < 1 < max is required rather than merely min < max: a band that excluded
  // the modelled magnitude itself would reject every healthy unit on every cycle,
  // which is a configuration error and not a tight gate.
  const bool band = std::isfinite(min_field_ratio) && std::isfinite(max_field_ratio) &&
                    min_field_ratio > 0.0 && min_field_ratio < 1.0 && max_field_ratio > 1.0;
  return band && std::isfinite(max_attitude_sigma_rad) && max_attitude_sigma_rad > 0.0 &&
         core.isValid();
}

MagVoter::MagVoter(const MagVoteConfig& config) {
  if (config.isValid()) {
    config_ = config;
    UnitVoteConfig core{};
    core.disagreement = config.disagreement_tesla;
    core.readmit_cycles = config.readmit_cycles;
    core.identify_confirm_cycles = config.identify_confirm_cycles;
    core_ = UnitVoter(core);
  }
}

bool MagVoter::vote(const MagVoteInput* units, int count, double modelled_magnitude_tesla,
                    const MagVoteReference* reference, MagVoteResult& out) {
  out = MagVoteResult{};
  if (!core_.isConfigured() || units == nullptr || count < 0 || count > kMaxMagUnits) {
    return false;
  }
  // Refused rather than run with the band disabled. Without a modelled field the
  // plausibility gate has nothing to compare against, and a vote whose only
  // surviving gate is finiteness would admit a railed unit into the pair — which
  // is the exact reading the band exists to catch. The caller has no magnetic pair
  // on such a cycle anyway (mag_voting.hpp).
  if (!std::isfinite(modelled_magnitude_tesla) || !(modelled_magnitude_tesla > 0.0)) {
    return false;
  }

  const double low = config_.min_field_ratio * modelled_magnitude_tesla;
  const double high = config_.max_field_ratio * modelled_magnitude_tesla;

  UnitVoteInput staged[kMaxVoteUnits];
  for (int i = 0; i < count; ++i) {
    staged[i].value = units[i].field_tesla.eigen();
    staged[i].present = units[i].present;
    const double magnitude = units[i].field_tesla.norm();
    staged[i].in_range = units[i].field_tesla.isFinite() && magnitude >= low && magnitude <= high;
  }

  // The reference is usable only if the attitude behind it is good enough to be
  // worth believing — a *quality* condition, not a validity flag, because a
  // validity flag cannot tell a 0.5° solution from a 10° one and only the former
  // predicts the field to better than the disagreement gate (mag_voting.hpp).
  Eigen::Vector3d reference_field;
  const bool reference_usable =
      reference != nullptr && reference->attitude_valid && reference->field_tesla.isFinite() &&
      std::isfinite(reference->attitude_sigma_rad) && reference->attitude_sigma_rad >= 0.0 &&
      reference->attitude_sigma_rad <= config_.max_attitude_sigma_rad;
  if (reference_usable) {
    reference_field = reference->field_tesla.eigen();
  }

  UnitVoteResult core_out;
  const bool ok =
      core_.vote(staged, count, reference_usable ? &reference_field : nullptr, core_out);

  out.field_tesla = pm::Vec3<Body>(core_out.value);
  out.valid = core_out.valid;
  out.status = core_out.status;
  out.contributing = core_out.contributing;
  out.exclusion_mask = core_out.exclusion_mask;
  for (int i = 0; i < kMaxMagUnits; ++i) {
    out.reason[i] = core_out.reason[i];
    out.newly_excluded[i] = core_out.newly_excluded[i];
    out.newly_readmitted[i] = core_out.newly_readmitted[i];
    if (out.published_index < 0 && core_out.reason[i] == VoteReason::kContributing) {
      out.published_index = i;
    }
  }
  return ok;
}

}  // namespace polaris::gnc

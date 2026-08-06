#include "gnc/bdot.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::gnc {
namespace {

constexpr double kNsPerSecond = 1.0e9;

/// Fill @p out with a refusal: zero dipole (the safe command) and the reason.
bool refuse(BdotRefusal reason, BdotResult& out) {
  out = BdotResult{};
  out.refusal = reason;
  return false;
}

}  // namespace

bool BdotConfig::isValid() const {
  return std::isfinite(gain_nms) && gain_nms > 0.0 && std::isfinite(duty_factor) &&
         duty_factor > 0.0 && duty_factor <= 1.0 && std::isfinite(min_sample_dt_s) &&
         min_sample_dt_s > 0.0 && std::isfinite(max_sample_dt_s) &&
         max_sample_dt_s > min_sample_dt_s;
}

BdotController::BdotController(const BdotConfig& config)
    : config_(config), configured_(config.isValid()) {}

void BdotController::reset() {
  have_previous_ = false;
  previous_field_ = math::Vec3<math::frames::Body>{};
  previous_time_ns_ = 0;
}

bool BdotController::update(const math::Vec3<math::frames::Body>& field_tesla,
                            std::int64_t time_tag_tai_ns, BdotResult& out) {
  if (!configured_) {
    return refuse(BdotRefusal::kUnconfigured, out);
  }

  const Eigen::Vector3d field = field_tesla.eigen();
  const double field_norm = field.norm();
  if (!field.allFinite() || !(field_norm > 0.0)) {
    // A non-finite or null field cannot be differenced *or* normalised. Drop the
    // stored sample: the next good one starts a fresh pair rather than being
    // differenced across an unknown gap.
    reset();
    return refuse(BdotRefusal::kBadInput, out);
  }

  if (!have_previous_) {
    previous_field_ = field_tesla;
    previous_time_ns_ = time_tag_tai_ns;
    have_previous_ = true;
    return refuse(BdotRefusal::kNoPreviousSample, out);
  }

  if (time_tag_tai_ns <= previous_time_ns_) {
    // A stuck clock is the dangerous one: re-differencing at the same epoch
    // divides by zero, and a backwards tag inverts the sign of the command.
    // Neither is recoverable by using the sample, so the pair is dropped and the
    // newer sample becomes the anchor.
    previous_field_ = field_tesla;
    previous_time_ns_ = time_tag_tai_ns;
    return refuse(BdotRefusal::kNonMonotonicTime, out);
  }

  const double dt_s = static_cast<double>(time_tag_tai_ns - previous_time_ns_) / kNsPerSecond;
  if (dt_s < config_.min_sample_dt_s) {
    // Too close together: keep the older anchor and wait for separation, so a
    // burst of samples does not throw away the one usable baseline.
    return refuse(BdotRefusal::kIntervalTooShort, out);
  }
  if (dt_s > config_.max_sample_dt_s) {
    previous_field_ = field_tesla;
    previous_time_ns_ = time_tag_tai_ns;
    return refuse(BdotRefusal::kIntervalTooLong, out);
  }

  const Eigen::Vector3d field_rate = (field - previous_field_.eigen()) / dt_s;
  previous_field_ = field_tesla;
  previous_time_ns_ = time_tag_tai_ns;

  // m = -(k / |B|^2) dB/dt, then the duty-factor scale-up so the *average* dipole
  // over a control period is the demanded one (§7, §8.5). Unclamped: the rods'
  // rated moments are a limit in the *rod* basis, and clamping here as well would
  // throw away authority on any triad that is not body-aligned (see the header).
  const Eigen::Vector3d dipole =
      -(config_.gain_nms / (field_norm * field_norm * config_.duty_factor)) * field_rate;
  if (!dipole.allFinite()) {
    return refuse(BdotRefusal::kNonFiniteOutput, out);
  }

  out.dipole_am2 = math::Vec3<math::frames::Body>(dipole);
  out.field_rate_tps = math::Vec3<math::frames::Body>(field_rate);
  out.sample_dt_s = dt_s;
  out.valid = true;
  out.refusal = BdotRefusal::kNone;
  return true;
}

bool RateHysteresisConfig::isValid() const {
  return std::isfinite(enter_radps) && enter_radps > 0.0 && std::isfinite(exit_radps) &&
         exit_radps > 0.0 && exit_radps < enter_radps && confirm_cycles > 0;
}

RateHysteresis::RateHysteresis(const RateHysteresisConfig& config)
    : config_(config), configured_(config.isValid()) {}

void RateHysteresis::reset() {
  tumbling_ = true;
  below_streak_ = 0;
}

void RateHysteresis::clear() {
  tumbling_ = false;
  below_streak_ = config_.confirm_cycles;
}

bool RateHysteresis::update(double rate_norm_radps) {
  if (!configured_ || !std::isfinite(rate_norm_radps)) {
    return tumbling_;
  }
  if (rate_norm_radps > config_.enter_radps) {
    below_streak_ = 0;
    tumbling_ = true;
    return tumbling_;
  }
  if (rate_norm_radps < config_.exit_radps) {
    if (below_streak_ < config_.confirm_cycles) {
      below_streak_++;
    }
    if (below_streak_ >= config_.confirm_cycles) {
      tumbling_ = false;
    }
    return tumbling_;
  }
  // Inside the deadband: no new evidence either way, so the streak is broken
  // (the rate is not confirmed low) but the verdict is held.
  below_streak_ = 0;
  return tumbling_;
}

}  // namespace polaris::gnc

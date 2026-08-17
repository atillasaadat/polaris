#include "sensors/gnss.hpp"

#include <cmath>

namespace polaris::sim::sensors {
namespace {

/// Look a key up, defaulting to 0 (term disabled) when absent — the shared
/// convention across the sensor `fromParams` (sun_sensor.cpp, magnetometer.cpp).
double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// √2 — a 2D horizontal DRMS is √(σ_e²+σ_n²) = σ·√2 for equal per-axis σ.
constexpr double kSqrt2 = 1.4142135623730951;
/// Vertical is poorer than horizontal for the same geometry; the usual VDOP/HDOP
/// ratio when a datasheet quotes only the horizontal figure.
constexpr double kVerticalToHorizontal = 1.5;
constexpr double kNsPerSecond = 1.0e9;

/// Local geodetic east/north/up basis at ECEF position @p r, using geocentric up
/// (r̂). Geocentric and geodetic up differ by <0.2° for an oblate Earth — well
/// inside the point of splitting horizontal from vertical error at all. At the
/// poles east is degenerate; fall back to the X axis there.
void enuBasis(const Eigen::Vector3d& r, Eigen::Vector3d& east, Eigen::Vector3d& north,
              Eigen::Vector3d& up) {
  up = r.normalized();
  east = Eigen::Vector3d::UnitZ().cross(up);
  if (east.norm() < 1.0e-9) {
    east = Eigen::Vector3d::UnitX();
  }
  east.normalize();
  north = up.cross(east);
}

}  // namespace

GnssSpec GnssSpec::fromParams(const std::map<std::string, double>& p) {
  GnssSpec s;

  const double h_rms = get(p, "horizontal_position_rms_m");
  s.position_sigma_h_m = (h_rms > 0.0) ? h_rms / kSqrt2 : 0.0;
  const double v_rms = get(p, "vertical_position_rms_m");
  s.position_sigma_v_m = (v_rms > 0.0) ? v_rms : s.position_sigma_h_m * kVerticalToHorizontal;

  s.velocity_sigma_m_s = get(p, "velocity_accuracy_m_s_rms");
  s.time_sigma_s = get(p, "time_accuracy_ns_rms") * 1.0e-9;

  s.max_rate_hz = get(p, "max_rate_hz");
  s.sample_period_s = (s.max_rate_hz > 0.0) ? 1.0 / s.max_rate_hz : 0.0;

  s.fix_latency_s = get(p, "fix_latency_s");
  s.fix_latency_jitter_s = get(p, "fix_latency_jitter_s");

  s.cold_start_s = get(p, "cold_start_s");
  s.hot_start_s = get(p, "hot_start_s");
  s.reacquisition_s = get(p, "reacquisition_s");
  return s;
}

GnssMeasurement Gnss::sample(const time::Tai& epoch, const GnssInput& input) {
  const time::Gps gps = time::toGps(epoch);
  const std::int64_t gps_ns = gps.nanosecondsSinceEpoch();

  // Sample-period gating: within one period of the last genuine fix, the receiver
  // has not produced a new solution — return the previous one, flagged stale.
  if (has_sampled_ && spec_.sample_period_s > 0.0) {
    const double elapsed =
        static_cast<double>(gps_ns - last_fix_time_.nanosecondsSinceEpoch()) / kNsPerSecond;
    if (elapsed >= 0.0 && elapsed < spec_.sample_period_s) {
      GnssMeasurement repeated = last_fix_;
      repeated.fresh = false;
      return repeated;
    }
  }

  // Geographic jamming is a loss of fix gated by the sub-satellite point, so it
  // folds into the same "outage" signal as a commanded dropout: same
  // invalidation, same reacquisition delay on exit.
  const std::string* jam_region =
      (jamming_ != nullptr) ? jamming_->jammedRegion(input.position_m) : nullptr;
  const bool outage = fault_outage_ || jam_region != nullptr;

  // Acquisition / reacquisition timing. A cold start withholds fixes for the
  // time-to-first-fix; the falling edge of an outage arms the reacquisition delay.
  if (!has_sampled_) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.cold_start_s * kNsPerSecond);
  } else if (prev_outage_ && !outage) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.reacquisition_s * kNsPerSecond);
  }
  prev_outage_ = outage;

  GnssMeasurement m;
  m.position_sigma_h_m = spec_.position_sigma_h_m;
  m.position_sigma_v_m = spec_.position_sigma_v_m;
  m.velocity_sigma_m_s = spec_.velocity_sigma_m_s;
  m.time_sigma_s = spec_.time_sigma_s;

  // Position error, split horizontal vs vertical in the local geodetic frame.
  // With noise disabled the fix is truth-exact (the spoof offset still applies —
  // it is a fault, not measurement noise).
  Eigen::Vector3d pos_err = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_err = Eigen::Vector3d::Zero();
  double clock_noise = 0.0;
  if (spec_.noise_enabled) {
    Eigen::Vector3d east;
    Eigen::Vector3d north;
    Eigen::Vector3d up;
    enuBasis(input.position_m.eigen(), east, north, up);
    pos_err = spec_.position_sigma_h_m * rng_.gaussian() * east +
              spec_.position_sigma_h_m * rng_.gaussian() * north +
              spec_.position_sigma_v_m * rng_.gaussian() * up;
    vel_err = Eigen::Vector3d(spec_.velocity_sigma_m_s * rng_.gaussian(),
                              spec_.velocity_sigma_m_s * rng_.gaussian(),
                              spec_.velocity_sigma_m_s * rng_.gaussian());
    clock_noise = spec_.time_sigma_s * rng_.gaussian();
  }
  m.position_m =
      math::Vec3<math::frames::ECEF>(input.position_m.eigen() + pos_err + fault_pos_offset_);
  m.velocity_m_s = math::Vec3<math::frames::ECEF>(input.velocity_m_s.eigen() + vel_err);

  // Receiver-clock bias on the time tag (noise + any injected jump).
  m.clock_bias_s = clock_noise + fault_clock_jump_s_;
  m.time_tag = time::Gps::fromNanosecondsSinceEpoch(
      gps_ns + static_cast<std::int64_t>(std::llround(m.clock_bias_s * kNsPerSecond)));

  m.fresh = true;
  m.valid = !outage && gps_ns >= valid_from_gps_ns_;
  m.jammed = jam_region != nullptr;
  if (jam_region != nullptr) {
    m.jamming_region = *jam_region;
  }

  has_sampled_ = true;
  last_fix_time_ = gps;

  if (!(spec_.fix_latency_s > 0.0)) {
    last_fix_ = m;
    return m;
  }

  // --- Fix latency: the solution just computed is not the one delivered ------
  // `m` describes the vehicle at `gps_ns` and enters the delay line; what leaves
  // it is the newest solution that has been in there at least `fix_latency_s`.
  // The tag stays at the *measurement* epoch — a receiver stamps when it
  // measured, not when it finished talking — which is precisely what makes the
  // latency correctable downstream instead of merely wrong.
  if (pending_count_ == kMaxPending) {
    for (std::size_t i = 1; i < kMaxPending; ++i) {
      pending_[i - 1] = pending_[i];
      pending_epoch_ns_[i - 1] = pending_epoch_ns_[i];
    }
    pending_count_ -= 1;
    pending_dropped_ += 1;
  }
  // Each fix carries its own delivery epoch: the latency plus a per-fix jitter
  // draw, clamped at zero. The tag `m` carries stays the measurement epoch.
  double latency_s = spec_.fix_latency_s;
  if (spec_.fix_latency_jitter_s > 0.0) {
    latency_s = std::max(0.0, latency_s + spec_.fix_latency_jitter_s * rng_.gaussian());
  }
  pending_[pending_count_] = m;
  pending_epoch_ns_[pending_count_] =
      gps_ns + static_cast<std::int64_t>(std::llround(latency_s * kNsPerSecond));
  pending_count_ += 1;

  std::size_t ready = 0;
  while (ready < pending_count_ && pending_epoch_ns_[ready] <= gps_ns) {
    ready += 1;
  }
  if (ready == 0) {
    // Nothing has cleared the receiver yet. Report the sigmas and flags of a
    // receiver with no solution rather than a stale or truth-exact position:
    // this is the same "no fix available" state a cold start is in, and a
    // consumer must treat it the same way.
    GnssMeasurement none;
    none.position_sigma_h_m = spec_.position_sigma_h_m;
    none.position_sigma_v_m = spec_.position_sigma_v_m;
    none.velocity_sigma_m_s = spec_.velocity_sigma_m_s;
    none.time_sigma_s = spec_.time_sigma_s;
    none.time_tag = gps;
    none.fresh = true;
    none.valid = false;
    last_fix_ = none;
    return none;
  }

  const GnssMeasurement delivered = pending_[ready - 1];
  const std::size_t remaining = pending_count_ - ready;
  for (std::size_t i = 0; i < remaining; ++i) {
    pending_[i] = pending_[ready + i];
    pending_epoch_ns_[i] = pending_epoch_ns_[ready + i];
  }
  pending_count_ = remaining;
  // `last_fix_` is what the receiver last *reported*, not what it last computed:
  // the sample-period repeat path above replays it, and replaying a solution
  // still sitting in the delay line would deliver it early.
  last_fix_ = delivered;
  return delivered;
}

}  // namespace polaris::sim::sensors

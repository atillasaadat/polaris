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

  // Acquisition / reacquisition timing. A cold start withholds fixes for the
  // time-to-first-fix; the falling edge of an outage arms the reacquisition delay.
  if (!has_sampled_) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.cold_start_s * kNsPerSecond);
  } else if (prev_outage_ && !fault_outage_) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.reacquisition_s * kNsPerSecond);
  }
  prev_outage_ = fault_outage_;

  GnssMeasurement m;
  m.position_sigma_h_m = spec_.position_sigma_h_m;
  m.position_sigma_v_m = spec_.position_sigma_v_m;
  m.velocity_sigma_m_s = spec_.velocity_sigma_m_s;
  m.time_sigma_s = spec_.time_sigma_s;

  // Position error, split horizontal vs vertical in the local geodetic frame.
  Eigen::Vector3d east;
  Eigen::Vector3d north;
  Eigen::Vector3d up;
  enuBasis(input.position_m.eigen(), east, north, up);
  const Eigen::Vector3d pos_err = spec_.position_sigma_h_m * rng_.gaussian() * east +
                                  spec_.position_sigma_h_m * rng_.gaussian() * north +
                                  spec_.position_sigma_v_m * rng_.gaussian() * up;
  m.position_m = math::Vec3<math::frames::ECEF>(input.position_m.eigen() + pos_err +
                                                fault_pos_offset_);

  // Velocity error, per-axis white in ECEF (the datasheet quotes no H/V split).
  const Eigen::Vector3d vel_err(spec_.velocity_sigma_m_s * rng_.gaussian(),
                                spec_.velocity_sigma_m_s * rng_.gaussian(),
                                spec_.velocity_sigma_m_s * rng_.gaussian());
  m.velocity_m_s = math::Vec3<math::frames::ECEF>(input.velocity_m_s.eigen() + vel_err);

  // Receiver-clock bias on the time tag.
  m.clock_bias_s = spec_.time_sigma_s * rng_.gaussian() + fault_clock_jump_s_;
  m.time_tag = time::Gps::fromNanosecondsSinceEpoch(
      gps_ns + static_cast<std::int64_t>(std::llround(m.clock_bias_s * kNsPerSecond)));

  m.fresh = true;
  m.valid = !fault_outage_ && gps_ns >= valid_from_gps_ns_;

  has_sampled_ = true;
  last_fix_time_ = gps;
  last_fix_ = m;
  return m;
}

}  // namespace polaris::sim::sensors

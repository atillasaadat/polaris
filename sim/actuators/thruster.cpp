#include "actuators/thruster.hpp"

#include <algorithm>

#include "constants/constants.hpp"

namespace polaris::sim::actuators {

namespace {
double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}
}  // namespace

ThrusterSpec ThrusterSpec::fromParams(const std::map<std::string, double>& p) {
  ThrusterSpec s;
  s.thrust_n = get(p, "thrust_n");
  s.isp_s = get(p, "isp_s");
  s.min_impulse_bit_ns = get(p, "min_impulse_bit_ns");
  s.rise_time_s = get(p, "rise_time_s");
  s.fall_time_s = get(p, "fall_time_s");
  s.thrust_scale_error = get(p, "thrust_scale_error");
  s.thrust_noise_frac = get(p, "thrust_noise_frac");
  s.misalignment_rad = get(p, "misalignment_rad");
  s.misalignment_azimuth_rad = get(p, "misalignment_azimuth_rad");
  return s;
}

ThrusterOutput Thruster::step(double dt) {
  ThrusterOutput out;
  if (!(dt > 0.0)) {
    out.delivered_thrust_n = thrust_;
    out.throttle = commanded_;
    return out;
  }
  // 1. Effective command: faults override the throttle; the scale error is a
  //    property of the unit, so it multiplies whatever is asked for.
  double throttle = commanded_;
  if (fault_off_) {
    throttle = 0.0;
  } else if (fault_on_) {
    throttle = 1.0;
  }
  const double target = throttle * spec_.thrust_n * (1.0 + spec_.thrust_scale_error);

  // 2. Valve/chamber lag, exact over the step; zero constant delivers now.
  const double tau = target > thrust_ ? spec_.rise_time_s : spec_.fall_time_s;
  thrust_ = tau > 0.0 ? target + (thrust_ - target) * std::exp(-dt / tau) : target;

  // 3. Minimum impulse bit: a pulse the valve cannot resolve delivers nothing.
  double delivered = thrust_;
  if (spec_.min_impulse_bit_ns > 0.0 && delivered * dt < spec_.min_impulse_bit_ns) {
    delivered = 0.0;
  }
  // 4. White thrust noise, per step.
  if (spec_.thrust_noise_frac > 0.0 && delivered > 0.0) {
    delivered *= 1.0 + spec_.thrust_noise_frac * rng_.gaussian();
    delivered = std::max(0.0, delivered);
  }

  // 5. Direction: +z tilted by the misalignment at its azimuth.
  const double a = spec_.misalignment_rad;
  const double az = spec_.misalignment_azimuth_rad;
  const Eigen::Vector3d dir(std::sin(a) * std::cos(az), std::sin(a) * std::sin(az), std::cos(a));
  out.force_n = delivered * dir;
  out.delivered_thrust_n = delivered;
  out.mass_flow_kg_s =
      spec_.isp_s > 0.0 ? delivered / (spec_.isp_s * constants::physical::kStandardGravity) : 0.0;
  out.throttle = throttle;
  return out;
}

}  // namespace polaris::sim::actuators

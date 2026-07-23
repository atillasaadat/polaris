#include "actuators/magnetorquer.hpp"

#include <algorithm>

namespace polaris::sim::actuators {

namespace {
double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}
}  // namespace

MagnetorquerSpec MagnetorquerSpec::fromParams(const std::map<std::string, double>& p) {
  MagnetorquerSpec s;
  s.max_dipole_am2 = get(p, "max_dipole_am2");
  s.residual_dipole_am2 = get(p, "residual_dipole_am2");
  s.linearity = get(p, "linearity");
  s.power_max_w = get(p, "power_max_w");
  return s;
}

double Magnetorquer::applyAxis(double cmd, double& state) const {
  // Saturate the command, then apply the scale-factor (linearity) error.
  double x = cmd;
  if (spec_.max_dipole_am2 > 0.0) {
    x = std::clamp(x, -spec_.max_dipole_am2, spec_.max_dipole_am2);
  }
  x *= (1.0 + spec_.linearity);

  // Play (backlash) hysteresis: the moment sticks within ±r of the command, so it
  // lags on reversal and retains ±r (the residual) when commanded to zero.
  const double r = spec_.residual_dipole_am2;
  state = std::clamp(state, x - r, x + r);
  double y = state;
  if (spec_.max_dipole_am2 > 0.0) {
    y = std::clamp(y, -spec_.max_dipole_am2, spec_.max_dipole_am2);
  }
  return y;
}

math::Vec3<math::frames::Body> Magnetorquer::commandDipole(
    const math::Vec3<math::frames::Body>& dipole_cmd) {
  if (fault_stuck_) {
    return actual_;  // hold the last produced moment
  }

  Eigen::Vector3d cmd = dipole_cmd.eigen();
  if (fault_dropout_) {
    cmd.setZero();  // no drive; the play operator relaxes toward ±residual
  }

  Eigen::Vector3d out;
  for (int i = 0; i < 3; ++i) {
    out[i] = applyAxis(cmd[i], hysteresis_state_[i]);
  }
  actual_ = math::Vec3<math::frames::Body>(out);
  return actual_;
}

double Magnetorquer::busPower() const {
  if (spec_.max_dipole_am2 <= 0.0 || spec_.power_max_w <= 0.0) {
    return 0.0;
  }
  // Each of the three rods has its own winding drawing power at once, and per rod
  // power ∝ dipole² (dipole ∝ current, P = I²R). Sum the per-axis contributions,
  // so full drive on all three axes draws three times the single-rod rating.
  double power = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double frac = actual_.eigen()[i] / spec_.max_dipole_am2;
    power += spec_.power_max_w * frac * frac;
  }
  return power;
}

}  // namespace polaris::sim::actuators

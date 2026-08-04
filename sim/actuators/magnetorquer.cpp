#include "actuators/magnetorquer.hpp"

#include <algorithm>
#include <cmath>

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
  s.settle_time_s = get(p, "settle_time_s");
  return s;
}

math::Vec3<math::frames::Body> dipoleNearField(const math::Vec3<math::frames::Body>& dipole_am2,
                                               const Eigen::Vector3d& source_m,
                                               const Eigen::Vector3d& observer_m) {
  // mu_0 / (4 pi) [T·m/A].
  constexpr double kMu0Over4Pi = 1.0e-7;
  const Eigen::Vector3d r = observer_m - source_m;
  const double distance = r.norm();
  if (!(distance > 0.0) || !r.allFinite()) {
    return math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  }
  const Eigen::Vector3d unit = r / distance;
  const Eigen::Vector3d m = dipole_am2.eigen();
  const Eigen::Vector3d field =
      kMu0Over4Pi / (distance * distance * distance) * (3.0 * m.dot(unit) * unit - m);
  return math::Vec3<math::frames::Body>(field);
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

void Magnetorquer::deenergize() {
  pre_off_ = actual_;
  // The rod really is commanded to zero at the end of each on-window, so the
  // play operator advances again — which is what leaves the remanent moment.
  // A stuck-on rod ignores it, exactly as commandDipole does, and that is the
  // fault the §7 interlock monitor exists to catch.
  (void)commandDipole(math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero()));
}

math::Vec3<math::frames::Body> Magnetorquer::settlingDipole(double seconds_since_off) const {
  const Eigen::Vector3d off = actual_.eigen();
  if (!(spec_.settle_time_s > 0.0) || !(seconds_since_off > 0.0)) {
    return seconds_since_off > 0.0 ? actual_ : pre_off_;
  }
  // tau = settle/3, so ~95% of the transient is gone at the catalog settle time.
  const double decay = std::exp(-3.0 * seconds_since_off / spec_.settle_time_s);
  return math::Vec3<math::frames::Body>(Eigen::Vector3d(off + (pre_off_.eigen() - off) * decay));
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

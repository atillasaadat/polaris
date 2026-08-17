#include "gnc/momentum.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Cholesky>

namespace polaris::gnc {
namespace {

bool refuseDesat(DesatRefusal reason, MtqDesatResult& out) {
  out = MtqDesatResult{};
  out.refusal = reason;
  return false;
}

bool refuseMomentum(MomentumRefusal reason, MomentumState& out, int refused_wheel = -1) {
  out = MomentumState{};
  out.refusal = reason;
  out.refused_wheel = refused_wheel;
  return false;
}

}  // namespace

// ----------------------------------------------------------------------
// Cross-product desaturation
// ----------------------------------------------------------------------

bool MtqDesatConfig::isValid() const {
  return std::isfinite(gain_per_s) && gain_per_s > 0.0 && std::isfinite(duty_factor) &&
         duty_factor > 0.0 && duty_factor <= 1.0;
}

bool mtqDesaturation(const MtqDesatConfig& config,
                     const math::Vec3<math::frames::Body>& momentum_error_nms,
                     const math::Vec3<math::frames::Body>& field_tesla, MtqDesatResult& out) {
  if (!config.isValid()) {
    return refuseDesat(DesatRefusal::kUnconfigured, out);
  }
  const Eigen::Vector3d dh = momentum_error_nms.eigen();
  const Eigen::Vector3d b = field_tesla.eigen();
  const double b_norm = b.norm();
  if (!dh.allFinite() || !b.allFinite() || !(b_norm > 0.0)) {
    return refuseDesat(DesatRefusal::kBadInput, out);
  }

  // m = (k_d / |B|^2) (dh x B), then the duty scale-up so the *average* dipole
  // over a control period is what the gain asked for (§7, §8.5). Unclamped: the
  // rated moment is a limit in the rod basis and the caller clamps there.
  const Eigen::Vector3d dipole =
      (config.gain_per_s / (b_norm * b_norm * config.duty_factor)) * dh.cross(b);
  const Eigen::Vector3d torque = dipole.cross(b);
  if (!dipole.allFinite() || !torque.allFinite()) {
    return refuseDesat(DesatRefusal::kNonFiniteOutput, out);
  }

  out.dipole_am2 = math::Vec3<math::frames::Body>(dipole);
  out.torque_nm = math::Vec3<math::frames::Body>(torque);
  out.valid = true;
  out.refusal = DesatRefusal::kNone;
  return true;
}

// ----------------------------------------------------------------------
// Momentum manager
// ----------------------------------------------------------------------

bool MomentumConfig::isValid() const {
  if (wheel_count < 3 || wheel_count > kMaxWheels) {
    return false;
  }
  for (int i = 0; i < wheel_count; ++i) {
    const Eigen::Vector3d axis = spin_axes.col(i);
    // Unit norm required, not merely non-zero: update() multiplies I_w*omega by
    // this column, so a column of length 2 doubles that wheel's momentum
    // contribution with no symptom anywhere downstream.
    if (!axis.allFinite() || std::abs(axis.norm() - 1.0) > 1.0e-9) {
      return false;
    }
  }
  // The axes must span the body: W W^T singular means a body axis no wheel can
  // load or unload, and the min-norm decomposition below has no meaning.
  Eigen::Matrix3d gram = Eigen::Matrix3d::Zero();
  for (int i = 0; i < wheel_count; ++i) {
    gram += spin_axes.col(i) * spin_axes.col(i).transpose();
  }
  if (!(gram.ldlt().isPositive()) || !(gram.determinant() > 1.0e-9)) {
    return false;
  }
  if (!std::isfinite(rotor_inertia_kgm2) || !(rotor_inertia_kgm2 > 0.0)) {
    return false;
  }
  if (!target_nms.allFinite()) {
    return false;
  }
  if (!std::isfinite(desat_enter_nms) || !(desat_enter_nms > 0.0) ||
      !std::isfinite(desat_exit_nms) || !(desat_exit_nms > 0.0) ||
      !(desat_exit_nms < desat_enter_nms) || desat_confirm_cycles == 0) {
    return false;
  }
  // The envelope is the FDIR ceiling on ||h_stored|| and the desat threshold is
  // the action that keeps the vehicle under it — but the threshold gates
  // ||h_stored - target||, a different quantity whenever the vehicle carries a
  // momentum bias. The ordering that makes the pair coherent is therefore
  // envelope >= ||target|| + enter: anything less lets a biased vehicle trip
  // the envelope at a stored momentum the desat law was never asked to unload.
  return std::isfinite(envelope_nms) && envelope_nms >= target_nms.norm() + desat_enter_nms;
}

Eigen::Vector3d MomentumManager::gramInverseTimes(const Eigen::Vector3d& v) const {
  Eigen::Matrix3d gram = Eigen::Matrix3d::Zero();
  for (int i = 0; i < config_.wheel_count; ++i) {
    gram += config_.spin_axes.col(i) * config_.spin_axes.col(i).transpose();
  }
  return gram.ldlt().solve(v);
}

MomentumManager::MomentumManager(const MomentumConfig& config)
    : config_(config), configured_(config.isValid()) {
  if (configured_) {
    RateHysteresisConfig h;
    h.enter_radps = config.desat_enter_nms;
    h.exit_radps = config.desat_exit_nms;
    h.confirm_cycles = config.desat_confirm_cycles;
    hysteresis_ = RateHysteresis(h);
    // The predicate starts "not desaturating". `RateHysteresis::reset` assumes
    // the unsafe answer, which is right for detumble and wrong here: a vehicle
    // whose momentum is not yet known must not begin by driving the rods.
    hysteresis_.clear();
  }
}

void MomentumManager::reset() {
  hysteresis_.clear();
}

bool MomentumManager::update(const double* wheel_speeds_radps, const bool* wheel_valid,
                             MomentumState& out) {
  if (!configured_) {
    return refuseMomentum(MomentumRefusal::kUnconfigured, out);
  }
  // A null array is a caller bug, not a tuning problem; naming it kUnconfigured
  // would send the operator to the parameter table for a defect in the code.
  if (wheel_speeds_radps == nullptr || wheel_valid == nullptr) {
    return refuseMomentum(MomentumRefusal::kBadInput, out);
  }
  Eigen::Vector3d stored = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, kMaxWheels, 1> wheel_nms = Eigen::Matrix<double, kMaxWheels, 1>::Zero();
  double max_wheel = 0.0;
  for (int i = 0; i < config_.wheel_count; ++i) {
    if (!wheel_valid[i]) {
      return refuseMomentum(MomentumRefusal::kWheelInvalid, out, i);
    }
    if (!std::isfinite(wheel_speeds_radps[i])) {
      return refuseMomentum(MomentumRefusal::kBadInput, out, i);
    }
    wheel_nms(i) = config_.rotor_inertia_kgm2 * wheel_speeds_radps[i];
    max_wheel = std::max(max_wheel, std::abs(wheel_nms(i)));
    stored += wheel_nms(i) * config_.spin_axes.col(i);
  }
  const Eigen::Vector3d error = stored - config_.target_nms;
  if (!stored.allFinite() || !error.allFinite()) {
    return refuseMomentum(MomentumRefusal::kBadInput, out);
  }
  // Null-space content: the wheel momenta less their minimum-norm realisation
  // of the body momentum, W^+ h. For N = 3 the pseudo-inverse is the inverse
  // and this is identically zero; for N = 4 it is the against-each-other spin
  // the body-momentum thresholds cannot see.
  const Eigen::Vector3d gram_solve = gramInverseTimes(stored);
  double null_space_sq = 0.0;
  for (int i = 0; i < config_.wheel_count; ++i) {
    const double min_norm_i = config_.spin_axes.col(i).dot(gram_solve);  // (W^T (WW^T)^-1 h)_i
    null_space_sq += (wheel_nms(i) - min_norm_i) * (wheel_nms(i) - min_norm_i);
  }
  const double null_space = std::sqrt(null_space_sq);

  out.stored_nms = math::Vec3<math::frames::Body>(stored);
  out.error_nms = math::Vec3<math::frames::Body>(error);
  out.stored_norm_nms = stored.norm();
  out.error_norm_nms = error.norm();
  out.max_wheel_nms = max_wheel;
  out.null_space_nms = std::isfinite(null_space) ? null_space : 0.0;
  out.desat_required = hysteresis_.update(out.error_norm_nms);
  out.envelope_exceeded = out.stored_norm_nms > config_.envelope_nms;
  out.valid = true;
  out.refusal = MomentumRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc

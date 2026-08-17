#include "gnc/rw_bias.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/SVD>

namespace polaris::gnc {

namespace {
/// Singular values below this fraction of the largest belong to the null space.
constexpr double kNullTol = 1.0e-9;
}  // namespace

bool RwBiasConfig::isValid() const {
  if (wheel_count < 3 || wheel_count > kMaxWheels) {
    return false;
  }
  if (!axes.leftCols(wheel_count).allFinite() || !std::isfinite(gain_per_s) ||
      !std::isfinite(max_torque_nm) || gain_per_s < 0.0 || max_torque_nm < 0.0) {
    return false;
  }
  if (gain_per_s > 0.0 && !(max_torque_nm > 0.0)) {
    return false;
  }
  for (int i = 0; i < wheel_count; ++i) {
    if (!std::isfinite(bias_nms[i])) {
      return false;
    }
  }
  return true;
}

RwBiasServo::RwBiasServo(const RwBiasConfig& config) : config_(config) {
  if (!config.isValid()) {
    return;
  }
  const int n = config.wheel_count;
  // Full SVD of the 3×n torque map: right singular vectors past the rank span
  // the null space. Fixed-size, bounded.
  Eigen::Matrix<double, 3, Eigen::Dynamic, 0, 3, kMaxWheels> a = config.axes.leftCols(n);
  Eigen::JacobiSVD<decltype(a)> svd(a, Eigen::ComputeFullV);
  const auto& sigma = svd.singularValues();
  const double smax = sigma.size() > 0 ? sigma(0) : 0.0;
  int rank = 0;
  for (int i = 0; i < sigma.size(); ++i) {
    if (sigma(i) > kNullTol * smax) {
      ++rank;
    }
  }
  null_dim_ = n - rank;
  for (int k = 0; k < null_dim_; ++k) {
    null_basis_.col(k).head(n) = svd.matrixV().col(rank + k);
  }
  Eigen::Matrix<double, kMaxWheels, 1> pattern = Eigen::Matrix<double, kMaxWheels, 1>::Zero();
  for (int i = 0; i < n; ++i) {
    pattern(i) = config.bias_nms[i];
  }
  const auto nb = null_basis_.leftCols(std::max(null_dim_, 1));
  effective_bias_ = (null_dim_ > 0)
                        ? Eigen::Matrix<double, kMaxWheels, 1>(nb * (nb.transpose() * pattern))
                        : Eigen::Matrix<double, kMaxWheels, 1>::Zero();
  configured_ = true;
}

double RwBiasServo::effectiveBiasNms(int wheel) const {
  if (!configured_ || wheel < 0 || wheel >= config_.wheel_count) {
    return 0.0;
  }
  return effective_bias_(wheel);
}

bool RwBiasServo::update(const double* wheel_momentum_nms, RwBiasResult& out) const {
  out = RwBiasResult{};
  if (!configured_ || wheel_momentum_nms == nullptr) {
    out.refusal = RwBiasRefusal::kUnconfigured;
    return false;
  }
  const int n = config_.wheel_count;
  Eigen::Matrix<double, kMaxWheels, 1> h = Eigen::Matrix<double, kMaxWheels, 1>::Zero();
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(wheel_momentum_nms[i])) {
      out.refusal = RwBiasRefusal::kBadInput;
      return false;
    }
    h(i) = wheel_momentum_nms[i];
  }
  out.valid = true;
  out.refusal = RwBiasRefusal::kNone;
  if (null_dim_ == 0) {
    return true;  // nothing to steer: inert by construction
  }
  const auto nb = null_basis_.leftCols(null_dim_);
  const Eigen::Matrix<double, kMaxWheels, 1> h_null = nb * (nb.transpose() * h);
  out.null_space_nms = h_null.norm();
  const bool off = config_.gain_per_s <= 0.0 || effective_bias_.norm() <= 0.0;
  if (off) {
    return true;
  }
  // Error in the null space only; the projection of h_b is already null-space.
  Eigen::Matrix<double, kMaxWheels, 1> torque = config_.gain_per_s * (effective_bias_ - h_null);
  // One scale factor for the whole vector, never a per-wheel clip: a clipped
  // component leaves the null space and would put torque on the body.
  const double peak = torque.head(n).cwiseAbs().maxCoeff();
  if (peak > config_.max_torque_nm) {
    torque *= config_.max_torque_nm / peak;
  }
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(torque(i))) {
      out.valid = false;
      out.refusal = RwBiasRefusal::kBadInput;
      return false;
    }
    out.torque_nm[i] = torque(i);
    out.active = out.active || torque(i) != 0.0;
  }
  return true;
}

}  // namespace polaris::gnc

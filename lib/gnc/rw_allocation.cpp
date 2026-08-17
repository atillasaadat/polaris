#include "gnc/rw_allocation.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace polaris::gnc {
namespace {

/// Largest candidate list the L-∞ line search evaluates: alpha = 0, one zero
/// crossing per wheel, and two equal-magnitude crossings per wheel pair.
constexpr int kMaxCandidates = 1 + kMaxWheels + kMaxWheels * (kMaxWheels - 1);

/// Denominators below this are treated as zero (parallel lines: no crossing).
constexpr double kTiny = 1.0e-300;

bool refuse(RwAllocationRefusal reason, RwAllocationResult& out) {
  out = RwAllocationResult{};
  out.refusal = reason;
  return false;
}

/// The 3x3 Gram matrix A A^T over the installed columns.
Eigen::Matrix3d gram(const RwAllocationConfig& config) {
  Eigen::Matrix3d g = Eigen::Matrix3d::Zero();
  for (int i = 0; i < config.wheel_count; ++i) {
    g += config.axes.col(i) * config.axes.col(i).transpose();
  }
  return g;
}

}  // namespace

bool RwAllocationConfig::isValid() const {
  if (wheel_count < 3 || wheel_count > kMaxWheels) {
    return false;
  }
  if (!std::isfinite(min_conditioning) || min_conditioning <= 0.0) {
    return false;
  }
  for (int i = 0; i < wheel_count; ++i) {
    if (!axes.col(i).allFinite() || !(axes.col(i).norm() > 0.0)) {
      return false;
    }
    if (!std::isfinite(max_torque_nm[i]) || !(max_torque_nm[i] > 0.0)) {
      return false;
    }
  }
  const Eigen::Matrix3d g = gram(*this);
  if (!g.allFinite()) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(g);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const double lambda_min = solver.eigenvalues()(0);
  const double lambda_max = solver.eigenvalues()(2);
  if (!(lambda_max > 0.0)) {
    return false;
  }
  return lambda_min / lambda_max >= min_conditioning;
}

RwAllocator::RwAllocator(const RwAllocationConfig& config)
    : config_(config), configured_(config.isValid()) {
  if (!configured_) {
    return;
  }
  const int n = config_.wheel_count;

  // A^+ = A^T (A A^T)^-1, factorised once: the Gram matrix passed the
  // conditioning gate in isValid(), so the inverse below is well posed.
  const Eigen::Matrix3d g_inv = gram(config_).inverse();
  for (int i = 0; i < n; ++i) {
    pinv_.row(i) = config_.axes.col(i).transpose() * g_inv;
  }
  if (!pinv_.allFinite()) {
    configured_ = false;
    return;
  }

  // Four wheels: the null space is exactly one-dimensional and its direction is
  // the generalized cross product of the three-row array —
  // n_i = (-1)^i det(A with column i deleted). Exact, allocation-free, and no
  // decomposition; A n = 0 is the Laplace expansion of a 4x4 with a repeated row.
  if (n == 4) {
    Eigen::Matrix<double, kMaxWheels, 1> null = Eigen::Matrix<double, kMaxWheels, 1>::Zero();
    for (int i = 0; i < 4; ++i) {
      Eigen::Matrix3d minor;
      int col = 0;
      for (int j = 0; j < 4; ++j) {
        if (j == i) {
          continue;
        }
        minor.col(col) = config_.axes.col(j);
        col++;
      }
      null[i] = ((i % 2) == 0 ? 1.0 : -1.0) * minor.determinant();
    }
    const double norm = null.norm();
    if (norm > 0.0 && null.allFinite()) {
      null_ = null / norm;
      has_null_ = true;
    }
  }
}

Eigen::Vector3d RwAllocator::achieved(const Eigen::Matrix<double, kMaxWheels, 1>& u) const {
  Eigen::Vector3d torque = Eigen::Vector3d::Zero();
  for (int i = 0; i < config_.wheel_count; ++i) {
    torque += config_.axes.col(i) * u[i];
  }
  return torque;
}

bool RwAllocator::allocate(const math::Vec3<math::frames::Body>& torque_cmd,
                           RwAllocationMethod method, RwAllocationResult& out) const {
  if (!configured_) {
    return refuse(RwAllocationRefusal::kUnconfigured, out);
  }
  const Eigen::Vector3d tau = torque_cmd.eigen();
  if (!tau.allFinite()) {
    return refuse(RwAllocationRefusal::kBadInput, out);
  }
  if (method == RwAllocationMethod::kMinMax && !supportsMinMax()) {
    // Null space of dimension > 1: the exact min-max is a linear program this
    // module does not solve (see the header). Refuse rather than return an L2
    // answer wearing an L-infinity label; the caller falls back explicitly.
    return refuse(RwAllocationRefusal::kMinMaxUnsupported, out);
  }

  const int n = config_.wheel_count;
  Eigen::Matrix<double, kMaxWheels, 1> u = pinv_ * tau;
  for (int i = n; i < kMaxWheels; ++i) {
    u[i] = 0.0;
  }

  if (method == RwAllocationMethod::kMinMax && has_null_) {
    // f(alpha) = max_i |w_i + alpha m_i| with w_i = u_i / L_i and m_i = n_i / L_i
    // — the wheel torques *in units of each wheel's own limit*, so what is
    // minimised is the box utilisation, which is what "saturates as late as
    // possible" means when the limits differ (with equal limits it is the plain
    // L-infinity). Convex and piecewise linear, so the minimum sits at a
    // breakpoint of the upper envelope: a crossing of two of the |affine|
    // pieces, or a piece's own zero. Evaluate every candidate and keep the
    // best. alpha = 0 is always in the list, so the answer is never worse than
    // the L2 one it started from.
    double w[kMaxWheels];
    double m[kMaxWheels];
    for (int i = 0; i < n; ++i) {
      w[i] = u[i] / config_.max_torque_nm[i];
      m[i] = null_[i] / config_.max_torque_nm[i];
    }
    double candidates[kMaxCandidates];
    int count = 0;
    candidates[count++] = 0.0;
    for (int i = 0; i < n; ++i) {
      if (std::abs(m[i]) > kTiny) {
        candidates[count++] = -w[i] / m[i];
      }
    }
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double diff = m[i] - m[j];
        if (std::abs(diff) > kTiny) {
          candidates[count++] = (w[j] - w[i]) / diff;
        }
        const double sum = m[i] + m[j];
        if (std::abs(sum) > kTiny) {
          candidates[count++] = -(w[i] + w[j]) / sum;
        }
      }
    }

    double best_alpha = 0.0;
    double best_max = 0.0;
    for (int i = 0; i < n; ++i) {
      best_max = std::max(best_max, std::abs(w[i]));
    }
    for (int c = 0; c < count; ++c) {
      const double alpha = candidates[c];
      if (!std::isfinite(alpha)) {
        continue;
      }
      double worst = 0.0;
      for (int i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(w[i] + alpha * m[i]));
      }
      // Strict improvement only: ties keep the earlier candidate, so the result
      // is a deterministic function of the inputs.
      if (worst < best_max) {
        best_max = worst;
        best_alpha = alpha;
      }
    }
    if (best_alpha != 0.0) {
      for (int i = 0; i < n; ++i) {
        u[i] += best_alpha * null_[i];
      }
    }
  }

  // Saturation: one scale for the whole vector, so the delivered torque keeps the
  // commanded direction (see the header).
  double scale = 1.0;
  for (int i = 0; i < n; ++i) {
    const double magnitude = std::abs(u[i]);
    if (magnitude > config_.max_torque_nm[i]) {
      scale = std::min(scale, config_.max_torque_nm[i] / magnitude);
    }
  }
  const bool saturated = scale < 1.0;
  if (saturated) {
    for (int i = 0; i < n; ++i) {
      u[i] *= scale;
    }
  }

  double max_wheel = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(u[i])) {
      return refuse(RwAllocationRefusal::kNonFiniteOutput, out);
    }
    max_wheel = std::max(max_wheel, std::abs(u[i]));
  }
  const Eigen::Vector3d delivered = achieved(u);
  if (!delivered.allFinite()) {
    return refuse(RwAllocationRefusal::kNonFiniteOutput, out);
  }

  out = RwAllocationResult{};
  for (int i = 0; i < n; ++i) {
    out.torque_nm[i] = u[i];
  }
  out.achieved_torque_nm = math::Vec3<math::frames::Body>(delivered);
  out.max_wheel_torque_nm = max_wheel;
  out.scale = scale;
  out.saturated = saturated;
  out.valid = true;
  out.refusal = RwAllocationRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc

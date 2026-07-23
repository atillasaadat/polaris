#include "actuators/rw_assembly.hpp"

#include <Eigen/SVD>

namespace polaris::sim::actuators {

RwAssembly RwAssembly::fromAxes(const std::vector<Eigen::Vector3d>& axes) {
  RwAssembly a;
  for (const Eigen::Vector3d& axis : axes) {
    if (axis.norm() < 1.0e-12) {
      return RwAssembly{};  // a wheel with no spin direction — reject the whole set
    }
  }
  a.w_.resize(3, static_cast<Eigen::Index>(axes.size()));
  for (std::size_t i = 0; i < axes.size(); ++i) {
    a.w_.col(static_cast<Eigen::Index>(i)) = axes[i].normalized();
  }
  return a;
}

bool RwAssembly::spansThreeAxes() const {
  if (w_.cols() < 3) {
    return false;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(w_);
  // Rank via the singular values, tolerant to the unit-axis scale (σ ≤ √N).
  const double tol = 1.0e-9 * static_cast<double>(w_.cols());
  return svd.singularValues().tail(1)(0) > tol;
}

}  // namespace polaris::sim::actuators

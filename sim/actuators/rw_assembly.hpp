#ifndef POLARIS_SIM_ACTUATORS_RW_ASSEMBLY_HPP
#define POLARIS_SIM_ACTUATORS_RW_ASSEMBLY_HPP

/// @file
/// @brief Reaction-wheel assembly geometry — the distribution matrix W (design doc §7, §8.5).
///
/// A wheel array is defined by one thing: where each rotor's spin axis points in
/// the body frame. Collect those N unit axes as the columns of a 3×N matrix **W**
/// and the whole array — pyramid, NASA 4-wheel skew, orthogonal triad, arbitrary N
/// — is captured without special cases. This replaces reasoning about individual
/// mounting DCMs: a wheel needs only its spin direction, not a full orientation.
///
/// W gives both directions the sim and the controller need:
///  - **Forward (plant):** the body torque a set of per-wheel reaction torques
///    produces is `τ_body = W · τ_wheels`, and the momentum the wheels store in the
///    body frame is `h_body = W · h_wheels`. This side is truth-model physics and
///    is what this class provides.
///  - **Inverse (control):** allocation `τ_wheels = W⁺ · τ_body` distributes a
///    commanded body torque across the wheels. That is the Phase-5 control layer
///    (§8.5, L-norm / L-∞) and lives with the FSW, not here — but it is the same W.
///
/// Sim-side: dynamic sizes and heap are fine.

#include <Eigen/Core>
#include <vector>

namespace polaris::sim::actuators {

/// The geometry of a reaction-wheel array: W (3×N), columns = unit spin axes in
/// the body frame, in wheel order.
class RwAssembly {
 public:
  RwAssembly() = default;

  /// Build from per-wheel spin axes (any nonzero vectors; each is normalised).
  /// Zero-length axes are rejected — a wheel with no spin direction has no
  /// meaning — returning an empty assembly is the caller's signal.
  static RwAssembly fromAxes(const std::vector<Eigen::Vector3d>& axes);

  /// Number of wheels (columns of W).
  [[nodiscard]] int size() const { return static_cast<int>(w_.cols()); }

  [[nodiscard]] bool empty() const { return w_.cols() == 0; }

  /// The 3×N distribution matrix.
  [[nodiscard]] const Eigen::Matrix<double, 3, Eigen::Dynamic>& matrix() const { return w_; }

  /// Net body torque from per-wheel reaction torques (about each spin axis):
  /// `τ_body = W · τ_wheels`. @p wheel_torques must be length N.
  [[nodiscard]] Eigen::Vector3d bodyTorque(const Eigen::VectorXd& wheel_torques) const {
    return w_ * wheel_torques;
  }

  /// Net stored wheel momentum in the body frame: `h_body = W · h_wheels`.
  [[nodiscard]] Eigen::Vector3d bodyMomentum(const Eigen::VectorXd& wheel_momenta) const {
    return w_ * wheel_momenta;
  }

  /// Whether the axes span all three body axes (rank W = 3) — the condition for
  /// the array to produce torque about any direction. A pyramid of ≥3 skewed
  /// wheels does; a set of collinear wheels does not, which is exactly the
  /// misconfiguration this flags.
  [[nodiscard]] bool spansThreeAxes() const;

 private:
  Eigen::Matrix<double, 3, Eigen::Dynamic> w_;
};

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_RW_ASSEMBLY_HPP

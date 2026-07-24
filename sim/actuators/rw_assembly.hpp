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
///
/// Implements the array-geometry half of REQ-SIM-003; the allocation inverse is
/// the §8.5 control layer. Reference: [markley2014] §7 (actuator arrays and
/// distribution matrices).

#include <Eigen/Core>
#include <vector>

namespace polaris::sim::actuators {

/// The geometry of a reaction-wheel array: W (3×N), columns = unit spin axes in
/// the body frame, in wheel order.
///
/// **Distribution.** With \f$W = [\hat{\mathbf{a}}_1\ \cdots\ \hat{\mathbf{a}}_N]
/// \in \mathbb{R}^{3\times N}\f$ (unit spin axes), the forward (truth) maps of the
/// array are linear in the per-wheel scalars:
/// \f[
///   \boldsymbol{\tau}_\mathrm{body} = W\,\boldsymbol{\tau}_\mathrm{wheels}, \qquad
///   \mathbf{h}_\mathrm{body} = W\,\mathbf{h}_\mathrm{wheels},
/// \f]
/// where \f$\tau_{\mathrm{wheel},i}\f$ is wheel \f$i\f$'s reaction torque about its
/// own spin axis and \f$h_{\mathrm{wheel},i}\f$ its stored momentum. (The control
/// inverse \f$\boldsymbol{\tau}_\mathrm{wheels} = W^{+}\,\boldsymbol{\tau}_\mathrm{body}\f$
/// is the §8.5 allocation layer, not part of this truth model.) The array can
/// produce torque about every body axis iff \f$\operatorname{rank} W = 3\f$
/// (Markley & Crassidis §7 [markley2014]).
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
  /// the array to produce torque about any direction. Rank is tested from the
  /// smallest singular value of \f$W\f$: \f$\sigma_{\min}(W) > \varepsilon\f$ with
  /// \f$\varepsilon\f$ scaled to the unit-axis magnitude. A pyramid of ≥3 skewed
  /// wheels satisfies it; a set of collinear wheels does not, which is exactly the
  /// misconfiguration this flags.
  [[nodiscard]] bool spansThreeAxes() const;

 private:
  Eigen::Matrix<double, 3, Eigen::Dynamic> w_;
};

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_RW_ASSEMBLY_HPP

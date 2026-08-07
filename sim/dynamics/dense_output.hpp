#ifndef POLARIS_SIM_DYNAMICS_DENSE_OUTPUT_HPP
#define POLARIS_SIM_DYNAMICS_DENSE_OUTPUT_HPP

/// @file
/// @brief Cubic-Hermite dense output over an integration's accepted-step nodes
/// (design doc §2.4, §5.1).
///
/// A sensor sample is an *observation*: it reads the plant, it does not change
/// it. So the integrator must not stop at one. `integrate` records its accepted
/// steps as `StepNode`s and this interpolant serves any instant inside them, so
/// a 2000 Hz IMU no longer fragments the RK8(9) grid to 0.5 ms — each fragment
/// otherwise pays the tableau's 16-stage minimum. Dynamics discontinuities (a
/// macro boundary where new commands apply, the magnetorquer duty-window edge)
/// remain exact integration stops; only observations are interpolated.
///
/// The interpolant is the standard cubic Hermite on \f$(\mathbf{y},
/// \dot{\mathbf{y}})\f$ at both ends of an accepted step, i.e. \f$O(h^4)\f$
/// local — coarser than the RK8(9) solution at the nodes, which is deliberate
/// and bounded: over a <=100 ms macro step at orbital rates the interpolation
/// residual sits far below any sensor's noise floor (pinned by
/// `tests/unit/sim_dynamics_test.cpp`). Where a state must be *exact* — the
/// macro-boundary truth published to the trace, the FSW and the stream tap — the
/// integration endpoint is used, never this.
///
/// References:
///  - Hairer, Nørsett & Wanner, *Solving Ordinary Differential Equations I*,
///    2nd ed., 1993, §II.6 (dense output / continuous extensions).

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <Eigen/Core>
#include <vector>

#include "dynamics/integrator.hpp"

namespace polaris::sim::dynamics {

/// State at @p t (same time origin as the nodes) by cubic Hermite between the
/// bracketing accepted-step nodes. @p nodes must be non-empty and ordered; @p t
/// outside their span is clamped to the end interval (extrapolation is never
/// asked for by the loop — the caller integrates to the event first).
///
/// On the interval \f$[t_0, t_1]\f$ with \f$h = t_1 - t_0\f$ and \f$\theta =
/// (t - t_0)/h\f$,
/// \f[
///   \mathbf{y}(\theta) = h_{00}\mathbf{y}_0 + h\,h_{10}\dot{\mathbf{y}}_0
///                      + h_{01}\mathbf{y}_1 + h\,h_{11}\dot{\mathbf{y}}_1,
/// \f]
/// with the usual basis \f$h_{00} = 2\theta^3 - 3\theta^2 + 1\f$, \f$h_{10} =
/// \theta^3 - 2\theta^2 + \theta\f$, \f$h_{01} = -2\theta^3 + 3\theta^2\f$,
/// \f$h_{11} = \theta^3 - \theta^2\f$ (Hairer et al. §II.6).
template <int N>
Eigen::Matrix<double, N, 1> hermiteAt(const std::vector<StepNode<N>>& nodes, double t) {
  assert(!nodes.empty());
  if (nodes.size() == 1) {
    return nodes.front().y;
  }
  // First node strictly past t, then step back to its left neighbour; clamped so
  // the first and last intervals cover the ends exactly.
  const auto it = std::upper_bound(nodes.begin(), nodes.end(), t,
                                   [](double x, const StepNode<N>& n) { return x < n.t; });
  const std::size_t hi =
      std::clamp<std::size_t>(static_cast<std::size_t>(it - nodes.begin()), 1, nodes.size() - 1);
  const StepNode<N>& a = nodes[hi - 1];
  const StepNode<N>& b = nodes[hi];

  const double h = b.t - a.t;
  if (!(h > 0.0)) {
    return b.y;
  }
  const double th = (t - a.t) / h;
  const double th2 = th * th;
  const double th3 = th2 * th;
  const double h00 = 2.0 * th3 - 3.0 * th2 + 1.0;
  const double h10 = th3 - 2.0 * th2 + th;
  const double h01 = -2.0 * th3 + 3.0 * th2;
  const double h11 = th3 - th2;
  return h00 * a.y + (h * h10) * a.ydot + h01 * b.y + (h * h11) * b.ydot;
}

}  // namespace polaris::sim::dynamics

#endif  // POLARIS_SIM_DYNAMICS_DENSE_OUTPUT_HPP

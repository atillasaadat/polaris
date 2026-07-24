#ifndef POLARIS_SIM_DYNAMICS_INTEGRATOR_HPP
#define POLARIS_SIM_DYNAMICS_INTEGRATOR_HPP

/// @file
/// @brief Adaptive Runge-Kutta 8(9) integrator (design doc §5.1, REQ-SIM-001).
///
/// A generic embedded RK integrator over a fixed-size state
/// `Eigen::Matrix<double, N, 1>`. The tableau is Verner's 16-stage RK8(9) — the
/// same coefficient set GMAT's `RungeKutta89` uses (Verner, *SIAM J. Numer.
/// Anal.* 15(4), 1978) — so the propagator matches the V&V reference tool once
/// the environment models land. The advancing solution is the higher-order
/// member; the embedded difference gives the per-step error estimate that drives
/// step-size control.
///
/// Not flight code (`sim/CLAUDE.md`): exceptions/heap are allowed here, but the
/// state is fixed-size and the hot loop stays allocation-free.
///
/// References:
///  - Verner, "Explicit Runge-Kutta Methods with Estimates of the Local
///    Truncation Error," SIAM J. Numer. Anal. 15(4):772-790, 1978. [verner1978]
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §4.2 (step control).
///    [montenbruck2000]

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>

namespace polaris::sim::dynamics {

/// Butcher tableau for an explicit embedded RK method (16 stages max).
struct RkTableau {
  static constexpr std::size_t kStages = 16;

  /// Order of the *error estimate* — sets the step-control exponent 1/(p+1).
  int error_order = 8;
  std::array<double, kStages> c{};                       ///< stage nodes
  std::array<std::array<double, kStages>, kStages> a{};  ///< stage coupling (strictly lower)
  std::array<double, kStages> b{};                       ///< advancing-solution weights
  std::array<double, kStages> e{};                       ///< error weights (b_high - b_low)
};

/// Verner RK8(9) — GMAT's `RungeKutta89` coefficients (Verner 1978).
const RkTableau& verner89();

/// Adaptive step-size control (Montenbruck & Gill §4.2, elementary controller).
struct StepControl {
  double abs_tol = 1.0e-12;  ///< absolute error floor per state component
  double rel_tol = 1.0e-12;  ///< relative error tolerance
  double min_step = 1.0e-6;  ///< [s] smallest step (forced-accept floor)
  double max_step = 60.0;    ///< [s] largest step
  double safety = 0.9;       ///< step-growth safety factor
  double min_scale = 0.2;    ///< max per-step shrink
  double max_scale = 5.0;    ///< max per-step growth
};

/// One RK step of size @p h from (@p t0, @p y): advancing solution -> @p y_next,
/// embedded error estimate -> @p err. @p f is `Vec f(double t, const Vec& y)`.
///
/// **Embedded step.** With the Butcher tableau \f$(c_i, a_{ij}, b_i, e_i)\f$ of
/// \f$s = 16\f$ stages, the stage derivatives are evaluated left-to-right on the
/// strictly lower-triangular coupling,
/// \f[
///   \mathbf{k}_i = f\!\left(t_0 + c_i h,\ \mathbf{y} + h \sum_{j<i} a_{ij}\,\mathbf{k}_j\right),
///   \qquad i = 0,\dots,s-1,
/// \f]
/// and combined into the advancing (higher-order) solution and the embedded error
/// estimate
/// \f[
///   \mathbf{y}_\mathrm{next} = \mathbf{y} + h \sum_{i} b_i\,\mathbf{k}_i, \qquad
///   \mathbf{err} = h \sum_{i} e_i\,\mathbf{k}_i,
/// \f]
/// where \f$e_i = b_i^{\text{high}} - b_i^{\text{low}}\f$ is the difference of the
/// order-9 and order-8 weight rows, so \f$\mathbf{err}\f$ estimates the local
/// truncation error of the order-8 member (Verner 1978 [verner1978]).
template <int N, class Deriv>
void rk89_step(const RkTableau& tab, Deriv& f, double t0, const Eigen::Matrix<double, N, 1>& y,
               double h, Eigen::Matrix<double, N, 1>& y_next, Eigen::Matrix<double, N, 1>& err) {
  using Vec = Eigen::Matrix<double, N, 1>;
  std::array<Vec, RkTableau::kStages> k;
  for (std::size_t i = 0; i < RkTableau::kStages; ++i) {
    Vec yi = y;
    for (std::size_t j = 0; j < i; ++j) {
      yi.noalias() += (h * tab.a[i][j]) * k[j];
    }
    k[i] = f(t0 + tab.c[i] * h, yi);
  }
  y_next = y;
  err = Vec::Zero();
  for (std::size_t i = 0; i < RkTableau::kStages; ++i) {
    y_next.noalias() += (h * tab.b[i]) * k[i];
    err.noalias() += (h * tab.e[i]) * k[i];
  }
}

/// No-op state projection (default for `integrate`).
struct NoProjection {
  template <class Vec>
  void operator()(Vec&) const {}
};

/// Integrate @p y from @p t0 to @p t1 with adaptive RK8(9). After each accepted
/// step, @p project is applied to the state (e.g. renormalize a quaternion so it
/// stays on the unit manifold). @p f is `Vec f(double t, const Vec& y)`.
///
/// **Step-size control** (Montenbruck & Gill §4.2 [montenbruck2000]). The embedded
/// error is scaled per component by a mixed absolute/relative tolerance and reduced
/// to an RMS norm over the \f$N\f$ states,
/// \f[
///   \mathrm{sc}_i = \mathrm{atol} + \mathrm{rtol}\,\max(|y_i|, |y_{\mathrm{next},i}|),
///   \qquad
///   E = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{\mathrm{err}_i}{\mathrm{sc}_i}\right)^2}.
/// \f]
/// A step is accepted when \f$E \le 1\f$ (or force-accepted at the \f$h =
/// h_{\min}\f$ floor so the loop cannot stall). The next step is scaled by
/// \f[
///   h \leftarrow \operatorname{clamp}\!\Big(
///     h\cdot\operatorname{clamp}\big(\rho\,E^{-1/(p+1)},\ s_{\min},\ s_{\max}\big),
///     \ h_{\min},\ h_{\max}\Big),
/// \f]
/// with safety factor \f$\rho\f$ (`safety`), error-estimate order \f$p\f$
/// (`error_order`, so the exponent is \f$-1/(p+1) = -1/9\f$), and per-step growth
/// bounded to \f$[s_{\min}, s_{\max}]\f$. When \f$E = 0\f$ the growth defaults to
/// \f$s_{\max}\f$. The final step is truncated to land exactly on \f$t_1\f$.
template <int N, class Deriv, class Project = NoProjection>
Eigen::Matrix<double, N, 1> integrate(const RkTableau& tab, Deriv& f, double t0, double t1,
                                      Eigen::Matrix<double, N, 1> y, const StepControl& ctl = {},
                                      Project project = {}) {
  using Vec = Eigen::Matrix<double, N, 1>;
  // Validate the controller invariants up front: std::clamp is UB if
  // min_step > max_step (§3.6, validate at boundaries), and the exponent/norm
  // math assumes positive tolerances and a sane scale window.
  assert(ctl.min_step > 0.0 && ctl.min_step <= ctl.max_step);
  assert(ctl.abs_tol > 0.0 && ctl.rel_tol >= 0.0);
  assert(ctl.min_scale > 0.0 && ctl.min_scale <= ctl.max_scale);
  double t = t0;
  double h = std::min(ctl.max_step, t1 - t0);
  if (h <= 0.0) {
    return y;
  }
  const double exponent = -1.0 / (static_cast<double>(tab.error_order) + 1.0);
  Vec y_next, err;
  while (t < t1) {
    if (t + h > t1) {
      h = t1 - t;
    }
    rk89_step<N>(tab, f, t, y, h, y_next, err);

    double sum_sq = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
      const double scale =
          ctl.abs_tol + ctl.rel_tol * std::max(std::abs(y[i]), std::abs(y_next[i]));
      const double ei = err[i] / scale;
      sum_sq += ei * ei;
    }
    const double err_norm = std::sqrt(sum_sq / static_cast<double>(N));

    // Accept on tolerance, or force-accept a floor-sized step so we never stall.
    // ponytail: forced accept keeps the loop finite; tighten min_step if a real
    // step is being clipped (the error norm at accept is reported nowhere yet).
    if (err_norm <= 1.0 || h <= ctl.min_step) {
      t += h;
      y = y_next;
      project(y);
    }

    double growth = (err_norm == 0.0) ? ctl.max_scale : ctl.safety * std::pow(err_norm, exponent);
    growth = std::clamp(growth, ctl.min_scale, ctl.max_scale);
    h = std::clamp(h * growth, ctl.min_step, ctl.max_step);
  }
  return y;
}

}  // namespace polaris::sim::dynamics

#endif  // POLARIS_SIM_DYNAMICS_INTEGRATOR_HPP

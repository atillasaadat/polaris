#ifndef POLARIS_MATH_FOV_OVERLAP_HPP
#define POLARIS_MATH_FOV_OVERLAP_HPP

/// @file
/// @brief Angular overlap of a circular field of view with a circular body
/// (design doc §6.1).
///
/// Lives in `lib/math` rather than in the sim's occlusion model because **both
/// sides of the boundary need the same number**: the sim's optical sensors use
/// it to grade Earth-in-field degradation, and the flight albedo correction
/// (`lib/gnc/albedo_correction.hpp`) uses it to reconstruct the same Earthshine
/// weighting from onboard geometry. Two implementations of this fraction would
/// mean a correction that removes an error the sensor never had — the same "one
/// geometry, one answer" rule §6.1 states for optical sensors, applied across
/// the sim/flight seam. `sim/sensors/occlusion.hpp` re-exports it, so the sim
/// side keeps its existing spelling.
///
/// **Approximation.** The overlap of the FOV cone with the body's cone is the
/// planar two-circle lens formula applied to the *angular* radii, the same
/// approximation §5.2 uses for the overlapping Sun and Earth disks. Exact in the
/// small-angle limit and degrading as the field grows: measured against a
/// brute-force spherical quadrature in
/// `tests/unit/sim_sensors_occlusion_test.cpp`, better than 0.008 in fraction
/// for half-fields up to 10°, 0.01 at 15°, 0.02 at 30° — even against the
/// Earth's ~68° disk.
///
/// **Flight-safe:** a pure function of three doubles. No heap, no exceptions, no
/// recursion, no loops, and every degenerate input returns a finite value rather
/// than a NaN.
///
/// References:
///  - Weisstein, *Circle-Circle Intersection*, MathWorld (the lens area).
///    [weisstein_circlecircle]

#include <algorithm>
#include <cmath>

namespace polaris::math {

/// Fraction of a circular field of view of half-angle @p half_fov_rad covered by
/// a disk of angular radius @p body_radius_rad whose centre is
/// @p separation_rad from the boresight.
///
/// With FOV radius \f$r_f\f$, body radius \f$r_b\f$, and separation
/// \f$d = |\text{separation\_rad}|\f$, the three degenerate cases are
/// \f[
///   f = \begin{cases}
///     0 & d \ge r_f + r_b \quad(\text{disjoint})
///     \\ 1 & d \le r_b - r_f \quad(\text{FOV inside body})
///     \\ (r_b/r_f)^2 & d \le r_f - r_b \quad(\text{body inside FOV})
///   \end{cases}
/// \f]
/// and the partial overlap is the planar two-circle lens area over the FOV disk:
/// \f[
///   f = \frac{r_f^2\big(\alpha - \tfrac12\sin 2\alpha\big) + r_b^2\big(\beta -
///   \tfrac12\sin 2\beta\big)}{\pi\, r_f^2},
/// \f]
/// \f[
///   \alpha = \arccos\frac{d^2 + r_f^2 - r_b^2}{2\,d\,r_f}, \qquad
///   \beta  = \arccos\frac{d^2 + r_b^2 - r_f^2}{2\,d\,r_b},
/// \f]
/// clamped to \f$[0,1]\f$.
///
/// Returns 0 for a non-positive field of view: a sensor with no field has no
/// fraction to report, and dividing by its area would put a NaN in a telemetry
/// channel — or, on the flight side, in an applied correction.
inline double fovCoveredFraction(double half_fov_rad, double separation_rad,
                                 double body_radius_rad) {
  if (!(half_fov_rad > 0.0) || !std::isfinite(separation_rad)) {
    return 0.0;
  }
  if (!(body_radius_rad > 0.0)) {
    return 0.0;
  }

  const double r_fov = half_fov_rad;
  const double r_body = body_radius_rad;
  const double d = std::abs(separation_rad);

  if (d >= r_fov + r_body) {
    return 0.0;  // disjoint
  }
  if (d <= r_body - r_fov) {
    return 1.0;  // FOV entirely inside the body
  }
  if (d <= r_fov - r_body) {
    // Body entirely inside the FOV: the ratio of the two disk areas.
    const double ratio = r_body / r_fov;
    return ratio * ratio;
  }

  // Partial overlap: the classical circular-lens area, on angular radii.
  const double d2 = d * d;
  const double rf2 = r_fov * r_fov;
  const double rb2 = r_body * r_body;
  const double cos_fov = std::clamp((d2 + rf2 - rb2) / (2.0 * d * r_fov), -1.0, 1.0);
  const double cos_body = std::clamp((d2 + rb2 - rf2) / (2.0 * d * r_body), -1.0, 1.0);
  const double alpha = std::acos(cos_fov);
  const double beta = std::acos(cos_body);
  // Each circular segment is (r² · angle) minus the triangle it contains.
  const double area =
      rf2 * (alpha - std::sin(2.0 * alpha) * 0.5) + rb2 * (beta - std::sin(2.0 * beta) * 0.5);
  return std::clamp(area / (M_PI * rf2), 0.0, 1.0);
}

}  // namespace polaris::math

#endif  // POLARIS_MATH_FOV_OVERLAP_HPP

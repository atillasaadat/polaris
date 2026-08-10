#ifndef POLARIS_GNC_GEOPOTENTIAL_HPP
#define POLARIS_GNC_GEOPOTENTIAL_HPP

/// @file
/// @brief Low-degree onboard geopotential — the unnormalized Cunningham V/W
/// recursion over a compiled-in EGM2008 truncation (design doc §8.3).
///
/// The force model the onboard orbit filter (@ref polaris::gnc::OrbitOd)
/// propagates on. It replaces that filter's original closed-form two-body + J2
/// term, which was the dominant error in its coast: measured against the
/// full-fidelity truth sim, the J2-only model diverged 3.59 m over the 300 s
/// coast horizon and 14.3 m over 699 s, an error budget led entirely by the
/// degree-3+ zonals and the tesserals this file adds.
///
/// ## Why a second implementation, when `sim/world/gravity_field` exists
///
/// Because the sim's is not flight code and cannot become it cheaply. It carries
/// `std::vector` coefficient and normalization tables sized at construction, a
/// `std::function` frame resolver, and a runtime-settable degree — all correct
/// for a truth model that runs a 200x200 field, all disallowed on the flight path
/// (§3.6: no heap, fixed-size, bounded). Templating it on a compile-time degree
/// to serve both would force the sim's 200x200 case into a 20301-entry
/// `std::array` and rewrite the model that GMAT cross-validation already
/// certified.
///
/// So this is a third, independent evaluation of the same field, and it is
/// treated the way the repo treats the other two: the sim already runs
/// `potential()` and `gradient()` as mutually-checking engines, and
/// `tests/unit/geopotential_test.cpp` pins *this* one against
/// `sim::world::SphericalHarmonicGravity` at matched degree and order. An
/// implementation that agrees with a GMAT-validated one to 1e-9 of its own
/// magnitude is not a duplicate free to drift; it is a cross-check.
///
/// ## Why unnormalized Cunningham, and where that stops working
///
/// The recursion is Montenbruck & Gill §3.2.4 [montenbruck2000], stated in terms
/// of the auxiliary functions
/// \f[
///   V_{nm} + \mathrm{i}\,W_{nm}
///     = \frac{R_e^{\,n}}{r^{\,n+1}}\,P_{nm}(\sin\phi)\;e^{\mathrm{i}m\lambda},
/// \f]
/// which are built by pure recursion from `V_00 = R_e/r`, `W_00 = 0` — no
/// Legendre evaluation, no trigonometry, and no division by `cos φ`, so the poles
/// are not special. The acceleration falls out of the *same* table one degree
/// higher, which is why V/W is the standard low-degree onboard formulation: it is
/// ~60 lines, has no lookup tables to precompute, and needs `(N+2)²` doubles of
/// scratch — 100 of them at degree 8, on the stack.
///
/// Its limit is real and is the reason the sim does not use it. The unnormalized
/// sectoral coefficients grow like `(2n-1)!!`, so the recursion overflows past
/// degree ~40 and loses precision well before that; the truth sim's 200x200 field
/// therefore uses the normalized Gottlieb recursion instead
/// (`sim/world/gravity_field.hpp` documents that trade). At degree 8 the largest
/// term is `15!!  ≈ 2e6` — nowhere near trouble — and
/// `tools/gravity/cxxtable.py` refuses to emit a table above degree 20 so this
/// evaluator can never be handed coefficients it cannot carry.
///
/// ## The frame is ECEF, at every order
///
/// The recursion is defined in the Earth-fixed frame and this one is used there
/// without exception. The filter's previous J2 term could evaluate in ECI about
/// the true pole, because a zonal field is axisymmetric and only its *axis*
/// matters; a tesseral field is not, and evaluating `m > 0` terms in ECI is wrong
/// in longitude by the whole Earth-rotation angle. Callers pass an ECEF position
/// and rotate the returned acceleration back to ECI themselves — @ref OrbitOd
/// does this per integration sub-step, since Earth turns 0.004 deg/s and holding
/// one rotation across a 60 s propagation would smear the tesserals by 0.25 deg
/// of longitude.
///
/// The same trap in the truth sim (zonal evaluation left in ECI, ~100 m per
/// revolution) was found by GMAT cross-validation and is documented in
/// `sim/world/gravity_field.hpp`. It is not repeated here.
///
/// **Frames, units, conventions.** Input position ECEF [m]; output acceleration
/// ECEF [m/s²]. `mu` [m³/s²] and `reference_radius` [m] must be the values the
/// coefficients were solved with — `egm2008::kGm` and `egm2008::kReferenceRadius`
/// travel with the table for exactly this reason. Flight path: fixed-size
/// arrays, no heap, no exceptions, no recursion, bounded loops.
///
/// References:
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.2.4 (the V/W recursion,
///    Eqs. 3.29-3.33, and the unnormalized coefficients it consumes).
///    [montenbruck2000]
///  - Cunningham, "On the computation of the spherical harmonic terms needed
///    during the numerical integration of the orbital motion of an artificial
///    satellite", Celestial Mechanics 2, 1970. [cunningham1970]
///  - Pavlis et al., "The development and evaluation of EGM2008", JGR 117, 2012
///    (the model the committed coefficients come from). [pavlis2012]

#include <Eigen/Core>

#include "gnc/egm2008_low_degree.hpp"

namespace polaris::gnc {

/// Largest degree/order this evaluator is compiled for — the size of the
/// committed coefficient table (@ref egm2008::kMaxDegree).
inline constexpr int kGeopotentialMaxDegree = egm2008::kMaxDegree;

/// Smallest geocentric radius the recursion is evaluated at [m]. Below it the
/// `R_e/r` powers blow up; a caller that has produced a position inside the Earth
/// has a fault to handle, and returning zero rather than an infinity keeps the
/// finiteness guard downstream meaningful. Well below @ref OrbitOd's own
/// plausibility band, so in practice this only ever fires on a diverged filter.
inline constexpr double kGeopotentialMinRadiusM = 1.0e5;

/// Geopotential acceleration at ECEF position @p r_ecef [m], in ECEF [m/s²].
///
/// Includes the point-mass term (`C_00 = 1`), so this is the *whole*
/// gravitational acceleration, not a perturbation to be added to `-mu r / r³`.
///
/// @param r_ecef            position in the Earth-fixed frame [m].
/// @param degree            harmonic degree to evaluate, clamped to
///                          [0, @ref kGeopotentialMaxDegree]. 0 is point-mass,
///                          2 with @p order 0 reproduces J2.
/// @param order             harmonic order, clamped to [0, @p degree].
/// @param mu                gravitational parameter the coefficients were solved
///                          with [m³/s²].
/// @param reference_radius  reference radius the coefficients are scaled to [m].
///
/// Returns the zero vector when the radius is below
/// @ref kGeopotentialMinRadiusM or an input is non-finite; every other path
/// returns a finite acceleration.
Eigen::Vector3d geopotentialAcceleration(const Eigen::Vector3d& r_ecef, int degree, int order,
                                         double mu = egm2008::kGm,
                                         double reference_radius = egm2008::kReferenceRadius);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_GEOPOTENTIAL_HPP

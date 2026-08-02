#ifndef POLARIS_SIM_WORLD_ECLIPSE_HPP
#define POLARIS_SIM_WORLD_ECLIPSE_HPP

/// @file
/// @brief Conical (umbra/penumbra) Earth-shadow model (REQ-SIM-002, §5.2).
///
/// `shadowFactor()` returns the fraction of the solar disk still visible from a
/// point in orbit — 1 in full sunlight, 0 in the umbra, and a continuously
/// varying value in between across the penumbra. That gradual limb crossing is
/// the whole point of a *conical* model: a cylindrical shadow snaps 1 -> 0 in one
/// step, which puts a discontinuity in the SRP acceleration that an adaptive
/// RK8(9) step controller then chases with ever-smaller steps.
///
/// Geometry (Montenbruck & Gill §3.4.2; Vallado §5.3): project the Sun and Earth
/// onto the sky as seen from the spacecraft and overlap the two apparent disks,
///
///   a = asin(R_sun / |r_sun - r_sat|)   apparent radius of the Sun,
///   b = asin(R_e   / |r_sat|)           apparent radius of the Earth,
///   c = angle between (-r_sat) and (r_sun - r_sat), their apparent separation,
///
/// which gives four regimes: disjoint disks (full sun), Sun's disk wholly inside
/// Earth's (umbra), Earth's disk wholly inside the Sun's (annular — geometrically
/// possible only far beyond the umbral cone tip, ~1.4e6 km, so never for an Earth
/// orbiter, but the branch keeps the function total), and partial overlap
/// (penumbra), where the visible fraction follows from the circular lens area.
///
/// The Earth is treated as a sphere of the WGS84 equatorial radius; oblateness
/// and atmospheric refraction near the limb are not modelled (both perturb only
/// the penumbra edge, well below the fidelity SRP itself warrants).
///
/// References:
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.4.2 (shadow function).
///    [montenbruck2000]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §5.3
///    (umbra/penumbra geometry). [vallado2013]

#include <Eigen/Core>

namespace polaris::sim::world {

/// Fraction of the solar disk visible from @p r_sat, in [0, 1].
///
/// **Model.** Project the Sun and Earth onto the sky as seen from the spacecraft
/// and overlap the two apparent disks (Montenbruck & Gill §3.4.2; Vallado §5.3)
/// [montenbruck2000; vallado2013]:
/// \f[
///   a = \arcsin\!\frac{R_\odot}{|\mathbf r_\odot - \mathbf r_{\mathrm{sat}}|},
///   \quad
///   b = \arcsin\!\frac{R_\oplus}{|\mathbf r_{\mathrm{sat}}|},
///   \quad
///   c = \angle\!\left(-\mathbf r_{\mathrm{sat}},\ \mathbf r_\odot - \mathbf
///   r_{\mathrm{sat}}\right),
/// \f]
/// the apparent solar radius, apparent Earth radius, and apparent separation. The
/// regimes are: full sun for \f$c \ge a+b\f$ (disjoint), umbra for \f$c+a\le b\f$
/// (solar disk fully occulted, \f$\nu=0\f$), and annular for \f$c+b\le a\f$
/// (\f$\nu = 1 - b^2/a^2\f$, unreachable for an Earth orbiter). Otherwise the disks
/// partially overlap and \f$\nu\f$ is one minus the circular-lens overlap area over
/// the solar-disk area (Montenbruck & Gill Eq. 3.87):
/// \f[
///   x = \frac{c^2 + a^2 - b^2}{2c},
///   \quad y = \sqrt{a^2 - x^2},
///   \quad A = a^2\arccos\frac{x}{a} + b^2\arccos\frac{c-x}{b} - c\,y,
///   \quad \nu = 1 - \frac{A}{\pi a^2}.
/// \f]
/// The two arccosines are evaluated as \f$\mathrm{atan2}(\sqrt{1-t^2},\,t)\f$ on
/// the clamped ratio \f$t\f$ (\c lib/README.md), which makes them total rather
/// than more accurate — the ratio is the conditioning limit. The apparent
/// separation \f$c\f$, being an angle between two vectors, genuinely does gain
/// from the \c atan2 form and uses it.
///
/// @param r_sat Geocentric ECI position of the spacecraft [m].
/// @param r_sun Geocentric ECI position of the Sun [m].
/// @return 1.0 in full sunlight, 0.0 in the umbra, fractional in the penumbra.
///         Degenerate inputs fail dark (0.0): a spacecraft at or below the
///         Earth's surface, a Sun coincident with the spacecraft, or a Sun at or
///         inside the Earth — the last of which signals a bad ephemeris result
///         rather than a geometry, and must not yield SRP.
double shadowFactor(const Eigen::Vector3d& r_sat, const Eigen::Vector3d& r_sun);

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_ECLIPSE_HPP

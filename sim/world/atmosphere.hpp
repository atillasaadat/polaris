#ifndef POLARIS_SIM_WORLD_ATMOSPHERE_HPP
#define POLARIS_SIM_WORLD_ATMOSPHERE_HPP

/// @file
/// @brief Neutral-density resolver contract + exponential model (REQ-SIM-002).
///
/// Drag needs one number from the atmosphere — the neutral mass density at the
/// spacecraft — and everything that makes atmosphere modelling hard (solar and
/// geomagnetic activity, diurnal bulge, seasonal-latitudinal variation) lives
/// behind that number. So the drag model takes an injected `DensityFn`, the same
/// decoupling `BodyPositionFn` gives the ephemeris consumers: drag is written,
/// built and tested without any atmosphere data on disk.
///
/// REQ-SIM-002 specifies **NRLMSIS 2.1** as the truth atmosphere, and it is
/// implemented in `nrlmsis.hpp` as exactly such a resolver — nothing in
/// `drag.hpp` knows which of the two it is holding. NRLMSIS is optional at build
/// time (research-only license, plus a Fortran toolchain; see
/// `THIRD_PARTY_NOTICES.md`), so the exponential model below is both the coarse
/// tier and the fallback when it is not built.
///
/// The exponential model is the one available *without* external data: the
/// piecewise-exponential fit to the US Standard Atmosphere 1976 / CIRA-72,
/// rho = rho_0 exp(-(h - h_0)/H) over 28 tabulated altitude bands (Vallado
/// Table 8-4). It carries no space-weather dependence at all, so it cannot
/// reproduce the order-of-magnitude solar-cycle swing in thermospheric density
/// above ~200 km. It is a static baseline for analytic decay checks and for
/// exercising the drag model — not a substitute for NRLMSIS in truth runs.
///
/// Altitude is **geodetic** (WGS84 ellipsoid), not geocentric: the 21 km
/// equator-to-pole difference in Earth radius is several density scale heights
/// in the thermosphere. It is computed straight from the ECI position, without
/// EOP or an ECI->ECEF reduction, because geodetic altitude depends only on the
/// distance from the spin axis and the height along it — both invariant under
/// the Earth rotation that separates ECI from ECEF. (Polar motion tilts the pole
/// by well under an arcsecond; that is irrelevant at these accuracies.)
///
/// Sim-side (`sim/CLAUDE.md`): heap / std::function / virtual dispatch are fine.
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
///    §8.6.2 and Table 8-4 (exponential atmosphere). [vallado2013]
///  - Emmert et al., "NRLMSIS 2.0/2.1", Earth and Space Science, 2021.
///    [emmert2021]
///  - Vallado, 4th ed., §3.2 / Algorithm 12 (ECEF to geodetic latitude and
///    height, the fixed-point iteration used here). [vallado2013]

#include <Eigen/Core>
#include <functional>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// Neutral mass density [kg/m^3] at an ECI position and TAI epoch. The epoch is
/// what a space-weather-driven model (NRLMSIS) needs to look up F10.7/Ap and the
/// diurnal bulge; a static model simply ignores it. Never negative; a model with
/// no valid answer (e.g. below the ground) returns 0, meaning "no drag here".
using DensityFn = std::function<double(const time::Tai&, const math::Vec3<math::frames::ECI>&)>;

/// Geodetic position on the WGS84 ellipsoid.
struct Geodetic {
  double latitude_rad{0.0};   ///< Geodetic latitude, [-pi/2, pi/2].
  double longitude_rad{0.0};  ///< Longitude east of the frame's prime meridian.
  double altitude_m{0.0};     ///< Height above the ellipsoid; negative inside it.
};

/// Convert a Cartesian position to geodetic coordinates on the WGS84 ellipsoid.
///
/// The **longitude is only meaningful for an ECEF input**; given an ECI vector it
/// comes out as right ascension, not longitude. Latitude and altitude are valid
/// for either, since both are invariant under rotation about the spin axis (see
/// the file header) — which is exactly why `geodeticAltitude()` below can take an
/// ECI position with no reduction, while NRLMSIS (`nrlmsis.hpp`) must first go to
/// ECEF to get a longitude it can turn into local solar time.
Geodetic geodetic(const Eigen::Vector3d& r);

/// Geodetic altitude above the WGS84 ellipsoid [m] for an ECI position.
/// Negative below the ellipsoid. See the file header on why no EOP is needed.
double geodeticAltitude(const math::Vec3<math::frames::ECI>& r_eci);

/// Piecewise-exponential US Standard Atmosphere 1976 (Vallado Table 8-4).
///
/// **Model.** Within the altitude band whose base is \f$(h_0,\rho_0,H)\f$
/// [vallado2013]:
/// \f[
///   \rho(h) = \rho_0\,\exp\!\left(-\frac{h - h_0}{H}\right),
/// \f]
/// with \f$h\f$ the geodetic altitude, \f$\rho_0\f$ the base density and \f$H\f$
/// the scale height of that band. This is the static baseline and NRLMSIS
/// fallback; a space-weather-driven truth run instead injects the NRLMSIS 2.1
/// resolver [emmert2021] through the same `DensityFn`, which this model knows
/// nothing about.
///
/// @param altitude_m Geodetic altitude [m].
/// @return Density [kg/m^3]. Zero below the ellipsoid; the topmost band (1000 km)
///         is extrapolated upward, where the density is negligible anyway.
double exponentialDensity(double altitude_m);

/// `DensityFn` adaptor for `exponentialDensity()` — the default atmosphere.
inline double exponentialAtmosphere(const time::Tai&, const math::Vec3<math::frames::ECI>& r) {
  return exponentialDensity(geodeticAltitude(r));
}

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_ATMOSPHERE_HPP

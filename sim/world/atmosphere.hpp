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
/// REQ-SIM-002 specifies **NRLMSIS 2.1** driven by space-weather files as the
/// truth atmosphere. That model is a separate deliverable (its source and the
/// F10.7/Ap history are committed verbatim per §3.7); when it lands it becomes
/// another `DensityFn` and nothing in `drag.hpp` changes.
///
/// `ExponentialAtmosphere` is the model available *without* external data: the
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

/// Geodetic altitude above the WGS84 ellipsoid [m] for an ECI position.
/// Negative below the ellipsoid. See the file header on why no EOP is needed.
double geodeticAltitude(const math::Vec3<math::frames::ECI>& r_eci);

/// Piecewise-exponential US Standard Atmosphere 1976 (Vallado Table 8-4).
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

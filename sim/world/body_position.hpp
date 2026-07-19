#ifndef POLARIS_SIM_WORLD_BODY_POSITION_HPP
#define POLARIS_SIM_WORLD_BODY_POSITION_HPP

/// @file
/// @brief The shared celestial-body position resolver contract (design doc §5.2).
///
/// Environment models that need where the Sun or Moon *is* — third-body gravity,
/// solar radiation pressure, eclipse, solid-Earth tides, optical-sensor occlusion
/// — all need the same thing and must not each invent their own hook. They take
/// an injected `BodyPositionFn`, which decouples every one of them from the
/// ephemeris source (mirroring `setEciToEcef` in `gravity_field.hpp`).
///
/// In the truth sim the resolver wraps the DE440-fed `ephemeris::EphemerisTable`;
/// onboard it would wrap the uploaded Chebyshev fits (REQ-CDH-002). One lambda
/// can therefore drive every consumer at once.
///
/// The argument is **TDB**, the ephemeris time argument — never TAI. Consumers
/// hold a TAI-stamped `TruthState` and convert with `toTdb(toTt(epoch))`.

#include <functional>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// Geocentric ECI position [m] of a celestial body at a TDB epoch. Returns
/// false — leaving @p out untouched — if the epoch is outside the ephemeris
/// coverage, which every consumer must treat as "this body contributes nothing"
/// rather than as a hard failure mid-integration.
using BodyPositionFn = std::function<bool(const time::Tdb&, math::Vec3<math::frames::ECI>& out)>;

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_BODY_POSITION_HPP

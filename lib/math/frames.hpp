#ifndef POLARIS_MATH_FRAMES_HPP
#define POLARIS_MATH_FRAMES_HPP

/// @file
/// @brief Compile-time coordinate-frame tags (design doc §3.1, Golden Rule 4).
///
/// Each frame is an empty tag type used to parameterize `Vec3<>` and `Quat<>`
/// at module/port/state boundaries, so that mixing frames across a boundary is
/// a compile error rather than a silent runtime bug. Tags carry only a name for
/// telemetry/diagnostics; they hold no data.

namespace polaris::math::frames {

/// Earth-centered inertial (ICRF / J2000 = attitude reference).
struct ECI {
  static constexpr const char* kName = "ECI";
};

/// Earth-centered Earth-fixed (ITRF).
struct ECEF {
  static constexpr const char* kName = "ECEF";
};

/// True Equator, Mean Equinox — the frame **SGP4 works in and only SGP4**
/// (Vallado 2006, §3.7 committed TLE fixtures).
///
/// It is deliberately its own tag rather than an alias for ECI. TEME differs
/// from J2000/GCRF by precession, nutation and the equation of the equinoxes —
/// order 100 km of position after a few decades of precession, and tens of
/// metres within a year — and the two are numerically close enough that mixing
/// them produces a *plausible* wrong answer rather than an obvious one. That is
/// the classic SGP4 integration bug, and it is exactly the class Golden Rule 4
/// exists to make a compile error. A TEME state reaches the rest of the system
/// only through an explicit conversion, never by assignment.
struct TEME {
  static constexpr const char* kName = "TEME";
};

/// Local-vertical / local-horizontal (nadir/orbit-relative pointing).
struct LVLH {
  static constexpr const char* kName = "LVLH";
};

/// Radial / In-track / Cross-track (Hill frame; RTN is a synonym).
struct RIC {
  static constexpr const char* kName = "RIC";
};

/// Spacecraft structural / body frame.
struct Body {
  static constexpr const char* kName = "Body";
};

}  // namespace polaris::math::frames

#endif  // POLARIS_MATH_FRAMES_HPP

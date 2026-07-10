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

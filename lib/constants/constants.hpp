#ifndef POLARIS_CONSTANTS_CONSTANTS_HPP
#define POLARIS_CONSTANTS_CONSTANTS_HPP

/// @file
/// @brief Shared physical-constants registry (design doc §3.1, Golden Rule 3).
///
/// Single source of truth for physical constants used by both the flight
/// software and the truth simulation, so the two can never silently disagree.
/// All values are SI. Every constant cites its source.
///
/// References:
///  - WGS84: NIMA TR8350.2, 3rd ed. (2000), "Department of Defense World
///    Geodetic System 1984". [bibkey: wgs84]
///  - Time scales: Vallado, *Fundamentals of Astrodynamics and Applications*,
///    4th ed., §3. [bibkey: vallado2013]

namespace polaris::constants {

/// @brief WGS84 reference-ellipsoid and Earth constants (NIMA TR8350.2).
namespace wgs84 {

/// Semi-major axis of the reference ellipsoid [m].
inline constexpr double kSemiMajorAxis = 6'378'137.0;

/// Flattening [-] (defining parameter): f = 1/298.257223563.
inline constexpr double kFlattening = 1.0 / 298.257'223'563;

/// First eccentricity squared [-]: e^2 = f(2 - f).
inline constexpr double kEccentricitySq = kFlattening * (2.0 - kFlattening);

/// Semi-minor axis [m]: b = a(1 - f).
inline constexpr double kSemiMinorAxis = kSemiMajorAxis * (1.0 - kFlattening);

/// Geocentric gravitational constant GM (mass of Earth incl. atmosphere)
/// [m^3/s^2] (WGS84 defining parameter).
inline constexpr double kGM = 3.986'004'418e14;

/// Earth nominal mean angular velocity [rad/s] (WGS84).
inline constexpr double kEarthRate = 7.292'115e-5;

}  // namespace wgs84

/// @brief Time-scale constants (Vallado §3).
namespace time {

/// Constant offset TAI - GPS [s]. GPS time runs 19 s behind TAI. Onboard
/// ingest applies TAI = GPS + kTaiMinusGps (design doc §3.2, Golden Rule 1).
inline constexpr double kTaiMinusGps = 19.0;

/// Constant offset TT - TAI [s]. Terrestrial Time leads TAI by 32.184 s.
inline constexpr double kTtMinusTai = 32.184;

/// Seconds per day [s].
inline constexpr double kSecondsPerDay = 86'400.0;

/// Julian Date of the J2000.0 epoch (2000-01-01T12:00:00 TT) [days] — the
/// reference epoch for the astronomical time arguments (Vallado §3.5).
inline constexpr double kJulianDateJ2000 = 2'451'545.0;

/// Julian Date of the uniform-scale epoch 1970-01-01T00:00:00 [days]. Bridges
/// the onboard nanosecond count (since 1970) to the JD-based astronomical
/// arguments; JD is a scale-agnostic calendar count (Vallado §3.5).
inline constexpr double kJulianDate1970 = 2'440'587.5;

/// Days per Julian century [days] — the unit of the astronomical time argument
/// `T = (JD - JD_J2000) / 36525`.
inline constexpr double kDaysPerJulianCentury = 36'525.0;

}  // namespace time

/// @brief TDB−TT periodic-term series (Astronomical Almanac / Vallado eq. 3-49).
///
/// Barycentric Dynamical Time differs from Terrestrial Time by a mainly annual
/// periodic term (no secular drift), driven by Earth's orbital eccentricity.
/// This low-precision two-harmonic series is accurate to ~30 µs — ample for
/// onboard Sun/Moon/planet ephemeris evaluation (design doc §3.2, §11.3).
namespace tdb {

/// Amplitude of the fundamental (annual) TDB−TT term [s].
inline constexpr double kAmplitude1 = 0.001'658;

/// Amplitude of the second harmonic [s].
inline constexpr double kAmplitude2 = 0.000'014;

/// Earth mean-anomaly constant term at J2000 [deg] (`g = kMeanAnomalyDeg + …`).
inline constexpr double kMeanAnomalyDeg = 357.53;

/// Earth mean-anomaly rate [deg/day].
inline constexpr double kMeanAnomalyRateDegPerDay = 0.985'600'28;

}  // namespace tdb

/// @brief Universal physical constants (CODATA / IAU defining values).
namespace physical {

/// Speed of light in vacuum [m/s] (exact, SI defining constant).
inline constexpr double kSpeedOfLight = 299'792'458.0;

/// Standard gravitational acceleration [m/s^2] (CGPM defining value).
inline constexpr double kStandardGravity = 9.806'65;

}  // namespace physical

}  // namespace polaris::constants

#endif  // POLARIS_CONSTANTS_CONSTANTS_HPP

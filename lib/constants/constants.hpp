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

/// @brief Earth zonal geopotential coefficients (EGM96/EGM2008 agree to the
/// digits carried here).
///
/// Unnormalized physical zonal harmonics \f$J_n\f$; the unnormalized zonal
/// \f$C_{n,0} = -J_n\f$, and the fully-normalized \f$\bar C_{n,0} = -J_n /
/// \sqrt{2n+1}\f$. They live in the shared registry rather than in either
/// consumer because **both** sides need them: the truth sim builds its zonal
/// coefficient table from these (`sim/world/gravity_field.hpp`), and the onboard
/// orbit filter's deliberately-coarse force model uses \f$J_2\f$ directly
/// (`lib/gnc/orbit_od.hpp`, design doc §8.3). Two copies of a geopotential
/// coefficient is exactly the "same number written twice" failure the truth-vs-
/// flight parameter checks exist to prevent, one level below the config.
///
/// **A harmonic coefficient is meaningless without the scale it was solved
/// with**, so the reference radius travels with it here rather than being
/// supplied by whichever `R_e` the caller had lying around. These values are
/// EGM96's (`J_2 = -\sqrt5\,\bar C_{20}` with
/// \f$\bar C_{20} = -4.84165371736\times10^{-4}\f$), and EGM96 — like EGM2008 —
/// is referenced to `kReferenceRadius = 6378136.3 m`, **not** to WGS84's
/// `kSemiMajorAxis = 6378137.0 m`. The 0.7 m difference looks negligible and is
/// not: the `J_2` term scales as `(R_e/r)²`, so substituting the WGS84 radius
/// rescales it by 2.2e-7, which is 3.3e-9 m/s² of acceleration and **~4.8 cm of
/// LEO position error per 1.5 revolutions**. That is a systematic that conserves
/// energy perfectly and therefore survives every self-consistency check; it was
/// found in the onboard propagator by the GMAT cross-validation
/// (`tests/golden/orbit_od_golden_test.cpp`), which is the same way the truth
/// sim's version of the mistake was found — see the note in
/// `sim/scenario/sim_runner.cpp` about using the model's own GM and radius.
///
/// [bibkey: vallado2013] (Table, zonal coefficients); [bibkey: pavlis2012]
/// (EGM2008, and the EGM96 lineage of the scale below).
namespace gravity {

inline constexpr double kJ2 = 1.082'626'683'5e-3;
inline constexpr double kJ3 = -2.532'656'485'3e-6;
inline constexpr double kJ4 = -1.619'621'591'4e-6;
inline constexpr double kJ5 = -2.272'721'801'1e-7;
inline constexpr double kJ6 = 5.406'815'991'0e-7;

/// Reference radius \f$R_e\f$ [m] the coefficients above are scaled to (EGM96 /
/// EGM2008). Pair it with them; see the namespace comment for what pairing them
/// with `wgs84::kSemiMajorAxis` instead costs.
inline constexpr double kReferenceRadius = 6'378'136.3;

/// Gravitational parameter GM [m³/s²] the coefficients above were solved with
/// (EGM96 / EGM2008; the value in the `.gfc` header's
/// `earth_gravity_constant`). It differs from `wgs84::kGM` by 7.5e-10 relative,
/// which is ~5 cm of LEO along-track drift per 1.5 revolutions.
///
/// **Use this one whenever a zonal term is switched on**, and `wgs84::kGM` only
/// for a pure point mass. A geopotential model is a single fit of GM *and* the
/// harmonics together, and mixing one model's GM with another's coefficients is
/// the same class of error as mixing reference radii. GMAT behaves this way too,
/// which is how the pairing was confirmed rather than assumed: the
/// `tests/golden/orbit_od_golden_test.cpp` ablation shows its `zonal_j2` case is
/// propagated on the potential file's GM and not on the `Earth.Mu = 398600.4418`
/// its own script sets, and swapping this constant in moves the onboard
/// propagator's agreement with that case from 6.9 cm to 1.0 cm — while the same
/// swap *degrades* the file-less `two_body` case from 3.5 mm to 6.1 cm. The
/// truth sim reaches the same place by reading the header (see
/// `sim/scenario/sim_runner.cpp`).
inline constexpr double kGM = 3.986'004'415e14;

}  // namespace gravity

/// @brief Third-body gravitational parameters and scale (JPL DE440 / IAU).
///
/// GM values consistent with the JPL DE440 ephemeris, for third-body point-mass
/// perturbations in the truth sim (design doc §5.2). [bibkey: park2021]
namespace bodies {

/// Heliocentric gravitational constant GM_sun [m^3/s^2] (DE440).
inline constexpr double kSunGM = 1.327'124'400'412'794'19e20;

/// Selenocentric gravitational constant GM_moon [m^3/s^2] (DE440,
/// NAIF gm_de440.tpc BODY301_GM). The previous value here (4.902800066...e12)
/// was DE430's, mislabelled as DE440 — 1e-8 relative, physically irrelevant,
/// but the provenance claim was wrong.
inline constexpr double kMoonGM = 4.902'800'118'457'55e12;

/// Planetary gravitational parameters [m^3/s^2] (DE440: Park et al. 2021,
/// machine-readable in NAIF gm_de440.tpc as km^3/s^2, converted here once).
/// Values are the **barycenter/system** GM (planet + satellites) — the right
/// mass for a point-mass perturbation acting from planetary distance, where the
/// system is unresolved.
inline constexpr double kMercuryGM = 2.203'186'855'14e13;
inline constexpr double kVenusGM = 3.248'585'92e14;
inline constexpr double kMarsGM = 4.282'837'581'575'61e13;    ///< Mars system
inline constexpr double kJupiterGM = 1.267'127'641'0e17;      ///< Jupiter system
inline constexpr double kSaturnGM = 3.794'058'484'18e16;      ///< Saturn system
inline constexpr double kUranusGM = 5.794'556'4e15;           ///< Uranus system
inline constexpr double kNeptuneGM = 6.836'527'100'580'4e15;  ///< Neptune system

/// Astronomical unit [m] (IAU 2012 defining value, exact).
inline constexpr double kAstronomicalUnit = 1.495'978'707e11;

/// Nominal solar radius [m] (IAU 2015 Resolution B3, `R_sun^N`). Sets the
/// angular size of the solar disk in the conical eclipse model.
inline constexpr double kSunRadius = 6.957e8;

/// Mean lunar radius [m] (IAU/IAG working group on cartographic coordinates).
/// Sets the Moon's angular size for optical-sensor keep-out (§6.1).
inline constexpr double kMoonRadius = 1.7374e6;

}  // namespace bodies

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

/// Modified Julian Date offset [days]: `MJD = JD - 2400000.5` (IERS/IAU
/// convention; MJD 0 = 1858-11-17T00:00). EOP tables are published on MJD.
inline constexpr double kJulianDateToMjd = 2'400'000.5;

/// Seconds from the uniform-scale epoch (1970-01-01T00:00:00) to J2000.0 [s].
/// The single definition of the 1970→J2000 span: the astronomical time arguments
/// count from J2000 while the master clock counts from 1970 (Vallado §3.5).
inline constexpr double kSecondsToJ2000 = (kJulianDateJ2000 - kJulianDate1970) * kSecondsPerDay;

}  // namespace time

/// @brief IAU/IERS reduction constants (SOFA/ERFA defining values).
///
/// Used by the IAU 2006/2000A ECI↔ECEF reduction (`lib/frames`, REQ-CONV-002).
/// IERS publishes polar motion in arcseconds; ERFA consumes radians.
///
/// References:
///  - IERS Conventions (2010), IERS TN 36, §5 (Earth orientation). [iers2010]
///  - IAU SOFA / ERFA `erfam.h`, `ERFA_DAS2R`. [erfa2021]
namespace iau {

/// Arcseconds to radians [rad/arcsec] = π / (180 × 3600). Matches SOFA/ERFA's
/// `ERFA_DAS2R` bit-for-bit.
inline constexpr double kArcsecToRad = 4.848'136'811'095'359'935'899'141e-6;

}  // namespace iau

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

/// @brief Solar radiation constants (IAU 2015 Res. B3; design doc §5.2).
namespace srp {

/// Nominal total solar irradiance at 1 AU [W/m^2] (IAU 2015 Resolution B3,
/// `S_sun^N`) — the modern TSI value, superseding the older 1367 W/m^2.
inline constexpr double kSolarConstant = 1361.0;

/// Solar radiation pressure at 1 AU [N/m^2] = S/c, for a fully absorbing
/// surface. Derived rather than tabulated so it can never drift from the
/// irradiance and speed of light above (Montenbruck & Gill §3.4).
inline constexpr double kPressureAt1Au = kSolarConstant / physical::kSpeedOfLight;

}  // namespace srp

}  // namespace polaris::constants

#endif  // POLARIS_CONSTANTS_CONSTANTS_HPP

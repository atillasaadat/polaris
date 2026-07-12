#ifndef POLARIS_TIME_TDB_HPP
#define POLARIS_TIME_TDB_HPP

/// @file
/// @brief TT↔TDB conversion — the periodic dynamical-time term (design doc §3.2,
/// §11.3).
///
/// Barycentric Dynamical Time (TDB) is the independent argument of the JPL
/// planetary ephemerides, so it is needed exactly where the onboard Chebyshev
/// ephemeris is evaluated (`ephemeris/chebyshev.hpp`). Unlike the TAI/GPS/TT
/// relationships, TDB−TT is **not** a constant offset: it is a mainly annual
/// periodic term (peak ≈ 1.66 ms) driven by Earth's orbital eccentricity, with
/// no secular drift. It cannot therefore live among the `constexpr` constant
/// offsets in `timescales.hpp`.
///
/// The implementation uses the standard low-precision two-harmonic series
/// (Astronomical Almanac; Vallado eq. 3-49), accurate to ~30 µs — far below the
/// onboard ephemeris error budget. No heap, no exceptions (flight-safe, §3.6).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
///    eq. 3-49 (TDB). [vallado2013]

#include "time/timescales.hpp"

namespace polaris::time {

/// TDB−TT for the given TT instant [s]. The mainly annual periodic term; the
/// result is bounded in magnitude by ~1.66 ms.
double tdbMinusTtSeconds(const Tt& tt);

/// Convert Terrestrial Time to Barycentric Dynamical Time: `TDB = TT + (TDB−TT)`
/// with the periodic term above. The offset is sub-millisecond, so the returned
/// instant differs from @p tt by at most a few million nanoseconds.
Tdb toTdb(const Tt& tt);

/// Convert Barycentric Dynamical Time back to Terrestrial Time: `TT = TDB − (TDB−TT)`.
///
/// The periodic term is evaluated at the TDB argument rather than TT. Because
/// the term varies slowly (period ≈ 1 year) and the TDB−TT offset is < 2 ms, this
/// single-shot inversion is self-consistent to well under 1 µs — no iteration is
/// needed for the onboard use.
Tt toTt(const Tdb& tdb);

}  // namespace polaris::time

#endif  // POLARIS_TIME_TDB_HPP

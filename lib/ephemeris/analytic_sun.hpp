#ifndef POLARIS_EPHEMERIS_ANALYTIC_SUN_HPP
#define POLARIS_EPHEMERIS_ANALYTIC_SUN_HPP

/// @file
/// @brief Low-precision analytic geocentric Sun ephemeris (design doc §8.1,
/// §11.3; REQ-CDH-002).
///
/// The **table-independent** Sun-direction source for the coarse (Safe-mode)
/// attitude floor. The onboard MEKF/coarse estimator normally reads the Sun
/// position from the uploaded DE440 Chebyshev fit (`ephemeris_table.hpp`); when
/// that table is missing or its coverage has lapsed, the coarse sun-pointing
/// state — the lowest safe-mode state (§10) — must still produce an inertial Sun
/// reference from arithmetic alone. This is that fallback: a mean-element
/// polynomial evaluated purely from the clock, with **no data dependency**.
///
/// Units & frames: input is a `time::Tdb` instant (the master TAI clock reduced
/// through `time/tdb.hpp`); output is a geocentric Sun position in **ECI metres**.
/// The model's natural frame is the **mean equator and equinox of date**; it is
/// returned tagged `ECI` (ICRF/J2000) under the deliberate approximation that
/// precession/nutation between date and J2000 are neglected. That frame slip —
/// dominated by general precession, ≈ 0.014°/yr ≈ 0.4° at the mid-2020s mission
/// epoch — is the largest error term and swamps the ~0.01° intrinsic model
/// accuracy; it is acceptable for coarse sun pointing (array-to-sun to a few
/// degrees) but callers needing better must use the precise table. No heap, no
/// exceptions, pure function (flight-safe, §3.6).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §5.1,
///    Algorithm 29 (Sun position, low precision). [vallado2013]

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::ephemeris {

/// Geocentric Sun position in ECI [m] at TDB instant @p tdb (Vallado Algorithm
/// 29). Pure analytic model; no table dependency. Frame is mean-of-date treated
/// as ECI (see file header for the precession approximation and its error).
///
/// \b Model. With Julian centuries \f$ T = d_{J2000}/36525 \f$ (days since
/// J2000 on the TDB scale), the mean longitude, mean anomaly, ecliptic
/// longitude, geocentric range, and obliquity are
/// \f[
///   \lambda_M = 280.460^\circ + 36000.771\,T, \qquad
///   M = 357.5291092^\circ + 35999.05034\,T,
/// \f]
/// \f[
///   \lambda_{ecl} = \lambda_M + 1.914666471^\circ \sin M
///                 + 0.019994643^\circ \sin 2M,
/// \f]
/// \f[
///   r = \bigl(1.000140612 - 0.016708617 \cos M
///           - 0.000139589 \cos 2M\bigr)\,\mathrm{AU}, \qquad
///   \varepsilon = 23.439291^\circ - 0.0130042\,T,
/// \f]
/// and the ECI position is
/// \f[
///   \mathbf{r}_\odot = r \begin{bmatrix}
///     \cos\lambda_{ecl}
///     \\ \cos\varepsilon \,\sin\lambda_{ecl}
///     \\ \sin\varepsilon \,\sin\lambda_{ecl}
///   \end{bmatrix}.
/// \f]
[[nodiscard]] math::Vec3<math::frames::ECI> sunPositionEci(const time::Tdb& tdb);

}  // namespace polaris::ephemeris

#endif  // POLARIS_EPHEMERIS_ANALYTIC_SUN_HPP

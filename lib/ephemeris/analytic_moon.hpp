#ifndef POLARIS_EPHEMERIS_ANALYTIC_MOON_HPP
#define POLARIS_EPHEMERIS_ANALYTIC_MOON_HPP

/// @file
/// @brief Low-precision analytic geocentric Moon ephemeris (design doc §8.1,
/// §11.3; REQ-CDH-002).
///
/// The **table-independent** lunar-position source that accompanies the analytic
/// Sun model (`analytic_sun.hpp`) in the coarse fallback. When the uploaded
/// DE440 Chebyshev fit is missing or its coverage has lapsed, the Moon position
/// is produced from a mean-element trigonometric series evaluated purely from the
/// clock, with **no data dependency**. The Moon is a much weaker constraint than
/// the Sun for coarse attitude, but the same fallback path serves third-body and
/// geometry consumers that would otherwise get no answer.
///
/// Units & frames: input is a `time::Tdb` instant (the master TAI clock reduced
/// through `time/tdb.hpp`); output is a geocentric Moon position in **ECI
/// metres**. As with the Sun model the natural frame is the **mean equator and
/// equinox of date**, returned tagged `ECI` (ICRF/J2000) under the deliberate
/// approximation that precession/nutation between date and J2000 are neglected.
/// The intrinsic series accuracy is ~0.3° in angle (Vallado); the neglected
/// precession adds ≈ 0.4° at the mid-2020s epoch, so a few tenths of a degree to
/// ~0.7° total against a precise ephemeris is expected. No heap, no exceptions,
/// pure function (flight-safe, §3.6).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §5.3.2
///    (Moon position, low precision). [vallado2013]

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::ephemeris {

/// Geocentric Moon position in ECI [m] at TDB instant @p tdb (Vallado §5.3.2
/// low-precision lunar series). Pure analytic model; no table dependency. Frame
/// is mean-of-date treated as ECI (see file header for the precession
/// approximation and its error).
///
/// \b Model. With Julian centuries \f$ T = d_{J2000}/36525 \f$ (days since
/// J2000 on the TDB scale), the ecliptic longitude, ecliptic latitude, and
/// horizontal parallax [deg] are
/// \f[
///   \lambda_{ecl} = 218.32^\circ + 481267.8813\,T
///     + 6.29\sin(134.9^\circ + 477198.85\,T)
///     - 1.27\sin(259.2^\circ - 413335.38\,T)
///     + 0.66\sin(235.7^\circ + 890534.23\,T)
/// \f]
/// \f[
///     + 0.21\sin(269.9^\circ + 954397.70\,T)
///     - 0.19\sin(357.5^\circ + 35999.05\,T)
///     - 0.11\sin(186.6^\circ + 966404.05\,T),
/// \f]
/// \f[
///   \phi_{ecl} = 5.13\sin(93.3^\circ + 483202.03\,T)
///     + 0.28\sin(228.2^\circ + 960400.87\,T)
///     - 0.28\sin(318.3^\circ + 6003.18\,T)
///     - 0.17\sin(217.6^\circ - 407332.20\,T),
/// \f]
/// \f[
///   \mathfrak{P} = 0.9508^\circ
///     + 0.0518\cos(134.9^\circ + 477198.85\,T)
///     + 0.0095\cos(259.2^\circ - 413335.38\,T)
///     + 0.0078\cos(235.7^\circ + 890534.23\,T)
///     + 0.0028\cos(269.9^\circ + 954397.70\,T).
/// \f]
/// With the obliquity \f$ \varepsilon = 23.439291^\circ - 0.0130042\,T \f$ and
/// geocentric range \f$ r = R_\oplus / \sin\mathfrak{P} \f$, the ECI position is
/// \f[
///   \mathbf{r}_\leftmoon = r \begin{bmatrix}
///     \cos\phi\cos\lambda
///     \\ \cos\varepsilon\cos\phi\sin\lambda - \sin\varepsilon\sin\phi
///     \\ \sin\varepsilon\cos\phi\sin\lambda + \cos\varepsilon\sin\phi
///   \end{bmatrix}.
/// \f]
[[nodiscard]] math::Vec3<math::frames::ECI> moonPositionEci(const time::Tdb& tdb);

}  // namespace polaris::ephemeris

#endif  // POLARIS_EPHEMERIS_ANALYTIC_MOON_HPP

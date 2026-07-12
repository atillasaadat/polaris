#ifndef POLARIS_EPHEMERIS_CHEBYSHEV_HPP
#define POLARIS_EPHEMERIS_CHEBYSHEV_HPP

/// @file
/// @brief Onboard Chebyshev ephemeris evaluation (design doc §11.3, REQ-CDH-002).
///
/// Sun/Moon/planet positions are carried onboard as **Chebyshev polynomial fits**
/// generated on the ground from SPICE (DE440) and uploaded as coefficient sets
/// per time interval (design doc §11.3, §18.6). SPICE itself is never linked into
/// flight — this header is the flight-side evaluator that turns an uploaded
/// coefficient segment into a position (and, optionally, velocity) at a query
/// epoch, exactly as CSPICE's SPK Type 2/3 records are read on the ground.
///
/// The query epoch is **TDB** — the argument of the planetary ephemerides — so
/// callers convert their onboard TAI/TT clock through `time/tdb.hpp` first.
///
/// Frames & units: positions are returned in **ECI (ICRF/J2000)** metres and
/// velocities in metres/second, per the SI convention (§3.1). The `ECI` tag
/// asserts both an **Earth-centered origin** and ICRF/J2000 orientation, so the
/// uploaded coefficients **must be fit geocentric** (Earth-centered): DE440's
/// native records are barycentric / center-relative, and the ground fitter is
/// responsible for translating to Earth-centered before upload (design doc
/// §11.3). Feeding barycentric coefficients here silently mislabels the origin.
///
/// Flight discipline (§3.6): fixed-capacity coefficient storage (no heap), no
/// exceptions, all inputs and outputs finiteness/range-checked. A query outside
/// the segment's coverage, a non-finite input, or a non-positive interval radius
/// returns `false` and leaves the output untouched.
///
/// References:
///  - Newhall, "Numerical representation of planetary ephemerides",
///    *Celestial Mechanics* 45 (1989) — Chebyshev fitting of DE ephemerides.
///    [newhall1989]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §8
///    (interpolation). [vallado2013]

#include <cstdint>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::ephemeris {

/// Highest Chebyshev coefficient index storable per component (16 coefficients).
/// DE440's densest body (the Moon) uses 13 coefficients per component, so this
/// fixed capacity covers every onboard body with margin (no heap, §3.6).
inline constexpr int kMaxChebyshevDegree = 15;

/// One uploaded Chebyshev coefficient set covering a single time interval for a
/// single body, mirroring an SPK Type 2 record. The interval is centred on
/// `mid_ns` (TDB nanoseconds since the uniform-scale epoch 1970-01-01T00:00:00;
/// see `time::Instant`) with half-width `radius_seconds`, and normalized time is
/// `τ = (t − mid) / radius ∈ [−1, 1]`. The midpoint is carried as **int64 ns**,
/// not double seconds, so the query offset `t − mid` is differenced at full
/// nanosecond precision before the (small, bounded) result is cast to double —
/// preserving the int64-ns clock discipline (REQ-CONV-005) instead of losing
/// ~0.2 µs to catastrophic cancellation on the absolute epoch past 2038.
struct ChebyshevSegment {
  std::int64_t mid_ns{0};      ///< interval midpoint, TDB nanoseconds since epoch
  double radius_seconds{0.0};  ///< interval half-width [s] (> 0)
  int degree{-1};              ///< highest coefficient index; `degree + 1` coefficients used

  double cx[kMaxChebyshevDegree + 1]{};  ///< X coefficients (Earth-centered ECI metres)
  double cy[kMaxChebyshevDegree + 1]{};  ///< Y coefficients (Earth-centered ECI metres)
  double cz[kMaxChebyshevDegree + 1]{};  ///< Z coefficients (Earth-centered ECI metres)

  /// True if this segment covers the TDB instant @p t (within its interval).
  bool covers(const time::Tdb& t) const;
};

/// Evaluate the segment position at @p t. Returns false and leaves @p pos_out
/// untouched if @p t is outside the segment, the segment is malformed
/// (`degree < 0`, `radius_seconds <= 0`), or any value is non-finite (§3.6).
[[nodiscard]] bool evaluate(const ChebyshevSegment& seg, const time::Tdb& t,
                            math::Vec3<math::frames::ECI>& pos_out);

/// Evaluate position and velocity together (analytic derivative of the fit).
/// Same failure contract as the position-only overload; on failure neither
/// output is modified.
[[nodiscard]] bool evaluate(const ChebyshevSegment& seg, const time::Tdb& t,
                            math::Vec3<math::frames::ECI>& pos_out,
                            math::Vec3<math::frames::ECI>& vel_out);

}  // namespace polaris::ephemeris

#endif  // POLARIS_EPHEMERIS_CHEBYSHEV_HPP

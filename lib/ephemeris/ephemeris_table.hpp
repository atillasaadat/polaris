#ifndef POLARIS_EPHEMERIS_EPHEMERIS_TABLE_HPP
#define POLARIS_EPHEMERIS_EPHEMERIS_TABLE_HPP

/// @file
/// @brief Fixed-capacity onboard ephemeris table (design doc §11.3, REQ-CDH-002).
///
/// One body's uploaded Chebyshev coefficient sets — a contiguous run of
/// `ChebyshevSegment`s covering successive time intervals — stored in a
/// fixed-capacity container (no heap, §3.6). A position query selects the segment
/// whose interval contains the epoch and evaluates it. This is the flight-side
/// realization of "Sun/Moon/planet positions onboard from uploaded Chebyshev
/// coefficient sets per interval" (design doc §11.3, §18.6).
///
/// The onboard EOP / leap-second tables that accompany the ephemeris upload live
/// in `lib/time`; this container holds only the position fits.

#include <cstddef>

#include "ephemeris/chebyshev.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::ephemeris {

/// A body's ephemeris as up to @p Capacity consecutive Chebyshev segments.
/// @tparam Capacity maximum number of intervals (compile-time; no heap).
template <std::size_t Capacity>
class EphemerisTable {
 public:
  static_assert(Capacity > 0, "an ephemeris table needs room for at least one segment");

  /// Append a segment. Returns false if the table is full or the segment is
  /// malformed (non-positive radius, out-of-range degree) — the table only ever
  /// holds evaluable segments, so a query need not re-validate structure.
  bool addSegment(const ChebyshevSegment& seg) {
    if (count_ >= Capacity) {
      return false;
    }
    if (!(seg.radius_seconds > 0.0) || seg.degree < 0 || seg.degree > kMaxChebyshevDegree) {
      return false;
    }
    segments_[count_] = seg;
    ++count_;
    return true;
  }

  /// Number of loaded segments.
  std::size_t size() const { return count_; }

  /// True when the table holds no segments at all. A non-empty table can
  /// still refuse a query outside its span; check positionAt's return.
  bool empty() const { return count_ == 0; }

  /// Position of the body at @p t in ECI [m]. Returns false and leaves @p pos_out
  /// untouched if no loaded segment covers @p t (§3.6 — no throw, no extrapolation).
  [[nodiscard]] bool position(const time::Tdb& t, math::Vec3<math::frames::ECI>& pos_out) const {
    const ChebyshevSegment* seg = find(t);
    return seg != nullptr && evaluate(*seg, t, pos_out);
  }

  /// Position and velocity of the body at @p t (ECI [m], [m/s]). Same coverage
  /// contract as `position`.
  [[nodiscard]] bool state(const time::Tdb& t, math::Vec3<math::frames::ECI>& pos_out,
                           math::Vec3<math::frames::ECI>& vel_out) const {
    const ChebyshevSegment* seg = find(t);
    return seg != nullptr && evaluate(*seg, t, pos_out, vel_out);
  }

 private:
  /// First loaded segment covering @p t, or nullptr. Linear scan — segment counts
  /// per body are small (a handful of active intervals), so no index is warranted.
  const ChebyshevSegment* find(const time::Tdb& t) const {
    for (std::size_t i = 0; i < count_; ++i) {
      if (segments_[i].covers(t)) {
        return &segments_[i];
      }
    }
    return nullptr;
  }

  ChebyshevSegment segments_[Capacity]{};
  std::size_t count_{0};
};

}  // namespace polaris::ephemeris

#endif  // POLARIS_EPHEMERIS_EPHEMERIS_TABLE_HPP

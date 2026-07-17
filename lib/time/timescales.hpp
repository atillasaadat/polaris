#ifndef POLARIS_TIME_TIMESCALES_HPP
#define POLARIS_TIME_TIMESCALES_HPP

/// @file
/// @brief Uniform timescales and the onboard master clock (design doc §3.2).
///
/// The onboard master timescale is **TAI** — continuous, monotonic, leap-second
/// free — represented as a monotonic signed int64 nanosecond count since the
/// epoch **1970-01-01T00:00:00 TAI** (REQ-SYS-001). GPS and TT are the other
/// uniform scales; each differs from TAI by a documented constant offset. UTC is
/// *not* here — it is leap-second-bearing and non-uniform, so it lives as a
/// broken-down calendar type in `utc.hpp` and is derived for ground use only.
///
/// Scales are **strongly typed** (`Tai`, `Gps`, `Tt`), mirroring the frame
/// tagging of §3.1: you cannot add a GPS reading to a TAI instant, and the only
/// way between scales is the documented conversion functions below — no silent
/// "it's just an int64" mixups (the single largest class of time bugs, §3.2).
///
/// All uniform-scale instants count nanoseconds since 1970-01-01T00:00:00 *in
/// their own scale*, so the same physical event read on two scales differs by
/// exactly that scale-pair's constant offset. Conversions are int64-exact:
/// `TAI - GPS = 19 s` and `TT - TAI = 32.184 s` are both whole nanosecond counts.
///
/// No heap, no exceptions, `constexpr` where possible (flight-safe, §3.6).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3
///    (TAI/GPS/TT/UTC relationships). [vallado2013]

#include <cstdint>

#include "constants/constants.hpp"
#include "time/duration.hpp"

namespace polaris::time {

/// Scale tags. Empty types used only as compile-time labels on `Instant`.
namespace scale {
struct Tai {
  static constexpr const char* kName = "TAI";
};

struct Gps {
  static constexpr const char* kName = "GPS";
};

struct Tt {
  static constexpr const char* kName = "TT";
};

/// Barycentric Dynamical Time — the independent argument of the JPL planetary
/// ephemerides. It differs from TT by a *periodic* (not constant) term, so the
/// TT↔TDB conversion is a function (see `time/tdb.hpp`), not one of the constant
/// offsets below.
struct Tdb {
  static constexpr const char* kName = "TDB";
};

/// Universal Time 1 — the Earth-rotation angle expressed as a time. Like UTC it
/// is **not uniform**: it tracks the actual, irregular rotation of the Earth, so
/// no constant offset reaches it from TAI and none of the conversions below
/// apply. UT1 is only obtainable from the measured/predicted `UT1 - TAI` in the
/// uploaded IERS EOP table (`frames/eop.hpp`) — hence `ut1FromTai()` there, not
/// here. It is a full `Instant` (unlike UTC's calendar-only form) because the
/// ECI↔ECEF reduction needs it as a high-precision JD argument, and it carries
/// no leap seconds to break the nanosecond count.
struct Ut1 {
  static constexpr const char* kName = "UT1";
};
}  // namespace scale

/// Two-part high-precision time: whole seconds plus a fraction in `[0, 1)`
/// (design doc §3.2 / REQ-CONV-005). Preserves nanosecond resolution across
/// long propagation/ephemeris arcs where a single double would lose bits.
struct TwoPart {
  std::int64_t seconds{0};  ///< whole seconds since the scale epoch (floored)
  double fraction{0.0};     ///< sub-second remainder, always in `[0, 1)`
};

/// A point in time on uniform scale @c Scale: int64 nanoseconds since
/// 1970-01-01T00:00:00 on that scale. Monotonic; the ordering of instants is the
/// ordering of the underlying count (REQ-SYS-001).
/// @tparam Scale one of `scale::Tai`, `scale::Gps`, `scale::Tt`.
template <class Scale>
class Instant {
 public:
  /// The epoch instant (count 0): 1970-01-01T00:00:00 on @c Scale.
  constexpr Instant() = default;

  /// Named constructor from an exact nanosecond count since the scale epoch.
  static constexpr Instant fromNanosecondsSinceEpoch(std::int64_t ns) { return Instant(ns); }

  /// Exact nanoseconds since the scale epoch.
  constexpr std::int64_t nanosecondsSinceEpoch() const { return ns_; }

  /// High-precision two-part decomposition (REQ-CONV-005). The fraction is
  /// always in `[0, 1)`; for negative instants the seconds floor accordingly
  /// (e.g. -0.5 s -> {seconds:-1, fraction:0.5}).
  constexpr TwoPart twoPart() const {
    std::int64_t s = ns_ / Duration::kNsPerSecond;
    std::int64_t rem = ns_ % Duration::kNsPerSecond;
    if (rem < 0) {  // floor toward negative infinity so fraction stays in [0,1)
      s -= 1;
      rem += Duration::kNsPerSecond;
    }
    return TwoPart{s, static_cast<double>(rem) / static_cast<double>(Duration::kNsPerSecond)};
  }

  /// @name Point ± interval arithmetic
  /// @{
  constexpr Instant operator+(const Duration& d) const { return Instant(ns_ + d.nanoseconds()); }

  constexpr Instant operator-(const Duration& d) const { return Instant(ns_ - d.nanoseconds()); }

  /// Interval between two instants on the same scale.
  constexpr Duration operator-(const Instant& o) const {
    return Duration::fromNanoseconds(ns_ - o.ns_);
  }

  /// @}

  /// @name Monotonic comparisons
  /// @{
  constexpr bool operator==(const Instant& o) const { return ns_ == o.ns_; }

  constexpr bool operator!=(const Instant& o) const { return ns_ != o.ns_; }

  constexpr bool operator<(const Instant& o) const { return ns_ < o.ns_; }

  constexpr bool operator<=(const Instant& o) const { return ns_ <= o.ns_; }

  constexpr bool operator>(const Instant& o) const { return ns_ > o.ns_; }

  constexpr bool operator>=(const Instant& o) const { return ns_ >= o.ns_; }

  /// @}

  static constexpr const char* scale_name() { return Scale::kName; }

 private:
  explicit constexpr Instant(std::int64_t ns) : ns_(ns) {}

  std::int64_t ns_{0};
};

/// TAI — the onboard master clock (design doc §3.2, REQ-SYS-001).
using Tai = Instant<scale::Tai>;
/// GPS time — the GNSS receiver interface scale (design doc §6.2).
using Gps = Instant<scale::Gps>;
/// Terrestrial Time — the ephemeris scale (design doc §3.2).
using Tt = Instant<scale::Tt>;
/// Barycentric Dynamical Time — the argument of the planetary ephemerides
/// (design doc §11.3). Reached from TT via the periodic term in `time/tdb.hpp`.
using Tdb = Instant<scale::Tdb>;
/// Universal Time 1 — the Earth-rotation argument of the ECI↔ECEF reduction
/// (design doc §3.1, REQ-CONV-002). Reached only via the EOP table's `UT1 - TAI`.
using Ut1 = Instant<scale::Ut1>;

/// @name Astronomical time arguments
///
/// Bridges the onboard nanosecond count (since 1970, on the instant's own scale)
/// to the JD/J2000 arguments the astronomical routines want. JD is a
/// scale-agnostic calendar count, so shifting by the fixed 1970→J2000 (or
/// 1970→JD) span is all it takes; the *scale* is carried by which `Instant` you
/// hand in (Vallado §3.5).
/// @{

/// Two-part Julian Date: `JD = day + fraction`, with @c day an exact integral
/// day number and @c fraction in `[0, 1)`.
///
/// The split is what preserves precision. A single double holding JD ≈ 2.45e6
/// resolves only ~50 µs, but an integral @c day is exact and @c fraction spends
/// its whole mantissa on the sub-day part — well under a nanosecond. This is the
/// two-part form SOFA/ERFA routines take as `(d1, d2)` (REQ-CONV-005).
struct JulianDate {
  double day{0.0};       ///< integral JD day number (exactly representable)
  double fraction{0.0};  ///< sub-day remainder, always in `[0, 1)`
};

/// Two-part Julian Date of @p t **on its own scale** (e.g. a `Tt` in, JD(TT) out).
template <class Scale>
inline JulianDate julianDate(const Instant<Scale>& t) {
  const TwoPart tp = t.twoPart();  // seconds since 1970 (floored) + fraction [0,1)
  constexpr std::int64_t kSecPerDay = static_cast<std::int64_t>(constants::time::kSecondsPerDay);
  // Floor toward -inf so `sec_of_day` stays non-negative for pre-1970 instants.
  std::int64_t days = tp.seconds / kSecPerDay;
  std::int64_t sec_of_day = tp.seconds % kSecPerDay;
  if (sec_of_day < 0) {
    days -= 1;
    sec_of_day += kSecPerDay;
  }
  // JD(1970-01-01T00:00) = 2440587.5, i.e. .5 into day 2440587 — so start the
  // fraction at 0.5 and carry into the day number when it reaches 1.
  double day = (constants::time::kJulianDate1970 - 0.5) + static_cast<double>(days);
  double fraction =
      0.5 + (static_cast<double>(sec_of_day) + tp.fraction) / constants::time::kSecondsPerDay;
  if (fraction >= 1.0) {
    day += 1.0;
    fraction -= 1.0;
  }
  return JulianDate{day, fraction};
}

/// Days elapsed since the J2000.0 epoch for @p t, on @p t's own scale. The
/// low-precision day argument of the periodic series (`time/tdb.hpp`); use
/// `julianDate()` where full nanosecond precision matters.
template <class Scale>
inline double daysSinceJ2000(const Instant<Scale>& t) {
  const double days_since_1970 =
      static_cast<double>(t.nanosecondsSinceEpoch()) /
      (static_cast<double>(Duration::kNsPerSecond) * constants::time::kSecondsPerDay);
  return days_since_1970 + (constants::time::kJulianDate1970 - constants::time::kJulianDateJ2000);
}

/// @}

/// @name Constant-offset scale conversions
///
/// Derived int64-ns offsets, cross-checked against the shared physical-constants
/// registry so the two representations cannot drift.
/// @{

/// `TAI - GPS = 19 s` as an exact nanosecond count (design doc §3.2 Golden Rule 1).
inline constexpr std::int64_t kTaiMinusGpsNs = 19'000'000'000;
static_assert(kTaiMinusGpsNs ==
                  static_cast<std::int64_t>(constants::time::kTaiMinusGps) * Duration::kNsPerSecond,
              "GPS->TAI offset must match the constants registry");

/// `TT - TAI = 32.184 s` as an exact nanosecond count, cross-checked against the
/// shared constants registry so the two representations cannot drift. The
/// registry value is a `double` (32.184 s is not exactly representable), so the
/// check rounds it to the nearest nanosecond (`+ 0.5`, positive by construction)
/// rather than truncating.
inline constexpr std::int64_t kTtMinusTaiNs = 32'184'000'000;
static_assert(kTtMinusTaiNs ==
                  static_cast<std::int64_t>(constants::time::kTtMinusTai * 1'000'000'000.0 + 0.5),
              "TT-TAI offset must match the constants registry");

/// Convert a GPS-time reading to the TAI master clock: `TAI = GPS + 19 s`
/// (REQ-CONV-001). Applied on every GNSS fix before the inertial-frame filter
/// and propagator run.
inline constexpr Tai toTai(const Gps& gps) {
  return Tai::fromNanosecondsSinceEpoch(gps.nanosecondsSinceEpoch() + kTaiMinusGpsNs);
}

/// Convert a TAI instant to a GPS-time reading: `GPS = TAI - 19 s`.
inline constexpr Gps toGps(const Tai& tai) {
  return Gps::fromNanosecondsSinceEpoch(tai.nanosecondsSinceEpoch() - kTaiMinusGpsNs);
}

/// Convert a TAI instant to Terrestrial Time: `TT = TAI + 32.184 s`.
inline constexpr Tt toTt(const Tai& tai) {
  return Tt::fromNanosecondsSinceEpoch(tai.nanosecondsSinceEpoch() + kTtMinusTaiNs);
}

/// Convert a Terrestrial Time instant to TAI: `TAI = TT - 32.184 s`.
inline constexpr Tai toTai(const Tt& tt) {
  return Tai::fromNanosecondsSinceEpoch(tt.nanosecondsSinceEpoch() - kTtMinusTaiNs);
}

/// @}

}  // namespace polaris::time

#endif  // POLARIS_TIME_TIMESCALES_HPP

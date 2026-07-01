#ifndef POLARIS_TIME_LEAP_SECONDS_HPP
#define POLARIS_TIME_LEAP_SECONDS_HPP

/// @file
/// @brief Leap-second table: TAI - UTC (ΔAT) lookup (design doc §3.2).
///
/// UTC is TAI minus an integer number of leap seconds (ΔAT). Onboard, the table
/// is an *uploaded* low-rate parameter (§3.2, §11.3); it is used only to derive
/// UTC for ground-facing output. A **settable/frozen** table (a single constant
/// ΔAT for all time) supports bit-reproducible Monte Carlo (§3.5).
///
/// Fixed capacity, no heap, no exceptions (flight-safe, §3.6). Lookups are over
/// a small monotonically-increasing table, so a linear scan is bounded and cheap.
///
/// References:
///  - IERS Bulletin C (leap-second announcements); ΔAT history per IERS/USNO.
///    [iers2010]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3.5
///    (UTC and leap seconds). [vallado2013]

#include <array>
#include <cstddef>
#include <cstdint>

namespace polaris::time {

/// One step in ΔAT, effective from @c date at 00:00:00 UTC.
struct LeapSecondEntry {
  std::int64_t year{1972};    ///< effective UTC year
  unsigned month{1};          ///< effective UTC month [1, 12]
  unsigned day{1};            ///< effective UTC day [1, 31]
  std::int32_t delta_at{10};  ///< TAI - UTC [s] in effect from this date onward
};

/// Ordered ΔAT table with a fixed maximum number of entries.
class LeapSecondTable {
 public:
  /// Maximum entries the fixed-size table can hold (ample for the historical
  /// record plus decades of future leaps).
  static constexpr std::size_t kMaxEntries = 64;

  /// Empty table (ΔAT = 0 everywhere until entries are added).
  LeapSecondTable() = default;

  /// The historical integer-leap-second record (post-1972 regime) through the
  /// latest announced leap. Entries are IERS ΔAT step dates.
  static LeapSecondTable historical();

  /// A frozen table: a single constant ΔAT for **all representable time**, for
  /// reproducible MC (§3.5). No leap-second steps occur. The single entry is
  /// seated far before the epoch so the constant offset also applies to `tai = 0`
  /// and to any pre-epoch instant (no near-epoch ΔAT=0 window).
  static LeapSecondTable frozen(std::int32_t delta_at_seconds);

  /// Append an entry. Entries must be added in ascending date order. Returns
  /// false (and ignores the entry) if the table is full or the date is not
  /// strictly after the previous entry.
  bool addEntry(const LeapSecondEntry& e);

  /// Number of entries currently held.
  std::size_t size() const { return count_; }

  /// ΔAT [s] in effect at UTC calendar date @p y-@p m-@p d (00:00:00). Returns
  /// the ΔAT of the latest entry whose effective date is on or before that date;
  /// 0 if the date precedes the first entry.
  std::int32_t deltaAtForUtcDate(std::int64_t y, unsigned m, unsigned d) const;

  /// ΔAT [s] in effect at a given count of TAI seconds since the TAI epoch
  /// (1970-01-01T00:00:00 TAI). Used to derive UTC from TAI.
  std::int32_t deltaAtForTaiSeconds(std::int64_t tai_seconds) const;

  /// If @p tai_seconds is the single inserted leap second immediately preceding
  /// a +1 ΔAT step (i.e. it displays as `hh:mm:60`), report it. Writes the ΔAT
  /// in effect *during* that second (the pre-step value) to @p delta_at_during
  /// and returns true; otherwise returns false and leaves @p delta_at_during
  /// unchanged.
  ///
  /// Only **positive** (+1) leap seconds are represented — none other has ever
  /// occurred and the design assumes leaps are "inserted, never removed" (§5.2).
  /// A step other than +1 (skipped/negative second) is not surfaced here.
  bool isInsertedLeapSecond(std::int64_t tai_seconds, std::int32_t& delta_at_during) const;

 private:
  /// TAI seconds since the TAI epoch at which entry @p i's ΔAT takes effect.
  std::int64_t taiThreshold(std::size_t i) const;

  std::array<LeapSecondEntry, kMaxEntries> entries_{};
  std::size_t count_{0};
};

}  // namespace polaris::time

#endif  // POLARIS_TIME_LEAP_SECONDS_HPP

/// @file
/// @brief Leap-second table implementation (design doc §3.2). See
/// `leap_seconds.hpp` for the interface contract and references.

#include "time/leap_seconds.hpp"

#include <cassert>

#include "constants/constants.hpp"
#include "time/civil.hpp"

namespace polaris::time {

namespace {
/// Seconds per day as an integer (leap seconds are inserted, never removed, at
/// day boundaries; the nominal day is exactly 86400 s).
constexpr std::int64_t kSecondsPerDay = static_cast<std::int64_t>(constants::time::kSecondsPerDay);
}  // namespace

LeapSecondTable LeapSecondTable::historical() {
  // IERS ΔAT (TAI - UTC) step history, integer-leap-second regime (from 1972).
  // Source: IERS Bulletin C / USNO leap-second record. [iers2010]
  static constexpr LeapSecondEntry kHistory[] = {
      {1972, 1, 1, 10}, {1972, 7, 1, 11}, {1973, 1, 1, 12}, {1974, 1, 1, 13}, {1975, 1, 1, 14},
      {1976, 1, 1, 15}, {1977, 1, 1, 16}, {1978, 1, 1, 17}, {1979, 1, 1, 18}, {1980, 1, 1, 19},
      {1981, 7, 1, 20}, {1982, 7, 1, 21}, {1983, 7, 1, 22}, {1985, 7, 1, 23}, {1988, 1, 1, 24},
      {1990, 1, 1, 25}, {1991, 1, 1, 26}, {1992, 7, 1, 27}, {1993, 7, 1, 28}, {1994, 7, 1, 29},
      {1996, 1, 1, 30}, {1997, 7, 1, 31}, {1999, 1, 1, 32}, {2006, 1, 1, 33}, {2009, 1, 1, 34},
      {2012, 7, 1, 35}, {2015, 7, 1, 36}, {2017, 1, 1, 37},
  };
  static_assert(sizeof(kHistory) / sizeof(kHistory[0]) <= LeapSecondTable::kMaxEntries,
                "historical leap-second record must fit the fixed-size table");
  LeapSecondTable t;
  for (const auto& e : kHistory) {
    const bool ok = t.addEntry(e);
    assert(ok && "historical leap-second entry rejected (capacity or ordering)");
    (void)ok;  // release builds: static_assert bounds capacity and the literal
               // table is ordered by construction.
  }
  return t;
}

LeapSecondTable LeapSecondTable::frozen(std::int32_t delta_at_seconds) {
  LeapSecondTable t;
  // Seat the single entry far before the epoch (year 1) so its ΔAT threshold is
  // deeply negative in TAI seconds. The constant offset then applies to every
  // representable instant — including tai = 0 and pre-epoch times — with no
  // near-epoch ΔAT=0 window and no spurious inserted leap second.
  const bool ok = t.addEntry(LeapSecondEntry{1, 1, 1, delta_at_seconds});
  assert(ok && "frozen leap-second entry rejected");
  (void)ok;
  return t;
}

bool LeapSecondTable::addEntry(const LeapSecondEntry& e) {
  if (count_ >= kMaxEntries) {
    return false;
  }
  if (count_ > 0) {
    const LeapSecondEntry& prev = entries_[count_ - 1];
    const std::int64_t prev_days = daysFromCivil(prev.year, prev.month, prev.day);
    const std::int64_t new_days = daysFromCivil(e.year, e.month, e.day);
    if (new_days <= prev_days) {
      return false;  // must be strictly increasing in date
    }
  }
  entries_[count_] = e;
  ++count_;
  return true;
}

std::int32_t LeapSecondTable::deltaAtForUtcDate(std::int64_t y, unsigned m, unsigned d) const {
  const std::int64_t days = daysFromCivil(y, m, d);
  std::int32_t delta = 0;
  for (std::size_t i = 0; i < count_; ++i) {
    const std::int64_t entry_days =
        daysFromCivil(entries_[i].year, entries_[i].month, entries_[i].day);
    if (entry_days <= days) {
      delta = entries_[i].delta_at;
    } else {
      break;  // entries are date-ordered; no later entry can apply
    }
  }
  return delta;
}

std::int64_t LeapSecondTable::taiThreshold(std::size_t i) const {
  // The step takes effect at 00:00:00 UTC on the entry date; in TAI seconds that
  // instant is the nominal UTC second count plus the NEW ΔAT.
  const std::int64_t nominal_utc_sec =
      daysFromCivil(entries_[i].year, entries_[i].month, entries_[i].day) * kSecondsPerDay;
  return nominal_utc_sec + entries_[i].delta_at;
}

std::int32_t LeapSecondTable::deltaAtForTaiSeconds(std::int64_t tai_seconds) const {
  std::int32_t delta = 0;
  for (std::size_t i = 0; i < count_; ++i) {
    if (taiThreshold(i) <= tai_seconds) {
      delta = entries_[i].delta_at;
    } else {
      break;
    }
  }
  return delta;
}

bool LeapSecondTable::isInsertedLeapSecond(std::int64_t tai_seconds,
                                           std::int32_t& delta_at_during) const {
  for (std::size_t i = 0; i < count_; ++i) {
    const std::int64_t threshold = taiThreshold(i);
    if (threshold > tai_seconds) {
      // `i` is the next upcoming step. A +1 step inserts one leap second at
      // (threshold - 1); larger/negative steps have no single inserted second.
      const std::int32_t prev = (i == 0) ? 0 : entries_[i - 1].delta_at;
      const std::int32_t step = entries_[i].delta_at - prev;
      if (step == 1 && tai_seconds == threshold - 1) {
        delta_at_during = prev;
        return true;
      }
      return false;
    }
  }
  return false;
}

}  // namespace polaris::time

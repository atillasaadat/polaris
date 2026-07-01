#ifndef POLARIS_TIME_CIVIL_HPP
#define POLARIS_TIME_CIVIL_HPP

/// @file
/// @brief Proleptic-Gregorian calendar <-> serial-day conversions.
///
/// Branch-free, overflow-safe integer algorithms for converting between a civil
/// (year, month, day) date and a serial day count relative to 1970-01-01. Used
/// by the leap-second table and the UTC calendar conversions (`utc.hpp`); shared
/// so both agree on the calendar. Valid for any date in the proleptic Gregorian
/// calendar; `constexpr`, no heap, no exceptions (flight-safe, §3.6).
///
/// References:
///  - H. Hinnant, "chrono-Compatible Low-Level Date Algorithms"
///    (`days_from_civil` / `civil_from_days`), public domain. [hinnant2016]

#include <cstdint>

namespace polaris::time {

/// Days from 1970-01-01 to the civil date @p y-@p m-@p d (negative before 1970).
/// @p m in `[1, 12]`, @p d in `[1, last-day-of-month]`. Exact for all inputs.
constexpr std::int64_t daysFromCivil(std::int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);                  // [0, 399]
  const unsigned doy = (153u * (m + (m > 2 ? -3u : 9u)) + 2u) / 5u + d - 1u;  // [0, 365]
  const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;              // [0, 146096]
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

/// A broken-down proleptic-Gregorian date.
struct CivilDate {
  std::int64_t year{1970};
  unsigned month{1};  ///< [1, 12]
  unsigned day{1};    ///< [1, 31]
};

/// Inverse of `daysFromCivil`: the civil date @p z days from 1970-01-01.
constexpr CivilDate civilFromDays(std::int64_t z) {
  z += 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);                // [0, 146096]
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
  const std::int64_t y = static_cast<std::int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);  // [0, 365]
  const unsigned mp = (5u * doy + 2u) / 153u;                       // [0, 11]
  const unsigned d = doy - (153u * mp + 2u) / 5u + 1u;              // [1, 31]
  const unsigned m = mp + (mp < 10u ? 3u : -9u);                    // [1, 12]
  return CivilDate{y + (m <= 2), m, d};
}

}  // namespace polaris::time

#endif  // POLARIS_TIME_CIVIL_HPP

/// @file
/// @brief UTC <-> TAI calendar conversion (design doc §3.2). See `utc.hpp`.

#include "time/utc.hpp"

#include <cassert>
#include <cmath>

#include "constants/constants.hpp"
#include "time/civil.hpp"
#include "time/duration.hpp"

namespace polaris::time {

namespace {

constexpr std::int64_t kSecondsPerDay = static_cast<std::int64_t>(constants::time::kSecondsPerDay);
constexpr std::int64_t kNsPerSecond = Duration::kNsPerSecond;

/// Floor division (round toward negative infinity) for signed integers.
std::int64_t floorDiv(std::int64_t a, std::int64_t b) {
  const std::int64_t q = a / b;
  const std::int64_t r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

/// Non-negative remainder consistent with `floorDiv`.
std::int64_t floorMod(std::int64_t a, std::int64_t b) {
  return a - floorDiv(a, b) * b;
}

}  // namespace

bool isValidUtc(const UtcDateTime& utc) {
  if (utc.month < 1 || utc.month > 12) {
    return false;
  }
  if (utc.day < 1 || utc.day > 31) {
    return false;
  }
  if (utc.hour > 23 || utc.minute > 59 || utc.second > 60) {
    return false;
  }
  if (utc.nanosecond < 0 || utc.nanosecond >= Duration::kNsPerSecond) {
    return false;
  }
  // Reject impossible calendar days (e.g. Feb 30) via a civil round-trip.
  const CivilDate rt = civilFromDays(daysFromCivil(utc.year, utc.month, utc.day));
  return rt.year == utc.year && rt.month == utc.month && rt.day == utc.day;
}

Tai taiFromUtc(const UtcDateTime& utc, const LeapSecondTable& leap) {
  assert(isValidUtc(utc) && "taiFromUtc requires a valid UtcDateTime (see isValidUtc)");
  // Nominal UTC seconds since 1970-01-01T00:00:00 UTC (calendar treated as
  // uniform 86400 s/day). A leap second arrives as second == 60, which pushes
  // the count to the following midnight — consistent with the ΔAT of the
  // pre-step date returned below.
  const std::int64_t nominal_utc_sec =
      daysFromCivil(utc.year, utc.month, utc.day) * kSecondsPerDay +
      static_cast<std::int64_t>(utc.hour) * 3600 + static_cast<std::int64_t>(utc.minute) * 60 +
      static_cast<std::int64_t>(utc.second);
  const std::int32_t delta = leap.deltaAtForUtcDate(utc.year, utc.month, utc.day);
  const std::int64_t tai_sec = nominal_utc_sec + delta;
  const std::int64_t tai_ns = tai_sec * kNsPerSecond + utc.nanosecond;
  return Tai::fromNanosecondsSinceEpoch(tai_ns);
}

UtcDateTime utcFromTai(const Tai& tai, const LeapSecondTable& leap) {
  const std::int64_t tai_ns = tai.nanosecondsSinceEpoch();
  const std::int64_t tai_sec = floorDiv(tai_ns, kNsPerSecond);
  const std::int32_t frac_ns = static_cast<std::int32_t>(floorMod(tai_ns, kNsPerSecond));

  UtcDateTime out;
  out.nanosecond = frac_ns;

  std::int32_t delta_during = 0;
  if (leap.isInsertedLeapSecond(tai_sec, delta_during)) {
    // The inserted second displays as 23:59:60 of the day *before* the ΔAT step.
    const std::int64_t nominal = tai_sec - delta_during;  // = midnight of step day
    const CivilDate date = civilFromDays(floorDiv(nominal, kSecondsPerDay) - 1);
    out.year = date.year;
    out.month = date.month;
    out.day = date.day;
    out.hour = 23;
    out.minute = 59;
    out.second = 60;
    return out;
  }

  const std::int32_t delta = leap.deltaAtForTaiSeconds(tai_sec);
  const std::int64_t nominal = tai_sec - delta;
  const std::int64_t day_index = floorDiv(nominal, kSecondsPerDay);
  const std::int64_t sec_of_day = nominal - day_index * kSecondsPerDay;
  const CivilDate date = civilFromDays(day_index);
  out.year = date.year;
  out.month = date.month;
  out.day = date.day;
  out.hour = static_cast<unsigned>(sec_of_day / 3600);
  out.minute = static_cast<unsigned>((sec_of_day % 3600) / 60);
  out.second = static_cast<unsigned>(sec_of_day % 60);
  return out;
}

bool decimalYear(const Tai& epoch, const LeapSecondTable& leap, double& out) {
  const UtcDateTime utc = utcFromTai(epoch, leap);
  const std::int64_t year_start = daysFromCivil(utc.year, 1, 1);
  const std::int64_t next_year_start = daysFromCivil(utc.year + 1, 1, 1);
  const double days_in_year = static_cast<double>(next_year_start - year_start);
  if (!(days_in_year > 0.0)) {
    return false;
  }

  const double day_of_year =
      static_cast<double>(daysFromCivil(utc.year, utc.month, utc.day) - year_start);
  const double seconds_of_day =
      static_cast<double>(utc.hour) * 3600.0 + static_cast<double>(utc.minute) * 60.0 +
      static_cast<double>(utc.second) + static_cast<double>(utc.nanosecond) * 1e-9;

  const double year =
      static_cast<double>(utc.year) +
      (day_of_year + seconds_of_day / static_cast<double>(kSecondsPerDay)) / days_in_year;
  if (!std::isfinite(year)) {
    return false;
  }
  out = year;
  return true;
}

}  // namespace polaris::time

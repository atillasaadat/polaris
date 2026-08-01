#ifndef POLARIS_TIME_UTC_HPP
#define POLARIS_TIME_UTC_HPP

/// @file
/// @brief UTC calendar time, derived from TAI for ground use only (§3.2).
///
/// UTC is not a uniform onboard scale — it carries leap seconds — so it is
/// represented as a broken-down calendar value, never as the master clock. The
/// FSW keeps time in TAI (`timescales.hpp`); UTC is produced at the ground/
/// telemetry boundary via the uploaded leap-second table (REQ-SYS-001).
///
/// A UTC-with-leap-table round-trip (`taiFromUtc` ∘ `utcFromTai`) is exact,
/// including the inserted leap second (which displays as `hh:mm:60`).
///
/// No heap, no exceptions (flight-safe, §3.6).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3.5
///    (UTC, leap seconds, calendar conversions). [vallado2013]

#include <cstdint>

#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::time {

/// A broken-down UTC calendar instant. `second == 60` represents a positive
/// leap second; `nanosecond` is the sub-second remainder.
struct UtcDateTime {
  std::int64_t year{1970};
  unsigned month{1};           ///< [1, 12]
  unsigned day{1};             ///< [1, 31]
  unsigned hour{0};            ///< [0, 23]
  unsigned minute{0};          ///< [0, 59]
  unsigned second{0};          ///< [0, 60] (60 only during a positive leap second)
  std::int32_t nanosecond{0};  ///< [0, 1e9)
};

/// True if every field of @p utc is in range and (year, month, day) is a real
/// proleptic-Gregorian date: month `[1,12]`, day valid for that month, hour
/// `[0,23]`, minute `[0,59]`, second `[0,60]` (60 for a leap second), nanosecond
/// `[0, 1e9)`. Boundary validator (§3.6) — callers on the flight side should
/// check this before `taiFromUtc`, which requires a valid input as a precondition.
bool isValidUtc(const UtcDateTime& utc);

/// Convert a UTC calendar instant to the TAI master clock, applying the ΔAT in
/// effect at that UTC date from @p leap (design doc §3.2). During a leap second
/// pass `second = 60`. Precondition: `isValidUtc(utc)` — an out-of-range field
/// would otherwise produce a silently wrong (non-invertible) TAI.
Tai taiFromUtc(const UtcDateTime& utc, const LeapSecondTable& leap);

/// Derive UTC from a TAI instant using @p leap (design doc §3.2). Emits
/// `second = 60` for the inserted leap second so the round-trip is exact.
UtcDateTime utcFromTai(const Tai& tai, const LeapSecondTable& leap);

/// Convert a TAI instant to the **decimal year** the IGRF geomagnetic model is
/// parameterised by (e.g. 2026.5), using @p leap for the UTC reduction.
///
/// The fraction divides by the actual length of *that* year rather than a fixed
/// 365.25, so a date lands at the same fraction the IGRF epoch grid means by it.
/// The difference is under a day — negligible against a field that drifts tens
/// of nT per year — but a fixed divisor also drifts 1 January off `.0` in leap
/// years, which is confusing to read in a log.
///
/// Flight-safe: pure arithmetic, no allocation. Both the truth sim's magnetic
/// field and the FSW's onboard IGRF reference (§8.1) go through this one
/// conversion, so their epochs cannot disagree.
///
/// @return false, leaving @p out untouched, if the conversion is not finite.
bool decimalYear(const Tai& epoch, const LeapSecondTable& leap, double& out);

}  // namespace polaris::time

#endif  // POLARIS_TIME_UTC_HPP

/// @file
/// @brief TT↔TDB periodic-term conversion (design doc §3.2). See tdb.hpp.

#include "time/tdb.hpp"

#include <cmath>

#include "constants/constants.hpp"
#include "time/duration.hpp"

namespace polaris::time {

namespace {

constexpr double kDegToRad = 3.141'592'653'589'793'238 / 180.0;

/// Days elapsed since the J2000.0 epoch for a uniform-scale nanosecond count.
/// The instant's nanoseconds are measured since 1970-01-01T00:00:00 on its own
/// scale; JD is a scale-agnostic calendar count, so shifting by the fixed
/// 1970→J2000 span gives the astronomical day argument.
double daysSinceJ2000(std::int64_t ns_since_1970) {
  const double days_since_1970 =
      static_cast<double>(ns_since_1970) /
      (static_cast<double>(Duration::kNsPerSecond) * constants::time::kSecondsPerDay);
  return days_since_1970 + (constants::time::kJulianDate1970 - constants::time::kJulianDateJ2000);
}

/// The two-harmonic TDB−TT series [s] for a day-count since J2000.
double tdbSeriesSeconds(double days_from_j2000) {
  const double g_deg =
      constants::tdb::kMeanAnomalyDeg + constants::tdb::kMeanAnomalyRateDegPerDay * days_from_j2000;
  const double g = g_deg * kDegToRad;
  return constants::tdb::kAmplitude1 * std::sin(g) +
         constants::tdb::kAmplitude2 * std::sin(2.0 * g);
}

}  // namespace

double tdbMinusTtSeconds(const Tt& tt) {
  return tdbSeriesSeconds(daysSinceJ2000(tt.nanosecondsSinceEpoch()));
}

Tdb toTdb(const Tt& tt) {
  const Duration offset = Duration::fromSecondsF(tdbMinusTtSeconds(tt));
  return Tdb::fromNanosecondsSinceEpoch(tt.nanosecondsSinceEpoch() + offset.nanoseconds());
}

Tt toTt(const Tdb& tdb) {
  // Evaluate the periodic term at the TDB argument; see the header for why this
  // single-shot inversion is accurate to well under 1 µs.
  const Duration offset =
      Duration::fromSecondsF(tdbSeriesSeconds(daysSinceJ2000(tdb.nanosecondsSinceEpoch())));
  return Tt::fromNanosecondsSinceEpoch(tdb.nanosecondsSinceEpoch() - offset.nanoseconds());
}

}  // namespace polaris::time

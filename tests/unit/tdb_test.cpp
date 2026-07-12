/// @file Tests for the TT↔TDB periodic-term conversion (design doc §3.2, §11.3).
///
/// Validated by an independent recomputation of the two-harmonic series at known
/// epochs (which also checks the 1970→J2000 day conversion and the deg→rad
/// scaling), the bounded magnitude of the term, and TT→TDB→TT round-trips.

#include "time/tdb.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "constants/constants.hpp"
#include "time/timescales.hpp"

namespace pt = polaris::time;

namespace {

constexpr double kPi = 3.141'592'653'589'793'238;
constexpr std::int64_t kNsPerSecond = 1'000'000'000;

// Seconds from the 1970 uniform-scale epoch to the J2000 epoch (2000-01-01T12:00 TT).
constexpr std::int64_t kSecondsToJ2000 = 946'728'000;  // (2451545.0 - 2440587.5) * 86400

pt::Tt TtAtDaysFromJ2000(double days) {
  const double seconds =
      static_cast<double>(kSecondsToJ2000) + days * polaris::constants::time::kSecondsPerDay;
  return pt::Tt::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(seconds) * kNsPerSecond);
}

// The reference series, recomputed independently of the library.
double referenceSeries(double days_from_j2000) {
  namespace c = polaris::constants::tdb;
  const double g =
      (c::kMeanAnomalyDeg + c::kMeanAnomalyRateDegPerDay * days_from_j2000) * kPi / 180.0;
  return c::kAmplitude1 * std::sin(g) + c::kAmplitude2 * std::sin(2.0 * g);
}

}  // namespace

TEST(Tdb, MatchesReferenceSeriesAtJ2000) {
  RecordProperty("verifies", "REQ-CDH-002");
  // daysSinceJ2000 must be exactly 0 at the J2000 epoch — catches any 1970→J2000
  // or scale-epoch bug in the day conversion.
  const pt::Tt j2000 = TtAtDaysFromJ2000(0.0);
  EXPECT_NEAR(pt::tdbMinusTtSeconds(j2000), referenceSeries(0.0), 1e-15);
}

TEST(Tdb, MatchesReferenceSeriesAtOffsetEpoch) {
  RecordProperty("verifies", "REQ-CDH-002");
  // 100 days past J2000 exercises the mean-anomaly rate term.
  const double days = 100.0;
  EXPECT_NEAR(pt::tdbMinusTtSeconds(TtAtDaysFromJ2000(days)), referenceSeries(days), 1e-15);
}

TEST(Tdb, MagnitudeStaysUnderTwoMilliseconds) {
  RecordProperty("verifies", "REQ-CDH-002");
  // The term is purely periodic (no secular drift); sweep ~3 years densely.
  for (double days = -400.0; days <= 800.0; days += 1.0) {
    EXPECT_LT(std::abs(pt::tdbMinusTtSeconds(TtAtDaysFromJ2000(days))), 1.75e-3);
  }
}

TEST(Tdb, IsAnnuallyPeriodic) {
  RecordProperty("verifies", "REQ-CDH-002");
  // One Earth mean-anomaly period is 360 / rate days; the term repeats.
  const double period_days = 360.0 / polaris::constants::tdb::kMeanAnomalyRateDegPerDay;
  const double v0 = pt::tdbMinusTtSeconds(TtAtDaysFromJ2000(50.0));
  const double v1 = pt::tdbMinusTtSeconds(TtAtDaysFromJ2000(50.0 + period_days));
  EXPECT_NEAR(v0, v1, 5e-8);  // sub-µs repeatability
}

TEST(Tdb, RoundTripsTtThroughTdb) {
  RecordProperty("verifies", "REQ-CDH-002");
  for (double days : {-200.0, 0.0, 93.0, 365.0}) {
    const pt::Tt tt = TtAtDaysFromJ2000(days);
    const pt::Tt back = pt::toTt(pt::toTdb(tt));
    // Forward and inverse round the same sub-ms offset; agreement to a couple ns.
    EXPECT_LE(std::abs(back.nanosecondsSinceEpoch() - tt.nanosecondsSinceEpoch()), 2);
  }
}

TEST(Tdb, OffsetIsSubMillisecondAndSignedByTheSeries) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pt::Tt tt = TtAtDaysFromJ2000(93.0);  // near the annual maximum
  const pt::Tdb tdb = pt::toTdb(tt);
  const std::int64_t applied = tdb.nanosecondsSinceEpoch() - tt.nanosecondsSinceEpoch();
  const double expected_ns = pt::tdbMinusTtSeconds(tt) * 1e9;
  EXPECT_NEAR(static_cast<double>(applied), expected_ns, 1.0);  // within rounding
  EXPECT_LT(std::abs(applied), 2'000'000);                      // < 2 ms
}

/// @file Unit tests for the lib/time library (design doc §3.2).
///
/// Covers the TAI master clock, two-part high-precision form, the constant-offset
/// GPS/TT scale conversions, the leap-second table, and UTC derivation. Values
/// are checked by self-consistency (round-trips), known IERS anchors, and the
/// exact integer offsets; external GMAT-golden cross-validation lands in Push 5.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "time/civil.hpp"
#include "time/duration.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace pt = polaris::time;

// --- Duration ---------------------------------------------------------------

TEST(Duration, NanosecondAndSecondRoundTrip) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto d = pt::Duration::fromSeconds(90);
  EXPECT_EQ(d.nanoseconds(), 90'000'000'000);
  EXPECT_DOUBLE_EQ(d.seconds(), 90.0);
}

TEST(Duration, Arithmetic) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto a = pt::Duration::fromNanoseconds(1'000);
  const auto b = pt::Duration::fromNanoseconds(250);
  EXPECT_EQ((a + b).nanoseconds(), 1'250);
  EXPECT_EQ((a - b).nanoseconds(), 750);
  EXPECT_EQ((-b).nanoseconds(), -250);
  EXPECT_EQ((b * 4).nanoseconds(), 1'000);
  EXPECT_TRUE(b < a);
  EXPECT_TRUE(a >= a);
}

TEST(Duration, FromFractionalSecondsRoundsHalfAwayFromZero) {
  RecordProperty("verifies", "REQ-CONV-005");
  EXPECT_EQ(pt::Duration::fromSecondsF(1.5).nanoseconds(), 1'500'000'000);
  EXPECT_EQ(pt::Duration::fromSecondsF(-1.5).nanoseconds(), -1'500'000'000);
  EXPECT_EQ(pt::Duration::fromSecondsF(2.6e-9).nanoseconds(), 3);
}

TEST(Duration, FromFractionalSecondsGuardsNonFiniteAndRange) {
  RecordProperty("verifies", "REQ-CONV-005");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // Non-finite -> zero (no std::llround UB).
  EXPECT_EQ(pt::Duration::fromSecondsF(nan).nanoseconds(), 0);
  EXPECT_EQ(pt::Duration::fromSecondsF(inf).nanoseconds(), 0);
  EXPECT_EQ(pt::Duration::fromSecondsF(-inf).nanoseconds(), 0);
  // Beyond the representable range -> saturates rather than overflowing.
  const std::int64_t max_ns = static_cast<std::int64_t>(pt::Duration::kMaxSeconds) * 1'000'000'000;
  EXPECT_EQ(pt::Duration::fromSecondsF(1e30).nanoseconds(), max_ns);
  EXPECT_EQ(pt::Duration::fromSecondsF(-1e30).nanoseconds(), -max_ns);
}

// --- TAI master clock -------------------------------------------------------

TEST(Tai, MonotonicOrderingAndInterval) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto t0 = pt::Tai::fromNanosecondsSinceEpoch(0);
  const auto t1 = t0 + pt::Duration::fromSeconds(10);
  EXPECT_TRUE(t0 < t1);
  EXPECT_TRUE(t1 > t0);
  EXPECT_EQ((t1 - t0).nanoseconds(), 10'000'000'000);
  EXPECT_EQ((t1 - pt::Duration::fromSeconds(10)).nanosecondsSinceEpoch(), 0);
}

TEST(Tai, TwoPartHighPrecisionPositive) {
  RecordProperty("verifies", "REQ-CONV-005");
  const auto t = pt::Tai::fromNanosecondsSinceEpoch(3'500'000'000);  // 3.5 s
  const pt::TwoPart tp = t.twoPart();
  EXPECT_EQ(tp.seconds, 3);
  EXPECT_DOUBLE_EQ(tp.fraction, 0.5);
}

TEST(Tai, TwoPartHighPrecisionNegativeFloors) {
  RecordProperty("verifies", "REQ-CONV-005");
  const auto t = pt::Tai::fromNanosecondsSinceEpoch(-500'000'000);  // -0.5 s
  const pt::TwoPart tp = t.twoPart();
  EXPECT_EQ(tp.seconds, -1);           // floored toward negative infinity
  EXPECT_DOUBLE_EQ(tp.fraction, 0.5);  // fraction stays in [0, 1)
}

TEST(Tai, TwoPartRetainsNanosecondResolutionAcrossLongArc) {
  RecordProperty("verifies", "REQ-CONV-005");
  // ~20 years of nanoseconds: a single double would lose sub-microsecond bits.
  const std::int64_t ns = 631'152'000LL * 1'000'000'000LL + 123'456'789LL;
  const auto t = pt::Tai::fromNanosecondsSinceEpoch(ns);
  const pt::TwoPart tp = t.twoPart();
  EXPECT_EQ(tp.seconds, 631'152'000);
  EXPECT_NEAR(tp.fraction, 0.123456789, 1e-15);
}

// --- Constant-offset scale conversions --------------------------------------

TEST(Scales, GpsToTaiIsPlus19Seconds) {
  RecordProperty("verifies", "REQ-CONV-001");
  const auto gps = pt::Gps::fromNanosecondsSinceEpoch(0);
  const auto tai = pt::toTai(gps);
  EXPECT_EQ((tai.nanosecondsSinceEpoch() - gps.nanosecondsSinceEpoch()), 19'000'000'000);
  // Round-trip back to GPS.
  EXPECT_EQ(pt::toGps(tai).nanosecondsSinceEpoch(), gps.nanosecondsSinceEpoch());
}

TEST(Scales, TtToTaiIsMinus32point184Seconds) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto tai = pt::Tai::fromNanosecondsSinceEpoch(1'000'000'000);
  const auto tt = pt::toTt(tai);
  EXPECT_EQ((tt.nanosecondsSinceEpoch() - tai.nanosecondsSinceEpoch()), 32'184'000'000);
  EXPECT_EQ(pt::toTai(tt).nanosecondsSinceEpoch(), tai.nanosecondsSinceEpoch());
}

// --- Civil calendar ---------------------------------------------------------

TEST(Civil, KnownDayNumbersAndRoundTrip) {
  RecordProperty("verifies", "REQ-SYS-001");
  EXPECT_EQ(pt::daysFromCivil(1970, 1, 1), 0);
  EXPECT_EQ(pt::daysFromCivil(1972, 1, 1), 730);
  EXPECT_EQ(pt::daysFromCivil(2000, 1, 1), 10957);
  for (std::int64_t z : {-1000, 0, 1, 10957, 20000, 30000}) {
    const pt::CivilDate c = pt::civilFromDays(z);
    EXPECT_EQ(pt::daysFromCivil(c.year, c.month, c.day), z);
  }
}

// --- Leap-second table ------------------------------------------------------

TEST(LeapSeconds, HistoricalDeltaAtKnownValues) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto leap = pt::LeapSecondTable::historical();
  EXPECT_EQ(leap.deltaAtForUtcDate(1971, 1, 1), 0);   // before the table
  EXPECT_EQ(leap.deltaAtForUtcDate(1972, 1, 1), 10);  // first entry
  EXPECT_EQ(leap.deltaAtForUtcDate(1999, 6, 1), 32);
  EXPECT_EQ(leap.deltaAtForUtcDate(2017, 1, 1), 37);  // latest announced
  EXPECT_EQ(leap.deltaAtForUtcDate(2025, 1, 1), 37);
}

TEST(LeapSeconds, FrozenTableIsConstant) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto leap = pt::LeapSecondTable::frozen(37);
  EXPECT_EQ(leap.deltaAtForUtcDate(1970, 1, 1), 37);
  EXPECT_EQ(leap.deltaAtForUtcDate(2100, 1, 1), 37);
}

TEST(LeapSeconds, FrozenIsConstantAtAndBeforeEpoch) {
  RecordProperty("verifies", "REQ-SYS-001");
  // Regression: the frozen offset must hold for tai = 0, the first seconds after
  // the epoch, and pre-epoch instants (no near-epoch ΔAT=0 window).
  const auto leap = pt::LeapSecondTable::frozen(37);
  EXPECT_EQ(leap.deltaAtForTaiSeconds(0), 37);
  EXPECT_EQ(leap.deltaAtForTaiSeconds(10), 37);
  EXPECT_EQ(leap.deltaAtForTaiSeconds(-1'000'000), 37);
  // And frozen(1) must not fabricate a spurious 1969-12-31T23:59:60.
  std::int32_t during = 0;
  EXPECT_FALSE(pt::LeapSecondTable::frozen(1).isInsertedLeapSecond(0, during));
}

TEST(LeapSeconds, RejectsOutOfOrderEntries) {
  RecordProperty("verifies", "REQ-SYS-001");
  pt::LeapSecondTable t;
  EXPECT_TRUE(t.addEntry({2000, 1, 1, 32}));
  EXPECT_FALSE(t.addEntry({1999, 1, 1, 31}));  // earlier date rejected
  EXPECT_EQ(t.size(), 1u);
}

// --- UTC <-> TAI ------------------------------------------------------------

TEST(Utc, EpochWithZeroOffset) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto leap = pt::LeapSecondTable::frozen(0);
  const pt::UtcDateTime epoch{1970, 1, 1, 0, 0, 0, 0};
  EXPECT_EQ(pt::taiFromUtc(epoch, leap).nanosecondsSinceEpoch(), 0);
}

TEST(Utc, TaiLeadsUtcByDeltaAt) {
  RecordProperty("verifies", "REQ-SYS-001");
  // 2017-01-01T00:00:00 UTC == 2017-01-01T00:00:37 TAI (ΔAT = 37).
  const auto leap = pt::LeapSecondTable::historical();
  const pt::UtcDateTime utc{2017, 1, 1, 0, 0, 0, 0};
  const auto tai = pt::taiFromUtc(utc, leap);
  const std::int64_t expected_sec = pt::daysFromCivil(2017, 1, 1) * 86400 + 37;
  EXPECT_EQ(tai.nanosecondsSinceEpoch(), expected_sec * 1'000'000'000);
}

TEST(Utc, RoundTripNormalTimes) {
  RecordProperty("verifies", "REQ-SYS-001");
  const auto leap = pt::LeapSecondTable::historical();
  const pt::UtcDateTime samples[] = {
      {2000, 1, 1, 12, 0, 0, 0},
      {2023, 7, 4, 18, 30, 15, 250'000'000},
      {1985, 3, 21, 6, 45, 59, 999'999'999},
  };
  for (const auto& u : samples) {
    const pt::UtcDateTime r = pt::utcFromTai(pt::taiFromUtc(u, leap), leap);
    EXPECT_EQ(r.year, u.year);
    EXPECT_EQ(r.month, u.month);
    EXPECT_EQ(r.day, u.day);
    EXPECT_EQ(r.hour, u.hour);
    EXPECT_EQ(r.minute, u.minute);
    EXPECT_EQ(r.second, u.second);
    EXPECT_EQ(r.nanosecond, u.nanosecond);
  }
}

TEST(Utc, InsertedLeapSecondIsExact) {
  RecordProperty("verifies", "REQ-SYS-001");
  // 2016-12-31T23:59:60 UTC (the leap second) == 2017-01-01T00:00:36 TAI.
  const auto leap = pt::LeapSecondTable::historical();
  const pt::UtcDateTime leap_sec{2016, 12, 31, 23, 59, 60, 0};
  const auto tai = pt::taiFromUtc(leap_sec, leap);
  const std::int64_t expected_sec = pt::daysFromCivil(2017, 1, 1) * 86400 + 36;
  EXPECT_EQ(tai.nanosecondsSinceEpoch(), expected_sec * 1'000'000'000);

  const pt::UtcDateTime r = pt::utcFromTai(tai, leap);
  EXPECT_EQ(r.year, 2016);
  EXPECT_EQ(r.month, 12u);
  EXPECT_EQ(r.day, 31u);
  EXPECT_EQ(r.hour, 23u);
  EXPECT_EQ(r.minute, 59u);
  EXPECT_EQ(r.second, 60u);
}

TEST(Utc, IsValidUtcChecksFieldRanges) {
  RecordProperty("verifies", "REQ-SYS-001");
  EXPECT_TRUE(pt::isValidUtc({2023, 7, 4, 18, 30, 15, 250'000'000}));
  EXPECT_TRUE(pt::isValidUtc({2016, 12, 31, 23, 59, 60, 0}));          // leap second
  EXPECT_FALSE(pt::isValidUtc({2023, 13, 1, 0, 0, 0, 0}));             // month > 12
  EXPECT_FALSE(pt::isValidUtc({2023, 2, 30, 0, 0, 0, 0}));             // Feb 30
  EXPECT_FALSE(pt::isValidUtc({2023, 1, 1, 24, 0, 0, 0}));             // hour 24
  EXPECT_FALSE(pt::isValidUtc({2023, 1, 1, 0, 0, 61, 0}));             // second 61
  EXPECT_FALSE(pt::isValidUtc({2023, 1, 1, 0, 0, 0, 1'000'000'000}));  // ns == 1e9
}

TEST(Utc, GpsEpochAlignsWithUtcCalendar) {
  RecordProperty("verifies", "REQ-CONV-001");
  // At the GPS epoch (1980-01-06) ΔAT = 19, so GPS time equals UTC that day:
  // converting UTC -> TAI -> GPS must land on the plain UTC-calendar second count.
  const auto leap = pt::LeapSecondTable::historical();
  const pt::UtcDateTime utc{1980, 1, 6, 0, 0, 0, 0};
  const auto gps = pt::toGps(pt::taiFromUtc(utc, leap));
  const std::int64_t utc_cal_sec = pt::daysFromCivil(1980, 1, 6) * 86400;
  EXPECT_EQ(gps.nanosecondsSinceEpoch(), utc_cal_sec * 1'000'000'000);
}

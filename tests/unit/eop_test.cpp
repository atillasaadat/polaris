/// @file Tests for the onboard IERS EOP table (design doc §11.3, REQ-CDH-002).
///
/// Validated by independently recomputed interpolants (the expected values are
/// hand-computed from the entry data, not read back from the implementation),
/// the container's ordering/capacity contracts, and — the reason this file
/// exists separately — the leap-second continuity property: `UT1 - TAI` must
/// stay smooth across a ΔAT step even though the published `UT1 - UTC` it is
/// derived from jumps by a full second there.

#include "frames/eop.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "time/civil.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace pf = polaris::frames;
namespace pt = polaris::time;

namespace {

/// UTC MJD of a civil date, independently of the table's internals.
double MjdOf(std::int64_t y, unsigned m, unsigned d) {
  return static_cast<double>(pt::daysFromCivil(y, m, d)) + pf::kMjd1970;
}

/// TAI instant at 00:00:00 UTC on a civil date, plus @p offset_sec.
pt::Tai TaiAtUtc(std::int64_t y, unsigned m, unsigned d, double offset_sec = 0.0) {
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const std::int64_t nominal = pt::daysFromCivil(y, m, d) * 86400;
  const std::int64_t tai_sec = nominal + leap.deltaAtForUtcDate(y, m, d);
  return pt::Tai::fromNanosecondsSinceEpoch(tai_sec * 1'000'000'000) +
         pt::Duration::fromSecondsF(offset_sec);
}

// Two consecutive quiet days in 2020 (no leap second nearby), with representative
// IERS-scale magnitudes. Deliberately unequal slopes in x and y so a mixed-up
// component would show.
constexpr double kDut1Day0 = -0.1770;
constexpr double kDut1Day1 = -0.1782;
constexpr double kXpDay0 = 0.0730;
constexpr double kXpDay1 = 0.0742;
constexpr double kYpDay0 = 0.2850;
constexpr double kYpDay1 = 0.2830;

pf::EopTable<8> QuietTable() {
  pf::EopTable<8> t;
  EXPECT_TRUE(t.addEntry({MjdOf(2020, 6, 1), kDut1Day0, kXpDay0, kYpDay0}));
  EXPECT_TRUE(t.addEntry({MjdOf(2020, 6, 2), kDut1Day1, kXpDay1, kYpDay1}));
  return t;
}

}  // namespace

TEST(EopTable, RejectsOutOfOrderDuplicateAndNonFiniteEntries) {
  RecordProperty("verifies", "REQ-CDH-002");
  pf::EopTable<8> t;
  ASSERT_TRUE(t.addEntry({MjdOf(2020, 6, 2), -0.1, 0.0, 0.0}));
  // Strictly-ascending MJD is a load-time contract, so a lookup never has to
  // re-validate ordering.
  EXPECT_FALSE(t.addEntry({MjdOf(2020, 6, 1), -0.1, 0.0, 0.0}));  // earlier
  EXPECT_FALSE(t.addEntry({MjdOf(2020, 6, 2), -0.1, 0.0, 0.0}));  // duplicate
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(t.addEntry({MjdOf(2020, 6, 3), nan, 0.0, 0.0}));
  EXPECT_FALSE(t.addEntry({nan, -0.1, 0.0, 0.0}));
  EXPECT_EQ(t.size(), 1u);  // none of the rejects landed
}

TEST(EopTable, EnforcesFixedCapacity) {
  RecordProperty("verifies", "REQ-CDH-002");
  pf::EopTable<2> t;
  ASSERT_TRUE(t.addEntry({MjdOf(2020, 6, 1), -0.1, 0.0, 0.0}));
  ASSERT_TRUE(t.addEntry({MjdOf(2020, 6, 2), -0.1, 0.0, 0.0}));
  EXPECT_FALSE(t.addEntry({MjdOf(2020, 6, 3), -0.1, 0.0, 0.0}));  // full: no heap growth
  EXPECT_EQ(t.size(), 2u);
}

TEST(EopTable, InterpolatesLinearlyAtTheMidpoint) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pf::EopTable<8> t = QuietTable();
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const double delta_at = 37.0;  // ΔAT in effect through 2020

  pf::EopValue mid;
  ASSERT_TRUE(t.lookup(TaiAtUtc(2020, 6, 1, 43200.0), leap, mid));  // 12:00, halfway
  EXPECT_NEAR(mid.ut1_minus_tai, 0.5 * ((kDut1Day0 - delta_at) + (kDut1Day1 - delta_at)), 1e-12);
  EXPECT_NEAR(mid.xp_arcsec, 0.5 * (kXpDay0 + kXpDay1), 1e-12);
  EXPECT_NEAR(mid.yp_arcsec, 0.5 * (kYpDay0 + kYpDay1), 1e-12);

  // Quarter point: check it is genuinely linear, not just symmetric.
  pf::EopValue q;
  ASSERT_TRUE(t.lookup(TaiAtUtc(2020, 6, 1, 21600.0), leap, q));  // 06:00
  EXPECT_NEAR(q.xp_arcsec, kXpDay0 + 0.25 * (kXpDay1 - kXpDay0), 1e-12);
}

TEST(EopTable, ReproducesEndpointsExactly) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pf::EopTable<8> t = QuietTable();
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();

  pf::EopValue v;
  ASSERT_TRUE(t.lookup(TaiAtUtc(2020, 6, 1), leap, v));
  EXPECT_NEAR(v.ut1_minus_tai, kDut1Day0 - 37.0, 1e-9);
  EXPECT_NEAR(v.xp_arcsec, kXpDay0, 1e-12);
  ASSERT_TRUE(t.lookup(TaiAtUtc(2020, 6, 2), leap, v));
  EXPECT_NEAR(v.ut1_minus_tai, kDut1Day1 - 37.0, 1e-9);
  EXPECT_NEAR(v.yp_arcsec, kYpDay1, 1e-12);
}

TEST(EopTable, RejectsEpochsOutsideTheSpanRatherThanExtrapolating) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pf::EopTable<8> t = QuietTable();
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();

  pf::EopValue v;
  v.xp_arcsec = 12345.0;  // sentinel: must survive a failed lookup untouched
  EXPECT_FALSE(t.lookup(TaiAtUtc(2020, 5, 31), leap, v));       // before the span
  EXPECT_FALSE(t.lookup(TaiAtUtc(2020, 6, 3), leap, v));        // after the span
  EXPECT_FALSE(t.lookup(TaiAtUtc(2020, 6, 1, -1.0), leap, v));  // 1 s early
  EXPECT_DOUBLE_EQ(v.xp_arcsec, 12345.0);

  pf::EopTable<8> too_small;  // a single record cannot be interpolated
  ASSERT_TRUE(too_small.addEntry({MjdOf(2020, 6, 1), -0.1, 0.0, 0.0}));
  EXPECT_TRUE(too_small.empty());
  EXPECT_FALSE(too_small.lookup(TaiAtUtc(2020, 6, 1), leap, v));
  EXPECT_DOUBLE_EQ(v.xp_arcsec, 12345.0);
}

/// The regression this table's design exists to prevent. IERS publishes ΔUT1 =
/// UT1 - UTC, which steps by +1 s at a leap second because UTC steps and UT1 does
/// not. Interpolating that raw series smears a ~1 s error (≈465 m of ground
/// track) across the days around the leap. Storing UT1 - TAI cancels the step
/// exactly, because ΔAT steps by the same +1 s at the same instant.
///
/// The 2016-12-31 → 2017-01-01 leap (ΔAT 36 → 37) with real-scale IERS values:
/// ΔUT1 jumps -0.5928 → +0.4064, yet UT1 - TAI drifts only ~0.8 ms/day.
TEST(EopTable, Ut1MinusTaiIsContinuousAcrossALeapSecond) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  ASSERT_EQ(leap.deltaAtForUtcDate(2016, 12, 31), 36);  // the leap really is bracketed
  ASSERT_EQ(leap.deltaAtForUtcDate(2017, 1, 1), 37);

  pf::EopTable<8> t;
  ASSERT_TRUE(t.addEntry({MjdOf(2016, 12, 30), -0.5920, 0.0, 0.0}));
  ASSERT_TRUE(t.addEntry({MjdOf(2016, 12, 31), -0.5928, 0.0, 0.0}));
  ASSERT_TRUE(t.addEntry({MjdOf(2017, 1, 1), +0.4064, 0.0, 0.0}));  // ΔUT1 steps by +1 s
  ASSERT_TRUE(t.addEntry({MjdOf(2017, 1, 2), +0.4056, 0.0, 0.0}));

  // Walk 30 min at a time from 2016-12-30 to 2017-01-02 straight through the
  // leap. UT1 - TAI must never move more than a few ms between samples; the
  // raw-ΔUT1 bug would show a ~1 s excursion somewhere in here.
  const pt::Tai start = TaiAtUtc(2016, 12, 30);
  double prev = 0.0;
  bool have_prev = false;
  int samples = 0;
  for (double dt = 0.0; dt <= 3.0 * 86400.0; dt += 1800.0) {
    pf::EopValue v;
    ASSERT_TRUE(t.lookup(start + pt::Duration::fromSecondsF(dt), leap, v)) << "dt=" << dt;
    if (have_prev) {
      EXPECT_LT(std::abs(v.ut1_minus_tai - prev), 1e-3) << "discontinuity at dt=" << dt;
    }
    // Real UT1 - TAI near this leap; a 1 s smear would blow this bound.
    EXPECT_NEAR(v.ut1_minus_tai, -36.5928, 5e-3) << "dt=" << dt;
    prev = v.ut1_minus_tai;
    have_prev = true;
    ++samples;
  }
  EXPECT_EQ(samples, 145);

  // And the step is cancelled exactly, not merely smoothed: at each entry the
  // stored value is the published ΔUT1 less that date's ΔAT.
  pf::EopValue before;
  pf::EopValue after;
  ASSERT_TRUE(t.lookup(TaiAtUtc(2016, 12, 31), leap, before));
  ASSERT_TRUE(t.lookup(TaiAtUtc(2017, 1, 1), leap, after));
  EXPECT_NEAR(before.ut1_minus_tai, -0.5928 - 36.0, 1e-9);
  EXPECT_NEAR(after.ut1_minus_tai, +0.4064 - 37.0, 1e-9);
  EXPECT_NEAR(after.ut1_minus_tai - before.ut1_minus_tai, -0.0008, 1e-9);  // ~0.8 ms/day drift
}

TEST(EopTable, Ut1FromTaiAppliesTheOffset) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pt::Tai t = TaiAtUtc(2020, 6, 1);
  const pt::Ut1 u = pf::ut1FromTai(t, -37.1770);
  EXPECT_EQ(u.nanosecondsSinceEpoch(), t.nanosecondsSinceEpoch() - 37'177'000'000);
}

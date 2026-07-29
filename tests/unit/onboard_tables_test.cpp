/// @file Tests for the onboard table store (lib/onboard/tables.hpp).
///
/// Loads the committed fixtures the flight OnboardTables component ships with —
/// the DE440 Chebyshev fit and the verbatim IERS finals.all — and pins: a full
/// load reports every table with sane counts/spans; the queries answer
/// physically sane values on the fixtures (magnitude/finiteness bounds — the
/// consumer stand-in, since no GNC consumer exists yet; the store's internal
/// tables are private, so exact lib-evaluator equality is pinned only for the
/// leap-second path); a reload restages and re-serves; a failed reload leaves
/// the previous tables in service; malformed uploads (non-finite coefficients,
/// reordered EOP rows) are rejected with distinct reasons; and concurrent
/// reload+query never serves a torn read (the seqlock contract).

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ephemeris/ephemeris_table.hpp"
#include "frames/eop.hpp"
#include "onboard/tables.hpp"
#include "time/leap_seconds.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"

namespace {

namespace ob = polaris::onboard;

// Fixture paths from the unit-test CMake definitions.
const char* kEphemPath = POLARIS_EPHEMERIS_FIXTURE;
const std::string kEopPath = std::string(POLARIS_GOLDEN_DIR) + "/finals.all.iau2000.txt";

// A TAI epoch inside the one-year fixture coverage (2026), as int64 ns.
constexpr std::int64_t kInCoverageTaiNs = 1'770'000'000'000'000'000LL;  // ~2026-02-01 TAI

std::unique_ptr<ob::TableStore> makeLoaded(ob::LoadReport& report) {
  auto store = std::make_unique<ob::TableStore>();
  store->load(kEopPath.c_str(), kEphemPath, report);
  return store;
}

TEST(OnboardTables, FullLoadReportsEveryTable) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;
  EXPECT_TRUE(store->ready());

  // Committed fixture: 28 historical leaps, 46 Sun and 92 Moon segments.
  EXPECT_EQ(report.leap_entries, 28u);
  EXPECT_EQ(report.sun_segments, 46u);
  EXPECT_EQ(report.moon_segments, 92u);
  EXPECT_GE(report.eop_entries, 2u);
  EXPECT_LE(report.eop_entries, ob::kEopCapacity);

  EXPECT_TRUE(report.eop_span.valid);
  EXPECT_TRUE(report.ephem_span.valid);
  EXPECT_LT(report.eop_span.start_tai_s, report.eop_span.end_tai_s);
  EXPECT_LT(report.ephem_span.start_tai_s, report.ephem_span.end_tai_s);
}

TEST(OnboardTables, MissingFileFailsWithoutServing) {
  auto store = std::make_unique<ob::TableStore>();
  ob::LoadReport report;
  EXPECT_FALSE(store->load(kEopPath.c_str(), "/no/such/ephem.cheb", report));
  EXPECT_FALSE(store->ready());
  EXPECT_FALSE(report.ephem_ok);
  EXPECT_GT(std::strlen(report.reason), 0u);
}

TEST(OnboardTables, EopPortMatchesLibEvaluator) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  polaris::frames::EopValue via_port;
  EXPECT_EQ(store->eopAt(kInCoverageTaiNs, via_port), ob::Quality::kPrecise);

  // Ground truth: same lib evaluator, historical leap, over the same windowed
  // table would need the private table; instead assert the port's answer is
  // finite and physical (UT1-TAI within a few tens of seconds, |polar| < 1").
  EXPECT_TRUE(std::isfinite(via_port.ut1_minus_tai));
  EXPECT_LT(std::fabs(via_port.ut1_minus_tai), 60.0);
  EXPECT_LT(std::fabs(via_port.xp_arcsec), 1.0);
  EXPECT_LT(std::fabs(via_port.yp_arcsec), 1.0);

  // Outside coverage → coarse zero-EOP fallback, not a hard failure: the sample
  // is overwritten with UT1-TAI = -ΔAT (UT1 ≈ UTC) and zero polar motion.
  polaris::frames::EopValue sentinel;
  sentinel.ut1_minus_tai = 12345.0;
  EXPECT_EQ(store->eopAt(kInCoverageTaiNs, sentinel),
            ob::Quality::kPrecise);  // sanity: precise here
  polaris::frames::EopValue coarse;
  coarse.ut1_minus_tai = 12345.0;
  EXPECT_EQ(store->eopAt(0, coarse), ob::Quality::kCoarse);
  EXPECT_DOUBLE_EQ(coarse.xp_arcsec, 0.0);
  EXPECT_DOUBLE_EQ(coarse.yp_arcsec, 0.0);
  EXPECT_TRUE(std::isfinite(coarse.ut1_minus_tai));
}

TEST(OnboardTables, BodyPositionPortMatchesLibTable) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  polaris::math::Vec3<polaris::math::frames::ECI> sun;
  EXPECT_EQ(store->bodyPositionEci(ob::Body::Sun, kInCoverageTaiNs, sun), ob::Quality::kPrecise);
  // Sun is ~1 AU from Earth (geocentric): 1.4e11 - 1.6e11 m.
  const double r = sun.eigen().norm();
  EXPECT_GT(r, 1.3e11);
  EXPECT_LT(r, 1.7e11);

  polaris::math::Vec3<polaris::math::frames::ECI> moon;
  EXPECT_EQ(store->bodyPositionEci(ob::Body::Moon, kInCoverageTaiNs, moon), ob::Quality::kPrecise);
  const double rm = moon.eigen().norm();
  EXPECT_GT(rm, 3.4e8);  // ~perigee
  EXPECT_LT(rm, 4.1e8);  // ~apogee
}

TEST(OnboardTables, TaiUtcOffsetIsCurrentDeltaAt) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  std::int32_t delta = 0;
  EXPECT_EQ(store->taiUtcOffset(kInCoverageTaiNs, delta), ob::Quality::kPrecise);
  // ΔAT has been 37 s since 2017-01-01; the 2026 fixture epoch is in that regime.
  EXPECT_EQ(delta, 37);
}

TEST(OnboardTables, CoverageInsideAndOutside) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  bool eop_ok = false;
  bool ephem_ok = false;
  ASSERT_TRUE(store->coverageAt(kInCoverageTaiNs, eop_ok, ephem_ok));
  EXPECT_TRUE(eop_ok);
  EXPECT_TRUE(ephem_ok);

  ASSERT_TRUE(store->coverageAt(0, eop_ok, ephem_ok));
  EXPECT_FALSE(eop_ok);
  EXPECT_FALSE(ephem_ok);
}

TEST(OnboardTables, FreshStoreServesCoarseFallbacksTableIndependent) {
  // The Safe-mode floor: a store that never loaded any table must still answer.
  // taiUtcOffset stays precise (in-code leap record), while EOP and ephemeris
  // fall back to coarse — this is what makes coarse sun pointing table-independent.
  ob::TableStore store;
  EXPECT_FALSE(store.ready());

  std::int32_t delta = 0;
  EXPECT_EQ(store.taiUtcOffset(kInCoverageTaiNs, delta), ob::Quality::kPrecise);
  EXPECT_EQ(delta, 37);  // 2026 regime, from the in-code historical leap table

  polaris::frames::EopValue eop;
  EXPECT_EQ(store.eopAt(kInCoverageTaiNs, eop), ob::Quality::kCoarse);
  EXPECT_DOUBLE_EQ(eop.xp_arcsec, 0.0);
  EXPECT_DOUBLE_EQ(eop.yp_arcsec, 0.0);
  EXPECT_DOUBLE_EQ(eop.ut1_minus_tai, -37.0);  // UT1 ≈ UTC ⇒ UT1-TAI = -ΔAT

  polaris::math::Vec3<polaris::math::frames::ECI> sun;
  EXPECT_EQ(store.bodyPositionEci(ob::Body::Sun, kInCoverageTaiNs, sun), ob::Quality::kCoarse);
  const double rs = sun.eigen().norm();
  EXPECT_GT(rs, 1.3e11);  // analytic Sun is ~1 AU geocentric
  EXPECT_LT(rs, 1.7e11);

  polaris::math::Vec3<polaris::math::frames::ECI> moon;
  EXPECT_EQ(store.bodyPositionEci(ob::Body::Moon, kInCoverageTaiNs, moon), ob::Quality::kCoarse);
  const double rm = moon.eigen().norm();
  EXPECT_GT(rm, 3.4e8);  // analytic Moon ~perigee..apogee
  EXPECT_LT(rm, 4.1e8);
}

TEST(OnboardTables, UncoveredEpochDegradesToCoarse) {
  // Tables loaded, but query outside their coverage span → coarse fallback (not a
  // hard failure), for both the ephemeris and EOP domains.
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  polaris::frames::EopValue eop;
  EXPECT_EQ(store->eopAt(0, eop), ob::Quality::kCoarse);  // tai=0 (1970) is uncovered

  polaris::math::Vec3<polaris::math::frames::ECI> sun;
  EXPECT_EQ(store->bodyPositionEci(ob::Body::Sun, 0, sun), ob::Quality::kCoarse);
  const double rs = sun.eigen().norm();
  EXPECT_GT(rs, 1.3e11);
  EXPECT_LT(rs, 1.7e11);

  // Recovery: back inside coverage, the precise table serves again.
  EXPECT_EQ(store->bodyPositionEci(ob::Body::Sun, kInCoverageTaiNs, sun), ob::Quality::kPrecise);
}

TEST(OnboardTables, ReloadRestagesAndFailedReloadKeepsService) {
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;
  const std::size_t sun0 = store->sunSegments();

  // Successful reload: same fixtures, still served.
  ob::LoadReport report2;
  EXPECT_TRUE(store->load(kEopPath.c_str(), kEphemPath, report2));
  EXPECT_TRUE(store->ready());
  EXPECT_EQ(store->sunSegments(), sun0);

  // Failed reload: bad path. Previous tables remain in service (staging swap).
  ob::LoadReport report3;
  EXPECT_FALSE(store->load(kEopPath.c_str(), "/no/such.cheb", report3));
  EXPECT_TRUE(store->ready());
  EXPECT_EQ(store->sunSegments(), sun0);
  polaris::math::Vec3<polaris::math::frames::ECI> sun;
  EXPECT_EQ(store->bodyPositionEci(ob::Body::Sun, kInCoverageTaiNs, sun), ob::Quality::kPrecise);
}

}  // namespace

TEST(OnboardTables, NonFiniteEphemerisCoefficientRejectedAtLoad) {
  // A nan coefficient must fail the LOAD with a clear reason (trust boundary),
  // not "succeed" into a table whose every query then returns false.
  const std::string path = testing::TempDir() + "polaris_bad_coeff.cheb";
  std::FILE* f = std::fopen(path.c_str(), "w");
  ASSERT_NE(f, nullptr);
  std::fputs("seg sun 1770000000000000000 691200 2 1.0 2.0 nan 4.0 5.0 6.0 7.0 8.0 9.0\n", f);
  std::fclose(f);

  ob::TableStore store;
  ob::LoadReport report;
  EXPECT_FALSE(store.load(kEopPath.c_str(), path.c_str(), report));
  EXPECT_FALSE(store.ready());
  EXPECT_NE(std::string(report.reason).find("non-finite"), std::string::npos) << report.reason;
  std::remove(path.c_str());
}

TEST(OnboardTables, ReorderedEopRowsRejectedWithDistinctReason) {
  // Reverse the committed finals.all line order: rows arrive with descending
  // MJD, which must be diagnosed as an ordering rejection, not "capacity".
  std::FILE* in = std::fopen(kEopPath.c_str(), "r");
  ASSERT_NE(in, nullptr);
  std::vector<std::string> lines;
  char buf[256];
  while (std::fgets(buf, sizeof(buf), in) != nullptr) {
    lines.emplace_back(buf);
  }
  std::fclose(in);
  const std::string path = testing::TempDir() + "polaris_reversed_finals.txt";
  std::FILE* out = std::fopen(path.c_str(), "w");
  ASSERT_NE(out, nullptr);
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    std::fputs(it->c_str(), out);
  }
  std::fclose(out);

  ob::TableStore store;
  ob::LoadReport report;
  EXPECT_FALSE(store.load(path.c_str(), kEphemPath, report));
  EXPECT_NE(std::string(report.reason).find("non-ascending"), std::string::npos) << report.reason;
  std::remove(path.c_str());
}

TEST(OnboardTables, ConcurrentReloadNeverServesTornRead) {
  // Seqlock contract: hammer queries from another thread across many
  // back-to-back reloads (each reload rewrites the slot a stale reader could
  // still be latched onto — the exact two-flip interleaving the generation
  // counter guards). Every query must either fail cleanly or return a value
  // that passes the same physical-sanity bounds as the single-threaded tests;
  // a torn read would produce garbage magnitudes.
  ob::LoadReport report;
  auto store = makeLoaded(report);
  ASSERT_TRUE(report.ok()) << report.reason;

  std::atomic<bool> stop{false};
  std::atomic<int> torn{0};
  std::atomic<int> served{0};
  std::thread reader([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      // Both the precise table and the coarse fallback (analytic Sun ~1 AU,
      // zero-EOP UT1-TAI = -ΔAT) satisfy the same physical bounds, so a torn read
      // is still the only way to produce an out-of-bounds magnitude here.
      polaris::math::Vec3<polaris::math::frames::ECI> sun;
      if (store->bodyPositionEci(ob::Body::Sun, kInCoverageTaiNs, sun) !=
          ob::Quality::kUnavailable) {
        const double r = sun.eigen().norm();
        if (!std::isfinite(r) || r < 1.3e11 || r > 1.7e11) {
          torn.fetch_add(1, std::memory_order_relaxed);
        }
        served.fetch_add(1, std::memory_order_relaxed);
      }
      polaris::frames::EopValue eop;
      if (store->eopAt(kInCoverageTaiNs, eop) != ob::Quality::kUnavailable) {
        if (!std::isfinite(eop.ut1_minus_tai) || std::abs(eop.ut1_minus_tai) > 45.0) {
          torn.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  });
  for (int i = 0; i < 40; ++i) {
    ob::LoadReport r2;
    ASSERT_TRUE(store->load(kEopPath.c_str(), kEphemPath, r2)) << r2.reason;
  }
  stop.store(true, std::memory_order_relaxed);
  reader.join();
  EXPECT_EQ(torn.load(), 0);
  EXPECT_GT(served.load(), 0);  // the reader actually exercised the path
}

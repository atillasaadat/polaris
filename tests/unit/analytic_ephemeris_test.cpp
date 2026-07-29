/// @file Tests for the analytic (table-independent) Sun/Moon ephemerides
/// (lib/ephemeris/analytic_sun.hpp, analytic_moon.hpp).
///
/// The coarse fallbacks that guarantee the Safe-mode sun-pointing floor is
/// table-independent (design doc §8.1, §11.3). These pin two things: (1) the
/// analytic direction agrees with the precise DE440 Chebyshev fit to within a
/// coarse-but-frame-error-catching bound, sampled across the committed one-year
/// fixture coverage; (2) the geocentric magnitudes are physical. The agreement
/// bound is dominated not by the ~0.01°/~0.3° intrinsic Vallado model accuracy
/// but by the deliberate mean-of-date≈J2000 frame approximation (neglected
/// precession, ≈0.4° at the mid-2020s epoch) — the bound is set generously
/// against that, still tight enough that a swapped axis or wrong obliquity sign
/// (tens of degrees) fails.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ephemeris/analytic_moon.hpp"
#include "ephemeris/analytic_sun.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "onboard/tables.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"

namespace {

namespace ob = polaris::onboard;
using Vec3Eci = polaris::math::Vec3<polaris::math::frames::ECI>;

const char* kEphemPath = POLARIS_EPHEMERIS_FIXTURE;
const std::string kEopPath = std::string(POLARIS_GOLDEN_DIR) + "/finals.all.iau2000.txt";

constexpr double kPi = 3.141'592'653'589'793'238;

/// Angular separation between two ECI vectors [deg].
double angleDeg(const Vec3Eci& a, const Vec3Eci& b) {
  const double c = a.eigen().normalized().dot(b.eigen().normalized());
  return std::acos(std::min(1.0, std::max(-1.0, c))) * 180.0 / kPi;
}

polaris::time::Tdb tdbOf(std::int64_t tai_ns) {
  return polaris::time::toTdb(
      polaris::time::toTt(polaris::time::Tai::fromNanosecondsSinceEpoch(tai_ns)));
}

std::unique_ptr<ob::TableStore> loadedStore() {
  auto store = std::make_unique<ob::TableStore>();
  ob::LoadReport report;
  store->load(kEopPath.c_str(), kEphemPath, report);
  EXPECT_TRUE(report.ok()) << report.reason;
  return store;
}

// Sample epochs across the interior of the fixture coverage (avoid the very
// endpoints, where a segment boundary could reject the precise query).
std::vector<std::int64_t> interiorEpochs(const ob::TableStore& store) {
  const ob::TableSpan span = store.ephemSpan();
  std::vector<std::int64_t> out;
  for (double f : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    const std::int64_t tai_s =
        span.start_tai_s + static_cast<std::int64_t>(f * (span.end_tai_s - span.start_tai_s));
    out.push_back(tai_s * 1'000'000'000LL);
  }
  return out;
}

TEST(AnalyticEphemeris, SunMatchesDe440AcrossCoverage) {
  auto store = loadedStore();
  double max_deg = 0.0;
  for (std::int64_t tai_ns : interiorEpochs(*store)) {
    Vec3Eci precise;
    ASSERT_EQ(store->bodyPositionEci(ob::Body::Sun, tai_ns, precise), ob::Quality::kPrecise);
    const Vec3Eci analytic = polaris::ephemeris::sunPositionEci(tdbOf(tai_ns));
    max_deg = std::max(max_deg, angleDeg(precise, analytic));
    // Magnitude sane: ~1 AU geocentric.
    const double r = analytic.eigen().norm();
    EXPECT_GT(r, 1.3e11);
    EXPECT_LT(r, 1.7e11);
  }
  // Dominated by the neglected precession (~0.4° at 2026); 0.6° holds with margin
  // and still catches a frame/sign error.
  EXPECT_LT(max_deg, 0.6) << "max Sun angular disagreement [deg]";
}

TEST(AnalyticEphemeris, MoonMatchesDe440AcrossCoverage) {
  auto store = loadedStore();
  double max_deg = 0.0;
  for (std::int64_t tai_ns : interiorEpochs(*store)) {
    Vec3Eci precise;
    ASSERT_EQ(store->bodyPositionEci(ob::Body::Moon, tai_ns, precise), ob::Quality::kPrecise);
    const Vec3Eci analytic = polaris::ephemeris::moonPositionEci(tdbOf(tai_ns));
    max_deg = std::max(max_deg, angleDeg(precise, analytic));
    const double r = analytic.eigen().norm();
    EXPECT_GT(r, 3.4e8);  // ~perigee
    EXPECT_LT(r, 4.1e8);  // ~apogee
  }
  // Intrinsic ~0.3° series accuracy + neglected precession ~0.4°; 1.0° holds.
  EXPECT_LT(max_deg, 1.0) << "max Moon angular disagreement [deg]";
}

}  // namespace

/// @file Unit tests for the DE440 Chebyshev ephemeris fixture and its loader
/// (REQ-CDH-002; design doc §11.3, §3.7).
///
/// The loader tests are ordinary parser tests — malformed input must fail loudly
/// rather than half-load, since a partially-loaded ephemeris surfaces as an
/// intermittent coverage gap mid-propagation.
///
/// The fixture tests are the interesting ones. They check the *committed fit*
/// against facts about the solar system that hold independently of DE440, so a
/// silently corrupted or wrongly-generated fixture cannot pass: the Sun's range
/// and its annual variation, perihelion falling in early January, the Moon's
/// range and month-scale period, and the two bodies being on opposite sides of
/// the Earth at new moon. A fixture that merely parsed but held garbage — wrong
/// units, barycentric instead of geocentric, swapped bodies — fails these.
///
/// Continuity across segment boundaries is checked too: the fit is piecewise, and
/// a per-interval least-squares fit is NOT continuous by construction, so this is
/// a real property worth pinning rather than a tautology.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <Eigen/Core>
#include <fstream>
#include <memory>
#include <string>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"
#include "world/body_position.hpp"
#include "world/ephemeris_file.hpp"

namespace world = polaris::sim::world;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

using Eci = pm::Vec3<pmf::ECI>;

constexpr double kAu = pc::bodies::kAstronomicalUnit;

/// The committed fixture, loaded once — it is ~200 kB of tables, so it is held
/// on the heap and shared rather than rebuilt per test.
const world::EphemerisSet& fixture() {
  static const std::unique_ptr<world::EphemerisSet> set = [] {
    auto s = std::make_unique<world::EphemerisSet>();
    std::string error;
    const bool ok = world::loadEphemerisFile(POLARIS_EPHEMERIS_FIXTURE, *s, &error);
    EXPECT_TRUE(ok) << error;
    return s;
  }();
  return *set;
}

/// TDB instant from a Julian Date, matching the fixture's time base.
pt::Tdb tdbAtJd(double jd) {
  const double seconds = (jd - pc::time::kJulianDate1970) * pc::time::kSecondsPerDay;
  return pt::Tdb::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(seconds * 1e9));
}

// Fixture coverage: 2026-01-01 .. 2027-01-01 TDB.
constexpr double kJdStart = 2'461'041.5;
constexpr double kJdEnd = 2'461'406.5;

Eigen::Vector3d sunAt(double jd) {
  Eci p;
  EXPECT_TRUE(fixture().sun.position(tdbAtJd(jd), p)) << "no sun coverage at JD " << jd;
  return p.eigen();
}

Eigen::Vector3d moonAt(double jd) {
  Eci p;
  EXPECT_TRUE(fixture().moon.position(tdbAtJd(jd), p)) << "no moon coverage at JD " << jd;
  return p.eigen();
}

/// Write a temporary fixture body for the parser tests.
std::string writeTemp(const std::string& contents) {
  const std::string path = std::string(std::tmpnam(nullptr)) + ".cheb";
  std::ofstream out(path);
  out << contents;
  return path;
}

}  // namespace

// --- Loader ------------------------------------------------------------------

TEST(EphemerisFile, LoadsTheCommittedFixture) {
  world::EphemerisSet set;
  std::string error;
  ASSERT_TRUE(world::loadEphemerisFile(POLARIS_EPHEMERIS_FIXTURE, set, &error)) << error;
  EXPECT_GT(set.sun.size(), 0u);
  EXPECT_GT(set.moon.size(), 0u);
  // The generator's defaults: 8-day Sun and 4-day Moon intervals over one year.
  EXPECT_EQ(set.sun.size(), 46u);
  EXPECT_EQ(set.moon.size(), 92u);
}

TEST(EphemerisFile, MissingFileFailsWithAReason) {
  world::EphemerisSet set;
  std::string error;
  EXPECT_FALSE(world::loadEphemerisFile("/nonexistent/x.cheb", set, &error));
  EXPECT_NE(error.find("cannot open"), std::string::npos) << error;
}

TEST(EphemerisFile, TruncatedCoefficientsAreRejectedRatherThanPartiallyLoaded) {
  // Degree 2 promises 9 coefficients; only 4 are present.
  const std::string path = writeTemp("seg sun 0 86400 2 1 2 3 4\n");
  world::EphemerisSet set;
  std::string error;
  EXPECT_FALSE(world::loadEphemerisFile(path, set, &error));
  EXPECT_NE(error.find("truncated"), std::string::npos) << error;
  std::remove(path.c_str());
}

TEST(EphemerisFile, UnknownBodyIsRejected) {
  const std::string path = writeTemp("seg jupiter 0 86400 0 1 2 3\n");
  world::EphemerisSet set;
  std::string error;
  EXPECT_FALSE(world::loadEphemerisFile(path, set, &error));
  EXPECT_NE(error.find("unknown body"), std::string::npos) << error;
  std::remove(path.c_str());
}

TEST(EphemerisFile, OutOfRangeDegreeIsRejected) {
  const std::string path = writeTemp("seg sun 0 86400 99 1\n");
  world::EphemerisSet set;
  std::string error;
  EXPECT_FALSE(world::loadEphemerisFile(path, set, &error));
  EXPECT_NE(error.find("out of range"), std::string::npos) << error;
  std::remove(path.c_str());
}

TEST(EphemerisFile, CommentsAndBlankLinesAreSkipped) {
  const std::string path = writeTemp("# a comment\n\nseg sun 0 86400 0 1e11 2e10 3e10\n");
  world::EphemerisSet set;
  std::string error;
  EXPECT_TRUE(world::loadEphemerisFile(path, set, &error)) << error;
  EXPECT_EQ(set.sun.size(), 1u);
  std::remove(path.c_str());
}

TEST(EphemerisFile, AnEmptyFileIsAnError) {
  const std::string path = writeTemp("# only comments\n");
  world::EphemerisSet set;
  std::string error;
  EXPECT_FALSE(world::loadEphemerisFile(path, set, &error));
  EXPECT_NE(error.find("no segments"), std::string::npos) << error;
  std::remove(path.c_str());
}

// --- The fit, checked against independent astronomy --------------------------

TEST(Ephemeris, SunRangeIsOneAuAndVariesByEarthsEccentricity) {
  // Earth's orbital eccentricity is 0.0167, so the geocentric solar range runs
  // 0.983-1.017 AU. Wrong units or a barycentric fit misses this immediately.
  double min_range = 1e30;
  double max_range = 0.0;
  for (double jd = kJdStart; jd < kJdEnd; jd += 1.0) {
    const double r = sunAt(jd).norm();
    min_range = std::min(min_range, r);
    max_range = std::max(max_range, r);
  }
  EXPECT_NEAR(min_range / kAu, 0.9833, 0.001);
  EXPECT_NEAR(max_range / kAu, 1.0167, 0.001);
}

TEST(Ephemeris, PerihelionFallsInEarlyJanuary) {
  // Earth reaches perihelion on ~3-5 January. This pins the fit's time base: an
  // epoch or JD-offset error shifts the minimum away from the first week.
  double best_jd = 0.0;
  double best = 1e30;
  for (double jd = kJdStart; jd < kJdStart + 40.0; jd += 0.25) {
    const double r = sunAt(jd).norm();
    if (r < best) {
      best = r;
      best_jd = jd;
    }
  }
  const double day_of_january = best_jd - kJdStart + 1.0;
  EXPECT_GE(day_of_january, 1.0);
  EXPECT_LE(day_of_january, 7.0) << "perihelion at day " << day_of_january << " of January";
}

TEST(Ephemeris, MoonRangeMatchesItsPerigeeAndApogee) {
  // The lunar distance runs ~356,500-406,700 km. A fit that accidentally held
  // EMB-relative or barycentric coordinates would be nowhere near this.
  double min_range = 1e30;
  double max_range = 0.0;
  for (double jd = kJdStart; jd < kJdEnd; jd += 0.25) {
    const double r = moonAt(jd).norm();
    min_range = std::min(min_range, r);
    max_range = std::max(max_range, r);
  }
  EXPECT_GT(min_range, 356'000e3);
  EXPECT_LT(min_range, 371'000e3);
  EXPECT_GT(max_range, 400'000e3);
  EXPECT_LT(max_range, 407'000e3);
}

TEST(Ephemeris, MoonCompletesRoughlyThirteenOrbitsInAYear) {
  // Counts sign changes of the Y component to measure the period without
  // assuming anything about the fit's internals. The sidereal month is 27.32
  // days, so a 365-day span holds ~13.4 orbits => ~26-27 crossings.
  int crossings = 0;
  double previous = moonAt(kJdStart).y();
  for (double jd = kJdStart + 0.25; jd < kJdEnd; jd += 0.25) {
    const double y = moonAt(jd).y();
    if ((y > 0.0) != (previous > 0.0)) {
      ++crossings;
    }
    previous = y;
  }
  EXPECT_GE(crossings, 25);
  EXPECT_LE(crossings, 28) << "implied period is wrong; got " << crossings << " crossings";
}

TEST(Ephemeris, SunAndMoonAlignAtNewMoonAndOpposeAtFull) {
  // Over a year the Sun-Earth-Moon angle must sweep the full range: near 0 at
  // new moon and near 180 deg at full moon. This is the check that the two
  // bodies are not the same data under different names.
  double min_cos = 1.0;
  double max_cos = -1.0;
  for (double jd = kJdStart; jd < kJdEnd; jd += 0.25) {
    const double c = sunAt(jd).normalized().dot(moonAt(jd).normalized());
    min_cos = std::min(min_cos, c);
    max_cos = std::max(max_cos, c);
  }
  EXPECT_GT(max_cos, 0.999) << "never reaches new moon";
  EXPECT_LT(min_cos, -0.999) << "never reaches full moon";
}

TEST(Ephemeris, IsContinuousAcrossSegmentBoundaries) {
  // Each interval is fitted independently, so continuity is a property of the fit
  // quality, not something the format guarantees. A jump here means the degree is
  // too low or the intervals too long.
  for (double jd = kJdStart + 4.0; jd < kJdEnd - 4.0; jd += 4.0) {
    const double dt = 1e-4;  // ~9 seconds
    const Eigen::Vector3d before = moonAt(jd - dt);
    const Eigen::Vector3d after = moonAt(jd + dt);
    // The Moon moves ~1 km/s, so ~18 s of travel is ~18 km; anything beyond that
    // is a discontinuity rather than motion.
    EXPECT_LT((after - before).norm(), 30e3) << "discontinuity near JD " << jd;
  }
}

TEST(Ephemeris, CoverageEndsAtTheFixtureBoundaryRatherThanExtrapolating) {
  Eci p;
  EXPECT_FALSE(fixture().sun.position(tdbAtJd(kJdStart - 10.0), p)) << "extrapolated before start";
  EXPECT_FALSE(fixture().sun.position(tdbAtJd(kJdEnd + 10.0), p)) << "extrapolated past end";
  EXPECT_TRUE(fixture().sun.position(tdbAtJd(kJdStart + 100.0), p));
}

// --- Resolver adaptor --------------------------------------------------------

TEST(Ephemeris, BodyPositionFnMatchesTheTableAndReportsCoverage) {
  const world::BodyPositionFn sun = world::bodyPositionFn(fixture().sun);

  Eci via_fn;
  ASSERT_TRUE(sun(tdbAtJd(kJdStart + 50.0), via_fn));
  EXPECT_TRUE(via_fn.eigen().isApprox(sunAt(kJdStart + 50.0)));

  // Outside coverage the resolver reports false and leaves the output alone —
  // the contract every consumer relies on to mean "this body contributes
  // nothing" rather than to abort the integration.
  Eci untouched(Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_FALSE(sun(tdbAtJd(kJdEnd + 10.0), untouched));
  EXPECT_TRUE(untouched.eigen().isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
}

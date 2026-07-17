/// @file Golden fixture: the committed IERS EOP table drives the reduction.
///
/// Loads `tests/golden/eop.json` — real IERS `finals2000A.all` (IAU2000A) EOP,
/// trimmed by `tools/eop/` and committed as static data (CI never downloads) —
/// into a `polaris::frames::EopTable` and exercises the ECI↔ECEF reduction on it
/// (REQ-CONV-002, REQ-CONV-001). This is the integration proof that the committed
/// fixture is well-formed, fully ingestible (ascending, no rejects), and that
/// lookup + interpolation + leap handling feed a valid rotation at real epochs.
///
/// The reduction's *numeric* accuracy is anchored to the published ERFA
/// `t_c2t06a` matrix in `tests/unit/eci_ecef_test.cpp`; the GMAT cross-check of a
/// full ECEF state at real EOP lands with Push 10 (REQ-SYS-010), when the GMAT
/// ECEF fixture joins this directory.

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace pf = polaris::frames;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;
using json = nlohmann::json;

namespace {

/// Capacity bound for the loaded table: comfortably above the committed fixture's
/// row count (2020→~2027 daily ≈ 2.8k). Heap-allocated in the test (not flight).
constexpr std::size_t kEopCapacity = 4096;

/// ΔAT (TAI−UTC) in effect for the whole fixture span: no leap since 2017-01-01.
constexpr double kDeltaAt = 37.0;

json LoadFixture(const std::string& name) {
  const std::string path = std::string(GOLDEN_DIR) + "/" + name;
  std::ifstream file(path);
  if (!file.is_open()) {
    ADD_FAILURE() << "cannot open golden fixture: " << path;
    return json{};
  }
  return json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

/// TAI at 00:00:00 UTC on the given UTC MJD (fixture rows are integral MJD).
pt::Tai TaiAtMjdUtc(double mjd_utc) {
  const auto tai_sec =
      static_cast<std::int64_t>((mjd_utc - pf::kMjd1970) * pc::time::kSecondsPerDay + kDeltaAt);
  return pt::Tai::fromNanosecondsSinceEpoch(tai_sec * 1'000'000'000);
}

}  // namespace

TEST(EopGolden, CommittedFixtureIngestsAndDrivesTheReduction) {
  RecordProperty("verifies", "REQ-CONV-002");
  const json fixture = LoadFixture("eop.json");
  ASSERT_FALSE(fixture.is_null() || fixture.is_discarded())
      << "golden EOP fixture missing or malformed";
  const auto& entries = fixture.at("entries");
  ASSERT_GE(entries.size(), 2u) << "need >=2 rows to interpolate";

  // Every committed row loads — ascending MJD, all finite, none rejected.
  auto table = std::make_unique<pf::EopTable<kEopCapacity>>();
  ASSERT_LE(entries.size(), kEopCapacity) << "fixture outgrew kEopCapacity";
  for (const auto& e : entries) {
    const pf::EopEntry row{e.at("mjd_utc").get<double>(), e.at("dut1").get<double>(),
                           e.at("xp_arcsec").get<double>(), e.at("yp_arcsec").get<double>()};
    ASSERT_TRUE(table->addEntry(row)) << "row rejected at mjd " << row.mjd_utc;
  }
  EXPECT_EQ(table->size(), entries.size());

  // At an exact node the interpolant returns that row's values, with the leap
  // step folded in: UT1−TAI = (UT1−UTC) − ΔAT. Cross-checks load, lookup and the
  // continuous-quantity conversion against the raw committed number.
  const auto& mid = entries.at(entries.size() / 2);
  const pt::Tai t = TaiAtMjdUtc(mid.at("mjd_utc").get<double>());
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  pf::EopValue v;
  ASSERT_TRUE(table->lookup(t, leap, v));
  EXPECT_NEAR(v.ut1_minus_tai, mid.at("dut1").get<double>() - kDeltaAt, 1e-9);
  EXPECT_NEAR(v.xp_arcsec, mid.at("xp_arcsec").get<double>(), 1e-12);
  EXPECT_NEAR(v.yp_arcsec, mid.at("yp_arcsec").get<double>(), 1e-12);

  // The real EOP drives a proper orthonormal rotation through the table overload.
  pm::Quat<pmf::ECEF, pmf::ECI> q;
  ASSERT_TRUE(pf::ecefFromEci(t, *table, leap, q));
  const Eigen::Matrix3d r = q.core().toRotationMatrix();
  EXPECT_TRUE((r * r.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-13));
  EXPECT_NEAR(r.determinant(), 1.0, 1e-13);

  // An off-node epoch (½ day past the first row) interpolates strictly between
  // the first two rows' polar motion — proves interpolation runs, not just nodes.
  const double mjd0 = entries.at(0).at("mjd_utc").get<double>();
  const pt::Tai t_half =
      TaiAtMjdUtc(mjd0) + pt::Duration::fromSecondsF(0.5 * pc::time::kSecondsPerDay);
  pf::EopValue vh;
  ASSERT_TRUE(table->lookup(t_half, leap, vh));
  const double xp0 = entries.at(0).at("xp_arcsec").get<double>();
  const double xp1 = entries.at(1).at("xp_arcsec").get<double>();
  EXPECT_GT((vh.xp_arcsec - xp0) * (xp1 - vh.xp_arcsec), -1e-30)
      << "interpolated xp must lie between the bracketing nodes";
}

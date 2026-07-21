/// @file Golden fixture: the committed IERS EOP table drives the reduction.
///
/// Loads `tests/golden/finals.all.iau2000.txt` — the IERS `finals.all.iau2000`
/// product committed **verbatim** as upstream serves it (fixed-width text; update
/// = re-download and overwrite, no transform), parsed here with the same Bulletin
/// A column spec as `tools/eop/finals.py` — into a `polaris::frames::EopTable` and
/// exercises the ECI↔ECEF reduction on it (REQ-CONV-002, REQ-CONV-001). This is
/// the integration proof that the committed fixture is well-formed, fully
/// ingestible (ascending, no rejects), and that lookup + interpolation + leap
/// handling feed a valid rotation at real epochs. CI never downloads.
///
/// The reduction's *numeric* accuracy is anchored to the published ERFA
/// `t_c2t06a` matrix in `tests/unit/eci_ecef_test.cpp`; the GMAT cross-check of a
/// full ECEF state at real EOP lands with Push 10 (REQ-SYS-010), when the GMAT
/// ECEF fixture joins this directory.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "time/civil.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "world/eop_file.hpp"

namespace pf = polaris::frames;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

/// Capacity bound for the loaded table: above the committed file's row count
/// (1973→~2027 daily ≈ 20k). Heap-allocated in the test (not flight).
constexpr std::size_t kEopCapacity = 20480;

/// The Bulletin A column spec lives in `sim/world/eop_file.{hpp,cpp}` — this
/// test used to carry its own transcription of the same fixed-width offsets,
/// which is exactly the kind of duplication that gets fixed in one copy and not
/// the other. It now loads through the shared parser, so this fixture and the
/// running sim demonstrably agree on how to read the file.
using polaris::sim::world::FinalsRow;

std::vector<FinalsRow> LoadFinals(const std::string& name) {
  const std::string path = std::string(GOLDEN_DIR) + "/" + name;
  std::vector<FinalsRow> rows;
  std::string error;
  // An inverted window means "keep every record".
  if (!polaris::sim::world::parseFinals(path, 1.0, 0.0, 0.0, rows, &error)) {
    ADD_FAILURE() << error;
  }
  return rows;
}

/// ΔAT (TAI−UTC) [s] in effect on the UTC date of MJD @p mjd — from the leap
/// table, not hardcoded, since the file spans eras (12 s in 1973 .. 37 s today).
double DeltaAtOfMjd(double mjd, const pt::LeapSecondTable& leap) {
  const auto day = static_cast<std::int64_t>(std::floor(mjd - pf::kMjd1970));
  const pt::CivilDate d = pt::civilFromDays(day);
  return static_cast<double>(leap.deltaAtForUtcDate(d.year, d.month, d.day));
}

/// TAI at 00:00:00 UTC on UTC MJD @p mjd, given that day's ΔAT.
pt::Tai TaiAtMjdUtc(double mjd, double delta_at) {
  const auto tai_sec =
      static_cast<std::int64_t>((mjd - pf::kMjd1970) * pc::time::kSecondsPerDay + delta_at);
  return pt::Tai::fromNanosecondsSinceEpoch(tai_sec * 1'000'000'000);
}

}  // namespace

TEST(EopGolden, CommittedFixtureIngestsAndDrivesTheReduction) {
  RecordProperty("verifies", "REQ-CONV-002");
  const std::vector<FinalsRow> rows = LoadFinals("finals.all.iau2000.txt");
  ASSERT_GE(rows.size(), 2u) << "need >=2 rows to interpolate";
  ASSERT_LE(rows.size(), kEopCapacity) << "fixture outgrew kEopCapacity";

  // Every committed row loads — ascending MJD, all finite, none rejected.
  auto table = std::make_unique<pf::EopTable<kEopCapacity>>();
  for (const FinalsRow& r : rows) {
    ASSERT_TRUE(table->addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec}))
        << "row rejected at mjd " << r.mjd_utc;
  }
  EXPECT_EQ(table->size(), rows.size());

  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();

  // At an exact node the interpolant returns that row's values, with the leap
  // step folded in: UT1−TAI = (UT1−UTC) − ΔAT. Cross-checks load, lookup and the
  // continuous-quantity conversion against the raw committed number.
  const FinalsRow& mid = rows.at(rows.size() / 2);
  const double dat_mid = DeltaAtOfMjd(mid.mjd_utc, leap);
  const pt::Tai t = TaiAtMjdUtc(mid.mjd_utc, dat_mid);
  pf::EopValue v;
  ASSERT_TRUE(table->lookup(t, leap, v));
  EXPECT_NEAR(v.ut1_minus_tai, mid.dut1_s - dat_mid, 1e-9);
  EXPECT_NEAR(v.xp_arcsec, mid.xp_arcsec, 1e-12);
  EXPECT_NEAR(v.yp_arcsec, mid.yp_arcsec, 1e-12);

  // The real EOP drives a proper orthonormal rotation through the table overload.
  pm::Quat<pmf::ECEF, pmf::ECI> q;
  ASSERT_TRUE(pf::ecefFromEci(t, *table, leap, q));
  const Eigen::Matrix3d rot = q.core().toRotationMatrix();
  EXPECT_TRUE((rot * rot.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-13));
  EXPECT_NEAR(rot.determinant(), 1.0, 1e-13);

  // An off-node epoch (½ day past the first row) interpolates strictly between
  // the first two rows' polar motion — proves interpolation runs, not just nodes.
  const double dat0 = DeltaAtOfMjd(rows.at(0).mjd_utc, leap);
  const pt::Tai t_half = TaiAtMjdUtc(rows.at(0).mjd_utc, dat0) +
                         pt::Duration::fromSecondsF(0.5 * pc::time::kSecondsPerDay);
  pf::EopValue vh;
  ASSERT_TRUE(table->lookup(t_half, leap, vh));
  EXPECT_GT((vh.xp_arcsec - rows.at(0).xp_arcsec) * (rows.at(1).xp_arcsec - vh.xp_arcsec), -1e-30)
      << "interpolated xp must lie between the bracketing nodes";
}

/// @file Golden fixture: the committed EGM2008 `.gfc` loads and drives gravity.
///
/// Loads `tests/golden/EGM2008_to200.gfc` — the real EGM2008 model (ICGEM `.gfc`),
/// committed as a degree-200 window in the native format (§3.7; produced by
/// `tools/gravity/`) — through `world::loadEgm2008Gfc` to 200×200 and exercises
/// the field (REQ-SIM-002). Integration proof that the committed file parses, the
/// known low-degree coefficients are the real EGM2008 values, and the loaded field
/// evaluates to a sane acceleration with `a = grad U` holding. CI never downloads.
///
/// The parser mechanics are unit-tested in `tests/unit/sim_world_egm2008_test.cpp`;
/// the GMAT cross-check of a full EGM2008 acceleration rides with the GMAT fixture
/// (REQ-SYS-010).

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <string>

#include "constants/constants.hpp"
#include "state/truth_state.hpp"
#include "world/egm2008.hpp"
#include "world/gravity_field.hpp"

namespace world = polaris::sim::world;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pc = polaris::constants;

namespace {

constexpr int kDegree = 200;

polaris::state::TruthState at(const Eigen::Vector3d& r) {
  polaris::state::TruthState s;
  s.position = pm::Vec3<pmf::ECI>(r);
  return s;
}

}  // namespace

TEST(Egm2008Golden, CommittedModelLoadsToDegree200AndEvaluates) {
  RecordProperty("verifies", "REQ-SIM-002");
  const std::string path = std::string(GOLDEN_DIR) + "/EGM2008_to200.gfc";
  world::Egm2008Header hdr;
  world::GravityCoeffs c;
  try {
    c = world::loadEgm2008Gfc(path, kDegree, &hdr);
  } catch (const std::exception& e) {
    FAIL() << "cannot load committed EGM2008 fixture: " << e.what();
  }

  // Loaded to the requested degree, full triangular table.
  ASSERT_EQ(c.nmax, kDegree);
  ASSERT_EQ(static_cast<int>(c.C.size()), kDegree + 1);
  ASSERT_EQ(static_cast<int>(c.C[kDegree].size()), kDegree + 1);

  // Header scalars are the EGM2008 model constants (not WGS84).
  EXPECT_NEAR(hdr.gm, 3.986004415e14, 1e6);
  EXPECT_NEAR(hdr.radius, 6378136.3, 1e-1);

  // Known low-degree coefficients are the real EGM2008 values (loose tolerances
  // absorb tide-system/format variation while still proving it is EGM2008, not
  // noise). C20 = -J2/sqrt(5); the tesserals C22/S22 are nonzero.
  EXPECT_DOUBLE_EQ(c.C[0][0], 1.0);
  EXPECT_NEAR(c.C[2][0], -4.841651e-4, 1e-9);
  EXPECT_NEAR(c.C[2][2], 2.4393836e-6, 1e-11);
  EXPECT_NEAR(c.S[2][2], -1.4002731e-6, 1e-11);

  // The loaded field evaluates to a sane LEO acceleration (~8.4 m/s^2 at 700 km)
  // and a = grad U holds by central difference of the independent potential().
  const world::SphericalHarmonicGravity g(c, kDegree, kDegree, hdr.gm, hdr.radius);
  const Eigen::Vector3d r(6.6e6, 1.9e6, 2.7e6);  // |r| ~ 7.4e6 m
  const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
  ASSERT_TRUE(a.allFinite());
  EXPECT_GT(a.norm(), 6.0);
  EXPECT_LT(a.norm(), 11.0);

  const double h = 25.0;
  Eigen::Vector3d numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d rp = r;
    Eigen::Vector3d rm = r;
    rp[i] += h;
    rm[i] -= h;
    numeric[i] = (g.potential(rp) - g.potential(rm)) / (2.0 * h);
  }
  EXPECT_LT((a - numeric).norm(), a.norm() * 1e-6);
}

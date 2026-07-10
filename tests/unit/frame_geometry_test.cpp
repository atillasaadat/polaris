/// @file Tests for the orbit-relative frame builders (RIC / LVLH).
///
/// Validated by known-orbit geometry and triad self-consistency: for a canonical
/// equatorial state the frames reduce to closed-form rotations, and for a general
/// state the radial/cross axes map exactly. External cross-validation against an
/// independent reference lands with the GMAT golden fixtures (§23.1, Push 5).

#include "math/frame_geometry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace pm = polaris::math;
namespace pf = polaris::math::frames;

namespace {

pm::Vec3<pf::ECI> Eci(double x, double y, double z) {
  return pm::Vec3<pf::ECI>(x, y, z);
}

// A representative low-Earth orbit radius / speed (m, m/s).
constexpr double kR = 7.0e6;
constexpr double kV = 7.5e3;

}  // namespace

TEST(FrameGeometry, RicIsIdentityForCanonicalEquatorialOrbit) {
  RecordProperty("verifies", "REQ-SYS-013");
  // r along +X, v along +Y => h along +Z; R=X, I=Y, C=Z => DCM = identity.
  const auto r = Eci(kR, 0.0, 0.0);
  const auto v = Eci(0.0, kV, 0.0);
  pm::Quat<pf::RIC, pf::ECI> q;
  ASSERT_TRUE(pm::ricFromEci(r, v, q));

  const auto r_ric = q.rotate(r);  // radial -> +R axis (index 0)
  const auto v_ric = q.rotate(v);  // velocity -> +I axis (index 1)
  EXPECT_NEAR(r_ric.x(), kR, 1e-6);
  EXPECT_NEAR(r_ric.y(), 0.0, 1e-6);
  EXPECT_NEAR(r_ric.z(), 0.0, 1e-6);
  EXPECT_NEAR(v_ric.x(), 0.0, 1e-6);
  EXPECT_NEAR(v_ric.y(), kV, 1e-6);
  EXPECT_NEAR(v_ric.z(), 0.0, 1e-6);
}

TEST(FrameGeometry, RicMapsRadialAndCrossTrackExactly) {
  RecordProperty("verifies", "REQ-SYS-013");
  // General inclined state: r must map to pure radial (+X), h to pure cross (+Z).
  const auto r = Eci(kR, 0.0, 0.0);
  const auto v = Eci(0.0, 6.0e3, 4.0e3);
  pm::Quat<pf::RIC, pf::ECI> q;
  ASSERT_TRUE(pm::ricFromEci(r, v, q));

  const auto r_ric = q.rotate(r);
  EXPECT_NEAR(r_ric.x(), r.norm(), 1e-6);
  EXPECT_NEAR(r_ric.y(), 0.0, 1e-6);
  EXPECT_NEAR(r_ric.z(), 0.0, 1e-6);

  const Eigen::Vector3d h = r.eigen().cross(v.eigen());
  const auto h_ric = q.rotate(pm::Vec3<pf::ECI>(h));
  EXPECT_NEAR(h_ric.x(), 0.0, 1e-3);
  EXPECT_NEAR(h_ric.y(), 0.0, 1e-3);
  EXPECT_NEAR(h_ric.z(), h.norm(), 1e-3);
}

TEST(FrameGeometry, LvlhZIsNadirAndXIsVelocityForCanonicalOrbit) {
  RecordProperty("verifies", "REQ-SYS-013");
  const auto r = Eci(kR, 0.0, 0.0);
  const auto v = Eci(0.0, kV, 0.0);
  pm::Quat<pf::LVLH, pf::ECI> q;
  ASSERT_TRUE(pm::lvlhFromEci(r, v, q));

  const auto r_lvlh = q.rotate(r);  // radial-out -> nadir is -Z, so z = -|r|
  const auto v_lvlh = q.rotate(v);  // velocity -> +X
  EXPECT_NEAR(r_lvlh.x(), 0.0, 1e-6);
  EXPECT_NEAR(r_lvlh.y(), 0.0, 1e-6);
  EXPECT_NEAR(r_lvlh.z(), -kR, 1e-6);
  EXPECT_NEAR(v_lvlh.x(), kV, 1e-6);
  EXPECT_NEAR(v_lvlh.y(), 0.0, 1e-6);
  EXPECT_NEAR(v_lvlh.z(), 0.0, 1e-6);
}

TEST(FrameGeometry, LvlhZAxisPointsNadirForGeneralOrbit) {
  RecordProperty("verifies", "REQ-SYS-013");
  const auto r = Eci(kR, 1.0e6, -2.0e6);
  const auto v = Eci(-1.0e3, 7.0e3, 1.0e3);
  pm::Quat<pf::LVLH, pf::ECI> q;
  ASSERT_TRUE(pm::lvlhFromEci(r, v, q));

  // The radial-out direction maps entirely onto the LVLH -Z (nadir) axis.
  const auto r_lvlh = q.rotate(r);
  EXPECT_NEAR(r_lvlh.x(), 0.0, 1e-3);
  EXPECT_NEAR(r_lvlh.y(), 0.0, 1e-3);
  EXPECT_NEAR(r_lvlh.z(), -r.norm(), 1e-3);
}

TEST(FrameGeometry, DegenerateRadialParallelVelocityReturnsFalse) {
  RecordProperty("verifies", "REQ-SYS-013");
  const auto r = Eci(kR, 0.0, 0.0);
  const auto v = Eci(3.0, 0.0, 0.0);  // v ∥ r => orbit normal vanishes
  pm::Quat<pf::RIC, pf::ECI> qr = pm::Quat<pf::RIC, pf::ECI>::Identity();
  pm::Quat<pf::LVLH, pf::ECI> ql = pm::Quat<pf::LVLH, pf::ECI>::Identity();
  EXPECT_FALSE(pm::ricFromEci(r, v, qr));
  EXPECT_FALSE(pm::lvlhFromEci(r, v, ql));
  // Output left untouched (still identity).
  EXPECT_DOUBLE_EQ(qr.core().scalar(), 1.0);
  EXPECT_DOUBLE_EQ(ql.core().scalar(), 1.0);
}

TEST(FrameGeometry, NonFiniteOrZeroInputReturnsFalse) {
  RecordProperty("verifies", "REQ-SYS-013");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  pm::Quat<pf::RIC, pf::ECI> qr;
  pm::Quat<pf::LVLH, pf::ECI> ql;
  EXPECT_FALSE(pm::ricFromEci(Eci(nan, 0.0, 0.0), Eci(0.0, kV, 0.0), qr));
  EXPECT_FALSE(pm::ricFromEci(Eci(kR, 0.0, 0.0), Eci(nan, 0.0, 0.0), qr));  // non-finite v
  EXPECT_FALSE(pm::ricFromEci(Eci(0.0, 0.0, 0.0), Eci(0.0, kV, 0.0), qr));  // zero radius
  EXPECT_FALSE(pm::lvlhFromEci(Eci(kR, 0.0, 0.0), Eci(nan, 0.0, 0.0), ql));
}

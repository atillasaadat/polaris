/// @file Unit tests for the conical eclipse and cannon-ball SRP models
/// (REQ-SIM-002; design doc §5.2).
///
/// The eclipse tests exercise each geometric regime independently of SRP — full
/// sun, umbra, penumbra — and check that the penumbra actually connects the two
/// extremes continuously and monotonically, which is the property a cylindrical
/// shadow lacks and the reason for the conical model. The SRP tests check the
/// closed-form magnitude against an INDEPENDENT hand computation (P·C_R·A/m at
/// 1 AU), the anti-sunward direction, the inverse-square falloff, and the
/// coupling to the shadow, plus the TAI→TDB argument and the c.p.-offset torque.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"
#include "world/body_position.hpp"
#include "world/eclipse.hpp"
#include "world/srp.hpp"

namespace world = polaris::sim::world;
namespace dyn = polaris::sim::dynamics;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

using Eci = pm::Vec3<pmf::ECI>;

constexpr double kAu = pc::bodies::kAstronomicalUnit;
constexpr double kRe = pc::wgs84::kSemiMajorAxis;

/// Sun parked on +X at 1 AU — the reference geometry for most cases below.
const Eigen::Vector3d kSunX(kAu, 0.0, 0.0);

polaris::state::TruthState at(const Eigen::Vector3d& r) {
  polaris::state::TruthState s;
  s.position = Eci(r);
  s.epoch = pt::Tai::fromNanosecondsSinceEpoch(1'600'000'000'000'000'000LL);  // ~2020
  return s;
}

world::BodyPositionFn fixedAt(const Eigen::Vector3d& pos) {
  return [pos](const pt::Tdb&, Eci& out) {
    out = Eci(pos);
    return true;
  };
}

}  // namespace

// --- Conical eclipse geometry ----------------------------------------------

TEST(Eclipse, FullSunOnTheSunlitSide) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Sub-solar point at LEO: the spacecraft is between the Earth and the Sun, so
  // nothing can occult it.
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(6.9e6, 0.0, 0.0), kSunX), 1.0);
}

TEST(Eclipse, UmbraDirectlyBehindEarth) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Anti-solar point at LEO — deep in the umbral cone (whose tip is ~1.4e6 km
  // from Earth, far beyond any Earth orbit).
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(-6.9e6, 0.0, 0.0), kSunX), 0.0);
}

TEST(Eclipse, UmbraStillTotalAtGeo) {
  RecordProperty("verifies", "REQ-SIM-002");
  // GEO (42164 km) on the anti-solar line is still well inside the umbral cone.
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(-4.2164e7, 0.0, 0.0), kSunX), 0.0);
}

TEST(Eclipse, FullSunWellOffTheShadowAxis) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Behind the Earth in X, but displaced far enough in Z to clear the penumbra.
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(-6.9e6, 0.0, 3.0e7), kSunX), 1.0);
}

TEST(Eclipse, PenumbraIsPartialAndBracketedByTheExtremes) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Sweep across the shadow terminator on the anti-solar side: at some offset
  // the disks partially overlap, giving a factor strictly between 0 and 1.
  const double x = -4.2164e7;  // GEO range, anti-solar
  bool saw_penumbra = false;
  for (double z = 0.0; z <= 1.2e7; z += 1.0e5) {
    const double nu = world::shadowFactor(Eigen::Vector3d(x, 0.0, z), kSunX);
    EXPECT_GE(nu, 0.0);
    EXPECT_LE(nu, 1.0);
    if (nu > 1e-6 && nu < 1.0 - 1e-6) {
      saw_penumbra = true;
    }
  }
  EXPECT_TRUE(saw_penumbra) << "no partial-shadow region found across the terminator";
}

TEST(Eclipse, ShadowFactorIsMonotoneLeavingTheShadow) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Moving away from the shadow axis can only ever reveal more of the solar
  // disk — the factor must be non-decreasing in the off-axis distance.
  const double x = -4.2164e7;
  double prev = world::shadowFactor(Eigen::Vector3d(x, 0.0, 0.0), kSunX);
  EXPECT_DOUBLE_EQ(prev, 0.0);
  for (double z = 1.0e5; z <= 1.2e7; z += 1.0e5) {
    const double nu = world::shadowFactor(Eigen::Vector3d(x, 0.0, z), kSunX);
    EXPECT_GE(nu, prev - 1e-12) << "shadow factor decreased at z = " << z;
    prev = nu;
  }
  EXPECT_DOUBLE_EQ(prev, 1.0);
}

TEST(Eclipse, ShadowFactorIsContinuousAcrossThePenumbra) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The point of a conical model: no 1 -> 0 jump. Adjacent samples across the
  // whole terminator must differ by far less than the full swing (a cylindrical
  // shadow would show a step of 1.0 between two neighbouring samples).
  const double x = -4.2164e7;
  const double dz = 1.0e4;
  double prev = world::shadowFactor(Eigen::Vector3d(x, 0.0, 0.0), kSunX);
  for (double z = dz; z <= 1.2e7; z += dz) {
    const double nu = world::shadowFactor(Eigen::Vector3d(x, 0.0, z), kSunX);
    EXPECT_LT(std::abs(nu - prev), 0.05) << "shadow factor jumped at z = " << z;
    prev = nu;
  }
}

TEST(Eclipse, DegenerateGeometryIsDark) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Inside the Earth, and a Sun at the geocenter: both unphysical, both return
  // 0 rather than NaN (§3.6).
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(0.5 * kRe, 0.0, 0.0), kSunX), 0.0);
  EXPECT_DOUBLE_EQ(world::shadowFactor(Eigen::Vector3d(6.9e6, 0.0, 0.0), Eigen::Vector3d::Zero()),
                   0.0);
}

// --- Solar radiation pressure ----------------------------------------------

TEST(Srp, MagnitudeMatchesClosedForm) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Independent hand computation at ~1 AU: a = P * C_R * A/m, with
  // P = S/c = 1361 / 299792458 = 4.5405e-6 N/m^2.
  const double area = 2.0, mass = 100.0, cr = 1.3;
  world::SolarRadiationPressure srp(area, mass, cr, fixedAt(kSunX));

  // Sub-solar point: full sun, and the Sun-spacecraft range is 1 AU - 6.9e6 m.
  const Eigen::Vector3d r(6.9e6, 0.0, 0.0);
  const double range = kAu - 6.9e6;
  const double pressure = (1361.0 / 299'792'458.0) * (kAu / range) * (kAu / range);
  const double expected = pressure * cr * area / mass;

  EXPECT_NEAR(srp.acceleration(at(r)).eigen().norm(), expected, expected * 1e-9);
  EXPECT_DOUBLE_EQ(srp.areaToMass(), area / mass);
}

TEST(Srp, PushesDirectlyAwayFromTheSun) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  // Off-axis sunlit point so the direction test is not trivially along an axis.
  const Eigen::Vector3d r(6.0e6, 3.0e6, 1.0e6);
  const Eigen::Vector3d a = srp.acceleration(at(r)).eigen();
  const Eigen::Vector3d anti_sun = (r - kSunX).normalized();
  // Parallel to the Sun -> spacecraft direction, i.e. pointing away from the Sun.
  EXPECT_LT((a.normalized() - anti_sun).norm(), 1e-12);
  EXPECT_GT(a.dot(anti_sun), 0.0);
}

TEST(Srp, FallsOffAsInverseSquareOfSolarRange) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Same spacecraft, Sun twice as far: the acceleration must drop by 4x. Put the
  // spacecraft at the geocenter-adjacent sunlit point so the range is the Sun
  // distance minus a fixed offset in both cases.
  world::SolarRadiationPressure near(2.0, 100.0, 1.3, fixedAt(Eigen::Vector3d(kAu, 0.0, 0.0)));
  world::SolarRadiationPressure far(2.0, 100.0, 1.3, fixedAt(Eigen::Vector3d(2.0 * kAu, 0.0, 0.0)));
  const Eigen::Vector3d r(0.0, 6.9e6, 0.0);  // ~perpendicular, so range ~ Sun distance

  const double a_near = near.acceleration(at(r)).eigen().norm();
  const double a_far = far.acceleration(at(r)).eigen().norm();
  EXPECT_NEAR(a_near / a_far, 4.0, 1e-6);
}

TEST(Srp, ScalesLinearlyWithCrAndAreaToMass) {
  RecordProperty("verifies", "REQ-SIM-002");
  const Eigen::Vector3d r(6.9e6, 0.0, 0.0);
  world::SolarRadiationPressure base(2.0, 100.0, 1.0, fixedAt(kSunX));
  world::SolarRadiationPressure double_cr(2.0, 100.0, 2.0, fixedAt(kSunX));
  world::SolarRadiationPressure double_area(4.0, 100.0, 1.0, fixedAt(kSunX));
  world::SolarRadiationPressure double_mass(2.0, 200.0, 1.0, fixedAt(kSunX));

  const double a0 = base.acceleration(at(r)).eigen().norm();
  EXPECT_NEAR(double_cr.acceleration(at(r)).eigen().norm(), 2.0 * a0, a0 * 1e-12);
  EXPECT_NEAR(double_area.acceleration(at(r)).eigen().norm(), 2.0 * a0, a0 * 1e-12);
  EXPECT_NEAR(double_mass.acceleration(at(r)).eigen().norm(), 0.5 * a0, a0 * 1e-12);
}

TEST(Srp, VanishesInTheUmbra) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  // Anti-solar point at LEO — fully eclipsed, so no photons, no acceleration.
  EXPECT_EQ(srp.acceleration(at(Eigen::Vector3d(-6.9e6, 0.0, 0.0))).eigen(),
            Eigen::Vector3d::Zero());
}

TEST(Srp, IsScaledByThePenumbraFactor) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Find a penumbral point, then check SRP there equals the unshadowed value
  // times the shadow factor — i.e. the eclipse enters exactly as a scale factor.
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  const double x = -4.2164e7;
  bool checked = false;
  for (double z = 0.0; z <= 1.2e7; z += 5.0e4) {
    const Eigen::Vector3d r(x, 0.0, z);
    const double nu = world::shadowFactor(r, kSunX);
    if (nu <= 1e-3 || nu >= 1.0 - 1e-3) {
      continue;
    }
    // Same geometry with the Sun far off-axis would be unshadowed; instead
    // reconstruct the unshadowed magnitude from the closed form.
    const double range = (r - kSunX).norm();
    const double pressure = pc::srp::kPressureAt1Au * (kAu / range) * (kAu / range);
    const double unshadowed = pressure * 1.3 * 2.0 / 100.0;
    EXPECT_NEAR(srp.acceleration(at(r)).eigen().norm(), nu * unshadowed, unshadowed * 1e-9);
    checked = true;
    break;
  }
  EXPECT_TRUE(checked) << "no penumbral sample found";
}

TEST(Srp, VanishesWithoutEphemerisCoverage) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, [](const pt::Tdb&, Eci&) { return false; });
  EXPECT_EQ(srp.acceleration(at(Eigen::Vector3d(6.9e6, 0.0, 0.0))).eigen(),
            Eigen::Vector3d::Zero());
}

TEST(Srp, ResolverReceivesTdbNotTai) {
  RecordProperty("verifies", "REQ-SIM-002");
  const polaris::state::TruthState s = at(Eigen::Vector3d(6.9e6, 0.0, 0.0));
  pt::Tdb seen{};
  bool called = false;
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, [&](const pt::Tdb& t, Eci& out) {
    seen = t;
    called = true;
    out = Eci(kSunX);
    return true;
  });
  srp.acceleration(s);
  ASSERT_TRUE(called);
  EXPECT_EQ(seen, pt::toTdb(pt::toTt(s.epoch)));
}

TEST(Srp, IsTorqueFreeWithCenterOfPressureAtCenterOfMass) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  EXPECT_EQ(srp.torque(at(Eigen::Vector3d(6.9e6, 0.0, 0.0))).eigen(), Eigen::Vector3d::Zero());
}

TEST(Srp, CenterOfPressureOffsetProducesPerpendicularDisturbanceTorque) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  const pm::Vec3<pmf::Body> r_cp(Eigen::Vector3d(0.1, 0.0, 0.0));
  srp.setCenterOfPressureOffset(r_cp);

  // Identity attitude keeps Body aligned with ECI, so the force stays along -X
  // (anti-sunward) and a +X lever arm is parallel to it -> still no torque.
  polaris::state::TruthState s = at(Eigen::Vector3d(6.9e6, 0.0, 0.0));
  EXPECT_LT(srp.torque(s).eigen().norm(), 1e-30);

  // A lever arm perpendicular to the force does produce a torque, and tau must
  // be perpendicular to both the arm and the force (definition of the cross
  // product) with magnitude |r_cp| |F| for the perpendicular case.
  const pm::Vec3<pmf::Body> r_cp_perp(Eigen::Vector3d(0.0, 0.1, 0.0));
  srp.setCenterOfPressureOffset(r_cp_perp);
  const Eigen::Vector3d tau = srp.torque(s).eigen();
  const Eigen::Vector3d f_body = 100.0 * srp.acceleration(s).eigen();  // identity attitude

  EXPECT_GT(tau.norm(), 0.0);
  EXPECT_NEAR(tau.norm(), 0.1 * f_body.norm(), tau.norm() * 1e-12);
  EXPECT_LT(std::abs(tau.dot(f_body)) / (tau.norm() * f_body.norm()), 1e-12);
  EXPECT_LT(std::abs(tau.dot(r_cp_perp.eigen())) / (tau.norm() * 0.1), 1e-12);
}

TEST(Srp, TorqueIsTheExactLeverArmCrossProductAtANonTrivialAttitude) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Identity attitude hides a missing ECI->Body rotation of the force, because
  // the two frames coincide. Rotate the vehicle and compare against the cross
  // product computed here, component by component (design doc §5.3).
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  const Eigen::Vector3d r_cp(0.03, -0.07, 0.11);
  srp.setCenterOfPressureOffset(pm::Vec3<pmf::Body>(r_cp));

  polaris::state::TruthState s = at(Eigen::Vector3d(4.0e6, 5.0e6, 1.0e6));
  s.attitude = pm::Quat<pmf::Body, pmf::ECI>(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 2.0, -1.0).normalized(), 0.9));

  const Eigen::Matrix3d a_body_from_eci = s.attitude.core().toRotationMatrix();
  const Eigen::Vector3d f_body = a_body_from_eci * (100.0 * srp.acceleration(s).eigen());
  const Eigen::Vector3d expected = r_cp.cross(f_body);

  ASSERT_GT(expected.norm(), 0.0);
  EXPECT_LT((srp.torque(s).eigen() - expected).norm(), 1.0e-16 * expected.norm());
}

TEST(Srp, TorqueVanishesInTheUmbra) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  srp.setCenterOfPressureOffset(pm::Vec3<pmf::Body>(Eigen::Vector3d(0.0, 0.1, 0.0)));
  EXPECT_EQ(srp.torque(at(Eigen::Vector3d(-6.9e6, 0.0, 0.0))).eigen(), Eigen::Vector3d::Zero());
}

TEST(Srp, ComposesWithOtherForceModels) {
  RecordProperty("verifies", "REQ-SIM-002");
  const dyn::TwoBodyGravity earth(pc::wgs84::kGM);
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));

  dyn::CompositeForceModel composite;
  composite.add(&earth);
  composite.add(&srp);
  EXPECT_EQ(composite.size(), 2u);

  const polaris::state::TruthState s = at(Eigen::Vector3d(6.9e6, 1.0e6, 2.0e6));
  const Eigen::Vector3d expected = earth.acceleration(s).eigen() + srp.acceleration(s).eigen();
  EXPECT_LT((composite.acceleration(s).eigen() - expected).norm(), 1e-18);
}

TEST(Srp, IsSmallButNotNegligibleAtLeo) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Sanity magnitude: a typical smallsat (A/m ~ 0.02 m^2/kg, C_R 1.3) sees an
  // SRP acceleration of order 1e-7 m/s^2 — ~8 orders below Earth's gravity.
  world::SolarRadiationPressure srp(2.0, 100.0, 1.3, fixedAt(kSunX));
  const double a = srp.acceleration(at(Eigen::Vector3d(6.9e6, 0.0, 0.0))).eigen().norm();
  EXPECT_GT(a, 1e-8);
  EXPECT_LT(a, 1e-6);
}

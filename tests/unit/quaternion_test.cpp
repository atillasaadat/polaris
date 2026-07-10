/// @file Self-consistency tests for the JPL scalar-first quaternion library.
///
/// These validate that the convention is internally consistent (the composition
/// rule A(a*b)=A(a)A(b), DCM round-trips, conjugate=transpose). External
/// correctness against an independent reference is validated separately against
/// GMAT golden fixtures (design doc §23.1, Push 5).

#include "math/quaternion.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace pm = polaris::math;
using pm::Quaternion;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Quaternion AxisAngle(double ax, double ay, double az, double angle) {
  Eigen::Vector3d axis(ax, ay, az);
  axis.normalize();
  return Quaternion::FromAxisAngle(axis, angle);
}

::testing::AssertionResult MatNear(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b, double tol) {
  const double d = (a - b).cwiseAbs().maxCoeff();
  if (d <= tol) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "max|A-B| = " << d << " > " << tol;
}

}  // namespace

TEST(Quaternion, IdentityHasNoEffect) {
  RecordProperty("verifies", "REQ-SYS-003");
  const Quaternion q;  // identity
  EXPECT_TRUE(q.isUnit());
  EXPECT_DOUBLE_EQ(q.scalar(), 1.0);
  EXPECT_TRUE(MatNear(q.toRotationMatrix(), Eigen::Matrix3d::Identity(), 1e-15));
  const Eigen::Vector3d v(1.0, -2.0, 3.0);
  EXPECT_LT((q.rotate(v) - v).norm(), 1e-15);
}

TEST(Quaternion, ZRotationMatchesPassiveRot3) {
  RecordProperty("verifies", "REQ-SYS-003");
  const double angle = 30.0 * kPi / 180.0;
  const Quaternion q = AxisAngle(0.0, 0.0, 1.0, angle);
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  Eigen::Matrix3d rot3;
  rot3 << c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0;  // passive ROT3 (Vallado)
  EXPECT_TRUE(MatNear(q.toRotationMatrix(), rot3, 1e-12));
}

TEST(Quaternion, RotateMatchesDcm) {
  RecordProperty("verifies", "REQ-SYS-003");
  const Quaternion q = AxisAngle(0.3, -0.7, 0.5, 1.1);
  const Eigen::Vector3d v(2.0, -1.0, 0.5);
  EXPECT_LT((q.rotate(v) - q.toRotationMatrix() * v).norm(), 1e-12);
}

TEST(Quaternion, CompositionMatchesDcmProduct) {
  RecordProperty("verifies", "REQ-SYS-003");
  const Quaternion a = AxisAngle(0.0, 0.0, 1.0, 0.5);
  const Quaternion b = AxisAngle(0.0, 1.0, 0.0, -0.8);
  const Quaternion ab = a * b;
  EXPECT_TRUE(MatNear(ab.toRotationMatrix(), a.toRotationMatrix() * b.toRotationMatrix(), 1e-12));
}

TEST(Quaternion, CompositionAppliesInDcmOrder) {
  RecordProperty("verifies", "REQ-SYS-003");
  // A(a*b) v == a applied after b, i.e. a.rotate(b.rotate(v)).
  const Quaternion a = AxisAngle(0.0, 0.0, 1.0, 0.7);
  const Quaternion b = AxisAngle(1.0, 0.0, 0.0, 0.4);
  const Eigen::Vector3d v(1.0, 2.0, -0.5);
  const Eigen::Vector3d lhs = (a * b).rotate(v);
  const Eigen::Vector3d rhs = a.rotate(b.rotate(v));
  EXPECT_LT((lhs - rhs).norm(), 1e-12);
}

TEST(Quaternion, ConjugateIsInverse) {
  RecordProperty("verifies", "REQ-CONV-004");
  const Quaternion q = AxisAngle(1.0, 2.0, 3.0, 0.9);
  const Quaternion qi = q.conjugate();
  const Quaternion id = q * qi;
  EXPECT_NEAR(std::abs(id.scalar()), 1.0, 1e-12);
  EXPECT_NEAR(id.vec().norm(), 0.0, 1e-12);
  EXPECT_TRUE(MatNear(qi.toRotationMatrix(), q.toRotationMatrix().transpose(), 1e-12));
}

TEST(Quaternion, RoundTripThroughDcm) {
  RecordProperty("verifies", "REQ-CONV-004");
  const double angles[] = {0.01, 0.5, kPi / 2.0, 3.0, kPi - 1e-3};
  const double axes[][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {-1, 2, 3}};
  for (const double angle : angles) {
    for (const auto& ax : axes) {
      const Quaternion q = AxisAngle(ax[0], ax[1], ax[2], angle).canonical();
      const Eigen::Matrix3d dcm = q.toRotationMatrix();
      const Quaternion r = Quaternion::FromRotationMatrix(dcm);
      // Compare via DCM (robust); acos-based angularDistance is ill-conditioned
      // near identity and would amplify ~1e-16 component error to ~1e-8 rad.
      EXPECT_TRUE(MatNear(dcm, r.toRotationMatrix(), 1e-12)) << "angle=" << angle;
    }
  }
}

TEST(Quaternion, CanonicalEnforcesPositiveScalar) {
  RecordProperty("verifies", "REQ-SYS-003");
  const Quaternion q(-0.5, 0.5, -0.5, 0.5);
  const Quaternion c = q.canonical();
  EXPECT_GE(c.scalar(), 0.0);
  EXPECT_NEAR(c.angularDistance(q), 0.0, 1e-12);  // q and -q are the same rotation
}

TEST(Quaternion, NormalizeRejectsDegenerate) {
  RecordProperty("verifies", "REQ-CONV-004");
  Quaternion z(0.0, 0.0, 0.0, 0.0);
  EXPECT_FALSE(z.normalize());
  Quaternion q(2.0, 0.0, 0.0, 0.0);
  EXPECT_TRUE(q.normalize());
  EXPECT_NEAR(q.norm(), 1.0, 1e-15);
}

TEST(Quaternion, IsFiniteDetectsNonFinite) {
  RecordProperty("verifies", "REQ-SYS-006");
  EXPECT_TRUE(Quaternion::Identity().isFinite());
  EXPECT_FALSE(Quaternion(std::nan(""), 0.0, 0.0, 0.0).isFinite());
}

TEST(TaggedQuat, ComposeAndRotateAreFrameSafe) {
  RecordProperty("verifies", "REQ-SYS-004");
  using pm::Quat;
  using pm::Vec3;
  using pm::frames::Body;
  using pm::frames::ECI;
  using pm::frames::LVLH;

  const Quat<Body, ECI> q_be(AxisAngle(0.0, 0.0, 1.0, 0.5));
  const Quat<ECI, LVLH> q_el(AxisAngle(0.0, 1.0, 0.0, 0.3));
  const Quat<Body, LVLH> q_bl = q_be * q_el;  // only compiles because frames chain

  const Vec3<LVLH> v(1.0, 0.0, 0.0);
  const Vec3<Body> vb = q_bl.rotate(v);
  const Eigen::Vector3d expected = (q_be.core() * q_el.core()).rotate(v.eigen());
  EXPECT_LT((vb.eigen() - expected).norm(), 1e-12);
}

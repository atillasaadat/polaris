/// @file Why the repo computes angles with `atan2` and never a bare `acos`.
///
/// This is the test the house rule in `lib/README.md` points at. It does not
/// exercise a Polaris algorithm so much as pin the numerical fact that forced
/// the convention: for two nearly-parallel (or nearly-antiparallel) unit
/// vectors, `acos(a·b)` returns an angle wrong by ~1e-8 rad even though `a·b`
/// is correct to machine epsilon, while `atan2(|a×b|, a·b)` is correct to
/// ~1e-15 rad on the same inputs.
///
/// The cause is stationarity, not rounding in `acos` itself. Near θ = 0 the
/// cosine is 1 - θ²/2, so the last ~1e-16 of the cosine spans a range of θ of
/// order √(2·1e-16) ≈ 1.5e-8: the information about θ is simply not present in
/// the cosine any more. The sine is O(θ) there and retains all of it, and the
/// cross-product norm computes that sine directly from the vectors rather than
/// recovering it from the (already-degraded) cosine. `atan2` uses both, so it
/// is well conditioned across the whole range. The same argument holds at
/// θ = π with the roles of the two branches swapped, and for a quaternion
/// angle, where the scalar part plays the role of the cosine of θ/2.
///
/// Reference: Kahan, "Computing Cross-Products and Rotations in 2- and
/// 3-Dimensional Euclidean Spaces" (2016) §"Mangled Angles" — the standard
/// statement that the angle between vectors must be formed from atan2 of the
/// cross-product norm against the dot product. Applied to attitude error
/// angles as in Markley & Crassidis, *Fundamentals of Spacecraft Attitude
/// Determination and Control* (2014) §3.2 (quaternion/rotation-vector
/// equivalence). Cited in `docs/refs.bib` as `kahan2016cross`.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "math/quaternion.hpp"

namespace pm = polaris::math;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

/// The form this repo forbids, kept here as the thing being measured against.
double acosAngle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::acos(std::max(-1.0, std::min(1.0, a.dot(b))));
}

/// The house form (lib/README.md).
double atan2Angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

/// Unit vector at exactly @p theta from +x, in the x–y plane.
Eigen::Vector3d atAngle(double theta) {
  return Eigen::Vector3d(std::cos(theta), std::sin(theta), 0.0);
}

}  // namespace

// A pair separated by nanoradians to a few times 1e-8: the regime a converged
// attitude estimate, a limb clearance and a boresight incidence all live in.
//
// The claim is asserted on the *magnitude* of the acos error rather than on an
// exact value, because the exact value depends on how the platform's libm
// rounds cos(theta) right at the boundary. At theta = 1e-8, theta^2/2 = 5e-17
// sits 0.05 ulp below half an ulp of 1.0 (5.55e-17): a correctly-rounded libm
// returns exactly 1.0, but a legal 0.5-ulp-accurate one may return the next
// double down, and the test should not hinge on which. What does not depend on
// the tie is that acos gets the angle wrong by a large fraction of the angle
// itself — the information is not in the cosine to recover.
TEST(AngleConditioning, AcosLosesMostOfTheAngleWhereAtanIsExact) {
  const Eigen::Vector3d a = Eigen::Vector3d::UnitX();

  for (const double theta : {1.0e-9, 5.0e-9, 1.0e-8}) {
    const Eigen::Vector3d b = atAngle(theta);

    EXPECT_LT(std::abs(atan2Angle(a, b) - theta), 1.0e-15) << "atan2 form at theta = " << theta;
    EXPECT_GT(std::abs(acosAngle(a, b) - theta), 0.4 * theta)
        << "acos form at theta = " << theta << " should have lost most of the angle";
  }

  // Below the tie the margin is comfortable (theta^2/2 <= 1.25e-17, under a
  // quarter of a half-ulp), so the collapse really is to a hard zero — which is
  // the sharper statement: acos cannot tell these two angles apart at all, so
  // anything gated or graded on such an angle sees a step where the atan2 form
  // sees an ordering.
  EXPECT_EQ(acosAngle(a, atAngle(1.0e-9)), 0.0);
  EXPECT_EQ(acosAngle(a, atAngle(5.0e-9)), 0.0);
  EXPECT_LT(atan2Angle(a, atAngle(1.0e-9)), atan2Angle(a, atAngle(5.0e-9)));
}

// The same collapse at the other stationary point, which is where a magnetometer
// and a sun vector sit during an eclipse-adjacent geometry and where a TRIAD
// pair degenerates. `theta` here is the double nearest pi - delta, which is what
// both forms are measured against; the magnitude assertion is used for the same
// libm-tie reason as above, since cos(pi - delta) sits at the same boundary.
TEST(AngleConditioning, AcosLosesMostOfTheAngleNearAntiparallelToo) {
  const Eigen::Vector3d a = Eigen::Vector3d::UnitX();

  for (const double delta : {1.0e-9, 5.0e-9, 1.0e-8}) {
    const double theta = kPi - delta;
    const Eigen::Vector3d b = atAngle(theta);

    EXPECT_LT(std::abs(atan2Angle(a, b) - theta), 1.0e-15) << "atan2 form at delta = " << delta;
    EXPECT_GT(std::abs(acosAngle(a, b) - theta), 0.4 * delta)
        << "acos form at delta = " << delta << " should have lost most of the gap to pi";
  }

  // Below the tie, again the sharper statement: a hard pi, indistinguishable.
  EXPECT_EQ(acosAngle(a, atAngle(kPi - 1.0e-9)), std::acos(-1.0));
  EXPECT_EQ(acosAngle(a, atAngle(kPi - 5.0e-9)), std::acos(-1.0));
}

// Away from the stationary points both forms are fine — the convention costs
// nothing where acos would have worked, which is why it is applied uniformly
// rather than only at the sites someone judged to be near-parallel.
TEST(AngleConditioning, BothFormsAgreeAwayFromTheEndpoints) {
  const Eigen::Vector3d a = Eigen::Vector3d::UnitX();
  for (const double theta : {0.3, 1.0, kPi / 2.0, 2.5}) {
    const Eigen::Vector3d b = atAngle(theta);
    EXPECT_NEAR(atan2Angle(a, b), theta, 1.0e-15);
    EXPECT_NEAR(acosAngle(a, b), theta, 1.0e-14);
  }
}

// Scale invariance is the practical bonus: the norms cancel between the two
// atan2 arguments, so a caller with unnormalised vectors needs no division —
// which is what sim/sensors/occlusion.cpp relies on.
TEST(AngleConditioning, AtanFormNeedsNoNormalisation) {
  const Eigen::Vector3d a(3.0e6, -1.0e6, 4.0e5);
  const Eigen::Vector3d b(1.5e11, 2.0e10, -7.0e9);
  EXPECT_NEAR(atan2Angle(a, b), atan2Angle(a.normalized(), b.normalized()), 1.0e-15);
}

// Quaternion::angularDistance is the same statement with the scalar part in the
// role of cos(theta/2): near identity it is 1 - theta^2/8, so 2*acos of it is
// wrong by ~1e-8 rad while the vector part is O(theta) and carries the answer.
TEST(AngleConditioning, QuaternionAngleIsExactAtSmallAngles) {
  const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, -3.0).normalized();

  // cos(theta/2) rounds to exactly 1.0 for theta <= 2e-8, so the banned form
  // returns a hard zero over this whole range while atan2 stays exact.
  for (const double theta : {1.0e-9, 5.0e-9, 1.0e-8}) {
    const pm::Quaternion q = pm::Quaternion::FromAxisAngle(axis, theta);
    const pm::Quaternion identity;

    EXPECT_NEAR(identity.angularDistance(q), theta, 1.0e-15) << "at theta = " << theta;

    // The banned form, for contrast: 2*acos of the four-vector dot product.
    const double d = std::min(1.0, std::abs(q.scalar()));
    EXPECT_EQ(2.0 * std::acos(d), 0.0) << "2*acos(scalar) at theta = " << theta;
  }
}

// Sign of the quaternion must not matter (q and -q are the same rotation), and
// the result must stay on the theta <= pi branch — the property the absolute
// value on the scalar part provides.
TEST(AngleConditioning, QuaternionAngleIsSignAndBranchStable) {
  const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  const pm::Quaternion identity;

  for (const double theta : {1.0e-6, 0.5, 2.0, kPi - 1.0e-6}) {
    const pm::Quaternion q = pm::Quaternion::FromAxisAngle(axis, theta);
    const pm::Quaternion negated(-q.scalar(), -q.vec().x(), -q.vec().y(), -q.vec().z());

    EXPECT_NEAR(identity.angularDistance(q), theta, 1.0e-14) << "at theta = " << theta;
    EXPECT_NEAR(identity.angularDistance(negated), theta, 1.0e-14)
        << "negated quaternion at theta = " << theta;
    EXPECT_LE(identity.angularDistance(q), kPi + 1.0e-12);
  }

  // Past pi the short way round is 2*pi - theta, not theta.
  const pm::Quaternion far = pm::Quaternion::FromAxisAngle(axis, 1.75 * kPi);
  EXPECT_NEAR(identity.angularDistance(far), 0.25 * kPi, 1.0e-14);
}

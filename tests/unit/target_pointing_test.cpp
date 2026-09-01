/// @file Unit tests for boresight-to-target guidance (§8.4; REQ-AGN-004).
///
/// The guidance that turns "point the camera at that object" into the quaternion
/// and rate the §8.5 controller tracks. What needs proving is mostly about the
/// *under-determined* degree of freedom: aligning a boresight is two DOF and an
/// attitude is three, so the interesting failures all live in how roll is
/// resolved and what happens when the constraint that resolves it says nothing.

#include "gnc/target_pointing.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;

namespace {

const pm::Vec3<pmf::Body> kBodyX(1.0, 0.0, 0.0);
const pm::Vec3<pmf::Body> kBodyY(0.0, 1.0, 0.0);
const pm::Vec3<pmf::Body> kBodyZ(0.0, 0.0, 1.0);

/// A vehicle at 7000 km on +x, a target 500 km "above" it on +y.
const pm::Vec3<pmf::ECI> kObserver(7000.0e3, 0.0, 0.0);
const pm::Vec3<pmf::ECI> kTarget(7000.0e3, 500.0e3, 0.0);
/// Orbit normal, a natural secondary reference and safely off the line of sight.
const pm::Vec3<pmf::ECI> kNormal(0.0, 0.0, 1.0);

}  // namespace

TEST(TargetPointing, PutsTheBoresightExactlyOnTheTarget) {
  pm::Quat<pmf::Body, pmf::ECI> q;
  ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, kNormal, q),
            pg::TargetPointingStatus::kOk);

  // Rotate the line of sight into body axes; it must land on the boresight.
  const Eigen::Vector3d los = (kTarget.eigen() - kObserver.eigen()).normalized();
  const Eigen::Vector3d in_body = q.rotate(pm::Vec3<pmf::ECI>(los)).eigen();
  EXPECT_NEAR(in_body.x(), 0.0, 1e-12);
  EXPECT_NEAR(in_body.y(), 0.0, 1e-12);
  EXPECT_NEAR(in_body.z(), 1.0, 1e-12) << "the boresight is not on the target";
}

TEST(TargetPointing, WorksForAnyBoresightAxis) {
  // The boresight is a mission parameter, not a convention baked into the math.
  for (const auto& axis : {kBodyX, kBodyY, kBodyZ}) {
    const pm::Vec3<pmf::Body> secondary = (std::fabs(axis.eigen().z()) > 0.5) ? kBodyX : kBodyZ;
    pm::Quat<pmf::Body, pmf::ECI> q;
    ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, axis, secondary, kNormal, q),
              pg::TargetPointingStatus::kOk);
    const Eigen::Vector3d los = (kTarget.eigen() - kObserver.eigen()).normalized();
    const Eigen::Vector3d in_body = q.rotate(pm::Vec3<pmf::ECI>(los)).eigen();
    EXPECT_LT((in_body - axis.eigen().normalized()).norm(), 1e-12);
  }
}

TEST(TargetPointing, TheSecondaryConstraintActuallyResolvesRoll) {
  // The load-bearing test for the second degree of freedom. Two different
  // secondary references, same target: the boresight must land identically and
  // the resulting attitudes must *differ*. If they came out the same, roll would
  // be being chosen by the implementation rather than by the caller — which is
  // exactly the silent failure this API exists to prevent.
  pm::Quat<pmf::Body, pmf::ECI> q_normal;
  pm::Quat<pmf::Body, pmf::ECI> q_radial;
  const pm::Vec3<pmf::ECI> radial(1.0, 0.0, 0.0);
  ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, kNormal, q_normal),
            pg::TargetPointingStatus::kOk);
  ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, radial, q_radial),
            pg::TargetPointingStatus::kOk);

  const Eigen::Vector3d los = (kTarget.eigen() - kObserver.eigen()).normalized();
  EXPECT_LT((q_normal.rotate(pm::Vec3<pmf::ECI>(los)).eigen() -
             q_radial.rotate(pm::Vec3<pmf::ECI>(los)).eigen())
                .norm(),
            1e-12)
      << "the boresight moved when only the roll reference changed";

  const Eigen::Vector4d a = q_normal.core().coeffs();
  const Eigen::Vector4d b = q_radial.core().coeffs();
  EXPECT_GT((a - b).norm(), 1e-3) << "roll is not responding to the secondary reference at all";
}

TEST(TargetPointing, TheSecondaryAxisLandsAsCloseAsPossibleToItsReference) {
  // "As close as possible" is a claim about optimality, so it is tested as one:
  // the achieved separation must be no worse than any small perturbation of the
  // solution about the boresight.
  pm::Quat<pmf::Body, pmf::ECI> q;
  ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, kNormal, q),
            pg::TargetPointingStatus::kOk);

  const Eigen::Vector3d los = (kTarget.eigen() - kObserver.eigen()).normalized();
  const Eigen::Vector3d ref = kNormal.eigen().normalized();
  // Secondary body axis, expressed in ECI.
  const Eigen::Vector3d secondary_eci = q.inverse().rotate(kBodyX).eigen();
  const double best = std::acos(std::clamp(secondary_eci.dot(ref), -1.0, 1.0));

  for (double droll : {-0.05, -0.01, 0.01, 0.05}) {
    const Eigen::AngleAxisd twist(droll, los);
    const Eigen::Vector3d perturbed = twist * secondary_eci;
    const double sep = std::acos(std::clamp(perturbed.dot(ref), -1.0, 1.0));
    EXPECT_LE(best, sep + 1e-12) << "a roll of " << droll << " rad beats the returned solution";
  }
}

TEST(TargetPointing, RefusesWhenTheReferenceIsParallelToTheLineOfSight) {
  // The degenerate case, refused rather than resolved arbitrarily: with the
  // reference along the line of sight the secondary constraint says nothing
  // about roll and every answer satisfies it equally. An arbitrary pick would be
  // stable and plausible and therefore invisible.
  const pm::Vec3<pmf::ECI> along_los(0.0, 1.0, 0.0);
  pm::Quat<pmf::Body, pmf::ECI> q;
  EXPECT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, along_los, q),
            pg::TargetPointingStatus::kSecondaryDegenerate);
}

TEST(TargetPointing, RefusesCoincidentObserverAndTarget) {
  pm::Quat<pmf::Body, pmf::ECI> q;
  EXPECT_EQ(pg::targetPointingQuaternion(kObserver, kObserver, kBodyZ, kBodyX, kNormal, q),
            pg::TargetPointingStatus::kCoincident);
}

TEST(TargetPointing, RefusesParallelOrNullBodyAxes) {
  pm::Quat<pmf::Body, pmf::ECI> q;
  EXPECT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyZ, kNormal, q),
            pg::TargetPointingStatus::kBadAxes);
  EXPECT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, pm::Vec3<pmf::Body>::Zero(), kBodyX,
                                         kNormal, q),
            pg::TargetPointingStatus::kBadAxes);
}

TEST(TargetPointing, RefusesNonFiniteInput) {
  pm::Quat<pmf::Body, pmf::ECI> q;
  const pm::Vec3<pmf::ECI> bad(std::nan(""), 0.0, 0.0);
  EXPECT_EQ(pg::targetPointingQuaternion(kObserver, bad, kBodyZ, kBodyX, kNormal, q),
            pg::TargetPointingStatus::kBadInput);
}

TEST(TargetPointing, TheReturnedQuaternionIsUnitAndCanonical) {
  pm::Quat<pmf::Body, pmf::ECI> q;
  ASSERT_EQ(pg::targetPointingQuaternion(kObserver, kTarget, kBodyZ, kBodyX, kNormal, q),
            pg::TargetPointingStatus::kOk);
  EXPECT_NEAR(q.core().coeffs().norm(), 1.0, 1e-12);
  EXPECT_GE(q.core().w(), 0.0) << "not canonical; the controller's error term assumes q0 >= 0";
}

TEST(TargetPointing, TheRateIsPerpendicularToTheLineOfSight) {
  // omega = (r x v)/|r|^2 by construction, so it can carry no component along
  // the line of sight. Asserted because a formulation that let range rate leak
  // in would command a roll about the boresight that nothing asked for.
  const pm::Vec3<pmf::ECI> obs_v(0.0, 7500.0, 0.0);
  const pm::Vec3<pmf::ECI> tgt_v(100.0, 7400.0, 30.0);
  pm::Vec3<pmf::ECI> omega;
  ASSERT_EQ(pg::targetPointingRate(kObserver, obs_v, kTarget, tgt_v, omega),
            pg::TargetPointingStatus::kOk);
  const Eigen::Vector3d los = (kTarget.eigen() - kObserver.eigen()).normalized();
  EXPECT_NEAR(omega.eigen().dot(los), 0.0, 1e-15);
}

TEST(TargetPointing, PureRangeRateProducesNoAngularRate) {
  // Closing on a target does not rotate the direction to it. This is the
  // property that separates a line-of-sight rate from a relative velocity, and
  // getting it wrong would make the feedforward chase range changes.
  const pm::Vec3<pmf::ECI> obs_v = pm::Vec3<pmf::ECI>::Zero();
  const pm::Vec3<pmf::ECI> closing(0.0, -100.0, 0.0);  // straight down the line of sight
  pm::Vec3<pmf::ECI> omega;
  ASSERT_EQ(pg::targetPointingRate(kObserver, obs_v, kTarget, closing, omega),
            pg::TargetPointingStatus::kOk);
  EXPECT_LT(omega.norm(), 1e-18);
}

TEST(TargetPointing, TheRateMatchesAFiniteDifferenceOfTheDirection) {
  // The analytic rate against the thing it replaces. Differencing successive
  // directions is what a naive implementation would do; it is avoided in flight
  // because it amplifies the target's position noise by 1/dt, but it is exactly
  // the right cross-check here where the inputs are noiseless.
  const pm::Vec3<pmf::ECI> obs_v(0.0, 7500.0, 0.0);
  const pm::Vec3<pmf::ECI> tgt_v(-200.0, 7300.0, 400.0);
  pm::Vec3<pmf::ECI> omega;
  ASSERT_EQ(pg::targetPointingRate(kObserver, obs_v, kTarget, tgt_v, omega),
            pg::TargetPointingStatus::kOk);

  const double dt = 1.0e-3;
  auto direction = [&](double t) {
    const Eigen::Vector3d o = kObserver.eigen() + obs_v.eigen() * t;
    const Eigen::Vector3d g = kTarget.eigen() + tgt_v.eigen() * t;
    return (g - o).normalized();
  };
  const Eigen::Vector3d u0 = direction(0.0);
  const Eigen::Vector3d du = (direction(dt) - direction(-dt)) / (2.0 * dt);
  // d(u)/dt = omega x u for a unit vector rotating at omega.
  const Eigen::Vector3d predicted = omega.eigen().cross(u0);
  EXPECT_LT((du - predicted).norm(), 1e-9 * std::max(1.0, du.norm()));
}

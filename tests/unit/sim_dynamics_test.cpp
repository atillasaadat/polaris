/// @file Tests for the 6DOF rigid-body plant + RK8(9) integrator (REQ-SIM-001).
///
/// Coverage is layered so a defect is localized:
///  - tableau invariants (row sums == nodes, Sum b == 1, Sum e == 0) catch any
///    coefficient transcription error cheaply;
///  - the error estimate's order of accuracy validates the embedded weights;
///  - Kepler two-body energy/angular-momentum conservation validates the
///    advancing weights on the translational EOM;
///  - a torque-free closed-form attitude match pins the quaternion-kinematics
///    sign, and torque-free energy/inertial-momentum conservation validates the
///    coupled Euler + kinematics rotational EOM.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/rigid_body.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "state/truth_state.hpp"

namespace dyn = polaris::sim::dynamics;
namespace pf = polaris::math::frames;
namespace pm = polaris::math;
namespace ps = polaris::state;

namespace {

// --- Tableau integrity ------------------------------------------------------

TEST(Rk89Tableau, InvariantsHold) {
  RecordProperty("verifies", "REQ-SIM-001");
  const dyn::RkTableau& t = dyn::verner89();

  // Row sums of the coupling matrix equal the stage nodes: Sum_j a[i][j] = c[i].
  // This is the standard consistency condition and catches most coefficient
  // typos immediately.
  for (std::size_t i = 0; i < dyn::RkTableau::kStages; ++i) {
    double row = 0.0;
    for (std::size_t j = 0; j < i; ++j) {
      row += t.a[i][j];
    }
    EXPECT_NEAR(row, t.c[i], 1.0e-12) << "row sum != node at stage " << i;
  }

  // Advancing weights sum to 1 (first-order consistency); error weights sum to 0
  // (the two embedded solutions share the same leading quadrature).
  double sum_b = 0.0, sum_e = 0.0;
  for (std::size_t i = 0; i < dyn::RkTableau::kStages; ++i) {
    sum_b += t.b[i];
    sum_e += t.e[i];
  }
  EXPECT_NEAR(sum_b, 1.0, 1.0e-13);
  EXPECT_NEAR(sum_e, 0.0, 1.0e-13);
}

// --- Integrator order -------------------------------------------------------

TEST(Rk89Integrator, ErrorEstimateIsHighOrder) {
  RecordProperty("verifies", "REQ-SIM-001");
  // For y' = y the embedded error estimate is O(h^9); halving h must shrink it
  // by ~2^9 = 512. A wrong error weight would collapse this ratio.
  using Vec1 = Eigen::Matrix<double, 1, 1>;
  auto f = [](double, const Vec1& y) {
    Vec1 d;
    d[0] = y[0];
    return d;
  };
  Vec1 y0;
  y0[0] = 1.0;

  auto err_at = [&](double h) {
    Vec1 y_next, err;
    dyn::rk89_step<1>(dyn::verner89(), f, 0.0, y0, h, y_next, err);
    return std::abs(err[0]);
  };
  const double e1 = err_at(0.2);
  const double e2 = err_at(0.1);
  ASSERT_GT(e2, 0.0);
  const double ratio = e1 / e2;
  EXPECT_GT(ratio, 300.0) << "error-estimate order too low (ratio=" << ratio << ")";
}

// --- Kepler two-body: translational conservation ----------------------------

TEST(RigidBody6Dof, KeplerConservesEnergyAndMomentum) {
  RecordProperty("verifies", "REQ-SIM-001");
  const double mu = polaris::constants::wgs84::kGM;
  const double r0 = 7.0e6;  // ~630 km altitude
  const double v0 = std::sqrt(mu / r0);
  const double period = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / mu);

  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(r0, 0.0, 0.0);
  s.velocity = pm::Vec3<pf::ECI>(0.0, v0, 0.0);

  const dyn::TwoBodyGravity gravity(mu);
  const dyn::RigidBody6Dof body(Eigen::Matrix3d::Identity(), gravity);

  auto energy = [&](const ps::TruthState& x) {
    return 0.5 * x.velocity.squaredNorm() - mu / x.position.norm();
  };
  auto ang_mom = [&](const ps::TruthState& x) -> Eigen::Vector3d {
    return x.position.eigen().cross(x.velocity.eigen());
  };

  const double e0 = energy(s);
  const Eigen::Vector3d h0 = ang_mom(s);

  const ps::TruthState s1 = body.propagate(s, period);

  // Specific energy and angular momentum are constants of the two-body motion.
  EXPECT_NEAR(energy(s1), e0, std::abs(e0) * 1.0e-9);
  EXPECT_LT((ang_mom(s1) - h0).norm(), h0.norm() * 1.0e-9);
  // After exactly one period the orbit closes on itself.
  EXPECT_LT((s1.position - s.position).norm(), 1.0);
  EXPECT_LT((s1.velocity - s.velocity).norm(), v0 * 1.0e-6);
}

// --- Torque-free rotation: kinematics sign ----------------------------------

TEST(RigidBody6Dof, TorqueFreeAxisRotationMatchesClosedForm) {
  RecordProperty("verifies", "REQ-SIM-001");
  // Symmetric inertia (J = c I) makes the body rate constant under zero torque,
  // so the attitude is an exact axis-angle rotation about the rate axis. With a
  // *non-identity* initial attitude the closed form is delta(t) (x) q0 (LEFT
  // multiply) — that pins both the sign and the side of q' = 1/2 [0, w] (x) q
  // (a right-multiply bug survives an identity q0 but fails here).
  const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
  const double rate = 0.05;  // rad/s
  const double dt = 100.0;

  const pm::Quaternion q0 = pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.0, 1.0, 0.0), 0.7);

  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(7.0e6, 0.0, 0.0);  // arbitrary; unforced
  s.attitude = pm::Quat<pf::Body, pf::ECI>(q0);
  s.body_rate = pm::Vec3<pf::Body>(Eigen::Vector3d(rate * axis));

  const dyn::NoForceModel free_drift;
  const dyn::RigidBody6Dof body(2.5 * Eigen::Matrix3d::Identity(), free_drift);
  const ps::TruthState s1 = body.propagate(s, dt);

  const pm::Quaternion expected = pm::Quaternion::FromAxisAngle(axis, rate * dt) * q0;
  EXPECT_NEAR(s1.attitude.core().angularDistance(expected), 0.0, 1.0e-9);
  // Rate is unchanged for a symmetric body.
  EXPECT_LT((s1.body_rate - s.body_rate).norm(), 1.0e-12);
}

// --- Torque-free rotation: coupled Euler + kinematics conservation ----------

TEST(RigidBody6Dof, TorqueFreeConservesEnergyAndInertialMomentum) {
  RecordProperty("verifies", "REQ-SIM-001");
  // Asymmetric inertia -> genuine tumbling. Rotational KE and the *inertial*
  // angular-momentum vector are both conserved; the latter only holds if the
  // Euler dynamics and the quaternion kinematics are consistently coupled.
  Eigen::Matrix3d J = Eigen::Vector3d(1.0, 2.0, 3.0).asDiagonal();
  const Eigen::Vector3d w0(0.03, 0.02, -0.05);

  ps::TruthState s;
  s.body_rate = pm::Vec3<pf::Body>(w0);
  const dyn::NoForceModel free_drift;
  const dyn::RigidBody6Dof body(J, free_drift);

  auto kinetic = [&](const ps::TruthState& x) {
    const Eigen::Vector3d w = x.body_rate.eigen();
    return 0.5 * w.dot(J * w);
  };
  auto inertial_momentum = [&](const ps::TruthState& x) -> Eigen::Vector3d {
    // L_ECI = A(q)^T (J w_body), with A = Body<-ECI so A^T maps Body -> ECI.
    const Eigen::Matrix3d A = x.attitude.core().toRotationMatrix();
    return A.transpose() * (J * x.body_rate.eigen());
  };

  const double t0 = kinetic(s);
  const Eigen::Vector3d L0 = inertial_momentum(s);

  const ps::TruthState s1 = body.propagate(s, 200.0);

  EXPECT_NEAR(kinetic(s1), t0, t0 * 1.0e-9);
  EXPECT_LT((inertial_momentum(s1) - L0).norm(), L0.norm() * 1.0e-9);
}

TEST(RigidBody6Dof, NonPositiveDtIsAConsistentNoOp) {
  RecordProperty("verifies", "REQ-SIM-001");
  // A non-positive dt must return s0 verbatim — never a state advanced against a
  // rewound epoch (boundary guard, §3.6).
  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(7.0e6, 0.0, 0.0);
  s.velocity = pm::Vec3<pf::ECI>(0.0, 7.5e3, 0.0);
  s.body_rate = pm::Vec3<pf::Body>(0.0, 0.0, 0.01);

  const dyn::TwoBodyGravity gravity;
  const dyn::RigidBody6Dof body(Eigen::Matrix3d::Identity(), gravity);

  for (const double dt : {0.0, -10.0}) {
    const ps::TruthState s1 = body.propagate(s, dt);
    EXPECT_EQ(s1.epoch.nanosecondsSinceEpoch(), s.epoch.nanosecondsSinceEpoch());
    EXPECT_DOUBLE_EQ((s1.position - s.position).norm(), 0.0);
    EXPECT_DOUBLE_EQ((s1.velocity - s.velocity).norm(), 0.0);
    EXPECT_DOUBLE_EQ((s1.body_rate - s.body_rate).norm(), 0.0);
  }
}

}  // namespace

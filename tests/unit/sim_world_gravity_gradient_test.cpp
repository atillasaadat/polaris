/// @file Unit tests for the gravity-gradient disturbance torque (REQ-SIM-002;
/// design doc §5.3).
///
/// Two layers. The algebraic ones pin the torque law itself — the equilibria, the
/// 45° maximum, the vanishing for a spherical body — where the expected value is
/// obvious by inspection and a failure points at one line of code.
///
/// The libration test is the one that matters. It closes the loop against the
/// classical closed-form result (Hughes ch. 9 [hughes1986]; Wertz §18.2
/// [wertz1978]): a gravity-gradient body in a circular orbit, held near the
/// local-vertical/local-horizontal attitude, oscillates in pitch at
///
///   omega_pitch = n * sqrt( 3 (I_x - I_z) / I_y ),
///
/// with n the mean motion, I_x the along-track (roll-axis) moment, I_z the nadir
/// (yaw-axis) moment and I_y the orbit-normal (pitch-axis) moment. Nothing in the
/// implementation knows about that frequency: it falls out of the coupled 6DOF
/// integration of the torque against Euler's equation, so matching it exercises
/// the torque law, the frame handling and the plant together. The sign of
/// (I_x - I_z) is what separates the stable configuration (minimum inertia along
/// nadir) from the unstable one, and both are checked.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "dynamics/rigid_body.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/timescales.hpp"
#include "world/gravity_gradient.hpp"

namespace {

namespace dyn = polaris::sim::dynamics;
namespace pm = polaris::math;
namespace world = polaris::sim::world;

using Body = pm::Vec3<pm::frames::Body>;
using Eci = pm::Vec3<pm::frames::ECI>;
using BodyFromEci = pm::Quat<pm::frames::Body, pm::frames::ECI>;

constexpr double kMu = polaris::constants::wgs84::kGM;

/// A truth state at ECI position @p r with attitude @p q; nothing else matters to
/// this torque.
polaris::state::TruthState at(const Eigen::Vector3d& r,
                              const BodyFromEci& q = BodyFromEci::Identity()) {
  polaris::state::TruthState s{};
  s.position = Eci(r);
  s.attitude = q;
  return s;
}

}  // namespace

// --- The torque law ----------------------------------------------------------

TEST(GravityGradient, VanishesForASphericallySymmetricBody) {
  RecordProperty("verifies", "REQ-SIM-002");
  // J = I * identity makes J r_hat parallel to r_hat, so the cross product is
  // identically zero at every attitude. This is why a sphere cannot be
  // gravity-gradient stabilised.
  const Eigen::Matrix3d j = 12.0 * Eigen::Matrix3d::Identity();
  const world::GravityGradientTorque model(j, kMu);

  const std::vector<Eigen::Vector3d> axes = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                                             Eigen::Vector3d(1.0, 2.0, 3.0).normalized()};
  for (const Eigen::Vector3d& axis : axes) {
    const BodyFromEci q(pm::Quaternion::FromAxisAngle(axis, 0.7));
    // Zero to roundoff, against the 3 mu I / r^3 scale the terms are formed at
    // (~1e-5 N·m here) rather than against an absolute floor.
    EXPECT_LT(model.torque(at(Eigen::Vector3d(6.9e6, 1.0e6, -2.0e6), q)).eigen().norm(), 1.0e-18);
  }
  // A gradient torque is a couple, not a force.
  EXPECT_EQ(model.acceleration(at(Eigen::Vector3d(6.9e6, 0.0, 0.0))).eigen(),
            Eigen::Vector3d::Zero());
}

TEST(GravityGradient, VanishesWhenNadirLiesAlongAPrincipalAxis) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The three principal-axis alignments are the equilibria: J r_hat is parallel
  // to r_hat again, so the couple is zero. Two of them are stable and one is
  // not, but statically they are indistinguishable — all give zero torque.
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 8.0, 4.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  const double r = 6.9e6;

  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    direction[axis] = 1.0;
    // Identity attitude, so Body and ECI coincide and the position direction is
    // the body direction.
    EXPECT_LT(model.torque(at(r * direction)).eigen().norm(), 1.0e-25) << "axis " << axis;
    // Anti-parallel too: r_hat appears twice, so the sign cancels.
    EXPECT_LT(model.torque(at(-r * direction)).eigen().norm(), 1.0e-25) << "axis " << axis;
  }
}

TEST(GravityGradient, PeaksMidwayBetweenTwoPrincipalAxes) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Between the x and z principal axes the magnitude is
  // (3 mu / r^3) |I_x - I_z| sin(a) cos(a), maximal at 45 deg. Sweeping the angle
  // and checking both the maximum location and the closed-form value pins the
  // factor of 3 and the 1/r^3 together.
  const double ix = 10.0;
  const double iz = 4.0;
  const Eigen::Matrix3d j = Eigen::Vector3d(ix, 8.0, iz).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  const double r = 6.9e6;
  const double scale = 3.0 * kMu / (r * r * r);

  double best = -1.0;
  double best_angle = 0.0;
  for (int i = 0; i <= 90; ++i) {
    const double a = static_cast<double>(i) * M_PI / 180.0;
    const Eigen::Vector3d direction(std::sin(a), 0.0, std::cos(a));
    const Eigen::Vector3d tau = model.torque(at(r * direction)).eigen();
    // Closed form for a nadir vector in the x-z plane: r_hat x (J r_hat) has only
    // a y component, equal to (I_x - I_z) sin a cos a.
    const double expected = scale * (ix - iz) * std::sin(a) * std::cos(a);
    EXPECT_NEAR(tau.y(), expected, 1.0e-12 * scale * ix);
    EXPECT_LT(std::hypot(tau.x(), tau.z()), 1.0e-12 * scale * ix);
    if (tau.norm() > best) {
      best = tau.norm();
      best_angle = a;
    }
  }
  EXPECT_NEAR(best_angle, M_PI / 4.0, 1.0e-9);
  EXPECT_NEAR(best, scale * (ix - iz) * 0.5, 1.0e-9 * scale * ix);
}

TEST(GravityGradient, ScalesAsTheInverseCubeOfRadius) {
  RecordProperty("verifies", "REQ-SIM-002");
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 8.0, 4.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 0.0, 1.0).normalized();

  const double near = model.torque(at(7.0e6 * direction)).eigen().norm();
  const double far = model.torque(at(14.0e6 * direction)).eigen().norm();
  EXPECT_NEAR(near / far, 8.0, 1.0e-9);
}

TEST(GravityGradient, TakesTheNadirDirectionIntoTheBodyFrame) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The inertia tensor lives in Body, so the position direction has to be rotated
  // in. Rotating the vehicle while holding the orbit fixed must change the torque
  // — a frame-blind implementation would return the same value.
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 8.0, 4.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  const Eigen::Vector3d r(6.9e6, 0.0, 0.0);

  // Identity attitude puts nadir on the body x principal axis: an equilibrium.
  EXPECT_LT(model.torque(at(r)).eigen().norm(), 1.0e-25);

  // Rotating 45 deg about body y moves it midway between x and z: the maximum.
  const BodyFromEci q(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), M_PI / 4.0));
  const double scale = 3.0 * kMu / (r.norm() * r.norm() * r.norm());
  EXPECT_NEAR(model.torque(at(r, q)).eigen().norm(), scale * (10.0 - 4.0) * 0.5,
              1.0e-9 * scale * 10.0);
}

TEST(GravityGradient, MatchesAHandWorkedCase) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Identity attitude, r = -R (1,1,1)/sqrt(3), so the Body nadir direction is
  // (1,1,1)/sqrt(3). With J = diag(10,20,30):
  //   tau = 3 mu/R^3 * r_hat x (J r_hat) = mu/R^3 (10,-20,10).
  // (Moved here from the spherical-harmonic gravity tests when the torque became
  // its own provider — it is the same hand computation.)
  const double r = 7.0e6;
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);

  const Eigen::Vector3d tau =
      model.torque(at(-r * Eigen::Vector3d(1.0, 1.0, 1.0).normalized())).eigen();
  const Eigen::Vector3d expected = (kMu / (r * r * r)) * Eigen::Vector3d(10.0, -20.0, 10.0);
  EXPECT_LT((tau - expected).norm(), expected.norm() * 1.0e-12);
}

TEST(GravityGradient, UsesTheAttitudeRotationInTheRightDirection) {
  RecordProperty("verifies", "REQ-SIM-002");
  // A genuine non-identity attitude, so an A <-> A^T mix-up cannot pass silently.
  // Attitude = ROT3(30 deg) about ECI Z, position along +X, so in Body the nadir
  // direction is (-cos30, sin30, 0). With J = diag(10,20,30) that gives
  //   r_hat x (J r_hat) = (0, 0, -5 sqrt3 / 2),
  // and the sign flips under A^T.
  const double r = 7.0e6;
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);

  const BodyFromEci q(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), M_PI / 6.0));
  const Eigen::Vector3d tau = model.torque(at(Eigen::Vector3d(r, 0.0, 0.0), q)).eigen();
  const Eigen::Vector3d expected =
      (3.0 * kMu / (r * r * r)) * Eigen::Vector3d(0.0, 0.0, -5.0 * std::sqrt(3.0) / 2.0);
  EXPECT_LT((tau - expected).norm(), expected.norm() * 1.0e-9);
}

TEST(GravityGradient, GuardsTheSingularRadius) {
  RecordProperty("verifies", "REQ-SIM-002");
  const Eigen::Matrix3d j = Eigen::Vector3d(10.0, 8.0, 4.0).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  EXPECT_EQ(model.torque(at(Eigen::Vector3d::Zero())).eigen(), Eigen::Vector3d::Zero());
}

TEST(GravityGradient, HasAPlausibleMagnitudeInLowEarthOrbit) {
  RecordProperty("verifies", "REQ-SIM-002");
  // A 6U-class body with a few kg·m² of inertia asymmetry at 500 km sits in the
  // 1e-8..1e-6 N·m band — the same order as the residual-dipole torque, and the
  // reason gravity gradient dominates the LEO momentum budget for elongated
  // bodies. Pins the units: a km/m or a mu slip lands decades away.
  const Eigen::Matrix3d j = Eigen::Vector3d(0.12, 0.10, 0.05).asDiagonal();
  const world::GravityGradientTorque model(j, kMu);
  const BodyFromEci q(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), M_PI / 4.0));
  const double magnitude = model.torque(at(Eigen::Vector3d(6.871e6, 0.0, 0.0), q)).eigen().norm();
  EXPECT_GT(magnitude, 1.0e-9);
  EXPECT_LT(magnitude, 1.0e-6);
}

// --- Pitch libration against the closed form ---------------------------------

namespace {

/// One circular-orbit gravity-gradient run: the body starts at the LVLH attitude
/// with a small pitch offset and no libration rate, and the pitch angle is
/// sampled as it swings.
struct LibrationRun {
  std::vector<double> t_s;
  std::vector<double> pitch_rad;
};

/// Propagate a body in a circular equatorial orbit under two-body gravity plus
/// the gravity-gradient torque, sampling the pitch angle relative to the orbit
/// frame.
///
/// The orbit frame is the usual one: +z nadir, +y along the negative orbit
/// normal, +x completing it (along-track for a circular orbit). Body axes start
/// aligned with it, so the configured inertia is directly (I_roll, I_pitch,
/// I_yaw) and the closed form applies to the diagonal entries as written.
LibrationRun runLibration(const Eigen::Vector3d& principal_moments, double radius_m,
                          double pitch0_rad, double duration_s, double sample_s) {
  const Eigen::Matrix3d j = principal_moments.asDiagonal();
  const double n = std::sqrt(kMu / (radius_m * radius_m * radius_m));

  dyn::TwoBodyGravity gravity(kMu);
  world::GravityGradientTorque gg(j, kMu);
  dyn::CompositeForceModel composite;
  composite.add(&gravity);
  composite.add(&gg);
  const dyn::RigidBody6Dof plant(j, composite);

  // Orbit in the ECI x-y plane, starting on +x moving toward +y, so the orbit
  // normal is +z.
  const Eigen::Vector3d r0(radius_m, 0.0, 0.0);
  const Eigen::Vector3d v0(0.0, n * radius_m, 0.0);

  // Orbit-frame axes in ECI at t = 0: z_o = -r_hat, y_o = -h_hat, x_o = y_o x z_o.
  const Eigen::Vector3d z_o = -r0.normalized();
  const Eigen::Vector3d y_o = -r0.cross(v0).normalized();
  const Eigen::Vector3d x_o = y_o.cross(z_o);
  Eigen::Matrix3d a_body_from_eci;  // rows are the body axes expressed in ECI
  a_body_from_eci.row(0) = x_o.transpose();
  a_body_from_eci.row(1) = y_o.transpose();
  a_body_from_eci.row(2) = z_o.transpose();

  // Offset in pitch (about the body y axis) and give the body exactly the orbit
  // rate, so the libration starts at rest at its amplitude: theta(t) =
  // theta0 cos(omega_p t) with no sine term to muddy the zero crossing.
  const Eigen::Matrix3d pitch =
      Eigen::AngleAxisd(pitch0_rad, Eigen::Vector3d::UnitY()).toRotationMatrix().transpose();
  const Eigen::Matrix3d a0 = pitch * a_body_from_eci;

  polaris::state::TruthState s{};
  s.epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(0);
  s.position = Eci(r0);
  s.velocity = Eci(v0);
  s.attitude = BodyFromEci(pm::Quaternion::FromRotationMatrix(a0));
  // The orbit frame rotates about the orbit normal at n; in body coordinates
  // that is n along -y (y_o = -h_hat), up to the small pitch offset.
  s.body_rate = Body(Eigen::Vector3d(0.0, -n, 0.0));

  dyn::StepControl ctl;
  ctl.abs_tol = 1.0e-13;
  ctl.rel_tol = 1.0e-13;
  ctl.max_step = 10.0;

  LibrationRun out;
  for (double t = 0.0; t <= duration_s + 0.5 * sample_s; t += sample_s) {
    if (t > 0.0) {
      s = plant.propagate(s, sample_s, ctl);
    }
    // Rebuild the orbit frame from the propagated state rather than from the
    // analytic one: if the orbit itself drifted, the pitch angle must be measured
    // against where the vehicle actually is.
    const Eigen::Vector3d r = s.position.eigen();
    const Eigen::Vector3d z_orbit = -r.normalized();
    const Eigen::Vector3d y_orbit = -r.cross(s.velocity.eigen()).normalized();
    const Eigen::Matrix3d a = s.attitude.core().toRotationMatrix();
    const Eigen::Vector3d z_body = a.transpose() * Eigen::Vector3d::UnitZ();
    // Signed rotation of the body nadir axis about the pitch axis.
    const double theta = std::atan2(z_orbit.cross(z_body).dot(y_orbit), z_orbit.dot(z_body));
    out.t_s.push_back(t);
    out.pitch_rad.push_back(theta);
  }
  return out;
}

/// First sign change in @p run, refined by linear interpolation. Returns a
/// negative time if the series never crosses zero.
double firstZeroCrossing(const LibrationRun& run) {
  for (std::size_t i = 1; i < run.pitch_rad.size(); ++i) {
    const double a = run.pitch_rad[i - 1];
    const double b = run.pitch_rad[i];
    if (a == 0.0) {
      return run.t_s[i - 1];
    }
    if ((a < 0.0) != (b < 0.0)) {
      const double frac = a / (a - b);
      return run.t_s[i - 1] + frac * (run.t_s[i] - run.t_s[i - 1]);
    }
  }
  return -1.0;
}

}  // namespace

TEST(GravityGradientLibration, MatchesTheClosedFormPitchFrequency) {
  RecordProperty("verifies", "REQ-SIM-002");
  // I_roll = 10, I_pitch = 8, I_yaw = 4, so
  //   omega_p = n sqrt(3 (I_x - I_z) / I_y) = n sqrt(3 * 6 / 8) = 1.5 n,
  // a libration period of exactly two thirds of an orbit. The quarter period —
  // the first zero crossing of theta(t) = theta0 cos(omega_p t) — is the
  // sensitive place to measure: a 1% frequency error moves it by 1%, whereas the
  // amplitude at a full period would move by only 0.05%.
  const double radius = 6.878137e6;  // 500 km circular
  const Eigen::Vector3d moments(10.0, 8.0, 4.0);
  const double n = std::sqrt(kMu / (radius * radius * radius));
  const double omega_p = n * std::sqrt(3.0 * (moments.x() - moments.z()) / moments.y());
  ASSERT_NEAR(omega_p / n, 1.5, 1.0e-12) << "test fixture no longer the 1.5n case";

  const double quarter_period = (M_PI / 2.0) / omega_p;
  // 1 mrad amplitude: the exact pitch equation carries a sin(2 theta), so the
  // small-angle frequency is low by ~theta0^2/16 ~ 6e-8 relative — far below the
  // tolerance and far below the sampling resolution.
  const LibrationRun run = runLibration(moments, radius, 1.0e-3, 1.6 * quarter_period, 1.0);

  const double measured = firstZeroCrossing(run);
  ASSERT_GT(measured, 0.0) << "pitch never crossed zero";
  EXPECT_NEAR(measured, quarter_period, 1.0e-5 * quarter_period)
      << "measured quarter period " << measured << " s vs closed form " << quarter_period << " s";

  // And it really is an oscillation about the LVLH attitude, not a drift: the
  // amplitude at t = 0 is the offset, and the swing stays bounded by it.
  EXPECT_NEAR(run.pitch_rad.front(), 1.0e-3, 1.0e-9);
  for (double theta : run.pitch_rad) {
    EXPECT_LT(std::abs(theta), 1.05e-3);
  }
}

TEST(GravityGradientLibration, DivergesWhenTheMaximumInertiaAxisPointsAtNadir) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Swap roll and yaw: I_x < I_z, so 3 n^2 (I_x - I_z) / I_y is negative and the
  // pitch equation has real exponents instead of imaginary ones. This is the
  // unstable equilibrium — statically indistinguishable from the stable one (both
  // are zero-torque), which is exactly why it needs a dynamic test.
  const double radius = 6.878137e6;
  const Eigen::Vector3d moments(4.0, 8.0, 10.0);
  const double n = std::sqrt(kMu / (radius * radius * radius));
  const double rate = n * std::sqrt(3.0 * (moments.z() - moments.x()) / moments.y());

  // Two e-foldings of the growing mode. theta(t) = theta0 cosh(rate t) for a
  // release from rest, so growth by cosh(2) ~ 3.76 is the prediction.
  const double duration = 2.0 / rate;
  const LibrationRun run = runLibration(moments, radius, 1.0e-3, duration, 1.0);

  ASSERT_LT(firstZeroCrossing(run), 0.0) << "an unstable mode must not oscillate";
  const double growth = run.pitch_rad.back() / run.pitch_rad.front();
  EXPECT_NEAR(growth, std::cosh(2.0), 0.02 * std::cosh(2.0)) << "growth " << growth;
}

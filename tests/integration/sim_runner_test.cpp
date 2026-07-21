/// @file Integration tests for the assembled truth sim (REQ-SIM-001/002;
/// design doc §5, §23.1).
///
/// Unlike the unit tests, these exercise the whole stack in one process: the
/// compiled config, the loaded reference-data products (EOP, DE440, IGRF,
/// EGM2008), every wired resolver, the composed force model, and the RK89 plant.
///
/// The layering is deliberate. Conservation laws are checked against the
/// *analytic* baselines first — free drift and pure two-body, where the right
/// answer is known exactly and any drift is integrator error, not modelling. Only
/// then does the full perturbed stack run, where the checks necessarily become
/// bounds rather than identities. A failure in the first group localises to the
/// integrator; a failure only in the second localises to a model or its wiring.

#include "scenario/sim_runner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <Eigen/Core>
#include <fstream>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "scenario/sim_config.hpp"
#include "time/leap_seconds.hpp"

namespace {

namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;

constexpr double kMu = polaris::constants::wgs84::kGM;

scenario::DataPaths dataPaths() {
  return scenario::DataPaths::under(POLARIS_GOLDEN_DIR);
}

/// A 500 km circular orbit, built in code so the conservation tests do not
/// depend on whatever the committed scenario happens to say.
scenario::SimConfig circularOrbit(double duration_s, double output_step_s) {
  scenario::SimConfig c;
  c.scenario_name = "test-circular";
  c.spacecraft.name = "test-vehicle";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();

  const double radius = 6878137.0;
  const double speed = std::sqrt(kMu / radius);
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);  // ~2026-01-01
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(radius, 0.0, 0.0));
  c.initial_state.velocity = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, speed, 0.0));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.001, -0.002, 0.0015));

  c.environment.gravity_degree = -1;  // free drift unless a test says otherwise
  c.environment.magnetic_field = scenario::MagneticModel::kNone;
  c.propagation.duration_s = duration_s;
  c.propagation.output_step_s = output_step_s;
  return c;
}

double specificEnergy(const polaris::state::TruthState& s) {
  const double r = s.position.eigen().norm();
  const double v = s.velocity.eigen().norm();
  return 0.5 * v * v - kMu / r;
}

Eigen::Vector3d orbitalAngularMomentum(const polaris::state::TruthState& s) {
  return s.position.eigen().cross(s.velocity.eigen());
}

}  // namespace

// --- Analytic baselines ------------------------------------------------------

TEST(SimIntegration, FreeDriftIsAStraightLineAtConstantVelocity) {
  // No gravity, no perturbations: Newton's first law. Any deviation is the
  // integrator or the plant wiring, with no model in between to blame.
  scenario::SimConfig config = circularOrbit(600.0, 60.0);
  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
  EXPECT_EQ(runner.modelCount(), 0u) << "free drift should compose no models";

  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;
  ASSERT_GE(trajectory.size(), 2u);

  const Eigen::Vector3d r0 = config.initial_state.position.eigen();
  const Eigen::Vector3d v0 = config.initial_state.velocity.eigen();
  for (const scenario::TrajectorySample& sample : trajectory) {
    const Eigen::Vector3d expected = r0 + v0 * sample.t_s;
    EXPECT_LT((sample.state.position.eigen() - expected).norm(), 1.0e-6) << "t=" << sample.t_s;
    EXPECT_LT((sample.state.velocity.eigen() - v0).norm(), 1.0e-9) << "t=" << sample.t_s;
  }
}

TEST(SimIntegration, TorqueFreeRotationConservesAngularMomentum) {
  // With no torque, H = J*omega is constant in INERTIAL space while omega itself
  // is not (that is the whole content of Euler's equation). Checking |H| in the
  // body frame is therefore the meaningful invariant.
  scenario::SimConfig config = circularOrbit(600.0, 30.0);
  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;

  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;

  const Eigen::Matrix3d inertia = config.spacecraft.inertia_kgm2;
  const double h0 = (inertia * config.initial_state.body_rate.eigen()).norm();
  ASSERT_GT(h0, 0.0);
  for (const scenario::TrajectorySample& sample : trajectory) {
    const double h = (inertia * sample.state.body_rate.eigen()).norm();
    EXPECT_NEAR(h / h0, 1.0, 1.0e-10) << "t=" << sample.t_s;
    // And the quaternion stays on the unit manifold — the plant renormalises
    // after every accepted step, so this is a check that it actually runs.
    EXPECT_NEAR(sample.state.attitude.core().coeffs().norm(), 1.0, 1.0e-12);
  }
}

TEST(SimIntegration, TwoBodyConservesEnergyAndAngularMomentum) {
  // Pure Kepler: both the specific orbital energy and the angular-momentum
  // vector are exact invariants, so this measures integrator quality alone.
  scenario::SimConfig config = circularOrbit(5400.0, 60.0);
  config.environment.gravity_degree = 0;  // point mass

  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
  EXPECT_EQ(runner.modelCount(), 1u);

  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;

  const double e0 = specificEnergy(config.initial_state);
  const Eigen::Vector3d h0 = orbitalAngularMomentum(config.initial_state);
  for (const scenario::TrajectorySample& sample : trajectory) {
    EXPECT_NEAR(specificEnergy(sample.state) / e0, 1.0, 1.0e-12) << "t=" << sample.t_s;
    const Eigen::Vector3d h = orbitalAngularMomentum(sample.state);
    EXPECT_LT((h - h0).norm() / h0.norm(), 1.0e-12) << "t=" << sample.t_s;
  }
}

TEST(SimIntegration, TwoBodyOrbitClosesAfterOneKeplerPeriod) {
  // A circular two-body orbit returns exactly to its start after T = 2*pi*
  // sqrt(a^3/mu). This catches a wrong mu, a wrong time base, or a step
  // controller that quietly loses time — none of which the conservation checks
  // above would notice, since they hold for any consistent-but-wrong rate.
  const double radius = 6878137.0;
  const double period = 2.0 * M_PI * std::sqrt(radius * radius * radius / kMu);

  scenario::SimConfig config = circularOrbit(period, period / 8.0);
  config.environment.gravity_degree = 0;

  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;

  const Eigen::Vector3d r0 = config.initial_state.position.eigen();
  const Eigen::Vector3d closed = trajectory.back().state.position.eigen();
  // Sub-millimetre over a 43 000 km path.
  EXPECT_LT((closed - r0).norm(), 1.0e-3);
  EXPECT_NEAR(trajectory.back().t_s, period, 1.0e-9);
}

// --- The full perturbed stack ------------------------------------------------

TEST(SimIntegration, FullEnvironmentPropagatesACompleteOrbit) {
  // Everything on: EGM2008 to degree 8 in the ECEF frame, Sun and Moon third
  // bodies from the DE440 fit, SRP with conical eclipse, drag on the exponential
  // atmosphere, and the IGRF residual-dipole torque. This is the assembly the
  // whole push exists to produce.
  scenario::SimConfig config = circularOrbit(5677.0, 30.0);
  config.environment.gravity_degree = 8;
  config.environment.sun_third_body = true;
  config.environment.moon_third_body = true;
  config.environment.srp_enabled = true;
  config.environment.drag_enabled = true;
  config.environment.magnetic_field = scenario::MagneticModel::kIgrf;
  config.spacecraft.drag_area_m2 = 0.06;
  config.spacecraft.srp_area_m2 = 0.06;
  config.spacecraft.residual_dipole_am2 =
      pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.002, -0.001, 0.0015));

  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
  EXPECT_EQ(runner.modelCount(), 5u) << "gravity + third-body + SRP + drag + dipole";

  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;
  ASSERT_GT(trajectory.size(), 100u);

  const double r_earth = polaris::constants::wgs84::kSemiMajorAxis;
  for (const scenario::TrajectorySample& sample : trajectory) {
    const Eigen::Vector3d r = sample.state.position.eigen();
    ASSERT_TRUE(r.allFinite()) << "t=" << sample.t_s;
    ASSERT_TRUE(sample.state.velocity.eigen().allFinite()) << "t=" << sample.t_s;
    // The orbit stays an orbit: no re-entry, no escape. A wired-up-wrong model
    // (sign error, unit error) shows up here as a trajectory that leaves the
    // band within one revolution.
    const double altitude = r.norm() - r_earth;
    EXPECT_GT(altitude, 400.0e3) << "t=" << sample.t_s;
    EXPECT_LT(altitude, 600.0e3) << "t=" << sample.t_s;
    EXPECT_NEAR(sample.state.attitude.core().coeffs().norm(), 1.0, 1.0e-12);
  }
}

TEST(SimIntegration, DragRemovesOrbitalEnergy) {
  // Drag is dissipative, so the semi-major axis must DECREASE monotonically over
  // a full revolution — never increase. Sign errors in the relative-velocity
  // term produce an orbit that gains energy, which looks superficially fine on a
  // single sample and is unmistakable here.
  auto finalSma = [](bool drag_enabled) {
    scenario::SimConfig config = circularOrbit(5677.0, 300.0);
    config.environment.gravity_degree = 0;
    config.environment.drag_enabled = drag_enabled;
    // A deliberately large area-to-mass ratio so one orbit of decay is far above
    // integrator noise.
    config.spacecraft.drag_area_m2 = 5.0;
    scenario::SimRunner runner;
    std::string error;
    EXPECT_TRUE(runner.build(config, dataPaths(), &error)) << error;
    std::vector<scenario::TrajectorySample> trajectory;
    EXPECT_TRUE(runner.run(trajectory, &error)) << error;
    return -kMu / (2.0 * specificEnergy(trajectory.back().state));
  };

  const double without_drag = finalSma(false);
  const double with_drag = finalSma(true);
  EXPECT_LT(with_drag, without_drag) << "drag must remove energy, not add it";
  // And the loss is physically sized, not a runaway.
  EXPECT_GT(without_drag - with_drag, 1.0);
  EXPECT_LT(without_drag - with_drag, 50.0e3);
}

TEST(SimIntegration, RunsAreBitReproducible) {
  // Determinism is mandatory (sim/CLAUDE.md): a run must be bit-reproducible
  // from {config, seed}. Nothing here is stochastic yet, so this pins that no
  // uninitialised memory or address-dependent ordering has crept into the wiring.
  scenario::SimConfig config = circularOrbit(1800.0, 60.0);
  config.environment.gravity_degree = 8;
  config.environment.magnetic_field = scenario::MagneticModel::kIgrf;
  config.spacecraft.residual_dipole_am2 =
      pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.002, -0.001, 0.0015));

  std::vector<scenario::TrajectorySample> first;
  std::vector<scenario::TrajectorySample> second;
  std::string error;
  for (std::vector<scenario::TrajectorySample>* out : {&first, &second}) {
    scenario::SimRunner runner;
    ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
    ASSERT_TRUE(runner.run(*out, &error)) << error;
  }

  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].state.position.eigen(), second[i].state.position.eigen()) << "sample " << i;
    EXPECT_EQ(first[i].state.velocity.eigen(), second[i].state.velocity.eigen()) << "sample " << i;
    EXPECT_EQ(first[i].state.body_rate.eigen(), second[i].state.body_rate.eigen())
        << "sample " << i;
  }
}

TEST(SimIntegration, EpochAdvancesWithinTheIntegrationNotJustBetweenSteps) {
  // The plant evaluates its derivative at the RUNNING epoch. If it instead froze
  // the epoch per call, a single long propagation and a sequence of short ones
  // would disagree by far more than integrator tolerance, because the Earth,
  // Sun, and Moon would be held still across the long call.
  scenario::SimConfig config = circularOrbit(1200.0, 1200.0);  // one long step
  config.environment.gravity_degree = 8;                       // ECEF frame -> Earth rotation
  scenario::SimRunner one_shot;
  std::string error;
  ASSERT_TRUE(one_shot.build(config, dataPaths(), &error)) << error;
  std::vector<scenario::TrajectorySample> long_run;
  ASSERT_TRUE(one_shot.run(long_run, &error)) << error;

  config.propagation.output_step_s = 60.0;  // twenty short steps
  scenario::SimRunner stepped;
  ASSERT_TRUE(stepped.build(config, dataPaths(), &error)) << error;
  std::vector<scenario::TrajectorySample> short_run;
  ASSERT_TRUE(stepped.run(short_run, &error)) << error;

  const Eigen::Vector3d a = long_run.back().state.position.eigen();
  const Eigen::Vector3d b = short_run.back().state.position.eigen();
  ASSERT_NEAR(long_run.back().t_s, short_run.back().t_s, 1.0e-9);
  // Both see the same time-varying environment, so they agree to integrator
  // tolerance. With a frozen epoch the tesseral field would be misplaced by
  // 20 minutes of Earth rotation (5 degrees) and this would be kilometres.
  EXPECT_LT((a - b).norm(), 1.0);
}

// --- Configuration errors ----------------------------------------------------

TEST(SimIntegration, RefusesAMissingDataProduct) {
  scenario::SimConfig config = circularOrbit(60.0, 60.0);
  config.environment.gravity_degree = 8;
  scenario::DataPaths paths = dataPaths();
  paths.gravity = "/nonexistent/EGM2008.gfc";

  scenario::SimRunner runner;
  std::string error;
  EXPECT_FALSE(runner.build(config, paths, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(runner.ready());
}

TEST(SimIntegration, RefusesToRunBeforeBuild) {
  const scenario::SimRunner runner;
  std::vector<scenario::TrajectorySample> trajectory;
  std::string error;
  EXPECT_FALSE(runner.ready());
  EXPECT_FALSE(runner.run(trajectory, &error));
  EXPECT_NE(error.find("build"), std::string::npos) << error;
}

TEST(SimIntegration, WritesATraceableTrajectoryCsv) {
  scenario::SimConfig config = circularOrbit(120.0, 60.0);
  config.config_hash = "deadbeefcafe";
  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(config, dataPaths(), &error)) << error;
  std::vector<scenario::TrajectorySample> trajectory;
  ASSERT_TRUE(runner.run(trajectory, &error)) << error;

  const std::string path = testing::TempDir() + "/polaris_trajectory.csv";
  ASSERT_TRUE(scenario::writeTrajectoryCsv(path, config, trajectory, &error)) << error;

  std::ifstream file(path);
  ASSERT_TRUE(file.good());
  std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  // The config hash is what makes an output file traceable back to its input.
  EXPECT_NE(text.find("deadbeefcafe"), std::string::npos);
  EXPECT_NE(text.find("t_s,x_m,y_m,z_m"), std::string::npos);
  std::remove(path.c_str());
}

// --- Regressions -------------------------------------------------------------

TEST(SimIntegration, AFailedRebuildLeavesTheRunnerUnusableNotStale) {
  // Regression: build() reassigns the composite before the fallible data loads.
  // If a failed rebuild left `body_` pointing at the freed previous composite,
  // ready() would still report true and run() would use freed memory. A failed
  // build must invalidate the runner outright.
  scenario::SimConfig good = circularOrbit(120.0, 60.0);
  good.environment.gravity_degree = 0;

  scenario::SimRunner runner;
  std::string error;
  ASSERT_TRUE(runner.build(good, dataPaths(), &error)) << error;
  ASSERT_TRUE(runner.ready());

  scenario::SimConfig bad = good;
  bad.environment.gravity_degree = 8;  // now needs the .gfc
  scenario::DataPaths broken = dataPaths();
  broken.gravity = "/nonexistent/EGM2008.gfc";

  EXPECT_FALSE(runner.build(bad, broken, &error));
  EXPECT_FALSE(runner.ready()) << "a failed rebuild must not leave the old plant reachable";

  std::vector<scenario::TrajectorySample> trajectory;
  EXPECT_FALSE(runner.run(trajectory, &error));

  // And the runner is still usable after a successful rebuild.
  ASSERT_TRUE(runner.build(good, dataPaths(), &error)) << error;
  EXPECT_TRUE(runner.ready());
  EXPECT_TRUE(runner.run(trajectory, &error)) << error;
  EXPECT_GT(trajectory.size(), 1u);
}

TEST(SimIntegration, EclipseCanBeDisabled) {
  // Regression: `eclipse_enabled` was parsed and then never consulted, so a
  // scenario asking for no shadow silently got the full conical eclipse. The
  // flag has to change the answer.
  auto finalPosition = [](bool eclipse_enabled) {
    scenario::SimConfig config = circularOrbit(5677.0, 600.0);
    config.environment.gravity_degree = 0;
    config.environment.srp_enabled = true;
    config.environment.eclipse_enabled = eclipse_enabled;
    // A large sail so one orbit of SRP is well above integrator noise.
    config.spacecraft.srp_area_m2 = 200.0;
    scenario::SimRunner runner;
    std::string error;
    EXPECT_TRUE(runner.build(config, dataPaths(), &error)) << error;
    std::vector<scenario::TrajectorySample> trajectory;
    EXPECT_TRUE(runner.run(trajectory, &error)) << error;
    return trajectory.back().state.position.eigen();
  };

  const Eigen::Vector3d shadowed = finalPosition(true);
  const Eigen::Vector3d sunlit = finalPosition(false);
  // A LEO orbit spends roughly a third of each revolution in shadow, so
  // switching eclipse off leaves SRP acting the whole way round and the two
  // trajectories must visibly diverge.
  EXPECT_GT((shadowed - sunlit).norm(), 1.0);
}

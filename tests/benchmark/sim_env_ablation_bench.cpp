/// @file Environment-cost ablation benchmark for the truth sim (design doc
/// §5.1, §23.1; the recorded "per-micro-step environment evaluations" item).
///
/// Times the §2.4 closed loop over the same short arc with one environment
/// contributor removed per variant; the delta against the full-environment
/// baseline attributes wall-clock cost model by model. Attribution by ablation
/// rather than a sampling profiler because it needs no toolchain (WSL ships no
/// perf) and measures the system exactly as the SITL rows fly it — the same
/// technique that caught the GEO SRP fixture defect.
///
/// A standalone executable, deliberately NOT registered with ctest: this is a
/// measurement instrument, not a pass/fail gate. Run it from the repo root:
///
///     ./build-fprime-automatic-native-ut/bin/Linux/polaris_sim_env_bench [sim_seconds]
///
/// The scenario mirrors the SITL attitude rows' regime: 500 km orbit, full
/// environment, a 5 deg/s tumble (attitude stiffness is what drives the RK89
/// micro-step count), 10 Hz macro rate, no-op FSW.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Core>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "io/closed_loop.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"
#include "sitl_harness.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;
namespace pc = polaris::constants;

constexpr double kDeg2Rad = 0.017453292519943295;

/// The SITL-row regime: full environment on, tumbling.
scenario::SimConfig baseline(double duration_s) {
  scenario::SimConfig c;
  c.scenario_name = "env-ablation-bench";
  c.spacecraft.name = "bench-vehicle";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  c.spacecraft.drag_area_m2 = 0.06;
  c.spacecraft.srp_area_m2 = 0.06;

  const double radius = 6.878137e6;
  const double speed = std::sqrt(pc::wgs84::kGM / radius);
  const double inc = 45.0 * kDeg2Rad;
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(radius, 0.0, 0.0));
  c.initial_state.velocity =
      pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, speed * std::cos(inc), speed * std::sin(inc)));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  c.initial_state.body_rate =
      pm::Vec3<pm::frames::Body>(Eigen::Vector3d(5.0, -3.0, 4.0) * kDeg2Rad);

  c.environment.gravity_degree = 8;
  c.environment.sun_third_body = true;
  c.environment.moon_third_body = true;
  c.environment.srp_enabled = true;
  c.environment.drag_enabled = true;
  c.environment.magnetic_field = scenario::MagneticModel::kIgrf;
  c.environment.gravity_gradient_torque_enabled = true;
  c.environment.aero_torque_enabled = true;
  c.environment.srp_torque_enabled = true;
  c.environment.residual_dipole_torque_enabled = true;

  c.propagation.duration_s = duration_s;
  c.propagation.output_step_s = duration_s;
  c.propagation.fsw_rate_hz = 10.0;
  return c;
}

/// Wall-clock one closed-loop flight of *config* carrying *suite*; seconds (< 0 on error).
double flyOnce(const scenario::SimConfig& config,
               const scenario::SpacecraftConfig& suite = scenario::SpacecraftConfig{}) {
  scenario::Vehicle vehicle;
  std::string error;
  if (!scenario::buildVehicle(suite, 1, vehicle, &error)) {
    std::fprintf(stderr, "buildVehicle: %s\n", error.c_str());
    return -1.0;
  }
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  if (!runner.build(config, scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error,
                    loop.wrench())) {
    std::fprintf(stderr, "build: %s\n", error.c_str());
    return -1.0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (!loop.run({}, nullptr, &error)) {
    std::fprintf(stderr, "run: %s\n", error.c_str());
    return -1.0;
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Variant {
  const char* name;
  void (*mutate)(scenario::SimConfig&);
};

const Variant kVariants[] = {
    {"full (baseline)", [](scenario::SimConfig&) {}},
    {"no drag/atmosphere",
     [](scenario::SimConfig& c) {
       c.environment.drag_enabled = false;
       c.environment.aero_torque_enabled = false;
     }},
    {"no magnetic field",
     [](scenario::SimConfig& c) {
       c.environment.magnetic_field = scenario::MagneticModel::kNone;
       c.environment.residual_dipole_torque_enabled = false;
     }},
    {"no SRP",
     [](scenario::SimConfig& c) {
       c.environment.srp_enabled = false;
       c.environment.srp_torque_enabled = false;
     }},
    {"no third bodies",
     [](scenario::SimConfig& c) {
       c.environment.sun_third_body = false;
       c.environment.moon_third_body = false;
     }},
    {"gravity degree 2", [](scenario::SimConfig& c) { c.environment.gravity_degree = 2; }},
    {"gravity point-mass", [](scenario::SimConfig& c) { c.environment.gravity_degree = 0; }},
    {"tol 1e-9 (vs 1e-12)",
     [](scenario::SimConfig& c) {
       c.propagation.abs_tol = 1.0e-9;
       c.propagation.rel_tol = 1.0e-9;
     }},
    {"no tumble (rate=0)",
     [](scenario::SimConfig& c) {
       c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero());
     }},
};

/// Phase 2 — vehicle-model ablation in the SITL rows' own regime
/// (`estimationOrbit`: two-body, IGRF only), which the env phase showed is
/// nearly free without a vehicle. Attribution: remove one sensor class per
/// variant, and one variant drops every 100 Hz rate to 10 Hz — separating
/// "the models are expensive" from "the 100 Hz sample events fragment the
/// integration grid 10x, and every fragment pays RK89's 16 stages".
scenario::SpacecraftConfig without(scenario::SpacecraftConfig sc, const std::string& kind) {
  sc.sensors.erase(std::remove_if(sc.sensors.begin(), sc.sensors.end(),
                                  [&](const scenario::UnitConfig& u) { return u.kind == kind; }),
                   sc.sensors.end());
  return sc;
}

scenario::SpacecraftConfig ratesAt10Hz(scenario::SpacecraftConfig sc) {
  for (scenario::UnitConfig& u : sc.sensors) {
    for (const char* key : {"sample_rate_hz", "update_rate_hz"}) {
      auto it = u.params.find(key);
      if (it != u.params.end() && it->second > 10.0) {
        it->second = 10.0;
      }
    }
  }
  return sc;
}

struct SuiteVariant {
  const char* name;
  scenario::SpacecraftConfig (*make)();
};

const SuiteVariant kSuiteVariants[] = {
    {"no vehicle", [] { return scenario::SpacecraftConfig{}; }},
    {"full sensor suite", [] { return polaris::test::sitl::faultMatrixSuite(); }},
    {"  - star trackers",
     [] { return without(polaris::test::sitl::faultMatrixSuite(), "star_tracker"); }},
    {"  - IMUs", [] { return without(polaris::test::sitl::faultMatrixSuite(), "imu"); }},
    {"  - sun sensors",
     [] { return without(polaris::test::sitl::faultMatrixSuite(), "sun_sensor"); }},
    {"  - magnetometers",
     [] { return without(polaris::test::sitl::faultMatrixSuite(), "magnetometer"); }},
    {"all rates capped 10 Hz", [] { return ratesAt10Hz(polaris::test::sitl::faultMatrixSuite()); }},
    // The rates the flown config actually compiles to (configc): STIM300 IMUs
    // sample at their datasheet 2000 Hz, which fragments the integration grid
    // to 0.5 ms — the reproduction of the SITL rows' wall-clock cost.
    {"IMUs at flown 2000 Hz",
     [] {
       scenario::SpacecraftConfig sc = polaris::test::sitl::faultMatrixSuite();
       for (scenario::UnitConfig& u : sc.sensors) {
         if (u.kind == "imu") {
           u.params["sample_rate_hz"] = 2000.0;
         }
       }
       return sc;
     }},
};

/// Phase 3 — sensor-model micro-bench: the per-sample cost of each model
/// called directly, isolating "the model is expensive" from "the loop's
/// per-event plumbing is expensive" (skyAt, eclipse, input construction).
void modelMicroBench() {
  scenario::Vehicle vehicle;
  std::string error;
  if (!scenario::buildVehicle(polaris::test::sitl::faultMatrixSuite(), 1, vehicle, &error)) {
    std::fprintf(stderr, "buildVehicle: %s\n", error.c_str());
    return;
  }
  const pt::Tai t0 = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
  constexpr int kReps = 100000;

  std::printf("\nsensor-model micro-bench (%d samples each)\n\n", kReps);
  const auto time_us = [](auto&& fn) {
    const auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
      fn(i);
    }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count() /
           kReps;
  };

  if (!vehicle.imus.empty()) {
    auto& imu = vehicle.imus.front().model;
    const pm::Vec3<pm::frames::Body> w(Eigen::Vector3d(0.05, -0.03, 0.04));
    const pm::Vec3<pm::frames::Body> sf(Eigen::Vector3d(0.0, 0.0, 0.0));
    std::printf("  imu.sample:          %8.2f us\n", time_us([&](int i) {
                  imu.sample(t0 + pt::Duration::fromNanoseconds(500000LL * i), 5.0e-4, w, sf);
                }));
  }
  if (!vehicle.sun_sensors.empty()) {
    auto& ss = vehicle.sun_sensors.front().model;
    polaris::sim::sensors::SunSensorInput in;
    in.sun_dir_body = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.8, 0.1, 0.59).normalized());
    in.nadir_dir_body = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.0, 0.0, -1.0));
    in.shadow_factor = 1.0;
    in.sky.sat = Eigen::Vector3d(6.878e6, 0.0, 0.0);
    in.sky.sun = Eigen::Vector3d(1.4e11, 0.0, 6.0e10);
    std::printf("  sun_sensor.sample:   %8.2f us\n", time_us([&](int i) {
                  ss.sample(t0 + pt::Duration::fromNanoseconds(10000000LL * i), in);
                }));
  }
  if (!vehicle.magnetometers.empty()) {
    auto& mag = vehicle.magnetometers.front().model;
    const pm::Vec3<pm::frames::Body> b(Eigen::Vector3d(2.0e-5, -1.0e-5, 3.0e-5));
    std::printf("  magnetometer.sample: %8.2f us\n", time_us([&](int i) {
                  mag.sample(t0 + pt::Duration::fromNanoseconds(10000000LL * i), b);
                }));
  }
}

}  // namespace

int main(int argc, char** argv) {
  const double sim_s = argc > 1 ? std::atof(argv[1]) : 20.0;
  if (!(sim_s > 0.0)) {
    std::fprintf(stderr, "usage: %s [sim_seconds > 0]\n", argv[0]);
    return 2;
  }
  std::printf("env ablation over %.0f sim-seconds, 10 Hz macro rate, tumbling\n\n", sim_s);
  std::printf("%-24s %10s %14s %12s\n", "variant", "wall [s]", "x real-time", "vs baseline");
  double base = -1.0;
  for (const Variant& v : kVariants) {
    scenario::SimConfig config = baseline(sim_s);
    v.mutate(config);
    const double wall = flyOnce(config);
    if (wall < 0.0) {
      return 1;
    }
    if (base < 0.0) {
      base = wall;
    }
    std::printf("%-24s %10.2f %13.2fx %11.0f%%\n", v.name, wall, sim_s / wall, 100.0 * wall / base);
  }

  std::printf("\nvehicle ablation over %.0f sim-seconds (SITL regime: two-body + IGRF)\n\n", sim_s);
  std::printf("%-24s %10s %14s %12s\n", "variant", "wall [s]", "x real-time", "vs suite");
  double suite_base = -1.0;
  for (const SuiteVariant& v : kSuiteVariants) {
    const scenario::SimConfig config = polaris::test::sitl::faultMatrixOrbit(sim_s, "bench");
    const double wall = flyOnce(config, v.make());
    if (wall < 0.0) {
      return 1;
    }
    if (std::string(v.name) == "full sensor suite") {
      suite_base = wall;
    }
    std::printf("%-24s %10.2f %13.2fx %11.0f%%\n", v.name, wall, sim_s / wall,
                suite_base > 0.0 ? 100.0 * wall / suite_base : 0.0);
  }

  modelMicroBench();
  return 0;
}

/// @file
/// @brief `polaris_sim` — the truth-simulation entry point (design doc §5, §24).
///
/// Reads a compiled `sim_setup.json` (produced by `tools/configc` from the
/// authoritative YAML), builds the environment, propagates the 6DOF truth state,
/// and writes a trajectory CSV.
///
/// Usage:
///
///     polaris_sim --config build/config/sim_setup.json
///                 --data tests/golden
///                 --out build/trajectory.csv
///
/// This is the open-loop truth plant. There is no flight software in the loop
/// yet — the F´ TCP transport and the macro-step handshake (§2.2/§2.4) are
/// `sim/io/`, still empty. What this establishes is that the environment models
/// compose and propagate together, which is the prerequisite for closing that
/// loop.

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "time/leap_seconds.hpp"

namespace {

namespace scenario = polaris::sim::scenario;

struct Options {
  std::string config;
  std::string data = "tests/golden";
  std::string out;
};

void usage() {
  std::cerr
      << "usage: polaris_sim --config <sim_setup.json> [--data <dir>] [--out <trajectory.csv>]\n"
      << "\n"
      << "  --config  compiled scenario from tools/configc (required)\n"
      << "  --data    directory holding the committed reference products\n"
      << "            (EOP, DE440 fit, IGRF coefficients, EGM2008)\n"
      << "  --out     trajectory CSV; omit to print a summary only\n";
}

/// Parse argv. False on an unknown or incomplete flag.
bool parseArgs(int argc, char** argv, Options& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--config" && has_value) {
      out.config = argv[++i];
    } else if (arg == "--data" && has_value) {
      out.data = argv[++i];
    } else if (arg == "--out" && has_value) {
      out.out = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      return false;
    } else {
      std::cerr << "polaris_sim: unrecognized or incomplete argument '" << arg << "'\n";
      return false;
    }
  }
  return !out.config.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parseArgs(argc, argv, options)) {
    usage();
    return EXIT_FAILURE;
  }

  // The historical leap-second table: the scenario epoch is UTC and the master
  // clock is TAI, so this is needed before anything else can be resolved.
  const polaris::time::LeapSecondTable leap = polaris::time::LeapSecondTable::historical();

  std::string error;
  scenario::SimConfig config;
  if (!scenario::loadSimConfig(options.config, leap, config, &error)) {
    std::cerr << "polaris_sim: " << error << "\n";
    return EXIT_FAILURE;
  }

  scenario::SimRunner runner;
  if (!runner.build(config, scenario::DataPaths::under(options.data), &error)) {
    std::cerr << "polaris_sim: " << error << "\n";
    return EXIT_FAILURE;
  }

  std::vector<scenario::TrajectorySample> trajectory;
  if (!runner.run(trajectory, &error)) {
    std::cerr << "polaris_sim: " << error << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "scenario:   " << config.scenario_name << "\n"
            << "spacecraft: " << config.spacecraft.name << "\n"
            << "config:     " << config.config_hash << "\n"
            << "models:     " << runner.modelCount() << "\n"
            << "duration:   " << config.propagation.duration_s << " s\n"
            << "samples:    " << trajectory.size() << "\n";

  if (!trajectory.empty()) {
    const auto& last = trajectory.back().state;
    std::cout << "final |r|:  " << last.position.eigen().norm() << " m\n"
              << "final |v|:  " << last.velocity.eigen().norm() << " m/s\n";
  }

  if (!options.out.empty()) {
    if (!scenario::writeTrajectoryCsv(options.out, config, trajectory, &error)) {
      std::cerr << "polaris_sim: " << error << "\n";
      return EXIT_FAILURE;
    }
    std::cout << "wrote:      " << options.out << "\n";
  }
  return EXIT_SUCCESS;
}

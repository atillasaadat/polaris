#ifndef POLARIS_SIM_SCENARIO_SIM_RUNNER_HPP
#define POLARIS_SIM_SCENARIO_SIM_RUNNER_HPP

/// @file
/// @brief Assembles a `SimConfig` into a running 6DOF truth propagation
/// (design doc §5, §24 Phase 1).
///
/// This is the piece that had been missing: five pushes produced environment
/// models — EGM2008 gravity, third-body, SRP + eclipse, drag, IGRF + residual
/// dipole — and a 6DOF RK89 plant, with nothing that put them in the same
/// process. `SimRunner` owns the data layers (EOP, DE440 ephemeris, IGRF
/// coefficients, gravity coefficients), wires every resolver, composes the
/// enabled models into one `CompositeForceModel`, and steps the plant.
///
/// Ownership matters here and is the reason this is a class rather than a
/// function. `CompositeForceModel` holds **non-owning** pointers, the resolvers
/// are `std::function`s capturing `this`-adjacent state, and the ephemeris and
/// EOP tables are referenced by those resolvers. Everything therefore has to
/// outlive the propagation and must not move once wired — hence the members are
/// held in `unique_ptr` and the class is non-copyable, so a resolver can never
/// end up pointing at a moved-from table.
///
/// Sim-side: heap, exceptions-free error returns, `std::function` all fine.

#include <memory>
#include <string>
#include <vector>

#include "dynamics/force_torque.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/rigid_body.hpp"
#include "scenario/sim_config.hpp"
#include "state/truth_state.hpp"

namespace polaris::sim::scenario {

/// Where the committed reference-data products live. Defaults match the repo
/// layout; a deployment can point them elsewhere.
struct DataPaths {
  std::string eop;        ///< IERS finals.all.iau2000
  std::string ephemeris;  ///< DE440 Chebyshev fit (.cheb)
  std::string igrf;       ///< IAGA IGRF-14 coefficients
  std::string gravity;    ///< ICGEM EGM2008 .gfc

  /// Paths relative to a repo/data root.
  static DataPaths under(const std::string& root);
};

/// One trajectory sample.
struct TrajectorySample {
  double t_s{0.0};  ///< seconds since the scenario epoch
  state::TruthState state;
};

/// Builds and runs one scenario.
class SimRunner {
 public:
  SimRunner();
  /// Defined out of line: `Impl` is incomplete here, so the implicit destructor
  /// could not instantiate `unique_ptr`'s deleter.
  ~SimRunner();
  SimRunner(const SimRunner&) = delete;
  SimRunner& operator=(const SimRunner&) = delete;

  /// Load data products and wire every enabled model.
  ///
  /// Only the data a scenario actually asks for is loaded: a run with
  /// `gravity_degree: 0` and no third bodies touches no files at all. That keeps
  /// the unit tests honest — they can exercise the runner without the multi-
  /// megabyte fixtures — and it means a missing file is only an error when the
  /// scenario genuinely needed it.
  ///
  /// @return false if a required data product is missing or malformed, or the
  ///         configuration asks for a model that is not compiled in (NRLMSIS
  ///         without `POLARIS_HAS_NRLMSIS`). @p error receives the reason.
  bool build(const SimConfig& config, const DataPaths& paths, std::string* error = nullptr);

  /// True once @ref build has succeeded.
  bool ready() const { return body_ != nullptr; }

  /// Propagate for the configured duration, sampling at the configured cadence.
  ///
  /// Stepping happens output-step by output-step rather than in one call, so the
  /// samples are the integrator's own accepted states at those epochs, not
  /// interpolations, and so a long run reports progress. Each step re-enters the
  /// adaptive controller, which is free to take many internal steps.
  ///
  /// @return false if @ref ready() is false. @p out is cleared first.
  bool run(std::vector<TrajectorySample>& out, std::string* error = nullptr) const;

  /// The composed model, for tests that want to interrogate the force budget.
  const dynamics::ForceTorqueModel* forceModel() const { return composite_.get(); }

  /// Number of component models composed in.
  std::size_t modelCount() const;

 private:
  struct Impl;
  /// Everything the resolvers capture. Held behind a pointer so wiring survives
  /// the runner being stored by value elsewhere.
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<dynamics::CompositeForceModel> composite_;
  std::unique_ptr<dynamics::RigidBody6Dof> body_;
  SimConfig config_;
};

/// Write @p samples as CSV to @p path, with a provenance header naming the
/// scenario and its config hash so an output file traces back to its input.
///
/// @return false if the file cannot be written.
bool writeTrajectoryCsv(const std::string& path, const SimConfig& config,
                        const std::vector<TrajectorySample>& samples, std::string* error = nullptr);

}  // namespace polaris::sim::scenario

#endif  // POLARIS_SIM_SCENARIO_SIM_RUNNER_HPP

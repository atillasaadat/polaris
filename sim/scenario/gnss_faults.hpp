#ifndef POLARIS_SIM_SCENARIO_GNSS_FAULTS_HPP
#define POLARIS_SIM_SCENARIO_GNSS_FAULTS_HPP

/// @file
/// @brief Apply a scheduled GNSS fault set to a receiver (design doc §9.2/§23.1.1).
///
/// Bridges config (`EnvironmentConfig::gnss_fault_events`) to the runtime fault
/// hooks on `sensors::Gnss`. The schedule is declarative — a list of time windows
/// per unit — and `applyGnssFaults` reconciles the receiver's fault state to the
/// window active at time t on every call, so entering and leaving a window are
/// both handled (a window that has ended clears its fault). This is the piece the
/// §2.4 macro-step loop will call each step; it is standalone and unit-tested so
/// the fault semantics are pinned before that loop exists.

#include <string>
#include <vector>

#include <Eigen/Core>

#include "scenario/sim_config.hpp"
#include "sensors/gnss.hpp"

namespace polaris::sim::scenario {

/// The fault state active for one receiver at an instant — the union of every
/// scheduled event covering time t for that unit.
struct GnssFaultState {
  bool outage = false;
  bool spoof = false;
  Eigen::Vector3d spoof_offset_ecef_m{Eigen::Vector3d::Zero()};
  double clock_jump_s = 0.0;
};

/// Resolve the active fault state for @p unit at @p t_s (seconds since epoch) from
/// @p events. Overlapping spoofs sum their offsets and clock jumps sum; outage is
/// a logical OR.
GnssFaultState gnssFaultsAt(const std::vector<GnssFaultEvent>& events, const std::string& unit,
                            double t_s);

/// Drive @p receiver's fault hooks to match @p state. Idempotent: call it every
/// step with the resolved state and the receiver tracks the schedule, clearing a
/// fault when its window ends.
void applyGnssFaults(sensors::Gnss& receiver, const GnssFaultState& state);

}  // namespace polaris::sim::scenario

#endif  // POLARIS_SIM_SCENARIO_GNSS_FAULTS_HPP

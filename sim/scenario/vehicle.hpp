#ifndef POLARIS_SIM_SCENARIO_VEHICLE_HPP
#define POLARIS_SIM_SCENARIO_VEHICLE_HPP

/// @file
/// @brief Builds the installed sensor/actuator suite from the compiled config
/// (design doc §19.3/§19.4, REQ-CFG-001/002).
///
/// This is the piece that makes the hardware library real. `tools/configc`
/// resolves each `model_id` against `config/hardware/**.yaml` and inlines that
/// unit's parameters into `sim_setup.json`; `SimConfig` carries them through
/// untouched; here they are handed to the model's own `fromParams`, which is the
/// **only** place a catalog number becomes a `Spec`. There is deliberately no
/// in-code hardware catalog to fall back on (§19.4) — swapping a `model_id`
/// string in the YAML is the entire mechanism for re-flying a vehicle with a
/// different unit, and it cannot be silently overridden from C++.
///
/// **Seeding.** Each unit's noise stream id is derived from a hash of its
/// instance name rather than its position in the list, so adding, removing, or
/// reordering hardware leaves every other unit's random stream bit-identical
/// (§3.6/§182). Two units may not share a name — that would make them the same
/// stream, and correlated "independent" sensors is the kind of error that only
/// shows up as an implausibly good estimator.
///
/// Sim-side: heap, exceptions-free error returns.

#include <cstdint>
#include <Eigen/Core>
#include <string>
#include <vector>

#include "actuators/magnetorquer.hpp"
#include "actuators/reaction_wheel.hpp"
#include "actuators/rw_assembly.hpp"
#include "scenario/sim_config.hpp"
#include "sensors/gnss.hpp"
#include "sensors/imu.hpp"
#include "sensors/magnetometer.hpp"
#include "sensors/star_tracker.hpp"
#include "sensors/sun_sensor.hpp"

namespace polaris::sim::scenario {

/// A built unit: the model plus the placement and identity it was built from.
template <typename Model>
struct MountedModel {
  std::string name;
  std::string model_id;
  /// Unit→body rotation; the caller rotates commands in and outputs out.
  Eigen::Matrix3d mounting_dcm{Eigen::Matrix3d::Identity()};
  Model model;
};

/// The vehicle's installed hardware, one vector per modelled device class.
struct Vehicle {
  std::vector<MountedModel<sensors::Imu>> imus;
  std::vector<MountedModel<sensors::StarTracker>> star_trackers;
  std::vector<MountedModel<sensors::SunSensor>> sun_sensors;
  std::vector<MountedModel<sensors::Magnetometer>> magnetometers;
  std::vector<MountedModel<sensors::Gnss>> gnss_receivers;
  std::vector<MountedModel<actuators::ReactionWheel>> wheels;
  std::vector<MountedModel<actuators::Magnetorquer>> magnetorquers;

  /// The wheel array's distribution matrix W (§7), columns = each wheel's spin
  /// axis in body frame, in `wheels` order. Empty when there are no wheels. Use
  /// it to turn per-wheel torques/momenta into their body-frame totals.
  actuators::RwAssembly rw_assembly;

  /// Units the config asked for that have no truth model yet (e.g. a thruster),
  /// as "name:kind". Reported rather than dropped: a scenario
  /// quietly flying without a sensor it configured would produce a clean-looking
  /// run that answers a different question.
  std::vector<std::string> unmodelled;

  std::size_t modelledCount() const {
    return imus.size() + star_trackers.size() + sun_sensors.size() + magnetometers.size() +
           gnss_receivers.size() + wheels.size() + magnetorquers.size();
  }
};

/// Build the suite described by @p spacecraft, seeding every stochastic unit
/// from @p seed.
///
/// @return false on a duplicate unit name (streams would collide) or a unit whose
///         `kind` has a model but whose parameters are unusable. @p error
///         receives the reason. An unmodelled `kind` is not an error — it lands
///         in @ref Vehicle::unmodelled.
bool buildVehicle(const SpacecraftConfig& spacecraft, std::uint64_t seed, Vehicle& out,
                  std::string* error = nullptr);

}  // namespace polaris::sim::scenario

#endif  // POLARIS_SIM_SCENARIO_VEHICLE_HPP

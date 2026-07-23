#include "scenario/vehicle.hpp"

#include <set>
#include <utility>

namespace polaris::sim::scenario {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// FNV-1a over the instance name — the unit's noise-stream id.
///
/// Name-derived rather than index-derived on purpose: the §182 determinism rule
/// is that adding a noise source must not perturb the ones already there, and an
/// index shifts the moment a unit is inserted ahead of another. FNV-1a is used
/// only to spread names across the id space; `random::streamRng` does the actual
/// mixing, so the hash needs no statistical strength of its own.
std::uint64_t streamIdFor(const std::string& name) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char c : name) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

bool buildVehicle(const SpacecraftConfig& spacecraft, std::uint64_t seed, Vehicle& out,
                  std::string* error, const NoiseSettings& noise) {
  out = Vehicle{};

  // Spin axes in `wheels` order, for the assembly's W. A wheel takes its axis from
  // spin_axis when set (the clean form), else the third column of its mounting DCM.
  std::vector<Eigen::Vector3d> wheel_axes;
  std::set<std::string> names;
  for (const std::vector<UnitConfig>* suite : {&spacecraft.sensors, &spacecraft.actuators}) {
    for (const UnitConfig& unit : *suite) {
      if (!names.insert(unit.name).second) {
        return fail(error, "duplicate hardware unit name '" + unit.name +
                               "' — unit names key the per-unit random streams and "
                               "must be unique across the vehicle");
      }
      // Every modelled device takes all of its physics from these parameters, so
      // an empty map is not a degenerate unit, it is a resolution that went
      // wrong: the model would build as an ideal, noiseless, unlimited device.
      if (unit.params.empty()) {
        return fail(error, "hardware unit '" + unit.name + "' (model_id '" + unit.model_id +
                               "') resolved to no parameters");
      }

      const std::uint64_t stream = streamIdFor(unit.name);
      if (unit.kind == "imu") {
        out.imus.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm,
             sensors::Imu(sensors::ImuSpec::fromParams(unit.params), seed, stream, noise.sensors)});
      } else if (unit.kind == "star_tracker") {
        const auto spec = sensors::StarTrackerSpec::fromParams(unit.params);
        // With neither a field of view nor a quoted Earth exclusion angle the
        // Earth keep-out collapses to zero and the tracker solves happily while
        // staring at the ground. A silently over-optimistic sensor is worse than
        // a configuration error. Either key satisfies this — vendors quote an
        // exclusion angle, and it is what dominates when both are present.
        if (!(spec.keep_out.earth_rad > 0.0)) {
          return fail(error, "star tracker '" + unit.name +
                                 "' has neither fov_deg nor earth_exclusion_deg — it would "
                                 "report valid solutions while pointed at the Earth");
        }
        out.star_trackers.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm,
             sensors::StarTracker(spec, unit.mounting_dcm, seed, stream, noise.sensors)});
      } else if (unit.kind == "sun_sensor") {
        const auto spec = sensors::SunSensorSpec::fromParams(unit.params);
        // Without an acceptance cone the cosine cut-off never fires and a cell
        // reports sunlight while facing away from the Sun — required whichever
        // output the part has.
        if (!(spec.half_fov_rad > 0.0)) {
          return fail(error, "sun sensor '" + unit.name + "' needs half_fov_deg");
        }
        // Beyond that the two output contracts need different things, and
        // demanding both would reject every real part: a digital unit reports a
        // vector and has no full scale, an analogue one has no quoted accuracy
        // because the FSW is what turns its counts into an angle.
        if (spec.output == sensors::SunSensorOutput::kDiodeCounts) {
          if (!(spec.full_scale_counts > 0.0)) {
            return fail(error, "sun sensor '" + unit.name +
                                   "' reports diode counts but has no full_scale_counts");
          }
        } else if (!(spec.accuracy_outer_sigma > 0.0)) {
          return fail(error, "sun sensor '" + unit.name +
                                 "' reports a vector but has no accuracy_*_deg_3sigma");
        }
        out.sun_sensors.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm,
             sensors::SunSensor(spec, unit.mounting_dcm, seed, stream, noise.sensors)});
      } else if (unit.kind == "magnetometer") {
        const auto model = sensors::magnetometerErrorFromParams(unit.params, seed, stream);
        // A magnetometer with no range still measures, so there is nothing to
        // reject here: every term of the error stack is a no-op at zero, and a
        // range of zero simply means "no saturation modelled".
        out.magnetometers.push_back({unit.name, unit.model_id, unit.mounting_dcm,
                                     sensors::Magnetometer(model, seed, stream, noise.sensors)});
      } else if (unit.kind == "gnss") {
        auto spec = sensors::GnssSpec::fromParams(unit.params);
        // GNSS honours the global sensor switch and its own override.
        spec.noise_enabled = noise.sensors && noise.gnss;
        // A receiver with no quoted position accuracy would report truth-perfect
        // fixes — worse than a config error, because it silently hands the OD
        // filter the answer. Every real datasheet quotes a horizontal RMS.
        if (!(spec.position_sigma_h_m > 0.0)) {
          return fail(error, "gnss receiver '" + unit.name +
                                 "' has no horizontal_position_rms_m — it would report "
                                 "truth-perfect position fixes");
        }
        out.gnss_receivers.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm, sensors::Gnss(spec, seed, stream)});
      } else if (unit.kind == "reaction_wheel") {
        const auto spec = actuators::ReactionWheelSpec::fromParams(unit.params);
        if (!(spec.inertia() > 0.0)) {
          return fail(error, "reaction wheel '" + unit.name +
                                 "' has no rotor inertia — set rotor_inertia_kg_m2, or "
                                 "max_momentum_nms with max_speed_rpm");
        }
        out.wheels.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm, actuators::ReactionWheel(spec)});
        wheel_axes.push_back(unit.spin_axis.norm() > 0.0 ? unit.spin_axis
                                                         : unit.mounting_dcm.col(2));
      } else if (unit.kind == "magnetorquer") {
        const auto spec = actuators::MagnetorquerSpec::fromParams(unit.params);
        if (!(spec.max_dipole_am2 > 0.0)) {
          return fail(error, "magnetorquer '" + unit.name + "' has no max_dipole_am2");
        }
        out.magnetorquers.push_back(
            {unit.name, unit.model_id, unit.mounting_dcm, actuators::Magnetorquer(spec)});
      } else {
        out.unmodelled.push_back(unit.name + ":" + unit.kind);
      }
    }
  }

  // Consolidate the wheel geometry into the assembly's W (empty if no wheels).
  out.rw_assembly = actuators::RwAssembly::fromAxes(wheel_axes);
  return true;
}

}  // namespace polaris::sim::scenario

#include "io/closed_loop.hpp"

#include <algorithm>
#include <cmath>

#include "actuators/magnetorquer.hpp"
#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "scenario/gnss_faults.hpp"
#include "sensors/occlusion.hpp"
#include "time/leap_seconds.hpp"
#include "time/tdb.hpp"
#include "world/eclipse.hpp"
#include "world/eop_file.hpp"
#include "world/ephemeris_file.hpp"

namespace polaris::sim::io {
namespace {

constexpr std::int64_t kNsPerSecond = 1'000'000'000;

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// A sensor's native sample period in exact nanoseconds. Zero/absent rates fall
/// back to the macro period: a sensor with no quoted rate is read when the FSW
/// reads it, which is what a polled device does.
std::int64_t periodNs(double rate_hz, std::int64_t macro_ns) {
  if (!(rate_hz > 0.0)) {
    return macro_ns;
  }
  return static_cast<std::int64_t>(std::llround(1.0e9 / rate_hz));
}

/// EOP capacity for the loop's own table (GNSS ECEF output); same sizing
/// rationale as the runner's.
constexpr std::size_t kEopCapacity = 512;

}  // namespace

ClosedLoop::ClosedLoop(scenario::SimRunner& runner, scenario::Vehicle& vehicle)
    : runner_(runner), vehicle_(vehicle), paths_(scenario::DataPaths::under("tests/golden")) {}

bool ClosedLoop::run(const FswCallback& fsw, std::vector<MacroSample>* trace, std::string* error) {
  if (trace != nullptr) {
    trace->clear();
  }
  if (!runner_.ready()) {
    return fail(error, "ClosedLoop::run: the runner has not been built");
  }
  const scenario::SimConfig& config = runner_.config();
  const scenario::PropagationConfig& prop = config.propagation;
  const time::Tai epoch = config.initial_state.epoch;
  const time::LeapSecondTable leap = time::LeapSecondTable::historical();

  // --- Data products the sensors (not the forces) need -----------------------
  //
  // The runner loads only what the *dynamics* asked for; the sensing side can
  // need more. GNSS outputs ECEF, so it needs EOP even in a two-body run;
  // optical sensors need Sun/Moon geometry even with third-body gravity off.
  frames::EopTable<kEopCapacity> eop;
  const bool needs_eop = !vehicle_.gnss_receivers.empty();
  if (needs_eop) {
    const time::Tai end = epoch + time::Duration::fromSecondsF(prop.duration_s);
    if (!world::loadEopFile(paths_.eop, leap, epoch, end, eop, error)) {
      return false;
    }
  }

  world::BodyPositionFn sun_fn = runner_.sunPositionFn();
  world::BodyPositionFn moon_fn = runner_.moonPositionFn();
  world::EphemerisSet own_ephemeris;
  const bool needs_sky = !vehicle_.star_trackers.empty() || !vehicle_.sun_sensors.empty() ||
                         !vehicle_.payload_sensors.empty();
  if (needs_sky && !sun_fn) {
    if (!world::loadEphemerisFile(paths_.ephemeris, own_ephemeris, error)) {
      return false;
    }
    sun_fn = world::bodyPositionFn(own_ephemeris.sun);
    moon_fn = world::bodyPositionFn(own_ephemeris.moon);
  }
  const world::MagneticFieldFn field_fn = runner_.magneticFieldFn();

  // Geographic GNSS jamming (§9.2): one map, bound to every receiver.
  sensors::JammingRegions jamming;
  if (config.environment.gnss_jamming_enabled && !config.environment.gnss_jamming_kml.empty() &&
      !vehicle_.gnss_receivers.empty()) {
    if (!sensors::JammingRegions::loadKmlFile(config.environment.gnss_jamming_kml, jamming,
                                              error)) {
      return false;
    }
  }
  for (auto& receiver : vehicle_.gnss_receivers) {
    receiver.model.setJammingRegions(jamming.empty() ? nullptr : &jamming);
  }

  // --- The event grid, in exact integer nanoseconds --------------------------
  const std::int64_t macro_ns = static_cast<std::int64_t>(std::llround(1.0e9 / prop.fsw_rate_hz));
  auto sensorPeriod = [macro_ns](double rate_hz) { return periodNs(rate_hz, macro_ns); };

  struct Track {
    std::int64_t period_ns;
    std::int64_t next_ns;
  };

  std::vector<Track> imu_track;
  std::vector<Track> st_track;
  std::vector<Track> ss_track;
  std::vector<Track> mag_track;
  std::vector<Track> gnss_track;
  std::vector<Track> payload_track;
  for (const auto& u : vehicle_.imus) {
    imu_track.push_back({sensorPeriod(u.model.spec().sample_rate_hz), 0});
  }
  for (const auto& u : vehicle_.star_trackers) {
    st_track.push_back({sensorPeriod(u.model.spec().update_rate_hz), 0});
  }
  for (const auto& u : vehicle_.sun_sensors) {
    ss_track.push_back({sensorPeriod(u.model.spec().update_rate_hz), 0});
  }
  for (const auto& u : vehicle_.magnetometers) {
    (void)u;  // no native rate modelled: polled at the FSW boundary
    mag_track.push_back({macro_ns, 0});
  }
  for (const auto& u : vehicle_.gnss_receivers) {
    gnss_track.push_back({sensorPeriod(u.model.spec().max_rate_hz), 0});
  }
  for (const auto& u : vehicle_.payload_sensors) {
    payload_track.push_back({sensorPeriod(u.model.spec().update_rate_hz), 0});
  }
  // First samples come due one period in (nothing has been measured at t=0);
  // discrete tracks stay aligned to their own grid for the whole run.
  for (auto* tracks : {&imu_track, &st_track, &ss_track, &mag_track, &gnss_track, &payload_track}) {
    for (Track& track : *tracks) {
      track.next_ns = track.period_ns;
    }
  }

  // --- Per-run working state --------------------------------------------------
  dynamics::StepControl control;
  control.abs_tol = prop.abs_tol;
  control.rel_tol = prop.rel_tol;
  control.max_step = prop.max_step_s;

  state::TruthState s = config.initial_state;

  FswInputs inputs;
  inputs.imus.resize(vehicle_.imus.size());
  inputs.star_trackers.resize(vehicle_.star_trackers.size());
  inputs.sun_sensors.resize(vehicle_.sun_sensors.size());
  inputs.magnetometers.resize(vehicle_.magnetometers.size());
  inputs.gnss.resize(vehicle_.gnss_receivers.size());
  inputs.wheels.resize(vehicle_.wheels.size());
  for (std::size_t i = 0; i < vehicle_.imus.size(); ++i) {
    inputs.imus[i].name = vehicle_.imus[i].name;
  }
  for (std::size_t i = 0; i < vehicle_.star_trackers.size(); ++i) {
    inputs.star_trackers[i].name = vehicle_.star_trackers[i].name;
  }
  for (std::size_t i = 0; i < vehicle_.sun_sensors.size(); ++i) {
    inputs.sun_sensors[i].name = vehicle_.sun_sensors[i].name;
  }
  for (std::size_t i = 0; i < vehicle_.magnetometers.size(); ++i) {
    inputs.magnetometers[i].name = vehicle_.magnetometers[i].name;
  }
  for (std::size_t i = 0; i < vehicle_.gnss_receivers.size(); ++i) {
    inputs.gnss[i].name = vehicle_.gnss_receivers[i].name;
  }
  for (std::size_t i = 0; i < vehicle_.wheels.size(); ++i) {
    inputs.wheels[i].name = vehicle_.wheels[i].name;
  }

  // Payload geometry is a sim-side product, so it lives on the loop rather than
  // in `inputs` — nothing here crosses to the FSW (§2.3).
  payload_geometry_.assign(vehicle_.payload_sensors.size(), PayloadGeometry{});
  for (std::size_t i = 0; i < vehicle_.payload_sensors.size(); ++i) {
    payload_geometry_[i].name = vehicle_.payload_sensors[i].name;
  }

  FswOutputs commands;  // zero until the first boundary: nothing commanded yet
  std::vector<Eigen::Vector3d> mtq_actual(vehicle_.magnetorquers.size(), Eigen::Vector3d::Zero());
  // §7 MTQ/MAG duty-cycle state. The rods carry the commanded dipole over
  // [macro boundary, duty_off_ns) and are then de-energised, their field decaying
  // over the catalog settle time — which is what leaves a *quiet window* at the
  // end of each period for the magnetometer to be read in. `rods_off` tracks
  // which side of that boundary the march is on; `duty_off_ns` is when it
  // happened (or will).
  std::int64_t duty_off_ns = 0;
  bool rods_off = true;
  std::vector<Eigen::Vector3d> st_prev_rate(vehicle_.star_trackers.size(), Eigen::Vector3d::Zero());
  std::vector<bool> st_has_prev(vehicle_.star_trackers.size(), false);

  auto taiAt = [&](std::int64_t t_ns) {
    return time::Tai::fromNanosecondsSinceEpoch(epoch.nanosecondsSinceEpoch() + t_ns);
  };

  /// Sky geometry at truth time — shared by every optical sensor so they cannot
  /// disagree about where the Sun and Earth are (§6.1).
  auto skyAt = [&](const time::Tai& t) {
    sensors::SkyGeometry sky;
    sky.sat = s.position.eigen();
    sky.atmosphere_height_m = config.environment.occultation_atmosphere_m;
    const time::Tdb tdb = time::toTdb(time::toTt(t));
    math::Vec3<math::frames::ECI> p;
    if (sun_fn && sun_fn(tdb, p)) {
      sky.sun = p.eigen();
    }
    if (moon_fn && moon_fn(tdb, p)) {
      sky.moon = p.eigen();
    }
    return sky;
  };

  /// Advance actuators over [from, to) under zero-order-held commands and write
  /// the net wrench into the plant for that interval.
  /// This unit's produced dipole at truth time @p at_ns: the driven moment inside
  /// the on-window, the decaying transient after it. Zero-order held across each
  /// micro-interval, which is exact inside the on-window and a small
  /// approximation of the decay tail — whose torque contribution is a fraction of
  /// a per-cent of the driven one by construction.
  auto mtqMomentAt = [&](std::size_t i, std::int64_t at_ns) {
    if (!rods_off) {
      return mtq_actual[i];
    }
    const double since_off_s = static_cast<double>(at_ns - duty_off_ns) / 1.0e9;
    return vehicle_.magnetorquers[i].model.settlingDipole(since_off_s).eigen();
  };

  auto applyActuators = [&](std::int64_t at_ns, double dt_s) {
    Eigen::Vector3d torque = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < vehicle_.wheels.size(); ++i) {
      const auto out = vehicle_.wheels[i].model.step(dt_s);
      // Reaction torque acts about the wheel's spin axis — the assembly's W
      // column for this wheel (§7).
      torque +=
          vehicle_.rw_assembly.matrix().col(static_cast<Eigen::Index>(i)) * out.reaction_torque_nm;
    }
    if (field_fn) {
      math::Vec3<math::frames::ECI> b_eci;
      if (field_fn(s.epoch, s.position, b_eci)) {
        const Eigen::Vector3d b_body = s.attitude.rotate(b_eci).eigen();
        for (std::size_t i = 0; i < vehicle_.magnetorquers.size(); ++i) {
          torque += mtqMomentAt(i, at_ns).cross(b_body);
        }
      }
    }
    wrench_.set(math::Vec3<math::frames::ECI>::Zero(), math::Vec3<math::frames::Body>(torque));
  };

  /// Sample one sensor class if due at t_ns; fixed order keeps the run
  /// deterministic. dt is the sensor's own period (its integration interval).
  auto sampleDue = [&](std::int64_t t_ns) {
    const time::Tai t = taiAt(t_ns);
    for (std::size_t i = 0; i < imu_track.size(); ++i) {
      if (imu_track[i].next_ns != t_ns) {
        continue;
      }
      imu_track[i].next_ns += imu_track[i].period_ns;
      const double dt = static_cast<double>(imu_track[i].period_ns) / 1.0e9;
      const auto sf_body = s.attitude.rotate(runner_.nonGravAcceleration(s));
      const auto sample = vehicle_.imus[i].model.sample(t, dt, s.body_rate, sf_body);
      ImuAccumulation& acc = inputs.imus[i];
      acc.delta_angle_rad = math::Vec3<math::frames::Body>(acc.delta_angle_rad.eigen() +
                                                           sample.delta_angle_rad.eigen());
      acc.delta_velocity_mps = math::Vec3<math::frames::Body>(acc.delta_velocity_mps.eigen() +
                                                              sample.delta_velocity_mps.eigen());
      acc.samples += 1;
      acc.valid = acc.valid && sample.valid;
      acc.time_tag = sample.time_tag;
    }
    for (std::size_t i = 0; i < st_track.size(); ++i) {
      if (st_track[i].next_ns != t_ns) {
        continue;
      }
      st_track[i].next_ns += st_track[i].period_ns;
      const double dt = static_cast<double>(st_track[i].period_ns) / 1.0e9;
      sensors::StarTrackerInput in;
      in.attitude = s.attitude;
      in.body_rate = s.body_rate;
      // Angular acceleration from the tracker's own sample-to-sample rate
      // difference — what the unit itself could observe.
      const Eigen::Vector3d rate = s.body_rate.eigen();
      in.angular_accel = math::Vec3<math::frames::Body>(
          st_has_prev[i] ? Eigen::Vector3d(((rate - st_prev_rate[i]) / dt).eval())
                         : Eigen::Vector3d::Zero());
      st_prev_rate[i] = rate;
      st_has_prev[i] = true;
      in.sky = skyAt(t);
      inputs.star_trackers[i].measurement = vehicle_.star_trackers[i].model.sample(t, dt, in);
      inputs.star_trackers[i].ever_sampled = true;
    }
    for (std::size_t i = 0; i < ss_track.size(); ++i) {
      if (ss_track[i].next_ns != t_ns) {
        continue;
      }
      ss_track[i].next_ns += ss_track[i].period_ns;
      sensors::SunSensorInput in;
      in.sky = skyAt(t);
      const Eigen::Vector3d to_sun = in.sky.sun - in.sky.sat;
      if (to_sun.norm() > 0.0) {
        in.sun_dir_body = s.attitude.rotate(math::Vec3<math::frames::ECI>(to_sun.normalized()));
      }
      in.nadir_dir_body =
          s.attitude.rotate(math::Vec3<math::frames::ECI>(-s.position.eigen().normalized()));
      in.shadow_factor = world::shadowFactor(in.sky.sat, in.sky.sun);
      inputs.sun_sensors[i].measurement = vehicle_.sun_sensors[i].model.sample(t, in);
      inputs.sun_sensors[i].ever_sampled = true;
    }
    for (std::size_t i = 0; i < mag_track.size(); ++i) {
      if (mag_track[i].next_ns != t_ns) {
        continue;
      }
      mag_track[i].next_ns += mag_track[i].period_ns;
      math::Vec3<math::frames::Body> b_body{};
      if (field_fn) {
        math::Vec3<math::frames::ECI> b_eci;
        if (field_fn(t, s.position, b_eci)) {
          b_body = s.attitude.rotate(b_eci);
        }
      }
      // §7 interlock fidelity: every energised (or still-settling) rod adds its
      // near field at *this* magnetometer's mounted location. The FSW knows
      // nothing about this geometry by design — it protects itself with the
      // duty-cycle schedule, and a schedule that slips shows up here as a sample
      // hundreds of microtesla wrong rather than as an assumption.
      Eigen::Vector3d near_field = Eigen::Vector3d::Zero();
      for (std::size_t k = 0; k < vehicle_.magnetorquers.size(); ++k) {
        near_field +=
            actuators::dipoleNearField(math::Vec3<math::frames::Body>(mtqMomentAt(k, t_ns)),
                                       vehicle_.magnetorquers[k].position_body_m,
                                       vehicle_.magnetometers[i].position_body_m)
                .eigen();
      }
      b_body = math::Vec3<math::frames::Body>(Eigen::Vector3d(b_body.eigen() + near_field));
      inputs.magnetometers[i].measurement = vehicle_.magnetometers[i].model.sample(t, b_body);
      inputs.magnetometers[i].ever_sampled = true;
    }
    for (std::size_t i = 0; i < gnss_track.size(); ++i) {
      if (gnss_track[i].next_ns != t_ns) {
        continue;
      }
      gnss_track[i].next_ns += gnss_track[i].period_ns;
      auto& unit = vehicle_.gnss_receivers[i];
      // Scheduled faults (§9.2): reconcile this receiver to the window active now.
      scenario::applyGnssFaults(
          unit.model, scenario::gnssFaultsAt(config.environment.gnss_fault_events, unit.name,
                                             static_cast<double>(t_ns) / 1.0e9));
      sensors::GnssInput in;
      if (!frames::ecefStateFromEci(t, eop, leap, s.position, s.velocity, in.position_m,
                                    in.velocity_m_s)) {
        // Outside the EOP window: the receiver has no frame to report in. Refuse
        // loudly — this is a configuration error, not an outage to simulate.
        fail(error, "GNSS sampling needs EOP coverage at t=" +
                        std::to_string(static_cast<double>(t_ns) / 1.0e9) + " s");
        return false;
      }
      inputs.gnss[i].measurement = unit.model.sample(t, in);
      inputs.gnss[i].ever_sampled = true;
    }
    for (std::size_t i = 0; i < payload_track.size(); ++i) {
      if (payload_track[i].next_ns != t_ns) {
        continue;
      }
      payload_track[i].next_ns += payload_track[i].period_ns;
      sensors::PayloadSensorInput in;
      in.attitude = s.attitude;
      in.sky = skyAt(t);
      payload_geometry_[i].measurement = vehicle_.payload_sensors[i].model.sample(t, in);
      payload_geometry_[i].ever_sampled = true;
    }
    return true;
  };

  // --- The march ---------------------------------------------------------------
  const auto macro_count =
      static_cast<std::uint64_t>(std::floor(prop.duration_s * prop.fsw_rate_hz + 1.0e-9));
  if (trace != nullptr) {
    trace->push_back({0.0, s});
  }

  std::int64_t t_ns = 0;
  for (std::uint64_t macro = 0; macro < macro_count; ++macro) {
    const std::int64_t boundary_ns = static_cast<std::int64_t>(macro + 1) * macro_ns;
    while (t_ns < boundary_ns) {
      // Next event: the earliest due sensor, capped at the boundary.
      std::int64_t next_ns = boundary_ns;
      for (const auto* tracks :
           {&imu_track, &st_track, &ss_track, &mag_track, &gnss_track, &payload_track}) {
        for (const Track& track : *tracks) {
          next_ns = std::min(next_ns, track.next_ns);
        }
      }
      // The end of the MTQ-on window is an event of the grid, not something a
      // micro-interval is allowed to straddle: a step spanning it would apply the
      // driven moment across the quiet window too, which is precisely the
      // violation this model exists to make visible.
      if (!rods_off && duty_off_ns > t_ns) {
        next_ns = std::min(next_ns, duty_off_ns);
      }
      const double dt_s = static_cast<double>(next_ns - t_ns) / 1.0e9;
      if (dt_s > 0.0) {
        applyActuators(t_ns, dt_s);
        s = runner_.body()->propagate(s, dt_s, control);
        s.epoch = taiAt(next_ns);
      }
      t_ns = next_ns;
      if (!rods_off && t_ns >= duty_off_ns) {
        for (auto& rod : vehicle_.magnetorquers) {
          rod.model.deenergize();
        }
        rods_off = true;
      }
      if (!sampleDue(t_ns)) {
        return false;
      }
    }

    // Macro boundary: publish, fire the FSW, hold its commands for the next
    // interval (§2.4 — commands apply on the *next* step).
    inputs.epoch = taiAt(t_ns);
    inputs.macro_step = macro;
    // Wheel tachometers, read at the boundary the FSW is about to act on. The
    // wheels have been stepped to here by the micro-step loop, so this is the
    // rotor speed at this epoch and not the one at the last command.
    for (std::size_t i = 0; i < vehicle_.wheels.size(); ++i) {
      inputs.wheels[i].speed_rad_s = vehicle_.wheels[i].model.speed();
      inputs.wheels[i].valid = std::isfinite(inputs.wheels[i].speed_rad_s);
      inputs.wheels[i].time_tag = inputs.epoch;
    }
    commands = fsw ? fsw(inputs) : FswOutputs{};

    for (std::size_t i = 0; i < vehicle_.wheels.size(); ++i) {
      const WheelCommand cmd = i < commands.wheels.size() ? commands.wheels[i] : WheelCommand{};
      if (cmd.mode == WheelCommand::Mode::kSpeed) {
        vehicle_.wheels[i].model.commandSpeed(cmd.value);
      } else {
        vehicle_.wheels[i].model.commandTorque(cmd.value);
      }
    }
    // §7 duty cycle: the commanded dipole is the *peak* the rods are driven at
    // over the on-window, not an average over the step. A window the FSW never
    // scheduled arrives as zero, which leaves the rods off — the safe reading of
    // a flight side that has not taken ownership of the schedule.
    const std::int64_t on_ns =
        std::clamp(static_cast<std::int64_t>(std::llround(commands.mtq_on_window_s * 1.0e9)),
                   static_cast<std::int64_t>(0), macro_ns);
    duty_off_ns = t_ns + on_ns;
    rods_off = on_ns <= 0;
    for (std::size_t i = 0; i < vehicle_.magnetorquers.size(); ++i) {
      const math::Vec3<math::frames::Body> dipole = i < commands.magnetorquer_dipoles.size()
                                                        ? commands.magnetorquer_dipoles[i]
                                                        : math::Vec3<math::frames::Body>::Zero();
      // The rods' hysteresis advances once per command; the produced dipole is
      // then held for the on-window and decays through the quiet window.
      // A zero-length on-window means the rods are never driven this period, so
      // the zero command is what reaches the play operator — commanding the
      // dipole and de-energising in the same instant would leave the driven
      // moment as the start of a settle transient the rod never had.
      const math::Vec3<math::frames::Body> applied =
          rods_off ? math::Vec3<math::frames::Body>::Zero() : dipole;
      mtq_actual[i] = vehicle_.magnetorquers[i].model.commandDipole(applied).eigen();
      if (rods_off) {
        vehicle_.magnetorquers[i].model.deenergize();
      }
    }

    // Reset the IMU accumulators: the FSW has consumed this interval.
    for (ImuAccumulation& acc : inputs.imus) {
      acc.delta_angle_rad = math::Vec3<math::frames::Body>::Zero();
      acc.delta_velocity_mps = math::Vec3<math::frames::Body>::Zero();
      acc.samples = 0;
      acc.valid = true;
    }

    if (trace != nullptr) {
      trace->push_back({static_cast<double>(t_ns) / 1.0e9, s});
    }
  }
  return true;
}

}  // namespace polaris::sim::io

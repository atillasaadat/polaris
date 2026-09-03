#ifndef POLARIS_SIM_IO_CLOSED_LOOP_HPP
#define POLARIS_SIM_IO_CLOSED_LOOP_HPP

/// @file
/// @brief The §2.4 sim-time-driven closed loop: plant → sensors → FSW → actuators.
///
/// The execution model the whole architecture is built around, sim-side. The
/// loop owns simulation time and marches it on an event grid in exact integer
/// nanoseconds, so a long run cannot drift off the sample times. Each interval
/// it: steps the actuators under their zero-order-held commands, writes the net
/// actuator wrench into the plant (reaction-wheel reaction torques through the
/// assembly's W matrix, magnetorquer m×B), propagates the 6DOF state, and
/// samples whichever sensors came due.
///
/// **Only dynamics events break an integration step**: a macro boundary, where
/// the FSW's commands begin to apply, and the magnetorquer duty-window off edge
/// (§7), which a step must never straddle. A sensor sample is an *observation* —
/// it reads the plant and changes nothing — so the integrator runs through it
/// and the sample is served an interpolated state from the propagation's own
/// accepted-step nodes (`dynamics/dense_output.hpp`). Stopping at every sample
/// instead fragments the grid to the fastest sensor's period (0.5 ms for the
/// flown 2000 Hz IMUs), and every fragment pays RK89's 16-stage minimum. Sample
/// order, epochs and RNG draw order are unaffected; anything *published* is an
/// exact integration endpoint, never an interpolant.
///
/// At each **macro-step** boundary (the FSW rate) it publishes
/// the buffered measurements, invokes the FSW, and holds the returned commands
/// for the *next* interval — commands apply one step later, exactly the §2.4
/// contract, so the loop is causal and bit-reproducible from `{config, seed}`.
///
/// **Buffers** (§2.4): the IMU accumulates delta-angle/delta-velocity between
/// FSW reads, so a 10 Hz consumer receives everything a 250 Hz instrument
/// measured with no loss; discrete sensors (star tracker, sun sensor,
/// magnetometer, GNSS) publish their **latest** sample with its validity flag
/// and time tag. Payload sensors (§6.3) are sampled the same way but their
/// products stay on the loop, not in `FswInputs` — they are pointing geometry
/// for the sim side, not measurements for the FSW.
///
/// **The FSW is a callback.** Until the F´ SITL transport lands (Phase 3), the
/// flight side is anything satisfying `FswCallback` — the default returns zero
/// commands (open-loop truth run), and tests script command profiles. The F´
/// two-process barrier will drive this same interface over TCP.
///
/// **This is where the deferred bindings land**: the GNSS jamming map and the
/// scheduled fault timeline are applied per step here (§9.2), and actuator
/// output finally feeds back into the dynamics (§7).
///
/// Everything stochastic stays inside the sensor models; the loop itself is
/// deterministic. Sim-side: heap / std::function are fine.
///
/// Implements the §2.4 execution model and REQ-SIM-004's truth/onboard
/// separation boundary (the FSW sees measurements, never `TruthState`).

#include <cstdint>
#include <Eigen/Core>
#include <functional>
#include <string>
#include <vector>

#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"
#include "sensors/gnss.hpp"
#include "sensors/imu.hpp"
#include "sensors/magnetometer.hpp"
#include "sensors/payload_sensor.hpp"
#include "sensors/star_tracker.hpp"
#include "sensors/sun_sensor.hpp"
#include "time/timescales.hpp"
#include "world/tracked_object.hpp"

namespace polaris::sim::io {

/// Accumulated IMU information since the last FSW read (§2.4): the sums of the
/// per-sample delta-angles/velocities, so no fast-rate information is lost.
struct ImuAccumulation {
  std::string name;  ///< unit instance name
  math::Vec3<math::frames::Body> delta_angle_rad{};
  math::Vec3<math::frames::Body> delta_velocity_mps{};
  int samples{0};        ///< native-rate samples folded in since the last read
  bool valid{true};      ///< false if any contributing sample was invalid
  time::Tai time_tag{};  ///< time of the newest contributing sample
};

/// The latest sample of a discrete sensor, tagged with its unit name.
template <typename Measurement>
struct Latest {
  std::string name;
  Measurement measurement{};
  bool ever_sampled{false};  ///< false until the sensor's first native sample
};

/// One reaction wheel's tachometer reading at a macro boundary — the feedback a
/// real wheel drive reports, and the input the §8.5 momentum management runs on.
/// A measurement, so it crosses the §2.3 boundary; the rotor inertia that turns
/// it into stored momentum is a catalog fact the FSW carries itself.
struct WheelTelemetry {
  std::string name;  ///< unit instance name
  double speed_rad_s{0.0};
  bool valid{true};      ///< false when the drive reports no usable reading
  time::Tai time_tag{};  ///< the macro boundary the speed was read at
};

/// Everything the FSW receives at a macro-step boundary. Measurements only —
/// the §2.3 truth/onboard separation means no `TruthState` crosses this line.
struct FswInputs {
  time::Tai epoch{};  ///< the macro-step boundary, TAI
  std::uint64_t macro_step{0};
  std::vector<ImuAccumulation> imus;
  std::vector<Latest<sensors::StarTrackerMeasurement>> star_trackers;
  std::vector<Latest<sensors::SunSensorMeasurement>> sun_sensors;
  std::vector<Latest<sensors::MagnetometerMeasurement>> magnetometers;
  std::vector<Latest<sensors::GnssMeasurement>> gnss;
  std::vector<WheelTelemetry> wheels;
};

/// One reaction-wheel command: torque mode or speed mode (§7), matching the two
/// interfaces the wheel drive exposes.
struct WheelCommand {
  enum class Mode { kTorque, kSpeed };
  Mode mode{Mode::kTorque};
  double value{0.0};  ///< [N·m] in torque mode, [rad/s] in speed mode
};

/// Everything the FSW returns. Sized to the vehicle's suites (in `Vehicle`
/// vector order); short vectors are zero-padded, extras ignored.
struct FswOutputs {
  std::vector<WheelCommand> wheels;
  /// Commanded dipole per magnetorquer unit, body frame [A·m²]. This is the
  /// **peak** value the rods are driven at during the on-window, not an average
  /// over the step.
  std::vector<math::Vec3<math::frames::Body>> magnetorquer_dipoles;
  /// §7 MTQ/MAG duty-cycle on-window [s] measured from the start of the interval
  /// these commands apply over: the rods carry `magnetorquer_dipoles` for this
  /// long and are then de-energised, their field decaying over the rod model's
  /// settle time. Clamped to the macro step by the loop; zero (the default)
  /// leaves the rods off, so an FSW that never schedules a window produces no
  /// magnetic torque rather than a full-period one.
  double mtq_on_window_s{0.0};
  /// Commanded throttle per thruster unit, 0..1, held for the whole macro step
  /// (§17 finite burns: the FSW holds a throttle for a duration, the plant
  /// integrates what the thruster delivers). Short = zero-padded (off).
  std::vector<double> thruster_throttles;

  /// The flight software's own attitude estimate, Body<-ECI, echoed back for
  /// **diagnosis only**.
  ///
  /// Everything else in this struct is a command the plant acts on. This is not
  /// one, and the plant must never read it: `ClosedLoop` records it in the trace
  /// and passes it nowhere else, so nothing the vehicle does can depend on it.
  /// That separation is what makes a later comparison against truth meaningful
  /// rather than circular.
  ///
  /// It exists because truth alone cannot distinguish "the vehicle pointed
  /// badly" from "the vehicle pointed exactly where it believed, and the belief
  /// was wrong" — different faults, different fixes. `estimate_valid` false
  /// means the FSW had no usable attitude, which is a different state from an
  /// identity quaternion and must not be confused with one.
  math::Quat<math::frames::Body, math::frames::ECI> estimate_attitude{};
  bool estimate_valid{false};
};

/// The flight side of the macro-step handshake. Phase 3's F´ SITL transport
/// implements this over TCP; tests script it; the default flies open loop.
using FswCallback = std::function<FswOutputs(const FswInputs&)>;

/// One payload sensor's latest pointing geometry (§6.3). Deliberately **not** in
/// `FswInputs`: it is truth-derived, and there is no flight-side payload
/// component to receive it — the same place the star tracker's FOV coverage
/// fractions stay. Read it from the loop for analysis and for pointing metrics.
using PayloadGeometry = Latest<sensors::PayloadSensorSample>;

/// One thruster's truth at a macro boundary (§17): what the plant delivered
/// over the step just ended, not what the FSW asked for.
struct ThrusterTelemetry {
  double throttle{0.0};            ///< command applied over the step
  double delivered_thrust_n{0.0};  ///< thrust the plant integrated [N]
  double mass_flow_kg_s{0.0};      ///< propellant rate at that thrust
};

/// One record of the loop's own trace: the truth state at a macro boundary.
struct MacroSample {
  double t_s{0.0};
  state::TruthState state;
  /// Vehicle mass after propellant depletion [kg] (the config mass while no
  /// thruster has fired). What the thrust acceleration is computed with; drag
  /// and SRP keep the build-time mass (see `ClosedLoop::massKg`).
  double mass_kg{0.0};
  std::vector<ThrusterTelemetry> thrusters;
  /// The FSW's attitude estimate at this step, and whether it had one — see
  /// `FswOutputs`. Truth-vs-estimate is the one comparison a SITL row cannot
  /// make from the plant alone, and it is the difference between reporting a
  /// pointing error and attributing it. Kept last so the positional aggregate
  /// initialisers that build a sample stay valid.
  math::Quat<math::frames::Body, math::frames::ECI> estimate_attitude{};
  bool estimate_valid{false};
};

/// The §2.4 closed loop. Build a `SimRunner` **with this loop's wrench** (see
/// `run`), build a `Vehicle`, then run.
///
/// @code
/// scenario::Vehicle vehicle;                          // buildVehicle(...)
/// scenario::SimRunner runner;
/// io::ClosedLoop loop(runner, vehicle);
/// runner.build(config, paths, &err, loop.wrench());   // wrench MUST be composed
///
/// std::vector<io::MacroSample> trace;
/// loop.run(
///     [](const io::FswInputs& in) {                   // the FSW: stub / script / F´
///       io::FswOutputs out;
///       out.wheels.resize(in.imus.size());            // fill wheel + MTQ commands
///       return out;
///     },
///     &trace, &err);
/// @endcode
///
/// The callback sees **measurements only** — `TruthState` never crosses it
/// (§2.3). The default (no callback) flies open loop.
/// An instrument boresight the viewer should draw a field of view for (§8.4.1).
///
/// The sim already models these — `Vehicle::payload_sensors` carries the
/// as-mounted boresight and the field shape — so this is a projection of a
/// truth model rather than a second definition of the camera. It exists because
/// the stream reader cannot see the vehicle.
struct CameraOverlay {
  std::string name;
  Eigen::Vector3d boresight_body = Eigen::Vector3d::UnitZ();
  double half_fov_x_deg = 0.0;
  double half_fov_y_deg = 0.0;
  /// True for a star tracker rather than a payload camera. They are drawn the
  /// same way and answer different questions: the camera's field says whether
  /// the vehicle is looking at what it was told to, the trackers' say whether it
  /// can *know* where it is looking. A row where the second is blocked explains
  /// a pointing error the first cannot.
  bool is_star_tracker = false;
};

class ClosedLoop {
 public:
  /// @param runner   Built plant + environment. The runner must have been built
  ///                 with `wrench()` passed as its extra model, or actuator
  ///                 commands will not reach the dynamics — the momentum-
  ///                 conservation test is what pins this wiring.
  /// @param vehicle  Built sensor/actuator suite (seeded, noise-configured).
  ///                 Held by reference; must outlive the loop.
  ClosedLoop(scenario::SimRunner& runner, scenario::Vehicle& vehicle);

  /// The actuator-feedback model to pass to `SimRunner::build`.
  const dynamics::CommandedWrench* wrench() const { return &wrench_; }

  /// March the configured duration in macro-steps, invoking @p fsw at each
  /// boundary. @p trace receives the truth state at every macro boundary
  /// (cleared first); pass nullptr to discard.
  ///
  /// When the environment variable `POLARIS_SIM_STREAM` names a file, the same
  /// per-boundary truth samples are also appended there as line-flushed JSONL
  /// for an external live viewer (`tools/freeflyer/viz.py`); see
  /// `sim/io/README.md`. Output-only — determinism is untouched.
  ///
  /// @return false on a configuration the loop cannot honour (runner not
  ///         ready / wrench not composed / a data product the vehicle needs is
  ///         missing). @p error receives the reason.
  bool run(const FswCallback& fsw, std::vector<MacroSample>* trace = nullptr,
           std::string* error = nullptr);

  /// Secondary objects the truth side should propagate and stream (§8.4.1).
  ///
  /// Held by pointer and not owned; must outlive the loop. Purely an output
  /// concern — the objects never reach the plant, the sensor models or the
  /// flight software, so adding one cannot change what a run does. That is what
  /// makes it safe to draw them beside the vehicle and still call the picture
  /// evidence.
  void setTrackedObjects(const world::TrackedObjectSet* objects) { tracked_ = objects; }

  /// The payload sensors' latest pointing geometry, in `Vehicle::payload_sensors`
  /// order, as of the end of the last `run`. Sim-side only (see
  /// @ref PayloadGeometry).
  const std::vector<PayloadGeometry>& payloadGeometry() const { return payload_geometry_; }

  /// Current vehicle mass [kg]: the config's `mass_kg` less the propellant the
  /// thrusters have spent. **What sees it:** the thrust acceleration F/m and
  /// this trace. **What does not:** the drag and SRP models, which took the
  /// build-time mass (`SimRunner::build`) — a ponytail: on a 17 kg bus a 0.5 N
  /// thruster spends 0.23 g/s, so a 10-minute burn is 0.8 % of the mass, under
  /// the drag coefficient's own uncertainty; make the environment models read a
  /// shared mass when a mission burns a real fraction of itself.
  double massKg() const { return mass_kg_; }

  /// Data products the loop itself loads (EOP for GNSS ECEF output; the
  /// ephemeris when the runner did not load one but optical sensors need sky
  /// geometry). Defaults to the repo layout via `DataPaths::under`.
  void setDataPaths(const scenario::DataPaths& paths) { paths_ = paths; }

 private:
  scenario::SimRunner& runner_;
  scenario::Vehicle& vehicle_;
  dynamics::CommandedWrench wrench_;
  double mass_kg_{0.0};
  scenario::DataPaths paths_;
  std::vector<PayloadGeometry> payload_geometry_;
  const world::TrackedObjectSet* tracked_ = nullptr;
};

}  // namespace polaris::sim::io

#endif  // POLARIS_SIM_IO_CLOSED_LOOP_HPP

#ifndef POLARIS_SIM_IO_CLOSED_LOOP_HPP
#define POLARIS_SIM_IO_CLOSED_LOOP_HPP

/// @file
/// @brief The §2.4 sim-time-driven closed loop: plant → sensors → FSW → actuators.
///
/// The execution model the whole architecture is built around, sim-side. The
/// loop owns simulation time and marches it in **micro-steps** set by the
/// sensors' own native rates (an event grid in exact integer nanoseconds, so a
/// long run cannot drift off the sample times). Each micro-interval it: steps
/// the actuators under their zero-order-held commands, writes the net actuator
/// wrench into the plant (reaction-wheel reaction torques through the assembly's
/// W matrix, magnetorquer m×B), propagates the 6DOF state, and samples whichever
/// sensors came due. At each **macro-step** boundary (the FSW rate) it publishes
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
  /// Commanded dipole per magnetorquer unit, body frame [A·m²].
  std::vector<math::Vec3<math::frames::Body>> magnetorquer_dipoles;
};

/// The flight side of the macro-step handshake. Phase 3's F´ SITL transport
/// implements this over TCP; tests script it; the default flies open loop.
using FswCallback = std::function<FswOutputs(const FswInputs&)>;

/// One payload sensor's latest pointing geometry (§6.3). Deliberately **not** in
/// `FswInputs`: it is truth-derived, and there is no flight-side payload
/// component to receive it — the same place the star tracker's FOV coverage
/// fractions stay. Read it from the loop for analysis and for pointing metrics.
using PayloadGeometry = Latest<sensors::PayloadSensorSample>;

/// One record of the loop's own trace: the truth state at a macro boundary.
struct MacroSample {
  double t_s{0.0};
  state::TruthState state;
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
  /// @return false on a configuration the loop cannot honour (runner not
  ///         ready / wrench not composed / a data product the vehicle needs is
  ///         missing). @p error receives the reason.
  bool run(const FswCallback& fsw, std::vector<MacroSample>* trace = nullptr,
           std::string* error = nullptr);

  /// The payload sensors' latest pointing geometry, in `Vehicle::payload_sensors`
  /// order, as of the end of the last `run`. Sim-side only (see
  /// @ref PayloadGeometry).
  const std::vector<PayloadGeometry>& payloadGeometry() const { return payload_geometry_; }

  /// Data products the loop itself loads (EOP for GNSS ECEF output; the
  /// ephemeris when the runner did not load one but optical sensors need sky
  /// geometry). Defaults to the repo layout via `DataPaths::under`.
  void setDataPaths(const scenario::DataPaths& paths) { paths_ = paths; }

 private:
  scenario::SimRunner& runner_;
  scenario::Vehicle& vehicle_;
  dynamics::CommandedWrench wrench_;
  scenario::DataPaths paths_;
  std::vector<PayloadGeometry> payload_geometry_;
};

}  // namespace polaris::sim::io

#endif  // POLARIS_SIM_IO_CLOSED_LOOP_HPP

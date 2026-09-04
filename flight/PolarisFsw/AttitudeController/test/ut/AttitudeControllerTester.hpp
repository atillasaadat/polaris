// ======================================================================
// \title  AttitudeControllerTester.hpp
// \brief  Test harness for the AttitudeController component (§8.5, §7, §23.1)
//
// The control *laws* are pinned in tests/unit/gnc_control_test.cpp against
// lib/gnc, so what this harness tests is the component: the mode ladder and its
// refusal paths, the parameter-missing refusal, the §7 duty-cycle schedule the
// magnetometer consumers gate on, and the §9 stuck-on monitor's confirmation and
// re-admission edges.
//
// The harness drives `estimateIn` directly, so the estimate the controller acts
// on — including its validity flags, its covariance and its magnetic block — is
// fully controlled by the test, and captures the three output ports.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDECONTROLLER_TESTER_HPP
#define FLIGHT_POLARISFSW_ATTITUDECONTROLLER_TESTER_HPP

#include <Eigen/Core>

#include "AttitudeControllerGTestBase.hpp"
#include "flight/PolarisFsw/AttitudeController/AttitudeController.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"

namespace flight {

class AttitudeControllerTester : public AttitudeControllerGTestBase {
 public:
  static const U32 MAX_HISTORY_SIZE = 2000;
  static const FwEnumStoreType TEST_INSTANCE_ID = 0;

  AttitudeControllerTester();

  ~AttitudeControllerTester();

  // ----------------------------------------------------------------------
  // Tests
  // ----------------------------------------------------------------------

  //! A cycle with no parameters in ParameterDb refuses, warns once, stays IDLE
  //! and commands zero on every actuator — it never substitutes a gain (§19.3).
  void testRefusesWithoutParameters();

  //! CTRL_MODE_SET(POINT) is refused, leaving the mode where it was, when the
  //! estimate is missing, stale, invalid, above the quality floor, or when no
  //! target has been set; IDLE is always accepted.
  void testPointRefusalPaths();

  //! With a fine-quality estimate and a target, POINT engages, the pointing
  //! error falls, and the allocated wheel torques reproduce the commanded body
  //! torque.
  //! Saturation is counted per mode and the paired event carries the
  //! pre-clip demand (Push 69 follow-on).
  void testSaturationIsCountedAndReportedUnclipped();

  void testPointEngagesAndReducesError();

  //! DETUMBLE commands a dipole whose components oppose the measured field
  //! derivative, and the §7 schedule it publishes leaves a quiet window inside
  //! the control period.
  void testDetumbleCommandsOpposingDipole();

  //! The published schedule's invariants: the quiet window opens a full settle
  //! time after the on-window, closes at the period boundary, and is the whole
  //! period on a cycle that commands no dipole.
  void testDutyCycleScheduleInvariants();

  //! A field-magnitude residual through the quiet window latches the stuck-on
  //! monitor after the confirmation count, marks the interlock unhealthy, and
  //! clears again on the same criterion after the clear count.
  void testStuckOnMonitorLatchesAndClears();

  //! A single-rod command makes the attribution decisive; three rods driven
  //! together make it ambiguous, and the EVR says so rather than guessing.
  void testStuckOnAttribution();

  //! CTRL_RESET drops the latch, the integrator and the mode.
  void testResetClearsState();

  //! Desaturation engages autonomously inside POINT when the wheels pass the
  //! momentum threshold, drives the rods concurrently with the wheels, and
  //! disengages once the momentum has been low for the confirmation count. Both
  //! edges are asserted, so "it disengages" is not read off a latch never set.
  void testDesatEngagesAndDisengagesInPoint();

  //! Desaturation is never active outside POINT — least of all in DETUMBLE,
  //! where B-dot owns the rods and two laws on one actuator would be two
  //! vehicles' worth of commands on one set of coils.
  void testDesatExcludedFromDetumbleAndIdle();

  //! CTRL_DESAT: INHIBIT stops a desaturation in progress, FORCE starts one the
  //! momentum predicate would not have asked for, and AUTO hands the decision
  //! back. FORCE is a permission, so it commands nothing without a field.
  void testDesatGroundOverride();

  //! §8.5 drive friction feedforward (REQ-ACTL-010): on a spinning array the
  //! commanded wheel torque is the allocation's demand plus the modelled
  //! friction, the two are separable from telemetry, disabling the feedforward
  //! puts the demand back exactly, a wheel with no usable tachometer is passed
  //! through uncompensated rather than given a guessed sign, and a missing
  //! friction coefficient refuses the whole configuration rather than flying the
  //! feedforward silently off.
  void testWheelFrictionFeedforward();

  //! Stored momentum past the envelope raises the §9 event once and recovers on
  //! the same comparison; a wheel with no usable speed refuses the momentum sum
  //! rather than understating it, and holds the latch where it was.
  void testMomentumEnvelopeAndWheelDropout();
  void testWheelCapacityMonitorSeesNullSpaceMomentum();
  //! The wheel-speed bias servo adds a null-space trim (zero body torque),
  //! rests when the wheels sit at the pattern, and refuses a pattern past
  //! capacity.
  void testWheelBiasServoAddsNullSpaceTorqueOnly();
  //! A bias pattern at or past WheelCapacityNms is a configuration refusal.
  void testWheelBiasPastCapacityIsRefused();

 private:
  // ----------------------------------------------------------------------
  // Captured outputs
  // ----------------------------------------------------------------------

  void from_wheelCmdOut_handler(FwIndexType portNum, const WheelTorqueSet& cmds) override;

  void from_mtqCmdOut_handler(FwIndexType portNum, const MtqDipoleSet& cmds,
                              F64 onWindowSec) override;

  void from_mtqActuationOut_handler(FwIndexType portNum, const MtqActuation& state) override;

  //! Generated by fpp-to-cpp into AttitudeControllerTesterHelpers.cpp.
  void connectPorts();
  void initComponents();

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Load a complete, self-consistent tuning set. @p dutyFactor and @p settleSec
  //! are exposed because the schedule tests vary them. @p withFriction drops the
  //! §8.5 friction coefficients, for the one test that has to see a vehicle whose
  //! friction model is missing refused rather than flown.
  void setValidParameters(F64 dutyFactor = 0.5, F64 settleSec = 0.03, bool withFriction = true);

  //! Publish one estimate and run one control cycle at @p taiNs.
  void runCycleAt(I64 taiNs);

  //! Stage the estimate the next @ref runCycleAt publishes.
  void setEstimate(const polaris::math::Quaternion& q, const Eigen::Vector3d& rate, double sigmaRad,
                   I64 epochNs);

  //! Stage the magnetic block of the estimate: measured field, its time tag, and
  //! the attitude-free modelled magnitude.
  void setMagnetic(const Eigen::Vector3d& fieldT, I64 tagNs, double modelMagnitudeT,
                   bool fieldValid = true, bool modelValid = true);

  //! Clear the magnetic block (no admissible sample this cycle).
  void clearMagnetic();

  //! Stage a common speed [rad/s] on every wheel, all readings valid. With the
  //! body-diagonal pyramid an equal-speed set stores momentum along +Z only,
  //! which makes the momentum under test a single number. Published by the next
  //! @ref runCycleAt with that cycle's time tag, because the controller gates the
  //! tachometers on staleness exactly as it gates the estimate.
  void setWheelSpeeds(double speedRadps, bool valid = true);

  //! Wheel speed [rad/s], common to all four, that stores @p momentumNms about
  //! body +Z on the pyramid.
  static double speedForMomentum(double momentumNms);

  AttitudeController component;

  //! Last captured commands and schedule.
  WheelTorqueSet last_wheels_{};
  MtqDipoleSet last_dipoles_{};
  F64 last_on_window_s_{-1.0};
  MtqActuation last_schedule_{};
  U32 wheel_cmd_count_{0};
  U32 mtq_cmd_count_{0};
  U32 schedule_count_{0};

  //! The estimate and the wheel tachometers staged for the next cycle.
  AttitudeEstimate estimate_{};
  double wheel_speed_radps_[4] = {0.0, 0.0, 0.0, 0.0};
  bool wheel_speed_valid_[4] = {true, true, true, true};
};

}  // namespace flight

#endif  // FLIGHT_POLARISFSW_ATTITUDECONTROLLER_TESTER_HPP

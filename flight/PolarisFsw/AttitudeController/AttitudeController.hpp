// ======================================================================
// \title  AttitudeController.hpp
// \brief  Attitude control component: B-dot detumble, quaternion PID pointing,
//         wheel allocation, momentum management and the MTQ/MAG duty-cycle
//         interlock (§8.5, §7, §9; REQ-ACTL-001..005, -009..-011)
//
// The F´ wrapper around polaris::gnc::BdotController, polaris::gnc::AttitudePid,
// polaris::gnc::RwAllocator, polaris::gnc::MomentumManager and
// polaris::gnc::DisturbanceObserver. It consumes the §8.0 attitude estimate the
// AttitudeEstimator published earlier in this same rate-group cycle plus the
// wheel tachometers, produces the actuator commands for the next interval, owns
// the §7 duty-cycle schedule and runs the §9 stuck-on rod, momentum-envelope and
// momentum-anomaly monitors. No control math lives here.
//
// Flight rules — no heap after init, no exceptions, fixed-size storage, every
// return code checked, finiteness guards on every published command.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDECONTROLLER_HPP
#define FLIGHT_POLARISFSW_ATTITUDECONTROLLER_HPP

#include <cstdint>

#include "flight/PolarisFsw/AttitudeController/AttitudeControllerComponentAc.hpp"
// AllocMethod appears in no port, command, event or telemetry channel — it names
// the *values* of the U8 AllocMethodSel parameter — so the component base header
// does not pull it in and it is included here explicitly.
#include "flight/PolarisFsw/AttitudeController/AttitudeController_AllocMethodEnumAc.hpp"
#include "gnc/attitude_pid.hpp"
#include "gnc/bdot.hpp"
#include "gnc/disturbance.hpp"
#include "gnc/momentum.hpp"
#include "gnc/rw_allocation.hpp"
#include "gnc/rw_friction.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace flight {

class AttitudeController final : public AttitudeControllerComponentBase {
 public:
  //! Short names for the component-scoped enums the autocoder emits as
  //! `flight::AttitudeController_*`. Aliased rather than spelled out so the mode
  //! ladder below reads as the FPP model does.
  using CtrlMode = AttitudeController_CtrlMode;
  using AllocMethod = AttitudeController_AllocMethod;
  using CtrlRefusal = AttitudeController_CtrlRefusal;
  using StuckAttribution = AttitudeController_StuckAttribution;
  using DesatOverride = AttitudeController_DesatOverride;
  using MomentumRefusalEv = AttitudeController_MomentumRefusal;

  //! Rods this component drives. The §7 interlock resolves the commanded body
  //! dipole onto an orthogonal triad, clamps each rod, and re-expands; anything
  //! else needs the allocation layer the wheels use, which is not what this push
  //! ships. MtqCount is checked against this.
  static constexpr U32 kRodCount = 3;

  //! Tolerance on the rod triad's mutual orthogonality (|axis_i . axis_j|).
  //! Loose enough for hand-written unit vectors, tight enough that a genuinely
  //! skewed set is refused rather than approximated.
  static constexpr F64 kRodOrthogonalityTol = 1.0e-6;

  explicit AttitudeController(const char* const compName);

  ~AttitudeController();

  //! Dispatch CTRL_SET_TARGET_Q and CTRL_MODE_SET at topology setup, for SITL
  //! and bench runs (design doc §23.1.1). On a flight vehicle both come from the
  //! ground — or, from Phase 7, from the mode manager — and the deployment in
  //! those runs has no uplink. The commands go through this component's own
  //! command port, so what executes is the flight handler with its full argument
  //! deserialisation and refusal logic; the only thing skipped is the radio.
  //!
  //! @param mode 0 = leave in IDLE (no command issued), otherwise the CtrlMode
  //!        value to request.
  //! @param q the inertial-hold target (JPL scalar-first), issued before the mode
  //!        so a POINT request has a target to be accepted against. A null-norm
  //!        quaternion skips the target command.
  //!
  //! Must follow loadParameters(): a mode request against an unconfigured
  //! controller is refused, which is the correct behaviour and the wrong test.
  void commandModeAtStartup(U32 mode, const F64 q[4]);

  //! Override the §8.5 feedforward enables at topology setup, for SITL and bench
  //! runs (design doc §23.1.1). It exists for exactly one experiment: flying the
  //! *same* vehicle with and without disturbance feedforward, which is the only
  //! way to measure what the feedforward buys. On a flight vehicle both values
  //! come from ParameterDb (§19.3) and change by uplink.
  //!
  //! Writes the component's own parameter cache and re-reads the tuning, so what
  //! runs afterwards is the ordinary `applyParameters` path. Must follow
  //! loadParameters(), which would otherwise overwrite it.
  //!
  //! @param model    tier 1 (gravity gradient + residual dipole) enabled.
  //! @param observer tier 2 (the momentum-based observer) contributing to the
  //!        feedforward. The observer itself runs either way — it is also the §9
  //!        anomaly monitor.
  void setFeedforwardAtStartup(bool model, bool observer);

 private:
  // ----------------------------------------------------------------------
  // Port handlers
  // ----------------------------------------------------------------------

  //! Control cycle: one law evaluation and one actuator command set (§8.5).
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Latch the estimator's product for this cycle.
  void estimateIn_handler(FwIndexType portNum, const AttitudeEstimate& estimate) override;

  //! Latch one wheel's tachometer reading for this cycle.
  void wheelSpeedIn_handler(FwIndexType portNum, const WheelSpeedMeas& meas) override;

  // ----------------------------------------------------------------------
  // Commands
  // ----------------------------------------------------------------------

  void CTRL_MODE_SET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, CtrlMode mode) override;

  void CTRL_SET_TARGET_Q_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 q0, F64 q1, F64 q2,
                                    F64 q3) override;

  void CTRL_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  void CTRL_DESAT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, DesatOverride action) override;

  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Configuration
  // ----------------------------------------------------------------------

  //! Read every parameter and rebuild the three laws. Emits ConfigInvalid (edge
  //! gated) and leaves the controller inert on a missing or out-of-range value —
  //! there are no flight defaults (§19.3), and a control law on an invented gain
  //! is worse than no control law.
  //! @return true when the controller is configured and may command actuators.
  bool applyParameters();

  //! Read the per-unit axis parameter @p axes into @p out, normalising each of
  //! the first @p count slots. @return false on a non-finite or null slot.
  static bool readAxes(const Vec3F64PerUnit& axes, U32 count, Eigen::Vector3d* out);

  // ----------------------------------------------------------------------
  // Cycle helpers
  // ----------------------------------------------------------------------

  //! Master clock as TAI nanoseconds.
  I64 currentTaiNs() const;

  //! Whether the latched estimate can be acted on this cycle at @p nowNs, and
  //! why not. Checks arrival, staleness, the validity flags the active mode
  //! needs, and (for POINT) the MaxAttSigmaRad quality floor.
  bool estimateUsable(I64 nowNs, CtrlMode::T mode, CtrlRefusal::T& reason) const;

  //! Run the B-dot law on this cycle's admissible field sample; fills @p dipole
  //! with the body-frame command [A*m^2] and @ref commanded_mask_ with the rods
  //! it energises. @return false when no command was produced (@p reason says
  //! why), leaving @p dipole zero.
  bool runDetumble(polaris::math::Vec3<polaris::math::frames::Body>& dipole,
                   CtrlRefusal::T& reason);

  //! Run the pointing PID and the wheel allocation; fills @p wheelTorque with
  //! `wheel_count_` commanded torques [N*m]. @return false when no command was
  //! produced, leaving @p wheelTorque zeroed.
  bool runPoint(double dtSec, double* wheelTorque, CtrlRefusal::T& reason);

  //! Fold this cycle's wheel tachometers into the momentum manager, run the §9
  //! envelope monitor, and telemeter the result. Runs in **every** mode: the
  //! envelope is a vehicle-level fault condition, not a POINT diagnostic.
  //! @param nowNs cycle epoch, TAI ns — a tachometer reading older than
  //!        `MaxEstimateAgeSec` is treated as no reading at all (§9.1), which
  //!        refuses the momentum sum rather than computing it from a speed the
  //!        wheel had at some earlier time.
  //! @return true when @ref momentum_ produced a usable state this cycle.
  bool updateMomentum(I64 nowNs);

  //! Run the §8.5 tier-1/tier-2 disturbance chain and produce this cycle's
  //! feedforward torque (already negated — it is the torque the actuators must
  //! supply). Also drives the §9 momentum-anomaly monitor, which is the same
  //! estimator read by a second consumer. Runs in **every** mode, for the same
  //! reason the stuck-on monitor does: an unmodelled torque is a vehicle fault
  //! whatever the controller happens to be doing.
  //! @param nowNs cycle epoch, TAI ns. Leaves the feedforward in
  //!        @ref feedforward_nm_, zeroed when nothing is enabled or available.
  void updateDisturbance(I64 nowNs);

  //! Whether the rods should desaturate this cycle, honouring the mode, the
  //! ground override and the momentum predicate. DETUMBLE and IDLE always answer
  //! false — in DETUMBLE the rods are B-dot's, and two laws on one actuator is
  //! not a schedule. Pure: the engage/disengage edges are emitted by the caller,
  //! once, where the transition is visible.
  bool desatDue() const;

  //! Run the cross-product desaturation law on this cycle's field and momentum
  //! error. @return false when no dipole was produced, leaving @p dipole zero.
  bool runDesat(polaris::math::Vec3<polaris::math::frames::Body>& dipole);

  //! Resolve @p demand onto the rod triad, clamp each rod to its rating, and
  //! re-expand — the one clamp on the magnetic path, shared by B-dot and
  //! desaturation so the two cannot saturate differently. Sets
  //! @ref commanded_mask_ and raises the bounded-cadence DipoleSaturated warning.
  //! @return false when the clamped result is not finite.
  bool clampDipoleToRods(const polaris::math::Vec3<polaris::math::frames::Body>& demand,
                         polaris::math::Vec3<polaris::math::frames::Body>& applied);

  //! Record the body torque @p dipole will produce in @p field over the interval
  //! this cycle commands, duty-averaged. Both magnetic laws call it; see the
  //! member it writes for why the observer reads the previous cycle's value.
  void noteMagneticTorque(const polaris::math::Vec3<polaris::math::frames::Body>& dipole,
                          const polaris::math::Vec3<polaris::math::frames::Body>& field);

  //! Emit the actuator commands and the duty-cycle schedule for the next
  //! interval. Called exactly once per cycle from every path, including the
  //! refusal paths — an actuator that is not commanded holds its last command,
  //! so a refused cycle must still command zero rather than say nothing.
  void commandActuators(I64 nowNs, const polaris::math::Vec3<polaris::math::frames::Body>& dipole,
                        const double* wheelTorque, bool rodsActive);

  //! §9 stuck-on rod monitor: a field-magnitude residual against the onboard
  //! IGRF that persists through the quiet window with the rods de-energised.
  //! Attitude-free by construction (magnitudes, not vectors), because judging a
  //! magnetometer through an attitude that magnetometer helped build is the
  //! circularity that latches out healthy units.
  void runStuckOnMonitor();

  //! Change mode, emitting ModeChanged on a transition only.
  void setMode(CtrlMode::T next);

  //! The guards a mode request has to clear, shared by the command handler and
  //! the startup latch. IDLE is unconditional; everything else needs a usable
  //! estimate, and POINT additionally needs a target.
  //! @return true when the mode was entered; @p reason says why not otherwise.
  bool tryEnterMode(CtrlMode::T requested, CtrlRefusal::T& reason);

  //! An event is due at the AlertCycles cadence for a condition that has now
  //! persisted @p streak cycles: the first cycle of the run, then one per
  //! AlertCycles. A 10 Hz loop that events every cycle floods the downlink and
  //! buries the transition that mattered.
  bool alertDue(U32 streak) const;

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  //! The three laws, rebuilt whenever the tuning changes. Default-constructed
  //! (unconfigured) they refuse every call, which is the inert state.
  polaris::gnc::BdotController bdot_{};
  polaris::gnc::AttitudePid pid_{};
  polaris::gnc::RwAllocator allocator_{};
  polaris::gnc::RwFrictionCompensator friction_{};
  polaris::gnc::RateHysteresis rate_hysteresis_{};
  polaris::gnc::MomentumManager momentum_{};
  polaris::gnc::DisturbanceObserver observer_{};
  //! The desaturation law is stateless, so its tuning is held rather than an
  //! object built from it.
  polaris::gnc::MtqDesatConfig desat_config_{};

  //! Latest estimate off `estimateIn`, and whether one has ever arrived.
  AttitudeEstimate estimate_{};
  bool have_estimate_{false};

  //! Latest wheel tachometer readings off `wheelSpeedIn`, indexed by port. A
  //! reading that has never arrived stays invalid, so an unwired wheel refuses
  //! the momentum sum rather than contributing a zero that would understate it.
  F64 wheel_speed_radps_[polaris::gnc::kMaxWheels] = {};
  bool wheel_speed_valid_[polaris::gnc::kMaxWheels] = {};
  I64 wheel_speed_time_ns_[polaris::gnc::kMaxWheels] = {};

  //! This cycle's per-wheel "valid **and** fresh" verdict on the tachometers,
  //! computed once in @ref updateMomentum and read again by the friction
  //! feedforward. One question, one answer: a second staleness gate for the same
  //! sensors is a second chance for the two to disagree about which wheels the
  //! vehicle can currently reason about.
  bool wheel_speed_fresh_[polaris::gnc::kMaxWheels] = {};

  //! This cycle's momentum state and whether it is usable.
  polaris::gnc::MomentumState momentum_state_{};

  //! Body torque the rods will apply over the interval this cycle commands
  //! [N*m] — from **either** magnetic law, averaged over the control period by
  //! the duty factor and computed from the **clamped** dipole. Zero when the rods
  //! are not driven.
  polaris::math::Vec3<polaris::math::frames::Body> magnetic_torque_nm_{Eigen::Vector3d::Zero()};

  //! The same quantity from the *previous* cycle, which is the one the
  //! disturbance observer must subtract: the momentum change it differences
  //! happened over the interval the previous cycle commanded (§2.4 — commands
  //! apply one step later). Using this cycle's would leave the vehicle's own
  //! magnetic actuation in the residual, and B-dot's torque alone is more than
  //! ten times the §9 anomaly budget — a detumble would alarm every time.
  polaris::math::Vec3<polaris::math::frames::Body> magnetic_torque_prev_nm_{
      Eigen::Vector3d::Zero()};

  //! This cycle's feedforward torque [N*m] — minus the modelled and observed
  //! disturbance, as enabled. Computed once per cycle in every mode and consumed
  //! by POINT.
  polaris::math::Vec3<polaris::math::frames::Body> feedforward_nm_{Eigen::Vector3d::Zero()};

  //! Commanded control mode and the inertial-hold target.
  CtrlMode::T mode_{CtrlMode::IDLE};
  polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI> target_{};
  bool have_target_{false};

  //! Cached tuning the cycle reads directly.
  F64 control_period_s_{0.0};
  F64 max_estimate_age_s_{0.0};
  F64 max_att_sigma_rad_{0.0};
  F64 pid_max_torque_nm_{0.0};
  F64 bdot_max_dipole_am2_{0.0};
  F64 mtq_duty_factor_{0.0};
  F64 mtq_settle_s_{0.0};
  F64 mtq_window_tolerance_s_{0.0};
  F64 mtq_stuck_residual_t_{0.0};
  U32 mtq_stuck_confirm_cycles_{0};
  U32 mtq_stuck_clear_cycles_{0};
  U32 alert_cycles_{0};
  U32 wheel_count_{0};
  polaris::gnc::RwAllocationMethod alloc_method_{polaris::gnc::RwAllocationMethod::kMinNorm};
  //! Whether the §8.5 drive friction feedforward is applied. Its coefficients are
  //! read and validated whatever this says, so a configuration missing the
  //! friction model is refused rather than quietly flown with it disabled.
  bool friction_enabled_{false};

  //! Momentum-management tuning the cycle reads directly.
  F64 wheel_inertia_kgm2_{0.0};
  Eigen::Vector3d inertia_diag_kgm2_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d residual_dipole_am2_{Eigen::Vector3d::Zero()};
  F64 momentum_envelope_nms_{0.0};
  F64 disturbance_budget_nm_{0.0};
  bool feedforward_model_{false};
  bool feedforward_observer_{false};
  //! SITL/bench feedforward override (@ref setFeedforwardAtStartup). Held rather
  //! than written into the parameter cache so a later `parameterUpdated` cannot
  //! silently revert the experiment mid-run.
  bool feedforward_override_{false};
  bool feedforward_model_override_{false};
  bool feedforward_observer_override_{false};
  //! Wheel spin axes, body frame, unit norm — the array's W columns. Held
  //! alongside the allocator's *negated* copy because momentum and torque
  //! authority genuinely differ by that sign (see gnc/momentum.hpp).
  Eigen::Vector3d wheel_axes_[polaris::gnc::kMaxWheels] = {};

  //! Friction feedforward applied to each wheel this cycle [N*m], or NaN where
  //! none was (feedforward off, no usable tachometer, a refused compensation, or
  //! a cycle that commanded no wheel torque at all). Reset every cycle and
  //! published by @ref commandActuators rather than by the pointing law, so the
  //! channel describes what was **commanded** on every path — including the
  //! refusal paths, where the honest answer is "none".
  F64 friction_nm_[polaris::gnc::kMaxWheels] = {};

  //! Rod dipole axes, body frame, unit norm. Index i is rod i's command axis.
  Eigen::Vector3d rod_axes_[kRodCount] = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                          Eigen::Vector3d::Zero()};

  //! Set when loadParameters succeeded; false leaves the controller inert.
  bool configured_{false};
  //! ConfigInvalid is edge-gated: one event per transition into the bad state.
  bool config_alerted_{false};
  //! AllocationFallback is emitted once per configuration, not per cycle.
  bool alloc_fallback_alerted_{false};

  //! TAI ns of the last accepted control cycle, for the PID integrator step.
  I64 last_cycle_ns_{0};
  bool have_last_cycle_{false};

  //! TAI ns of the last cycle of any kind, and the streak of cycles whose
  //! measured spacing disagrees with ControlPeriodSec. Separate from
  //! `last_cycle_ns_` because the §7 schedule is published on **every** cycle,
  //! refused or not, so the period that matters is every cycle's.
  I64 last_run_ns_{0};
  bool have_last_run_{false};
  U32 period_mismatch_streak_{0};

  //! Consecutive refused cycles and the reason, for the bounded-cadence EVR.
  U32 refusal_streak_{0};
  CtrlRefusal::T refusal_reason_{CtrlRefusal::NOT_CONFIGURED};
  U32 saturation_streak_{0};
  U32 dipole_saturation_streak_{0};

  //! §7 stuck-on latch and its persistence counters. `stuck_mask_` is the set of
  //! rods named by a decisive attribution; the interlock is unhealthy whenever
  //! `stuck_confirmed_` is set, whether or not a rod could be named.
  U32 stuck_mask_{0};
  U32 stuck_candidate_mask_{0};
  U32 stuck_streak_{0};
  U32 clear_streak_{0};
  bool stuck_confirmed_{false};

  //! Rods carrying a non-zero command in the period now being judged — the
  //! candidate set for an attribution.
  U32 commanded_mask_{0};

  //! Desaturation state: the ground override, whether the rods are unloading the
  //! wheels this cycle, and the §9 envelope/anomaly edges (edge-gated, so a
  //! persistent condition costs one event pair and not one per 10 Hz cycle).
  DesatOverride::T desat_override_{DesatOverride::AUTO};
  bool desat_active_{false};
  bool envelope_alerted_{false};
  bool anomaly_alerted_{false};

  //! Consecutive cycles the momentum accounting has refused — the
  //! MomentumUnavailable cadence counter, zeroed by the first usable state. A
  //! refusal blinds desaturation and both §9 momentum monitors at once, which is
  //! why it gets an event stream and not just a NaN channel.
  U32 momentum_refusal_streak_{0};

  //! Control mode latched by @ref commandModeAtStartup and retried each cycle
  //! until it is accepted. Zero when there is nothing pending.
  U32 pending_mode_{0};

  //! Cycle counters (telemetry).
  U32 cycles_run_{0};
  U32 cycles_refused_{0};
};

}  // namespace flight

#endif  // FLIGHT_POLARISFSW_ATTITUDECONTROLLER_HPP

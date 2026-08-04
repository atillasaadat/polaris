// ======================================================================
// \title  AttitudeController.hpp
// \brief  Attitude control component: B-dot detumble, quaternion PID pointing,
//         wheel allocation and the MTQ/MAG duty-cycle interlock
//         (§8.5, §7, §9; REQ-ACTL-001..004)
//
// The F´ wrapper around polaris::gnc::BdotController, polaris::gnc::AttitudePid
// and polaris::gnc::RwAllocator. It consumes the §8.0 attitude estimate the
// AttitudeEstimator published earlier in this same rate-group cycle, produces
// the actuator commands for the next interval, owns the §7 duty-cycle schedule
// and runs the §9 stuck-on rod monitor. No control math lives here.
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
#include "gnc/rw_allocation.hpp"
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

 private:
  // ----------------------------------------------------------------------
  // Port handlers
  // ----------------------------------------------------------------------

  //! Control cycle: one law evaluation and one actuator command set (§8.5).
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Latch the estimator's product for this cycle.
  void estimateIn_handler(FwIndexType portNum, const AttitudeEstimate& estimate) override;

  // ----------------------------------------------------------------------
  // Commands
  // ----------------------------------------------------------------------

  void CTRL_MODE_SET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, CtrlMode mode) override;

  void CTRL_SET_TARGET_Q_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 q0, F64 q1, F64 q2,
                                    F64 q3) override;

  void CTRL_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

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
  void runStuckOnMonitor(I64 nowNs);

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
  polaris::gnc::RateHysteresis rate_hysteresis_{};

  //! Latest estimate off `estimateIn`, and whether one has ever arrived.
  AttitudeEstimate estimate_{};
  bool have_estimate_{false};

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

  //! Control mode latched by @ref commandModeAtStartup and retried each cycle
  //! until it is accepted. Zero when there is nothing pending.
  U32 pending_mode_{0};

  //! Cycle counters (telemetry).
  U32 cycles_run_{0};
  U32 cycles_refused_{0};
};

}  // namespace flight

#endif  // FLIGHT_POLARISFSW_ATTITUDECONTROLLER_HPP

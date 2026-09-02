// ======================================================================
// \title  SitlBridge.hpp
// \brief  Flight end of the SITL lockstep transport (design doc §2.2, §2.4)
//
// Decodes deframed SITL payloads (lib/sitl/wire.hpp) arriving from a dedicated
// SITL comm stack and answers them, per the two-process macro-step barrier. The
// byte-level protocol lives in polaris::sitl::SitlHandler; this component is the
// F´ shell: buffer plumbing, events, telemetry, and the §2.4 rate-group cycle.
// Flight rules apply — no heap after init, no exceptions, fixed reply buffer,
// every message validated.
//
// Each STEP_REQ drives the FSW's 10 Hz cycle synchronously (§2.4 steps 3-4):
// decode the request, push the sim epoch to the time provider, fire the SITL
// rate group (which commands actuators back on wheelCmdIn/mtqCmdIn), then build
// the STEP_REPLY from those latched commands.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_SITLBRIDGE_HPP
#define FLIGHT_POLARISFSW_SITLBRIDGE_HPP

#include "flight/PolarisFsw/SitlBridge/SitlBridgeComponentAc.hpp"
#include "sitl/handler.hpp"
#include "sitl/wire.hpp"

namespace flight {

class SitlBridge final : public SitlBridgeComponentBase {
 public:
  //! Construct SitlBridge object
  explicit SitlBridge(const char* const compName);

  //! Destroy SitlBridge object
  ~SitlBridge();

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! Deframed SITL payload from the SITL FprimeDeframer. Validates, builds a
  //! reply via SitlHandler, and frames it back out on dataOut. Runs on the
  //! TcpClient receive task.
  void dataIn_handler(FwIndexType portNum, Fw::Buffer& data,
                      const ComCfg::FrameContext& context) override;

  //! Ownership of the reply buffer returned by the framer. The reply lives in a
  //! fixed member buffer, so there is nothing to deallocate.
  void dataReturnIn_handler(FwIndexType portNum, Fw::Buffer& data,
                            const ComCfg::FrameContext& context) override;

  //! Latch the rate group's reaction-wheel torque commands for the next reply.
  //! Invoked synchronously during sitlCycleOut, on this same task.
  void wheelCmdIn_handler(FwIndexType portNum, const flight::WheelTorqueSet& cmds) override;

  //! Latch the rate group's magnetorquer dipole commands and the §7 duty-cycle
  //! on-window for the next reply.
  void mtqCmdIn_handler(FwIndexType portNum, const flight::MtqDipoleSet& cmds,
                        F64 onWindowSec) override;

  //! Latch the burn executor's thruster throttles for the next reply.
  void attitudeEstimateIn_handler(FwIndexType portNum, const AttitudeEstimate& estimate) override;
  void thrusterCmdIn_handler(FwIndexType portNum, const flight::ThrusterThrottleSet& cmds) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Run the §2.4 macro-step cycle for a decoded STEP_REQ and send the reply:
  //! push the sim epoch, fire the SITL rate group, build the STEP_REPLY from the
  //! latched commands. Emits MalformedMessage and sends nothing on reply overflow.
  void runStepCycle(const polaris::sitl::HandleResult& result, const ComCfg::FrameContext& context,
                    U32 reqBytes);

  //! Wrap the fixed reply buffer (first @p len bytes) and send it to the SITL
  //! framer for framing and downlink.
  void sendReply(FwSizeType len, const ComCfg::FrameContext& context);

  //! Republish this STEP_REQ's decoded sensor records on the GncPorts
  //! measurement outputs, unit i to port i, before the rate group is cycled.
  //! @p epochTaiNs is this step's sim epoch, used to derive the IMU
  //! accumulation interval from the previous step.
  void publishMeasurements(I64 epochTaiNs);

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  polaris::sitl::SitlHandler handler_;  //!< Byte-level decode/reply logic
  bool connected_ = false;              //!< Emit SitlConnected once, on first message
  bool quiescent_ = false;              //!< True after SHUTDOWN: stop answering
  U64 steps_ = 0;                       //!< Macro steps exchanged (telemetry)

  //! Latest actuator commands from the SITL rate group, indexed by unit build
  //! order. Written by wheelCmdIn/mtqCmdIn during the cycle, read when building
  //! the STEP_REPLY. Fixed-size (kMaxUnits); zero until the rate group commands.
  polaris::sitl::WheelCommandRecord latest_wheel_[polaris::sitl::kMaxUnits] = {};
  polaris::sitl::MtqCommandRecord latest_mtq_[polaris::sitl::kMaxUnits] = {};
  polaris::sitl::ThrusterCommandRecord latest_thruster_[polaris::sitl::kMaxUnits] = {};

  //! Latest attitude estimate (Body<-ECI, scalar-first), echoed on the
  //! STEP_REPLY for diagnosis only — see SitlBridge.fpp and
  //! `sitl::StepReplyHeader`. Not latched across an invalid estimate: an
  //! estimator that has lost the attitude must read as "no estimate", not as
  //! the last good one, or a test would score a stale belief as a current one.
  double latest_est_q_[4] = {1.0, 0.0, 0.0, 0.0};
  bool latest_est_valid_ = false;

  //! §7 MTQ-on window [s] for the next step, latched with the dipoles. Zero
  //! until the rate group commands, which is the rods-off schedule.
  F64 latest_mtq_on_window_s_ = 0.0;

  //! Fixed reply buffer; largest possible STEP_REPLY (no heap). Reused each
  //! step: the framer copies out synchronously before we are re-entered.
  U8 reply_[polaris::sitl::kMaxStepReplyBytes] = {};

  //! Previous step's sim epoch, for the IMU accumulation interval. Zero/unset
  //! on the first step, where no interval can be formed and the increments are
  //! published invalid rather than divided by a guessed dt.
  I64 last_epoch_tai_ns_ = 0;
  bool have_last_epoch_ = false;
};

}  // namespace flight

#endif

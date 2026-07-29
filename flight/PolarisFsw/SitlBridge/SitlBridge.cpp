// ======================================================================
// \title  SitlBridge.cpp
// \brief  Flight end of the SITL lockstep transport (design doc §2.2, §2.4)
// ======================================================================

#include "flight/PolarisFsw/SitlBridge/SitlBridge.hpp"

#include <Os/RawTime.hpp>

namespace flight {

namespace {
// Emit a progress EVR on the first step and every kStepMilestone steps, so a
// long run leaves a bounded event trail rather than one per macro step.
constexpr U64 kStepMilestone = 100;
}  // namespace

// The rate-group command ports carry fixed arrays sized to kMaxUnits; the reply
// buffer indexing below relies on that, so pin it at compile time.
static_assert(WheelTorqueSet::SIZE == polaris::sitl::kMaxUnits,
              "WheelTorqueSet must be sized to sitl::kMaxUnits");
static_assert(MtqDipoleSet::SIZE == polaris::sitl::kMaxUnits,
              "MtqDipoleSet must be sized to sitl::kMaxUnits");

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

SitlBridge ::SitlBridge(const char* const compName) : SitlBridgeComponentBase(compName) {}

SitlBridge ::~SitlBridge() {}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

void SitlBridge ::dataIn_handler(FwIndexType portNum, Fw::Buffer& data,
                                 const ComCfg::FrameContext& context) {
  namespace sitl = polaris::sitl;

  // After SHUTDOWN the bridge goes quiet: release the buffer and answer nothing.
  if (this->quiescent_) {
    this->dataReturnOut_out(0, data, context);
    return;
  }

  const sitl::HandleResult result = this->handler_.handle(
      data.getData(), static_cast<FwSizeType>(data.getSize()), this->reply_, sizeof(this->reply_));

  // Return ownership of the received payload buffer to the deframer. The handler
  // has already copied everything it needs into reply_.
  this->dataReturnOut_out(0, data, context);

  if (!this->connected_) {
    this->connected_ = true;
    this->log_ACTIVITY_HI_SitlConnected();
  }

  switch (result.status) {
    case sitl::HandleStatus::kHelloAck: {
      const sitl::HelloMsg& h = result.hello;
      this->log_ACTIVITY_HI_HelloReceived(h.n_imu, h.n_star_tracker, h.n_sun_sensor,
                                          h.n_magnetometer, h.n_gnss, h.n_wheel, h.n_mtq);
      this->sendReply(result.reply_len, context);
      break;
    }
    case sitl::HandleStatus::kStepReq:
      this->runStepCycle(result, context, static_cast<U32>(data.getSize()));
      break;
    case sitl::HandleStatus::kShutdown:
      this->quiescent_ = true;
      this->log_ACTIVITY_HI_SitlShutdown();
      break;
    case sitl::HandleStatus::kMalformed:
    default:
      this->log_WARNING_HI_MalformedMessage(result.msg_type, static_cast<U32>(data.getSize()));
      break;
  }
}

void SitlBridge ::dataReturnIn_handler(FwIndexType portNum, Fw::Buffer& data,
                                       const ComCfg::FrameContext& context) {
  // The reply buffer is a fixed member (reply_), not pool-allocated, so there is
  // nothing to deallocate — the framer has finished with it by the time we are
  // called (synchronous downlink on the receive task).
}

void SitlBridge ::wheelCmdIn_handler(FwIndexType portNum, const flight::WheelTorqueSet& cmds) {
  // Latch this cycle's wheel torques as wire records (torque mode). Runs inside
  // sitlCycleOut on this same task, so no locking is needed against the reply build.
  for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
    this->latest_wheel_[i].value = cmds[i];
    this->latest_wheel_[i].mode = 0;  // torque
  }
}

void SitlBridge ::mtqCmdIn_handler(FwIndexType portNum, const flight::MtqDipoleSet& cmds) {
  for (U32 i = 0; i < MtqDipoleSet::SIZE; ++i) {
    this->latest_mtq_[i].dipole_am2[0] = cmds[i][0];
    this->latest_mtq_[i].dipole_am2[1] = cmds[i][1];
    this->latest_mtq_[i].dipole_am2[2] = cmds[i][2];
  }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void SitlBridge ::runStepCycle(const polaris::sitl::HandleResult& result,
                               const ComCfg::FrameContext& context, U32 reqBytes) {
  // §2.4 steps 3-4: serve sim time, fire the FSW cycle, then reply from the
  // commands it produced. Order matters — the rate group's command source reads
  // sim time, so publish the epoch before cycling.
  this->timeSetOut_out(0, result.epoch_tai_ns);

  Os::RawTime cycleStart;
  (void)cycleStart.now();
  // Runs the SITL PassiveRateGroup to completion on this task; its members call
  // back into wheelCmdIn/mtqCmdIn, updating latest_wheel_/latest_mtq_.
  this->sitlCycleOut_out(0, cycleStart);

  const FwSizeType reply_len =
      this->handler_.buildStepReply(result.macro_step, this->latest_wheel_, this->latest_mtq_,
                                    this->reply_, sizeof(this->reply_));
  if (reply_len == 0) {
    // Reply would overflow the fixed buffer (never in flight sizing) — treat as
    // a malformed exchange rather than sending a truncated frame.
    this->log_WARNING_HI_MalformedMessage(result.msg_type, reqBytes);
    return;
  }

  this->steps_++;
  this->tlmWrite_MacroStep(this->steps_);
  if (this->steps_ == 1 || (this->steps_ % kStepMilestone) == 0) {
    this->log_ACTIVITY_LO_StepMilestone(result.macro_step);
  }
  this->sendReply(reply_len, context);
}

void SitlBridge ::sendReply(FwSizeType len, const ComCfg::FrameContext& context) {
  // Wrap the fixed reply buffer and hand it to the SITL framer. The framer
  // serializes header + payload + CRC into its own buffer synchronously and
  // returns this one on dataReturnIn.
  Fw::Buffer out(this->reply_, static_cast<Fw::Buffer::SizeType>(len));
  this->dataOut_out(0, out, context);
}

}  // namespace flight

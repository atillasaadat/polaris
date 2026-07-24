// ======================================================================
// \title  SitlBridge.cpp
// \brief  Flight end of the SITL lockstep transport (design doc §2.2, §2.4)
// ======================================================================

#include "flight/PolarisFsw/SitlBridge/SitlBridge.hpp"

namespace flight {

namespace {
// Emit a progress EVR on the first step and every kStepMilestone steps, so a
// long run leaves a bounded event trail rather than one per macro step.
constexpr U64 kStepMilestone = 100;
}  // namespace

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
    case sitl::HandleStatus::kStepReply: {
      this->steps_++;
      this->tlmWrite_MacroStep(this->steps_);
      if (this->steps_ == 1 || (this->steps_ % kStepMilestone) == 0) {
        this->log_ACTIVITY_LO_StepMilestone(result.macro_step);
      }
      this->sendReply(result.reply_len, context);
      break;
    }
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

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void SitlBridge ::sendReply(FwSizeType len, const ComCfg::FrameContext& context) {
  // Wrap the fixed reply buffer and hand it to the SITL framer. The framer
  // serializes header + payload + CRC into its own buffer synchronously and
  // returns this one on dataReturnIn.
  Fw::Buffer out(this->reply_, static_cast<Fw::Buffer::SizeType>(len));
  this->dataOut_out(0, out, context);
}

}  // namespace flight

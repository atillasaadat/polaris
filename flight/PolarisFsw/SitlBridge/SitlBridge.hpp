// ======================================================================
// \title  SitlBridge.hpp
// \brief  Flight end of the SITL lockstep transport (design doc §2.2, §2.4)
//
// Decodes deframed SITL payloads (lib/sitl/wire.hpp) arriving from a dedicated
// SITL comm stack and answers them, per the two-process macro-step barrier. The
// byte-level protocol lives in polaris::sitl::SitlHandler; this component is the
// F´ shell: buffer plumbing, events, and telemetry. Flight rules apply — no heap
// after init, no exceptions, fixed reply buffer, every message validated.
//
// THIS PUSH the bridge answers with zero actuator commands; it is not yet wired
// to the control rate group (next push).
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

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Wrap the fixed reply buffer (first @p len bytes) and send it to the SITL
  //! framer for framing and downlink.
  void sendReply(FwSizeType len, const ComCfg::FrameContext& context);

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  polaris::sitl::SitlHandler handler_;  //!< Byte-level decode/reply logic
  bool connected_ = false;              //!< Emit SitlConnected once, on first message
  bool quiescent_ = false;              //!< True after SHUTDOWN: stop answering
  U64 steps_ = 0;                       //!< Macro steps exchanged (telemetry)

  //! Fixed reply buffer; largest possible STEP_REPLY (no heap). Reused each
  //! step: the framer copies out synchronously before we are re-entered.
  U8 reply_[polaris::sitl::kMaxStepReplyBytes] = {};
};

}  // namespace flight

#endif

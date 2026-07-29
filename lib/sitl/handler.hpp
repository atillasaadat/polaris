#ifndef POLARIS_SITL_HANDLER_HPP
#define POLARIS_SITL_HANDLER_HPP

/// @file
/// @brief FSW-side SITL message handler — the pure decode/reply logic the
/// `SitlBridge` F´ component wraps (design doc §2.2, §2.4).
///
/// This is the flight counterpart of `sim/io/sitl_server.cpp`: the sim
/// serializes and blocks, the FSW decodes here and answers. Factored out of the
/// component so the byte-level protocol is unit-testable without a running
/// topology — `SitlBridge::dataIn_handler` does nothing but hand the deframed
/// payload to `SitlHandler::handle` and frame whatever bytes come back.
///
/// **Flight-safe:** no heap, no exceptions, no recursion, bounded loops.
/// Everything is memcpy over caller-owned fixed buffers (`wire.hpp` PODs). Every
/// message is validated — magic/version/type via `checkHeader`, declared unit
/// counts against `kMaxUnits`, and length against the fixed struct sizes —
/// before any reply is built; a bad message yields `kMalformed` and no reply,
/// never an assert.
///
/// **This push:** the FSW answers autonomously with *zero* actuator commands —
/// every `WheelCommandRecord`/`MtqCommandRecord` is left default (torque mode,
/// 0). Coupling the reply to the control rate group is the NEXT push; until then
/// the reply-record loop below is the seam that GNC output will fill.

#include <cstddef>
#include <cstdint>

#include "sitl/wire.hpp"

namespace polaris::sitl {

/// What `handle` decoded and how the caller should react.
enum class HandleStatus : std::uint8_t {
  kHelloAck,   ///< reply buffer holds a HELLO_ACK; send it
  kStepReply,  ///< reply buffer holds a STEP_REPLY; send it
  kShutdown,   ///< peer asked to stop; no reply, bridge should go quiet
  kMalformed,  ///< unusable message; no reply (caller emits a warning EVR)
};

/// Outcome of decoding one SITL message.
struct HandleResult {
  HandleStatus status = HandleStatus::kMalformed;
  std::size_t reply_len = 0;     ///< bytes written into the reply buffer
  std::uint16_t msg_type = 0;    ///< raw MsgHeader::type seen (for the EVR/telemetry)
  std::uint64_t macro_step = 0;  ///< echoed step (valid for kStepReply)
  HelloMsg hello{};              ///< decoded HELLO (valid for kHelloAck)
};

/// Decodes SITL requests and builds replies. Holds the only cross-message state
/// the barrier needs: the wheel/MTQ counts negotiated at HELLO, which size every
/// later STEP_REPLY (STEP_REQ does not repeat them). One instance per link.
class SitlHandler {
 public:
  /// Decode one message from @p in (@p in_len bytes) and, when a reply is due,
  /// write it into @p out (capacity @p out_cap). Returns the outcome; on
  /// `kHelloAck`/`kStepReply` `reply_len` bytes of @p out are the payload to
  /// frame. Touches nothing it cannot fully validate.
  HandleResult handle(const std::uint8_t* in, std::size_t in_len, std::uint8_t* out,
                      std::size_t out_cap) {
    HandleResult r;
    MsgHeader hdr;
    std::size_t off = 0;
    if (!readRecord(in, in_len, off, hdr)) {
      return r;  // too short even for a header → kMalformed
    }
    r.msg_type = hdr.type;
    if (hdr.magic != kMagic || hdr.version != kVersion) {
      return r;  // kMalformed
    }
    switch (static_cast<MsgType>(hdr.type)) {
      case MsgType::kHello:
        return buildHelloAck(in, in_len, out, out_cap);
      case MsgType::kStepReq:
        return buildStepReply(in, in_len, out, out_cap);
      case MsgType::kShutdown:
        r.status = HandleStatus::kShutdown;
        return r;
      default:
        return r;  // kHelloAck/kStepReply are FSW→sim only; anything else is bad
    }
  }

  bool helloSeen() const { return hello_seen_; }

  std::uint32_t nWheel() const { return n_wheel_; }

  std::uint32_t nMtq() const { return n_mtq_; }

 private:
  /// HELLO → echo the struct back with type flipped to HELLO_ACK, after latching
  /// the reply counts. Rejects counts above the wire maxima.
  HandleResult buildHelloAck(const std::uint8_t* in, std::size_t in_len, std::uint8_t* out,
                             std::size_t out_cap) {
    HandleResult r;
    r.msg_type = static_cast<std::uint16_t>(MsgType::kHello);
    HelloMsg hello;
    std::size_t off = 0;
    if (!readRecord(in, in_len, off, hello)) {
      return r;  // kMalformed
    }
    if (hello.n_imu > kMaxUnits || hello.n_star_tracker > kMaxUnits ||
        hello.n_sun_sensor > kMaxUnits || hello.n_magnetometer > kMaxUnits ||
        hello.n_gnss > kMaxUnits || hello.n_wheel > kMaxUnits || hello.n_mtq > kMaxUnits) {
      return r;  // kMalformed: implausible unit count
    }
    n_wheel_ = hello.n_wheel;
    n_mtq_ = hello.n_mtq;
    hello_seen_ = true;

    HelloMsg ack = hello;  // echo all counts (§2.2 HELLO_ACK contract)
    ack.hdr.type = static_cast<std::uint16_t>(MsgType::kHelloAck);
    std::size_t woff = 0;
    if (!writeRecord(out, out_cap, woff, ack)) {
      return r;  // kMalformed: caller buffer too small (never in flight sizing)
    }
    r.status = HandleStatus::kHelloAck;
    r.reply_len = woff;
    r.hello = hello;
    return r;
  }

  /// STEP_REQ → STEP_REPLY: echo the barrier step, then n_wheel + n_mtq zeroed
  /// command records. Sensor records in the request are intentionally not read
  /// this push (no rate-group coupling yet) — only the header is validated.
  HandleResult buildStepReply(const std::uint8_t* in, std::size_t in_len, std::uint8_t* out,
                              std::size_t out_cap) {
    HandleResult r;
    r.msg_type = static_cast<std::uint16_t>(MsgType::kStepReq);
    StepReqHeader req;
    std::size_t off = 0;
    if (!readRecord(in, in_len, off, req)) {
      return r;  // kMalformed
    }
    if (!hello_seen_) {
      return r;  // kMalformed: STEP before HELLO breaks the handshake order
    }
    r.macro_step = req.macro_step;

    std::size_t woff = 0;
    StepReplyHeader rhdr;
    rhdr.macro_step = req.macro_step;  // barrier echo (§2.4)
    if (!writeRecord(out, out_cap, woff, rhdr)) {
      return r;  // kMalformed
    }
    // NEXT PUSH: replace these zeros with the control loop's actuator commands.
    for (std::uint32_t i = 0; i < n_wheel_; ++i) {
      if (!writeRecord(out, out_cap, woff, WheelCommandRecord{})) {
        return r;  // kMalformed: reply would overflow the caller buffer
      }
    }
    for (std::uint32_t i = 0; i < n_mtq_; ++i) {
      if (!writeRecord(out, out_cap, woff, MtqCommandRecord{})) {
        return r;  // kMalformed
      }
    }
    r.status = HandleStatus::kStepReply;
    r.reply_len = woff;
    return r;
  }

  std::uint32_t n_wheel_ = 0;
  std::uint32_t n_mtq_ = 0;
  bool hello_seen_ = false;
};

}  // namespace polaris::sitl

#endif  // POLARIS_SITL_HANDLER_HPP

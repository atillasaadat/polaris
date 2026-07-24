/// @file Tests for the FSW-side SITL message handler (lib/sitl/handler.hpp).
///
/// Pins the §2.2/§2.4 reply contract the sim's `SitlServer` depends on: HELLO →
/// HELLO_ACK echoing every count, STEP_REQ → STEP_REPLY echoing the barrier step
/// with n_wheel + n_mtq zeroed command records, SHUTDOWN → no reply, and malformed
/// input rejected (never asserted). The byte layout is exercised end-to-end against
/// the same `wire.hpp` structs the sim serializes, so the two sides cannot drift.

#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "sitl/handler.hpp"
#include "sitl/wire.hpp"

namespace ps = polaris::sitl;

namespace {

ps::HelloMsg makeHello(std::uint32_t n_wheel, std::uint32_t n_mtq) {
  ps::HelloMsg h;
  h.n_imu = 1;
  h.n_star_tracker = 1;
  h.n_sun_sensor = 2;
  h.n_magnetometer = 1;
  h.n_gnss = 1;
  h.n_wheel = n_wheel;
  h.n_mtq = n_mtq;
  h.macro_dt_ns = 100000000;  // 10 Hz macro step
  return h;
}

// Drive a handler through the HELLO handshake, asserting the ACK is well-formed.
void doHello(ps::SitlHandler& h, std::uint32_t n_wheel, std::uint32_t n_mtq) {
  const ps::HelloMsg hello = makeHello(n_wheel, n_mtq);
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      out.data(), out.size());
  ASSERT_EQ(r.status, ps::HandleStatus::kHelloAck);
  ASSERT_EQ(r.reply_len, sizeof(ps::HelloMsg));
  ps::HelloMsg ack;
  std::memcpy(&ack, out.data(), sizeof(ack));
  EXPECT_TRUE(ps::checkHeader(ack.hdr, ps::MsgType::kHelloAck));
  EXPECT_EQ(ack.n_wheel, n_wheel);
  EXPECT_EQ(ack.n_mtq, n_mtq);
  EXPECT_EQ(ack.n_imu, hello.n_imu);
  EXPECT_EQ(ack.macro_dt_ns, hello.macro_dt_ns);
}

}  // namespace

TEST(SitlHandler, HelloAckEchoesCounts) {
  RecordProperty("verifies", "REQ-SIM-004");
  ps::SitlHandler h;
  doHello(h, 4, 3);
  EXPECT_TRUE(h.helloSeen());
  EXPECT_EQ(h.nWheel(), 4u);
  EXPECT_EQ(h.nMtq(), 3u);
}

TEST(SitlHandler, StepReplyEchoesStepAndZeroesCommands) {
  ps::SitlHandler h;
  doHello(h, 4, 3);

  ps::StepReqHeader req;
  req.epoch_tai_ns = 123456789;
  req.macro_step = 42;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r =
      h.handle(reinterpret_cast<const std::uint8_t*>(&req), sizeof(req), out.data(), out.size());

  ASSERT_EQ(r.status, ps::HandleStatus::kStepReply);
  EXPECT_EQ(r.macro_step, 42u);
  // Header + 4 wheels + 3 MTQs, all zeroed.
  const std::size_t expected = sizeof(ps::StepReplyHeader) + 4 * sizeof(ps::WheelCommandRecord) +
                               3 * sizeof(ps::MtqCommandRecord);
  ASSERT_EQ(r.reply_len, expected);

  std::size_t off = 0;
  ps::StepReplyHeader rhdr;
  ASSERT_TRUE(ps::readRecord(out.data(), r.reply_len, off, rhdr));
  EXPECT_TRUE(ps::checkHeader(rhdr.hdr, ps::MsgType::kStepReply));
  EXPECT_EQ(rhdr.macro_step, 42u);
  for (int i = 0; i < 4; ++i) {
    ps::WheelCommandRecord w;
    ASSERT_TRUE(ps::readRecord(out.data(), r.reply_len, off, w));
    EXPECT_EQ(w.mode, 0);  // torque
    EXPECT_EQ(w.value, 0.0);
  }
  for (int i = 0; i < 3; ++i) {
    ps::MtqCommandRecord m;
    ASSERT_TRUE(ps::readRecord(out.data(), r.reply_len, off, m));
    EXPECT_EQ(m.dipole_am2[0], 0.0);
    EXPECT_EQ(m.dipole_am2[1], 0.0);
    EXPECT_EQ(m.dipole_am2[2], 0.0);
  }
  EXPECT_EQ(off, r.reply_len);  // reply is exactly full, no trailing bytes
}

TEST(SitlHandler, ShutdownYieldsNoReply) {
  ps::SitlHandler h;
  ps::MsgHeader bye{ps::kMagic, ps::kVersion, static_cast<std::uint16_t>(ps::MsgType::kShutdown)};
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r =
      h.handle(reinterpret_cast<const std::uint8_t*>(&bye), sizeof(bye), out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kShutdown);
  EXPECT_EQ(r.reply_len, 0u);
}

TEST(SitlHandler, StepBeforeHelloIsMalformed) {
  ps::SitlHandler h;
  ps::StepReqHeader req;
  req.macro_step = 1;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r =
      h.handle(reinterpret_cast<const std::uint8_t*>(&req), sizeof(req), out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_EQ(r.reply_len, 0u);
}

TEST(SitlHandler, BadMagicIsMalformed) {
  ps::SitlHandler h;
  ps::HelloMsg hello = makeHello(1, 1);
  hello.hdr.magic = 0xBADBAD00u;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_FALSE(h.helloSeen());
}

TEST(SitlHandler, TooShortForHeaderIsMalformed) {
  ps::SitlHandler h;
  std::array<std::uint8_t, 4> tiny{};  // shorter than MsgHeader
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(tiny.data(), tiny.size(), out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
}

TEST(SitlHandler, OversizeUnitCountRejected) {
  ps::SitlHandler h;
  ps::HelloMsg hello = makeHello(1, 1);
  hello.n_wheel = ps::kMaxUnits + 1;  // implausible
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_FALSE(h.helloSeen());
}

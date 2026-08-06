// Unit tests for the sim-side SITL lockstep endpoint (design doc §2.2, §2.4).
//
// The fake FSW client here speaks the exact wire contract (lib/sitl/wire.hpp)
// over the exact F´ frame format, from a test thread — so these tests pin the
// sim side of the protocol deterministically with no second process. The real
// F´ SitlBridge is exercised by the two-process integration test.
//
// Verifies REQ-SIM-004 (only measurements cross the boundary — the wire
// records carry no TruthState fields).

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

#include "io/sitl_server.hpp"
#include "sitl/wire.hpp"
#include "sitl_test_util.hpp"

namespace {

using polaris::sim::io::FswInputs;
using polaris::sim::io::FswOutputs;
using polaris::sim::io::SitlServer;
namespace sitl = polaris::sitl;
using namespace polaris::sitl_testutil;

/// A fake FSW: connect, ack HELLO, answer @p n_steps STEP_REQs with scripted
/// wheel torques (step index encoded in the command values), then wait for
/// SHUTDOWN.
void fakeFsw(std::uint16_t port, int n_steps, std::vector<sitl::StepReqHeader>* seen) {
  const int fd = connectTo(port);
  ASSERT_GE(fd, 0);

  std::vector<std::uint8_t> payload;
  ASSERT_TRUE(recvFramed(fd, payload));
  sitl::HelloMsg hello;
  std::size_t off = 0;
  ASSERT_TRUE(sitl::readRecord(payload.data(), payload.size(), off, hello));
  ASSERT_TRUE(sitl::checkHeader(hello.hdr, sitl::MsgType::kHello));

  sitl::HelloMsg ack = hello;
  ack.hdr.type = static_cast<std::uint16_t>(sitl::MsgType::kHelloAck);
  ASSERT_TRUE(sendFramed(fd, &ack, sizeof(ack)));

  for (int s = 0; s < n_steps; ++s) {
    ASSERT_TRUE(recvFramed(fd, payload));
    sitl::StepReqHeader req;
    off = 0;
    ASSERT_TRUE(sitl::readRecord(payload.data(), payload.size(), off, req));
    ASSERT_TRUE(sitl::checkHeader(req.hdr, sitl::MsgType::kStepReq));
    seen->push_back(req);

    // Scripted reply: wheel i torque = 0.01*(step+1) + i, MTQ dipole = step.
    std::vector<std::uint8_t> reply(sitl::kMaxStepReplyBytes);
    std::size_t roff = 0;
    sitl::StepReplyHeader rh;
    rh.macro_step = req.macro_step;
    ASSERT_TRUE(sitl::writeRecord(reply.data(), reply.size(), roff, rh));
    for (std::uint32_t w = 0; w < hello.n_wheel; ++w) {
      sitl::WheelCommandRecord rec;
      rec.mode = 0;
      rec.value = 0.01 * (static_cast<double>(req.macro_step) + 1.0) + w;
      ASSERT_TRUE(sitl::writeRecord(reply.data(), reply.size(), roff, rec));
    }
    for (std::uint32_t m = 0; m < hello.n_mtq; ++m) {
      sitl::MtqCommandRecord rec;
      rec.dipole_am2[0] = static_cast<double>(req.macro_step);
      ASSERT_TRUE(sitl::writeRecord(reply.data(), reply.size(), roff, rec));
    }
    ASSERT_TRUE(sendFramed(fd, reply.data(), roff));
  }

  // Expect SHUTDOWN.
  if (recvFramed(fd, payload)) {
    sitl::MsgHeader bye;
    off = 0;
    if (sitl::readRecord(payload.data(), payload.size(), off, bye)) {
      EXPECT_TRUE(sitl::checkHeader(bye, sitl::MsgType::kShutdown));
    }
  }
  ::close(fd);
}

TEST(SitlServer, LockstepExchangeCarriesScriptedCommandsAndBarrierEcho) {
  SitlServer::Counts counts;
  counts.imu = 1;
  counts.magnetometer = 1;
  counts.gnss = 1;
  counts.wheel = 4;
  counts.mtq = 3;
  SitlServer server(counts, 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::vector<sitl::StepReqHeader> seen;
  std::thread fsw(fakeFsw, server.port(), 3, &seen);

  auto cb = server.callback();
  for (std::uint64_t s = 0; s < 3; ++s) {
    const FswOutputs out = cb(makeInputs(s));
    ASSERT_TRUE(server.healthy()) << server.lastError();
    ASSERT_EQ(out.wheels.size(), 4u);
    ASSERT_EQ(out.magnetorquer_dipoles.size(), 3u);
    // The scripted values prove the reply was decoded, not defaulted.
    EXPECT_DOUBLE_EQ(out.wheels[2].value, 0.01 * (static_cast<double>(s) + 1.0) + 2.0);
    EXPECT_DOUBLE_EQ(out.magnetorquer_dipoles[0].eigen()[0], static_cast<double>(s));
  }
  server.stop();
  fsw.join();

  ASSERT_EQ(seen.size(), 3u);
  for (std::uint64_t s = 0; s < 3; ++s) {
    EXPECT_EQ(seen[s].macro_step, s);
    EXPECT_EQ(seen[s].epoch_tai_ns, 1'000'000'000LL * static_cast<std::int64_t>(s));
  }
  EXPECT_EQ(server.stepsExchanged(), 3u);
}

TEST(SitlServer, BarrierMismatchDegradesToOpenLoopNotCrash) {
  SitlServer::Counts counts;
  counts.wheel = 1;
  SitlServer server(counts, 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // HELLO
    sitl::HelloMsg hello;
    std::size_t off = 0;
    ASSERT_TRUE(sitl::readRecord(payload.data(), payload.size(), off, hello));
    hello.hdr.type = static_cast<std::uint16_t>(sitl::MsgType::kHelloAck);
    ASSERT_TRUE(sendFramed(fd, &hello, sizeof(hello)));

    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0
    // Reply with the WRONG step echo: the barrier must reject it.
    std::vector<std::uint8_t> reply(sitl::kMaxStepReplyBytes);
    std::size_t roff = 0;
    sitl::StepReplyHeader rh;
    rh.macro_step = 99;
    ASSERT_TRUE(sitl::writeRecord(reply.data(), reply.size(), roff, rh));
    sitl::WheelCommandRecord rec;
    ASSERT_TRUE(sitl::writeRecord(reply.data(), reply.size(), roff, rec));
    ASSERT_TRUE(sendFramed(fd, reply.data(), roff));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_NE(server.lastError().find("barrier"), std::string::npos) << server.lastError();
  // Degraded output: correct sizes, zero commands.
  ASSERT_EQ(out.wheels.size(), 1u);
  EXPECT_DOUBLE_EQ(out.wheels[0].value, 0.0);
  fsw.join();
  server.stop();
}

TEST(SitlWire, RecordSizesAreTheFrozenLayout) {
  // The wire contract: any change here is a version bump, not a silent edit.
  // Version 2 (Push 56) added WheelTachRecord to the STEP_REQ; every earlier
  // record is unchanged, which is what the sizes below pin.
  EXPECT_EQ(sitl::kVersion, 2);
  EXPECT_EQ(sizeof(sitl::HelloMsg), 48u);
  EXPECT_EQ(sizeof(sitl::ImuRecord), 64u);
  EXPECT_EQ(sizeof(sitl::StarTrackerRecord), 56u);
  EXPECT_EQ(sizeof(sitl::SunSensorRecord), 176u);
  EXPECT_EQ(sizeof(sitl::MagnetometerRecord), 40u);
  EXPECT_EQ(sizeof(sitl::GnssRecord), 104u);
  EXPECT_EQ(sizeof(sitl::WheelCommandRecord), 16u);
  EXPECT_EQ(sizeof(sitl::MtqCommandRecord), 24u);
  EXPECT_EQ(sizeof(sitl::WheelTachRecord), 24u);
}

}  // namespace

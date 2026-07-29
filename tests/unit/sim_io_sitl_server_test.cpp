// Unit tests for the sim-side SITL lockstep endpoint (design doc §2.2, §2.4).
//
// The fake FSW client here speaks the exact wire contract (lib/sitl/wire.hpp)
// over the exact F´ frame format, from a test thread — so these tests pin the
// sim side of the protocol deterministically with no second process. The real
// F´ SitlBridge is exercised by the two-process integration test.
//
// Verifies REQ-SIM-004 (only measurements cross the boundary — the wire
// records carry no TruthState fields).

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include "io/sitl_server.hpp"
#include "sitl/wire.hpp"

namespace {

using polaris::sim::io::FswInputs;
using polaris::sim::io::FswOutputs;
using polaris::sim::io::SitlServer;
namespace sitl = polaris::sitl;

// --- Minimal fake-FSW framing (mirrors Svc::FprimeProtocol) -----------------

std::uint32_t crc32Ref(const std::uint8_t* d, std::size_t n) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= d[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
    }
  }
  return ~crc;
}

void putBe(std::uint8_t* p, std::uint32_t v) {
  p[0] = static_cast<std::uint8_t>(v >> 24);
  p[1] = static_cast<std::uint8_t>(v >> 16);
  p[2] = static_cast<std::uint8_t>(v >> 8);
  p[3] = static_cast<std::uint8_t>(v);
}

std::uint32_t getBe(const std::uint8_t* p) {
  return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) |
         std::uint32_t(p[3]);
}

bool sendAll(int fd, const std::uint8_t* d, std::size_t n) {
  std::size_t off = 0;
  while (off < n) {
    const ssize_t r = ::send(fd, d + off, n - off, MSG_NOSIGNAL);
    if (r <= 0) {
      return false;
    }
    off += static_cast<std::size_t>(r);
  }
  return true;
}

bool sendFramed(int fd, const void* payload, std::size_t len) {
  std::vector<std::uint8_t> f(8 + len + 4);
  putBe(f.data(), 0xDEADBEEFu);
  putBe(f.data() + 4, static_cast<std::uint32_t>(len));
  std::memcpy(f.data() + 8, payload, len);
  putBe(f.data() + 8 + len, crc32Ref(f.data(), 8 + len));
  return sendAll(fd, f.data(), f.size());
}

/// Blocking read of one frame; returns payload bytes.
bool recvFramed(int fd, std::vector<std::uint8_t>& payload) {
  auto recvExact = [fd](std::uint8_t* d, std::size_t n) {
    std::size_t off = 0;
    while (off < n) {
      const ssize_t r = ::recv(fd, d + off, n - off, 0);
      if (r <= 0) {
        return false;
      }
      off += static_cast<std::size_t>(r);
    }
    return true;
  };
  std::uint8_t hdr[8];
  if (!recvExact(hdr, 8) || getBe(hdr) != 0xDEADBEEFu) {
    return false;
  }
  const std::uint32_t len = getBe(hdr + 4);
  payload.resize(len);
  std::uint8_t crc[4];
  if (!recvExact(payload.data(), len) || !recvExact(crc, 4)) {
    return false;
  }
  std::vector<std::uint8_t> whole(8 + len);
  std::memcpy(whole.data(), hdr, 8);
  std::memcpy(whole.data() + 8, payload.data(), len);
  return getBe(crc) == crc32Ref(whole.data(), whole.size());
}

int connectTo(std::uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

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

FswInputs makeInputs(std::uint64_t step) {
  FswInputs in;
  in.epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(1'000'000'000LL *
                                                           static_cast<std::int64_t>(step));
  in.macro_step = step;
  in.imus.resize(1);
  in.imus[0].delta_angle_rad =
      polaris::math::Vec3<polaris::math::frames::Body>(Eigen::Vector3d(1e-3, 2e-3, 3e-3));
  in.imus[0].samples = 25;
  in.magnetometers.resize(1);
  in.gnss.resize(1);
  return in;
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

TEST(SitlWire, RecordSizesAreTheFrozenV1Layout) {
  // The wire contract: any change here is a version bump, not a silent edit.
  EXPECT_EQ(sizeof(sitl::HelloMsg), 48u);
  EXPECT_EQ(sizeof(sitl::ImuRecord), 64u);
  EXPECT_EQ(sizeof(sitl::StarTrackerRecord), 56u);
  EXPECT_EQ(sizeof(sitl::SunSensorRecord), 176u);
  EXPECT_EQ(sizeof(sitl::MagnetometerRecord), 40u);
  EXPECT_EQ(sizeof(sitl::GnssRecord), 104u);
  EXPECT_EQ(sizeof(sitl::WheelCommandRecord), 16u);
  EXPECT_EQ(sizeof(sitl::MtqCommandRecord), 24u);
}

}  // namespace

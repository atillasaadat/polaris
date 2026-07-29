// Adversarial/edge tests for the sim-side SITL lockstep transport (design doc
// §2.2, §2.4). The happy path and the barrier-mismatch degrade live in
// sim_io_sitl_server_test.cpp; here we abuse the wire — corrupt CRCs, garbage
// prefixes, fragmentation, coalescing, silent/dead peers, implausible lengths,
// truncated payloads — and require the server to either continue healthy (when
// the byte stream is recoverable) or degrade to open loop (healthy() false,
// lastError() set) without crashing, hanging, or over-allocating.
//
// ASan/UBSan is on for this target, so any accumulator overrun or bad pointer
// surfaces here directly.

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "io/sitl_server.hpp"
#include "sitl/wire.hpp"
#include "sitl_test_util.hpp"

namespace {

using polaris::sim::io::FswOutputs;
using polaris::sim::io::SitlServer;
namespace sitl = polaris::sitl;
using namespace polaris::sitl_testutil;

/// One-wheel suite shared by these tests; the plant is not under test here, the
/// transport is. (`SitlServer` is non-copyable, so tests construct it inline.)
SitlServer::Counts oneWheel() {
  SitlServer::Counts counts;
  counts.imu = 1;
  counts.magnetometer = 1;
  counts.gnss = 1;
  counts.wheel = 1;
  return counts;
}

// 1. A framed STEP_REPLY whose CRC trailer is corrupted → the server rejects it
//    and degrades; no crash.
TEST(SitlServerFault, CorruptCrcDegradesNotCrash) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0

    // Build a valid frame, then flip a payload byte so the CRC no longer matches.
    const std::vector<std::uint8_t> reply = makeStepReply(hello, 0);
    std::vector<std::uint8_t> f(8 + reply.size() + 4);
    putBe(f.data(), 0xDEADBEEFu);
    putBe(f.data() + 4, static_cast<std::uint32_t>(reply.size()));
    std::memcpy(f.data() + 8, reply.data(), reply.size());
    putBe(f.data() + 8 + reply.size(), crc32Ref(f.data(), 8 + reply.size()));
    f[8] ^= 0xFFu;  // corrupt one payload byte after the CRC was computed
    ASSERT_TRUE(sendAll(fd, f.data(), f.size()));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_NE(server.lastError().find("CRC"), std::string::npos) << server.lastError();
  ASSERT_EQ(out.wheels.size(), 1u);
  EXPECT_DOUBLE_EQ(out.wheels[0].value, 0.0);
  fsw.join();
  server.stop();
}

// 2. Garbage bytes (no start word) ahead of a valid frame → the accumulator
//    resyncs by dropping the junk and the real frame still decodes; run healthy.
TEST(SitlServerFault, GarbagePrefixResyncsAndDecodes) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0

    // 0xA5 never forms the 0xDEADBEEF start word at any alignment, so the
    // server drops all seven junk bytes then locks onto the real frame.
    const std::vector<std::uint8_t> junk(7, 0xA5u);
    ASSERT_TRUE(sendAll(fd, junk.data(), junk.size()));
    const std::vector<std::uint8_t> reply = makeStepReply(hello, 0);
    ASSERT_TRUE(sendFramed(fd, reply.data(), reply.size()));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_TRUE(server.healthy()) << server.lastError();
  ASSERT_EQ(out.wheels.size(), 1u);
  EXPECT_EQ(server.stepsExchanged(), 1u);
  fsw.join();
  server.stop();
}

// 3. A frame delivered as many single-byte sends → the accumulator reassembles
//    it across recv() boundaries; exchange succeeds.
TEST(SitlServerFault, ByteDribbleReassembles) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0

    const std::vector<std::uint8_t> reply = makeStepReply(hello, 0);
    std::vector<std::uint8_t> f(8 + reply.size() + 4);
    putBe(f.data(), 0xDEADBEEFu);
    putBe(f.data() + 4, static_cast<std::uint32_t>(reply.size()));
    std::memcpy(f.data() + 8, reply.data(), reply.size());
    putBe(f.data() + 8 + reply.size(), crc32Ref(f.data(), 8 + reply.size()));
    for (std::uint8_t b : f) {
      ASSERT_TRUE(sendAll(fd, &b, 1));  // one byte per send()
    }
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_TRUE(server.healthy()) << server.lastError();
  EXPECT_EQ(server.stepsExchanged(), 1u);
  ASSERT_EQ(out.wheels.size(), 1u);
  fsw.join();
  server.stop();
}

// 4. HELLO_ACK and the first STEP_REPLY coalesced into a single write → the
//    accumulator yields both frames from one recv(); the leftover after the ACK
//    is decoded on the next barrier read with no further socket traffic.
TEST(SitlServerFault, CoalescedAckAndReplyBothParse) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    // Read HELLO, then write HELLO_ACK + STEP_REPLY(step 0) in one send. The
    // step is deterministic (the loop starts at macro_step 0), so the reply can
    // precede the STEP_REQ that will follow on the wire.
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));
    sitl::HelloMsg hello;
    std::size_t off = 0;
    ASSERT_TRUE(sitl::readRecord(payload.data(), payload.size(), off, hello));
    hello.hdr.type = static_cast<std::uint16_t>(sitl::MsgType::kHelloAck);

    const std::vector<std::uint8_t> reply = makeStepReply(hello, 0);
    std::vector<std::uint8_t> both(8 + sizeof(hello) + 4);
    putBe(both.data(), 0xDEADBEEFu);
    putBe(both.data() + 4, static_cast<std::uint32_t>(sizeof(hello)));
    std::memcpy(both.data() + 8, &hello, sizeof(hello));
    putBe(both.data() + 8 + sizeof(hello), crc32Ref(both.data(), 8 + sizeof(hello)));
    std::vector<std::uint8_t> rf(8 + reply.size() + 4);
    putBe(rf.data(), 0xDEADBEEFu);
    putBe(rf.data() + 4, static_cast<std::uint32_t>(reply.size()));
    std::memcpy(rf.data() + 8, reply.data(), reply.size());
    putBe(rf.data() + 8 + reply.size(), crc32Ref(rf.data(), 8 + reply.size()));
    both.insert(both.end(), rf.begin(), rf.end());
    ASSERT_TRUE(sendAll(fd, both.data(), both.size()));

    ASSERT_TRUE(recvFramed(fd, payload));  // drain the STEP_REQ the server sends
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_TRUE(server.healthy()) << server.lastError();
  EXPECT_EQ(server.stepsExchanged(), 1u);
  ASSERT_EQ(out.wheels.size(), 1u);
  fsw.join();
  server.stop();
}

// 5. Peer closes the socket after HELLO_ACK but before any STEP_REPLY → the
//    barrier read hits EOF and the run degrades to open loop; no hang.
TEST(SitlServerFault, PeerCloseMidRunDegrades) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    ::close(fd);  // vanish before answering the first STEP_REQ
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_FALSE(server.lastError().empty());
  ASSERT_EQ(out.wheels.size(), 1u);
  EXPECT_DOUBLE_EQ(out.wheels[0].value, 0.0);
  fsw.join();
  server.stop();
}

// 6. Peer ACKs HELLO then goes silent → the barrier read times out (short
//    timeout) and the run degrades; the whole test stays well under the bound.
TEST(SitlServerFault, SilentPeerTimesOutAndDegrades) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(1.0);  // 1 s barrier-read timeout
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    recvFramed(fd, payload);  // read the STEP_REQ, then never reply
    // Hold the socket open past the server's timeout so it is a silence, not a
    // close; the server owns the lifetime and joins us after it degrades.
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_FALSE(server.lastError().empty());
  ASSERT_EQ(out.wheels.size(), 1u);
  fsw.join();
  server.stop();
}

// 7. A frame advertising an implausibly huge length → the server treats the
//    start word as a chance collision, drops a byte, and resyncs; it never
//    tries to allocate the advertised size, and degrades cleanly at EOF.
TEST(SitlServerFault, ImplausibleLengthDoesNotAllocateOrCrash) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0

    std::uint8_t hdr[10];
    putBe(hdr, 0xDEADBEEFu);
    putBe(hdr + 4, 0xFFFFFFFFu);  // ~4 GiB payload claim
    hdr[8] = 0xA5u;
    hdr[9] = 0xA5u;
    ASSERT_TRUE(sendAll(fd, hdr, sizeof(hdr)));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_FALSE(server.lastError().empty());
  ASSERT_EQ(out.wheels.size(), 1u);
  fsw.join();
  server.stop();
}

// 8. A well-framed STEP_REPLY that is too short for the declared command records
//    (header only, no wheel record) → the barrier read decodes the header, runs
//    out of bytes for the wheel, and degrades with zero commands out.
TEST(SitlServerFault, TruncatedReplyPayloadDegrades) {
  SitlServer server(oneWheel(), 100'000'000LL);
  server.setTimeoutS(5.0);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  std::thread fsw([port = server.port()] {
    const int fd = connectTo(port);
    ASSERT_GE(fd, 0);
    sitl::HelloMsg hello;
    ASSERT_TRUE(ackHello(fd, hello));
    std::vector<std::uint8_t> payload;
    ASSERT_TRUE(recvFramed(fd, payload));  // STEP_REQ 0

    // Header only: barrier step echoes, but the promised wheel record is absent.
    sitl::StepReplyHeader rh;
    rh.macro_step = 0;
    ASSERT_TRUE(sendFramed(fd, &rh, sizeof(rh)));
    ::close(fd);
  });

  const FswOutputs out = server.callback()(makeInputs(0));
  EXPECT_FALSE(server.healthy());
  EXPECT_NE(server.lastError().find("short"), std::string::npos) << server.lastError();
  ASSERT_EQ(out.wheels.size(), 1u);
  EXPECT_DOUBLE_EQ(out.wheels[0].value, 0.0);
  fsw.join();
  server.stop();
}

}  // namespace

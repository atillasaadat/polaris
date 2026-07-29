#ifndef POLARIS_TESTS_SITL_TEST_UTIL_HPP
#define POLARIS_TESTS_SITL_TEST_UTIL_HPP

/// @file Shared fake-FSW helpers for the sim-side SITL server tests.
///
/// These let a test thread speak the exact §2.2 wire contract (lib/sitl/wire.hpp)
/// inside the exact F´ frame format (`Svc::FprimeProtocol`: big-endian start word
/// 0xdeadbeef, length, payload, CRC-32), so both the happy-path suite
/// (sim_io_sitl_server_test.cpp) and the adversarial fault suite
/// (sim_io_sitl_server_fault_test.cpp) drive the real `SitlServer` with no second
/// process and without duplicating the framing.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <Eigen/Core>
#include <vector>

#include "io/sitl_server.hpp"
#include "sitl/wire.hpp"

namespace polaris::sitl_testutil {

namespace sitl = polaris::sitl;

/// CRC-32/ISO-HDLC, matching the sim's `crc32` (and F´ `Utils::Hash`).
inline std::uint32_t crc32Ref(const std::uint8_t* d, std::size_t n) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= d[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
    }
  }
  return ~crc;
}

inline void putBe(std::uint8_t* p, std::uint32_t v) {
  p[0] = static_cast<std::uint8_t>(v >> 24);
  p[1] = static_cast<std::uint8_t>(v >> 16);
  p[2] = static_cast<std::uint8_t>(v >> 8);
  p[3] = static_cast<std::uint8_t>(v);
}

inline std::uint32_t getBe(const std::uint8_t* p) {
  return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) |
         std::uint32_t(p[3]);
}

inline bool sendAll(int fd, const std::uint8_t* d, std::size_t n) {
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

/// Frame @p payload the way the F´ SitlBridge would and send it whole.
inline bool sendFramed(int fd, const void* payload, std::size_t len) {
  std::vector<std::uint8_t> f(8 + len + 4);
  putBe(f.data(), 0xDEADBEEFu);
  putBe(f.data() + 4, static_cast<std::uint32_t>(len));
  std::memcpy(f.data() + 8, payload, len);
  putBe(f.data() + 8 + len, crc32Ref(f.data(), 8 + len));
  return sendAll(fd, f.data(), f.size());
}

/// Blocking read of one frame; returns payload bytes. False on EOF or bad CRC.
inline bool recvFramed(int fd, std::vector<std::uint8_t>& payload) {
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

inline int connectTo(std::uint16_t port) {
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

/// Receive the server's HELLO and answer HELLO_ACK, returning the decoded HELLO
/// (with its type already flipped to HELLO_ACK). False on any framing error.
inline bool ackHello(int fd, sitl::HelloMsg& hello) {
  std::vector<std::uint8_t> payload;
  if (!recvFramed(fd, payload)) {
    return false;
  }
  std::size_t off = 0;
  if (!sitl::readRecord(payload.data(), payload.size(), off, hello) ||
      !sitl::checkHeader(hello.hdr, sitl::MsgType::kHello)) {
    return false;
  }
  hello.hdr.type = static_cast<std::uint16_t>(sitl::MsgType::kHelloAck);
  return sendFramed(fd, &hello, sizeof(hello));
}

/// Build a well-formed STEP_REPLY payload (header + n_wheel + n_mtq zeroed
/// command records) echoing @p step, for the counts declared in @p hello.
inline std::vector<std::uint8_t> makeStepReply(const sitl::HelloMsg& hello, std::uint64_t step) {
  std::vector<std::uint8_t> reply(sitl::kMaxStepReplyBytes);
  std::size_t roff = 0;
  sitl::StepReplyHeader rh;
  rh.macro_step = step;
  sitl::writeRecord(reply.data(), reply.size(), roff, rh);
  for (std::uint32_t w = 0; w < hello.n_wheel; ++w) {
    sitl::writeRecord(reply.data(), reply.size(), roff, sitl::WheelCommandRecord{});
  }
  for (std::uint32_t m = 0; m < hello.n_mtq; ++m) {
    sitl::writeRecord(reply.data(), reply.size(), roff, sitl::MtqCommandRecord{});
  }
  reply.resize(roff);
  return reply;
}

/// One deterministic macro-step's worth of sensor inputs.
inline polaris::sim::io::FswInputs makeInputs(std::uint64_t step) {
  polaris::sim::io::FswInputs in;
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

}  // namespace polaris::sitl_testutil

#endif  // POLARIS_TESTS_SITL_TEST_UTIL_HPP

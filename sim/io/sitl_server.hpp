#ifndef POLARIS_SIM_IO_SITL_SERVER_HPP
#define POLARIS_SIM_IO_SITL_SERVER_HPP

/// @file
/// @brief Sim-side SITL lockstep endpoint: an `FswCallback` backed by a live
/// F´ process over TCP (design doc §2.2, §2.4 — the two-process barrier).
///
/// The sim is the master clock, so it owns the socket: `SitlServer` listens,
/// the FSW's `Drv::TcpClient` connects. At each macro boundary the closed
/// loop's callback serializes `FswInputs` into the §2.2 wire payloads
/// (`lib/sitl/wire.hpp`), wraps them in a standard F´ frame, sends, and
/// **blocks** until the FSW's STEP_REPLY arrives — that blocking read *is* the
/// §2.4 barrier. Determinism is untouched: no wall-clock reads; the reply
/// content, not its timing, feeds the plant.
///
/// Framing is the F´ comms protocol byte-for-byte (`Svc::FprimeProtocol`):
/// big-endian start word `0xdeadbeef`, big-endian payload length, payload,
/// big-endian CRC-32 over header+payload — the same CRC-32/ISO-HDLC that
/// F´'s `Utils::Hash` computes (init `0xFFFFFFFF`, reflected poly, final
/// complement), so `Svc::FprimeDeframer` on the flight side validates these
/// frames unmodified.
///
/// Sim-side code: heap/std OK. Sockets are POSIX (Linux SITL only, like the
/// F´ native target).
///
/// Implements the §2.4 macro-step handshake, sim side (REQ-SIM-004 boundary:
/// only measurements cross).

#include <cstdint>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"

namespace polaris::sim::io {

/// The sim end of the plant↔FSW TCP lockstep. Construct, `start()` to listen,
/// then hand `callback()` to `ClosedLoop::run` — the first STEP blocks until
/// the FSW process has connected and answered HELLO.
///
/// Lifecycle: `start(port)` → (FSW connects; HELLO/HELLO_ACK exchange happens
/// lazily on the first callback) → N×STEP_REQ/STEP_REPLY → `stop()` sends
/// SHUTDOWN and closes. Errors (peer gone, malformed frame, barrier
/// step-echo mismatch, `timeoutS()` elapsing on accept or a barrier read)
/// surface as an invalid `FswOutputs` (zero commands) plus `lastError()`, so a
/// truth run degrades to open loop rather than crashing — and the run is
/// marked unusable for comparison via `healthy()`.
///
/// **One connection per run.** The lockstep contract is a single FSW process
/// for the lifetime of a run: the server accepts exactly once and never
/// re-accepts or re-runs HELLO. Losing the peer mid-run is a failed run
/// (degrade + `healthy()` false), not a reconnectable condition — a
/// reconnected FSW would have lost its state, so its outputs could not be
/// trusted anyway (§2.4 bit-reproducibility).
class SitlServer {
 public:
  /// @param counts  Per-type unit counts, in vehicle build order — the HELLO
  ///                contract. Wheel/MTQ counts size the expected replies.
  struct Counts {
    std::uint32_t imu = 0;
    std::uint32_t star_tracker = 0;
    std::uint32_t sun_sensor = 0;
    std::uint32_t magnetometer = 0;
    std::uint32_t gnss = 0;
    std::uint32_t wheel = 0;
    std::uint32_t mtq = 0;
    std::uint32_t thruster = 0;
  };

  SitlServer(const Counts& counts, std::int64_t macro_dt_ns);
  ~SitlServer();

  SitlServer(const SitlServer&) = delete;
  SitlServer& operator=(const SitlServer&) = delete;

  /// Bind + listen on 127.0.0.1:@p port (loopback only — §2.2 SITL scope).
  /// Port 0 lets the OS choose; read it back with `port()`. Returns false with
  /// `lastError()` set on failure.
  bool start(std::uint16_t port);

  /// The bound port (valid after `start`).
  std::uint16_t port() const { return port_; }

  /// Send SHUTDOWN (if a peer is connected) and close all sockets.
  void stop();

  /// The `FswCallback` to pass to `ClosedLoop::run`. The returned callable
  /// blocks on the FSW each macro step (the §2.4 barrier).
  FswCallback callback();

  /// False after any protocol/socket failure; the trace since is open-loop.
  bool healthy() const { return healthy_; }

  /// Bound on the accept wait and every barrier read [s] (default 30). A dead
  /// FSW then fails the run instead of hanging it (and CI).
  void setTimeoutS(double s) { timeout_s_ = s; }

  double timeoutS() const { return timeout_s_; }

  const std::string& lastError() const { return last_error_; }

  /// Macro steps successfully exchanged.
  std::uint64_t stepsExchanged() const { return steps_; }

 private:
  /// Block until a client is connected and HELLO/HELLO_ACK has run.
  bool ensurePeer();

  /// One barrier exchange. Returns false (and marks unhealthy) on any failure.
  bool exchange(const FswInputs& in, FswOutputs& out);

  /// Frame @p payload per Svc::FprimeProtocol and send it whole.
  bool sendFrame(const std::uint8_t* payload, std::size_t len);

  /// Block until one complete, CRC-valid frame arrives; payload → @p out.
  bool recvFrame(std::vector<std::uint8_t>& out);

  Counts counts_;
  std::int64_t macro_dt_ns_ = 0;
  double timeout_s_ = 30.0;
  int listen_fd_ = -1;
  int peer_fd_ = -1;
  std::uint16_t port_ = 0;
  bool hello_done_ = false;
  bool healthy_ = true;
  std::uint64_t steps_ = 0;
  std::string last_error_;
  std::vector<std::uint8_t> rx_;  ///< accumulated unparsed bytes
};

}  // namespace polaris::sim::io

#endif  // POLARIS_SIM_IO_SITL_SERVER_HPP

#include "io/sitl_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cstring>

#include "sitl/wire.hpp"

namespace polaris::sim::io {

namespace {

/// CRC-32/ISO-HDLC — bit-identical to F´ `Utils::Hash` (libcrc `update_crc_32`
/// with init 0xFFFFFFFF and final one's complement). Table built once.
std::uint32_t crc32(const std::uint8_t* data, std::size_t len) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFFu];
  }
  return ~crc;
}

void putU32BigEndian(std::uint8_t* p, std::uint32_t v) {
  p[0] = static_cast<std::uint8_t>(v >> 24);
  p[1] = static_cast<std::uint8_t>(v >> 16);
  p[2] = static_cast<std::uint8_t>(v >> 8);
  p[3] = static_cast<std::uint8_t>(v);
}

std::uint32_t getU32BigEndian(const std::uint8_t* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
         (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

constexpr std::uint32_t kStartWord = 0xDEADBEEFu;
constexpr std::size_t kHeaderBytes = 8;   // start word + length
constexpr std::size_t kTrailerBytes = 4;  // CRC32

/// Fill one wire IMU record from the loop's accumulation.
sitl::ImuRecord toWire(const ImuAccumulation& a) {
  sitl::ImuRecord r;
  for (int i = 0; i < 3; ++i) {
    r.delta_angle_rad[i] = a.delta_angle_rad.eigen()[i];
    r.delta_velocity_mps[i] = a.delta_velocity_mps.eigen()[i];
  }
  r.time_tag_tai_ns = a.time_tag.nanosecondsSinceEpoch();
  r.samples = a.samples;
  r.valid = a.valid ? 1 : 0;
  return r;
}

sitl::StarTrackerRecord toWire(const Latest<sensors::StarTrackerMeasurement>& l) {
  sitl::StarTrackerRecord r;
  const auto& m = l.measurement;
  const Eigen::Vector4d q = m.attitude.core().coeffs();
  for (int i = 0; i < 4; ++i) {
    r.q_body_eci[i] = q[i];
  }
  // The per-body FOV fractions in m.occlusion are truth-derived and stay
  // sim-side (§2.3); only the coarse occluder status crosses.
  r.acquisition_elapsed_s = m.acquisition_elapsed_s;
  r.time_tag_tai_ns = m.time_tag.nanosecondsSinceEpoch();
  r.valid = m.valid ? 1 : 0;
  r.mode = static_cast<std::uint8_t>(m.mode);
  r.occluder = static_cast<std::uint8_t>(m.occlusion.occluder);
  r.rate_limited = m.rate_limited ? 1 : 0;
  r.accel_limited = m.accel_limited ? 1 : 0;
  r.ever_sampled = l.ever_sampled ? 1 : 0;
  return r;
}

sitl::SunSensorRecord toWire(const Latest<sensors::SunSensorMeasurement>& l) {
  sitl::SunSensorRecord r;
  const auto& m = l.measurement;
  const std::size_t n = std::min<std::size_t>(m.counts.size(), sitl::kMaxDiodes);
  for (std::size_t i = 0; i < n; ++i) {
    r.counts[i] = m.counts[i];
  }
  r.n_counts = static_cast<std::uint32_t>(n);
  for (int i = 0; i < 3; ++i) {
    r.sun_dir_body[i] = m.sun_dir_body.eigen()[i];
  }
  r.accuracy_sigma_rad = m.accuracy_sigma_rad;
  r.time_tag_tai_ns = m.time_tag.nanosecondsSinceEpoch();
  r.fresh = m.fresh ? 1 : 0;
  r.sun_present = m.sun_present ? 1 : 0;
  r.valid = m.valid ? 1 : 0;
  r.ever_sampled = l.ever_sampled ? 1 : 0;
  return r;
}

sitl::MagnetometerRecord toWire(const Latest<sensors::MagnetometerMeasurement>& l) {
  sitl::MagnetometerRecord r;
  for (int i = 0; i < 3; ++i) {
    r.field_tesla[i] = l.measurement.field_tesla.eigen()[i];
  }
  r.time_tag_tai_ns = l.measurement.time_tag.nanosecondsSinceEpoch();
  r.valid = l.measurement.valid ? 1 : 0;
  r.ever_sampled = l.ever_sampled ? 1 : 0;
  return r;
}

sitl::WheelTachRecord toWire(const WheelTelemetry& w) {
  sitl::WheelTachRecord r;
  r.speed_rad_s = w.speed_rad_s;
  r.time_tag_tai_ns = w.time_tag.nanosecondsSinceEpoch();
  r.valid = w.valid ? 1 : 0;
  return r;
}

sitl::GnssRecord toWire(const Latest<sensors::GnssMeasurement>& l) {
  sitl::GnssRecord r;
  const auto& m = l.measurement;
  for (int i = 0; i < 3; ++i) {
    r.position_ecef_m[i] = m.position_m.eigen()[i];
    r.velocity_ecef_mps[i] = m.velocity_m_s.eigen()[i];
  }
  r.clock_bias_s = m.clock_bias_s;
  r.position_sigma_h_m = m.position_sigma_h_m;
  r.position_sigma_v_m = m.position_sigma_v_m;
  r.velocity_sigma_mps = m.velocity_sigma_m_s;
  r.time_sigma_s = m.time_sigma_s;
  r.time_tag_gps_ns = m.time_tag.nanosecondsSinceEpoch();
  r.fresh = m.fresh ? 1 : 0;
  r.valid = m.valid ? 1 : 0;
  r.jammed = m.jammed ? 1 : 0;
  r.ever_sampled = l.ever_sampled ? 1 : 0;
  return r;
}

}  // namespace

SitlServer::SitlServer(const Counts& counts, std::int64_t macro_dt_ns)
    : counts_(counts), macro_dt_ns_(macro_dt_ns) {}

SitlServer::~SitlServer() {
  stop();
}

bool SitlServer::start(std::uint16_t port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    last_error_ = "socket() failed";
    return false;
  }
  const int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // loopback only (§2.2)
  addr.sin_port = htons(port);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    last_error_ = "bind() failed";
    stop();
    return false;
  }
  socklen_t alen = sizeof(addr);
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &alen);
  port_ = ntohs(addr.sin_port);
  if (::listen(listen_fd_, 1) != 0) {
    last_error_ = "listen() failed";
    stop();
    return false;
  }
  return true;
}

void SitlServer::stop() {
  if (peer_fd_ >= 0) {
    sitl::MsgHeader bye{sitl::kMagic, sitl::kVersion,
                        static_cast<std::uint16_t>(sitl::MsgType::kShutdown)};
    sendFrame(reinterpret_cast<const std::uint8_t*>(&bye), sizeof(bye));
    ::close(peer_fd_);
    peer_fd_ = -1;
  }
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  hello_done_ = false;
}

bool SitlServer::ensurePeer() {
  if (peer_fd_ < 0) {
    // Bounded wait for the FSW process (its TcpClient retries until we are
    // up, so the only reason to wait long is a process that never launched —
    // a hung CI run is worse than a failed one).
    pollfd pfd{listen_fd_, POLLIN, 0};
    const int ready = ::poll(&pfd, 1, static_cast<int>(timeout_s_ * 1000.0));
    if (ready <= 0) {
      last_error_ = "timed out waiting for the FSW process to connect";
      return false;
    }
    peer_fd_ = ::accept(listen_fd_, nullptr, nullptr);
    if (peer_fd_ < 0) {
      last_error_ = "accept() failed";
      return false;
    }
    const int one = 1;
    ::setsockopt(peer_fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    // Bound every barrier read the same way: a dead FSW degrades the run to
    // open loop (healthy() false) instead of hanging it.
    timeval tv{};
    tv.tv_sec = static_cast<time_t>(timeout_s_);
    tv.tv_usec = static_cast<suseconds_t>((timeout_s_ - static_cast<double>(tv.tv_sec)) * 1e6);
    ::setsockopt(peer_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  if (!hello_done_) {
    sitl::HelloMsg hello;
    hello.n_imu = counts_.imu;
    hello.n_star_tracker = counts_.star_tracker;
    hello.n_sun_sensor = counts_.sun_sensor;
    hello.n_magnetometer = counts_.magnetometer;
    hello.n_gnss = counts_.gnss;
    hello.n_wheel = counts_.wheel;
    hello.n_mtq = counts_.mtq;
    hello.n_thruster = counts_.thruster;
    hello.macro_dt_ns = macro_dt_ns_;
    if (!sendFrame(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello))) {
      return false;
    }
    std::vector<std::uint8_t> reply;
    if (!recvFrame(reply) || reply.size() < sizeof(sitl::HelloMsg)) {
      last_error_ = "HELLO_ACK missing or short";
      return false;
    }
    sitl::HelloMsg ack;
    std::size_t off = 0;
    if (!sitl::readRecord(reply.data(), reply.size(), off, ack) ||
        !sitl::checkHeader(ack.hdr, sitl::MsgType::kHelloAck) || ack.n_wheel != counts_.wheel ||
        ack.n_mtq != counts_.mtq || ack.n_thruster != counts_.thruster) {
      last_error_ = "HELLO_ACK invalid";
      return false;
    }
    hello_done_ = true;
  }
  return true;
}

bool SitlServer::exchange(const FswInputs& in, FswOutputs& out) {
  if (!ensurePeer()) {
    return false;
  }

  // Serialize STEP_REQ: fixed prefix, then records in HELLO order and counts.
  std::vector<std::uint8_t> buf(sitl::kMaxStepReqBytes);
  std::size_t off = 0;
  sitl::StepReqHeader hdr;
  hdr.epoch_tai_ns = in.epoch.nanosecondsSinceEpoch();
  hdr.macro_step = in.macro_step;
  bool ok = sitl::writeRecord(buf.data(), buf.size(), off, hdr);
  for (const auto& u : in.imus) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  for (const auto& u : in.star_trackers) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  for (const auto& u : in.sun_sensors) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  for (const auto& u : in.magnetometers) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  for (const auto& u : in.gnss) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  for (const auto& u : in.wheels) {
    ok = ok && sitl::writeRecord(buf.data(), buf.size(), off, toWire(u));
  }
  if (!ok) {
    last_error_ = "STEP_REQ overflow (unit counts exceed wire maxima?)";
    return false;
  }
  if (!sendFrame(buf.data(), off)) {
    return false;
  }

  // The barrier: block for STEP_REPLY.
  std::vector<std::uint8_t> reply;
  if (!recvFrame(reply)) {
    return false;
  }
  std::size_t roff = 0;
  sitl::StepReplyHeader rhdr;
  if (!sitl::readRecord(reply.data(), reply.size(), roff, rhdr) ||
      !sitl::checkHeader(rhdr.hdr, sitl::MsgType::kStepReply)) {
    last_error_ = "STEP_REPLY invalid header";
    return false;
  }
  if (rhdr.macro_step != in.macro_step) {
    last_error_ = "STEP_REPLY barrier mismatch (echoed step != sent step)";
    return false;
  }
  out.wheels.resize(counts_.wheel);
  for (auto& w : out.wheels) {
    sitl::WheelCommandRecord rec;
    if (!sitl::readRecord(reply.data(), reply.size(), roff, rec)) {
      last_error_ = "STEP_REPLY short (wheels)";
      return false;
    }
    w.mode = rec.mode == 1 ? WheelCommand::Mode::kSpeed : WheelCommand::Mode::kTorque;
    w.value = rec.value;
  }
  out.magnetorquer_dipoles.resize(counts_.mtq);
  for (auto& d : out.magnetorquer_dipoles) {
    sitl::MtqCommandRecord rec;
    if (!sitl::readRecord(reply.data(), reply.size(), roff, rec)) {
      last_error_ = "STEP_REPLY short (mtq)";
      return false;
    }
    d = math::Vec3<math::frames::Body>(
        Eigen::Vector3d(rec.dipole_am2[0], rec.dipole_am2[1], rec.dipole_am2[2]));
  }
  out.thruster_throttles.resize(counts_.thruster);
  for (double& u : out.thruster_throttles) {
    sitl::ThrusterCommandRecord rec;
    if (!sitl::readRecord(reply.data(), reply.size(), roff, rec)) {
      last_error_ = "STEP_REPLY short (thrusters)";
      return false;
    }
    u = rec.throttle;
  }
  // The §7 duty-cycle on-window for the interval these commands apply over. The
  // loop clamps it to the macro step; a value the FSW never set arrives as zero,
  // which is the rods-off schedule.
  out.mtq_on_window_s = rhdr.mtq_on_window_s;
  ++steps_;
  return true;
}

FswCallback SitlServer::callback() {
  return [this](const FswInputs& in) {
    FswOutputs out;
    if (healthy_ && !exchange(in, out)) {
      healthy_ = false;
      // Degrade to open loop: zero commands, run flagged via healthy().
      out.wheels.assign(counts_.wheel, WheelCommand{});
      out.magnetorquer_dipoles.assign(counts_.mtq, math::Vec3<math::frames::Body>{});
      out.mtq_on_window_s = 0.0;
    }
    return out;
  };
}

bool SitlServer::sendFrame(const std::uint8_t* payload, std::size_t len) {
  std::vector<std::uint8_t> frame(kHeaderBytes + len + kTrailerBytes);
  putU32BigEndian(frame.data(), kStartWord);
  putU32BigEndian(frame.data() + 4, static_cast<std::uint32_t>(len));
  std::memcpy(frame.data() + kHeaderBytes, payload, len);
  putU32BigEndian(frame.data() + kHeaderBytes + len, crc32(frame.data(), kHeaderBytes + len));

  std::size_t sent = 0;
  while (sent < frame.size()) {
    const ssize_t n = ::send(peer_fd_, frame.data() + sent, frame.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      last_error_ = "send() failed (peer gone?)";
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool SitlServer::recvFrame(std::vector<std::uint8_t>& out) {
  std::array<std::uint8_t, 4096> chunk;
  for (;;) {
    // Parse any complete frame already accumulated.
    while (rx_.size() >= kHeaderBytes) {
      if (getU32BigEndian(rx_.data()) != kStartWord) {
        // Resync: drop one byte. F´'s FrameAccumulator does the same scan.
        rx_.erase(rx_.begin());
        continue;
      }
      const std::size_t len = getU32BigEndian(rx_.data() + 4);
      if (len > sitl::kMaxStepReplyBytes && len > sizeof(sitl::HelloMsg)) {
        // A chance 0xDEADBEEF in the stream: resync by dropping one byte,
        // exactly as the start-word scan above does (and as F´'s
        // FrameAccumulator does) — never abort on a resynchronizable state.
        rx_.erase(rx_.begin());
        continue;
      }
      const std::size_t total = kHeaderBytes + len + kTrailerBytes;
      if (rx_.size() < total) {
        break;  // need more bytes
      }
      const std::uint32_t want = getU32BigEndian(rx_.data() + kHeaderBytes + len);
      if (crc32(rx_.data(), kHeaderBytes + len) != want) {
        last_error_ = "frame CRC mismatch";
        return false;
      }
      out.assign(rx_.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes),
                 rx_.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes + len));
      rx_.erase(rx_.begin(), rx_.begin() + static_cast<std::ptrdiff_t>(total));
      return true;
    }
    const ssize_t n = ::recv(peer_fd_, chunk.data(), chunk.size(), 0);
    if (n <= 0) {
      last_error_ = "recv() failed or peer closed";
      return false;
    }
    rx_.insert(rx_.end(), chunk.data(), chunk.data() + n);
  }
}

}  // namespace polaris::sim::io

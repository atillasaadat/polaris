#ifndef POLARIS_SITL_HANDLER_HPP
#define POLARIS_SITL_HANDLER_HPP

/// @file
/// @brief FSW-side SITL message handler — the pure decode/reply logic the
/// `SitlBridge` F´ component wraps (design doc §2.2, §2.4).
///
/// This is the flight counterpart of `sim/io/sitl_server.cpp`: the sim
/// serializes and blocks, the FSW decodes here and answers. Factored out of the
/// component so the byte-level protocol is unit-testable without a running
/// topology — `SitlBridge::dataIn_handler` owns the transport-side state
/// (quiescing, the connected latch, its EVRs, and driving the rate group for
/// STEP) and hands the deframed payload here for the protocol itself.
///
/// **Flight-safe:** no heap, no exceptions, no recursion, bounded loops.
/// Everything is memcpy over caller-owned fixed buffers (`wire.hpp` PODs). Every
/// message is validated — magic/version/type via `checkHeader`, declared unit
/// counts against `kMaxUnits`, and length against the fixed struct sizes —
/// before any reply is built; a bad message yields `kMalformed` and no reply,
/// never an assert.
///
/// **STEP is two-phase (Push 35).** A STEP_REQ's reply carries the actuator
/// commands the FSW's 10 Hz rate group produces *for this step* (§2.4 steps
/// 3-4), so decoding and replying straddle the rate-group cycle. `handle`
/// therefore only decodes a STEP_REQ — it returns `kStepReq` with the barrier
/// step and the sim epoch and writes no reply. The caller drives the rate group
/// (setting sim time from that epoch first), then calls `buildStepReply` with
/// the latched per-unit commands. HELLO/SHUTDOWN still resolve entirely inside
/// `handle`. This keeps the byte protocol here and the topology plumbing in the
/// `SitlBridge` component, and stays unit-testable without a running topology.

#include <cstddef>
#include <cstdint>

#include "sitl/wire.hpp"

namespace polaris::sitl {

/// What `handle` decoded and how the caller should react.
enum class HandleStatus : std::uint8_t {
  kHelloAck,   ///< reply buffer holds a HELLO_ACK; send it
  kStepReq,    ///< a valid STEP_REQ was decoded; run the rate group, then
               ///< `buildStepReply` to produce the STEP_REPLY (no reply yet)
  kShutdown,   ///< peer asked to stop; no reply, bridge should go quiet
  kMalformed,  ///< unusable message; no reply (caller emits a warning EVR)
};

/// Outcome of decoding one SITL message.
struct HandleResult {
  HandleStatus status = HandleStatus::kMalformed;
  std::size_t reply_len = 0;      ///< bytes written into the reply buffer (kHelloAck)
  std::uint16_t msg_type = 0;     ///< raw MsgHeader::type seen (for the EVR/telemetry)
  std::uint64_t macro_step = 0;   ///< barrier step to echo (valid for kStepReq)
  std::int64_t epoch_tai_ns = 0;  ///< macro-step sim epoch (valid for kStepReq)
  HelloMsg hello{};               ///< decoded HELLO (valid for kHelloAck)
};

/// Decodes SITL requests and builds replies. Holds the cross-message state the
/// barrier needs: the per-type unit counts negotiated at HELLO, which size every
/// later STEP exchange (neither STEP message repeats them), and the sensor
/// records of the most recently decoded STEP_REQ, which the caller publishes to
/// the GNC components. One instance per link; ~3.7 kB by value, no heap.
class SitlHandler {
 public:
  /// Decode one message from @p in (@p in_len bytes). For HELLO the HELLO_ACK is
  /// written into @p out (capacity @p out_cap) and `kHelloAck`/`reply_len` are
  /// returned. For STEP_REQ nothing is written to @p out: `kStepReq` is returned
  /// with the barrier `macro_step` and sim `epoch_tai_ns`, and the caller must
  /// then call `buildStepReply` after running the rate group. SHUTDOWN yields
  /// `kShutdown`. Touches nothing it cannot fully validate.
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
        return decodeStepReq(in, in_len);
      case MsgType::kShutdown:
        r.status = HandleStatus::kShutdown;
        return r;
      default:
        return r;  // kHelloAck/kStepReply are FSW→sim only; anything else is bad
    }
  }

  /// Build the STEP_REPLY for barrier step @p macro_step into @p out (capacity
  /// @p out_cap): the reply header then `nWheel()` wheel records and `nMtq()`
  /// MTQ records. Command values come from the caller — @p wheels points to at
  /// least `nWheel()` `WheelCommandRecord`s (the rate group's latched wheel
  /// commands) and @p mtqs to at least `nMtq()` `MtqCommandRecord`s; passing
  /// `nullptr` for either fills that section with default (zero) records.
  /// @p mtq_on_window_s is the §7 duty-cycle on-window for the next step [s],
  /// which the plant needs to apply the rods over the right fraction of it.
  /// Returns the reply length, or 0 if @p out cannot hold it (caller emits a
  /// warning EVR — never overruns). Requires a prior HELLO (`helloSeen()`).
  /// @p thrusters points to at least `nThruster()` `ThrusterCommandRecord`s
  /// (throttles for the §17 burn executor); `nullptr` fills zeros (off).
  /// @p est_q_body_eci, when non-null, is the FSW's attitude estimate
  /// (Body<-ECI, scalar-first) carried back for diagnosis only — see
  /// `StepReplyHeader` on why it exists and why the plant must not read it.
  /// Null leaves the reply's `est_valid` clear, which is the honest state for a
  /// deployment with no estimator running.
  std::size_t buildStepReply(std::uint64_t macro_step, const WheelCommandRecord* wheels,
                             const MtqCommandRecord* mtqs, double mtq_on_window_s,
                             std::uint8_t* out, std::size_t out_cap,
                             const ThrusterCommandRecord* thrusters = nullptr,
                             const double* est_q_body_eci = nullptr) {
    if (!hello_seen_) {
      return 0;  // STEP_REPLY before HELLO breaks the handshake order
    }
    std::size_t woff = 0;
    StepReplyHeader rhdr;
    rhdr.macro_step = macro_step;            // barrier echo (§2.4)
    rhdr.mtq_on_window_s = mtq_on_window_s;  // §7 duty-cycle schedule
    if (est_q_body_eci != nullptr) {
      for (int i = 0; i < 4; ++i) {
        rhdr.est_q_body_eci[i] = est_q_body_eci[i];
      }
      rhdr.est_valid = 1;
    }
    if (!writeRecord(out, out_cap, woff, rhdr)) {
      return 0;  // reply would overflow the caller buffer
    }
    for (std::uint32_t i = 0; i < n_wheel_; ++i) {
      const WheelCommandRecord rec = wheels != nullptr ? wheels[i] : WheelCommandRecord{};
      if (!writeRecord(out, out_cap, woff, rec)) {
        return 0;
      }
    }
    for (std::uint32_t i = 0; i < n_mtq_; ++i) {
      const MtqCommandRecord rec = mtqs != nullptr ? mtqs[i] : MtqCommandRecord{};
      if (!writeRecord(out, out_cap, woff, rec)) {
        return 0;
      }
    }
    for (std::uint32_t i = 0; i < n_thruster_; ++i) {
      const ThrusterCommandRecord rec =
          thrusters != nullptr ? thrusters[i] : ThrusterCommandRecord{};
      if (!writeRecord(out, out_cap, woff, rec)) {
        return 0;
      }
    }
    return woff;
  }

  bool helloSeen() const { return hello_seen_; }

  std::uint32_t nThruster() const { return n_thruster_; }

  std::uint32_t nWheel() const { return n_wheel_; }

  std::uint32_t nMtq() const { return n_mtq_; }

  /// @name Latest decoded STEP_REQ measurements
  ///
  /// Valid after `handle` returned `kStepReq`; the arrays hold the records of
  /// the most recently decoded STEP_REQ, in vehicle build order, and are only
  /// written by a fully validated message — a rejected STEP_REQ leaves the
  /// previous step's measurements in place rather than tearing them. Indices
  /// past the matching count are unwritten and must not be read.
  /// @{
  std::uint32_t nImu() const { return n_imu_; }

  std::uint32_t nStarTracker() const { return n_star_tracker_; }

  std::uint32_t nSunSensor() const { return n_sun_sensor_; }

  std::uint32_t nMagnetometer() const { return n_magnetometer_; }

  std::uint32_t nGnss() const { return n_gnss_; }

  const ImuRecord& imu(std::uint32_t i) const { return imu_[i]; }

  const StarTrackerRecord& starTracker(std::uint32_t i) const { return star_tracker_[i]; }

  const SunSensorRecord& sunSensor(std::uint32_t i) const { return sun_sensor_[i]; }

  const MagnetometerRecord& magnetometer(std::uint32_t i) const { return magnetometer_[i]; }

  const GnssRecord& gnss(std::uint32_t i) const { return gnss_[i]; }

  /// Wheel tachometer of unit @p i (§8.5 momentum management). Bounded by
  /// `nWheel()`, which HELLO declares and which also sizes the reply.
  const WheelTachRecord& wheelTach(std::uint32_t i) const { return wheel_tach_[i]; }

  /// @}

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
        hello.n_gnss > kMaxUnits || hello.n_wheel > kMaxUnits || hello.n_mtq > kMaxUnits ||
        hello.n_thruster > kMaxUnits) {
      return r;  // kMalformed: implausible unit count
    }
    n_imu_ = hello.n_imu;
    n_star_tracker_ = hello.n_star_tracker;
    n_sun_sensor_ = hello.n_sun_sensor;
    n_magnetometer_ = hello.n_magnetometer;
    n_gnss_ = hello.n_gnss;
    n_wheel_ = hello.n_wheel;
    n_mtq_ = hello.n_mtq;
    n_thruster_ = hello.n_thruster;
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

  /// STEP_REQ decode: validate the header and the handshake order, read the
  /// per-unit sensor records that follow (in the HELLO-declared order and
  /// counts), then surface the barrier step and sim epoch. The reply is built
  /// later by `buildStepReply`, once the caller has run the rate group.
  ///
  /// The whole message is length-checked *before* any record is copied, so a
  /// truncated or over-long request is rejected whole and the previously decoded
  /// measurements stay intact — a partially overwritten sensor set would be
  /// indistinguishable downstream from a fresh one.
  HandleResult decodeStepReq(const std::uint8_t* in, std::size_t in_len) {
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
    if (req.epoch_tai_ns < 0) {
      // Trust-boundary check: a negative TAI epoch off the wire must be
      // rejected here, so downstream sim-time consumers (SitlTime) can keep
      // `epoch >= 0` as a true invariant rather than asserting on wire data.
      return r;  // kMalformed
    }
    const std::size_t expected =
        sizeof(StepReqHeader) + static_cast<std::size_t>(n_imu_) * sizeof(ImuRecord) +
        static_cast<std::size_t>(n_star_tracker_) * sizeof(StarTrackerRecord) +
        static_cast<std::size_t>(n_sun_sensor_) * sizeof(SunSensorRecord) +
        static_cast<std::size_t>(n_magnetometer_) * sizeof(MagnetometerRecord) +
        static_cast<std::size_t>(n_gnss_) * sizeof(GnssRecord) +
        static_cast<std::size_t>(n_wheel_) * sizeof(WheelTachRecord);
    if (in_len != expected) {
      return r;  // kMalformed: not the sensor set HELLO declared
    }

    // Counts are bounded by kMaxUnits at HELLO, so every loop below is bounded
    // by the fixed array sizes.
    for (std::uint32_t i = 0; i < n_imu_; ++i) {
      if (!readRecord(in, in_len, off, imu_[i])) {
        return r;
      }
    }
    for (std::uint32_t i = 0; i < n_star_tracker_; ++i) {
      if (!readRecord(in, in_len, off, star_tracker_[i])) {
        return r;
      }
    }
    for (std::uint32_t i = 0; i < n_sun_sensor_; ++i) {
      if (!readRecord(in, in_len, off, sun_sensor_[i])) {
        return r;
      }
    }
    for (std::uint32_t i = 0; i < n_magnetometer_; ++i) {
      if (!readRecord(in, in_len, off, magnetometer_[i])) {
        return r;
      }
    }
    for (std::uint32_t i = 0; i < n_gnss_; ++i) {
      if (!readRecord(in, in_len, off, gnss_[i])) {
        return r;
      }
    }
    for (std::uint32_t i = 0; i < n_wheel_; ++i) {
      if (!readRecord(in, in_len, off, wheel_tach_[i])) {
        return r;
      }
    }

    r.status = HandleStatus::kStepReq;
    r.macro_step = req.macro_step;
    r.epoch_tai_ns = req.epoch_tai_ns;
    return r;
  }

  std::uint32_t n_imu_ = 0;
  std::uint32_t n_star_tracker_ = 0;
  std::uint32_t n_sun_sensor_ = 0;
  std::uint32_t n_magnetometer_ = 0;
  std::uint32_t n_gnss_ = 0;
  std::uint32_t n_wheel_ = 0;
  std::uint32_t n_mtq_ = 0;
  std::uint32_t n_thruster_ = 0;
  bool hello_seen_ = false;

  // Latest decoded measurements, fixed capacity (kMaxUnits bounds the HELLO
  // counts). ~3.7 kB held by value: no heap, and the component owns one handler.
  ImuRecord imu_[kMaxUnits]{};
  StarTrackerRecord star_tracker_[kMaxUnits]{};
  SunSensorRecord sun_sensor_[kMaxUnits]{};
  MagnetometerRecord magnetometer_[kMaxUnits]{};
  GnssRecord gnss_[kMaxUnits]{};
  WheelTachRecord wheel_tach_[kMaxUnits]{};
};

}  // namespace polaris::sitl

#endif  // POLARIS_SITL_HANDLER_HPP

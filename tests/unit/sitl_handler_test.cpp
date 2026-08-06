/// @file Tests for the FSW-side SITL message handler (lib/sitl/handler.hpp).
///
/// Pins the §2.2/§2.4 reply contract the sim's `SitlServer` depends on: HELLO →
/// HELLO_ACK echoing every count; STEP_REQ decoded to the barrier step + sim
/// epoch, then STEP_REPLY built from caller-supplied per-unit commands (Push 35
/// two-phase API); SHUTDOWN → no reply; malformed input rejected (never
/// asserted). The byte layout is exercised end-to-end against the same
/// `wire.hpp` structs the sim serializes, so the two sides cannot drift.

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

// A full STEP_REQ for the suite makeHello() declares: the fixed header followed
// by 1 IMU, 1 star tracker, 2 sun sensors, 1 magnetometer, 1 GNSS record and
// `n_wheel` tachometer records, in that order. Sensor values are seeded from @p seed so a decode
// can be checked field by field.
struct StepReqMessage {
  std::array<std::uint8_t, ps::kMaxStepReqBytes> bytes{};
  std::size_t len = 0;
};

StepReqMessage makeStepReq(std::uint64_t macro_step, std::int64_t epoch_tai_ns, double seed = 1.0,
                           std::uint32_t n_wheel = 4) {
  StepReqMessage msg;
  ps::StepReqHeader hdr;
  hdr.macro_step = macro_step;
  hdr.epoch_tai_ns = epoch_tai_ns;
  std::size_t off = 0;
  EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, hdr));

  ps::ImuRecord imu;
  imu.delta_angle_rad[0] = 0.001 * seed;
  imu.delta_angle_rad[1] = 0.002 * seed;
  imu.delta_angle_rad[2] = 0.003 * seed;
  imu.time_tag_tai_ns = epoch_tai_ns;
  imu.valid = 1;
  EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, imu));

  ps::StarTrackerRecord st;
  st.q_body_eci[0] = 1.0;
  st.time_tag_tai_ns = epoch_tai_ns;
  st.valid = 1;
  EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, st));

  for (int i = 0; i < 2; ++i) {
    ps::SunSensorRecord ss;
    ss.sun_dir_body[0] = 1.0;
    ss.accuracy_sigma_rad = 0.01 * seed * static_cast<double>(i + 1);
    ss.time_tag_tai_ns = epoch_tai_ns;
    ss.fresh = 1;
    ss.sun_present = 1;
    ss.valid = 1;
    EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, ss));
  }

  ps::MagnetometerRecord mag;
  mag.field_tesla[1] = 3.0e-5 * seed;
  mag.time_tag_tai_ns = epoch_tai_ns;
  mag.valid = 1;
  EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, mag));

  ps::GnssRecord gnss;
  gnss.position_ecef_m[0] = 7.0e6 * seed;
  gnss.time_tag_gps_ns = epoch_tai_ns;
  gnss.fresh = 1;
  gnss.valid = 1;
  EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, gnss));

  // Wheel tachometers (wire v2): the §8.5 momentum management's input, and part
  // of the declared length, so a request that omitted them would — correctly —
  // be rejected as not the suite HELLO declared.
  for (std::uint32_t i = 0; i < n_wheel; ++i) {
    ps::WheelTachRecord tach;
    tach.speed_rad_s = 100.0 * seed * static_cast<double>(i + 1);
    tach.time_tag_tai_ns = epoch_tai_ns;
    tach.valid = 1;
    EXPECT_TRUE(ps::writeRecord(msg.bytes.data(), msg.bytes.size(), off, tach));
  }

  msg.len = off;
  return msg;
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

TEST(SitlHandler, StepReqDecodesStepAndEpoch) {
  ps::SitlHandler h;
  doHello(h, 4, 3);

  const StepReqMessage req = makeStepReq(42, 123456789);
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(req.bytes.data(), req.len, out.data(), out.size());

  // handle() only decodes a STEP_REQ; it writes no reply.
  ASSERT_EQ(r.status, ps::HandleStatus::kStepReq);
  EXPECT_EQ(r.macro_step, 42u);
  EXPECT_EQ(r.epoch_tai_ns, 123456789);
  EXPECT_EQ(r.reply_len, 0u);
}

TEST(SitlHandler, StepReqDecodesSensorRecords) {
  // Push 40: the per-unit sensor records the request carries are decoded and
  // held for the component to republish on the GNC measurement ports.
  RecordProperty("verifies", "REQ-ADET-002");
  ps::SitlHandler h;
  doHello(h, 4, 3);

  const StepReqMessage req = makeStepReq(7, 1'770'000'000'000'000'000LL, 2.0);
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  ASSERT_EQ(h.handle(req.bytes.data(), req.len, out.data(), out.size()).status,
            ps::HandleStatus::kStepReq);

  // Counts come from HELLO (makeHello: 1 IMU, 1 ST, 2 SS, 1 MAG, 1 GNSS).
  ASSERT_EQ(h.nImu(), 1u);
  ASSERT_EQ(h.nStarTracker(), 1u);
  ASSERT_EQ(h.nSunSensor(), 2u);
  ASSERT_EQ(h.nMagnetometer(), 1u);
  ASSERT_EQ(h.nGnss(), 1u);

  EXPECT_EQ(h.imu(0).valid, 1);
  EXPECT_DOUBLE_EQ(h.imu(0).delta_angle_rad[1], 0.004);
  EXPECT_EQ(h.imu(0).time_tag_tai_ns, 1'770'000'000'000'000'000LL);
  // Unit identity is positional: sun sensor 1 must not be sun sensor 0.
  EXPECT_DOUBLE_EQ(h.sunSensor(0).accuracy_sigma_rad, 0.02);
  EXPECT_DOUBLE_EQ(h.sunSensor(1).accuracy_sigma_rad, 0.04);
  EXPECT_EQ(h.sunSensor(0).sun_present, 1);
  EXPECT_DOUBLE_EQ(h.magnetometer(0).field_tesla[1], 6.0e-5);
  EXPECT_DOUBLE_EQ(h.gnss(0).position_ecef_m[0], 1.4e7);
  EXPECT_DOUBLE_EQ(h.starTracker(0).q_body_eci[0], 1.0);
  // Wheel tachometers, positional like every other unit: wheel 1 is not wheel 0.
  ASSERT_EQ(h.nWheel(), 4u);
  EXPECT_EQ(h.wheelTach(0).valid, 1);
  EXPECT_DOUBLE_EQ(h.wheelTach(0).speed_rad_s, 200.0);
  EXPECT_DOUBLE_EQ(h.wheelTach(3).speed_rad_s, 800.0);
}

TEST(SitlHandler, StepReqOfWrongLengthKeepsPreviousMeasurements) {
  // A request whose byte count does not match the HELLO-declared sensor suite is
  // rejected whole. The previously decoded measurements must survive: a
  // partially overwritten sensor set is indistinguishable downstream from a
  // fresh one, which is the failure mode worth designing out.
  ps::SitlHandler h;
  doHello(h, 4, 3);

  const StepReqMessage good = makeStepReq(1, 1'000'000'000LL, 3.0);
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  ASSERT_EQ(h.handle(good.bytes.data(), good.len, out.data(), out.size()).status,
            ps::HandleStatus::kStepReq);
  const double kept = h.gnss(0).position_ecef_m[0];

  // Same message one record short of what the counts imply.
  const ps::HandleResult r =
      h.handle(good.bytes.data(), good.len - sizeof(ps::GnssRecord), out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_DOUBLE_EQ(h.gnss(0).position_ecef_m[0], kept);
}

TEST(SitlHandler, BuildStepReplyZeroesCommandsWhenCallerSuppliesNone) {
  ps::SitlHandler h;
  doHello(h, 4, 3);

  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const std::size_t reply_len = h.buildStepReply(42, nullptr, nullptr, 0.0, out.data(), out.size());

  // Header + 4 wheels + 3 MTQs, all zeroed (nullptr command arrays).
  const std::size_t expected = sizeof(ps::StepReplyHeader) + 4 * sizeof(ps::WheelCommandRecord) +
                               3 * sizeof(ps::MtqCommandRecord);
  ASSERT_EQ(reply_len, expected);

  std::size_t off = 0;
  ps::StepReplyHeader rhdr;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, rhdr));
  EXPECT_TRUE(ps::checkHeader(rhdr.hdr, ps::MsgType::kStepReply));
  EXPECT_EQ(rhdr.macro_step, 42u);
  // No duty-cycle schedule from the caller means the rods stay off for the whole
  // step (§7) — the safe reading, not a full-period drive.
  EXPECT_EQ(rhdr.mtq_on_window_s, 0.0);
  for (int i = 0; i < 4; ++i) {
    ps::WheelCommandRecord w;
    ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, w));
    EXPECT_EQ(w.mode, 0);  // torque
    EXPECT_EQ(w.value, 0.0);
  }
  for (int i = 0; i < 3; ++i) {
    ps::MtqCommandRecord m;
    ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, m));
    EXPECT_EQ(m.dipole_am2[0], 0.0);
    EXPECT_EQ(m.dipole_am2[1], 0.0);
    EXPECT_EQ(m.dipole_am2[2], 0.0);
  }
  EXPECT_EQ(off, reply_len);  // reply is exactly full, no trailing bytes
}

TEST(SitlHandler, BuildStepReplyCarriesCallerSuppliedCommands) {
  ps::SitlHandler h;
  doHello(h, 2, 1);

  std::array<ps::WheelCommandRecord, 2> wheels{};
  wheels[0].value = 0.0125;
  wheels[0].mode = 0;
  wheels[1].value = -0.5;
  wheels[1].mode = 1;  // speed mode
  std::array<ps::MtqCommandRecord, 1> mtqs{};
  mtqs[0].dipole_am2[0] = 1.5;
  mtqs[0].dipole_am2[1] = -2.5;
  mtqs[0].dipole_am2[2] = 3.5;

  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const std::size_t reply_len =
      h.buildStepReply(7, wheels.data(), mtqs.data(), 0.05, out.data(), out.size());
  ASSERT_GT(reply_len, 0u);

  std::size_t off = 0;
  ps::StepReplyHeader rhdr;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, rhdr));
  EXPECT_EQ(rhdr.macro_step, 7u);
  // The §7 MTQ-on window crosses the wire alongside the dipoles: the plant needs
  // it to drive the rods over the right fraction of the step and to corrupt any
  // magnetometer sample landing inside it.
  EXPECT_EQ(rhdr.mtq_on_window_s, 0.05);
  ps::WheelCommandRecord w0;
  ps::WheelCommandRecord w1;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, w0));
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, w1));
  EXPECT_EQ(w0.value, 0.0125);
  EXPECT_EQ(w0.mode, 0);
  EXPECT_EQ(w1.value, -0.5);
  EXPECT_EQ(w1.mode, 1);
  ps::MtqCommandRecord m0;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, m0));
  EXPECT_EQ(m0.dipole_am2[0], 1.5);
  EXPECT_EQ(m0.dipole_am2[1], -2.5);
  EXPECT_EQ(m0.dipole_am2[2], 3.5);
  EXPECT_EQ(off, reply_len);
}

TEST(SitlHandler, BuildStepReplyBeforeHelloReturnsZero) {
  ps::SitlHandler h;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  EXPECT_EQ(h.buildStepReply(1, nullptr, nullptr, 0.0, out.data(), out.size()), 0u);
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

TEST(SitlHandler, OversizeSensorCountRejected) {
  // The count guard is an OR over every unit type, not just the reply-sizing
  // wheel/MTQ counts — a bogus sensor count is rejected too.
  ps::SitlHandler h;
  ps::HelloMsg hello = makeHello(1, 1);
  hello.n_imu = ps::kMaxUnits + 1;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_FALSE(h.helloSeen());
}

TEST(SitlHandler, WrongVersionIsMalformed) {
  ps::SitlHandler h;
  ps::HelloMsg hello = makeHello(1, 1);
  hello.hdr.version = ps::kVersion + 1;  // a future/incompatible layout
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_FALSE(h.helloSeen());
}

TEST(SitlHandler, TruncatedStepReqIsMalformed) {
  // A message that clears the MsgHeader but is shorter than the full StepReqHeader
  // must be rejected (readRecord of the header fails), never partially decoded.
  ps::SitlHandler h;
  doHello(h, 1, 1);
  ps::StepReqHeader req;
  req.macro_step = 7;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r =
      h.handle(reinterpret_cast<const std::uint8_t*>(&req), sizeof(req) - 1,  // one byte short
               out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_EQ(r.reply_len, 0u);
}

TEST(SitlHandler, UndersizedReplyBufferFailsSafely) {
  // Reply buffer too small for the STEP_REPLY the counts imply → writeRecord
  // stops before overrunning (ASan would flag any overflow) and buildStepReply
  // returns 0 (caller emits a warning EVR).
  ps::SitlHandler h;
  doHello(h, 4, 3);
  std::array<std::uint8_t, sizeof(ps::StepReplyHeader) + 4> tiny{};  // header fits, records don't
  EXPECT_EQ(h.buildStepReply(1, nullptr, nullptr, 0.0, tiny.data(), tiny.size()), 0u);
}

TEST(SitlHandler, UndersizedHelloAckBufferFailsSafely) {
  ps::SitlHandler h;
  const ps::HelloMsg hello = makeHello(2, 2);
  std::array<std::uint8_t, sizeof(ps::HelloMsg) - 1> tiny{};  // can't hold the echo
  const ps::HandleResult r = h.handle(reinterpret_cast<const std::uint8_t*>(&hello), sizeof(hello),
                                      tiny.data(), tiny.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
}

TEST(SitlHandler, ReHelloReLatchesCounts) {
  // The contract for a second HELLO on one link: it is accepted and re-latches
  // the reply counts (the handler is stateless across the count negotiation
  // beyond the latch), so a renegotiated suite sizes later replies correctly.
  ps::SitlHandler h;
  doHello(h, 4, 3);
  EXPECT_EQ(h.nWheel(), 4u);
  EXPECT_EQ(h.nMtq(), 3u);

  doHello(h, 2, 1);  // second HELLO with different counts
  EXPECT_TRUE(h.helloSeen());
  EXPECT_EQ(h.nWheel(), 2u);
  EXPECT_EQ(h.nMtq(), 1u);

  // A subsequent STEP_REPLY reflects the re-latched counts — and so does the
  // *request* length, since the tachometer records are sized by the same count.
  const StepReqMessage req = makeStepReq(5, 5'000'000'000LL, 1.0, /*n_wheel=*/2);
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r = h.handle(req.bytes.data(), req.len, out.data(), out.size());
  ASSERT_EQ(r.status, ps::HandleStatus::kStepReq);
  const std::size_t reply_len = h.buildStepReply(5, nullptr, nullptr, 0.0, out.data(), out.size());
  const std::size_t expected = sizeof(ps::StepReplyHeader) + 2 * sizeof(ps::WheelCommandRecord) +
                               1 * sizeof(ps::MtqCommandRecord);
  EXPECT_EQ(reply_len, expected);
}

TEST(SitlHandler, NegativeEpochIsMalformed) {
  // Trust-boundary check (JPL rule 7): a negative TAI epoch off the wire is
  // rejected in decode, so SitlTime's `epoch >= 0` FW_ASSERT stays a true
  // invariant and malformed input can never abort the FSW.
  ps::SitlHandler h;
  doHello(h, 1, 1);
  ps::StepReqHeader req;
  req.macro_step = 3;
  req.epoch_tai_ns = -1;
  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const ps::HandleResult r =
      h.handle(reinterpret_cast<const std::uint8_t*>(&req), sizeof(req), out.data(), out.size());
  EXPECT_EQ(r.status, ps::HandleStatus::kMalformed);
  EXPECT_EQ(r.reply_len, 0u);
}

TEST(SitlHandler, BuildStepReplySerializesCallerCommands) {
  // The non-null command path: caller-latched wheel/MTQ records must land in
  // the reply byte-for-byte, in unit order after the header.
  ps::SitlHandler h;
  doHello(h, 2, 1);

  std::array<ps::WheelCommandRecord, ps::kMaxUnits> wheels{};
  wheels[0].value = 0.011;
  wheels[1].value = -0.022;
  wheels[1].mode = 0;
  std::array<ps::MtqCommandRecord, ps::kMaxUnits> mtqs{};
  mtqs[0].dipole_am2[0] = 1.5;
  mtqs[0].dipole_am2[1] = -2.5;
  mtqs[0].dipole_am2[2] = 3.5;

  std::array<std::uint8_t, ps::kMaxStepReplyBytes> out{};
  const std::size_t reply_len =
      h.buildStepReply(9, wheels.data(), mtqs.data(), 0.04, out.data(), out.size());
  const std::size_t expected = sizeof(ps::StepReplyHeader) + 2 * sizeof(ps::WheelCommandRecord) +
                               1 * sizeof(ps::MtqCommandRecord);
  ASSERT_EQ(reply_len, expected);

  std::size_t off = 0;
  ps::StepReplyHeader rh;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, rh));
  EXPECT_EQ(rh.macro_step, 9u);
  EXPECT_EQ(rh.mtq_on_window_s, 0.04);
  ps::WheelCommandRecord w0, w1;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, w0));
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, w1));
  EXPECT_EQ(w0.value, 0.011);
  EXPECT_EQ(w1.value, -0.022);
  ps::MtqCommandRecord m0;
  ASSERT_TRUE(ps::readRecord(out.data(), reply_len, off, m0));
  EXPECT_EQ(m0.dipole_am2[0], 1.5);
  EXPECT_EQ(m0.dipole_am2[1], -2.5);
  EXPECT_EQ(m0.dipole_am2[2], 3.5);
  EXPECT_EQ(off, reply_len);
}

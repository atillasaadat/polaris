#ifndef POLARIS_SITL_WIRE_HPP
#define POLARIS_SITL_WIRE_HPP

/// @file
/// @brief SITL lockstep wire format — the §2.2/§2.4 plant↔FSW message payloads.
///
/// One header shared by **both** processes (the truth sim serializes, the F´
/// `SitlBridge` component deserializes, and vice versa for the reply), so the
/// two sides cannot drift: the layout is the POD structs below, memcpy'd
/// verbatim. Payloads ride inside standard F´ frames (`Svc::FprimeProtocol`
/// start word + length + CRC32) over the dedicated SITL TCP socket — the
/// §18.11 transport — never over the GDS ground link.
///
/// **Layout contract.** Little-endian, fixed-width types, natural 8-byte
/// alignment with explicit padding; every struct's size is static_asserted.
/// Records carry what the real electrical interface would: measurements and
/// their validity/freshness flags. Truth-side diagnostics that a physical part
/// would not report (true incidence angle, shadow factor, albedo split,
/// jamming-region name) deliberately do **not** cross — §2.3 separation.
/// The realised measurement σ does cross (`SunSensorRecord::accuracy_sigma_rad`,
/// the GNSS σs): the design exposes it so estimators weight with the noise the
/// sensor actually had (§6).
///
/// **Flight-safe:** no heap, no exceptions, bounded arrays (`kMaxUnits`,
/// `kMaxDiodes`). Unit identity is positional — HELLO declares per-type counts
/// and vector order follows the vehicle's build order (§19.4); names stay
/// sim-side.
///
/// Implements the §2.4 macro-step barrier messages; REQ-SIM-004 (truth/onboard
/// separation) constrains the field set.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace polaris::sitl {

inline constexpr std::uint32_t kMagic = 0x50534954u;  ///< "PSIT"
inline constexpr std::uint16_t kVersion = 1;

/// Bounded unit counts per sensor/actuator type (wire arrays are sized to the
/// HELLO-declared counts, never these maxima; these bound validation).
inline constexpr std::uint32_t kMaxUnits = 8;
inline constexpr std::uint32_t kMaxDiodes = 16;  ///< per analogue sun sensor

enum class MsgType : std::uint16_t {
  kHello = 1,      ///< sim → FSW at connect: counts + macro step
  kHelloAck = 2,   ///< FSW → sim: accepted (echoes HELLO)
  kStepReq = 3,    ///< sim → FSW at each macro boundary: FswInputs
  kStepReply = 4,  ///< FSW → sim: FswOutputs for the NEXT interval (§2.4)
  kShutdown = 5,   ///< sim → FSW: run complete, exit cleanly
};

/// Common prefix of every SITL payload.
struct MsgHeader {
  std::uint32_t magic = kMagic;
  std::uint16_t version = kVersion;
  std::uint16_t type = 0;  ///< MsgType
};

static_assert(sizeof(MsgHeader) == 8);

/// Connect-time contract: how many of each unit the STEP messages carry, in
/// vehicle build order, and the macro-step the barrier runs at.
struct HelloMsg {
  MsgHeader hdr{kMagic, kVersion, static_cast<std::uint16_t>(MsgType::kHello)};
  std::uint32_t n_imu = 0;
  std::uint32_t n_star_tracker = 0;
  std::uint32_t n_sun_sensor = 0;
  std::uint32_t n_magnetometer = 0;
  std::uint32_t n_gnss = 0;
  std::uint32_t n_wheel = 0;  ///< reply sizing
  std::uint32_t n_mtq = 0;    ///< reply sizing
  std::uint32_t pad = 0;
  std::int64_t macro_dt_ns = 0;
};

static_assert(sizeof(HelloMsg) == 48);

/// §2.4 IMU accumulation since the last FSW read.
struct ImuRecord {
  double delta_angle_rad[3] = {};
  double delta_velocity_mps[3] = {};
  std::int64_t time_tag_tai_ns = 0;  ///< newest contributing sample
  std::int32_t samples = 0;          ///< native-rate samples folded in
  std::uint8_t valid = 0;
  std::uint8_t pad[3] = {};
};

static_assert(sizeof(ImuRecord) == 64);

/// Latest star-tracker sample (§2.4 latest-valid publication). Carries what
/// the real unit reports: the attitude solution, its mode, and **coarse**
/// status flags — a tracker knows it is blinded (`occluder`, like a real
/// part's Earth/Sun-in-FOV flag) but not how much of its FOV each body covers.
/// The truth-side per-body FOV fractions (`OcclusionState`) deliberately do
/// not cross — they are computed from the truth ephemeris (§2.3).
struct StarTrackerRecord {
  double q_body_eci[4] = {};  ///< JPL scalar-first [q0,q1,q2,q3]
  double acquisition_elapsed_s = 0.0;
  std::int64_t time_tag_tai_ns = 0;
  std::uint8_t valid = 0;
  std::uint8_t mode = 0;      ///< sensors::StarTrackerMode
  std::uint8_t occluder = 0;  ///< sensors::Occluder (coarse blinded-by status)
  std::uint8_t rate_limited = 0;
  std::uint8_t accel_limited = 0;
  std::uint8_t ever_sampled = 0;
  std::uint8_t pad[2] = {};
};

static_assert(sizeof(StarTrackerRecord) == 56);

/// Latest sun-sensor sample. Carries both interfaces (§6.4): per-diode counts
/// for an analogue part (first `n_counts` entries), the processed unit vector
/// for a digital part. Truth diagnostics (incidence, shadow, albedo split) stay
/// sim-side.
struct SunSensorRecord {
  double counts[kMaxDiodes] = {};
  double sun_dir_body[3] = {};
  double accuracy_sigma_rad = 0.0;  ///< realised 1σ for estimator weighting
  std::int64_t time_tag_tai_ns = 0;
  std::uint32_t n_counts = 0;
  std::uint8_t fresh = 0;
  std::uint8_t sun_present = 0;
  std::uint8_t valid = 0;
  std::uint8_t ever_sampled = 0;
};

static_assert(sizeof(SunSensorRecord) == 176);

/// Latest magnetometer sample.
struct MagnetometerRecord {
  double field_tesla[3] = {};
  std::int64_t time_tag_tai_ns = 0;
  std::uint8_t valid = 0;
  std::uint8_t ever_sampled = 0;
  std::uint8_t pad[6] = {};
};

static_assert(sizeof(MagnetometerRecord) == 40);

/// Latest GNSS PVT fix. GPS time and ECEF exactly as the receiver reports
/// (§3.2); the FSW applies TAI = GPS + 19 s and ECEF→ECI on ingest.
struct GnssRecord {
  double position_ecef_m[3] = {};
  double velocity_ecef_mps[3] = {};
  double clock_bias_s = 0.0;
  double position_sigma_h_m = 0.0;
  double position_sigma_v_m = 0.0;
  double velocity_sigma_mps = 0.0;
  double time_sigma_s = 0.0;
  std::int64_t time_tag_gps_ns = 0;  ///< receiver-stamped, includes clock bias
  std::uint8_t fresh = 0;
  std::uint8_t valid = 0;
  std::uint8_t jammed = 0;
  std::uint8_t ever_sampled = 0;
  std::uint8_t pad[4] = {};
};

static_assert(sizeof(GnssRecord) == 104);

/// STEP_REQ fixed prefix; the per-unit records follow contiguously in HELLO
/// order and counts: ImuRecord×n_imu, StarTrackerRecord×n_star_tracker,
/// SunSensorRecord×n_sun_sensor, MagnetometerRecord×n_magnetometer,
/// GnssRecord×n_gnss.
struct StepReqHeader {
  MsgHeader hdr{kMagic, kVersion, static_cast<std::uint16_t>(MsgType::kStepReq)};
  std::int64_t epoch_tai_ns = 0;  ///< the macro-step boundary
  std::uint64_t macro_step = 0;
};

static_assert(sizeof(StepReqHeader) == 24);

/// One wheel command in the reply (§7 torque or wheel-local speed mode).
struct WheelCommandRecord {
  double value = 0.0;     ///< [N·m] torque mode, [rad/s] speed mode
  std::uint8_t mode = 0;  ///< 0 = torque, 1 = speed
  std::uint8_t pad[7] = {};
};

static_assert(sizeof(WheelCommandRecord) == 16);

/// One magnetorquer dipole command in the reply, body frame [A·m²].
struct MtqCommandRecord {
  double dipole_am2[3] = {};
};

static_assert(sizeof(MtqCommandRecord) == 24);

/// STEP_REPLY fixed prefix; WheelCommandRecord×n_wheel then
/// MtqCommandRecord×n_mtq follow. `macro_step` echoes the request — the barrier
/// check that neither side skipped a step.
struct StepReplyHeader {
  MsgHeader hdr{kMagic, kVersion, static_cast<std::uint16_t>(MsgType::kStepReply)};
  std::uint64_t macro_step = 0;
  /// §7 MTQ/MAG duty-cycle on-window [s]: how long, from the start of the next
  /// macro step, the rods are energised at the commanded dipole. One value for
  /// the set rather than one per rod, because the schedule is owned by the
  /// controller as a whole — a per-rod window would be three chances for the
  /// quiet window the magnetometer is judged against to disagree with itself.
  /// Zero or negative leaves the rods off for the whole step.
  double mtq_on_window_s = 0.0;
};

static_assert(sizeof(StepReplyHeader) == 24);

/// Largest possible STEP_REQ payload — sizes receive buffers on both ends.
inline constexpr std::size_t kMaxStepReqBytes =
    sizeof(StepReqHeader) +
    kMaxUnits * (sizeof(ImuRecord) + sizeof(StarTrackerRecord) + sizeof(SunSensorRecord) +
                 sizeof(MagnetometerRecord) + sizeof(GnssRecord));

/// Largest possible STEP_REPLY payload.
inline constexpr std::size_t kMaxStepReplyBytes =
    sizeof(StepReplyHeader) + kMaxUnits * (sizeof(WheelCommandRecord) + sizeof(MtqCommandRecord));

/// Validate a received header: magic, version, and expected type.
inline bool checkHeader(const MsgHeader& h, MsgType expected) {
  return h.magic == kMagic && h.version == kVersion &&
         h.type == static_cast<std::uint16_t>(expected);
}

/// Read a POD record out of a byte buffer. Returns false (touching nothing) if
/// fewer than sizeof(T) bytes remain. Advances @p offset on success.
template <typename T>
bool readRecord(const std::uint8_t* buf, std::size_t len, std::size_t& offset, T& out) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset + sizeof(T) > len) {
    return false;
  }
  std::memcpy(&out, buf + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

/// Append a POD record to a byte buffer of capacity @p cap. Returns false
/// (touching nothing) if it does not fit. Advances @p offset on success.
template <typename T>
bool writeRecord(std::uint8_t* buf, std::size_t cap, std::size_t& offset, const T& in) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset + sizeof(T) > cap) {
    return false;
  }
  std::memcpy(buf + offset, &in, sizeof(T));
  offset += sizeof(T);
  return true;
}

}  // namespace polaris::sitl

#endif  // POLARIS_SITL_WIRE_HPP

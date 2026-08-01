// ======================================================================
// \title  SitlBridge.cpp
// \brief  Flight end of the SITL lockstep transport (design doc §2.2, §2.4)
// ======================================================================

#include "flight/PolarisFsw/SitlBridge/SitlBridge.hpp"

#include <Os/RawTime.hpp>

namespace flight {

namespace {
// Emit a progress EVR on the first step and every kStepMilestone steps, so a
// long run leaves a bounded event trail rather than one per macro step.
constexpr U64 kStepMilestone = 100;

//! Copy a wire 3-vector into the GncPorts array type.
Vec3F64 toVec3(const double (&v)[3]) {
  Vec3F64 out;
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}
}  // namespace

// The rate-group command ports carry fixed arrays sized to kMaxUnits; the reply
// buffer indexing below relies on that, so pin it at compile time.
static_assert(WheelTorqueSet::SIZE == polaris::sitl::kMaxUnits,
              "WheelTorqueSet must be sized to sitl::kMaxUnits");
static_assert(MtqDipoleSet::SIZE == polaris::sitl::kMaxUnits,
              "MtqDipoleSet must be sized to sitl::kMaxUnits");

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

SitlBridge ::SitlBridge(const char* const compName) : SitlBridgeComponentBase(compName) {}

SitlBridge ::~SitlBridge() {}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

void SitlBridge ::dataIn_handler(FwIndexType portNum, Fw::Buffer& data,
                                 const ComCfg::FrameContext& context) {
  namespace sitl = polaris::sitl;

  // After SHUTDOWN the bridge goes quiet: release the buffer and answer nothing.
  if (this->quiescent_) {
    this->dataReturnOut_out(0, data, context);
    return;
  }

  const sitl::HandleResult result = this->handler_.handle(
      data.getData(), static_cast<FwSizeType>(data.getSize()), this->reply_, sizeof(this->reply_));

  // Return ownership of the received payload buffer to the deframer. The handler
  // has already copied everything it needs into reply_.
  this->dataReturnOut_out(0, data, context);

  if (!this->connected_) {
    this->connected_ = true;
    this->log_ACTIVITY_HI_SitlConnected();
  }

  switch (result.status) {
    case sitl::HandleStatus::kHelloAck: {
      const sitl::HelloMsg& h = result.hello;
      this->log_ACTIVITY_HI_HelloReceived(h.n_imu, h.n_star_tracker, h.n_sun_sensor,
                                          h.n_magnetometer, h.n_gnss, h.n_wheel, h.n_mtq);
      this->sendReply(result.reply_len, context);
      break;
    }
    case sitl::HandleStatus::kStepReq:
      this->runStepCycle(result, context, static_cast<U32>(data.getSize()));
      break;
    case sitl::HandleStatus::kShutdown:
      this->quiescent_ = true;
      this->log_ACTIVITY_HI_SitlShutdown();
      break;
    case sitl::HandleStatus::kMalformed:
    default:
      this->log_WARNING_HI_MalformedMessage(result.msg_type, static_cast<U32>(data.getSize()));
      break;
  }
}

void SitlBridge ::dataReturnIn_handler(FwIndexType portNum, Fw::Buffer& data,
                                       const ComCfg::FrameContext& context) {
  // The reply buffer is a fixed member (reply_), not pool-allocated, so there is
  // nothing to deallocate — the framer has finished with it by the time we are
  // called (synchronous downlink on the receive task).
}

void SitlBridge ::wheelCmdIn_handler(FwIndexType portNum, const flight::WheelTorqueSet& cmds) {
  // Latch this cycle's wheel torques as wire records (torque mode). Runs inside
  // sitlCycleOut on this same task, so no locking is needed against the reply build.
  for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
    this->latest_wheel_[i].value = cmds[i];
    this->latest_wheel_[i].mode = 0;  // torque
  }
}

void SitlBridge ::mtqCmdIn_handler(FwIndexType portNum, const flight::MtqDipoleSet& cmds) {
  for (U32 i = 0; i < MtqDipoleSet::SIZE; ++i) {
    this->latest_mtq_[i].dipole_am2[0] = cmds[i][0];
    this->latest_mtq_[i].dipole_am2[1] = cmds[i][1];
    this->latest_mtq_[i].dipole_am2[2] = cmds[i][2];
  }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void SitlBridge ::runStepCycle(const polaris::sitl::HandleResult& result,
                               const ComCfg::FrameContext& context, U32 reqBytes) {
  // §2.4 steps 3-4: serve sim time, publish this step's measurements, fire the
  // FSW cycle, then reply from the commands it produced. Order matters — the
  // rate group's members read sim time and expect this step's sensor data to be
  // latched, so both must be out before cycling.
  this->timeSetOut_out(0, result.epoch_tai_ns);
  this->publishMeasurements(result.epoch_tai_ns);

  Os::RawTime cycleStart;
  (void)cycleStart.now();
  // Runs the SITL PassiveRateGroup to completion on this task; its members call
  // back into wheelCmdIn/mtqCmdIn, updating latest_wheel_/latest_mtq_.
  this->sitlCycleOut_out(0, cycleStart);

  const FwSizeType reply_len =
      this->handler_.buildStepReply(result.macro_step, this->latest_wheel_, this->latest_mtq_,
                                    this->reply_, sizeof(this->reply_));
  if (reply_len == 0) {
    // Reply would overflow the fixed buffer (never in flight sizing) — treat as
    // a malformed exchange rather than sending a truncated frame.
    this->log_WARNING_HI_MalformedMessage(result.msg_type, reqBytes);
    return;
  }

  this->steps_++;
  this->tlmWrite_MacroStep(this->steps_);
  if (this->steps_ == 1 || (this->steps_ % kStepMilestone) == 0) {
    this->log_ACTIVITY_LO_StepMilestone(result.macro_step);
  }
  this->sendReply(reply_len, context);
}

void SitlBridge ::publishMeasurements(I64 epochTaiNs) {
  namespace sitl = polaris::sitl;

  // Every measurement output array is indexed by its HELLO-declared unit count,
  // which the handler bounds by kMaxUnits; each port array must be at least that
  // wide. Asserted per array rather than once: they are separate FPP
  // declarations and could drift independently.
  static_assert(NUM_IMUOUT_OUTPUT_PORTS >= static_cast<FwIndexType>(sitl::kMaxUnits),
                "imuOut must be >= sitl::kMaxUnits");
  static_assert(NUM_SUNSENSOROUT_OUTPUT_PORTS >= static_cast<FwIndexType>(sitl::kMaxUnits),
                "sunSensorOut must be >= sitl::kMaxUnits");
  static_assert(NUM_MAGNETOMETEROUT_OUTPUT_PORTS >= static_cast<FwIndexType>(sitl::kMaxUnits),
                "magnetometerOut must be >= sitl::kMaxUnits");
  static_assert(NUM_GNSSOUT_OUTPUT_PORTS >= static_cast<FwIndexType>(sitl::kMaxUnits),
                "gnssOut must be >= sitl::kMaxUnits");
  static_assert(NUM_STARTRACKEROUT_OUTPUT_PORTS >= static_cast<FwIndexType>(sitl::kMaxUnits),
                "starTrackerOut must be >= sitl::kMaxUnits");

  // The IMU records accumulate since the last FSW read, i.e. over exactly one
  // macro step, so the interval is this epoch minus the previous one. On the
  // first step there is no previous epoch and no interval can be formed: the
  // increments go out flagged invalid rather than divided by a guessed dt.
  //
  // This assumes consecutive STEP_REQs — which the §2.4 barrier guarantees, and
  // a skipped step would break the lockstep long before it reached here. If the
  // link ever tolerates gaps, derive the interval from consecutive record time
  // tags instead; the wire already carries them.
  const double interval_s = this->have_last_epoch_
                                ? static_cast<double>(epochTaiNs - this->last_epoch_tai_ns_) / 1.0e9
                                : 0.0;
  const bool interval_ok = interval_s > 0.0;
  this->last_epoch_tai_ns_ = epochTaiNs;
  this->have_last_epoch_ = true;

  // Every loop is bounded by the HELLO-declared count, itself bounded by
  // kMaxUnits == GncMaxUnits (static_asserted above). Each output is guarded by
  // isConnected: the topology wires as many units as the FSW consumes, and a
  // unit the sim declares but nothing consumes is dropped here rather than
  // asserting on an unconnected port.
  for (U32 i = 0; i < this->handler_.nImu(); ++i) {
    if (!this->isConnected_imuOut_OutputPort(static_cast<FwIndexType>(i))) {
      continue;
    }
    const sitl::ImuRecord& rec = this->handler_.imu(i);
    ImuMeas meas;
    meas.set_deltaAngleRad(toVec3(rec.delta_angle_rad));
    meas.set_deltaVelMps(toVec3(rec.delta_velocity_mps));
    meas.set_intervalSec(interval_s);
    meas.set_timeTagNs(rec.time_tag_tai_ns);
    meas.set_valid(rec.valid != 0 && interval_ok);
    this->imuOut_out(static_cast<FwIndexType>(i), meas);
  }
  for (U32 i = 0; i < this->handler_.nSunSensor(); ++i) {
    if (!this->isConnected_sunSensorOut_OutputPort(static_cast<FwIndexType>(i))) {
      continue;
    }
    const sitl::SunSensorRecord& rec = this->handler_.sunSensor(i);
    SunSensorMeas meas;
    meas.set_dirBody(toVec3(rec.sun_dir_body));
    meas.set_sigmaRad(rec.accuracy_sigma_rad);
    meas.set_timeTagNs(rec.time_tag_tai_ns);
    meas.set_sunPresent(rec.sun_present != 0);
    // `fresh` is the unit's own flag for "this sample is from this step"; a
    // stale-but-valid sample must not masquerade as current, and the consumer's
    // age gate cannot see it because the time tag alone does not say.
    meas.set_valid(rec.valid != 0 && rec.fresh != 0);
    this->sunSensorOut_out(static_cast<FwIndexType>(i), meas);
  }
  for (U32 i = 0; i < this->handler_.nMagnetometer(); ++i) {
    if (!this->isConnected_magnetometerOut_OutputPort(static_cast<FwIndexType>(i))) {
      continue;
    }
    const sitl::MagnetometerRecord& rec = this->handler_.magnetometer(i);
    MagnetometerMeas meas;
    meas.set_fieldTesla(toVec3(rec.field_tesla));
    meas.set_timeTagNs(rec.time_tag_tai_ns);
    meas.set_valid(rec.valid != 0);
    this->magnetometerOut_out(static_cast<FwIndexType>(i), meas);
  }
  for (U32 i = 0; i < this->handler_.nGnss(); ++i) {
    if (!this->isConnected_gnssOut_OutputPort(static_cast<FwIndexType>(i))) {
      continue;
    }
    const sitl::GnssRecord& rec = this->handler_.gnss(i);
    GnssMeas meas;
    meas.set_posEcefM(toVec3(rec.position_ecef_m));
    meas.set_velEcefMps(toVec3(rec.velocity_ecef_mps));
    meas.set_timeTagGpsNs(rec.time_tag_gps_ns);
    meas.set_valid(rec.valid != 0 && rec.fresh != 0);
    this->gnssOut_out(static_cast<FwIndexType>(i), meas);
  }
  for (U32 i = 0; i < this->handler_.nStarTracker(); ++i) {
    if (!this->isConnected_starTrackerOut_OutputPort(static_cast<FwIndexType>(i))) {
      continue;
    }
    const sitl::StarTrackerRecord& rec = this->handler_.starTracker(i);
    StarTrackerMeas meas;
    QuatF64 q;
    for (U32 k = 0; k < 4; ++k) {
      q[k] = rec.q_body_eci[k];
    }
    meas.set_qBodyEci(q);
    meas.set_timeTagNs(rec.time_tag_tai_ns);
    meas.set_valid(rec.valid != 0);
    this->starTrackerOut_out(static_cast<FwIndexType>(i), meas);
  }
}

void SitlBridge ::sendReply(FwSizeType len, const ComCfg::FrameContext& context) {
  // Wrap the fixed reply buffer and hand it to the SITL framer. The framer
  // serializes header + payload + CRC into its own buffer synchronously and
  // returns this one on dataReturnIn.
  Fw::Buffer out(this->reply_, static_cast<Fw::Buffer::SizeType>(len));
  this->dataOut_out(0, out, context);
}

}  // namespace flight

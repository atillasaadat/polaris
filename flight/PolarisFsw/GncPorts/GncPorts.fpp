module flight {

  # ----------------------------------------------------------------------
  # Shared GNC sensor-measurement and estimate port/type definitions
  # (design doc §8.0, §8.1, §9.1).
  #
  # The typed seam between whatever *produces* sensor measurements — the
  # SitlBridge under SITL today, `Drv` sensor drivers on hardware tomorrow — and
  # the GNC estimators that consume them, plus the estimate product the
  # estimators publish to guidance/control/FDIR. Interface-only, like SitlPorts
  # and OnboardTablesPorts, so producer and consumer share exactly one definition
  # and cannot drift.
  #
  # Conventions: time tags are TAI nanoseconds (the FSW master clock, §3.2);
  # vectors are SI and body-frame unless the field name says otherwise; every
  # record carries its own validity flag, which a consumer MUST gate on (§9.1) —
  # an invalid or stale measurement is excluded, never silently used.
  # ----------------------------------------------------------------------

  @ Maximum number of units of any one sensor type carried on the measurement
  @ port arrays. The vehicle will fly multiple sun sensors, magnetometers and
  @ IMUs and one or more star trackers, fused by the §8.2 layer; the ports are
  @ arrays from the start so adding units — or the fusion itself — is a topology
  @ change, not a port-interface change. Sized to match the SITL wire bound
  @ (polaris::sitl::kMaxUnits = 8), which SitlBridge static_asserts, so every
  @ unit the sim declares has somewhere to land.
  constant GncMaxUnits = 8

  @ A 3-vector [SI], component order x,y,z.
  array Vec3F64 = [3] F64

  @ A JPL scalar-first quaternion [q0,q1,q2,q3].
  array QuatF64 = [4] F64

  @ Active attitude-estimation mode (design doc §8.1), mirroring
  @ polaris::state::EstimationMode. Ordered by increasing fidelity.
  enum EstimationMode : U8 {
    INVALID = 0 @< no valid solution (cold start / fault / coast expired)
    COARSE = 1 @< sun sensor + magnetometer + IMU coarse attitude (safe/acq)
    FINE = 2 @< MEKF fine solution (nominal; not yet implemented)
  }

  @ One IMU's accumulated inertial increments since the last FSW read (§2.4).
  @ Delta-angle/delta-velocity rather than rate/acceleration is what a real IMU
  @ integrates and reports; the consumer divides by @ intervalSec.
  struct ImuMeas {
    deltaAngleRad: Vec3F64 @< integrated body rate over the interval [rad]
    deltaVelMps: Vec3F64 @< integrated specific force over the interval [m/s]
    intervalSec: F64 @< accumulation interval [s]; positive when valid
    timeTagNs: I64 @< TAI ns of the newest contributing sample
    valid: bool @< the unit reports this sample as usable (§9.1)
  }

  @ One sun sensor's processed unit vector. `sunPresent` is the unit's own
  @ sun-in-field-of-view flag: false in eclipse or when pointed away, which is a
  @ normal condition, not a fault. `sigmaRad` is the realised 1-sigma accuracy at
  @ this incidence (§6.4) — carried so an estimator weights with the noise the
  @ sensor actually had rather than a headline number.
  struct SunSensorMeas {
    dirBody: Vec3F64 @< measured sun direction, body frame [dimensionless]
    sigmaRad: F64 @< realised 1-sigma accuracy [rad]
    timeTagNs: I64 @< TAI ns of the sample
    sunPresent: bool @< the sun is in the unit's field of view
    valid: bool @< the unit reports this sample as usable (§9.1)
  }

  @ One magnetometer's field measurement.
  struct MagnetometerMeas {
    fieldTesla: Vec3F64 @< measured field, body frame [T]
    timeTagNs: I64 @< TAI ns of the sample
    valid: bool @< the unit reports this sample as usable (§9.1)
  }

  @ One GNSS receiver's PVT fix, in the frame and timescale the receiver reports
  @ (§3.2): ECEF metres and GPS time. The consumer applies TAI = GPS + 19 s and
  @ the ECEF->ECI reduction on ingest (REQ-CONV-001) — never the producer.
  struct GnssMeas {
    posEcefM: Vec3F64 @< position, ECEF [m]
    velEcefMps: Vec3F64 @< velocity, ECEF [m/s]
    timeTagGpsNs: I64 @< GPS ns as stamped by the receiver
    valid: bool @< the receiver reports this fix as usable (§9.1)
  }

  @ One star tracker's attitude solution. Defined now so the fine-mode (MEKF)
  @ push adds a consumer rather than a port; nothing reads it yet.
  struct StarTrackerMeas {
    qBodyEci: QuatF64 @< attitude Body <- ECI (JPL scalar-first, q0 >= 0)
    timeTagNs: I64 @< TAI ns of the solution
    valid: bool @< the unit reports a tracking solution (§9.1)
  }

  @ The attitude part of the canonical onboard state (§8.0) as published by the
  @ attitude estimator: what guidance, control and FDIR need every cycle. The
  @ orbit fields of `polaris::state::EstimatedState` are deliberately absent
  @ until the §8.3 orbit filter owns them — publishing zeros for a field nobody
  @ estimates yet is how a consumer ends up trusting one.
  struct AttitudeEstimate {
    epochTaiNs: I64 @< TAI ns this estimate is valid at
    qBodyEci: QuatF64 @< attitude Body <- ECI (JPL scalar-first, q0 >= 0)
    bodyRateRadps: Vec3F64 @< bias-corrected body rate [rad/s]
    attCovDiagRad2: Vec3F64 @< diagonal of the body-frame attitude-error covariance [rad^2]
    ageSec: F64 @< time since the last accepted vector fix [s]
    mode: EstimationMode @< active estimation mode
    attitudeValid: bool @< qBodyEci and attCovDiagRad2 are usable
    rateValid: bool @< bodyRateRadps is usable
  }

  @ IMU increments, sensor source -> estimator.
  port ImuMeasPort(meas: ImuMeas)

  @ Sun-sensor unit vector, sensor source -> estimator.
  port SunSensorMeasPort(meas: SunSensorMeas)

  @ Magnetometer field, sensor source -> estimator.
  port MagnetometerMeasPort(meas: MagnetometerMeas)

  @ GNSS PVT fix, sensor source -> estimator.
  port GnssMeasPort(meas: GnssMeas)

  @ Star-tracker attitude, sensor source -> estimator.
  port StarTrackerMeasPort(meas: StarTrackerMeas)

  @ Attitude estimate, estimator -> guidance/control/FDIR.
  port AttitudeEstimatePort(estimate: AttitudeEstimate)

}

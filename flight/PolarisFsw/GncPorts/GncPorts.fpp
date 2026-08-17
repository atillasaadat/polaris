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

  @ One body-frame 3-vector per unit of a sensor type, flattened: unit i occupies
  @ elements [3i, 3i+2]. Used for the per-unit sun-sensor boresights the §8.2
  @ fusion layer needs (`AttitudeEstimator.SunAlbedoBoresightsBody`).
  @
  @ Flat rather than an array of `Vec3F64` because F´ parameter serialization and
  @ the config compiler encode an array of scalars, not an array of arrays; and
  @ one parameter rather than `GncMaxUnits` separately named ones because that
  @ would be eight chances for them to disagree about which vector they describe.
  @ A slot for a unit that is not installed is written as the zero vector, which
  @ every consumer reads as "no value for this index" — it cannot be mistaken for
  @ a direction.
  array Vec3F64PerUnit = [GncMaxUnits * 3] F64

  @ One scalar per unit of a type, indexed in vehicle build order. Used for the
  @ per-wheel torque commands the §8.5 allocation produces
  @ (`AttitudeController.WheelTorque`). A slot for a unit that is not installed is
  @ written as zero.
  array F64PerUnit = [GncMaxUnits] F64

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
  @ this incidence (§6.2) — carried so an estimator weights with the noise the
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
  @ the ECEF->ECI reduction on ingest (REQ-CONV-001) — never the producer. The
  @ per-fix accuracies are the receiver's own reported figures (§6.2, §8.3): the
  @ orbit filter's R is built from them, in the local horizontal/vertical basis
  @ they are stated in, so a fix that does not report them (zero) is one the
  @ filter refuses rather than one it guesses a covariance for.
  struct GnssMeas {
    posEcefM: Vec3F64 @< position, ECEF [m]
    velEcefMps: Vec3F64 @< velocity, ECEF [m/s]
    timeTagGpsNs: I64 @< GPS ns as stamped by the receiver
    posSigmaHM: F64 @< reported per-axis horizontal position 1-sigma [m]
    posSigmaVM: F64 @< reported vertical (up) position 1-sigma [m]
    velSigmaMps: F64 @< reported per-axis velocity 1-sigma [m/s]
    velValid: bool @< velEcefMps/velSigmaMps carry a solution this fix
    valid: bool @< the receiver reports this fix as usable (§9.1)
  }

  @ The onboard orbit solution (§8.3), orbit estimator -> every consumer that
  @ needs where the vehicle is: the attitude estimator's magnetic and sun
  @ references, guidance, FDIR. ECI (ICRF/J2000) metres and metres per second at
  @ @ epochTaiNs, with the 1-sigma figures the filter's own covariance carries.
  @ `valid` is the filter's coast-horizon verdict (§8.3): false means the
  @ solution was dropped, and a consumer MUST treat position as unavailable
  @ rather than reuse the last vector (§9.1).
  struct OrbitEstimate {
    epochTaiNs: I64 @< TAI ns this solution is valid at
    posEciM: Vec3F64 @< position, ECI [m]
    velEciMps: Vec3F64 @< velocity, ECI [m/s]
    posSigmaM: F64 @< sqrt(trace) of the position covariance [m]
    velSigmaMps: F64 @< sqrt(trace) of the velocity covariance [m/s]
    ageSec: F64 @< time since the last accepted fix [s]
    valid: bool @< the solution is inside the coast horizon and finite
  }

  @ One star tracker's attitude solution. Defined now so the fine-mode (MEKF)
  @ push adds a consumer rather than a port; nothing reads it yet.
  struct StarTrackerMeas {
    qBodyEci: QuatF64 @< attitude Body <- ECI (JPL scalar-first, q0 >= 0)
    timeTagNs: I64 @< TAI ns of the solution
    valid: bool @< the unit reports a tracking solution (§9.1)
  }

  @ One reaction wheel's tachometer reading (design doc §7, §8.5). A wheel drive
  @ reports rotor speed as a matter of course — it is what its own speed loop
  @ closes on — and the §8.5 momentum management runs on it: stored momentum
  @ `h = W (I_w omega_w)` is the basis of the desaturation demand and of the §9
  @ momentum envelope. The **speed** is the measurement; the rotor inertia that
  @ turns it into momentum is a catalog fact carried as an FSW parameter, so a
  @ recalibrated wheel is a parameter change and not a wire change.
  struct WheelSpeedMeas {
    speedRadps: F64 @< rotor speed about the wheel's own spin axis [rad/s]
    timeTagNs: I64 @< TAI ns of the reading
    valid: bool @< the drive reports this reading as usable (§9.1)
  }

  @ The magnetorquer duty-cycle schedule for one control period, published by the
  @ controller that owns it and consumed by every magnetometer consumer (design
  @ doc §7, layers 1-2 of the MTQ/MAG interlock).
  @
  @ It is the *schedule*, not a live actuation flag, because that is what makes
  @ the interlock deterministic: the rods are energised over
  @ [periodStartTaiNs, onWindowEndTaiNs), the field then decays for the rod
  @ model's settle time, and only samples time-tagged inside
  @ [quietStartTaiNs, quietEndTaiNs] are measurements of the geomagnetic field.
  @ A consumer compares its sample's own time tag against that window, so a
  @ scheduling slip shows up as a rejected sample rather than as corrupted data
  @ wearing a valid flag.
  @
  @ Published once per control cycle for the period that cycle is *commanding*,
  @ so a consumer running earlier in the rate group reads the schedule of the
  @ period its current sample was taken in. That one-cycle offset is the correct
  @ pairing, not a staleness bug.
  struct MtqActuation {
    periodStartTaiNs: I64 @< TAI ns the commanded control period begins
    onWindowEndTaiNs: I64 @< TAI ns the rods are de-energised at
    quietStartTaiNs: I64 @< TAI ns the quiet window opens (on-window end + settle time)
    quietEndTaiNs: I64 @< TAI ns the quiet window closes (the period boundary)
    commandedMask: U32 @< bit i set = rod i carried a non-zero dipole this period
    interlockHealthy: bool @< no rod is latched stuck-on; false makes every sample in this period suspect whatever its time tag says
  }

  @ The attitude part of the canonical onboard state (§8.0) as published by the
  @ attitude estimator: what guidance, control and FDIR need every cycle. The
  @ orbit block of `polaris::state::EstimatedState` lives on OrbitEstimate, the
  @ §8.3 orbit estimator's own product; it is not repeated here.
  @
  @ `posEciM` is the **one** orbit field carried, and it is here on purpose: it
  @ is the position the estimator's own magnetic and sun references were
  @ evaluated at this cycle — the OrbitEstimate it consumed, not a second
  @ estimate — so a consumer of this attitude can place it without racing the
  @ orbit estimator's port. The §8.5 gravity-gradient feedforward needs the nadir
  @ direction and nothing else. It carries `posValid`, false whenever the orbit
  @ solution was unavailable, and no velocity: a consumer that wants to
  @ propagate reads OrbitEstimate.
  @
  @ The magnetic block is here rather than on a second port because the estimator
  @ is the vehicle's **one** gate on magnetometer data: it votes the units (§8.2),
  @ applies the hard/soft-iron calibration (§8.1) and enforces the §7 quiet-window
  @ interlock. A controller that read the raw port array instead would be a second
  @ opinion about which magnetometer the vehicle believes, and the two would
  @ disagree on exactly the cycles that matter. `magModelMagnitudeT` is the
  @ onboard IGRF magnitude at the estimated position — **attitude-free**, so a
  @ consumer can compare measured against modelled field strength without the
  @ circularity of judging a sensor through an attitude that sensor helped build.
  @
  @ `magRawMagnitudeT` is the *diagnostic* twin of `magFieldBody` and exists for
  @ one consumer: the §7 stuck-on monitor. It is the largest raw magnitude among
  @ the units the interlock admitted this cycle — **before** the plausibility band
  @ and the vote, and deliberately so. A rod stuck on puts hundreds of microtesla
  @ on the sensor, which the §8.2 magnitude gate rejects as implausible before the
  @ vote ever runs; a monitor reading the voted field would therefore go blind at
  @ exactly the disturbance it exists to name. Being out of band is *evidence* for
  @ this monitor, not a reason to look away. It is not a measurement and no
  @ estimator reads it.
  struct AttitudeEstimate {
    epochTaiNs: I64 @< TAI ns this estimate is valid at
    qBodyEci: QuatF64 @< attitude Body <- ECI (JPL scalar-first, q0 >= 0)
    bodyRateRadps: Vec3F64 @< bias-corrected body rate [rad/s]
    attCovDiagRad2: Vec3F64 @< diagonal of the body-frame attitude-error covariance [rad^2]
    ageSec: F64 @< time since the last accepted vector fix [s]
    magFieldBody: Vec3F64 @< voted, calibration-corrected body-frame field [T]
    magFieldTimeTagNs: I64 @< TAI ns the accepted magnetometer sample was taken at
    magModelMagnitudeT: F64 @< onboard IGRF field magnitude at the estimated position [T]
    magRawMagnitudeT: F64 @< largest raw magnetometer magnitude admitted by the §7 interlock this cycle [T]
    posEciM: Vec3F64 @< spacecraft position, ECI [m] — the §8.3 orbit solution the estimator's references were computed at this cycle
    mode: EstimationMode @< active estimation mode
    attitudeValid: bool @< qBodyEci and attCovDiagRad2 are usable
    rateValid: bool @< bodyRateRadps is usable
    magFieldValid: bool @< magFieldBody/magFieldTimeTagNs are usable this cycle
    magModelValid: bool @< magModelMagnitudeT is usable this cycle
    magRawValid: bool @< magRawMagnitudeT is usable this cycle
    posValid: bool @< posEciM is usable this cycle (the orbit solution was valid)
  }

  @ IMU increments, sensor source -> estimator.
  port ImuMeasPort(meas: ImuMeas)

  @ Sun-sensor unit vector, sensor source -> estimator.
  port SunSensorMeasPort(meas: SunSensorMeas)

  @ Magnetometer field, sensor source -> estimator.
  port MagnetometerMeasPort(meas: MagnetometerMeas)

  @ GNSS PVT fix, sensor source -> orbit estimator.
  port GnssMeasPort(meas: GnssMeas)

  @ Orbit solution, orbit estimator -> attitude estimator / guidance / FDIR.
  port OrbitEstimatePort(estimate: OrbitEstimate)

  @ Star-tracker attitude, sensor source -> estimator.
  port StarTrackerMeasPort(meas: StarTrackerMeas)

  @ Attitude estimate, estimator -> guidance/control/FDIR.
  port AttitudeEstimatePort(estimate: AttitudeEstimate)

  @ Magnetorquer duty-cycle schedule, controller -> magnetometer consumers (§7).
  port MtqActuationPort($state: MtqActuation)

  @ Wheel tachometer, actuator source -> controller (§8.5 momentum management).
  port WheelSpeedMeasPort(meas: WheelSpeedMeas)

}

module flight {

  @ Coarse attitude estimator (design doc §8.1, §10; REQ-ADET-002, REQ-ADET-003,
  @ REQ-ADET-004).
  @
  @ The F´ wrapper around the flight-safe `polaris::gnc::CoarseAttitudeEstimator`
  @ (lib/gnc/coarse_attitude.hpp): sun sensor + magnetometer + gyro, TRIAD-seeded
  @ and gyro-propagated. This is the estimator the §10 Safe-mode floor rests on,
  @ so it is deliberately star-tracker- and table-independent — the sun reference
  @ falls back to the analytic ephemeris (§11.3) and the magnetic reference is
  @ the onboard IGRF-14 snapshot, neither of which needs a working uplink.
  @
  @ Per cycle the component: picks the first valid, fresh unit of each sensor
  @ type off its measurement port arrays; builds the inertial references (Sun
  @ position from OnboardTables, geomagnetic field from the onboard IGRF-14
  @ evaluated at the GNSS position and rotated ECEF->ECI with onboard EOP); runs
  @ one estimator cycle; and publishes the estimate plus its health telemetry.
  @ No math and no I/O live here — the algorithm is in lib/gnc, the field model
  @ in lib/environment, the tables behind the OnboardTables ports.
  @
  @ **Multiple units are the design point.** The measurement inputs are port
  @ arrays sized `GncMaxUnits` because the vehicle will fly several sun sensors,
  @ magnetometers and IMUs and one or more star trackers. This push consumes the
  @ first valid unit of each type — genuine multi-unit fusion is the §8.2 layer,
  @ and the star-tracker input is declared but only counted until fine mode (the
  @ MEKF push). Adding units is then a topology change, not a port change.
  @
  @ **Tuning has no defaults, by design (§19.3).** Every parameter below is
  @ mission configuration; a coarse estimator quietly running on an invented
  @ noise budget reports a covariance that is fiction, and the whole point of
  @ this estimator is that its covariance can be trusted enough to seed the MEKF.
  @ A missing or out-of-range parameter therefore refuses the cycle with a
  @ `ConfigInvalid` warning rather than substituting a value.
  @
  @ Passive: it runs to completion on its rate group's task (the barrier-driven
  @ 10 Hz GNC cycle under SITL, §2.4), calls only synchronous query ports, and
  @ holds no queue. A flight component — it ships to hardware and has no
  @ lib/sitl dependency; only the *source* of its measurements is SITL today.
  passive component AttitudeEstimator {

    # ----------------------------------------------------------------------
    # Rate group
    # ----------------------------------------------------------------------

    @ Estimation cycle entry: one `CoarseAttitudeEstimator::update` per call, at
    @ the GNC rate (10 Hz, §2.4). Measurements must already be latched for this
    @ epoch — under SITL the bridge publishes them before cycling the group.
    sync input port run: Svc.Sched

    # ----------------------------------------------------------------------
    # Sensor measurement inputs (arrays: multi-unit is the design point, §8.2)
    # ----------------------------------------------------------------------

    @ IMU delta-angle/delta-velocity increments, one port per unit in vehicle
    @ build order. Latched on arrival; consumed by the next `run`.
    sync input port imuIn: [GncMaxUnits] ImuMeasPort

    @ Sun-sensor unit vectors, one port per unit in vehicle build order.
    sync input port sunSensorIn: [GncMaxUnits] SunSensorMeasPort

    @ Magnetometer field measurements, one port per unit in vehicle build order.
    sync input port magnetometerIn: [GncMaxUnits] MagnetometerMeasPort

    @ GNSS PVT fixes, one port per unit. The position is what the IGRF reference
    @ is evaluated at; this component does not estimate orbit state (§8.3 owns
    @ that) and does not consume the velocity.
    sync input port gnssIn: [GncMaxUnits] GnssMeasPort

    @ Star-tracker attitude solutions, one port per unit. TODO(fine mode): the
    @ MEKF push consumes these; declared now so that push adds a handler body
    @ rather than reworking the port interface. Counted for health telemetry only
    @ — the solutions themselves are not even stored, and nothing is fused into
    @ the coarse attitude, which must stay tracker-independent to remain the
    @ Safe-mode floor (§10).
    sync input port starTrackerIn: [GncMaxUnits] StarTrackerMeasPort

    # ----------------------------------------------------------------------
    # Reference queries and product output
    # ----------------------------------------------------------------------

    @ Geocentric ECI position [m] of the Sun at the cycle epoch, with its source
    @ grade (precise Chebyshev vs coarse analytic fallback).
    output port getBodyPosition: GetBodyPosition

    @ EOP at the cycle epoch, for the ECEF->ECI rotation of the modelled field
    @ (and of the GNSS position).
    output port getEopAt: GetEopAt

    @ The published attitude estimate (§8.0), for guidance/control/FDIR. Emitted
    @ every cycle the estimator runs, with validity flags set — a consumer gates
    @ on those, never on the mere arrival of the port call.
    output port estimateOut: AttitudeEstimatePort

    # ----------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------

    @ Drop the attitude solution and re-acquire from the next TRIAD (cold start).
    @ Configuration and parameters are retained. Use after a sensor calibration
    @ change, or to force re-acquisition when the solution is suspect.
    sync command RESET_ESTIMATOR

    # ----------------------------------------------------------------------
    # Parameters (mission configuration, §19.3 — no defaults on purpose)
    # ----------------------------------------------------------------------

    @ White (cycle-to-cycle independent) part of the sun-pair 1-sigma transverse
    @ uncertainty [rad]: sensor noise and quantisation. The part repeated fixes
    @ average down. Must be finite and >= 0, and not both zero with the
    @ systematic part.
    param SigmaSunWhiteRad: F64

    @ Systematic part of the sun-pair 1-sigma transverse uncertainty [rad]:
    @ ephemeris error (the analytic fallback is ~0.4 deg) and sensor alignment.
    @ Constant across cycles, so it becomes the covariance floor. May be zero.
    param SigmaSunSysRad: F64

    @ White part of the magnetic-pair 1-sigma transverse uncertainty [rad].
    param SigmaMagWhiteRad: F64

    @ Systematic part of the magnetic-pair 1-sigma transverse uncertainty [rad]:
    @ IGRF model error, hard/soft-iron residual, alignment. May be zero.
    param SigmaMagSysRad: F64

    @ Gyro angle random walk [rad/s^(1/2)]: the square root of the attitude-error
    @ variance accumulated per second of gyro-only propagation.
    param GyroArw: F64

    @ Minimum |sin(angle)| between the sun and field directions for a TRIAD solve
    @ [dimensionless]. Below this the roll about the sun is unobservable.
    param MinSinAngle: F64

    @ Complementary blend gain on the TRIAD-vs-propagated error, in (0, 1]. 1
    @ snaps to TRIAD; smaller trades measurement noise for gyro smoothing.
    param TriadGain: F64

    @ Longest gyro-only coast before the attitude is declared invalid [s].
    param MaxCoastSec: F64

    @ Largest accepted propagation step [s]. A longer gap is a dropout (attitude
    @ held), not an extrapolation on a stale rate.
    param MaxDtSec: F64

    @ Largest measurement age accepted at a cycle [s] (§9.1 staleness gate).
    @ An older sample is excluded, exactly as an invalid one is. Must be positive
    @ and no larger than MaxCoastSec: a staleness window wider than the coast
    @ horizon would keep feeding the estimator data it has already outlived.
    param MaxMeasAgeSec: F64

    @ Smallest geocentric radius accepted from a GNSS fix [m] (§9.1 range gate).
    @ A position below this is not a place this vehicle can be, so the fix is
    @ excluded rather than propagated into the field model and the sun reference.
    param MinPositionRadiusM: F64

    @ Largest geocentric radius accepted from a GNSS fix [m] (§9.1 range gate).
    param MaxPositionRadiusM: F64

    # ----------------------------------------------------------------------
    # Telemetry (health: mode, solution, margins, source quality)
    # ----------------------------------------------------------------------

    @ Active estimation mode (REQ-ADET-004): INVALID until acquisition, COARSE
    @ once a TRIAD has been accepted and the coast horizon has not expired.
    telemetry EstMode: EstimationMode

    @ Attitude Body <- ECI, JPL scalar-first [q0,q1,q2,q3]. Meaningful only when
    @ AttitudeValid is true.
    telemetry AttQuat: QuatF64

    @ Bias-corrected body rate [rad/s]. Meaningful only when RateValid is true.
    telemetry BodyRate: Vec3F64

    @ Trace of the body-frame attitude-error covariance [rad^2] — the one-number
    @ summary of solution quality; grows through eclipse and coast.
    telemetry AttCovTrace: F64

    @ Time since the last accepted TRIAD fix [s]. Compared against MaxCoastSec,
    @ this is the margin before the solution is declared invalid.
    telemetry SolutionAge: F64

    @ TRIAD solutions accepted and blended in since start.
    telemetry TriadAccepted: U32

    @ Cycles where both vector pairs were present but TRIAD refused the solve
    @ (degenerate geometry, or a body/inertial pair disagreeing beyond the
    @ geometry gate) since start.
    telemetry TriadRejected: U32

    @ Estimator cycles refused since start: unconfigured, non-increasing clock,
    @ or a non-finite internal result. A non-zero and rising count is a fault.
    telemetry CyclesRefused: U32

    @ Source grade of the inertial references this cycle: the worse of the Sun
    @ ephemeris and EOP grades served by OnboardTables. COARSE is safe but
    @ degraded (analytic ephemeris / zero-EOP); UNAVAILABLE means no reference.
    telemetry RefGrade: TableGrade

    @ A valid, fresh gyro measurement was found this cycle.
    telemetry GyroValid: bool

    @ A valid, fresh sun measurement with the sun in view, paired with a
    @ reference, was found this cycle.
    telemetry SunValid: bool

    @ A valid, fresh magnetometer measurement paired with a modelled field was
    @ found this cycle.
    telemetry MagValid: bool

    @ A valid, fresh GNSS position was available this cycle. Without one there is
    @ no magnetic reference, so the estimator coasts on the gyro.
    telemetry PositionValid: bool

    @ Star-tracker solutions received since start. Counted for health only — the
    @ coarse mode never fuses them (fine mode is the MEKF push).
    telemetry StarTrackerCount: U32

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ The estimator produced a valid attitude after having none — cold-start
    @ acquisition, or re-acquisition after a coast expiry.
    @ Action: none; informational. Expected once after boot and after each
    @ eclipse long enough to expire the coast horizon.
    event AttitudeAcquired(ageSec: F64, covTraceRad2: F64) \
      severity activity high \
      format "Coarse attitude acquired: age={} s, cov trace={} rad^2"

    @ The attitude was valid and no longer is: the gyro-only coast ran past
    @ MaxCoastSec, or the estimator refused a cycle. Consumers see the validity
    @ flag drop in the same cycle. Edge-gated (emitted on the transition only),
    @ so an eclipse produces one event rather than one per cycle.
    @ Action: expected through a long eclipse or a sun-sensor dropout; if it
    @ repeats outside eclipse, check sun-sensor and magnetometer validity.
    event AttitudeLost(ageSec: F64) \
      severity warning high \
      format "Coarse attitude lost: coasted {} s without a vector fix"

    @ A parameter is missing from ParameterDb or outside its valid range, so the
    @ estimator refuses to run: there are no flight defaults to fall back on
    @ (§19.3). Edge-gated to the transition into the invalid state.
    @ Action: uplink the missing/corrected parameter (PRM_SET + PRM_SAVE), then
    @ RESET_ESTIMATOR. The vehicle has no attitude solution until then.
    event ConfigInvalid(detail: string size 80) \
      severity warning high \
      format "Attitude estimator configuration invalid: {}"

    @ The inertial-reference source grade dropped (typically PRECISE->COARSE when
    @ the uploaded tables stop covering the current time). The coarse fallback is
    @ safe — it is what makes the Safe-mode floor table-independent — but the
    @ systematic error budget it implies is larger. Edge-gated per transition,
    @ following the OnboardTables TableDegraded pattern.
    @ Action: refresh the onboard tables (OnboardTables RELOAD_TABLES) and check
    @ the configured systematic sigmas still bound the coarse budget.
    event ReferenceDegraded(domain: TableDomain, grade: TableGrade) \
      severity warning high \
      format "Attitude reference degraded: domain={} grade={}"

    @ The inertial-reference source grade improved again (e.g. after a successful
    @ table reload). Re-arms the ReferenceDegraded alert.
    event ReferenceRecovered(domain: TableDomain, grade: TableGrade) \
      severity activity high \
      format "Attitude reference recovered: domain={} grade={}"

    @ No valid GNSS position was available, so the geomagnetic reference could
    @ not be evaluated and the magnetic pair was excluded this cycle. With no
    @ magnetic pair there is no TRIAD, so the estimator gyro-coasts. Edge-gated.
    @ Action: check GNSS validity/jamming telemetry. Sustained loss ends in
    @ AttitudeLost once the coast horizon expires; an onboard orbit propagator
    @ (§8.3) is what will remove this dependency.
    event PositionUnavailable \
      severity warning low \
      format "No valid GNSS position: magnetic reference unavailable, coasting"

    @ The onboard IGRF-14 snapshot could not be loaded at setup, so there is no
    @ modelled field and the magnetic pair can never be formed. The estimator
    @ still gyro-propagates and still publishes a body rate (Safe-mode rate
    @ damping needs it), but it cannot acquire attitude.
    @ Action: uplink a valid IAGA coefficient file and restart, or command
    @ RESET_ESTIMATOR after the file is in place.
    event IgrfLoadFailed(reason: string size 80) \
      severity warning high \
      format "Onboard IGRF load failed: {}"

    @ The current epoch has run past the horizon the loaded IGRF snapshot's
    @ linear model was published for (the next tabulated IAGA epoch, or five
    @ years past the last one). The magnetic reference is refused (excluded, not
    @ degraded) so the estimator coasts instead of blending an extrapolated field
    @ nobody validated.
    @ Edge-gated. Action: uplink a fresh IAGA snapshot for the current epoch.
    event MagneticReferenceStale(cycleYear: F64, validUntilYear: F64) \
      severity warning high \
      format "Onboard IGRF snapshot expired: cycle epoch {} yr, valid until {} yr"

    @ The onboard IGRF-14 snapshot loaded and the magnetic reference is available.
    event IgrfLoaded(epochYear: F64, validUntilYear: F64, degree: U32) \
      severity activity high \
      format "Onboard IGRF loaded: epoch {} yr, valid until {} yr, degree {}"

    @ The estimator was reset by operator command; the solution is dropped and
    @ re-acquires from the next TRIAD.
    event EstimatorReset \
      severity activity high \
      format "Attitude estimator reset: solution dropped, awaiting re-acquisition"

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    @ Port for requesting the current time (TAI under SITL, via SitlTime).
    time get port timeCaller

    @ Parameter get port
    param get port prmGetOut

    @ Parameter set port
    param set port prmSetOut

    @ Command registration port
    command reg port cmdRegOut

    @ Command receive port
    command recv port cmdIn

    @ Command response port
    command resp port cmdResponseOut

    @ Event port
    event port logOut

    @ Text event port
    text event port logTextOut

    @ Telemetry port
    telemetry port tlmOut

  }

}

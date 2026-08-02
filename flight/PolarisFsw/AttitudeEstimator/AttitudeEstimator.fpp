module flight {

  @ Why a fine solution was given up and the published product fell back to the
  @ coarse chain (design doc §8.1 arbitration, §9; REQ-ADET-004). Carried on
  @ FineModeDemoted so FDIR can tell a geometry/outage fallback apart from a
  @ measurement stream the filter no longer believes.
  enum FineDemotionReason : U8 {
    REFUSAL_STREAK = 0 @< the filter refused propagate/update on N consecutive cycles
    NIS_STREAK = 1 @< the NIS gate rejected a measurement on N consecutive cycles
    COAST = 2 @< no accepted update inside the fine coast horizon
    COMMANDED = 3 @< RESET_ESTIMATOR, or a tuning change that rebuilt the filter
    FILTER_FAULT = 4 @< a non-finite internal result dropped the filter itself
  }

  @ Attitude estimator — coarse chain plus MEKF fine mode with arbitration
  @ (design doc §8.1, §10; REQ-ADET-002, REQ-ADET-003, REQ-ADET-004).
  @
  @ The F´ wrapper around the two flight-safe estimators in lib/gnc. The
  @ **coarse chain** (`polaris::gnc::CoarseAttitudeEstimator`) is sun sensor +
  @ magnetometer + gyro, TRIAD-seeded and gyro-propagated: the estimator the §10
  @ Safe-mode floor rests on, deliberately star-tracker- and table-independent —
  @ the sun reference falls back to the analytic ephemeris (§11.3) and the
  @ magnetic reference is the onboard IGRF-14 snapshot, neither of which needs a
  @ working uplink. The **fine mode** (`polaris::gnc::Mekf`) is the 6-state
  @ multiplicative EKF that estimates gyro bias alongside attitude.
  @
  @ Per cycle the component: picks the first valid, fresh unit of each sensor
  @ type off its measurement port arrays; builds the inertial references (Sun
  @ position from OnboardTables, geomagnetic field from the onboard IGRF-14
  @ evaluated at the GNSS position and rotated ECEF->ECI with onboard EOP); runs
  @ one coarse cycle; arbitrates fine vs coarse; and publishes whichever solution
  @ is active plus its health telemetry. No math and no I/O live here — the
  @ algorithms are in lib/gnc, the field model in lib/environment, the tables
  @ behind the OnboardTables ports.
  @
  @ **Arbitration (REQ-ADET-004).** The coarse chain runs **every** cycle,
  @ whether or not fine mode is engaged, so the fallback is always a live
  @ solution rather than one that has to re-acquire from cold. Cold start
  @ acquires coarse; once the coarse attitude is valid and both vector pairs are
  @ present, a Davenport q-method solve over that cycle's pairs seeds the MEKF
  @ (attitude and its covariance from the solve, bias zero at the configured
  @ turn-on sigma) and fine mode engages. Fine mode is given up — demoted, with
  @ FineModeDemoted naming the reason — on a refusal streak, an NIS rejection
  @ streak, the fine coast horizon, an internal filter fault, or command. A
  @ demotion **drops the filter state**: a solution no longer trusted is not
  @ worth carrying, so re-promotion goes through a fresh Davenport seed, never a
  @ resumed filter. Demotion and promotion are one cycle apart at the earliest,
  @ so an oscillating condition is visible in the event stream rather than hidden
  @ inside one cycle.
  @
  @ **Multiple units are the design point.** The measurement inputs are port
  @ arrays sized `GncMaxUnits` because the vehicle will fly several sun sensors,
  @ magnetometers and IMUs and one or more star trackers. This push consumes the
  @ first valid unit of each type — genuine multi-unit fusion is the §8.2 layer,
  @ and the star-tracker input is declared but only counted until that layer
  @ lands. Adding units is then a topology change, not a port change.
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

    @ Star-tracker attitude solutions, one port per unit. TODO(§8.2): the fusion
    @ layer feeds these to the MEKF as a third measurement source; declared now
    @ so that push adds a handler body rather than reworking the port interface.
    @ Counted for health telemetry only — the solutions themselves are not even
    @ stored, and nothing is fused into the coarse attitude, which must stay
    @ tracker-independent to remain the Safe-mode floor (§10).
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

    @ Drop **both** solutions — the coarse chain and the MEKF — and re-acquire
    @ from cold: the next TRIAD for coarse, then a fresh Davenport seed for fine.
    @ Every edge-gated alert is re-armed and the tuning re-read. Configuration
    @ and parameters are retained. Use after a sensor calibration change, or to
    @ force re-acquisition when the solution is suspect.
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
    # Fine-mode (MEKF) parameters — a second, independent validity gate
    # ----------------------------------------------------------------------
    #
    # These are validated separately from the coarse set above. A coarse
    # parameter missing leaves the vehicle with no attitude at all; a *fine*
    # parameter missing leaves it with the Safe-mode floor, which is a working
    # vehicle. So the fine set failing emits FineConfigInvalid and the component
    # runs coarse-only, rather than refusing every cycle.
    #
    # The MEKF's angle random walk is `GyroArw` above: it is the same gyro and
    # the same parametrisation, and two parameters for one physical quantity is
    # two chances to disagree.

    @ Gyro **rate** random walk [rad*s^(-3/2)]: the bias standard deviation
    @ accumulated per sqrt(second). Convert a datasheet bias instability (deg/hr,
    @ Allan floor) and its correlation time to the equivalent random walk before
    @ setting this. May be zero — a bias modelled as exactly constant, which
    @ real hardware rarely is.
    param MekfRrw: F64

    @ NIS rejection threshold [dimensionless] for one 3-vector update. A
    @ chi-square quantile on **2** degrees of freedom, not 3: the innovation
    @ between two unit vectors is transverse by construction (chi2_2 at 99.9% =
    @ 13.82). Must be positive.
    param MekfNisGate: F64

    @ Longest interval without an accepted fine update before the fine solution
    @ is given up and the published product falls back to coarse [s]. Distinct
    @ from MaxCoastSec: the coarse horizon asks "is this solution still usable?",
    @ this one asks "is the filter still better than the floor underneath it?".
    param MekfMaxCoastSec: F64

    @ 1-sigma of the initial gyro-bias estimate [rad/s], per axis. Size it to the
    @ hardware's turn-on bias repeatability — the seed bias itself is zero, so
    @ this is the whole of what the filter is told about the bias at cold start.
    @ Must be positive: a zero would tell the filter the bias is known exactly
    @ and it would never learn one.
    param MekfBiasSigmaInit: F64

    @ Consecutive cycles on which the filter refused a propagate or an update
    @ (malformed input, not a gate rejection) before fine mode is demoted. Must
    @ be positive.
    param MekfRefusalStreak: U32

    @ Consecutive cycles on which the NIS gate rejected a measurement before fine
    @ mode is demoted. One rejection is a chi-square tail; a streak is a
    @ measurement stream the filter no longer believes. Must be positive.
    param MekfNisStreak: U32

    @ Observability gate for the Davenport seed [dimensionless], in (0, 1): the
    @ ratio lambda_min/lambda_max of the observation set's Fisher information
    @ matrix, required in both the body and the reference frame. Scale-free, so
    @ it gates geometry rather than noise level; two equally-weighted
    @ observations separated by theta give (1 - cos theta)/2 = sin^2(theta/2),
    @ i.e. 0.5 at 90 degrees and 0.0076 at 10 degrees.
    param SeedMinObservability: F64

    # ----------------------------------------------------------------------
    # Telemetry (health: mode, solution, margins, source quality)
    # ----------------------------------------------------------------------

    @ Active estimation mode (REQ-ADET-004): INVALID until acquisition, COARSE
    @ once a TRIAD has been accepted and the coast horizon has not expired, FINE
    @ while the MEKF is engaged and inside its own coast horizon.
    telemetry EstMode: EstimationMode

    @ Attitude Body <- ECI, JPL scalar-first [q0,q1,q2,q3]. Meaningful only when
    @ AttitudeValid is true.
    telemetry AttQuat: QuatF64

    @ Bias-corrected body rate [rad/s]. Meaningful only when RateValid is true.
    telemetry BodyRate: Vec3F64

    @ Trace of the body-frame attitude-error covariance [rad^2] of the **active**
    @ solution — the one-number summary of solution quality; grows through
    @ eclipse and coast. NaN when there is no valid attitude.
    telemetry AttCovTrace: F64

    @ Time since the last accepted fix [s] of the active solution: the last TRIAD
    @ in coarse mode, the last accepted MEKF update in fine mode. Compared
    @ against MaxCoastSec / MekfMaxCoastSec, this is the margin before the
    @ solution is given up.
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

    @ Star-tracker solutions received since start. Counted for health only —
    @ nothing fuses them until the §8.2 layer.
    telemetry StarTrackerCount: U32

    @ **Largest** NIS of this cycle's fine-mode updates [dimensionless], accepted
    @ or rejected. ~chi2_2 when the filter is consistent, so a channel that sits
    @ well above 2 is the covariance-consistency diagnostic of REQ-ADET-004. The
    @ maximum rather than the last, because a rejection is above the gate and an
    @ accepted update below it — reporting the last would let a good magnetic
    @ update hide the sun outlier that preceded it. NaN on a cycle with no fine
    @ update.
    telemetry MekfNis: F64

    @ Fine-mode measurements rejected by the NIS gate by the **currently seeded**
    @ filter. Resets to zero at every demotion, since the filter it counted for
    @ is gone; MekfRejectedTotal is the one to watch a trend on.
    telemetry MekfRejected: U32

    @ Fine-mode NIS-gate rejections since the last commanded reset, across every
    @ filter this component has seeded. A demotion clears the filter's own count
    @ (Mekf::reset treats that as the operator saying "start over"), which would
    @ erase the FDIR signal at the moment a NIS_STREAK demotion created it — so
    @ this total carries across demotions and only RESET_ESTIMATOR clears it.
    @ A rising count is the FDIR signal, not a single rejection.
    telemetry MekfRejectedTotal: U32

    @ Estimated gyro bias [rad/s], body axes. Zero while fine mode is not
    @ engaged — the coarse chain does not estimate bias, and reporting anything
    @ else would be inventing one.
    telemetry GyroBias: Vec3F64

    @ Trace of the MEKF's attitude-error covariance block [rad^2]. Telemetered
    @ next to AttCovTrace so the fine solution's uncertainty can be compared
    @ against the coarse floor directly. NaN while fine mode is not engaged.
    telemetry FineAttCovTrace: F64

    @ Trace of the MEKF's gyro-bias covariance block [rad^2/s^2]: how well the
    @ bias is known. Falls from MekfBiasSigmaInit^2 * 3 as the filter converges.
    @ NaN while fine mode is not engaged.
    telemetry BiasCovTrace: F64

    @ Fine->coarse demotions since start. Non-zero is not itself a fault — an
    @ eclipse with no magnetic reference will do it — but a rising count is a
    @ filter that cannot hold onto a solution.
    telemetry FineDemotions: U32

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ The estimator produced a valid attitude after having none — cold-start
    @ acquisition, or re-acquisition after a coast expiry. Keyed on the
    @ **published** solution, so a fine->coarse demotion with a live coarse
    @ solution underneath does not produce one (nothing was lost).
    @ Action: none; informational. Expected once after boot and after each
    @ eclipse long enough to expire the coast horizon.
    event AttitudeAcquired(ageSec: F64, covTraceRad2: F64) \
      severity activity high \
      format "Attitude acquired: age={} s, cov trace={} rad^2"

    @ The attitude was valid and no longer is: the gyro-only coast ran past
    @ MaxCoastSec, or the estimator refused a cycle. Consumers see the validity
    @ flag drop in the same cycle. Edge-gated (emitted on the transition only),
    @ so an eclipse produces one event rather than one per cycle.
    @ Action: expected through a long eclipse or a sun-sensor dropout; if it
    @ repeats outside eclipse, check sun-sensor and magnetometer validity.
    event AttitudeLost(ageSec: F64) \
      severity warning high \
      format "Attitude lost: coasted {} s without a vector fix"

    @ Fine mode engaged: a Davenport solve over this cycle's vector pairs seeded
    @ the MEKF and the published solution is now the filter's. The reported trace
    @ is the **seed** covariance, i.e. the single-frame solution's, which the
    @ filter converges below as it folds in further measurements.
    @ Action: none; informational, and the mode transition REQ-ADET-004 requires
    @ be surfaced to FDIR.
    event FineModeEngaged(seedCovTraceRad2: F64) \
      severity activity high \
      format "Fine mode engaged: MEKF seeded from Davenport, seed cov trace={} rad^2"

    @ Fine mode given up; the published solution falls back to the coarse chain,
    @ which has been running underneath all along. The filter state is dropped —
    @ re-promotion goes through a fresh Davenport seed, never a resumed filter.
    @ Action: depends on the reason. COAST is expected on a long dual-source
    @ outage and clears itself. NIS_STREAK means the measurements and the filter
    @ disagree persistently — check sensor calibration and the configured sigmas.
    @ REFUSAL_STREAK or FILTER_FAULT is a software/numerics fault: capture the
    @ telemetry and keep the vehicle on the coarse floor.
    event FineModeDemoted(reason: FineDemotionReason, ageSec: F64) \
      severity warning high \
      format "Fine mode demoted to coarse: reason={}, age={} s"

    @ A fine-mode promotion was attempted and refused: the Davenport solve did
    @ not clear the observability gate, or the MEKF rejected the seed (a
    @ covariance that is not positive-definite, a non-finite value). The coarse
    @ solution is unaffected. Edge-gated to the first failure of a run of them,
    @ so degenerate sun/field geometry does not warn at 10 Hz.
    @ Action: none if it clears within an orbit — near-parallel sun and field
    @ directions are a normal flight condition. Persistent failure with good
    @ geometry means the configured seed sigmas or SeedMinObservability are wrong.
    event FineInitFailed(detail: string size 80) \
      severity warning low \
      format "Fine mode could not be seeded: {}"

    @ A fine-mode parameter is missing from ParameterDb or outside its valid
    @ range. Unlike ConfigInvalid this is **not** fatal: the coarse chain still
    @ runs and the vehicle keeps the §10 Safe-mode floor, it just never promotes
    @ to fine. Edge-gated to the transition into the invalid state.
    @ Action: uplink the missing/corrected parameter (PRM_SET + PRM_SAVE). The
    @ vehicle is flyable meanwhile, on a coarse attitude.
    event FineConfigInvalid(detail: string size 80) \
      severity warning high \
      format "Fine mode configuration invalid, running coarse-only: {}"

    @ A **coarse-chain** parameter is missing from ParameterDb or outside its
    @ valid range, so the estimator refuses to run at all: there are no flight
    @ defaults to fall back on (§19.3), and without the coarse chain there is no
    @ floor for fine mode to fall back to either. The fine-mode set has its own,
    @ non-fatal gate — see FineConfigInvalid. Edge-gated to the transition into
    @ the invalid state.
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

    @ The estimator was reset by operator command; both solutions are dropped and
    @ the coarse chain re-acquires from the next TRIAD, fine mode from the
    @ Davenport seed after it.
    event EstimatorReset \
      severity activity high \
      format "Attitude estimator reset: solutions dropped, awaiting re-acquisition"

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

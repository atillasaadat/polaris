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

  @ Where the commanded magnetometer calibration is in its lifecycle (design doc
  @ §8.1). Derived state, not a stored one: COLLECTING while a window is open,
  @ else APPLIED while a solved calibration is correcting the field measurement,
  @ else IDLE. COLLECTING wins because a window opened over an already-applied
  @ calibration is the interesting condition, and the fit itself always runs on
  @ **raw** samples regardless — the applied correction never feeds its own refit.
  enum MagCalState : U8 {
    IDLE = 0 @< no window open, no calibration applied: the magnetometer runs raw
    COLLECTING = 1 @< a MAG_CAL_START window is accumulating samples
    APPLIED = 2 @< a solved calibration is correcting every magnetometer consumer
  }

  @ Which gate refused a magnetometer calibration fit (design doc §8.1). Carried
  @ on MagCalRejected so the ground knows whether to re-fly the window and how.
  enum MagCalRejectReason : U8 {
    SAMPLES = 0 @< fewer samples were accepted than MagCalMinSamples — collect longer, or the window ran out of cycles waiting for a stalled sensor
    COVERAGE = 1 @< the sampled field directions span too narrow a cone — tumble further
    CONDITION = 2 @< the normal matrix is worse conditioned than MagCalMaxCondition
    NO_IMPROVEMENT = 3 @< the fit does not beat the raw data by MagCalMinImprovement
    NUMERICAL = 4 @< a decomposition failed or the fitted ellipsoid was unusable
    CONFIG = 5 @< the calibration parameters are missing or out of range in ParameterDb
  }

  @ Why one IMU is not contributing to the voted body rate (design doc §8.2,
  @ §9.2; REQ-ADET-008, REQ-ADET-009). Mirrors `polaris::gnc::ImuVoteReason`.
  @ Carried on ImuUnitExcluded so the ground knows whether to expect recovery:
  @ a rate-limit trip is a unit that reported something the vehicle cannot be
  @ doing, a non-finite reading is a broken data path, and OUTVOTED is a
  @ two-unit disagreement the filter's own propagated rate attributed.
  enum ImuExclusionReason : U8 {
    NOT_FINITE = 0 @< NaN or Inf in the reported rate
    RATE_LIMIT = 1 @< magnitude above ImuMaxRateRadps — not a rate this vehicle can be at
    OUTVOTED = 2 @< the disagreeing unit of a pair, identified by the MEKF propagated rate
  }

  @ Why one magnetometer is not contributing to the voted field (design doc §8.2,
  @ §9.2; REQ-ADET-011). Mirrors `polaris::gnc::MagVoteReason`. A separate
  @ enumeration from ImuExclusionReason because the middle gate is a different
  @ physical test and an operator reading "RATE_LIMIT" on a magnetometer would be
  @ reading a lie: the magnetometer's plausibility gate is its magnitude against
  @ the onboard IGRF model, which tracks the field over an orbit instead of
  @ admitting everything below saturation.
  enum MagExclusionReason : U8 {
    NOT_FINITE = 0 @< NaN or Inf in the reported field
    FIELD_MAGNITUDE = 1 @< |m| outside [MagMinFieldRatio, MagMaxFieldRatio] x |B_IGRF|
    OUTVOTED = 2 @< the disagreeing unit of a pair, identified by the modelled field
  }

  @ Which measurement source the fine (MEKF) solution is currently being updated
  @ from — the §8.2 mode ladder, telemetered because "fine mode" no longer means
  @ one thing (design doc §8.1, §8.2; REQ-ADET-004, REQ-ADET-012).
  @
  @ The ladder is ST+IMU, then SS+MAG+IMU, then the coarse TRIAD chain, and the
  @ **finest rung fuses star trackers only**: with at least one valid tracker the
  @ sun and magnetic pairs are not folded into the filter at all. They are two
  @ orders of magnitude wider than a tracker, so folding them in can only pull the
  @ solution away from it, and the filter's white-R model has no way to represent
  @ the systematic floor that makes them wide. They are not discarded either —
  @ they demote to FDIR-monitored residuals against the tracker solution
  @ (ResidualMonitorAlert), which is a strictly better use of them: a sun sensor
  @ that has drifted is now *observable* instead of merely down-weighted.
  enum FineSource : U8 {
    NONE = 0 @< fine mode is not engaged; the coarse chain is the published product
    SUN_MAG = 1 @< MEKF updated from the sun and magnetic vector pairs (no tracker available)
    STAR_TRACKER = 2 @< MEKF updated from one or more star trackers; SS/MAG are monitors only
  }

  @ Which cross-check raised a ResidualMonitorAlert (design doc §8.2, §9.2;
  @ REQ-ADET-012). Each is a measurement the fine solution is **not** using this
  @ cycle, compared against the solution to see whether it still agrees.
  enum ResidualMonitor : U8 {
    SUN = 0 @< selected sun sensor vs the tracker-derived sun direction
    MAGNETOMETER = 1 @< voted field vs the tracker-derived modelled field
    SUN_CROSS_UNIT = 2 @< the selected sun sensor vs the runner-up unit of the suite
  }

  @ Where the commanded inter-star-tracker alignment calibration is in its
  @ lifecycle (design doc §8.2; REQ-ADET-013). Derived state, not a stored one,
  @ exactly as MagCalState is: COLLECTING while a window is open, else APPLIED
  @ while at least one non-king tracker carries a fitted correction, else IDLE.
  enum StAlignState : U8 {
    IDLE = 0 @< no window open, no alignment applied: every tracker reads as mounted
    COLLECTING = 1 @< a ST_ALIGN_CAL_START window is accumulating simultaneous pairs
    APPLIED = 2 @< a fitted correction is being applied to at least one tracker
  }

  @ Which gate refused an inter-tracker alignment fit (design doc §8.2). Carried
  @ on StAlignRejected so the ground knows whether to re-fly the window, suspect a
  @ unit, or fix the command.
  enum StAlignRejectReason : U8 {
    SAMPLES = 0 @< fewer simultaneous pairs than StAlignMinSamples — collect longer, or a tracker kept dropping out
    DEGENERATE = 1 @< the pairs do not share one fixed rotation (eigen-gap below StAlignMinEigenGap) — suspect a unit
    DISPERSION = 2 @< the fitted residual exceeds StAlignMaxResidualRad
    NUMERICAL = 3 @< the eigensolve failed or the average was unusable
    CONFIG = 4 @< the alignment parameters are missing or out of range in ParameterDb
    UNIT = 5 @< the commanded unit index is the king tracker, out of range, or has no configured boresight
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
  @ evaluated at the §8.3 orbit solution and rotated ECEF->ECI with onboard
  @ EOP); runs
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
  @ **Magnetometer calibration is commanded here (§8.1).** The component owns one
  @ streaming ellipsoid accumulator (`polaris::gnc::MagCalibrationAccumulator`)
  @ and one applied calibration. MAG_CAL_START opens a bounded window; each cycle
  @ that has both a selected magnetometer reading and a modelled IGRF field feeds
  @ the **raw** reading and that field's magnitude to the accumulator; at the
  @ target the fit runs, and on acceptance the correction is applied at a single
  @ point in the magnetometer path — between unit selection and every consumer,
  @ so the coarse chain, the MEKF and the future B-dot cannot disagree about what
  @ the field was. Collection never disturbs the running estimator: it is a tap,
  @ the fit is on raw data, and the published attitude changes only when a
  @ calibration is applied or cleared.
  @
  @ **Multiple units are combined, not merely accepted (§8.2).** The measurement
  @ inputs are port arrays sized `GncMaxUnits`, and each type has its own
  @ combination rule, chosen by what redundancy is worth on that sensor:
  @
  @  - **IMUs: a fault-tolerant vote** (`polaris::gnc::ImuVoter`). Per-unit
  @    plausibility gates (finiteness, rate magnitude against a configured
  @    physical vehicle limit, staleness), then a per-axis **median** at three or
  @    more units, or — the branch the two-IMU reference vehicle flies —
  @    pairwise-disagreement detection with the MEKF's propagated rate
  @    identifying the offender. Averaging is forbidden: its breakdown point is
  @    zero, so one railed unit drags the combined rate without limit. An
  @    excluded unit is an FDIR event with a latch and an automatic re-admission
  @    policy.
  @  - **Sun sensors: validity-gated selection of the best-illuminated unit** —
  @    the valid, fresh, sun-in-view unit reporting the smallest realised sigma,
  @    which is the incidence-cosine criterion expressed in the quantity the
  @    estimator actually consumes. Selection rather than a weighted combination
  @    because the units' errors are dominated by a *shared* systematic (albedo,
  @    ephemeris) that combining cannot average down, so a second unit at a worse
  @    incidence buys noise reduction on the small term and nothing on the large
  @    one. Ties break to the lowest index, so the choice is deterministic.
  @  - **Magnetometers: voted** (§8.2, two units on the reference vehicle);
  @    position is not selected here at all — it arrives once per cycle from
  @    the OrbitEstimator, which owns the receiver (§8.3).
  @  - **Star trackers: counted only.** The tracker joins the MEKF in the next
  @    §8.2 push; the coarse chain must stay tracker-independent to remain the
  @    Safe-mode floor (§10).
  @
  @ Adding a unit is a topology line plus a vehicle-config entry — no port or
  @ component change.
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
  @
  @ **`run` and every command are `guarded`, not `sync`, and both halves are
  @ required.** A passive component borrows its caller's thread, and the callers
  @ are two different threads: the rate group drives `run`, while the command
  @ dispatcher drives the command handlers. Those handlers write exactly the
  @ state `run` reads — RESET_ESTIMATOR rebuilds the estimators, MAG_CAL_CLEAR
  @ assigns the ~100-byte applied calibration, MAG_CAL_START resets the
  @ accumulator mid rank-1 update. A torn read there is the worst kind of fault
  @ this component can have, because the halves are individually *finite*: an
  @ identity soft-iron matrix against a stale hard-iron offset passes every
  @ finiteness gate and flows into both estimator chains as a measurement. F´
  @ serialises guarded ports on one component mutex, which costs one uncontended
  @ lock per 10 Hz cycle. Guarding the commands alone would be worse than
  @ guarding neither, since it would look like the problem was handled.
  @ The measurement input ports stay `sync`, as they were before this push: the
  @ SITL bridge publishes them from the same thread that then cycles the rate
  @ group. A hardware deployment that services sensors on a driver thread would
  @ have to guard those too, which is a topology-time decision for the push that
  @ brings the `Drv` layer up.
  passive component AttitudeEstimator {

    # ----------------------------------------------------------------------
    # Rate group
    # ----------------------------------------------------------------------

    @ Estimation cycle entry: one `CoarseAttitudeEstimator::update` per call, at
    @ the GNC rate (10 Hz, §2.4). Measurements must already be latched for this
    @ epoch — under SITL the bridge publishes them before cycling the group.
    guarded input port run: Svc.Sched

    # ----------------------------------------------------------------------
    # Sensor measurement inputs (arrays: multi-unit is the design point, §8.2)
    # ----------------------------------------------------------------------

    @ Sensor inputs are `guarded`, matching `run`: a `guarded` handler takes the
    @ component mutex and a `sync` one does not, so mixing them would leave the
    @ guard protecting nothing against a sensor push from another thread. Under
    @ SITL the pushes and `run` share the bridge's task and the mutex is
    @ uncontended; on the hardware topology (§2.4) the drivers get their own
    @ threads and this is what keeps the latch-then-consume handoff sound. The
    @ handlers only latch a few hundred bytes, so the cost is negligible.
    @
    @ IMU delta-angle/delta-velocity increments, one port per unit in vehicle
    @ build order. Latched on arrival; consumed by the next `run`.
    guarded input port imuIn: [GncMaxUnits] ImuMeasPort

    @ Sun-sensor unit vectors, one port per unit in vehicle build order.
    guarded input port sunSensorIn: [GncMaxUnits] SunSensorMeasPort

    @ Magnetometer field measurements, one port per unit in vehicle build order.
    guarded input port magnetometerIn: [GncMaxUnits] MagnetometerMeasPort

    @ The onboard orbit solution (§8.3), published by the OrbitEstimator earlier
    @ in the same rate-group cycle. Its ECI position is what the IGRF reference
    @ and the sun parallax are evaluated at; this component does not see the
    @ receiver and does not consume the velocity. Guarded like the sensor ports:
    @ the producer runs on the rate-group thread today, but the port contract
    @ must not depend on that.
    guarded input port orbitStateIn: OrbitEstimatePort

    @ Star-tracker attitude solutions, one port per unit — the finest rung of the
    @ §8.2 mode ladder, fused into the MEKF as **attitude** measurements
    @ (`Mekf::updateAttitude`, `H = [I 0]`) rather than as vector pairs.
    @
    @ One unit — StKingUnit — is the **king**: its mounting defines the body frame,
    @ so its reading is taken as the frame itself and no alignment is ever
    @ estimated for it. Every other unit is stated in the king's frame by the
    @ correction from the commanded ST_ALIGN_CAL flow before it reaches the filter.
    @ Two trackers on a structure observe only their *relative* rotation, so
    @ naming a king removes an unobservable degree of freedom rather than hiding
    @ one (see lib/gnc/st_alignment.hpp).
    @
    @ Nothing here is fused into the **coarse** attitude, which stays
    @ tracker-independent to remain the Safe-mode floor (§10).
    guarded input port starTrackerIn: [GncMaxUnits] StarTrackerMeasPort

    @ The magnetorquer duty-cycle schedule from the controller that owns it
    @ (design doc §7, interlock layer 2). A magnetometer sample whose time tag
    @ does not fall inside the published quiet window is **not a measurement of
    @ the geomagnetic field** — an energised rod puts a near-field on the sensor
    @ far above the ~30 uT ambient, and the core's hysteresis outlives the drive —
    @ so it is excluded here, at the vehicle's single gate on magnetometer data,
    @ and therefore reaches neither the estimators nor the §8.1 calibration
    @ accumulator.
    @
    @ Written on the rate-group thread by a member that runs *after* this one, so
    @ each cycle reads the schedule of the period its sample was taken in — the
    @ correct pairing (see `MtqActuation`). A topology with no magnetorquer
    @ control never calls it, and the estimator then admits every sample: the
    @ absence of a schedule means no rod has ever been commanded, which is the
    @ honest reading of a vehicle whose rods nothing drives.
    guarded input port mtqActuationIn: MtqActuationPort

    # ----------------------------------------------------------------------
    # Reference queries and product output
    # ----------------------------------------------------------------------

    @ Geocentric ECI position [m] of the Sun at the cycle epoch, with its source
    @ grade (precise Chebyshev vs coarse analytic fallback).
    output port getBodyPosition: GetBodyPosition

    @ EOP at the cycle epoch, for the ECEF->ECI rotation of the modelled field.
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
    @
    @ **Full reset semantics include the magnetometer calibration**: a collection
    @ window in progress is aborted and an applied calibration is cleared, so the
    @ magnetometer goes back to raw. That is deliberate — the command exists for
    @ "the solution is suspect, start over", and a calibration derived from data
    @ the operator no longer trusts is part of what is suspect. Re-calibrating is
    @ MAG_CAL_START; clearing a calibration alone (keeping the solutions) is
    @ MAG_CAL_CLEAR.
    guarded command RESET_ESTIMATOR

    @ Open a magnetometer hard/soft-iron calibration window of @p sampleCount
    @ **accepted** samples (design doc §8.1). The estimator keeps running
    @ normally throughout: collection is a tap on the magnetometer path, not a
    @ mode, and the published attitude is unaffected until the fit is applied.
    @
    @ **Samples, not seconds, on purpose.** Every gate the fit applies is in
    @ samples, and a sample only enters when that cycle had a valid magnetometer
    @ reading *and* a modelled IGRF field to compare it against — so a position
    @ outage (the orbit solution dropped, §8.3), a sensor dropout, or a §7
    @ magnetorquer-on window all cost samples without costing wall-clock. A
    @ duration would let the ground command a window that quietly collected a
    @ tenth of the data it asked for; a sample count says what the fit will
    @ actually be fitted on. Wall-clock is then bounded by the operator, who can
    @ MAG_CAL_ABORT at any time.
    @
    @ The window also closes itself after ten cycles per sample asked for, so a
    @ loss of magnetometer or position mid-collection cannot leave the vehicle
    @ telemetering COLLECTING forever waiting on a completion that can no longer
    @ arrive. The fit is attempted on whatever was collected: enough and it
    @ succeeds, too little and MagCalRejected(SAMPLES) is the honest report.
    @
    @ Rejected (EXECUTION_ERROR, no window opened) when the calibration
    @ parameters are missing or out of range — MagCalRejected(CONFIG) — or when
    @ @p sampleCount is below MagCalMinSamples or above the component's fixed
    @ sample-count ceiling (AttitudeEstimator::kMaxCalSamples).
    @ Starting a window while one is open restarts it: the accumulator is reset
    @ and the new target takes effect.
    guarded command MAG_CAL_START(
                                sampleCount: U32 @< accepted samples to collect before fitting
                              )

    @ Close a collection window without fitting: the accumulator is discarded and
    @ any previously applied calibration is left exactly as it was. A no-op (with
    @ OK) when no window is open, so an abort is always safe to send.
    guarded command MAG_CAL_ABORT

    @ Drop the applied magnetometer calibration; every consumer reverts to the
    @ raw field measurement. Leaves a collection window in progress alone —
    @ clearing the applied correction and re-fitting are separate decisions, and
    @ the fit runs on raw samples either way. A no-op (with OK) when no
    @ calibration is applied.
    @
    @ **The applied calibration does not survive a reboot.** It lives in
    @ component state only; nothing is written to ParameterDb. That is the
    @ deliberate deferral recorded in §8.1 — persistence rides on the §23.6
    @ non-volatile-state work — so an operator sees MagCalCleared on command and
    @ simply no MagCalComplete after a reset, and the vehicle flies uncalibrated
    @ until the window is re-flown.
    guarded command MAG_CAL_CLEAR

    @ Open an inter-star-tracker alignment calibration window on @p unit against
    @ the king tracker, of @p sampleCount **simultaneous solution pairs** (design
    @ doc §8.2; REQ-ADET-013). Same commanded shape as MAG_CAL_START, deliberately:
    @ start / abort / clear, a window counted in accepted samples, one application
    @ point, quality telemetered, parameters read at command time.
    @
    @ **Pairs, not samples.** A pair enters only on a cycle where the king *and*
    @ @p unit both delivered a fresh, valid solution — so an Earth or Sun keep-out
    @ on either unit costs pairs without costing wall-clock, exactly as a position
    @ outage costs the magnetometer window its samples. Simultaneity is what makes
    @ the estimate an alignment rather than a smear: at 0.1 deg/s a one-cycle skew
    @ is already 36 arcsec, comparable to what is being measured.
    @
    @ The window closes itself after ten cycles per pair asked for, so a tracker
    @ that stops solving cannot leave the vehicle telemetering COLLECTING forever.
    @ The fit is attempted on what was collected; too little and
    @ StAlignRejected(SAMPLES) is the honest report.
    @
    @ **Collection does not disturb the estimator.** It is a tap: the fit runs on
    @ **uncorrected** readings so an applied alignment never refits itself, and the
    @ published attitude changes only when a fit is applied or cleared.
    @
    @ Rejected (EXECUTION_ERROR, no window opened) when the alignment parameters
    @ are missing or out of range — StAlignRejected(CONFIG) — when @p unit is the
    @ king, out of range, or has no configured boresight — StAlignRejected(UNIT) —
    @ or when @p sampleCount is below StAlignMinSamples or above the component's
    @ fixed ceiling (AttitudeEstimator::kMaxCalSamples). Starting a window while
    @ one is open restarts it, on whichever unit the new command names.
    guarded command ST_ALIGN_CAL_START(
                                unit: U8 @< starTrackerIn port index to calibrate against the king
                                sampleCount: U32 @< simultaneous pairs to collect before fitting
                              )

    @ Close an alignment collection window without fitting; any previously applied
    @ alignment is left exactly as it was. A no-op (with OK) when no window is
    @ open, so an abort is always safe to send.
    guarded command ST_ALIGN_CAL_ABORT

    @ Drop the applied alignment for @p unit; that tracker reverts to its
    @ as-mounted reading. Leaves a collection window in progress alone — clearing
    @ the applied correction and re-fitting are separate decisions, and the fit
    @ runs on uncorrected readings either way. A no-op (with OK) when nothing is
    @ applied to that unit; EXECUTION_ERROR for an out-of-range index.
    @
    @ **The applied alignment does not survive a reboot**, on the same deferral as
    @ the magnetometer calibration: it lives in component state and nothing is
    @ written to ParameterDb (§23.6 owns non-volatile state). A vehicle whose event
    @ log carries no StAlignComplete is fusing its second tracker as mounted.
    guarded command ST_ALIGN_CAL_CLEAR(
                                unit: U8 @< starTrackerIn port index to revert to as-mounted
                              )

    # ----------------------------------------------------------------------
    # Parameters (mission configuration, §19.3 — no defaults on purpose)
    # ----------------------------------------------------------------------

    @ White (cycle-to-cycle independent) part of the sun-pair 1-sigma transverse
    @ uncertainty [rad]: sensor noise and quantisation. The part repeated fixes
    @ average down. Must be finite and >= 0, and not both zero with the
    @ systematic part.
    param SigmaSunWhiteRad: F64

    @ The sun pair's systematic uncertainty is composed **per cycle** from two
    @ independent terms, each of which the vehicle may or may not have working
    @ on any given cycle, so each is a parameter rather than a constant folded
    @ into one number:
    @
    @   sigma_sun_sys = hypot(albedo term, ephemeris term)
    @
    @ The **albedo** term is the sun sensor's: how much Earthshine error is left
    @ in the measured direction. It takes SigmaSunAlbedoRad when the §8.1
    @ correction ran this cycle and SigmaSunAlbedoUncorrRad when it did not (no
    @ position fix, night side, no Earth in the field, no attitude to place it
    @ with). A further attitude-quality term is added to the corrected case at
    @ runtime — see SunAlbedoPeakRad.
    @
    @ The **ephemeris** term is the *reference's*: how well the Sun's inertial
    @ direction is known. It takes SigmaSunEphemPreciseRad while the onboard
    @ DE440 Chebyshev tables answer the query (grade PRECISE) and SigmaSunEphemRad
    @ when the analytic fallback does. Two orders of magnitude separate them, and
    @ which one is in force is not a configuration choice — it depends on whether
    @ the uploaded tables cover the current epoch.
    @
    @ Keeping the four apart is what lets each be re-derived on its own: a better
    @ sun sensor moves the albedo pair and nothing else, and an ephemeris upload
    @ moves nothing at all because the vehicle already carries both grades.

    @ Albedo residual left in the sun direction **with** the Earth-albedo
    @ correction applied [rad]: the dispersion of the real Earth about the
    @ uniform Lambertian sphere the correction models it as. Contains no
    @ ephemeris error. May be zero (a perfectly modelled Earth, which no vehicle
    @ flies over).
    param SigmaSunAlbedoRad: F64

    @ Albedo error in the sun direction with **no** correction applied [rad] —
    @ the full Earthshine term, an order of magnitude larger. Used on any cycle
    @ the correction refuses. Must be at least SigmaSunAlbedoRad: a correction
    @ that made the measurement worse would be a configuration error, and is
    @ refused rather than flown.
    param SigmaSunAlbedoUncorrRad: F64

    @ Sun-direction error of the **analytic** ephemeris fallback [rad] (§11.3),
    @ used on any cycle the onboard tables cannot answer at grade PRECISE. This
    @ is the degraded floor the vehicle flies on with no ephemeris upload, or
    @ past the end of the uploaded span.
    param SigmaSunEphemRad: F64

    @ Sun-direction error while the onboard DE440 Chebyshev tables answer at
    @ grade PRECISE [rad]. Arcsecond-class, and dominated by geometry rather than
    @ by the fit: at r = 6878 km the geocentric-to-spacecraft-centric parallax is
    @ at most asin(r / 1 AU) = 4.598e-5 rad, against a Chebyshev residual of
    @ ~3.5e-12 rad — seven orders below, so it rounds away. 5e-5 rad covers the
    @ parallax with 9% margin. **Re-derive if the orbit changes**: the parallax
    @ scales with r, so a higher orbit needs a larger value.
    @
    @ The component subtracts that parallax itself when it has both a position
    @ fix **and** the ECEF->ECI rotation; this must cover the case where either
    @ is missing, which is why it is budgeted at the full unremoved value rather
    @ than at the residual after removal.
    @
    @ Deliberately a parameter rather than a hardcoded zero — the ground states
    @ what its uploaded tables are worth, and "we assumed zero" is not a thing to
    @ discover in flight. Must be non-negative and must not exceed
    @ SigmaSunEphemRad: a precise source worse than the fallback is a
    @ configuration error.
    param SigmaSunEphemPreciseRad: F64

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

    # --- Multi-IMU voting (§8.2) -------------------------------------------
    # Part of the **coarse** validity gate above rather than a set of their own,
    # and deliberately so: without a voted body rate there is no gyro
    # propagation, so a missing value here costs exactly what a missing
    # SigmaSunWhiteRad costs — the whole estimator. A separate gate would imply
    # a degraded-but-flying state that does not exist.

    @ Physical body-rate magnitude limit for this vehicle [rad/s]. A unit
    @ reporting above it is failed, not fast. Derive it from the vehicle's worst
    @ credible rate (separation tip-off, detumble entry) with margin — **not**
    @ from the gyro's measurement range, which is what a railed unit reports.
    @ Must be finite and positive.
    param ImuMaxRateRadps: F64

    @ Pairwise disagreement gate on the rate difference magnitude [rad/s], used
    @ only when the surviving set is exactly two units. Above the pair's combined
    @ noise and bias repeatability by a comfortable factor: a false disagreement
    @ costs the body rate for that cycle, because two units can detect a fault
    @ and never attribute one. Must be finite and positive.
    param ImuDisagreementRadps: F64

    @ Consecutive cycles passing **the criterion that excluded it** that re-admit
    @ an IMU. The hysteresis on the automatic recovery policy (§9.2): long enough
    @ that a marginal unit cannot flap in and out, short enough that a transient
    @ does not cost a unit of redundancy for the rest of the flight. Must be
    @ non-zero — zero would make the exclusion latch a no-op.
    @
    @ The criterion is per exclusion kind. A gate failure is undone by passing
    @ that gate; a unit excluded by *identification* was plausible by
    @ construction, so it earns credit only on cycles where it agrees with the
    @ combination — otherwise it would re-admit unconditionally and be outvoted
    @ again, flapping at this period with an FDIR event per lap.
    param ImuReadmitCycles: U32

    @ Consecutive cycles the same IMU must lose the pairwise identification
    @ before it is latched out. Must be non-zero.
    @
    @ A latch is permanent until re-admission earns it back, so one sample must
    @ not buy one — particularly under a slow common-mode drift, where the
    @ residual ordering can flip cycle to cycle. This costs detection latency,
    @ not rate availability: the winning unit is published throughout the
    @ confirmation window.
    param ImuIdentifyConfirmCycles: U32

    @ Consecutive cycles of unattributable IMU disagreement after which the
    @ estimator escalates (§9.2), and the period at which that escalation is
    @ re-reported while the condition persists. Must be non-zero.
    @
    @ The refusal itself is edge-gated, which is right for a transient and wrong
    @ for a permanent condition: without this the vehicle would fly rate-less
    @ indefinitely after a single warning. One parameter serves both the horizon
    @ and the repeat cadence, because a condition worth re-reporting is worth
    @ re-reporting at the interval that made it notable.
    param ImuAmbiguityEscalateCycles: U32

    # ----------------------------------------------------------------------
    # Multi-magnetometer voting (§8.2, §9.2; REQ-ADET-011)
    # ----------------------------------------------------------------------
    #
    # These ride in the **coarse** validity gate, on the same reasoning the IMU
    # voting values do: without a voted field there is no magnetic pair, hence no
    # TRIAD and no coarse attitude, so a missing value costs the whole estimator
    # and a separate gate would imply a degraded-but-flying state that does not
    # exist.
    #
    # The vote combines the **raw** readings. A commanded hard/soft-iron
    # calibration is applied *after* it, because the vote decides which unit's
    # reading the vehicle believes and a correction fitted for one unit must never
    # be used to judge another.

    @ Lower and upper bounds of the accepted |m| / |B_IGRF| ratio [-], the
    @ magnetometer's plausibility gate. This is the natural physical test for a
    @ magnetometer and it costs nothing new: the vehicle already evaluates IGRF-14
    @ at its own position every cycle to build the magnetic reference, and the
    @ measured magnitude has to sit in a band around the modelled one whatever the
    @ attitude is. A fixed full-scale check would admit everything below
    @ saturation; this tracks the field from ~22 to ~52 uT over an orbit.
    @
    @ Must satisfy 0 < Min < 1 < Max: a band excluding the modelled magnitude
    @ itself would reject every healthy unit on every cycle. Size them from the
    @ installed hard-iron and scale error with margin — it is a fault gate, not an
    @ accuracy gate.
    param MagMinFieldRatio: F64

    @ Upper bound of the accepted |m| / |B_IGRF| ratio [-]. See MagMinFieldRatio.
    param MagMaxFieldRatio: F64

    @ Pairwise disagreement gate on the field difference magnitude [T], used only
    @ when the surviving set is exactly two units. Above the pair's combined noise
    @ and installed hard-iron spread by a comfortable factor: a false disagreement
    @ costs the magnetic pair for that cycle. Must be finite and positive.
    param MagDisagreementT: F64

    @ Largest attitude-error 1-sigma [rad] at which the modelled field rotated into
    @ body axes is still trusted to attribute a two-magnetometer disagreement.
    @
    @ A **quality** gate, not a validity flag, and the distinction is the whole
    @ point: a validity flag cannot tell a 0.5 deg solution from a 10 deg one, and a
    @ 10 deg attitude error moves the predicted field by sigma*|B| — several uT on
    @ a 30 uT field, which is many times any sane MagDisagreementT. Identifying on
    @ that would hand the verdict to whichever unit happened to sit nearer a badly
    @ rotated prediction. Derive it so that sigma*|B| stays comfortably inside
    @ MagDisagreementT. Must be finite and positive.
    param MagMaxAttSigmaRad: F64

    @ Consecutive cycles passing **the criterion that excluded it** that re-admit a
    @ magnetometer. Same policy and same reasoning as ImuReadmitCycles. Must be
    @ non-zero.
    param MagReadmitCycles: U32

    @ Consecutive cycles the same magnetometer must lose the pairwise
    @ identification before it is latched out. Same policy and same reasoning as
    @ ImuIdentifyConfirmCycles. Must be non-zero.
    @
    @ The ambiguity escalation horizon is shared with the IMU vote
    @ (ImuAmbiguityEscalateCycles) rather than duplicated: it is a statement about
    @ how long the ground should wait before a persistent refusal is worth a
    @ console alert, which is a property of the operations concept and not of the
    @ sensor. What differs is the *cost* of the refusal — the IMU case loses the
    @ body rate, this one loses only the magnetic pair — and that difference is in
    @ the severity of what happens next, not in the horizon.
    param MagIdentifyConfirmCycles: U32

    # ----------------------------------------------------------------------
    # Star-tracker fusion (§8.2) — a fourth, independent validity gate
    # ----------------------------------------------------------------------
    #
    # Validated separately from the coarse, fine and albedo sets, and the cost
    # structure is why. A missing value here costs **tracker fusion**: the
    # component emits StConfigInvalid, the mode ladder cannot reach its finest
    # rung, and the vehicle flies the SS+MAG+IMU fine mode it flew before trackers
    # were fused — a fully flyable state, and not one to refuse the estimator over.
    #
    # MekfAttNisGate is the exception and rides in the *fine* set instead: it is
    # part of `MekfConfig::isValid`, so the filter cannot be built without it.

    @ Which starTrackerIn port index is the **king** tracker — the unit whose
    @ mounting *defines* the body frame (design doc §8.2). Its reading is taken as
    @ the frame itself: no alignment is estimated for it, ST_ALIGN_CAL refuses it,
    @ and every other tracker is stated in its frame before reaching the filter.
    @
    @ This is a vehicle-integration decision, not a runtime one. Changing it in
    @ flight redefines the body frame and therefore every mounting quaternion,
    @ every control gain axis and every payload boresight on the vehicle; it is a
    @ parameter only because a hardcoded index would be a vehicle constant in code
    @ (§19.4). Must be inside the port array.
    param StKingUnit: U32

    @ Cross-boresight and about-boresight 1-sigma of a star-tracker solution [rad],
    @ used to build each unit's measurement covariance
    @ R = sigma_xy^2 (I - b b') + sigma_z^2 b b' in body axes, with b the unit's
    @ boresight from StBoresightsBody.
    @
    @ **Carrying the anisotropy is the entire reason two trackers beat one.** A
    @ tracker barely constrains rotation about its own boresight — the identified
    @ stars hardly move — so sigma_z runs several times sigma_xy (about 6x for the
    @ reference vehicle's AURIGA). With two non-parallel boresights each unit's
    @ tight directions cover the other's weak one, and an isotropic sigma^2 I would
    @ throw exactly that away and report a covariance the geometry does not
    @ support.
    @
    @ Both are **inflated** figures, not the datasheet noise: the filter treats R
    @ as white, and a tracker's fixed bias and its low-frequency spatial term do
    @ not average down. Root-sum-square the datasheet's white and systematic terms
    @ per axis. One value each rather than per unit because they describe the
    @ *part*; a mixed suite would need them per unit, which is another parameter of
    @ the same shape rather than a redesign. Must be finite and positive.
    param StSigmaXyRad: F64

    @ About-boresight 1-sigma of a star-tracker solution [rad]. See StSigmaXyRad.
    @ Must be finite, positive, and >= StSigmaXyRad — a tracker tighter about its
    @ boresight than across it is a configuration error, and flying it would report
    @ a covariance tighter than the truth in the one direction that is weakest.
    param StSigmaZRad: F64

    @ Per-unit star-tracker boresights in body axes, flattened three at a time in
    @ **starTrackerIn port order** (§19.4 build order). Same shape and same
    @ conventions as SunAlbedoBoresightsBody: a slot for a unit that is not
    @ installed, or whose mounting has not been characterised, is written as the
    @ **zero vector**, and that unit is then not fused at all rather than fused
    @ with a guessed R.
    @
    @ These must match the mounting_quaternion_wxyz entries in the vehicle config —
    @ each is that quaternion applied to the sensor's +Z. The config compiler
    @ cross-checks them, as it does the sun-sensor set.
    @
    @ Non-parallel boresights are what the second tracker is *for* (§8.1): two
    @ units looking the same way share a weak axis and buy noise averaging on the
    @ strong ones, which is the smaller half of what a tracker costs.
    param StBoresightsBody: Vec3F64PerUnit

    # --- Residual monitors on the demoted sources (§8.2, §9.2) ---------------
    #
    # While a tracker is fused the sun and magnetic pairs are **not** folded into
    # the filter (see the FineSource enum). They are cross-checked against it
    # instead, which is a strictly better use of them: a sun sensor that has
    # drifted becomes observable rather than merely down-weighted. These three
    # thresholds are what "has drifted" means.
    #
    # They are FDIR thresholds, not accuracy budgets: set each several times the
    # source's own 3-sigma so the monitor fires on a fault and not on a bad day.

    @ Largest accepted angle [rad] between the selected sun sensor's measured
    @ direction and the tracker-derived one. Must be finite and positive.
    param MonitorSunResidualRad: F64

    @ Largest accepted angle [rad] between the voted magnetic field direction and
    @ the tracker-derived modelled one. Must be finite and positive. Wider than the
    @ sun threshold on any real vehicle: the IGRF model error is part of what this
    @ residual measures, and it is not a fault.
    param MonitorMagResidualRad: F64

    @ Largest accepted angle [rad] between the **selected** sun sensor's direction
    @ and the runner-up unit's, on the 47% of the sky where two or more units see
    @ the Sun (design doc §8.2).
    @
    @ This closes the gap the sigma-ordered selector leaves open: the selector
    @ takes the smallest *reported* sigma, and a unit that is confidently wrong
    @ reports a small sigma and wins. Nothing else checks it. On a persistent
    @ disagreement the estimator emits ResidualMonitorAlert(SUN_CROSS_UNIT) and, if
    @ it has a fine solution to judge with, switches to whichever of the two agrees
    @ better with it (SunUnitOverridden). Must be finite and positive.
    param MonitorSunCrossUnitRad: F64

    @ Consecutive cycles a residual monitor must be over its threshold before it
    @ alerts, and the period at which the alert is re-reported while the condition
    @ persists. Must be non-zero.
    @
    @ One parameter for all three monitors: they answer the same operational
    @ question — "has this been wrong long enough to be a fault rather than a
    @ transient?" — and three tunings for one question is three chances to set two
    @ of them wrong. Per-monitor persistence is what a mixed suite would need, and
    @ is another parameter of the same shape rather than a redesign.
    param MonitorAlertCycles: U32

    @ Containment the coarse-agreement test admits a star tracker at: the
    @ chi-square quantile for **3** degrees of freedom (an attitude error is
    @ three-axis) at the chosen confidence. It decides whether the filter is
    @ re-seeded from a tracker the NIS gate refused — whether the whole attitude
    @ solution is replaced — so the cost of admitting a unit that did not belong
    @ is one re-seed while the cost of refusing one that did is the vehicle's
    @ finest rung; the errors are not symmetric and the reference tuning
    @ (16.266, the 0.999 quantile) sits on the permissive side deliberately.
    @ The same quantile the MEKF's own attitude gate (MekfAttNisGate) uses; a
    @ parameter rather than a constant so the two cannot drift apart in code.
    param StCoarseAgreementGate: F64

    @ Consecutive cycles a latched-out star tracker must agree with the fine
    @ solution before its exclusion is lifted. Its own parameter and not
    @ MonitorAlertCycles: that one is the *reporting cadence* of the residual
    @ monitors, and an operator who quiets a noisy monitor by raising it must
    @ not silently lengthen an FDIR parole sentence on the vehicle's finest
    @ attitude source.
    param StReadmitCycles: U32

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

    @ NIS rejection threshold [dimensionless] for one **attitude** update — a star
    @ tracker's complete solution. A chi-square quantile on **3** degrees of
    @ freedom, not 2 (chi2_3 at 99.9% = 16.27): a tracker's innovation is a full
    @ rotation with no degenerate direction, unlike the transverse innovation
    @ between two unit vectors. Must be positive.
    @
    @ Kept as its own value rather than reusing MekfNisGate because the two gate
    @ different statistics: a 2-DOF threshold applied to a 3-DOF NIS gates at about
    @ 99.0% instead of 99.9%, i.e. ten times the false-rejection rate, on the one
    @ measurement source the finest mode is built around — and a rejection streak
    @ there demotes the mode.
    param MekfAttNisGate: F64

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
    # Magnetometer-calibration parameters — a third, independent validity gate
    # ----------------------------------------------------------------------
    #
    # Validated separately again, and on the *command* rather than per cycle. A
    # missing calibration parameter costs neither the coarse chain nor the fine
    # mode — it costs only the ability to *start* a calibration, which is a
    # commanded activity, not a flight function. So there is no per-cycle event
    # for this set: MAG_CAL_START answers MagCalRejected(CONFIG) and the vehicle
    # is otherwise completely unaffected. These map one-for-one onto
    # polaris::gnc::MagCalibrationConfig.

    @ Field magnitude the fit is non-dimensionalised by [T], e.g. 30e-6 in LEO.
    @ Numerics only — the solution is invariant to it up to round-off; without it
    @ the ten unknowns span 18 orders of dynamic range. Must be positive.
    param MagCalNominalFieldT: F64

    @ Smallest accepted |m_raw| and IGRF magnitude for a calibration sample [T].
    @ Rejects a dead or unpowered sensor and a nonsensical reference.
    param MagCalMinFieldT: F64

    @ Largest accepted |m_raw| and IGRF magnitude for a calibration sample [T].
    @ Rejects a saturated reading and — until the §7 MTQ/MAG interlock supplies
    @ the actuation state — the grossest magnetorquer-contaminated samples.
    @ Must exceed MagCalMinFieldT.
    param MagCalMaxFieldT: F64

    @ Fewest samples that may be fitted. The algebraic minimum is 10 (ten
    @ unknowns), which fits exactly and reports zero residual regardless of
    @ truth, so the residual and condition gates are meaningless below a
    @ well-overdetermined problem. Also the floor on MAG_CAL_START's sampleCount.
    param MagCalMinSamples: U32

    @ Smallest accepted orientation coverage 3*lambda_min(D) of the sampled field
    @ directions [dimensionless], in (0, 1]. A calibration fitted inside a narrow
    @ cone extrapolates the ellipsoid over directions it never saw, which is
    @ worse than no calibration — so this is a gate, not a diagnostic.
    param MagCalMinCoverage: F64

    @ Largest accepted condition number lambda_max/lambda_min of the
    @ preconditioned normal matrix [dimensionless], > 1. The rigorous
    @ observability gate: it sees every way the ten parameters can fail to
    @ separate, where MagCalMinCoverage only sees the direction spread.
    param MagCalMaxCondition: F64

    @ Factor by which the calibrated magnitude residual must beat the
    @ uncalibrated one [dimensionless], >= 1. A "calibration" that does not
    @ improve the scalar check is refused rather than applied — a calibration
    @ that makes the magnetometer worse is the outcome worth guarding hardest
    @ against.
    param MagCalMinImprovement: F64

    # --- Inter-star-tracker alignment calibration (§8.2) ---------------------
    #
    # Read at ST_ALIGN_CAL_START only, on the same reasoning the MagCal set is: a
    # missing value costs the ability to *start* an alignment calibration, so it is
    # a command-time refusal — StAlignRejected(CONFIG) — and never a flight event
    # on a vehicle that is otherwise entirely healthy.

    @ Fewest simultaneous solution pairs that may be fitted. At least 2: one pair
    @ determines all three parameters exactly and reports a zero residual whatever
    @ the truth, so the quality gates below only mean something on a
    @ well-overdetermined window. Also the floor on ST_ALIGN_CAL_START's
    @ sampleCount.
    @
    @ Unlike the magnetometer fit there is **no geometry requirement** to satisfy
    @ and therefore no coverage gate: an attitude pair determines the relative
    @ rotation at any attitude, so a window does not need the vehicle to tumble.
    @ What more pairs buy is averaging down the trackers' own noise, which is the
    @ only reason the floor is above two.
    param StAlignMinSamples: U32

    @ Largest accepted RMS residual of the fitted alignment [rad]. Size it from the
    @ two trackers' combined per-sample noise with margin: above it the pairs are
    @ not describing one fixed rotation, and the correction would be an average of
    @ something that is not constant. Must be finite and positive.
    param StAlignMaxResidualRad: F64

    @ Smallest accepted normalised eigen-gap (lambda_max - lambda_2)/N of the
    @ quaternion-average moment matrix [dimensionless], in (0, 1). Near 1 when
    @ every pair agrees on one rotation.
    @
    @ Worth being precise about what this catches, because it is **not** a geometry
    @ gate — attitude pairs have no degenerate geometry, so there is no analogue of
    @ TRIAD's near-parallel refusal here. It detects a *fault*: one tracker
    @ delivering solutions that do not sit at a fixed rotation from the other's — a
    @ mis-identified star field, a unit reporting stale or another unit's solution,
    @ a mounting that is moving. Refusing there is refusing to average a rotation
    @ that does not exist.
    param StAlignMinEigenGap: F64

    # ----------------------------------------------------------------------
    # Earth-albedo correction parameters — a fifth, independent validity gate
    # ----------------------------------------------------------------------
    #
    # Validated separately again, and for the same reason the fine set is: a
    # missing albedo parameter costs the *correction*, not the estimator. The
    # component then behaves exactly as it did before the correction existed —
    # every cycle weighted at SigmaSunAlbedoUncorrRad — which is a working vehicle
    # with a wider sun budget, so it emits AlbedoConfigInvalid once and carries
    # on rather than refusing cycles.
    #
    # This is a **model, not a commanded calibration** (§8.1): it needs geometry,
    # not collected data, so there is no command, no window, and no fitted state
    # to persist. It runs on every cycle whose geometry supports it.
    #
    # The peak and the field of view describe the sun-sensor **part**, of which
    # the vehicle carries one type, so they are one value each; the boresight is
    # per **unit**, since the whole point of the §8.2 suite is that different
    # units point at different faces. A suite of mixed parts would need the first
    # two per-unit as well — the change is another Vec3F64PerUnit-shaped
    # parameter each, not a redesign.

    @ Peak angular error from Earthshine [rad]: the value reached with the Earth
    @ filling the field on a fully sunlit day side, 90 deg from the Sun. The
    @ unit's datasheet figure — `albedo_error_deg` in its config/hardware entry,
    @ converted to radians. Must be finite, >= 0, and below a quarter turn (which
    @ catches degrees left unconverted).
    param SunAlbedoPeakRad: F64

    @ Acceptance half-angle of the sun-sensor part [rad], which sets how much
    @ Earth can be in its field at all. Must be positive and <= pi/2.
    param SunAlbedoHalfFovRad: F64

    @ Per-unit boresights in **body** axes, flattened three components at a time
    @ in `sunSensorIn` port-array order: each unit's mounting quaternion applied
    @ to the sensor's +Z. Mounting is configuration, not measurement, which is why
    @ it arrives here rather than on the measurement port.
    @
    @ A slot for a unit that is not installed — or one whose mounting has not been
    @ characterised — is the **zero vector**, and the correction is then skipped
    @ for that unit and the cycle weighted at SigmaSunAlbedoUncorrRad. Skipping
    @ rather than guessing: the correction places the Earth in *this* unit's
    @ field, so a wrong boresight injects a bias the size of the one being
    @ removed, pointed in an arbitrary direction (§8.1).
    param SunAlbedoBoresightsBody: Vec3F64PerUnit

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

    @ A valid, fresh gyro measurement was found this cycle. With the §8.2 vote in
    @ place this means "the vote produced a usable rate", which is not the same as
    @ "some unit reported": a two-unit disagreement nothing could attribute leaves
    @ this false with both units still talking.
    telemetry GyroValid: bool

    @ IMUs contributing to this cycle's voted rate (§8.2). The redundancy margin,
    @ read alongside ImuExclusionMask: 2 is the healthy reference vehicle, which
    @ can *detect* a disagreement and needs the MEKF's propagated rate to
    @ attribute one; 1 has lost detection as well; 0 means no rate this cycle. A
    @ vehicle carrying 3 or more reads 3+ here and takes the median branch, which
    @ needs no external reference.
    telemetry ImuContributing: U32

    @ Bit i set means IMU i is latched out of the vote (§9.2). One word rather
    @ than a channel per unit, so the whole array's health is one trend line; the
    @ transitions are ImuUnitExcluded / ImuUnitReadmitted.
    telemetry ImuExclusionMask: U32

    @ Port index of the sun sensor selected this cycle — the valid, fresh,
    @ sun-in-view unit with the smallest realised sigma (§8.2). 255 when no unit
    @ was selectable (eclipse, or the Sun outside every field). Watching this step
    @ between units is how a handoff shows up on the ground.
    telemetry SunUnitSelected: U8

    @ A valid, fresh sun measurement with the sun in view, paired with a
    @ reference, was found this cycle.
    telemetry SunValid: bool

    @ A valid, fresh magnetometer measurement paired with a modelled field was
    @ found this cycle.
    telemetry MagValid: bool

    @ A valid orbit solution was available this cycle. Without one there is
    @ no magnetic reference, so the estimator coasts on the gyro.
    telemetry PositionValid: bool

    @ Star-tracker solutions received since start, across every unit. A raw
    @ arrival count, not a fusion count: it rises whether or not the solution was
    @ fresh, whether or not the unit had a configured boresight, and whether or not
    @ the filter accepted it. StContributing is the fusion count.
    telemetry StarTrackerCount: U32

    @ Star trackers whose solution was **accepted into the filter** this cycle
    @ (§8.2). The redundancy margin that matters for REQ-ADET-007: 2 is the healthy
    @ reference vehicle and the configuration the 0.02 deg-class figure rests on,
    @ since each unit covers the other's weak about-boresight axis; 1 is a working
    @ but boresight-limited solution; 0 means the ladder has fallen to SS+MAG or
    @ coarse.
    telemetry StContributing: U32

    @ Bit i set means star tracker i delivered a valid, fresh, boresight-configured
    @ solution this cycle. Read against StContributing, this separates "the unit
    @ stopped solving" (Earth or Sun in the baffle, a slew past the tracking
    @ envelope) from "the filter rejected what it sent" — different faults with
    @ different responses.
    telemetry StValidMask: U32

    @ **Largest** NIS of this cycle's star-tracker updates [dimensionless],
    @ accepted or rejected. ~chi2_**3** when the filter is consistent, unlike
    @ MekfNis which is chi2_2: a tracker's innovation is a full rotation. Kept as
    @ its own channel for exactly that reason — one channel carrying two different
    @ null distributions is a channel nobody can set an alarm on. NaN on a cycle
    @ with no tracker update.
    telemetry StNis: F64

    @ Which measurement source the fine solution is being updated from — the §8.2
    @ mode ladder. NONE while the coarse chain is the published product.
    telemetry FineSource: FineSource

    @ Magnetometers contributing to this cycle's voted field (§8.2). Same reading
    @ as ImuContributing: 2 is the healthy reference vehicle, 1 has lost detection,
    @ 0 means no magnetic pair this cycle.
    telemetry MagContributing: U32

    @ Bit i set means magnetometer i is latched out of the vote (§9.2). The
    @ transitions are MagUnitExcluded / MagUnitReadmitted.
    telemetry MagExclusionMask: U32

    @ Port index of the magnetometer whose reading the vote published this cycle —
    @ the lowest-indexed contributing unit. 255 when none contributed. It is what
    @ picks the per-unit calibration applied downstream, so a handoff between units
    @ shows up here first.
    telemetry MagUnitSelected: U8

    @ Magnetometer samples excluded by the §7 MTQ/MAG duty-cycle interlock since
    @ start: the sample's time tag fell outside the controller's published quiet
    @ window, or a rod is latched stuck-on and no window is quiet. A steadily
    @ climbing count with the rods nominally off is the signature the §9 stuck-on
    @ monitor exists to name.
    telemetry MagInterlockRejects: U32

    @ Angle [rad] between the selected sun sensor's measured direction and the one
    @ the fine solution predicts, on cycles where a star tracker is fused and the
    @ sun pair is therefore a **monitor** rather than a measurement (§8.2). NaN when
    @ not computed. This is the channel that makes a drifting sun sensor visible
    @ instead of merely down-weighted.
    telemetry SunResidualRad: F64

    @ Angle [rad] between the voted magnetic field direction and the one the fine
    @ solution predicts, on the same cycles and for the same reason. NaN when not
    @ computed. Expect it to sit at the IGRF model error, which is not a fault —
    @ MonitorMagResidualRad is set well above it.
    telemetry MagResidualRad: F64

    @ Angle [rad] between the selected sun sensor's direction and the runner-up
    @ unit's, on the 47% of the sky where two or more units see the Sun. NaN when
    @ fewer than two units are selectable. Independent of the fine mode: it is a
    @ sensor-versus-sensor check and needs no attitude at all, which is what makes
    @ it the one cross-check available in Safe mode.
    telemetry SunCrossUnitRad: F64

    @ Where the commanded inter-tracker alignment calibration is (§8.2). Derived,
    @ not stored: COLLECTING outranks APPLIED because a window opened over an
    @ applied alignment is the condition worth seeing.
    telemetry StAlignState: StAlignState

    @ Simultaneous pairs accepted by the open alignment window, 0 when none is
    @ open. Watching it rise is how the ground tells a window that is collecting
    @ from one whose tracker has stopped solving — the latter stalls here while the
    @ self-close deadline runs down.
    telemetry StAlignSamples: U32

    @ RMS residual of the applied alignment [rad], NaN when none is applied. The
    @ number the ground grades a calibration on before deciding to keep it.
    telemetry StAlignResidualRad: F64

    @ Total misalignment angle the applied correction removes [rad], NaN when none
    @ is applied. Read against the mounting drawing: a correction far larger than
    @ the integration tolerance is a mounting or a boresight-parameter error, not a
    @ calibration success.
    telemetry StAlignAngleRad: F64

    @ Bit i set means star tracker i is being read through a fitted alignment
    @ correction. The king's bit is never set — its mounting *is* the body frame.
    telemetry StAlignMask: U32

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

    @ Magnetometer-calibration lifecycle state (design doc §8.1). Written every
    @ cycle so the ground can watch a window run without polling an event stream.
    telemetry MagCalState: MagCalState

    @ Samples accepted into the open collection window. Zero when none is open.
    @ Read against the commanded sampleCount, this is the progress bar.
    telemetry MagCalSamples: U32

    @ Live orientation coverage 3*lambda_min(D) of the samples collected so far
    @ [dimensionless]. **This is the "has it tumbled enough" signal** and it is
    @ the reason to watch a window rather than wait for it: coverage climbing
    @ toward MagCalMinCoverage says the window will succeed, coverage flat says
    @ it will be refused however long it runs, and the ground can abort and
    @ re-fly with a larger tumble instead of spending the whole window. NaN when
    @ no window is open.
    telemetry MagCalCoverage: F64

    @ Angle-equivalent RMS residual of the last **accepted** calibration [rad]:
    @ residual_rms / mean_field. Indicative rather than a bound — the scalar
    @ magnitude check is blind to the error component transverse to the field,
    @ which is the one that rotates the vector — but it is the only accuracy
    @ figure available without an attitude reference, and it is the number the
    @ ground grades a fit on. NaN until a calibration has been accepted, and
    @ again after MAG_CAL_CLEAR or RESET_ESTIMATOR.
    telemetry MagCalResidualAngle: F64

    @ Earth-albedo pull removed from this cycle's sun measurement [rad] (§8.1).
    @ **NaN means the correction did not run**, and that is the channel's second
    @ job: it is how the ground tells which of SigmaSunAlbedoRad and
    @ SigmaSunAlbedoUncorrRad was in force this cycle. Note that is the *albedo*
    @ term alone, not the whole composed systematic — the ephemeris term is
    @ chosen separately by the served TableGrade, and OnboardTables.EphemGrade is
    @ what shows that half. NaN is normal and
    @ frequent — eclipse, the night side, no Earth in the sensor's field, no
    @ position fix, or no attitude yet to place the field with.
    @
    @ On the day side with the Earth in view this should track the orbit smoothly
    @ over minutes, peaking well under SunAlbedoPeakRad. A value pinned at the
    @ peak, or one that jumps cycle to cycle, means the geometry feeding it is
    @ wrong rather than the sensor.
    telemetry SunAlbedoCorrection: F64

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
      format "Fine mode engaged: MEKF seeded, seed cov trace={} rad^2"

    @ The fine solution changed which measurement source it is updated from — the
    @ §8.2 mode ladder moving a rung while fine mode stays engaged (design doc
    @ §8.1, §8.2; REQ-ADET-004, REQ-ADET-012). SUN_MAG -> STAR_TRACKER when a
    @ tracker becomes available, at which point the sun and magnetic pairs stop
    @ being folded in and become monitors; STAR_TRACKER -> SUN_MAG when every
    @ tracker drops out (Earth or Sun in the baffle, a slew past the tracking
    @ envelope, a unit fault).
    @
    @ Deliberately **not** an AttitudeLost or a demotion: the filter keeps its
    @ state and its covariance across the change, and the published solution stays
    @ valid throughout — what changes is how fast the covariance grows from here.
    @ Losing the trackers costs about two orders of magnitude of accuracy, so a
    @ consumer with a knowledge requirement (a payload, REQ-PAY-001) gates on this
    @ channel and not merely on AttitudeValid.
    @ Action: none on a transition that clears within a few minutes — tracker
    @ outages are geometry. A vehicle that never reaches STAR_TRACKER with trackers
    @ installed means StBoresightsBody, StKingUnit or the keep-out geometry is
    @ wrong.
    event FineSourceChanged(from: FineSource, to: FineSource, trackers: U32) \
      severity activity high \
      format "Fine-mode source changed {} -> {} ({} tracker(s) fused)"

    @ The filter was **re-seeded from a star tracker** after persistently
    @ rejecting it while running on the sun/magnetic pairs (design doc §8.2,
    @ §9.2; REQ-FDIR-013). This is an arbitration, not a fault report, and the
    @ event exists so the ground reads it as one — the alternative reading of the
    @ same telemetry ("the tracker was bad") is exactly backwards.
    @
    @ Why it happens: the SS+MAG solution's error is dominated by the
    @ magnetometer's systematic, which the filter's white-R model averages its
    @ covariance down through, so the innovation covariance shrinks to a few
    @ milliradians while the true error stays tens. Every arriving tracker update
    @ then fails the chi-square gate however good the tracker is. The rejection is
    @ therefore evidence about the *filter*, not about the unit, and the response
    @ is to adopt the better source rather than to isolate it.
    @
    @ Guarded by a like-for-like comparison first: the tracker must agree with the
    @ **coarse** solution, whose covariance converges to the systematic floor and
    @ therefore does not lie, inside the chi-square-3 quantile at 0.999 (16.266).
    @ @p mahalanobis is that statistic; @p separationRad is the same disagreement
    @ as a plain angle, for a human reading the log.
    @ Action: none autonomous, and none expected from the ground — the vehicle has
    @ just moved to its best available source and FineSourceChanged follows on the
    @ same cycle. Repeated occurrences on the same unit without an intervening
    @ tracker outage mean something is pulling the filter back off the tracker;
    @ capture StNis and the sun/magnetic residual monitors.
    event FineReseededFromStarTracker(unit: U8, separationRad: F64, mahalanobis: F64) \
      severity warning low \
      format "Fine solution re-seeded from star tracker {}: {} rad from the coarse solution (d2={}), so the covariance was wrong, not the tracker"

    @ A star tracker the filter persistently rejects was **not** adopted: it also
    @ disagrees with the coarse solution, by more than that solution's own
    @ covariance supports (chi-square-3 at 0.999 = 16.266). Where
    @ FineReseededFromStarTracker says the filter was wrong, this says the *unit*
    @ is — it is inconsistent with everything the vector data can support.
    @
    @ **Nothing is latched.** The unit is not excluded and stays a candidate every
    @ cycle, because the criterion that would convict it here (agreement with the
    @ coarse solution) is not the criterion that would re-admit it (agreement with
    @ the fine one), and an exclusion whose parole test differs from its
    @ conviction test is permanent by construction. So the vehicle keeps looking:
    @ a later cycle with a better reference — a coarse fix on cleaner sun/field
    @ geometry, or the other tracker seeding the filter — can still adopt it.
    @
    @ Emitted at a bounded cadence (MonitorAlertCycles) while the condition lasts,
    @ and only while the coarse fix is fresh: a coasted covariance grows as a
    @ stated *lower* bound on the truth, so refusing on one would be an accusation
    @ built on a number known to be optimistic.
    @ Action: investigate the unit — a mis-identified star field, a mounting that
    @ has moved, or a wrong StBoresightsBody entry. The vehicle is meanwhile flying
    @ its sun/magnetic solution, which is REQ-ADET-006 accuracy and safe, so this
    @ is a pass-timescale action and not an immediate one. A vehicle showing this
    @ on *every* tracker points at the coarse chain or the vehicle's own tuning
    @ rather than at the trackers.
    event FineTrackerAdoptionRefused(unit: U8, separationRad: F64, mahalanobis: F64) \
      severity warning high \
      format "Star tracker {} not adopted: {} rad from the coarse solution (d2={}) is more than that solution supports"

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

    @ An Earth-albedo-correction parameter is missing from ParameterDb or outside
    @ its valid range. Not fatal and not even mode-limiting: every cycle is
    @ simply weighted at SigmaSunAlbedoUncorrRad and the sun measurement carries its
    @ full albedo, which is how the vehicle flew before the correction existed.
    @ Edge-gated to the transition into the invalid state.
    @ Action: uplink the missing/corrected parameter (PRM_SET + PRM_SAVE). Until
    @ then the attitude solution is degraded but honest — the wider sigma is the
    @ one being used, so the reported covariance is not overconfident.
    event AlbedoConfigInvalid(detail: string size 80) \
      severity warning high \
      format "Albedo correction configuration invalid, running uncorrected: {}"

    @ A star-tracker-fusion parameter is missing from ParameterDb or outside its
    @ valid range. Not fatal and not mode-limiting below the top rung: the ladder
    @ simply cannot reach STAR_TRACKER, and the vehicle flies the SS+MAG+IMU fine
    @ mode it flew before trackers were fused. Edge-gated to the transition into
    @ the invalid state.
    @ Action: uplink the missing/corrected parameter (PRM_SET + PRM_SAVE). The
    @ vehicle is flyable meanwhile, at REQ-ADET-006 accuracy rather than
    @ REQ-ADET-007 — which is a real loss for a payload but not for safety.
    event StConfigInvalid(detail: string size 80) \
      severity warning high \
      format "Star-tracker fusion configuration invalid, no tracker fused: {}"

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

    @ One IMU failed a plausibility gate and has been latched out of the vote
    @ (design doc §8.2, §9.2; REQ-ADET-008, REQ-ADET-009). Fires on the
    @ **transition** into exclusion, not once per cycle in it, so a permanently
    @ dead unit costs one event rather than 10 per second.
    @ Action: none autonomously — the vote has already isolated the unit and the
    @ remaining set carries the rate. Operator: trend ImuExclusionMask. A unit
    @ that re-excludes repeatedly after ImuUnitReadmitted is degrading rather
    @ than glitching, and is a candidate for a commanded power cycle.
    event ImuUnitExcluded(unit: U8, reason: ImuExclusionReason) \
      severity warning high \
      format "IMU {} excluded from the rate vote: {}"

    @ An excluded IMU behaved for ImuReadmitCycles consecutive cycles and is back
    @ in the vote. The recovery edge of ImuUnitExcluded.
    event ImuUnitReadmitted(unit: U8) \
      severity activity high \
      format "IMU {} re-admitted to the rate vote"

    @ Exactly two IMUs survived the plausibility gates, they disagree by more than
    @ ImuDisagreementRadps, and no MEKF propagated rate was available to say which
    @ of them to believe. **No body rate is published this cycle** — a rate that
    @ cannot be trusted is worse than none, because it propagates the fault into
    @ the attitude solution, whereas the estimator's dropout behaviour (hold and
    @ coast on a growing covariance) is designed for. Edge-gated.
    @
    @ On the two-IMU reference vehicle this is a reachable flight state rather
    @ than an already-degraded one: the pair is the whole suite, so a
    @ disagreement while the vehicle is in coarse mode — before the MEKF has
    @ seeded, or after a demotion — has nothing to attribute it with.
    @ Action: a genuine loss of rate knowledge, and the §9.2 conservative
    @ response is already taken (hold, coast, grow the covariance). This event is
    @ **edge-gated**, so a persistent condition is reported by
    @ ImuVoteAmbiguousPersistent rather than by this one repeating — read that
    @ event's Action block for the recovery path. If fine mode is available the
    @ identification runs instead and this event is not emitted; an
    @ ImuUnitExcluded(OUTVOTED) appears in its place.
    event ImuVoteAmbiguous(rateDifferenceRadps: F64) \
      severity warning high \
      format "Two IMUs disagree by {} rad/s and nothing can attribute it: no body rate"

    @ The unattributable disagreement above has persisted for
    @ ImuAmbiguityEscalateCycles, and the vehicle has been without a body rate
    @ for that whole time. Re-emitted at the same period while it continues, so
    @ the condition stays visible without becoming per-cycle noise.
    @
    @ **The onboard response is refusal plus this escalation, and nothing more.**
    @ That is a deliberate boundary, not an omission: choosing between two
    @ disagreeing units without evidence is exactly what the refusal exists to
    @ avoid, so there is no autonomous action that is better than continuing to
    @ refuse. Recovery is a **ground action**: disambiguate from telemetry (per-unit
    @ rates are not downlinked individually, but ImuContributing, the attitude
    @ solution's behaviour and the vehicle's commanded history are), then either
    @ RESET_ESTIMATOR after the faulty unit is known, or uplink a widened
    @ ImuDisagreementRadps if the pair is merely out of family rather than faulted.
    @ Action: an autonomous FDIR mode response (e.g. escalation to Safe) is
    @ explicitly **deferred to the §9 FDIR push**, which owns the mode ladder;
    @ this event is the signal that push will consume.
    event ImuVoteAmbiguousPersistent(durationSec: F64, cyclesWithoutRate: U32) \
      severity warning high \
      format "IMU disagreement unattributable for {} s ({} cycles without a body rate): ground intervention required"

    @ One magnetometer is latched out of the voted field, with the gate that closed
    @ it (design doc §8.2, §9.2; REQ-ADET-011). Reported once, on the transition,
    @ so a permanently dead unit costs one event rather than ten a second. The
    @ magnetic pair survives on the remaining unit — unlike the IMU case, where
    @ losing the pair costs the body rate.
    @ Action: FIELD_MAGNITUDE on a unit that recovers is a transient (the automatic
    @ policy re-admits it); one that stays out is a dead, unpowered or saturated
    @ sensor. OUTVOTED means the modelled field attributed a two-unit disagreement
    @ — the losing unit's hard-iron signature has changed, and a MAG_CAL_START
    @ window is the response once the healthy unit is confirmed.
    event MagUnitExcluded(unit: U8, reason: MagExclusionReason) \
      severity warning high \
      format "Magnetometer {} excluded from the voted field: {}"

    @ A previously excluded magnetometer passed the criterion that excluded it for
    @ MagReadmitCycles consecutive cycles and is contributing again.
    @ Action: none; informational. A unit that flaps between this and
    @ MagUnitExcluded is marginal — widen its gate or exclude it by command.
    event MagUnitReadmitted(unit: U8) \
      severity activity high \
      format "Magnetometer {} re-admitted to the voted field"

    @ The §7 interlock is excluding otherwise-usable magnetometer samples because
    @ the controller reports a rod latched stuck-on: no window is quiet, whatever
    @ the clock says. Operator-visible because the vehicle is flying without a
    @ magnetic pair — in a tracker-fused mode that costs nothing, in Safe mode it
    @ costs the attitude. Edge-gated and then repeated at the shared alert
    @ cadence (MonitorAlertCycles' twin, ImuAmbiguityEscalateCycles).
    event MagInterlockExcluding(rejected: U32, cycles: U32) \
      severity warning high \
      format "MTQ/MAG interlock excluding magnetometer samples: {} rejected in total, {} cycles"

    @ The interlock stopped excluding samples — the controller cleared its
    @ stuck-on latch and the magnetic pair is available again.
    event MagInterlockRestored(cycles: U32) \
      severity activity high \
      format "MTQ/MAG interlock restored after {} cycles: magnetometer samples usable again"

    @ Two plausible magnetometers disagree beyond MagDisagreementT and nothing can
    @ attribute it: either no attitude solution exists, or its sigma is above
    @ MagMaxAttSigmaRad so the rotated reference is not worth believing. **No
    @ magnetic pair this cycle** — the coarse chain refuses TRIAD and the MEKF skips
    @ the magnetic update, which is the estimator's existing dropout behaviour.
    @ Edge-gated.
    @
    @ Cheaper than the IMU equivalent by design: this costs one of two vector
    @ pairs, not the body rate, so the vehicle keeps propagating and — with the Sun
    @ in view — keeps acquiring.
    @ Action: as ImuVoteAmbiguous. The onboard response is refusal plus the
    @ persistence escalation below; choosing between two disagreeing units without
    @ evidence is what the refusal exists to avoid. Recovery is a ground action.
    event MagVoteAmbiguous(fieldDifferenceTesla: F64) \
      severity warning high \
      format "Two magnetometers disagree by {} T and nothing can attribute it: no magnetic pair"

    @ The unattributable magnetometer disagreement above has persisted for
    @ ImuAmbiguityEscalateCycles. Re-emitted at the same period while it continues.
    @ Shares that horizon with the IMU escalation deliberately: how long the ground
    @ should wait before a persistent refusal reaches a console is a property of the
    @ operations concept, not of the sensor.
    @ Action: as ImuVoteAmbiguousPersistent — disambiguate from telemetry, then
    @ RESET_ESTIMATOR once the faulty unit is known, or widen MagDisagreementT if
    @ the pair is merely out of family. An autonomous FDIR mode response is
    @ deferred to the §9 FDIR push.
    event MagVoteAmbiguousPersistent(durationSec: F64, cyclesWithoutPair: U32) \
      severity warning high \
      format "Magnetometer disagreement unattributable for {} s ({} cycles without a magnetic pair): ground intervention required"

    @ A measurement the fine solution is **not** using has disagreed with it past
    @ its threshold for MonitorAlertCycles consecutive cycles (design doc §8.2,
    @ §9.2; REQ-ADET-012). Re-emitted at that same period while the condition
    @ lasts, so a permanent fault stays visible without becoming per-cycle noise.
    @
    @ This is what the mode ladder buys beyond accuracy. With a tracker fused the
    @ sun and magnetic pairs are no longer folded in, so instead of being silently
    @ down-weighted they are checked — and a sensor that has drifted becomes
    @ *observable*. SUN_CROSS_UNIT is the one monitor that needs no fine solution at
    @ all: it compares two sun sensors against each other, so it runs in Safe mode
    @ too.
    @ Action: SUN means the selected sun sensor disagrees with the tracker solution
    @ — suspect that unit's mounting or a contaminated field of view, and check
    @ SunUnitSelected. MAGNETOMETER usually means the hard-iron signature has moved
    @ (MAG_CAL_START) or the IGRF snapshot is stale (MagneticReferenceStale).
    @ SUN_CROSS_UNIT names two disagreeing units without saying which is wrong; the
    @ estimator resolves it against the fine solution where it can
    @ (SunUnitOverridden) and otherwise leaves it to the ground.
    event ResidualMonitorAlert(monitor: ResidualMonitor, residualRad: F64, thresholdRad: F64) \
      severity warning high \
      format "Residual monitor {}: {} rad against a {} rad threshold"

    @ A residual monitor that had alerted came back inside its threshold. The
    @ recovery edge, so the ground can close the condition without waiting to
    @ notice that the alerts stopped.
    @ Action: none; informational.
    event ResidualMonitorCleared(monitor: ResidualMonitor) \
      severity activity high \
      format "Residual monitor {} back within threshold"

    @ The sun-sensor cross-check found the **selected** unit disagreeing with the
    @ runner-up, and the fine solution agreed better with the runner-up — so the
    @ estimator used the runner-up instead for this cycle (design doc §8.2).
    @
    @ Only ever emitted once the SUN_CROSS_UNIT monitor has already alerted, i.e.
    @ after MonitorAlertCycles of persistent disagreement, so a single noisy sample
    @ never moves the selection. The sigma-ordered selector is what makes this
    @ necessary: it takes the smallest *reported* sigma, and a unit that is
    @ confidently wrong reports a small sigma and wins.
    @ Action: the overridden unit is a calibration or mounting suspect. Nothing is
    @ latched — the override is re-decided every cycle from the current evidence —
    @ so a unit that recovers simply stops being overridden.
    event SunUnitOverridden(selected: U8, used: U8, angleRad: F64) \
      severity warning low \
      format "Sun sensor {} disagrees with the fine solution; using unit {} instead ({} rad apart)"

    @ One star tracker has been latched out of the fusion: the filter's chi-square
    @ gate rejected its solutions on @p nisStreak consecutive cycles (design doc
    @ §8.2; REQ-ADET-012). The **mode is not demoted** — that is the point of a
    @ per-unit streak. A cycle-global one would demote the fine mode for a
    @ single bad unit, drop the filter, re-promote off the same bad unit and flap
    @ at the streak period; isolating the unit keeps the solution running on the
    @ trackers that are still believed.
    @ Action: the unit's solutions disagree with an attitude the *other* trackers
    @ built, so suspect that unit — a mis-identified star field, a stale or
    @ cross-wired solution, or a mounting that has moved. It is re-admitted
    @ automatically once it agrees with the fine solution again
    @ (StUnitReadmitted); a unit that never does needs ground investigation, and
    @ RESET_ESTIMATOR is the commanded path back.
    event StUnitExcluded(unit: U8, nisStreak: U32) \
      severity warning high \
      format "Star tracker {} excluded from the fusion after {} consecutive NIS rejections"

    @ A latched-out star tracker agreed with the fine solution for MonitorAlertCycles
    @ consecutive cycles and is contributing again.
    @ Action: none; informational. A unit that flaps between this and
    @ StUnitExcluded is marginal — investigate rather than leaving it cycling.
    event StUnitReadmitted(unit: U8) \
      severity activity high \
      format "Star tracker {} re-admitted to the fusion"

    @ An inter-star-tracker alignment collection window opened (design doc §8.2).
    @ The estimator keeps running normally: collection is a tap on the tracker
    @ path, not a mode.
    @ Action: watch StAlignSamples rise. A window whose count stalls has lost one of
    @ the two trackers to a keep-out or a fault; ST_ALIGN_CAL_ABORT and re-fly it
    @ when both are solving.
    event StAlignStarted(unit: U8, targetSamples: U32) \
      severity activity high \
      format "Inter-tracker alignment collection started on unit {}: {} pairs"

    @ An alignment fit succeeded and is now applied to that tracker: its solutions
    @ reach the filter stated in the king's frame.
    @ Action: grade it. residualRad is the fit quality — it should sit at the two
    @ units' combined per-sample noise. misalignRad is what was found, and it should
    @ match the mounting tolerance; far larger means a mounting or a
    @ StBoresightsBody error rather than a calibration success. **Nothing is written
    @ to ParameterDb**: the correction does not survive a reboot (§23.6 owns
    @ non-volatile state), so the window must be re-flown after one.
    event StAlignComplete(unit: U8, residualRad: F64, misalignRad: F64, samples: U32) \
      severity activity high \
      format "Inter-tracker alignment fitted on unit {}: residual={} rad, misalignment={} rad, {} pairs"

    @ An alignment fit was refused; nothing is applied and any previously applied
    @ alignment for that unit is **retained** — a failed later window is no
    @ information about the correction already flying.
    @ Action: named by the reason. SAMPLES means collect longer, or both trackers
    @ were not solving together. DEGENERATE means the pairs do not describe one
    @ fixed rotation, which is a unit fault rather than a collection problem — do
    @ not simply re-fly it. DISPERSION means they do, but too loosely. CONFIG and
    @ UNIT are command/parameter errors and cost the vehicle nothing.
    event StAlignRejected(unit: U8, reason: StAlignRejectReason, samples: U32) \
      severity warning high \
      format "Inter-tracker alignment on unit {} rejected: {} after {} pairs"

    @ An alignment collection window was closed without fitting.
    @ Action: none; informational.
    event StAlignAborted(unit: U8, samples: U32) \
      severity activity high \
      format "Inter-tracker alignment collection on unit {} aborted after {} pairs"

    @ An applied alignment was dropped; that tracker reverts to its as-mounted
    @ reading.
    @ Action: none; informational. Re-fly ST_ALIGN_CAL_START to replace it.
    event StAlignCleared(unit: U8) \
      severity activity high \
      format "Inter-tracker alignment cleared on unit {}"

    @ No valid orbit solution was available, so the geomagnetic reference could
    @ not be evaluated and the magnetic pair was excluded this cycle. With no
    @ magnetic pair there is no TRIAD, so the estimator gyro-coasts. Edge-gated.
    @ Action: check the OrbitEstimator's telemetry — the orbit filter coasts a
    @ receiver outage for its coast horizon (§8.3), so this fires only once the
    @ solution has been dropped, or before it was ever seeded. Sustained loss
    @ ends in AttitudeLost once the attitude coast horizon expires.
    event PositionUnavailable \
      severity warning low \
      format "No valid orbit solution: magnetic reference unavailable, coasting"

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

    @ A magnetometer calibration window opened (design doc §8.1). The estimator
    @ keeps running unchanged; only the sampling tap is new.
    @ Action: watch MagCalCoverage climb. If it plateaus well below
    @ MagCalMinCoverage the window cannot succeed — MAG_CAL_ABORT and re-fly it
    @ over a larger tumble rather than waiting it out.
    event MagCalStarted(targetSamples: U32) \
      severity activity high \
      format "Magnetometer calibration started: collecting {} samples"

    @ A calibration window closed and the fit was accepted; the correction is now
    @ applied to every magnetometer consumer. The reported residual is the
    @ angle-equivalent magnitude residual — see MagCalResidualAngle for what it
    @ does and does not bound.
    @ Action: grade the fit. A residual in the few-milliradian class is the
    @ expected result on the reference budget. Then consider the **re-derivation
    @ the calibration forces**: SigmaMagSysRad describes an uncalibrated
    @ magnetometer, and SeedMinObservability was derived from the *ratio* of the
    @ sun and magnetic sigmas — a much tighter magnetic pair makes the shipped
    @ 0.0076 refuse every geometry in the band, and 5.1e-4 preserves its 10 deg
    @ meaning with the albedo correction also in force. Uplink both before
    @ relying on fine mode (SigmaMagWhiteRad is unchanged: the fit removes the
    @ systematic, not the sensor noise). §8.1 and flight/PolarisFsw/README.md
    @ carry the procedure.
    event MagCalComplete(residualAngleRad: F64, coverage: F64, samples: U32) \
      severity activity high \
      format "Magnetometer calibration applied: residual={} rad, coverage={}, samples={}"

    @ A calibration fit was refused; **nothing was applied** and any previously
    @ applied calibration is retained untouched. The reason names the gate.
    @ Action: SAMPLES — collect longer. COVERAGE — re-fly over a larger tumble.
    @ CONDITION or NUMERICAL — the window is unobservable or the data is
    @ pathological; check magnetometer health and the IGRF reference before
    @ re-flying. NO_IMPROVEMENT — the sensor is already as good as this fit can
    @ make it, which is a *result*, not a fault. CONFIG — uplink the missing
    @ MagCal* parameters (PRM_SET + PRM_SAVE), then re-command.
    event MagCalRejected(reason: MagCalRejectReason, samples: U32, coverage: F64) \
      severity warning high \
      format "Magnetometer calibration refused: reason={}, samples={}, coverage={}"

    @ A collection window was closed by command without fitting.
    @ Action: none; the operator asked for it.
    event MagCalAborted(samples: U32) \
      severity activity high \
      format "Magnetometer calibration aborted: {} samples discarded"

    @ The applied magnetometer calibration was dropped and every consumer is back
    @ on the raw field measurement — by MAG_CAL_CLEAR, or as part of the full
    @ RESET_ESTIMATOR semantics. Also the event whose *absence* after a reboot
    @ tells the story: a calibration is not persisted (§23.6 deferral), so a
    @ vehicle that comes up with no MagCalComplete in its log is running raw.
    @ Action: re-fly MAG_CAL_START if the correction is still wanted, and revert
    @ any re-derived SigmaMag*/SeedMinObservability values with it.
    event MagCalCleared \
      severity activity high \
      format "Magnetometer calibration cleared: magnetometer reverted to raw"

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

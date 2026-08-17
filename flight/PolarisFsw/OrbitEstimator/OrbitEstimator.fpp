module flight {

  # ----------------------------------------------------------------------
  # Onboard orbit determination (design doc §8.3, §9.2; REQ-ODP-001, -005, -006)
  # ----------------------------------------------------------------------
  #
  # The F´ seam around `polaris::gnc::OrbitOd`: the 6-state position/velocity
  # error-state filter that turns GNSS PVT fixes into the orbit half of the
  # canonical onboard state (§8.0). It runs on the barrier-driven 10 Hz GNC rate
  # group (§2.4) as the member **before** the attitude estimator, so the position
  # the estimator's magnetic and sun references are evaluated at is this
  # cycle's, and it publishes `OrbitEstimate` on every cycle — valid or not.
  #
  # No filter math lives here. The propagation, the GPS->TAI and ECEF->ECI
  # ingest (REQ-CONV-001), the latency correction, the NIS gates, the coast
  # horizon and the plausibility band are all in the flight-safe `lib/gnc`;
  # this component is the wiring, the parameters, the EOP query and the
  # telemetry/EVR surface — the split the AttitudeEstimator follows.
  #
  # **What the coast horizon buys the vehicle.** Until this component existed
  # the attitude estimator placed its magnetic reference on the raw GNSS fix, so
  # a receiver outage cost the magnetic pair on the next cycle. Through this
  # seam a dropout is coasted on the onboard force model for `MaxCoastS`
  # (300 s, §8.3), then the solution is **dropped** and the next fix re-seeds
  # whole — never blended against a stale prior (§8.3).
  #
  # **Passive, guarded, and the same threading argument as the estimator.** The
  # fixes arrive on the producer's thread (SitlBridge's under SITL, a receiver
  # driver's on hardware) while `run` executes on the rate group's, and the
  # command arrives on the dispatcher's; every port that touches the fix latch
  # or the filter is `guarded`. There is no re-entry risk: nothing this
  # component calls calls back into it.
  passive component OrbitEstimator {

    # ----------------------------------------------------------------------
    # Rate group
    # ----------------------------------------------------------------------

    @ Filter cycle entry, at the GNC rate (10 Hz, §2.4). Propagates to now,
    @ folds in a fresh fix if one arrived, publishes the solution.
    guarded input port run: Svc.Sched

    # ----------------------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------------------

    @ GNSS PVT fixes, one port per receiver in vehicle build order (§19.4).
    @ Each slot latches the newest fix and marks it fresh; `run` consumes the
    @ freshest valid one and clears the mark, so a fix is folded in exactly
    @ once — re-presenting one is the stuck-clock case the filter refuses.
    guarded input port gnssIn: [GncMaxUnits] GnssMeasPort

    # ----------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------

    @ EOP at an epoch, from OnboardTables (§11.3): the ECEF->ECI reduction on
    @ ingest and the Earth orientation the force model needs per propagation.
    @ An epoch the table does not cover refuses the cycle rather than
    @ converting on an extrapolated Earth orientation.
    output port getEopAt: GetEopAt

    @ The orbit solution, published every cycle. `valid` false when there is
    @ none — a consumer MUST treat position as unavailable then (§9.1).
    output port orbitStateOut: OrbitEstimatePort

    # ----------------------------------------------------------------------
    # Types
    # ----------------------------------------------------------------------

    @ Why a fix or a cycle was refused — the FPP mirror of
    @ `polaris::gnc::OrbitOdRefusal`, kept in the same order and pinned to it
    @ by a static_assert in the component so the two cannot drift.
    enum OdRefusal : U8 {
      NONE = 0
      UNCONFIGURED = 1 @< the tuning failed OrbitOdConfig::isValid
      UNINITIALISED = 2 @< no solution yet, and this entry point cannot seed one
      NON_MONOTONIC_EPOCH = 3 @< the epoch is not strictly after the last (backwards or stuck)
      STEP_TOO_LONG = 4 @< the propagation gap exceeds MaxDtS
      COAST_EXPIRED = 5 @< past MaxCoastS; the solution was dropped
      FIX_NOT_FINITE = 6 @< a non-finite component in the fix
      FIX_IMPLAUSIBLE = 7 @< fix radius outside [MinRadiusM, MaxRadiusM] (§9.1)
      FIX_SIGMA_INVALID = 8 @< a reported sigma that is not positive and finite
      FRAME_CONVERSION = 9 @< ECEF->ECI failed, or the epoch is outside EOP coverage
      NO_VELOCITY_FOR_SEED = 10 @< a seed or a latent fix needs a velocity and the fix has none
      MEASUREMENT_REJECTED = 11 @< the NIS gate refused the fix
      FILTER_FAULT = 12 @< non-finite internal result; the solution was dropped
    }

    # ----------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------

    @ Drop the solution (cold start): the next fix re-seeds whole. Configuration
    @ is retained and the rejection counter cleared — the operator saying
    @ "start over", as distinct from a coast expiry or a fault, which drop the
    @ solution and keep the count.
    guarded command OD_RESET \
      opcode 0

    # ----------------------------------------------------------------------
    # Parameters (design doc §19.3 — no defaults; a missing value refuses).
    # GM and the reference radius are **not** parameters: they are paired with
    # the compiled-in coefficient table (constants::gravity) and a parameter
    # would let an uplink separate them (§8.3, the constant-pairing defect).
    # ----------------------------------------------------------------------

    @ Harmonic degree of the compiled-in EGM2008 truncation (0..8). **Zero**
    @ selects the closed-form two-body + J2 path with `constants::gravity::kJ2`
    @ — the flight never flies pure two-body.
    param GeopotentialDegree: U32

    @ Harmonic order (0..GeopotentialDegree). Order 0 is a purely zonal field.
    param GeopotentialOrder: U32

    @ Ballistic coefficient C_d*A/m [m^2/kg]. Zero disables drag.
    param DragBallisticCoeffM2PerKg: F64

    @ Exponential-atmosphere reference density rho_0 [kg/m^3] at
    @ DragRefAltitudeM — one band of the Vallado Table 8-4 fit the truth sim
    @ interpolates, chosen at the operating altitude (§8.3).
    param DragRefDensityKgM3: F64

    @ Reference altitude h_0 [m] the band's density is stated at.
    param DragRefAltitudeM: F64

    @ Scale height H [m] of the band.
    param DragScaleHeightM: F64

    @ Process-noise acceleration PSD q_a [m^2/s^3], sized from the measured
    @ force-model truncation over the coast horizon, q_a = 3*dr(T)^2/T^3 (§8.3).
    param AccelPsdM2PerS3: F64

    @ NIS gate on the 3-row position update [-]; chi-square(3), 99.9% = 16.27.
    param PositionNisGate: F64

    @ NIS gate on the 3-row velocity update [-]; carried separately because
    @ the two measurements have different error budgets.
    param VelocityNisGate: F64

    @ Coast horizon [s]: time since the last accepted fix past which the
    @ solution is dropped (§8.3, §9.2). An FDIR-visible validity policy.
    param MaxCoastS: F64

    @ Longest single propagation step accepted [s]; a larger gap is a clock
    @ fault, refused rather than integrated across.
    param MaxDtS: F64

    @ RK4 sub-step ceiling [s]. MaxDtS must be at most 64 sub-steps of it.
    param MaxStepS: F64

    @ Largest fix latency corrected for [s]: a fix older than this against the
    @ filter epoch is a clock fault. **Paired with the receiver's catalogued
    @ `fix_latency_s`** (§19.4): configc refuses a value below it.
    param MaxFixLatencyS: F64

    @ Lower edge of the geocentric-radius plausibility band [m] (§9.1).
    param MinRadiusM: F64

    @ Upper edge of the plausibility band [m].
    param MaxRadiusM: F64

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------

    @ Estimated position, ECI [m].
    telemetry PosEciM: Vec3F64

    @ Estimated velocity, ECI [m/s].
    telemetry VelEciMps: Vec3F64

    @ sqrt(trace) of the position covariance [m].
    telemetry PosSigmaM: F64

    @ sqrt(trace) of the velocity covariance [m/s].
    telemetry VelSigmaMps: F64

    @ Time since the last accepted fix [s]; grows through an outage.
    telemetry SolutionAgeS: F64

    @ The solution exists and is inside MaxCoastS.
    telemetry SolutionValid: bool

    @ NIS of the last position update, accepted or not [-].
    telemetry PositionNis: F64

    @ NIS of the last velocity update, accepted or not [-].
    telemetry VelocityNis: F64

    @ Realised latency of the last fix [s]. Watched because a latency that
    @ climbs is invisible in the residuals precisely because the filter
    @ corrects for it (§8.3).
    telemetry FixLatencyS: F64

    @ Fixes that seeded or updated the solution since start or OD_RESET.
    telemetry FixesAccepted: U32

    @ Fixes refused for any reason since start or OD_RESET.
    telemetry FixesRefused: U32

    @ The most recent refusal.
    telemetry LastRefusal: OdRefusal

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ A fix seeded the filter — the first fix, or the first after a drop.
    event OrbitSeeded(unit: U8, posSigmaM: F64) \
      severity activity high \
      format "Orbit solution seeded from GNSS unit {} (position sigma {} m)"

    @ The solution was dropped at the coast horizon.
    event OrbitSolutionDropped(ageSec: F64, maxCoastSec: F64) \
      severity warning high \
      format "Orbit solution dropped: {} s since the last accepted fix, coast horizon {} s"

    @ A fix was refused; edge-gated per reason so a stream of the same refusal
    @ logs once, and a change of reason logs again.
    event FixRefused(unit: U8, reason: OdRefusal, count: U32) \
      severity warning low \
      format "GNSS fix from unit {} refused: {} ({} refusals so far)"

    @ A parameter is missing or the set fails validation; the filter is inert
    @ until a valid set arrives (§19.3).
    event OrbitTuningInvalid(detail: string size 80) \
      severity warning high \
      format "OrbitEstimator tuning invalid: {}"

    @ OnboardTables could not answer the EOP query for this epoch (unconnected,
    @ or outside coverage). Edge-gated.
    event EopUnavailable(taiNs: I64) \
      severity warning high \
      format "EOP unavailable at TAI {} ns: orbit filter cannot convert or propagate"

    @ A non-finite internal result dropped the solution.
    event OrbitFilterFault \
      severity warning high \
      format "Orbit filter internal fault: solution dropped, awaiting re-seed"

    @ OD_RESET accepted.
    event OrbitReset \
      severity activity high \
      format "Orbit estimator reset: solution dropped, next fix re-seeds"

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    time get port timeCaller
    param get port prmGetOut
    param set port prmSetOut
    command reg port cmdRegOut
    command recv port cmdIn
    command resp port cmdResponseOut
    event port logOut
    text event port logTextOut
    telemetry port tlmOut
  }
}

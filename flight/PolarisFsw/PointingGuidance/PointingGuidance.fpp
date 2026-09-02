module flight {

  @ Align/constrain pointing guidance and the target stores behind it
  @ (design doc §8.4; REQ-AGN-004, REQ-AGN-005, REQ-ODP-002).
  @
  @ Turns one operator command — align a body vector with an inertial target,
  @ constrain a second body vector toward a second target — into the attitude and
  @ body rate the §8.5 controller tracks, once per GNC cycle.
  @
  @ **Why this is a component and not part of AttitudeController.** The solve
  @ needs the orbit state, the Sun/Moon ephemeris and Earth orientation; the
  @ controller has none of those and should not grow ports for them. Keeping the
  @ split means the controller stays a controller: it receives a target attitude
  @ and a feedforward rate and does not know or care whether they came from a
  @ fixed quaternion, a nadir hold or a satellite track.
  @
  @ **Why the target stores live here too.** The TLE slots, state-vector slots,
  @ ground points and custom body vectors exist only to be pointed at. Putting
  @ them in the component that consumes them means one place validates an upload
  @ and one place reports what is loaded, and no port array has to carry a
  @ catalogue across a component boundary every cycle.
  @
  @ Passive, like the controller it feeds: the solve runs inline on the GNC rate
  @ group after the orbit estimator, so the target the controller acts on was
  @ computed from this step's orbit solution rather than from the previous one.
  @ The stores are shared between that cycle and the uplink commands that write
  @ them, so both are `guarded` — half a guarded pair is no mutual exclusion at
  @ all, which is the note `AttitudeController` already carries about its own
  @ inputs.
  passive component PointingGuidance {

    # ----------------------------------------------------------------------
    # Scheduling and inputs
    # ----------------------------------------------------------------------

    @ Guidance cycle, at the GNC rate. Ordered after the orbit estimator and
    @ before the attitude controller on the same cycle, so the target the
    @ controller acts on was computed from this step's orbit solution.
    guarded input port run: Svc.Sched

    @ The §8.3 orbit solution. Every orbit-relative target — nadir, LVLH, and
    @ every line of sight — is measured from it, so a DEGRADED or absent solution
    @ is reported as a specific refusal rather than silently degrading pointing.
    guarded input port orbitStateIn: OrbitEstimatePort

    @ Geocentric ECI position of the Sun or Moon at a TAI epoch, for SUN and
    @ MOON targets (§11.3).
    output port getBodyPosition: GetBodyPosition

    @ Earth orientation at a TAI epoch. Needed **only** by ECEF_TARGET; every
    @ other target kind works through an EOP outage, which is why a stale table
    @ refuses ground-station pointing specifically instead of taking the whole
    @ guidance mode down.
    output port getEopAt: GetEopAt

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

    @ This cycle's commanded attitude and feedforward rate, to the controller.
    @ `valid` false means the guidance could not be solved this cycle and the
    @ controller must fall back rather than fly a stale target.
    output port guidanceOut: AttitudeTargetPort

    # ----------------------------------------------------------------------
    # Types
    # ----------------------------------------------------------------------

    @ A named direction in the body frame (`lib/gnc/pointing_refs.hpp`). The
    @ sensor kinds resolve through the *estimator's* mounting parameters rather
    @ than a second copy of those numbers, so a boresight exists once on the
    @ vehicle.
    enum BodyVecKind : U8 {
      BODY_X = 0
      BODY_Y = 1
      BODY_Z = 2
      STAR_TRACKER = 3 @< boresight of star tracker `index`
      SUN_SENSOR = 4 @< normal of sun sensor `index`
      CAMERA = 5 @< boresight of camera `index`
      CUSTOM_BODY_VEC = 6 @< operator-stored vector `index` (0..9)
    }

    @ A named direction in inertial space (`lib/gnc/pointing_refs.hpp`).
    enum TargetKind : U8 {
      SUN = 0
      MOON = 1
      NADIR = 2
      ECEF_TARGET = 3 @< stored ground point `index` (0..29)
      STAR_J2000 = 4 @< right ascension / declination in the command
      J2000_X = 5
      J2000_Y = 6
      J2000_Z = 7
      LVLH_X = 8 @< ~ +velocity
      LVLH_Y = 9 @< -orbit normal
      LVLH_Z = 10 @< nadir
      SAT_TLE = 11 @< TLE slot `index` (0..4)
      SAT_STATE = 12 @< state-vector slot `index` (0..4)
    }

    @ Why a guidance command was refused, or a cycle produced no target. The FPP
    @ mirror of `gnc::GuidanceStatus`, so an EVR can carry the reason an operator
    @ needs rather than a bare failure.
    enum GuidanceRefusal : U8 {
      SAME_BODY_AXIS = 0 @< align and constrain name one axis: it cannot point two ways
      BODY_VECTOR_UNKNOWN = 1 @< the unit is not installed, or the custom slot is empty
      BODY_AXES_PARALLEL = 2 @< two different names resolved to one direction
      TARGET_UNAVAILABLE = 3 @< a target has no position right now
      TARGET_SLOT_EMPTY = 4 @< a named slot has nothing in it
      BAD_PARAMETERS = 5 @< an index or angle is out of range
      NO_EPHEMERIS = 6 @< SUN/MOON named with no ephemeris answer
      NO_EARTH_ORIENTATION = 7 @< ECEF_TARGET named with no usable EOP
      NO_ORBIT_STATE = 8 @< an orbit-relative target with no usable orbit solution
      DIRECTIONS_COLLINEAR = 9 @< the two targets are collinear *now*; roll is undetermined
      BAD_INPUT = 10 @< a non-finite value reached the solver
      NOT_COMMANDED = 11 @< no guidance command has been set since reset
    }

    # ----------------------------------------------------------------------
    # Commands — the pointing command
    # ----------------------------------------------------------------------

    @ Set the complete align/constrain pointing command (§8.4).
    @
    @ **One command, not four.** The align pair and the constrain pair are
    @ validated *against each other* — the two body vectors must not be the same
    @ axis, and must not resolve to one direction — so accepting them separately
    @ would leave the vehicle holding a half-updated, unvalidated pair between two
    @ uplinks. There is no state in which only half a pointing command applies.
    @
    @ Rejected commands leave the previous guidance in force and raise
    @ `GuidanceCommandRefused` with the reason. Note what is deliberately *not*
    @ checked here: whether the two targets are collinear. That is a property of
    @ the sky at a moment rather than of the command — a constraint that is fine
    @ now can degenerate an orbit later as the Sun, the target and the vehicle
    @ line up — so it is re-checked every cycle and reported then.
    guarded command SET_GUIDANCE(
                                  alignVecKind: BodyVecKind @< body vector to align
                                  alignVecIndex: U8 @< unit or custom slot
                                  alignVecNegate: bool @< use the opposite direction
                                  alignTgtKind: TargetKind @< what to align it with
                                  alignTgtIndex: U8 @< slot, where the kind takes one
                                  alignTgtNegate: bool
                                  alignTgtParam0: F64 @< STAR_J2000 right ascension [rad]
                                  alignTgtParam1: F64 @< STAR_J2000 declination [rad]
                                  conVecKind: BodyVecKind @< body vector to constrain
                                  conVecIndex: U8
                                  conVecNegate: bool
                                  conTgtKind: TargetKind @< what to bring it toward
                                  conTgtIndex: U8
                                  conTgtNegate: bool
                                  conTgtParam0: F64 @< STAR_J2000 right ascension [rad]
                                  conTgtParam1: F64 @< STAR_J2000 declination [rad]
                                ) \
      opcode 0

    @ Forget the pointing command. The next cycle publishes an invalid target and
    @ the controller falls back — the commanded way to stop tracking without
    @ having to name a replacement.
    guarded command CLEAR_GUIDANCE \
      opcode 1

    # ----------------------------------------------------------------------
    # Commands — the target stores
    # ----------------------------------------------------------------------

    @ Load a two-line element set into TLE slot `slot` (0..4).
    @
    @ The lines are parsed, the epoch recovered and the propagator initialised
    @ before anything is stored, so a rejected upload leaves the previous target
    @ in service — a half-loaded target is worse than a stale one.
    @
    @ `verifyChecksum` defaults on in operations: an uplinked TLE with one flipped
    @ character is a plausible orbit somewhere else, not obvious garbage. It is
    @ waivable because hand-written element sets — including the official AIAA
    @ verification fixture's own lines — legitimately carry stale checksums.
    @
    @ **Each 69-column line arrives in two halves, and that is the framework's
    @ constraint rather than a design choice.** A `string size N` in a command
    @ declaration is documentation: the autocoder emits `Fw::CmdStringArg` for
    @ every command string, whose capacity is the framework-wide
    @ `FW_CMD_STRING_MAX_SIZE` — **40** in the F´ default config this deployment
    @ uses. A 69-column line handed to that type is silently truncated to 40,
    @ which is what this command did until a SITL row tried to fly a TLE and was
    @ refused with a parse error whose real cause was five layers away. The
    @ alternative was forking F´'s config directory to raise a global constant
    @ for one command; halves are the smaller change. The handler concatenates
    @ and refuses anything that is not exactly 69 columns, so a truncation can
    @ never again be discovered as a mysterious parse failure.
    guarded command LOAD_TLE(
                              slot: U8 @< 0..4
                              line1a: string size 40 @< TLE line 1, columns 1-35
                              line1b: string size 40 @< TLE line 1, columns 36-69
                              line2a: string size 40 @< TLE line 2, columns 1-35
                              line2b: string size 40 @< TLE line 2, columns 36-69
                              verifyChecksum: bool @< true in operations
                            ) \
      opcode 2

    @ Load an osculating ECI state into state-vector slot `slot` (0..4),
    @ propagated with two-body + J2 and **no drag** (§8.3).
    @
    @ `sigmaM` is the ground solution's own 1-sigma position accuracy at the
    @ epoch. It is carried rather than assumed because it is the only thing that
    @ makes the propagated uncertainty meaningful, and a ground OD always knows
    @ it.
    guarded command LOAD_STATE_VECTOR(
                                       slot: U8 @< 0..4
                                       epochTaiNs: I64 @< TAI ns the state is valid at
                                       posXM: F64
                                       posYM: F64
                                       posZM: F64
                                       velXMps: F64
                                       velYMps: F64
                                       velZMps: F64
                                       sigmaM: F64 @< 1-sigma position accuracy at epoch [m]
                                     ) \
      opcode 3

    @ Empty a target slot so naming it is refused again.
    guarded command CLEAR_TARGET_SLOT(
                                       isTle: bool @< true for a TLE slot, false for state vector
                                       slot: U8 @< 0..4
                                     ) \
      opcode 4

    @ Store an Earth-fixed point in ground-point slot `slot` (0..29) — a ground
    @ station, imaging site or calibration target.
    @
    @ Geodetic rather than Cartesian ECEF, because that is how a station is
    @ published and checked by a human: storing the Cartesian form would make the
    @ uplinked number differ from the one in the mission document, which is the
    @ flight/sim parameter-pair failure class in a new place. The height bound is
    @ deliberately wide (-1 km to +100 km) — it exists to catch a
    @ metres-versus-kilometres unit error, the mistake that actually happens, not
    @ to police where a site may be.
    @
    @ Deciding *which* station to point at and when is not here; this is the
    @ storage that makes such a thing possible without another upload path.
    guarded command SET_GROUND_POINT(
                                      slot: U8 @< 0..29
                                      latitudeDeg: F64 @< geodetic latitude [deg]
                                      longitudeDeg: F64 @< east longitude [deg]
                                      heightM: F64 @< height above the WGS-84 ellipsoid [m]
                                    ) \
      opcode 5

    @ Forget a stored ground point.
    guarded command CLEAR_GROUND_POINT(
                                        slot: U8 @< 0..29
                                      ) \
      opcode 6

    @ Store a custom body vector in slot `slot` (0..9), normalised on receipt.
    @
    @ For what mounting parameters cannot cover: an antenna, a thruster axis, a
    @ payload aperture added after the parameter set was frozen. A sensor
    @ boresight must **not** be duplicated here — it already exists as
    @ `AttitudeEstimator.StBoresightsBody` and its siblings, and a second copy
    @ would drift silently from the first.
    guarded command SET_CUSTOM_BODY_VEC(
                                         slot: U8 @< 0..9
                                         x: F64
                                         y: F64
                                         z: F64
                                       ) \
      opcode 7

    @ Forget a custom body vector.
    guarded command CLEAR_CUSTOM_BODY_VEC(
                                           slot: U8 @< 0..9
                                         ) \
      opcode 8

    # ----------------------------------------------------------------------
    # Parameters (§19.3 — no defaults; a missing value refuses)
    # ----------------------------------------------------------------------

    @ Star-tracker boresights in body axes, per unit — the same convention and
    @ the same numbers as `AttitudeEstimator.StBoresightsBody`. Cross-checked
    @ against it by `configc` rather than trusted to stay in step: two
    @ independently maintained copies of a mounting vector is precisely the
    @ failure this repository keeps finding.
    param StBoresightsBody: Vec3F64PerUnit

    @ Sun-sensor normals in body axes, per unit; mirrors
    @ `AttitudeEstimator.SunAlbedoBoresightsBody`.
    param SunSensorBoresightsBody: Vec3F64PerUnit

    @ Camera boresights in body axes, per unit. Unlike the two above these have
    @ no estimator counterpart — a camera is not an attitude sensor — so this is
    @ the only definition of them on the vehicle.
    param CameraBoresightsBody: Vec3F64PerUnit

    @ Geopotential degree and order the **state-vector** target slots are
    @ propagated with (§8.3). TLE slots are unaffected: SGP4 is the theory a TLE
    @ was fitted with and is not a truncation choice.
    @
    @ **8 x 8 by default**, the same EGM2008 truncation the orbit filter already
    @ flies, because measurement said the previous two-body + J2 cost ~13 m over
    @ 900 s and ~1.3 km over the propagator's 12 h horizon — more error than the
    @ slot's own uncertainty model claimed for everything it omits.
    @
    @ **Order is also the EOP dependency.** A zonal harmonic is axisymmetric and
    @ needs no Earth-rotation angle; anything above order 0 does. So
    @ `TargetGeopotentialOrder: 0` is the setting that keeps target propagation
    @ working through an EOP outage, and `degree 2, order 0` reproduces the old
    @ two-body + J2 exactly. The propagator additionally falls back to order 0 by
    @ itself on any cycle the EOP tables cannot answer, so this parameter chooses
    @ the *ceiling*, not a promise.
    @
    @ Cost, measured on the flight tuning: 34 us per 10 Hz cycle in steady state
    @ (0.034 % of the cycle), because the propagator carries a grid-anchored
    @ cursor. The first call after an upload catches the whole span up at once —
    @ 3 ms for a 15-minute-old state, 141 ms at the 12 h bound.
    param TargetGeopotentialDegree: U8

    @ See TargetGeopotentialDegree. Clamped to that degree; 0 is the zonal,
    @ EOP-free setting.
    param TargetGeopotentialOrder: U8

    @ Maximum age of an orbit solution the guidance will use [s]. Beyond it every
    @ orbit-relative target is refused with NO_ORBIT_STATE rather than pointed
    @ from a coasted state whose error nobody bounded.
    param MaxOrbitStateAgeSec: F64

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------

    @ True when this cycle produced a usable attitude target.
    telemetry GuidanceValid: bool

    @ The most recent refusal reason, latched until the next successful cycle.
    telemetry LastRefusal: GuidanceRefusal

    @ Consecutive cycles the guidance has failed to solve. A climbing count with
    @ a stable reason is a diagnosis; a flickering one is a geometry problem.
    telemetry RefusalStreak: U32

    @ Angular separation between the commanded boresight and where it actually
    @ points, from the controller's own error [rad]. Zero when not tracking.
    telemetry PointingErrorRad: F64

    @ Range to the aligned target [m], when the target has one. Zero for a pure
    @ direction such as a star or an inertial axis.
    telemetry TargetRangeM: F64

    @ 1-sigma position uncertainty of the aligned target [m], when it has one —
    @ so a stale TLE is legibly worse than a fresh ground solution rather than
    @ looking identical.
    telemetry TargetSigmaM: F64

    @ Magnitude of the commanded feedforward body rate [rad/s].
    telemetry CommandedRateRadS: F64

    @ Occupied TLE slots, state-vector slots, ground points and custom vectors.
    telemetry TleSlotsUsed: U8
    telemetry StateSlotsUsed: U8
    telemetry GroundPointsUsed: U8
    telemetry CustomVecsUsed: U8

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ A pointing command was accepted.
    event GuidanceCommanded(
                             alignVec: BodyVecKind
                             alignVecIndex: U8
                             alignTgt: TargetKind
                             alignTgtIndex: U8
                           ) \
      severity activity high \
      format "Guidance set: align {} [{}] with {} [{}]"

    @ A pointing command was refused; the previous guidance still stands.
    event GuidanceCommandRefused(
                                  reason: GuidanceRefusal
                                ) \
      severity warning low \
      format "Guidance command refused: {}"

    @ Guidance stopped producing a target. Throttled: a geometry that has become
    @ unsatisfiable stays unsatisfiable for many cycles, and one event per cycle
    @ would bury the transition that matters.
    event GuidanceLost(
                        reason: GuidanceRefusal
                      ) \
      severity warning high \
      format "Guidance lost: {}" \
      throttle 5

    @ Guidance recovered after a run of refusals.
    event GuidanceRecovered(
                             afterCycles: U32
                           ) \
      severity activity high \
      format "Guidance recovered after {} refused cycles"

    @ A target upload was accepted.
    event TargetLoaded(
                        isTle: bool
                        slot: U8
                      ) \
      severity activity high \
      format "Target loaded: TLE={} slot {}"

    @ Why an upload was refused. One value per operator action: a checksum or
    @ format error is a re-uplink, an impossible date is a ground-tool bug, and
    @ an element set SGP4 cannot start from is a different element set. These
    @ were one value until a SITL row was refused and the event said only
    @ "refused", which left guessing as the only recovery.
    enum TargetRefusal : U8 {
      BAD_SLOT = 0 @< slot index out of range
      BAD_ELEMENTS = 1 @< the two lines did not parse, or the checksum did not verify
      BAD_EPOCH = 2 @< the epoch fields do not form a real date
      BAD_ORBIT = 3 @< parsed, but the propagator cannot be initialised from it
      LINE_LENGTH = 4 @< the reassembled line is not 69 columns: the uplink truncated it
      OTHER = 5 @< a refusal with no more specific value
    }

    @ A target upload was refused; the slot is unchanged.
    event TargetLoadRefused(
                             isTle: bool
                             slot: U8
                             reason: TargetRefusal
                           ) \
      severity warning low \
      format "Target upload refused: TLE={} slot {} reason {} (slot unchanged)"

    @ A ground point or custom body vector was stored or refused.
    event StoreUpdated(
                        isGroundPoint: bool
                        slot: U8
                        stored: bool @< false when the slot was cleared
                      ) \
      severity activity low \
      format "Store updated: groundPoint={} slot {} stored={}"

    event StoreRefused(
                        isGroundPoint: bool
                        slot: U8
                      ) \
      severity warning low \
      format "Store update refused: groundPoint={} slot {}"

    # ----------------------------------------------------------------------
    # Standard component ports
    # ----------------------------------------------------------------------

    time get port timeCaller
    command recv port cmdIn
    command reg port cmdRegOut
    command resp port cmdResponseOut
    event port eventOut
    text event port textEventOut
    telemetry port tlmOut
    param get port prmGetOut
    param set port prmSetOut
  }

}

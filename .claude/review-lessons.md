# Review Lessons — Defect Patterns Found in Polaris Development

A living catalog of defect classes actually found in this repo's review cycles,
kept so every future review checks for them and every future implementation
avoids them. Ordered by category, each with the push where it was caught.
Reviewers: treat every entry as a standing checklist item. Implementers: read
before starting FDIR, estimator, or requirements work.

## FDIR / fault-tolerance state machines

- **Re-admission must be judged on the criterion that excluded the unit** (P51).
  A unit excluded by a comparison (outvoted) is plausible by construction, so a
  plausibility-based re-admission streak always completes → permanent
  exclude/re-admit flap, one FDIR event pair per lap. Gate failures earn back on
  the gate; identifications earn back on agreement.
- **Any reference used to identify a faulty unit must pass the same plausibility
  gates as the data it judges** (P51). A finiteness-only check on the reference
  let an absurd value latch out the *healthy* unit. A failed reference is no
  reference (ambiguous), never a decision.
- **Single-cycle, zero-margin identification is a coin flip** (P51). Under slow
  common-mode drift the residual margin at the detection cycle is far below the
  noise. Require a decisive comparison (loser outside the gate, winner inside)
  plus K-cycle confirmation before latching. Exact ties must resolve to
  ambiguous, never identify silently.
- **Per-unit fault attribution, not cycle-global** (P52). A cycle-global
  "rejected" flag OR'd across units let one bad tracker demote the whole fine
  mode into a promote/demote flap while a healthy king was accepted every cycle.
  Track streaks per unit; the mode-level response fires only when *no* unit is
  accepted.
- **The finest-mode source needs at least the fault tolerance of the sources
  below it** (P52). IMUs and MAGs got voters; the star trackers — the source the
  top rung is built on — initially got none.
- **Every documented alert action must actually fire — test the mapping** (P51).
  An EVR promised coast-horizon escalation that a gyro-only fault provably never
  triggers (the vector pairs keep TRIAD alive). Either implement the escalation
  (persistence-counted, bounded cadence) or write the honest no-action.
- **A filter's covariance is not entitled to veto a better source** (P53). A
  solution built from systematic-dominated sources has a covariance the white-`R`
  model has averaged *down* while the true error stayed put, so `S = HPHᵀ + R` is
  tiny against a residual that is the systematic. An innovation gate on that `S`
  then rejects an arriving instrument two orders of magnitude better — every
  cycle, permanently — and the per-unit streak policy latches out the **good**
  unit. Before gating source A against a solution built from source B, ask which
  of the two the design says is better; the gate is only evidence when the
  comparison is like for like. **Enacted in P53**
  (`AttitudeEstimator::arbitrateRejectedTrackers`): the verdict moved to the
  *coarse* covariance, which carries a systematic floor and therefore does not
  lie — inside 3σ the filter is re-seeded from the tracker, outside it the unit
  is excluded. Note the fix's own failure mode and close it in the same change:
  adopting a rejected source is right only when it really is the better one, so
  the guard that refuses a grossly wrong one is not optional. And the rule that
  P53's own first attempt broke: **any new exclusion must ship its re-admission
  criterion in the same change.** The fix latched a unit out on disagreement with
  the *coarse* solution while the existing parole test was agreement with the
  *fine* one — a criterion mismatch, i.e. a life sentence, written by the same
  push that cites this catalog. If you cannot name the test that ends the
  exclusion, do not latch: refuse the cycle and report it instead.
- **An uncalibrated unit is not a calibrated unit at full weight** (P52). The
  launch-state second tracker was fused with the calibrated sigma — a 2–5σ
  systematic sold as noise, which is also what fed the demotion flap. Don't fuse
  (or inflate R) until the calibration is valid.

## Circularity

- **A reference derived from a state the checked sensor feeds is circular**
  (P52, three separate instances). The mag-vote reference through an attitude the
  mag helped build could latch out the healthy magnetometer; the sun cross-check
  resolution against a TRIAD attitude that fits the sun exactly confirms the
  incumbent; residual monitors read healthy on the sensor being fused. Gate
  every such reference on a mode where it is genuinely independent (e.g. ST
  fine), and refuse otherwise.
- **The filter that identifies a fault must survive it** (P51/P52). Assert "no
  demotion" through an identification in tests — a demotion there exposes the
  circularity.

## Requirements & verification

- **Verify against the configuration the vehicle actually ships** (P50). The
  requirement campaign ran post-calibration parameters the YAML does not carry
  (they arrive by ground uplink after a fit). State the condition in the
  requirement; record the as-delivered floor honestly.
- **Margins are asserted, not printed** (P50). A margin that is only logged
  inverts silently. `EXPECT_LE(bound, margin_fraction * limit)`.
- **Thresholds are requirement values, never tuned to measurements** — and a
  committed threshold within noise of its own measurement is too tight (P52:
  0.020° committed vs 0.021° measured; the number moved with a test-harness
  refactor). Carry the declared margin (20%) in the threshold itself.
- **The sim/test selector must mirror the flight rule exactly** (P51). The MC
  verified a min-incidence sun selection the vehicle does not fly (flight:
  min-sigma, ties to index); the load-bearing coverage claim was about the
  wrong rule.

## Statistics

- **Paired comparisons need per-unit RNG substreams** (P52). Adding a unit to
  one campaign shifted the global stream, so "same budget, different unit
  count" compared two independent samples; a 3% median gap asserted on that is
  RNG-fragile. Dedicated substreams per unit + run-for-run win fractions.
- **Systematics are drawn once per run as constants, not per sample as white**
  (P45/P47). Modelling a systematic as white makes campaigns vacuously easy
  (1/√N) and truth models uncorrectable by construction.
- **NIS gates are accept ranges** (P42): `0 ≤ NIS ≤ gate`. A plain upper bound
  accepts negative NIS, which means a broken covariance, not a good fit. Require
  R symmetric-PD (Cholesky) before use.

## Numerics

- **`acos(clamp(a·b))` loses half its digits near 0 and π** (P49). Use the
  atan2 forms (house rule, lib/README.md / design doc §3.3).
- **Fixed-size Eigen members and arrays are NOT zero-initialized by `{}`**
  (P52). `Eigen::Vector3d a[N]{}` value-initializes via Eigen's default ctor,
  which leaves storage uninitialized — zeros on a fresh stack, garbage in a
  suite run. Initialize explicitly; grep for the pattern when touching test
  fixtures or structs with Eigen members.

## Physics / modeling

- **Truth-model errors must be correctable by construction** (P47). A per-sample
  random albedo error cannot be corrected by any onboard model; the physical
  deterministic pull can. If the flight software is supposed to correct it, the
  truth model must contain the structure being corrected.
- **Torque/force double-counting across providers** (P43). The gravity-gradient
  couple was already inside the spherical-harmonic provider when a dedicated
  provider was added. When adding an environment term, grep for where it may
  already be implicit.

## F´ / infrastructure

- **Static table sizes fail as runtime asserts, not build errors**: PrmDb
  entries (`PRMDB_NUM_DB_ENTRIES`, P46) and the command-dispatcher table
  (`CMD_DISPATCHER_DISPATCH_TABLE_SIZE = 150`, P52 — fires the moment a
  component registers, well before the param count matters). Check both when
  adding components, commands, or parameters; override via
  `flight/config/*ImplCfg.hpp` and verify in **both** build trees.
- **`gtest_discover_tests` without `WORKING_DIRECTORY` runs from the build
  tree** (P52). Repo-root-relative test paths then GTEST_SKIP — the whole SITL
  suite skipped silently in CI from P41 to P52. Skips are not passes: check
  ctest output for SKIPPED, and prefer hard failure over skip for
  must-run suites.
- **A passive component's sync handlers run on the caller's thread** (P46).
  Command-thread vs rate-group access to shared state = torn reads that are
  finite, plausible, and wrong. `guarded` ports for everything both threads
  touch — including RESET-style commands.
- **Parameters must actually be loaded** (P41): `readParameters()` is not
  called by the framework; a topology that never calls it flies defaults
  silently. New topology = verify the prm path end-to-end (PrmDb file → load →
  component sees the value).
- **Wire/topology every unit of a port array** (P52): only unit 0 of the new
  pairs was connected. When a suite grows to N, grep the topology for `[1]`.

## Process

- `pre-commit run --all-files` **skips untracked files** — `git add -A` first
  (CI catches what local missed otherwise).
- Incremental Sphinx builds lie — trust only clean `-E` builds for the docs
  gate.
- clang-format reformats new files on its first pass — re-run gates after.
- Test-count phrasing: report "N collected — M passed, K skipped", never a bare
  count that conflates them.
- **A fault injected on the macro-step seam misses the first sample** (P53). The
  closed loop samples the sensors and *then* calls back, so "faulted from step 0"
  still lets one clean cycle through — and one clean cycle is all the estimator
  needs to promote off a Davenport seed and change the mode the whole case was
  about. A row whose premise is "this source was never available" has to inject
  before the run starts.
- **A conditional assertion that never fires is not a test** (P53). `if (found) {
  EXPECT... }` is green whether the behaviour happened or not. When the run shows
  the branch is never taken, assert the behaviour that *does* happen and say why
  in the comment; the honest negative ("no exclusion ever, and here is the
  mechanism that makes it impossible") is worth more than a dormant positive.
- Stale docs, comments, and PR text are defects, not afterthoughts — every
  claim edited must be re-verified against the tree, and a hygiene change that
  introduces a new stale claim is worse than none.

## Cross-domain interactions (found by closed-loop tests, not by inspection)

- **A downstream monitor must read upstream of the gate that hides its fault**
  (P54). The §7 stuck-on rod monitor was fed the *voted* magnetometer field, and
  a stuck rod puts hundreds of microtesla on the sensor — which the §8.2
  plausibility band rejects as implausible *before* the vote runs. The monitor
  therefore went blind at exactly the disturbance it exists to name, and the SITL
  row failed with no event at all. Fixed by publishing the largest **raw**
  admitted magnitude alongside the voted field, explicitly labelled a diagnostic
  no estimator reads. General form: when a health monitor and a data-quality gate
  look at the same signal, being out of band is *evidence* for the monitor and a
  reason to look away for the gate — so the monitor has to sit upstream.
- **The vehicle's own actuators are a permanent signal in its own sensors**
  (P54). Torque rods keep a remanent moment after every de-energise, so there is
  a static near-field on the magnetometers in *every* quiet window of a healthy
  vehicle. A placeholder catalog residual (0.5 A·m² on a 15 A·m² rod) put ~9 µT
  on a sensor 0.18 m away — above the plausibility band's low-field edge and
  above any sensible stuck-on threshold. Every FDIR threshold that compares a
  sensor against a model has to be derived from the vehicle's own static
  signature first and the fault second; a threshold set from the fault alone
  declares a healthy vehicle faulty.
- **Redundant sensors must be close enough that the vehicle's own signature is
  common-mode between them** (P54). With the two magnetometers spread across the
  bus, the rods' remanent field differed by ~5 µT between them — which the
  pairwise redundancy vote cannot distinguish from a sensor fault. It reported
  `MagVoteAmbiguous`, dropped the magnetic pair, and detumble starved, because
  B-dot's only input *is* the magnetic pair. Redundancy geometry is a
  requirement, not a layout convenience.
- **A guarded handler must not invoke another guarded port on the same
  component** (P54). The controller's startup mode latch re-dispatched
  `CTRL_MODE_SET` through `cmdIn` from inside the guarded `run` handler; F´
  component mutexes are not recursive, so the deployment aborted on
  `Os/Mutex.cpp` the first time it fired. Share the *guards* as a private method
  both entry points call; never re-enter through the port.
- **Write the requirement from what the physics does, not from what the phase
  plan hoped** (P54). The first REQ-ACTL-001 draft asked for 5 deg/s → 0.5 deg/s
  in 500 s. B-dot damps only the rate perpendicular to the field; the residual
  spin about the field line unwinds as the field direction turns over the
  *orbit*, so the vehicle reaches 2.97 deg/s in 200 s and then decays ~1 % per
  500 s. Measuring first and then writing the requirement on the phase the
  physics actually delivers — with the rest declared owed — is the honest order;
  writing the number first would have produced a permanently red gate or a
  quietly weakened one.

## Gates that swallow their own release conditions (P54, review)

- **An exclusion whose clearing evidence flows through the gate it closes can
  never be revoked.** The §7 stuck-on latch published `interlockHealthy = false`;
  the estimator's magnetometer gate returned false unconditionally on that; the
  raw magnitude the monitor needs was recorded *after* that gate; so the monitor
  saw nothing, the clear streak never advanced, and `MtqStuckCleared` was
  unreachable — a life sentence written by the same push that cites the
  criterion-matched-re-admission rule. The re-admission criterion being *correct*
  is not enough: trace the clearing signal through the **integrated topology**,
  from the sensor to the counter, and check it does not pass through the gate the
  latch closes. A component test cannot find this, because the harness hands the
  monitor the very data the flight gate withholds — if a test supplies an input
  that a real gate would suppress, it is testing the harness.
- **An inert or unconfigured publisher whose product is a gate must publish
  permissive or publish nothing — never a zero-valued product.** The
  unconfigured controller emitted a duty-cycle schedule with
  `quietStart == quietEnd == now`; a magnetometer time tag is never bit-exactly
  the cycle epoch, so every sample was rejected and a controller doing nothing
  disabled the estimator. Silence was already the permissive state (the consumer
  reads "no schedule" as "nothing has ever driven a rod"), so the fix was to not
  publish. Ask what a *default-constructed* product means to the consumer before
  emitting one.
- **A test harness that restates catalog values instead of reading them drifts
  toward passing.** The SITL suites transcribed `config/hardware/**` into C++
  maps. The rod settle time drifted 5x from the committed YAML while
  REQ-ACTL-004 was being verified against the transcription, and the GNSS
  receiver's `cold_start_s` was dropped entirely — so the rows flew a vehicle
  strictly better than the one that ships, and a real precondition of detumble
  (no position, no modelled field, no voted field, no B-dot) stayed invisible.
  The suites already ran the config compiler; the fix was to build the vehicle
  from its `sim_setup.json` output. Transcription is a copy, and every copy is a
  place for the two to disagree in the direction that passes.

## Cross-domain interactions (continued)

- **A disturbance the vehicle *commands* must be fed forward, not discovered**
  (P56). Magnetic desaturation runs concurrently with wheel pointing, so the rods
  apply a body torque the wheels have to take up. Left to the feedback loop, that
  torque costs an attitude error of *disturbance over proportional gain* for as
  long as the unloading lasts — measured at **2.9°** against a 1.0° requirement,
  on a vehicle whose control authority was never in question. The torque is known
  exactly and one cycle in advance (it is the vehicle's own command), so the fix
  is ordering: decide the desaturation *before* the pointing law, feed `−τ_mtq`
  into the demand ahead of saturation and anti-windup, and subtract the same term
  from the disturbance observer's input so a commanded action cannot look like an
  unmodelled fault. General form: when two control laws share a vehicle, the one
  that acts second should be *told* what the first commanded, never left to infer
  it from the error it causes.
- **The actuator's own friction can exceed the control torque, and then the
  pointing requirement is a momentum-dependent statement** (P56). The reference
  wheels carry `dry_friction_nm = 1e-4`; once the four-wheel pyramid is spinning
  the friction reactions sum to 2.3e-4 N·m on the body — five times the injected
  disturbance the row was studying, and more than twice the PID integrator's
  entire authority (`Ki × clamp`). The resulting 3.0° of steady-state error was
  invisible for two pushes because every earlier row flew wheels near zero speed;
  it appeared the moment a row deliberately loaded them. Two lessons: a
  requirement verified only at one operating point (here, empty wheels) is
  silently conditioned on it, and **a closed-loop row that fails should first be
  explained arithmetically** — the friction sum predicted 3.0° against a measured
  2.9° before any code was changed, which is what told the push it was looking at
  a vehicle fact and not at its own new feature.

## FDIR / fault-tolerance state machines (continued)

- **A term recorded as "commanded" must be cleared on every path that does not
  command it** (P56, review). The rod-torque ordering fix records the magnetic
  torque at the moment the desaturation is *decided*, then feeds it forward and
  subtracts it from the observer a cycle later. Every path that abandons the
  command downstream — a refused pointing law, a failed clamp — must clear the
  record too, or the observer subtracts a torque the vehicle never applied and
  the anomaly monitor fires on the vehicle's own inaction. The general form is
  the mirror of the ordering-fix entry above: when one law tells another what it
  commanded, the telling has to be retracted when the commanding is. Grep every
  path that zeroes an actuator command for the bookkeeping that accompanied it.
- **Report the edge from the command, never from the decision** (P56, review).
  `DesatEngaged` was emitted on the predicate that *chose* to desaturate,
  upstream of the refusal path that zeroes the dipole, so an operator
  correlating a payload anomaly against magnetic activity could be handed a
  window in which no rod was driven. A decision and a command are two different
  facts about a cycle; an event whose whole purpose is to timestamp a physical
  action must be derived from what was published, which means it has to be
  emitted after every path that can withdraw it.
- **A refusal that disables a monitor is an event, not a telemetry gap** (P56,
  review). One dead wheel tachometer correctly refused the momentum sum — and
  thereby took out both the desaturation and the §9 momentum-anomaly monitor,
  indefinitely, with nothing in the event stream to say so. A NaN on a strip
  chart is the right *telemetry* answer and the wrong *FDIR* answer: the refusal
  is well reasoned locally and invisible globally. Whenever a refusal gates a
  downstream monitor, the refusal itself needs a bounded-cadence event naming
  the reason and the unit, plus a latched flag, so the silence of the monitor it
  disabled is attributable.

## Requirements & verification (continued)

- **A tolerance-band attribution must be backed by an ablation, not a
  narrative** (P57). The GEO golden case carried a 150 m band whose rationale
  attributed the 120 m residual to "SRP model differences" between two
  cannonball models — plausible prose that no one had tested. A FreeFlyer
  force ablation (turn SRP off entirely, measure the divergence from the same
  golden samples) reproduced the 120 m to two decimal places: the "model
  difference" was the *entire SRP signature*, because the fixture omitted the
  ballistic properties and `srp_area_m2` silently defaulted to zero — a
  disabled force flying under a band wide enough to absorb it. With the force
  actually on, the two propagators agree to 0.017 m and the band tightened
  300x. Two generalisations: a struct field whose zero-default silently
  disables a physical effect must be required, not defaulted, wherever the
  effect is claimed to be on; and when a rationale names the dominant term of
  a residual, the review question is "what ablation established that?" — a
  third independent implementation makes that ablation a two-minute
  experiment.

## The same number written twice (P60, review)

- **A quantity that exists on both sides of the flight/sim boundary drifts
  silently, and no test you would think to write catches it.** Several physical
  values live twice by design: as sim/hardware truth (`config/hardware/**`, a
  `spacecraft.*` field) and as the `flight.*` parameter the FSW actually runs on.
  Edit one and leave the other and *nothing happens*: the build is clean, the
  suite is green, the SITL rows pass, and the flight software believes something
  the vehicle is not. Push 60 re-sized the reference wheel to
  `max_torque_nm: 0.002` and left `WheelMaxTorqueNm` at `0.025` — an FSW that
  would command **12.5x more torque than the wheel can deliver**, with every
  torque margin computed from that parameter optimistic by the same factor, and
  saturation reported nowhere because as far as the FSW knew it never saturated.
  It was caught by luck: a new analysis tool happened to read both. The earlier
  `MtqAxesBody` vs `dipole_axis` mismatch is the same class one level down.
  The rule: **every such pairing is enforced in `tools/configc`**, where the two
  halves are in the same room and the check runs on every SITL row and every
  flight parameter build — never in a unit test (which proves the drift is
  *detectable*, not that it is *impossible to commit*), and never as a comment
  asking the next engineer for care. When you add a `flight.*` parameter, the
  review question is "what else in the config says this number?"
- **The escape hatch has to be designed, or the check gets deleted instead of
  obeyed.** Divergence is sometimes the design — the onboard model is
  deliberately lower fidelity than truth (`sim/CLAUDE.md`), and derating a unit
  below its catalog rating is a real decision. A check with no legitimate escape
  is one the next engineer disables wholesale, taking the eleven cases that were
  working with it. So divergence is *declarable* in the config
  (`spacecraft.fsw_parameter_divergence`) — and deliberately not as a bare
  opt-out: the entry restates the truth value it was written against, so the
  waiver **expires when the hardware does** rather than outliving its subject,
  and a waiver naming a parameter no check covers is itself an error. A skipped
  check that can never come back is how a mechanism rots into decoration.
- **"Mismatch" is not an error message.** A cross-check that fires costs the
  reader more than it saves unless it names *both* values, *both* source
  locations (which YAML, which key, which installed units), what physically goes
  wrong if they stay apart, and the two ways out — fix the stale side, or declare
  the divergence, with the declaration spelled out ready to paste.

## A test whose reference stopped being above its subject (P63, measurement)

Push 63 raised the onboard force model from closed-form J2 to an 8×8 EGM2008
field. Two committed tests had characterised the old model's truncation against
references that were *also* degree 8 — the truth sim's `gravity_degree: 8` and
the GMAT fixture's `gravity_degree: 8`. Both kept passing. Both had become
meaningless: the unit test's 3.59 m fell to **0.045 m** and the golden test's
14.3 m to **1.2 cm**, not because the model got 100× better against full
fidelity, but because it was now being compared against *itself* through a
second implementation. The bounds were 5.0 m and 50 m, so nothing failed and
nothing prompted a look.

- **Improving a model can silently invalidate the test that characterised it.**
  A test with an upper bound reports a *smaller* number when it goes vacuous,
  which reads exactly like success. There is no failure mode here to catch it:
  the assertion, the test name, and the recorded property all still make sense
  as English. The only thing that changed is that the reference is no longer
  above the subject.
- **The review question is "what is this measured *against*, and is it still
  better than what it measures?"** Ask it every time a model's fidelity moves.
  For any characterisation test — truncation, model difference, residual budget
  — the reference's fidelity is a load-bearing input that lives somewhere else
  in the file (here, a `SimConfig` field 60 lines away and a JSON fixture's
  `environment` block) and is not mentioned in the assertion that depends on it.
- **State the reference's fidelity next to the number it produces.** Both tests
  now say what truth they run against and why it is above the model, so the next
  degree bump reads the constraint at the point it would break it rather than
  discovering it by having a number get suspiciously good.
- **Two honest repairs, and they are different.** Where a higher-fidelity
  reference existed, it was *raised* (truth sim 8×8 → 32×32) and the number
  re-derived — `q_a` moved with it, since it is sized from that measurement.
  Where none existed (GMAT's fixture is degree 8 and regenerating it is a
  separate job), the test was **repurposed to what it can now actually assert** —
  a matched-degree cross-validation of two independent implementations, which
  turned out to be *stronger* evidence than the truncation it replaced. Deleting
  it or loosening the bound to keep it green would both have been worse than
  either.
- **Pin the improvement, not just the improved number.** Both tests now fly the
  *old* model over the identical arc and assert the new one beats it. A
  characterisation number drifts with the epoch and the vehicle; "the field beats
  J2" is the claim that actually justifies the code, so that is what is asserted.
  It also means a silent revert to the old model fails, which a one-sided bound
  on the new model's error never would.

## A harness whose cycle order made a flight branch unreachable (P63, measurement)

The orbit-OD Monte Carlo driver polled the GNSS delay-line model once per 10 s
fix period and, on a valid fix, called `ingest()` alone. Two things followed,
and only the first announced itself.

- **A delay-line model polled slower than its own delay realises the poll
  period, not the delay.** `Gnss::sample()` returns the newest solution at least
  `fix_latency_s` old, and it only knows about solutions it was handed. Polled
  every 10 s with the datasheet's 50 ms, the newest one old enough is the
  *previous poll's* — so the campaign flew a 10 s latency. Measured: a constant
  **76.7 km** of along-track offset (10 s × 7.669 km/s), with the covariance
  sitting at 0.68 m and 354 of 355 fixes accepted. The tell was the shape, not
  the size: a diverging filter does not look like that. The filter was tracking
  its own trajectory perfectly and simply answering for an epoch 10 s behind the
  one the record scored it against. **When an error is constant and the
  covariance is healthy, suspect the epoch before the math.**
- **Ingest-only means the filter's epoch is always the last fix's, so an
  arriving fix is always *forward* and the latent-fix branch is dead code.**
  This is the one that stayed silent. Every guard the latency correction owns —
  `max_fix_latency_s`, the no-velocity refusal, the O(τ³) advance — had zero
  campaign coverage, and nothing failed to say so, because unreachable code
  reports no error. The gated unit tests exercised it; the campaign that was
  supposed to be the open-ended check did not.
- **A harness must march the real cycle order, not a convenient one.** The FSW
  propagates to *now* every cycle and then folds in whatever arrived; that
  ordering is what makes a latent fix latent. Reordering the driver to match
  changed nothing about the zero-latency arcs' numbers (2.01 m worst, identical)
  and turned a dead branch into a measured one — which is the signature of a
  harness bug rather than a model bug: the fix is invisible where the harness was
  already right.
- **"Where am I now" and "where was I at the last fix" are different questions.**
  Pointing, pass planning and maneuver targeting all ask the first. A harness
  that only ever evaluates the estimate at fix epochs never measures the quantity
  the vehicle actually uses.
- **When a modelled effect cannot be resolved at the campaign's cadence, give it
  its own scenario rather than arming it everywhere.** `latency_fast` runs 50 Hz
  over ten minutes with the real 50 ms; the long arcs pass zero. Leaving a
  datasheet value armed at a cadence that cannot see it does not model the effect
  conservatively — it models a different, much larger effect, and reports it as
  the filter's error.

## The band that was the only gate, sized as if it were the second (P63, measurement)

Building the OD campaign's report surfaced this from a scenario that was
*passing*. `bad_data` injects fixes at geostationary radius and its stated
intent is that they be refused on the §9.1 plausibility band "before the filter
sees it". They were being refused — as `measurement_rejected`, by the NIS gate,
one layer further in. The band, configured `6.4e6` to `5.0e7` m on a 400 km
vehicle, admits everything from just above the surface to beyond GEO and caught
nothing.

- **A seed has no prior, so it has no innovation, so it has no gate.** On the
  update path the NIS test is a genuine second line of defence and it worked.
  On the *seed* path — a cold filter, or one whose solution the coast horizon
  has just dropped — the plausibility band is the only thing between a wire
  value and the state the vehicle then flies on. Measured: a GEO-radius fix
  seeded the LEO filter outright, `seeded == true`, refusal `kNone`. The
  `outage_long` scenario drops the solution three times a week, so the seed path
  is not a cold-start curiosity; it is a path the vehicle takes in flight.
- **A trust boundary sized to "any Earth orbit" is not a trust boundary.** The
  band's job is to exclude what this vehicle cannot be doing, and a smallsat at
  400 km cannot be at geostationary radius under any dispersion. Re-sized to
  6.5e6–8.0e6 m (roughly 120–1600 km altitude): the whole LEO band with room for
  decay and dispersion, with MEO, GTO and GEO all outside it.
- **A defence that is passing because a *different* defence caught the case is
  not passing.** Nothing failed here — the fix was rejected, the campaign was
  green, and the scenario's own comment said what was supposed to happen. The
  only way to see it was to read *which* refusal came back, which is why the
  driver now emits the refusal by name rather than as an integer. Ask of any
  layered check: which layer actually fired, and is the one being tested the one
  that did?
- **Emit the reason, not the outcome.** Had the record carried only
  `fix_accepted: 0` this would have been invisible for as long as anyone cared
  to look. The name cost a few bytes on ~1 % of rows and is what made the
  finding legible at a glance.

## A label that records the cause ending, not the effect ending (P63, campaign)

The OD campaign's per-cycle `regime` tag flips back to `nominal` the instant a
fault window closes. The estimate does not: after a spoof the filter is still
kilometres out with its gate refusing honest fixes for tens of minutes. The
campaign-wide NEES gate pooled every scenario's nominal-*regime* cycles, so the
recovery tails — labelled nominal, drawn from no distribution the covariance
claims — put it at 2.5e6 against a ceiling of 7.3 on a healthy filter.

- **Armed and contaminated are different predicates.** A tag that records
  whether the fault generator is running says nothing about whether the state
  it corrupted has relaxed. Any statistic gated on "nominal" must decide which
  of the two it means, and a transient-bearing system means the second.
- **Three checks that can disagree are worth two that cannot.** NEES screamed
  while the truth-derived ensemble ratio read 0.98 and NIS passed. The
  contradiction did not just flag the defect, it *localised* it: the ensemble
  check was already scoped to the nominal scenario, so the difference between
  the populations was the entire suspect list.
- **Quadratic statistics have no breakdown resistance.** A mean of squares is
  moved arbitrarily far by arbitrarily few samples; 3k contaminated cycles
  outvoted 2.2M healthy ones. Pool into a quadratic form only what the claim
  under test covers.

## Margin stored in a tuning constant instead of on its fence (P63, campaign)

`q_a` is derived from the measured force-model truncation, which was carried at
1.8 m against a 1.28 m measurement so the CI assertion would not flap with the
epoch. The campaign then measured the filter conservative — NEES 4.22 under a
4.83 floor, all velocity — and a sweep across the margin (4.68/5.28/6.81 at
1.5/1.28/1.0 m) was consistent only at the measurement.

- **Headroom belongs on the assertion, not in the value.** One constant served
  two masters: a CI fence that wants slack and a flight tuning that wants the
  truth. Split them (`kTruncationAtHorizonM` = measurement,
  `kTruncationFenceM` = fence) so margin can never again ride silently into
  the covariance.
- **"Conservative" is a measured defect, not a virtue.** The pessimistic half
  of the chi-square interval exists because an over-budgeted filter discards
  information it has; the gate flagging it is the gate working.

## An MC driver built in the tree the framework sanitizes (P63, campaign)

The campaign ran 1.64x slow for a day because the driver was built into
`build-fprime-automatic-native-ut`, where F´'s own `cmake/sanitizers.cmake`
adds ASan+UBSan regardless of the project's `POLARIS_SANITIZE` option (OFF in
both caches — checking it proves nothing). `analysis/detumble/README.md`
already warned about exactly this; the new package's README documented the
wrong tree anyway.

- **Verify instrumentation on the binary, not in the cache:**
  `nm -C <bin> | grep -c __asan` answers it in one line.
- **A convention that lives only in a sibling's README is not a convention.**
  The warning existed and was re-learned at full price; it is now beside the
  build command in every campaign README.

## Code and docs written before a lesson do not re-sweep themselves (P64, audit)

Five independent review lanes over main converged on one signature. The JPL
audit found the P46 concurrency class alive in `OnboardTables` — the one
component written before that lesson landed, while its two siblings enacted
it. The flight-standard audit found the P60 same-number class displaced onto
flight FDIR: a χ² gate hard-coded in one file and parameterised in another,
five copies, two spellings. The docs audit found every stale figure traceable
to a push that re-measured a number and swept *most* of its citations.

- **A lesson is enacted at a point in time; the tree is not.** Appending a
  class to this catalog fixes the instance that taught it and everything
  written afterwards. Everything written before it keeps the defect until
  someone re-sweeps, and nothing in the process forces that.
- **The re-sweep is cheap once the class is named.** Each of the audits
  found its instances with a grep and an afternoon; the expensive part was
  ever learning the class, which was already paid.
- **Corollary for figures in prose:** a number quoted in a comment with no
  link to the constant it describes is a stale-docs defect waiting for its
  push. The OD area's pattern — declare the design input once, compute the
  derived constant from it, cite rather than restate — is what came through
  the audit clean.

## A fixture can lie about its own physics for fifty pushes (P65, SITL)

The shared SITL orbit fixture set `gravity_degree = -1` under the comment
"two-body: the plant is not under test". In the sim's enum -1 is *free
drift*: every SITL row from Push 39 to Push 64 — fault matrix, control,
calibration, tuning — flew a vehicle in a straight line at 7.6 km/s, and
every one of them passed, because nothing they measured depended on where
the vehicle actually went. Push 65's orbit estimator was the first consumer
whose whole model is gravity; it rejected every fix within a second and the
fixture was found in the innovation sequence, not in review.

- **A comment stating a config value's meaning is a claim about an enum,
  and enums are checkable.** The one-line fix (`gravity_degree = 8`, matched
  to the flight model) is trivial; the fifty pushes of a plant that did not
  orbit are not recoverable. Prefer named constants for enum-like sentinels
  (`kFreeDriftPlant`) so a reader sees the semantics rather than the number.
- **"The plant is not under test" is not the same as "the plant may be
  wrong".** A fixture that is *simplified* is fine; one that is *unphysical*
  quietly narrows what every row on it can ever notice. Three tuned rows
  broke on the corrected plant — two control A/B thresholds and the
  magnetometer-calibration residual (2.43 → 7.45 mrad) — which is what
  narrowing looks like when it lifts.
- **Per-step fault hooks and scheduled faults do not compose.** The closed
  loop reconciles every receiver to the scenario's fault schedule on each
  sample (idempotent by design), so a flag set directly on the model is
  undone before the next fix. A GNSS row must inject through the schedule;
  the sensor rows' per-step hooks work only because those sensors have no
  schedule. Documented on the row that learned it.

## The orbit filter's fault rows on the topology (P66, SITL)

- **A passive component cannot dispatch its own command from a guarded
  handler.** Guarded ports and the command port share the component's
  non-recursive mutex, so `get_cmdIn_InputPort(0)->invoke(...)` from inside
  `run_handler` deadlocks — the SITL row's only symptom was "sim not
  healthy" after a timeout. A bench hook that must fire mid-cycle runs the
  command handler's *body* (factored into a private method the handler also
  calls) and says so in the comment; the startup hooks (`-M`, `-A`, `-c`) may
  still dispatch through the port because they run before the rate group.
- **A row's premise must be re-checked in the log it produces, not in the
  fixture comment.** The Push 65 outage row's comment placed the first fix
  "near 37 s" (the catalogue cold start); the harness receiver has no cold
  start and seeds on the first cycle. Harmless there, but a horizon row
  whose timing derives from a comment is one changed default away from
  asserting nothing. Read one kept log (`POLARIS_KEEP_SITL_LOGS=1`) per row
  before trusting the assertion.
- **A stale run log is a false finding.** A leftover `build-artifacts/`
  directory from an earlier fixture state read as "the filter rejected every
  fix for 300 s" on main; the fresh run was healthy. Match the log's mtime
  to the run before diagnosing from it.

## A straight-line plant hid a fit defect for twenty pushes (P67, lib)

- **A parameter the model does not have is a defect waiting for the data
  that exposes it.** The ellipsoid fit carried the quadric's constant as a
  free tenth unknown; the model fixes it (`c = βᵀAβ`). Every test swept the
  field magnitude widely enough to pin it by accident; the first realistic
  window (6 % swing) let noise decide it and shipped a 0.7 % scale error.
  When a linear least-squares model has an algebraic constraint among its
  parameters, either eliminate the parameter or enforce the constraint after
  the solve — and put a unit test on the *least-informative* input the
  flight case produces, not the most.
- **"Same figure with the old code, so it is the plant" is not a root
  cause.** Push 65 correctly showed the residual tripling was not the orbit
  filter, then attributed it to plant physics and moved the bound to the
  class limit. Two hypotheses were left untested (pairing skew, fit
  conditioning) and one hour of experiment — an ideal sensor, then the same
  solve offline — settled both. A moved bound with an owed investigation is
  acceptable; an owed investigation that outlives the next push is not.
- **A convergence window tuned to pass has no margin by construction.**
  Both control rows converged with 10–40 s to spare on the plant they were
  tuned on; the plant change did not alter the physics (all disturbance
  torques off was bit-identical), it moved a chaotic settling by one cycle.
  Size a settling window from the measured settling time plus a stated
  margin, and record the settling time on the artifact so a drift shows up
  as a number before it shows up as a failure.

## A torque clip is not a slew-rate limit (P68, control)

- **Check the large-signal regime of a law tuned for the small one.** The
  PID's gains were sized for a small-angle bandwidth and every accuracy row
  measured a hold; the first row that entered POINT from 100° found `Kp·δθ`
  at 85× the torque limit and a bang-bang slew that cost the estimator a
  dozen demotions. Before shipping a linear law, compute its demand at the
  largest error the mode can be entered with, and if it exceeds the
  actuator by more than the clip can absorb, add the rate limit the
  literature already has (Wie & Lu 1995) rather than a bigger window.
- **Assert the mechanism, not only the outcome.** The row's outcome (a 2°
  tail) passed for two pushes while the storm ran underneath it. The
  demotion and saturation counts were on the log the whole time; they are
  now bounded on the row, so the storm fails a test the day it returns.

## The wheels audited: what three independent lanes found (P69, control)

- **A body-momentum threshold is not a wheel-momentum threshold.** A
  redundant array can hold every wheel at capacity in its null pattern with
  zero body momentum; envelope, desaturation and observer are all blind to
  it. Any monitor on a *projected* quantity needs a companion on the
  *per-actuator* quantity — and the L-∞ allocation injects null space every
  cycle, so this is drift by design, not by accident.
- **An optimum computed on the wrong norm is still an optimum.** The L-∞
  search minimised the unweighted max wheel torque while the config carried
  per-wheel limits; equal limits hid it. When a config admits heterogeneity
  the algorithm must be tested with it, even if the flight vehicle is
  homogeneous.
- **Edge-gated events hide sustained conditions from tests that grep.**
  ~80 saturation events per row were in the log for two pushes; nothing
  counted them. Every SITL row that exercises a mechanism now records and
  bounds the mechanism's counters (saturation, envelope, demotions, refused,
  peak-over-limit), not only its outcome.
- **A bias must be sized against the threshold that will erode it.** 10 % of
  wheel capacity sounds ample; the desaturation entry loads the worst wheel
  by 0.75× its threshold, which is 90 % of that bias. Write the sizing rule
  in configc, not in a comment.
- **Report the cost, not only the benefit.** The bias points *worse* on an
  unloaded hold because spinning wheels pay the under-trimmed friction. The
  row says so, and the design doc says why the stiction-held number is not
  one to fly.

## A paper is a checklist, not a design (P70, OD)

- **Read the numbers, not the abstract.** The paper's headline is a robust
  GNSS/INS architecture; its own results say the INS buys nothing while GNSS
  is present and the outage performance "is not remarkably higher than an
  isolated propagator". The one transferable result was the burn-in-outage
  comparison — and that is what was built. Adopting the architecture would
  have added a 15-state error filter for no measured gain.
- **A validity policy that costs a consumer must be justified by that
  consumer's tolerance.** The 300 s drop was sized for a metre-class use and
  cost a kilometre-class one (the magnetic reference) for nothing. Publish
  quality and sigma; let each consumer gate on its own tolerance.
- **When a lane dies mid-task, read the tree before re-planning.** The OD lane
  was killed by a spend limit after most of its work had landed; three unit
  tests were the whole gap. Rebuild, run everything, then fill holes.

## A best-practices document is a checklist against the seams, not the math (P71, OD/ADET)

- **The filters were textbook; the operability was not.** NASA/TP-2018-219822's
  math chapters were already satisfied; its Ch. 9 (editing flags, covariance
  re-init, backup ephemeris, uplink without loss) was entirely absent, and the
  most costly gap was a *component* behaviour — every parameter upload rebuilt
  the filter and threw a converged solution away. Audit component seams
  against operability rules, not only libraries against equations.
- **"Bad upload" must never make a running filter inert.** `fail()` paths that
  set `configured_ = false` were correct at bring-up and wrong forever after;
  the rule is: refuse the *new* set, keep the *old* one in force.
- **Force overrides the gate, never the fault.** A negative NIS is the
  covariance, not the residual; an operator flag that overrode it would fly an
  update against an indefinite `P` on request.
- **Record what a practice cannot fix.** The leap-second exposure on the seed
  path is real and bounded; a test that measures it (490 m, cold seed only) is
  worth more than a sentence claiming immunity.
- **Declining a recommendation needs the number.** Underweighting was declined
  because `H` is linear for GNSS/ST and the sun/mag second-order term is
  ~1e-3 rad against 1e-2 rad σ — written down, so the next reader does not
  re-derive it or add a parameter nobody can tune.

## Latency is a sim-and-FSW item, and the tag is the whole of it (P72, ADET)

- **A time tag that is only a freshness gate is a latency you are not
  correcting.** The tracker's tag had been carried since P48 and read once,
  as `fresh()`. Every sensor that stamps its own measurement epoch should be
  asked: does anything *use* the difference between that stamp and now?
- **Model the latency in the truth before compensating it in the flight
  code**, or the compensation is verified against nothing. The delay-line
  pattern already existed for GNSS; the tracker got the same one, and the
  SITL rows then flew the delayed tracker end to end for free.
- **A flight/sim pair needs its configc check the day it is born.**
  `MaxMeasAgeSec` vs `latency_s` was checked in the same push as the sim
  field, so a catalog change cannot silently stale every tracker sample.
- **Batching changes what an accessor means mid-cycle.** `attitude()` inside
  an open batch is the reference, not the running estimate; the residual
  monitors and the publish had to sit after `endBatch()`, and a re-seed
  inside a batch closes it. Say so in the accessor's doc.

## A best practice is a hypothesis until the campaign says otherwise (P73, OD)

- **Build the machinery, then measure before flying it.** The DMC states did
  exactly what the TP describes in the library (an unmodelled acceleration
  estimated, the coast closed) and *worsened* the coast on the vehicle's own
  truth. The flown yaml records both numbers and stays as it was.
- **Give the campaign the knobs on the command line.** `--dmc-tau-s`,
  `--dmc-psd`, `--q-rtn`, `--qa-scale` on the MC harness meant every candidate
  ran on the same shards in minutes; without them the study would have been
  a yaml edit and a rebuild per candidate — or would not have happened.
- **Zsh does not word-split an unquoted variable.** Three "different" MC runs
  produced byte-identical shards because `$extra` reached the binary as one
  argument; `${=extra}` fixes it. Check `md5sum` before believing a null result.
- **Closed forms that cancel are not "exact".** TP Eqs. 2.56–2.61 are correct
  and unusable at `h/τ ~ 1e-3`; a fixed Gauss–Legendre rule on the kernel is
  bounded, exact for the quartic, and was pinned against 20 000-node Simpson —
  Simpson at 16 nodes was 1e-5 off on the quartic, also measured.

## A criterion measures what its samples contain, not what its name says (P74, OD/analysis)

- **Three runs is a hypothesis; the gate is the answer.** The RTN restructuring
  that improved the coast by 13 % on three runs was inside the noise on ten and
  on twenty, with the NEES risen from 5.3 to 7.2. Nothing was flown on the
  three-run number, and the yaml never changed — but a whole push's headline had
  been written from it. Re-measure a candidate at gate size before believing it.
- **A failing baseline is the campaign talking about itself.** The *flown*
  tuning failed a criterion at 30 runs. The tempting reading is "the tuning
  regressed"; the true one was that `burn_outage_blind` deliberately denies the
  filter a 3.6 m/s burn, and the gate then refuses honest fixes because the
  *state* is wrong. Refusals that a fault earned are not gate false alarms.
- **Scope a metric by what the estimator says about itself, not by the fault
  tag.** The regime tag falls the instant a fault window closes, while the
  estimate is still contaminated — `summarise` already documented that trap for
  NEES/NIS, and the rejection rate had inherited it anyway. The fix was to scope
  the rate to the samples where the filter still calls its own solution *fine*;
  the filter's verdict is the honest boundary, and it costs one recorded field.
- **A rate cannot express a lockout.** 2.2 % over a day was one unbroken 898 s
  stretch per burn. A filter that refuses every honest fix for a quarter of an
  hour and one that scatters the same refusals across a day give the same rate,
  and only one of them cannot get back. Measure the stretch.
- **Carry the threshold out of the thing being judged.** The re-acquisition
  bound is the filter's own `max_degraded_coast_s`, emitted in the driver's meta
  record rather than copied into the analysis — the flight/sim parameter-pair
  rule applied to a criterion. A threshold the analysis owns is a threshold that
  drifts from the policy it claims to check.

## The obvious explanation for "it's slow" is worth one measurement (P75, freeflyer)

- **Measure the boundary you suspect before optimising it.** The WSL→Windows
  stream file looked like the obvious cost; it reads 4501 states in 36 ms
  (8 µs each). Preloading, caching or copying it would have bought nothing.
  The cost was four synchronous engine round-trips per frame — async queueing
  with one drain at the render took 188 ms/frame to 59 ms, 3.2x, and changed
  nothing about what is drawn.
- **Performance advice ages with the platform.** "Prefer one window, it halves
  the frame cost" was true on the CPU rasteriser and false the moment the GPU
  rendered (56.4 vs 58.6 ms). So was the `--fps 2` default. When a platform
  assumption changes, grep the docs and comments for the advice it justified.
- **A gitignored vendor directory can shadow the package that drives it.**
  Repo-root `freeflyer/` (installer, license) shadowed `tools/freeflyer` on the
  Windows child's `sys.path`, which imported a folder of RPMs. Regular packages
  beat namespace packages *within* one path entry, but `sys.path[0]` order wins
  first — run the child from the directory that owns the package.
- **Environment does not cross the WSL/Windows boundary.** Only `WSLENV`-listed
  variables reach a Windows process, so `PYTHONPATH` silently did not arrive.
  Pass what must cross explicitly, and prefer cwd/argv to env where you can.
- **An optional vendor component is a compatibility question, not a blocker.**
  The Windows FreeFlyer installer omits the Runtime API SDK but ships
  `ffrtapi.dll`; borrowing the client from the other install works and is safe
  *only* on an exact build match, because the client is generated against one
  engine ABI. Encode the check, name both builds when refusing.

## An optional state is a state that has to be *turned on*, not just skipped (P76, orbit_od)

- **A zero-variance block is a state the Kalman gain can never reach.** The drag
  scale factor's covariance block is identically zero while it is off, so
  enabling it by uplink on a running filter left a permanently frozen estimate —
  correct arithmetic, no error anywhere, and a state that could only ever be
  opened by months of random walk. Any conditionally-carried state needs seeding
  on the **off→on transition**, not only at cold start. `retune` seeds that one
  block and leaves the navigation solution untouched, which is the whole point
  of re-tuning in place.
- **The off→on path is the one no other test covers.** Every existing case ran
  with the state off, so all of them would have passed against a seam that was
  never connected. When a capability ships disabled, the test that turns it on is
  the only test of its wiring — and it has to use the *uplink* mechanism
  (`paramSend_`), because `paramSet_` + `loadParameters()` never re-applies once
  the component is configured.
- **Require both halves before carrying a state.** A drag scale factor with drag
  disabled has no path to any measurement: its variance grows every step and is
  never reduced. Refusing to carry it is cheaper than explaining a covariance
  that only ever gets worse.

## A tolerance on a differenced check belongs to the difference (P76, tests)

- **Do not assert an analytic partial to a tighter bound than the numeric
  reference can resolve.** Checking `∂a/∂s` against a central difference at
  `ds = 1e-6` on a ~1e-6 m/s² term cancels twelve significant figures before
  dividing, so ~1e-8 relative is its floor; the partial was exact and the test
  failed at 1e-9. Size the band from the reference's own error, and put the
  machine-exact assertion on a **structural** property instead — here, that the
  term is linear in the scale, checked with no differencing at all.

## Measure the thing before scoping the work around it (P76, orbit_od)

- **A capability can be correct and still resolve nothing, and that is a
  result.** The drag scale factor works; its entire signal is 39x under the
  process noise already budgeted for the geopotential truncation. The scan that
  found it (`q_a` x arc length, 12 points) took minutes and turned a guess about
  arc length into a measured statement about `q_a` — and named what would
  actually unblock it, a higher-degree onboard field. Prefer the cheap parameter
  scan to the confident assertion, and assert the negative result as an **upper
  bound** so the day it stops holding, the test says why.
- **Scope the named item down where a piece of it is speculative, and say so
  with a number.** "Drag/SRP scale factors" shipped as drag only: the onboard
  force model carries no SRP term, so the SRP half meant building a term an
  order below the drag beside it. Deferring it with its magnitude is a decision;
  deferring it silently is a gap.

## Sweep the tuning knob, not the whole gate (P77, analysis/od)

- **A tuning trial and an acceptance run are different runs.** NEES is measured
  on the `nominal` scenario alone, so a consistency tuning needs one scenario and
  a short arc — ~1 minute a point — while the full campaign is twelve scenarios
  of 24 h. Iterating on the gate is how tuning became an overnight loop; the fast
  sweep plus one gate run at the end is the same confidence for a fraction of the
  wall clock.
- **But keep the run count honest even when sweeping.** The chi-square interval
  widens as 1/sqrt(N); a three-run sweep ranks candidates and cannot accept one.
  Push 74 already paid for this lesson once, when a three-run improvement
  evaporated at gate size.
- **Put the knob in the driver, not in a scratch patch.** The three
  `--gnss-corr-*` flags cost a few lines and turned an ad-hoc printf scan into a
  reproducible sweep anyone can rerun from the README.

## An error's colour defeats a filter its magnitude does not (P77, gnss/orbit_od)

- **Model the spectrum, not just the RMS.** A datasheet quotes a magnitude and
  says nothing about correlation. Splitting that RMS into white and Gauss-Markov
  parts — as a *fraction of the variance*, so the total is preserved — changes
  only the colour, and the filter failed **both** consistency criteria on colour
  alone. Adding correlated error *on top* of the datasheet would have confounded
  "bigger" with "correlated" and proved nothing.
- **Give the model the generous reading of what the sensor knows.** The receiver
  reports the full total, not the white part — a formal covariance can size an
  error but not say how much survives to the next sample. Defeating the filter
  under the generous assumption is a stronger result than under a pessimistic one.

## Check every criterion, not the one you are steering (P77, analysis/od)

- **I grepped only the NEES lines while tuning, and shipped a regression.** NIS
  was failing at every point of the sweep — including zero inflation — and the
  sweep output would have said so. A tuning loop that reads one number cannot
  see the cost it is paying somewhere else, and here the two criteria were
  moving in *opposite* directions, which is the case where reading one is worst.
- **A statistic tuned against cannot falsify the thing it tuned.** The inflation
  factor was fitted until NEES entered its band, so any value that passed was by
  definition "right" — unfalsifiable by construction. The replacement is a
  *receiver property* cross-checked against the hardware entry by equality, and
  NIS is then a genuine two-sided test of it. Prefer a parameter the world fixes
  to one the gate fixes.
- **Predict before you measure.** The consider block's NEES/NIS/RMS/rejection
  numbers and its falsification conditions were written down *before* the code
  existed. Measuring 5.548/2.938 against a predicted 6.3±0.8 / 2.9±0.30 is
  evidence; fitting to 5.548 afterwards would have been decoration.
- **The convenient hypothesis deserves the harshest test.** "The criterion is
  mis-scoped" would have retired the failure at no cost, and Push 74 had set the
  precedent for exactly that. It was refuted three ways: the interval is ~470x
  *wider* than the true run-to-run spread (so it is loose, not tight); a white
  receiver passes at 2.986; and NIS returns to ~2.95 as tau collapses to the fix
  cadence. Test the escape hatch before taking it.

## A trade between two consistency statistics is a modelling error, not a tuning (P77, orbit_od)

- **`S = HPHᵀ + R` is a consequence of whiteness, not a definition.** It needs
  `E[e vᵀ] = 0`, which holds because the prior error is built from past
  measurements and white noise is independent of them. Colour the noise and
  `C = E[e bᵀ] != 0` appears, the innovation covariance loses `HC + CᵀHᵀ`, and
  the innovation is *smaller* than `S` by twice the bias already absorbed.
- **So NEES and NIS move oppositely and no `R` fixes both.** Padding `R` treats
  a correlation as a magnitude. Carrying the error as states puts `C` back where
  it belongs — as a covariance cross-block — and both statistics become valid at
  once. If two consistency tests disagree, look for a term the derivation
  dropped before looking for a knob.
- **A consider (Schmidt) block works even when the state is unobservable.** The
  covariance still propagates and still shapes the gain, so `S` is right whether
  or not the estimate is informative: unobservability costs the estimate, not
  the consistency. That is what let the fix ship on a vehicle where the bias is
  4.4x under the process-noise floor.
- **Pin consider states on *every* update, not the one whose H sees them.** The
  velocity measurement does not see the position bias, but `P` has a
  bias/velocity cross-block, so an unpinned velocity update walked the estimate
  anyway — the exact state a slow spoof would exploit.
- **A state you refuse to estimate can be a security property.** A FOGM position
  bias is precisely the shape to absorb a slow spoof ramp. Pinning the estimate
  is not only an observability concession; record it as a threat-model decision
  so nobody "improves" it later by enabling the estimator.

## Refuse unknown flags in a measurement driver (P77, tests/mc)

- **A silently ignored argument makes a campaign measure something it does not
  name.** A sweep kept passing a flag that had been renamed; nothing failed, and
  the runs quietly flew a different configuration than the command line claimed.
  A tool whose output is *evidence* must refuse what it does not understand.
- **And check which flag is load-bearing before accusing the data.** I called a
  teammate's control invalid on seeing the dead flag, when a second flag in the
  same command made the run correct regardless. The retraction cost more than
  the check would have.

## Agreement is not validity: bound a shared quantity, not just its equality (P78, configc/sim)

- **A cross-check that compares two copies cannot see that both are nonsense.**
  Push 77 made `GnssCorrFraction` and the receiver's
  `correlated_position_fraction` a plain equality, which is right and was not
  enough: `1.5` against `1.5` agrees perfectly, compiled clean, and produced a
  vehicle whose sim flew a receiver with zero white noise while the flight
  filter refused to configure. A flight/sim pair needs **both** halves — the two
  agree, *and* the agreed value is inside the range each side can fly.
- **Never let the two sides disagree on how to reject a bad value.** The sim
  clamped (`std::min(1.0, f)`) where the flight filter refused. The clamp is
  what turned a one-character config typo into the silent cascade — "no orbit
  solution, no magnetic reference, no attitude" — three layers from its cause.
  Defensive coercion in one implementation of a pair is not defensive; it hides
  the disagreement the pair exists to expose.
- **Validate a config value where the config is compiled, not only where it is
  consumed.** The range check belongs in `configc`, which both sides pass
  through, so the failure names the file and the value instead of surfacing as
  behaviour. Consumers still refuse it, but as a backstop for hand-built
  fixtures — which is exactly how P77's SITL rows went wrong.
- **A checker shipped without tests gets verified once, by hand, and then
  never.** P77's split check had no automated test at all; it was confirmed
  interactively and left. This push added the range cases *and* the equality
  case it should have had.

## An uninstalled hook is indistinguishable from a passing one (P79, tooling)

- **A gate that is declared, reviewed and documented can still never run.**
  `freeflyer-vv` sat in `.pre-commit-config.yaml` with `stages: [pre-push]` from
  Push 57, was described in `PROGRESS.md` as "a push gate", and had never
  executed on any developer machine: `pre-commit install` wires only the
  `pre-commit` hook type by default, and nothing documented `--hook-type
  pre-push`. Fix the *installability*, not just the declaration —
  `default_install_hook_types` makes the documented command the whole install.
- **Verify a gate by watching it fire, never by reading its config.** The same
  error as asserting an emitter's own bytes instead of the reader's (P78), and
  as a green CI job whose tests all `GTEST_SKIP`ped since Push 41. Silent
  success is this repo's recurring failure mode; assume it until a run proves
  otherwise.
- **Duplicating a CI gate locally is only worth it if you wait for it.**
  Running the SITL suite and pushing before it finishes spends the compute and
  buys none of the feedback, while looking like diligence. Decide per change:
  run and wait, or skip and let CI gate — never the middle.
- **And a local pass is not a CI pass.** The binary run directly executes rows
  serially; CI runs them under `ctest -j$(nproc)`, where contention has produced
  failures this repo has already debugged once.

## P81 — a harness artefact masquerading as a tolerance

**Class:** a measurement's own precision limit read as the thing being measured.

The Polaris-vs-GMAT SGP4 comparison came in at 0.16–0.21 arcsec while the
Polaris-vs-FreeFlyer comparison on the *same element sets* came in at
0.027–0.035. The tempting move is to give GMAT a looser band and note that
"GMAT agrees less well" — which is a sentence that explains nothing and buries
a defect.

The cause was in the harness. GMAT's `UTCGregorian` epoch format carries three
decimal places, so setting a spacecraft epoch that way rounds it to the nearest
millisecond, and 0.5 ms at LEO orbital speed is **4 metres** of along-track
position. Switching the generated script to GMAT's `UTCModJulian` double (ULP
~0.3 µs, ~2 mm at LEO) brought the disagreement to 0.008–0.056 arcsec, in line
with FreeFlyer.

**Why it belongs in this file:** it is the same shape as the silent-success
class, one level up. A silent-success check reports green while testing nothing;
this reports a *number* while measuring the wrong thing, and the number is
plausible enough to be written into a band and defended forever. The tell was
comparative — two independent tools should not disagree with Polaris by 6x when
they agree with each other — and the discipline is to treat an unexplained ratio
between two measurements of the same quantity as a defect in the measurement
until shown otherwise.

**How to apply:** before widening a band, ask what the *harness* can resolve.
Epoch precision, report column width, output format decimals and unit
conversions are all part of the error budget and none of them appear in the
physics. When two references disagree with you by different amounts, the
difference between them is the diagnostic, not the two absolute numbers.

## P81b — assert the decision, not just the behaviour

`lib/frames/teme_eci.cpp` composes `eraBp00` to cross the FK5→GCRS frame bias.
Every test written against it initially passed *whether or not the term was
there*, because they compared the code to itself or to a band wide enough to
contain both answers.

The fix was to find a reference that separates the two hypotheses. GMAT reports
a state in both ICRF (GCRS) and `EarthMJ2000Eq` (FK5), so the fixture carries
both columns and the test asserts Polaris lands **closer to the ICRF one**.
That assertion reverses if the bias term is removed — and this was verified by
removing it (39.7/57.0 m becomes 81.7/60.1 m, test red), not assumed.

**How to apply:** when a line of code encodes a *choice* between two defensible
conventions, a passing test that would also pass with the other choice is not
covering it. Look for a reference that distinguishes them, and confirm the test
fails when the choice is inverted.

## P82 — "available" is not "usable"

**Class:** a resource check that stops one gate short of the one that matters,
and convicts working code with a plausible number.

A SITL row bounded its pointing accuracy by whether a star tracker was
available, computed from truth geometry: boresight unobstructed, Sun and Earth
keep-outs applied. It reported availability 1.0 while the estimator fused zero
trackers, which looked like a clear estimator defect and was reported as one.

Strengthening the check did not help, and that is the interesting part. Adding
the model's acquisition rate and acceleration envelopes — the tighter gate a
lost tracker must pass to re-acquire — felt like ruling out the innocent
explanation. The number stayed at 1.0, which read as confirmation. It had ruled
out *an* innocent explanation while leaving the real one untouched.

The real one: the estimator fuses **king-only** until `ST_ALIGN_CAL` has run,
because a non-king unit's as-mounted reading carries the two units' bias
difference. The unobstructed tracker was the non-king one. The king was inside
the Earth keep-out. No fine-mode source was ever available, the coarse fallback
was correct, and the 3 deg result was the requirement being met.

**Why it belongs here:** this is the P81 class one turn further on. There the
harness's own precision was read as the measured quantity; here the harness's
own *model of eligibility* was. Both produce a number that is plausible, stable
under strengthening, and wrong — and both survive review because the number
looks like evidence.

**How to apply:** when a check says a resource is available, ask what the
consumer additionally requires before it can use it — calibration state,
ownership, mode, a latch. Availability that stops at physics will pass a
resource the software is correctly declining. And when strengthening a check
leaves the answer unchanged, treat that as *weak* evidence, not confirmation:
ask what the strengthening could not have detected.

---

## P82b — A test that never crosses the boundary the bug lives on

`LOAD_TLE` declared its two lines as `string size 72`. The F´ autocoder emits
`Fw::CmdStringArg` for every command string regardless, and its capacity is the
framework-wide `FW_CMD_STRING_MAX_SIZE` — 40. Every 69-column line was
truncated to 40 characters. The command could never have loaded a TLE, from the
day it was written.

The library beneath it was correct and well covered: `TargetCatalog::loadTle`
takes `std::string_view`, and the unit suite fed it good lines, garbage lines
and a deliberately corrupted checksum. Every one of those tests passed, and
none of them could have failed, because none crossed the command boundary where
the loss happened. The FPP's `size 72` read as a specification and was
decoration.

The symptom, when a SITL row finally flew the uplink path, pointed away from
the cause: a truncated line still parses far enough to fail on a *specific*
field, so the first report was a checksum error four fields downstream.

**Why it belongs here:** the P81/P82a class is a measurement that looks like
evidence. This is its structural sibling — a *test suite* that looks like
coverage. Both leave a defect standing behind something green.

**How to apply:** for any value that crosses a framework boundary — a command
argument, a telemetry channel, a parameter, a port struct — find the concrete
type the autocoder actually emits and its capacity, and do not trust the width
written in the model. Then ask of any component whose library is well tested:
is there a test that exercises the *component*, or only the library it calls?
A declared size that nothing asserts is a comment. The fix pairs with the
guard: the handler now refuses anything that does not reassemble to exactly 69
columns, so the next truncation is a named refusal rather than a mystery.

---

## P82c — Running a subset and reporting it as the suite

Two CI failures on a branch reported as locally verified, from one cause: the
three `polaris_*` gtest binaries were run directly and called the gate. They are
not the gate. The F´ per-component tests live beside their components and are
separate executables that only `ctest` builds and runs — 101 cases CI runs and
that recipe never touches.

The first attempt to reproduce one of them locally *passed*, which nearly buried
it a second time: `ctest` does not build. It ran the previous commit's binary
and reported green.

Both failures were themselves informative rather than incidental:

  - the PrmDb record-count guard fired because two parameters were added, which
    is precisely the change it exists to catch;
  - a component test counted every port of an output array, so widening
    `orbitStateOut` from `[1]` to `[2]` for an unrelated feature turned "the
    component published once this cycle" into 2. The count was a statement
    about the component's behaviour and had quietly become a statement about
    how many consumers were wired.

**Why it belongs here:** P82b was a test suite that looked like coverage because
it never crossed the boundary the bug lived on. This is the same shape one level
up — a *test run* that looks like verification because it never ran the tier the
bug lived in. Both end in a confident "verified" over an untested gap.

**How to apply:** verify with the command CI uses, not a command that resembles
it. `ctest` after a build, not hand-picked binaries; and when a test does not
fail where you expect it to, check the binary's timestamp before concluding the
code is fine. Any assertion that counts port invocations should count one port,
or the number becomes a function of the topology rather than of the component.

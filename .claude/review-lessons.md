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

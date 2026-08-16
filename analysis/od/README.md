# `analysis/od` — the orbit-determination Monte Carlo campaign

Answers the questions `gnc::OrbitOd`'s own unit tests cannot: over long arcs of
real dynamics, against a truth stack the filter does not have, does the
estimate stay bounded, does the covariance still mean what it says, and what
happens when the receiver misbehaves in each of the ways §9.2 says it can.

Design doc §8.3 (onboard OD), §9.2 (GNSS FDIR), §12 (analysis tools), §13 (Monte
Carlo). Requirements REQ-ODP-001, -005, -006 in
`docs/requirements/orbit_od_propagation.rst`.

## The split, and why

| Piece | Where | Why |
|---|---|---|
| Flying the runs | `tests/mc/orbit_od_mc.cpp` → `polaris_orbit_od_mc` | C++. The filter under test **is** `gnc::OrbitOd`, driven against the real receiver model over the real environment. No GNC math is reimplemented. |
| Statistics, figures, report | this package | Python. Sampling statistics, plotting and report generation are the named exceptions in `analysis/CLAUDE.md`. |
| The interchange | JSONL under `build-artifacts/` | A derived artifact. Never committed. |

Unlike the detumble driver this one runs **in-process** rather than forking the
deployment: `OrbitOd` is flight code called directly, and its F´ component seam
does not exist yet. When that seam lands, this driver should move to the forked
form for the same reason detumble uses it.

## Running the campaign

Build it in `build-fprime-automatic-native`, **not** the `-ut` tree: F´'s own
`cmake/sanitizers.cmake` compiles the unit-test tree with ASan and UBSan
regardless of this project's `POLARIS_SANITIZE` option, and the campaign pays
1.6x for instrumentation it does not need.

```bash
uv run cmake --build build-fprime-automatic-native \
    --target polaris_orbit_od_mc -j8
```

**Smoke case — one run, one scenario, a few seconds.**

```bash
./build-fprime-automatic-native/bin/Linux/polaris_orbit_od_mc \
    --runs 1 --duration-s 3000 --scenario nominal \
    --out build-artifacts/mc/od_smoke.jsonl
```

**The full campaign — 30 runs.** Runs are sequential within one process;
parallelise by sharding across processes. Each shard is a whole run (all nine
scenarios) and is independently restartable. Budget about **430 MB RSS and
16 minutes per shard** idle, so the concurrency ceiling is whichever of cores
and memory runs out first; under 30-way contention on 24 cores the whole
campaign lands around half an hour.

`--duration-s` sets `nominal`'s arc and only `nominal`'s: it is the one
scenario without a `max_duration_s` of its own in `orbit_od_scenarios.hpp`. The
seven fault scenarios are capped at a day each and `latency_fast` at two
minutes, so the flag never lengthens them. Pass `604800` for the week-long
nominal arc.

```bash
JOBS=$(nproc)
mkdir -p build-artifacts/mc/od1d
seq -w 0 29 | xargs -P "$JOBS" -I{} sh -c \
  './build-fprime-automatic-native/bin/Linux/polaris_orbit_od_mc \
       --first-run $(echo {} | sed "s/^0*//;s/^$/0/") --runs 1 --duration-s 86400 \
       --out build-artifacts/mc/od1d/shard_{}.jsonl'
```

**Cost is per-cycle, not per-arc.** A shard's price is its GNC-cycle count times
about 12.5 ms; the truth propagation is a couple of percent of it. Lengthening
an arc at the 10 s nominal cadence is nearly free, while raising a cadence is
not — `latency_fast` at 50 Hz bought a fifth of the whole campaign's cycles out
of 0.02 % of its flight time before it was trimmed. Budget in cycles.

**Reading the result.** The analysis takes the shard directory directly, and
tolerates a campaign that is still flying — a shard whose last line is a
partial write is read short and named in the report's warnings.

```bash
PYTHONPATH=tools uv run --group analysis python -m analysis.od \
    build-artifacts/mc/od1d
```

It writes the interactive page and the rendered report under
`build-artifacts/orbit-od/` (`--out` to choose), opens the page (`--no-browser` to suppress,
which is what CI uses), and exits non-zero on any FAIL — 2 rather than 1 when
there was no readable campaign at all, because "the campaign says the filter is
inconsistent" and "there was no campaign" are different answers.

## Reproducing one run

The seed is `0x0D0D * 1000003 + run`, a function of the run index alone, so
adding runs to a campaign never moves an existing run's trajectory:

```bash
./build-fprime-automatic-native/bin/Linux/polaris_orbit_od_mc \
    --first-run <run> --runs 1 --duration-s <same> --scenario <name>
```

Every scenario of one run shares that seed, which is deliberate — it makes a
fault's effect readable against a common nominal baseline — and is why the
statistics treat a run's nine scenarios as **replicates, not samples**. Pooling
them flat would inflate N ninefold and let a two-run campaign satisfy a gate
written to demand ten.

## The scenarios

| Scenario | What it is for |
|---|---|
| `nominal` | Steady-state, no faults. The baseline every other scenario is read against, and the only one the consistency and ensemble statistics are measured on. |
| `outage_short` | Dropouts well inside the 300 s coast horizon. The solution must coast and stay valid. |
| `outage_horizon` | Dropouts at the horizon — the policy boundary itself. |
| `outage_long` | Dropouts past it. The solution must be **dropped**, not coasted indefinitely. |
| `spoof_step` | A discontinuous position jump; caught by the innovation gate. |
| `spoof_ramp` | A slow walk, each innovation inside a gate sized for one fix's noise. How far it gets is the campaign's headline measurement. |
| `jamming` | The geographic jamming map (`config/scenarios/jamming/`), so degraded and absent fixes arrive where geography puts them. Skipped with a note if the KML is unreadable. |
| `bad_data` | Clock jumps, geostationary-radius fixes and inflated reported sigmas — the receiver lying rather than going quiet. |
| `latency_fast` | The only scenario with fix latency armed, at 50 Hz over two minutes. The long arcs pass zero: at a 10 s cadence the delay line realises a whole poll rather than the datasheet's 50 ms. |

Every fault scenario's events are spread across its arc rather than clustered,
so each lands at a different point in the orbit's precession and the day/night
cycle and recovery transients do not overlap. A day is 15 orbits at the
reference vehicle's 5677 s period, which is what makes a day enough.

## Three ways to be wrong about a covariance

The campaign's centre of gravity. Each check sees a failure the others cannot.

| Check | Where | Reference | Blind to |
|---|---|---|---|
| **NEES** — state error against the covariance | `statistics.py` | Chi-square interval over run means (Bar-Shalom §5.4.2) | Error and covariance wrong by the same factor. |
| **NIS** — innovation against its predicted covariance | `statistics.py` | Same | The same, on the measurement side. |
| **Ensemble spread** — truth-derived σ against reported σ, per RIC axis | `ensemble.py` | The runs themselves | Anything below the sampling noise floor, ~13 % at 30 runs. |

The first two normalise the error by the very covariance under test, which is
what makes them cheap and what makes them blind in one direction: a mis-scaled
process noise, a variance stored where a standard deviation was meant, a units
slip — each scales the reported covariance and the realised error together and
sails through both. The ensemble check estimates the covariance a second time
from **truth alone**, and holds the filter's claim up against it.

Resolved in RIC and not as a scalar because orbit uncertainty is overwhelmingly
in-track: a covariance with the right total size and the wrong split across the
three axes is wrong in the way that matters, and no scalar can see it. The
frame is built from truth (`ricFromEci` in the driver), so a filter wrong about
where the vehicle is is not also allowed to be wrong about which way in-track
points.

Two further checks live on the C++ side, in
`tests/unit/orbit_od_covariance_test.cpp`, and need no campaign: the propagated
covariance against a finite-difference state-transition matrix, and phase-space
volume conservation (Liouville — two-body and J2 are conservative, so `det Φ`
is exactly one and any deviation is pure truncation error). Between them they
bound the propagation's *shape* and *volume* error with no reference
implementation at all.

## What is a criterion, and what is only a measurement

No requirement writes a number on the filter's position accuracy. Inventing one
here and then passing against it is the threshold-tuned-to-its-own-measurement
defect in `.claude/review-lessons.md`, so the accuracy figures are carried as
**provenance** and the criteria are the claims with a threshold this campaign
did not choose:

- the chi-square consistency intervals, split into their optimistic and
  pessimistic halves because only one of the two directions is unsafe;
- the ensemble/reported σ ratio, with a band wide enough (0.5–2.0) not to fail
  a healthy filter on sampling noise and narrow enough to catch a covariance
  wrong by a factor;
- the NIS gate's own configured rejection rate under nominal conditions;
- the fault policy — coast inside the horizon, drop past it, and an implausible
  fix refused **on the plausibility band** rather than one layer in. That last
  is a criterion and not a nicety: the band is the only check on the seed path,
  where there is no prior and therefore no innovation gate, so a GEO-radius fix
  refused by the gate would have been accepted whole one cycle earlier;
- campaign integrity — enough runs, enough samples, every scenario flown.

The spoof-ramp drift is a **warning** carrying its measured value. It is the
number a spoofing requirement should be written from, and there is no
requirement to write it against yet.

## Output schema

One JSON object per line, two kinds. `kind: "meta"` opens each (run, scenario)
and carries the scenario's intent, cadence, latency and arc; `kind: "sample"` is
one GNC cycle, emitted whether or not a fix arrived — the interesting rows are
the ones where none did. Fields are documented on
`analysis.od.records.ScenarioRun`. The refusal arrives as its **name**
(`"fix_implausible"`), from `gnc::refusalName` beside the enum, so this package
never carries a copy of the enum that drifts the first time a value is inserted.

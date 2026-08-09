# `analysis/sizing` — ADCS actuator sizing and design validation

Point it at a spacecraft config and it answers three questions:

1. **How much can these actuators actually do?** — the exact achievable set of
   the wheel array and the rod set, and the capability **guaranteed in every
   direction** rather than in the best one.
2. **Is this design big enough for its environment?** — the §5.3 disturbance
   budget, the momentum and torque drivers, and a pass/fail verdict with 30 %
   margin on each.
3. **What should the flight parameters be, and why?** — the PID gains, momentum
   envelope, desaturation hysteresis, detumble exit threshold and B-dot gain the
   design implies, each with its formula, its inputs and the reasoning.

```bash
PYTHONPATH=tools uv run --group analysis python -m analysis.sizing config/spacecraft/leo_smallsat.yaml
```

That writes **`index.html`, an interactive report, and opens it in your
browser.** The page is **light-themed unconditionally** — there is no
`prefers-color-scheme` override, and the plotly figures set their own light
background, axis and grid colours rather than inheriting. Two reviewers reading
the same file see the same document; a design-review artifact whose appearance
depends on the reader's OS setting is a liability. It is one self-contained file — plotly.js and the static figures are
inlined, and there is no external stylesheet, web font or CDN, so it works
offline and survives being emailed.

It is laid out as an engineering document: a sticky header carrying the vehicle,
the verdict and the provenance; a section nav; the rotatable momentum and torque
envelopes; the disturbance budget; the criteria table, **grouped by family**
(wheel momentum, wheel torque, magnetorquer authority, control tuning) and
sortable within each group, with each margin shown in absolute *and* percentage
terms in one cell; the assumptions and warnings; and the derived tuning with its
justifications.

**Scannable by default, complete on demand.** Each criterion row, card, caption
and warning shows one sentence plus its numbers; anything longer collapses into
a `<details>` disclosure. Nothing is deleted, the justifications are the reason
this report exists, they are simply one click away rather than between the
reader and the next row.

The console strings are set for a fixed-width terminal, so the page typesets
them on the way in. **Formulae are real LaTeX**, rendered server-side to a tight
transparent SVG by matplotlib mathtext in [`texmath.py`](texmath.py) and
embedded as a `data:` URI, baseline-aligned and sized in `em` so an equation
scales with the text around it. That keeps typeset mathematics without MathJax,
KaTeX or a web font, so the page stays one offline file. The LaTeX comes from a
lookup table keyed on the formula strings the report emits rather than from a
parser guessing at ASCII math, and `tests/analysis/test_sizing_texmath.py`
asserts the table covers every formula the analysis produces.

Everything else goes through [`mathfmt.py`](mathfmt.py): `N.m.s` becomes N·m·s,
`wn` becomes ω<sub>n</sub>. It is conservative by construction, anything it does
not recognise is passed through unchanged rather than guessed at, every string
is HTML-escaped *before* substitution, and it is also the fallback whenever a
formula has no LaTeX form or mathtext refuses one. Em dashes and `**emphasis**`
are console conventions and are dropped at render time. **The report objects and
the plain-text rendering keep their original strings**; nothing about the
verdict changes.

| Flag | Effect |
|---|---|
| `--no-browser` | write the page, open nothing. What CI and the tests use. |
| `--print` | also print the plain-text report, budget and justifications to stdout. |
| `--out DIR` | where everything lands (default `build-artifacts/analysis/sizing`). |
| `--no-plots` | skip the matplotlib figures; the HTML is still written, without the two static figures it would embed. |

The plain-text rendering is always written to `<out>/sizing_report.txt` whether
or not you ask for it on the console — it is the record, and it is what the
tests assert on. **The command exits non-zero if any criterion fails** — it is a
design gate, meant to stop a run before the simulation spends an hour confirming
a problem arithmetic already knew about.

## Pointing it at your own vehicle

Nothing in this package is specific to any spacecraft. Everything it can read
comes from your `config/spacecraft/*.yaml` and the `config/hardware/` entries
your `model_id`s resolve to:

| From the config | Used for |
|---|---|
| `inertia_kgm2`, `mass_kg` | every momentum and torque driver |
| `WheelAxesBody`, `WheelMaxTorqueNm`, catalog `max_momentum_nms` | the wheel envelopes |
| catalog `max_torque_nm` | the commanded-torque consistency check |
| `MtqAxesBody`, `BdotMaxDipoleAm2`, `MtqDutyFactor` | rod authority |
| `drag_area_m2`, `drag_cd`, `srp_area_m2`, `srp_cr`, `cp_offset_*_m`, `residual_dipole_am2` | the disturbance budget |
| `scenario.initial_state.orbit` | altitude, period, orbital rate, the field model |
| catalog `noise_ut_rms`, `ControlPeriodSec` | the B-dot noise floor |
| the `flight.attitudeController.*` tuning | the derived-parameter comparisons |

Anything sizing needs that **no config carries** — a launch vehicle's tip-off
rate, how often you intend to desaturate, whether your reference attitude is
LVLH or inertial — is a field of `SizingAssumptions` with a documented default,
and every one of them is printed in the report's assumptions block. Three are
also command-line flags:

```bash
PYTHONPATH=tools uv run --group analysis python -m analysis.sizing my_sat.yaml \
    --tipoff-deg-s 10 --desat-orbits 0.5 --slew-deg-s 1.0
```

For the rest, import the package:

```python
from analysis.control.vehicle import load_vehicle
from analysis.sizing import SizingAssumptions, sizing_report

vehicle = load_vehicle("config/spacecraft/my_sat.yaml")
report = sizing_report(
    vehicle,
    "config/spacecraft/my_sat.yaml",
    SizingAssumptions(
        tipoff_rate_radps=0.05,
        # An inertially-pointing vehicle: the aerodynamic term is the secular
        # one and SRP averages out, the opposite of the LVLH default.
        secular_fraction_aero=1.0,
        secular_fraction_srp=0.0,
    ),
)
print(report.format_text())
```

## What each criterion means

### Wheels

| Criterion | Question |
|---|---|
| **D1 tip-off absorption** | Can the wheels catch the body momentum at separation, `|J·ω_tipoff|`? |
| **D1b post-B-dot handover** | Can they catch what is left after detumble, at `DetumbleExitRadps`? |
| **D2 cyclic storage** | Can they store the momentum a once-per-orbit disturbance parks in them and gives back, `0.707·τ_cyc·T/4`? |
| **D3 secular accumulation** | Can they hold what accumulates between desaturations, `τ_sec·T_desat`? |
| **D4 slew agility** | Can they supply `|J·ω_slew|` for a commanded slew? *Only judged if you declare a slew rate.* |
| **wheel torque** | Can the array deliver the commanded body torque **in every direction**, plus disturbance rejection? |
| **commanded torque limit** | Does the flight parameter `WheelMaxTorqueNm` fit inside the installed wheel's catalog `max_torque_nm`? |
| **oversizing factor** | Is the capability within 10× the largest driver — or is this the wrong unit class? |

Each momentum criterion is judged against the **usable** envelope,
`min(zonotope r_in, MomentumEnvelopeNms)`. Momentum the vehicle raises an FDIR
event over is momentum it does not have. The tip-off driver is *also* judged
against the raw hardware radius, so a design whose only problem is its certified
ceiling is distinguishable from one whose wheels are genuinely too small.

**The commanded-torque check is the one criterion about a config mistake rather
than a design.** `WheelMaxTorqueNm` is a flight parameter; `max_torque_nm` is a
property of the unit bolted to the deck, and nothing links them — so swapping the
wheel and leaving the parameter behind is silent. It happened here: the reference
wheel went RW-X (0.025 N·m) → RW-X (0.002 N·m) and the parameter stayed, leaving
the FSW authorised to command 12.5× what the wheel can produce. Commanding past
the catalog value is *not* conservative in the safe direction — the wheel simply
does not deliver it, so the allocator's authority assumption is wrong and every
torque margin built on it is optimistic by the same factor. The reverse, a flight
limit *below* the catalog value, is legitimate derating and passes; a gap of more
than 2× raises a report **warning** instead, because authority the vehicle never
commands is mass and power it is carrying for nothing.

### Magnetorquers

| Criterion | Question |
|---|---|
| **M1 desaturation authority** | Do the rods beat the **secular** disturbance torque? This is where the disturbance budget lands: fail it and the wheels saturate no matter how large they are. |
| **M2 detumble authority** | Can they remove the tip-off momentum inside the budget? The implied fast-phase duration is reported. |
| **M3 B-dot noise floor** | Is `DetumbleExitRadps` above `σ√2/(Δt·|B|)`, the rate at which B-dot's input is its own noise? |

### Derived tuning

Bandwidth against the sampling bound, the damping ratio `Kp` and `Kd` jointly
imply, the momentum envelope against the SISO validity boundary, both halves of
the desaturation ordering invariant (`exit < enter < envelope`), the B-dot gain
against the Avanzini & Giulietti convergence floor, and whether the wheels can
absorb the handover momentum the noise floor forces on them.

## Reading the envelope figure

The headline figure in `index.html` is the momentum envelope in 3D — rotate it,
zoom it, and toggle any surface off in the legend. Four surfaces in body
momentum space, plus the drivers:

- **Blue hull, drawn with its edges — the zonotope.** Everything the array can
  reach with each wheel inside its own limit; this is what the shipped L∞
  allocator delivers. The fill is nearly transparent and the **wireframe** is what
  gives it its shape, so the two surfaces nested inside it stay visible — the
  point of the figure is that they are *inside*, and a translucent blob with no
  edges cannot show that. Toggle `zonotope edges` in the legend to drop it.
- **Blue sphere — the hardware inscribed radius `r_in`.** The momentum the
  wheels guarantee in *every* direction.
- **Orange sphere — the *usable* envelope, `MomentumEnvelopeNms`.** What the
  certified analysis covers. **This is the distinction the figure exists to
  carry**: "the wheels can hold it" and "the analysis covers it" are different
  claims, and on the reference vehicle they differ by two orders of magnitude.
  Every momentum criterion is judged on the smaller of the two, because momentum
  the vehicle raises an envelope event over is momentum it does not have.
- **Purple ellipsoid — the L2 allocator's reach** under an RMS command limit.
  Always inside the zonotope; if you switch `AllocMethodSel` to 0, this becomes
  your capability.
- **Arrows — each momentum driver** at its ×1.3 margin, drawn along the array's
  *weakest* direction, which is where `r_in` is attained. Green and inside, or
  red and outside; the hover text says which in words.

The gap between the hull and the blue sphere is the layout's anisotropy — on
the reference four-wheel pyramid, the best direction reaches 1.15 N·m·s and the
guarantee is 0.82, a 29 % difference. A per-body-axis sizing check would have
quoted the larger number.

The drivers on this class of vehicle are orders of magnitude below the envelope,
so the arrows are short: `momentum_drivers.png`, embedded lower in the page, is
the readable comparison — a log axis is the only way the drivers and the
envelope share one plot. **A bar above the green line is a driver the design
cannot hold.** The same PNGs are still written to `--out` alongside the page.

## When a criterion fails

| Failing | What to change, cheapest first |
|---|---|
| A momentum driver, envelope-limited | The ceiling is the *analysis*, not the wheels. Widen the certified envelope (needs the MIMO work in §8.5), or lower the driver — a slower tip-off, more frequent desaturation, a smaller commanded slew. |
| A momentum driver, hardware-limited | More wheels before bigger wheels: a fourth wheel on a three-wheel vehicle raises `r_in` far more than the same mass in rotor. Then a better spread — the inscribed radius is set by the *worst* direction, so a layout with one weak axis wastes capability everywhere else. Then a larger `max_momentum_nms`. |
| Wheel torque | Same ladder. Check whether `PidMaxTorqueNm` really needs to be that large — the demand is usually the tuning, not the physics. |
| Commanded torque limit | A config edit, not a design change: lower `flight.attitudeController.WheelMaxTorqueNm` to the installed wheel's `max_torque_nm`, or install the wheel the parameter was written for. Re-read every torque margin afterwards — they were all computed on the old number. |
| Oversizing factor | Consider a smaller wheel. This is not a safety failure; it is mass, power and cost bought and not used. |
| **M1 desaturation** | Reduce the *secular* disturbance: tighter CP-CM control (the SRP and aero lever arms are the whole term), or magnetic cleanliness for the dipole. Then more or larger rods. Then a longer duty factor — bounded by the §7 quiet-window schedule. |
| **M2 detumble** | Larger rods, a longer budget, or accept a slower fast phase. Note the caveat: B-dot damps only the rate perpendicular to the field, so no rod size removes the residual spin about the field line. |
| **M3 noise floor** | A quieter magnetometer, a longer B-dot differencing interval (up to `BdotMaxSampleDtSec` — the secant stops being the tangent past that), or raise `DetumbleExitRadps` above the floor and check the wheels can still absorb the handover. |
| Ordering invariants | A config edit, not a design change: `exit < enter < envelope`. |
| B-dot gain | Raise `BdotGainNms`; the floor guarantees convergence and everything above it buys decay rate until the rods saturate. |

## Units and frames

SI throughout: momentum N·m·s, torque N·m, dipole A·m², field T, rates rad/s,
inertia kg·m², lengths m. Every vector is **body frame**; wheel spin axes and rod
dipole axes are unit vectors in build order. Friendly units (deg/s, µN·m, µT)
appear only in the report and on the figures, which is the presentation
boundary. No quaternion, frame transform or propagation is computed anywhere in
this package — see `analysis/CLAUDE.md` for why that boundary matters.

## What this tool is not

It is a **closed-form worst-case** budget. Every disturbance is its analytic
maximum at a static attitude, the atmosphere is the plant's exponential baseline
rather than its NRLMSIS truth model, and the geomagnetic field is a degree-1
truncation of IGRF. That is the right fidelity for sizing — a bound in closed
form beats a time history you have to interpret — but when a criterion is
*close*, the answer is a simulation, not a wider analytic margin.

No requirement in the baseline is written on actuator sizing, so no criterion
here carries a requirement ID; the report says so in a standing warning.

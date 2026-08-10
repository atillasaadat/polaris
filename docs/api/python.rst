Tools & Python
==============

The **config compiler** (``tools/configc/``, design doc §19.3) is the first
Python component: it validates the single-source-of-truth spacecraft/scenario
YAML against a Pydantic schema, resolves hardware ``model_id`` references against
the library in ``config/hardware/``, and emits provenance-stamped
F´-param / sim / analysis artifacts from **one resolved object**
(REQ-CFG-001/002/003). Run it with::

   PYTHONPATH=tools uv run python -m configc --help

The **GMAT golden harness** (``tools/gmat/``) and the **ephemeris fitter**
(``tools/ephem/``) generate committed reference fixtures; neither runs in the
normal test path (they need the GMAT binary / a NAIF kernel). See
:doc:`/guides/verification`.

The **fetch tools** refresh the committed external reference data under the
verbatim-data rule (design doc §3.7). Each downloads from the authoritative source, sanity-checks that the file
parses, and writes it **verbatim** — the parsing that flies happens on the C++
side, so these never emit a bespoke intermediate. They are ground-side and never
run in CI:

- ``tools/eop/`` — IERS ``finals.all.iau2000`` Earth-orientation product
  (mirrors tried in order); ``finals.py`` also exposes the column parser used to
  validate the download.
- ``tools/gravity/`` — the EGM2008 ``.gfc`` spherical-harmonic model, truncated
  to a requested maximum degree in its native format (the full model is ~100 MB).
- ``tools/igrf/`` — the IAGA IGRF-14 coefficient table (committed verbatim) plus
  ``golden``, which regenerates the derived field fixture with ``pyIGRF14``, the
  official IAGA reference implementation — a fetch input, never vendored.
- ``tools/spaceweather/`` — CelesTrak ``SW-All.csv``, the solar radio flux
  (F10.7) and geomagnetic (Ap) record NRLMSIS is driven by, parsed by
  ``sim/world/space_weather_file.cpp``.

``tools/dev/collect_gtest_trace.py`` is the C++ half of the traceability gate:
it converts GoogleTest JSON output (``--gtest_output=json:…``) into
``docs/_generated/verif_gtest.json``, the sphinx-needs external-needs file that
turns each test's ``RecordProperty("verifies", "REQ-…")`` into a ``verified by``
back-link in the RVTM. With no input files it writes an empty-but-valid file so
the docs build still resolves.

Analysis: shared reporting convention
-------------------------------------

``analysis/common/`` carries the pattern every analysis tool follows (design doc
§13; ``analysis/CLAUDE.md``). An analysis with a pass/fail criterion produces
**both** a structured result and plots that state their own verdict:

- ``report`` — ``Criterion`` (requirement ID, threshold, measured value, units,
  sense) computing its own margin in absolute and percentage terms, and
  ``AnalysisReport`` gathering them with the configuration provenance and the
  modelling assumptions in force. ``format_text()``/``write_text()`` render it;
  tests assert on the structured object, never on parsed text.
- ``plotting`` — threshold lines, measured-value annotations carrying the number
  *and* the word PASS or FAIL, and verdict titles. Colour is never load-bearing
  on its own.

Linear control analysis
-----------------------

``analysis/control/`` (design doc §8.5, §12) is the first live package under
``analysis/``. It measures the **as-flown** attitude loop from the same
committed ``config/spacecraft/*.yaml`` the flight software is tuned from, and
verifies REQ-ACTL-006/007/008. Built on **numpy and scipy only** — no
control-systems library; the margin extraction, the disk margin and the Gramians
are ours, and are validated against closed-form cases before being applied to
the vehicle.

- ``vehicle`` — resolves inertia, wheel and rod axes, PID gains, the control
  period, the wheel momentum capacity (from the hardware catalog the
  ``model_id`` resolves to) and the estimator's noise budget out of the config,
  through the config compiler's own loader. Nothing is transcribed.
- ``plant`` — the linearised attitude model: ``Loop`` (a pair of polynomials
  plus a sample period), the per-axis :math:`1/(J_{ii}s^2)` plant, the coupled
  six-state form, the shipped PID, the **sampled-data** loop at the GNC period
  (``scipy.signal.cont2discrete`` ZOH plus the flight code's forward-Euler
  integrator, realised in state space and converted once), and ``siso_coupling``
  — the validity check that says when a per-axis model stops describing the
  vehicle.
- ``margins`` — gain, phase, sensitivity-peak and disk margins extracted from
  our own frequency response. Reports the gain margin in **both directions**,
  because the loop is type 3 and therefore conditionally stable.
- ``report`` — the per-axis ``MarginReport`` and the shared
  ``AnalysisReport`` covering all three requirements.
- ``controllability`` — Kalman rank and finite-horizon Gramians for the wheel
  pyramid, its 3-of-4 failure subsets, and the magnetorquers (instantaneously
  rank-deficient; full rank only orbit-averaged).
- ``observability`` — the (attitude error, gyro bias) model under two vector
  measurements, on the same information-matrix metric the flight seed gate uses.
- ``field`` — a tilted dipole parsed from the committed IAGA IGRF-14 table, for
  the orbit-averaged magnetic case only.
- ``plots`` — annotated Bode/Nyquist/margin/controllability/geometry figures and
  the rendered text report, into a caller-supplied directory (default under
  ``build-artifacts/``, never committed).

Detumble Monte Carlo
--------------------

``analysis/detumble/`` (design doc §13, §23.2) is the second live package, and
the first built on the **campaign** pattern: the runs are C++
(``tests/mc/detumble_mc.cpp`` → ``polaris_detumble_mc``, which forks the real
deployment and drives the real closed loop, so no GNC math is reimplemented),
and this package owns the sampling statistics, the figures and the report. It
characterises the residual-spin tail REQ-ACTL-001 records as owed — the time to
the ``DetumbleExitRadps`` completion predicate across dispersed tip-off rate and
direction, attitude, RAAN, argument of latitude and epoch.

- ``records`` — the driver's JSONL per-run schema. Every record carries its
  dispersion draw, its derived seed and the compiled ``config_hash``, so a run
  is re-flyable from its own record.
- ``statistics`` — the distribution, and the **distribution-free** upper
  tolerance bound the handover time is read from (Wilks order statistics
  [wilks1941], [conover1999]: 59 runs for a 95/95 bound from the sample
  maximum, 93 from the second largest). Right-censored runs are counted, never
  quietly folded in.
- ``report`` — REQ-ACTL-001's fast-phase bound re-measured across the
  dispersion as pass/fail criteria, with the proposed handover time carried as a
  *proposal* in the provenance and warnings rather than as a criterion: no
  requirement is written on it yet, and passing against a threshold invented
  here would be circular.
- ``plots`` — the rate ensemble, the empirical CDF of time-to-completion, and
  the scatter against the geometric driver.

``analysis/detumble/README.md`` carries the recipe for flying the campaign and a
smoke case that proves the harness in a couple of minutes. The campaign itself
is deliberately not a ctest, for the same reason ``tests/benchmark/`` is not.

ADCS actuator sizing
--------------------

``analysis/sizing/`` (design doc §12, §7, §8.5) is the third live package and
fulfils the ``momentum/`` item on the Phase-11 list, broadened: it sizes the
actuators, validates a candidate design against its drivers with 30 % margin,
and derives the flight tuning the design implies. Like ``control/`` it reads
matrices and scalars from the committed ``config/`` YAML and computes no
quaternion, frame transform or propagation.

- ``envelope`` — the exact achievable set of an actuator array. The zonotope's
  support function :math:`h(u) = a_{\max}\lVert W^\top u\rVert_1`, its
  **inscribed** radius (found exactly by enumerating facet normals, and
  cross-checked against a dense spherical sample) and its circumscribed radius,
  plus the L2 ellipsoid inscribed in it. Sizing uses the inscribed radius —
  the capability guaranteed in *every* direction — never the best-direction
  figure. Degenerate layouts are refused rather than approximated.
- ``disturbances`` — the §5.3 budget in closed form at the config's own orbit:
  gravity gradient, aerodynamic, SRP and residual magnetic, each split into a
  **secular** part (which sizes desaturation) and a **cyclic** part (which sizes
  storage) by an assumption the report states. The atmosphere band table is
  parsed from ``sim/world/atmosphere.cpp`` and the field from the committed IAGA
  table through ``control.field``.
- ``wheels`` — momentum drivers D1–D4 and the torque driver, judged against
  ``min(zonotope r_in, MomentumEnvelopeNms)``, plus an oversizing check.
- ``magnetorquers`` — desaturation authority against the secular disturbance,
  detumble authority, and the **B-dot noise floor** :math:`\sigma\sqrt2/(\Delta
  t\,\lvert B\rvert)`: the body rate below which the law commands on its own
  measurement noise, and therefore a lower bound on any detumble exit threshold.
- ``parameters`` — the derived tuning with its justification: PID gains, the
  momentum envelope (reusing ``control.plant.siso_coupling``), the desaturation
  hysteresis ordering, the detumble exit threshold (refused if it would sit
  below the noise floor) and the B-dot gain against the Avanzini & Giulietti
  convergence floor.
- ``assumptions`` — everything sizing needs that no config carries, each with a
  documented default and each rendered into the report's assumptions block.
- ``report``/``plots`` — the shared ``AnalysisReport``, the momentum-envelope
  figure and its driver companion, the disturbance budget, and the magnetorquer
  authority/noise-floor figure.
- ``interactive``/``html`` — the **default output**: a single self-contained
  ``index.html``, opened in a browser unless ``--no-browser`` is given, laid out
  as an engineering document — sticky header with the vehicle, verdict and
  provenance; section nav; and the criteria **grouped by family** (wheel
  momentum, wheel torque, magnetorquer authority, control tuning), sortable
  within each group, each margin carrying its absolute *and* percentage figure
  in one cell. plotly figures for the momentum and torque envelopes in 3D (with
  each driver drawn as a labelled vector along the array's weakest direction,
  and the *usable* ``MomentumEnvelopeNms`` sphere drawn distinctly from the
  hardware zonotope), the disturbance budget and the per-criterion margins, plus
  the assumptions and warnings, and the derived tuning as cards carrying each
  formula, its inputs and its reasoning. plotly.js is inlined and the matplotlib
  figures are embedded as ``data:`` URIs, with no external stylesheet or web
  font, so the page fetches nothing at runtime; every value is escaped, because
  config-derived strings are untrusted input. ``--print`` keeps the console
  report, whose content is unchanged and which is written to
  ``sizing_report.txt`` regardless.
- ``mathfmt`` — the presentation formatter the page renders through, and the
  reason it can typeset mathematics while staying one offline file. It escapes
  first and then substitutes only tokens it recognises, so ``Kp = J * wn^2``
  becomes :math:`K_p = J\,\omega_n^2` and ``N.m.s`` becomes ``N·m·s`` using
  Unicode plus ``<sub>``/``<sup>`` rather than MathJax or KaTeX. It is
  deliberately conservative — a constant name such as
  ``MAX_SISO_COUPLING_RATIO``, a config path, or a note that opens with a symbol
  (``eta * m_in * ...``) is passed through untouched rather than guessed at, so
  the failure mode is an unconverted string and never a corrupted one. It is
  rendering only: the ``AnalysisReport`` and the plain-text report keep their
  original strings.

``analysis/sizing/README.md`` covers pointing it at your own config, what each
criterion means, how to read the envelope figure, and what to change when one
fails.

Run any of the three packages with the ``analysis`` dependency group::

   uv run --group analysis pytest tests/analysis

.. note::

   Like ``tools/``, these modules are documented here in prose rather than by
   autodoc: pulling SciPy and matplotlib into the docs-only toolchain would
   double the docs CI job for a rendering convenience. The
   numpydoc docstrings are still the authority and are written to that standard.
   ``bindings/`` (pybind11) has no modules yet; autodoc directives land here with
   it, at which point this page documents the Python-facing API of the same C++
   that flies.

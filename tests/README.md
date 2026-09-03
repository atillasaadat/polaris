# `tests/` — Verification & validation

Test pyramid (design doc §23.1):

| tier | where | cases | binary / runner |
|---|---|---|---|
| unit | `unit/` over `lib/` + `sim/` | 894 | `polaris_unit_tests` |
| F´ component | **beside the component**, `flight/PolarisFsw/<C>/test/ut/` | 101 in 4 exes | `ctest -R _ut_exe` |
| integration | `integration/` — closed-loop and SITL, incl. FDIR | 73 | `polaris_integration_tests` |
| golden | `golden/` — GMAT/FreeFlyer fixtures; comparison *is* the regression tier | 15 | `polaris_golden_tests` |
| python | `tools/`, `analysis/`, `freeflyer/`, `mc/` | 582 collected | `pytest` |

**The component tier is the one that gets missed, and it has bitten this repo.**
Those tests are not in any of the three `polaris_*` binaries — they are separate
executables that only `ctest` (or `fprime-util check`) builds and runs, which is
why the "Build & verify locally" recipe below runs `ctest` rather than the three
binaries alone. See that section for the trap that goes with it.

`freeflyer/` holds the FreeFlyer V&V and viewer suites; the cross-validation
cases **skip visibly** without a licensed local install (`tests/freeflyer/
conftest.py` owns the skip, so a machine without the seat is never silently
green). Two of the GMAT cases likewise skip without a local GMAT.

Tests annotate the requirement(s) they verify so traceability flows into the
Sphinx-Needs matrix:
- **Python:** `@pytest.mark.verifies("REQ-…")` + `record_property("margin_pct", …)`.
- **C++ (GoogleTest):** `RecordProperty("verifies", "REQ-…")` + `RecordProperty("margin_pct", …)`.

See `tests/conftest.py` and `tools/dev/collect_gtest_trace.py` for the collectors.

## Running them

The full recipe — including generating the UT tree with `-O2`, which matters
because unoptimised Eigen makes the SITL rows ~30x slower — is in
`PROGRESS.md` § "Build & verify locally". The short forms:

```bash
# Everything CI runs, the way CI runs it. BUILD FIRST: ctest does not build,
# and a stale binary passing is a green run that tested the previous commit.
uv run cmake --build build-fprime-automatic-native-ut -j"$(nproc)"
uv run ctest --test-dir build-fprime-automatic-native-ut -j"$(nproc)" --output-on-failure

# One tier at a time (the labels CI uses)
uv run ctest --test-dir build-fprime-automatic-native-ut -L unit   -j"$(nproc)"
uv run ctest --test-dir build-fprime-automatic-native-ut -L golden -j"$(nproc)"
uv run ctest --test-dir build-fprime-automatic-native-ut -R _ut_exe -j"$(nproc)"   # F´ components

# One case, with its output — the usual debugging loop
./build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests \
    --gtest_filter='SitlPointingGuidance.*'
```

SITL rows are slow by construction (each flies a closed-loop mission segment in
lockstep, minutes of wall clock each), so `-j` and a `--gtest_filter` are how
the loop stays usable. `POLARIS_KEEP_SITL_LOGS=1` keeps the per-row work
directory, whose `fsw.log` is the deployment's event stream and the only
channel back from the flight software.

## Watching a row instead of reading its assertions

Any SITL row can be rendered in FreeFlyer — the pointing rows have four windows
including the payload-camera and star-tracker points of view:

```bash
PYTHONPATH=tools uv run --group analysis python -m freeflyer run --replay --pace 100 \
    --scenario SitlPointingGuidance.TracksAnUploadedStateVectorTarget
```

`tools/freeflyer/README.md` carries the scenario table (what each row shows and
which to watch first), the live-vs-`--replay` choice, and the rule that every
new flight or sim capability ships its own row.

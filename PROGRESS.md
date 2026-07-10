# Polaris — Progress

Living status of the build-out, tracked against the design doc's development
phasing (`docs/design/Polaris_Design_Document.md` §24). This file summarizes
**what exists and is verified**; the authoritative spec and the requirements
baseline (`docs/requirements/`) remain the sources of truth.

**Current phase:** Phase 0 — Foundations (in progress)
**Last updated:** Phase 0, Push 2a + early F´ integration (Push 4 pulled forward)

---

## Status at a glance

| Area | State |
|---|---|
| Repo skeleton, licensing, tooling | ✅ done |
| Requirements ICD (Sphinx-Needs) | ✅ authored (72 reqs seeded) |
| Docs site + traceability gate | ✅ builds clean under `-W` |
| CI skeleton (docs gate + lint active) | ✅ in place; build/test stages stubbed |
| `lib/` math foundations (constants, typed vectors, quaternion) | ✅ done + tested |
| `lib/` time (TAI clock, GPS/TT scales, leap seconds, UTC) | ✅ done + tested |
| F´ v4.2.2 barebones deployment (buildable/runnable) | ✅ done (Push 4 pulled forward) |
| `lib/` frames, canonical state, ephemeris | ⏳ next (Push 2c–2d) |
| Config compiler · GMAT golden harness | ⏳ Pushes 3, 5 |

**Build/test health:** C++ builds clean under a strict `-Werror` warning set;
**39/39 unit tests pass** (also green under ASan/UBSan); docs build is green;
pre-commit (clang-format + ruff) is clean.

---

## Phase 0 — what's done

### Push 1 — Foundations (skeleton, licensing, requirements, docs, CI)
- **Repository skeleton** per design §22.3 (`lib/ flight/ sim/ bindings/ analysis/
  config/ tests/ tools/ mc/ docs/ .github/`).
- **Dual-license model:** PolyForm Noncommercial 1.0.0 for the public
  (`LICENSE`) — free for research/education/university smallsats with attribution
  — plus a separately-sold commercial license (`LICENSING.md`, `NOTICE`).
  Third-party attributions in `THIRD_PARTY_NOTICES.md`.
- **Tooling baseline:** `.gitignore`, `.editorconfig`, `.clang-format`
  (clang-format 18), `.clang-tidy` (JPL Power-of-Ten checks),
  `.pre-commit-config.yaml`.
- **Requirements ICD** (`docs/requirements/`) in **Sphinx-Needs**:
  `REQ-<SUBSYS>-NNN`, levels L0/L1/L2, T/A/I/D verification, margin fields, and
  an auto-generated RVTM. **72 requirements** seeded across 16 subsystem groups
  (5 mission, 14 system/program, 53 subsystem), all at status `reviewed`.
- **Docs site** (Sphinx + numpydoc + PyData theme + Breathe/Doxygen + bibtex +
  sphinx-needs), `refs.bib`, glossary. The build is a CI gate (`sphinx-build -W`):
  a baselined requirement without a verifying test fails the build.
- **CI** (`.github/workflows/`): `docs.yml` (doc-gate + Pages publish) and
  `ci.yml` (pre-commit lint active; §23.2 build/test stages stubbed for later
  pushes).
- **Traceability collectors** for pytest and GoogleTest wired into the RVTM.

### Push 2a — `lib/` math foundations
- **Build system:** top-level CMake; **Eigen 3.4.0** and **GoogleTest 1.15.2**
  pinned via FetchContent (marked SYSTEM so `-Werror` only polices Polaris code);
  C++17; CTest.
- **`lib/constants`** — shared physical-constants registry (WGS84 `a/f/GM/ω`,
  `TAI−GPS = 19 s`, `TT−TAI`, *c*), each value sourced.
- **`lib/math`** — the convention bedrock:
  - **Frame tags** + `Vec3<Frame>` typed vectors: mixing frames across a
    boundary is a **compile error** (Golden Rule 4).
  - **JPL scalar-first `Quaternion`** `[q0,q1,q2,q3]`, canonical `q0≥0`, JPL
    product (`A(a·b)=A(a)A(b)`), passive attitude matrix (`v_rot=A(q)v_ref`),
    Shepperd extraction; tagged `Quat<To,From>` boundary wrapper.
- **18 unit tests, all REQ-traced**, 100% pass. The quaternion convention is
  validated by self-consistency (composition vs DCM product, Shepperd
  round-trip, conjugate = transpose, passive ROT3). External GMAT-golden
  validation is deferred to Push 5.
- **Reviewed** with the `fsw-code-reviewer` (no critical/memory/exception
  issues; math verified by hand). Findings fixed: added `trawny2005` /
  `shepperd1978` / `wgs84` bibkeys, made `FromRotationMatrix` always return a
  unit quaternion, added `isFinite()` boundary guards.

**Requirements coverage:** 7 of 72 requirements now have a verifying test
(REQ-SYS-001/002/003/004/006, REQ-CONV-003/004) — the first traces in the RVTM.

### F´ integration — Push 4 pulled forward (out of dependency order)
- **F´ vendored as a submodule at tag `v4.2.2`** — the **latest** F´ release
  (published 2026-04-24; verified against `nasa/fprime` releases). No upgrade is
  available or needed.
- **Buildable, runnable barebones deployment** at `flight/PolarisFsw` — a minimal
  F´-native topology (`CdhCore` + `ComCcsds` + `DataProducts` + `FileHandling`
  subtopologies), unified into the top-level build. No GNC components yet; this is
  the plumbing that later hosts the Phase 3 FSW skeleton (§24).
- **Rationale for reordering:** the F´ deployment builds independently of the
  `lib/` numerics, so standing it up early de-risks the toolchain/topology work
  without blocking on Push 2b–2d. The design's §24 phasing is a dependency guide,
  not a strict sequence — this reordering respects the actual dependency graph.

### Push 2b — `lib/time` library
- **`lib/time`** — the onboard time foundation (design doc §3.2):
  - **`Duration`** — signed int64-nanosecond interval; `constexpr` arithmetic;
    exact-seconds and rounded fractional-seconds constructors.
  - **Strongly-typed uniform scales** `Tai` / `Gps` / `Tt` (`Instant<Scale>`):
    monotonic int64-ns count since the 1970 scale epoch. Mixing scales is a
    **compile error** — the same boundary-typing discipline as the frame tags.
  - **TAI master clock** (REQ-SYS-001) with a **two-part high-precision form**
    (int64 s + double frac in `[0,1)`) for long propagation/ephemeris arcs
    (REQ-CONV-005).
  - **Constant-offset conversions** `GPS→TAI = +19 s` (REQ-CONV-001) and
    `TT−TAI = +32.184 s`, int64-exact and `static_assert`-checked against the
    shared constants registry.
  - **Leap-second table** — fixed-capacity (no heap), historical IERS ΔAT record
    through 2017-01-01 (=37 s), plus a **frozen/settable** table for reproducible
    MC (§3.5); **UTC derivation** (`utc.hpp`) with exact round-trips including the
    inserted leap second (`hh:mm:60`). UTC is ground-only; the master clock stays
    TAI.
  - **`civil.hpp`** — branch-free proleptic-Gregorian date↔serial-day algorithms
    (Hinnant), shared by the leap table and UTC conversion.
- **21 new unit tests, all REQ-traced**, 100% pass (39/39 total; green under
  ASan/UBSan). Validated by round-trips, known IERS/GPS-epoch anchors, and the
  exact integer offsets. External GMAT-golden time/scale cross-validation is
  deferred to Push 5.
- **Scope note:** **TDB** (and the TT↔TDB periodic term) is deferred to **Push 2d**
  (ephemeris), where it is actually consumed; TAI/GPS/TT/UTC are complete here.
- **Reviewed** with the `fsw-code-reviewer` (no CRITICAL). Findings fixed:
  finiteness/range guard on `Duration::fromSecondsF` (no `llround` UB); checked
  `addEntry` returns + a capacity `static_assert` in the table factories; `frozen`
  table re-seated so its constant ΔAT holds at/before the epoch; `isValidUtc`
  boundary validator + `taiFromUtc` precondition; TT−TAI `static_assert` now
  cross-checks the constants registry; added the `hinnant2016` bibkey.

**Requirements coverage:** 9 of 72 requirements now have a verifying test
(adds REQ-CONV-001, REQ-CONV-005 to the Push 2a set) — 39 verifying tests in the
RVTM.

---

## What's next

- ~~**Push 2b** — time library: TAI/UTC/GPS/TT, int64-ns master clock + two-part
  high-precision form, leap-second handling.~~ ✅ **done** (TDB moved to 2d).
- **Push 2c** — frames (LVLH/RIC geometric now; full ECI↔ECEF reduction with
  GMAT in Push 5) + the canonical `EstimatedState` / `TruthState` structs.
- **Push 2d** — onboard Chebyshev / ground SPICE ephemeris interfaces; **TDB**
  (TT↔TDB periodic term) lands here where ephemeris consumes it.
- **Push 3** — config schema + hardware-model library + config-compiler stub.
- ~~**Push 4** — F´ submodule + buildable deployment.~~ ✅ **done early** (F´
  `v4.2.2`, see above).
- **Push 5** — GMAT golden-data harness + first fixtures.
- **Push 6** — activate the full §23.2 CI pipeline.

---

## Build & verify locally

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `uv sync`
creates `.venv`, provisions Python 3.12, and installs the pinned toolchain
(`cmake`/`ninja` as wheels). Only `doxygen` is a system package.

```bash
# One-time
uv sync                             # F´ toolchain + dev tools (from uv.lock)

# F´ build + unit tests
uv run fprime-util generate && uv run fprime-util build
uv run fprime-util build --ut && uv run fprime-util check   # 39/39

# Docs site (warnings are errors; same gate as CI — renders the lib/ C++ API)
sudo apt-get install -y doxygen     # one-time
uv run --group docs bash tools/dev/build_docs.sh            # -> docs/_build/html/index.html

# Lint
uv run --only-group dev pre-commit run --all-files
```

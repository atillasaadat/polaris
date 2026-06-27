# Polaris — Progress

Living status of the build-out, tracked against the design doc's development
phasing (`docs/design/Polaris_Design_Document.md` §24). This file summarizes
**what exists and is verified**; the authoritative spec and the requirements
baseline (`docs/requirements/`) remain the sources of truth.

**Current phase:** Phase 0 — Foundations (in progress)
**Last updated:** end of Phase 0, Push 2a

---

## Status at a glance

| Area | State |
|---|---|
| Repo skeleton, licensing, tooling | ✅ done |
| Requirements ICD (Sphinx-Needs) | ✅ authored (72 reqs seeded) |
| Docs site + traceability gate | ✅ builds clean under `-W` |
| CI skeleton (docs gate + lint active) | ✅ in place; build/test stages stubbed |
| `lib/` math foundations (constants, typed vectors, quaternion) | ✅ done + tested |
| `lib/` time, frames, canonical state, ephemeris | ⏳ next (Push 2b–2d) |
| Config compiler · F´ · GMAT golden harness | ⏳ Pushes 3–5 |

**Build/test health:** C++ builds clean under a strict `-Werror` warning set;
**18/18 unit tests pass**; docs build is green; pre-commit (clang-format + ruff)
is clean.

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

---

## What's next

- **Push 2b** — time library: TAI/UTC/GPS/TT/TDB, int64-ns master clock +
  two-part high-precision form, leap-second handling.
- **Push 2c** — frames (LVLH/RIC geometric now; full ECI↔ECEF reduction with
  GMAT in Push 5) + the canonical `EstimatedState` / `TruthState` structs.
- **Push 2d** — onboard Chebyshev / ground SPICE ephemeris interfaces.
- **Push 3** — config schema + hardware-model library + config-compiler stub.
- **Push 4** — F´ submodule (`fprime/` @ v4.2.1) + buildable deployment.
- **Push 5** — GMAT golden-data harness + first fixtures.
- **Push 6** — activate the full §23.2 CI pipeline.

---

## Build & verify locally

```bash
# C++ build + unit tests
cmake -S . -B build && cmake --build build -j
(cd build && ctest --output-on-failure)

# Requirements traceability (after tests)
./build/tests/unit/polaris_unit_tests --gtest_output=json:build/gtest.json
python tools/dev/collect_gtest_trace.py build/gtest.json

# Docs site (warnings are errors; same gate as CI)
python3 -m venv docs-venv && . docs-venv/bin/activate
pip install -r docs/requirements.txt
bash tools/dev/build_docs.sh        # -> docs/_build/html/index.html

# Lint
pre-commit run --all-files
```

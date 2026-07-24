# Polaris

**End-to-end spacecraft GNC: flight software, 6DOF simulation, and analysis — from scratch.**

**Docs site:** <https://atillasaadat.github.io/polaris/> — API reference,
requirements traceability (RVTM), and bibliography, published from `main` by CI.

Polaris is a ground-up Guidance, Navigation & Control software suite for a
spacecraft, built to flight-grade, NASA-aligned standards. It spans the full
loop — a high-fidelity truth simulation (the "plant") flying against real flight
software (the "controller") in software-in-the-loop, plus the ground-side tools
to size, validate, and verify the design.

## Architecture (four layers)

| Layer | Path | Language | Role |
|---|---|---|---|
| **Flight software (FSW)** | `flight/` | F´ (F Prime) / C++ | The deliverable that "flies": sensing → estimation → guidance → control → actuation → FDIR. **No heap, no exceptions** after init. |
| **Truth / environment sim** | `sim/` | C++ | High-fidelity 6DOF plant (RK89, gravity/drag/SRP/eclipse/IGRF/3-body), sensor & actuator truth models, fault injection. |
| **Shared library** | `lib/` | C++ | Math, frames, time, canonical state, constants, environment, ephemeris — used by both FSW and sim. |
| **Analysis** | `analysis/` | Python (via `bindings/` pybind11) | Momentum/sizing, detumble MC, contacts, link budget, post-processing — exercising the *same* C++ that flies. |

Plus `mc/` (Monte Carlo), `config/` (spacecraft/scenario/hardware config), `tests/`
(unit → component → integration → golden), and `tools/` (config compiler, GMAT
golden-data harness).

## Source of truth

- **Authoritative spec:** [`docs/design/Polaris_Design_Document.md`](docs/design/Polaris_Design_Document.md) — read the relevant section before non-trivial work.
- **Operational rules for agents/devs:** [`CLAUDE.md`](CLAUDE.md) (root) and per-layer `CLAUDE.md` in `flight/`, `sim/`, `analysis/`.
- **Requirements:** [`docs/requirements/`](docs/requirements/) — `REQ-<SUBSYS>-NNN`, with bidirectional traceability to tests (Sphinx-Needs).

## Conventions (the non-negotiables)

- **Time:** onboard master clock is **TAI** (int64 ns); GNSS GPS time + ECEF converted on ingest.
- **Attitude:** quaternions are **JPL convention, scalar-first** `[q0,q1,q2,q3]`, `q0 ≥ 0`.
- **Units:** **SI everywhere** inside FSW and `lib/`; display units only at the ground/analysis boundary.
- **Frames:** every vector/quaternion is frame-tagged; mixing frames across a boundary is a compile error.
- **Provenance:** every algorithm/model cites a textbook/paper source in `docs/refs.bib`.

## Getting started — build, run & test the baseline

The flight software is an **F´ (F Prime) v4.2.2** project (vendored as the `fprime/`
git submodule). The current baseline is a buildable, runnable **barebones deployment**
(`flight/PolarisFsw`) — F´ core command/telemetry/event/file/data-product services with
three rate groups — that the GNC components land into in later phases.

### Prerequisites

- Linux or macOS (WSL2 works), `git`, and a C++17 compiler (`g++` ≥ 11 or `clang` ≥ 14).
- **Python 3.12** (F´ toolchain requirement).
- `cmake` ≥ 3.24 and `ninja` — system packages, or installed into the venv via `pip` (below).

### 1. Clone (with submodules)

```bash
git clone --recurse-submodules git@github.com:atillasaadat/polaris.git
cd polaris
# already cloned without --recurse-submodules?  initialise the framework:
git submodule update --init --recursive
```

### 2. Set up the toolchain (one-time)

Dependencies are managed with [uv](https://docs.astral.sh/uv/). One command
creates the environment (`.venv`), provisions Python 3.12, and installs the
pinned F´ toolchain — `cmake` and `ninja` arrive as wheels, so nothing else is
needed on `PATH`:

```bash
uv sync                                # F´ toolchain + dev tools, from uv.lock
```

`uv run <cmd>` runs a tool inside that environment; add `--group docs` to pull
in the documentation toolchain. Only `doxygen` (for the C++ API) is a system
package: `sudo apt-get install -y doxygen`.

### 3. Build

```bash
uv run fprime-util generate            # configure the build cache (one-time per cache)
uv run fprime-util build               # compile F´ core + the PolarisFsw deployment
```

The binary lands at `build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw`.

### 4. Run (flight software + F´ GDS)

```bash
cd flight/PolarisFsw
uv run fprime-gds                      # launches the GDS web UI (http://127.0.0.1:5000) and the app
```

The GDS lets you send commands and watch live telemetry/events. To run the pieces
separately — GDS in one shell, the binary in another:

```bash
cd flight/PolarisFsw && uv run fprime-gds --no-app    # ground system only (TCP server on :50000)
# in a second shell, from the repo root:
./build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw -a 127.0.0.1 -p 50000
```

### 5. Test

```bash
# Build the test suites (unit runs under ASan/UBSan)
uv run cmake --build build-fprime-automatic-native-ut \
    --target polaris_unit_tests polaris_integration_tests polaris_golden_tests -j4

./build-fprime-automatic-native-ut/bin/Linux/polaris_unit_tests          # 378 tests
./build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests   # full-stack sim
./build-fprime-automatic-native-ut/bin/Linux/polaris_golden_tests        # GMAT cross-validation

uv run pytest tests/tools              # config compiler, GMAT harness, space weather
```

### 6. Docs (optional)

```bash
sudo apt-get install -y doxygen        # one-time; C++ API extraction
uv run --group docs bash tools/dev/build_docs.sh   # -> docs/_build/html/index.html
```

> The shared `lib/` (math, quaternions, typed vectors, constants) is plain CMake linked
> into both the flight deployment and the unit tests. `lib/` unit tests build on demand and
> are **not** part of `fprime-util build`; use `-DPOLARIS_BUILD_TESTS=OFF` for a pure flight build.

## Status

**Phase 2 — Sensor & actuator models** (complete except CMGs/thrusters), on top of a
finished Phase 0 (foundations, F´ baseline, config pipeline, requirements ICD) and
Phase 1 (6DOF + RK8(9) truth dynamics with the full environment suite, GMAT
cross-validated). The truth sim flies a config-defined vehicle: IMU, star tracker,
sun sensor, magnetometer, and GNSS truth models with full error stacks, shared
occlusion, and scriptable fault injection; reaction-wheel (W-matrix assembly) and
magnetorquer actuators. Next: **Phase 3** — the F´ FSW skeleton and the two-process
SITL loop that binds them together. Detail: [`PROGRESS.md`](PROGRESS.md); phase plan:
design doc §24.

## License

Polaris is **dual-licensed**: free for **noncommercial use** (research, education,
university smallsats) under the [PolyForm Noncommercial License 1.0.0](LICENSE) with
attribution; **commercial use requires a paid license** (see [`LICENSING.md`](LICENSING.md)).
Third-party components remain under their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

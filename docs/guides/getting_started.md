# Getting Started

Polaris is a from-scratch spacecraft GNC suite: flight software in F´ (F Prime) /
C++, a high-fidelity 6DOF truth simulation, a shared C++ library, and Python
tooling — verified against GMAT golden data.

## Prerequisites

- Linux or macOS (WSL2 works), `git`, a C++17 compiler (`g++` ≥ 11 / `clang` ≥ 14).
- **Python 3.12** (the F´ toolchain requirement).
- Everything else — `cmake`, `ninja`, the F´ toolchain — arrives as pinned
  wheels through [uv](https://docs.astral.sh/uv/). Only `doxygen` (for the C++
  API docs) is a system package.

## Clone and build

```bash
git clone --recurse-submodules git@github.com:atillasaadat/polaris.git
cd polaris
uv sync                                   # F´ toolchain + dev tools, from uv.lock

uv run fprime-util generate               # configure the build cache (once)
uv run fprime-util build                  # F´ core + the PolarisFsw deployment
```

## Run the test suites

The C++ suites build into the F´ native-UT cache; the unit suite runs under
AddressSanitizer/UBSan.

```bash
uv run cmake --build build-fprime-automatic-native-ut \
    --target polaris_unit_tests polaris_integration_tests polaris_golden_tests -j4
./build-fprime-automatic-native-ut/bin/Linux/polaris_unit_tests
./build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests
./build-fprime-automatic-native-ut/bin/Linux/polaris_golden_tests    # GMAT cross-validation

uv run pytest tests/tools                 # config compiler, GMAT harness, space weather
```

## What to read next

- {doc}`frames_and_time` — the conventions every quantity obeys (frames, the JPL
  quaternion, the TAI clock). Read this before touching any GNC math.
- {doc}`configuring_a_vehicle` — how a YAML config becomes a flown vehicle.
- {doc}`closed_loop` — the §2.4 execution model that ties sensors, the plant, and
  the flight software together.
- {doc}`/api/index` — the symbol-level reference.

The authoritative design baseline is `docs/design/Polaris_Design_Document.md`;
read the relevant section before non-trivial work.

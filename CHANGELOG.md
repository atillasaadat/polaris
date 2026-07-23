# Changelog

All notable changes to Polaris are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Tagged
releases begin once a buildable baseline exists (Phase 0).

## [Unreleased]

### Added — Phase 2, Pushes 19–28 (sensor & actuator truth models)
- Sensor truth models with full error stacks, per-source seeded RNG streams
  (§3.5), and scriptable fault injection: **magnetometer** (P19, #21), **IMU**
  (STIM300/377H; P20, #22), **star tracker** (Sodern AURIGA; acquisition/tracking
  state machine, anisotropic spatial/temporal errors; P23, #25), **sun sensor**
  (analogue diode counts + GomSpace NanoSense FSS digital vector,
  incidence-dependent accuracy, albedo; P24, #26), **GNSS** (NovAtel OEM7600
  PVT fix in ECEF/GPS time, cold-start/reacquisition; P25, #27).
- Shared line-of-sight **occlusion model** (Earth+atmosphere limb / Sun / Moon,
  fractional FOV coverage; P23, #25).
- Actuator truth models: **reaction wheel** (RW-0.4 rundown friction, torque +
  wheel-local speed command modes, static/dynamic imbalance; P21 #23, P26 #28),
  **magnetorquer** (dipole limit, hysteresis; P21, #23); **RW assembly** as a
  3×N W matrix with `spin_axis` config (P26, #28).
- **Config-driven hardware** (§19.4): in-code catalogs deleted; every unit built
  from `config/hardware/` datasheet-pinned YAML through the compiler (P22, #24).
- Scenario controls: master seed, **sensor-noise switches** (global + per-unit;
  P27, #29), **GNSS geographic jamming** from a KML region map, and a scheduled
  outage/spoof/clock-jump **fault-event timeline** (P25, #27).
- Holistic audit (P28): `com_m` carried through the config pipeline,
  `gravity_order` reachable from YAML, drag/SRP default alignment, C++
  validation parity, §-reference sweep (`§182`→`§3.5`), refs.bib completion,
  documentation standard (§21.3) + per-folder READMEs.

### Added — Phase 1, Pushes 6–18 (truth sim core)
- **6DOF rigid-body dynamics + RK8(9)** integrator with conservation and
  Kepler-closure tests (P6, #7).
- Gravity: spherical-harmonic + gravity-gradient torque (P7, #8), fully
  normalized to 200×200 (P8, #9), **EGM2008** `.gfc` loader in the ECEF frame
  (P10, #11) with a committed degree-200 fixture (#12).
- **ECI↔ECEF** reduction (IAU 2006/2000A via ERFA) + IERS EOP tables (P9, #10).
- Third-body point-mass gravity (Sun/Moon) (P11, #12); **DE440 Chebyshev
  ephemeris** layer (P14, #16).
- **SRP + conical eclipse** (P12, #14); **atmospheric drag** with exponential
  atmosphere (P13, #15), NRLMSIS 2.1 option + **CelesTrak space weather** (P18, #20).
- **IGRF-14** geomagnetic field + residual-dipole torque (P15, #17).
- Truth-sim executable (P16, #18); **GMAT force-model golden cross-validation**
  (P17, #19).

### Added — Phase 0, Pushes 2b–5 (foundations continued)
- `lib/time` (TAI/GPS/TT/UTC + leap seconds; P2b, #3), TDB + onboard Chebyshev
  ephemeris (P2d, #4), geometric frames (LVLH/RIC) + canonical
  `EstimatedState`/`TruthState` (P2c, #4).
- **F´ v4.2.2** vendored + buildable barebones deployment (P2b, #3).
- Config schema + hardware-model library + **config compiler** with provenance
  hashing (P3, #5).
- **GMAT golden-data harness** + first fixture (P5, #6).

### Infrastructure
- CI: lint (pre-commit), docs `-W` gate, F´ build + unit tests; branch
  protection on `main` with required checks + squash auto-merge; GitHub Pages
  docs publishing (<https://atillasaadat.github.io/polaris/>).

### Added — Phase 0, Push 1 (Foundations: skeleton, license, requirements, docs)
- Repository directory skeleton per design doc §22.3 (`lib/ flight/ sim/ bindings/
  analysis/ config/ tests/ tools/ mc/ docs/ .github/`).
- Dual-license model: **PolyForm Noncommercial 1.0.0** for the public
  (`LICENSE`) plus a separately-sold commercial license (`LICENSING.md`,
  `NOTICE`); third-party attributions in `THIRD_PARTY_NOTICES.md`.
- Tooling baseline: `.gitignore`, `.editorconfig`, `.clang-format` (clang-format 18),
  `.clang-tidy` (JPL Power-of-Ten checks), `.pre-commit-config.yaml`.
- **Requirements ICD** (`docs/requirements/`) authored with **Sphinx-Needs**:
  ID scheme `REQ-<SUBSYS>-NNN`, levels L0/L1/L2, T/A/I/D verification methods,
  margin fields, and an auto-generated RVTM. Seeded L0/L1 and firm
  conventions/CDH/sim/V&V requirements.
- **Docs site skeleton** (Sphinx + numpydoc + PyData theme + Breathe/Doxygen +
  sphinxcontrib-bibtex + sphinx-needs) with `refs.bib`, glossary, and a CI
  doc-build gate (`sphinx-build -W`).
- CI skeleton (`.github/workflows/`): runnable docs/lint gate now; staged §23.2
  pipeline stubbed for later phases.

### Added — Phase 0, Push 2a (lib/ math foundations)
- CMake build system with pinned **Eigen 3.4.0** and **GoogleTest 1.15.2**
  (FetchContent, marked SYSTEM), C++17, strict `-Werror` warning set, CTest.
- `lib/constants` — shared physical-constants registry (WGS84, time offsets, *c*).
- `lib/math` — frame tags + `Vec3<Frame>` typed vectors (compile-time frame
  safety) and the single JPL scalar-first `Quaternion` library + tagged
  `Quat<To,From>` boundary wrapper.
- 18 GoogleTest unit tests (all `REQ-###`-traced, 100% pass) validating the
  quaternion convention by self-consistency.
- Added `trawny2005`, `shepperd1978`, `wgs84` to `docs/refs.bib`.

### Decisions
- **License:** chose PolyForm Noncommercial + commercial dual-license over the
  design doc's original Apache-2.0 suggestion, to allow free academic/research use
  with attribution while requiring commercial licensees to pay. Design doc §23.4
  updated to match.

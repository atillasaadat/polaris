# Changelog

All notable changes to Polaris are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Tagged
releases begin once a buildable baseline exists (Phase 0).

## [Unreleased]

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

# Polaris

**End-to-end spacecraft GNC: flight software, 6DOF simulation, and analysis — from scratch.**

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

## Status

**Phase 0 — Foundations** (in progress): repo skeleton, conventions, requirements
ICD, docs site, F´/GMAT setup. See design doc §24 for the full phase plan. Not yet
buildable end-to-end; CI is being stood up incrementally.

## License

Polaris is **dual-licensed**: free for **noncommercial use** (research, education,
university smallsats) under the [PolyForm Noncommercial License 1.0.0](LICENSE) with
attribution; **commercial use requires a paid license** (see [`LICENSING.md`](LICENSING.md)).
Third-party components remain under their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

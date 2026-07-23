# `lib/` — Shared C++

Flight/sim-shared C++ used by both the FSW (`flight/`) and the truth sim (`sim/`),
and exposed to Python via `bindings/`. **Flight paths of `lib/` obey the flight
memory/exception rules** (no heap after init, fixed-size Eigen, no exceptions) —
see root `CLAUDE.md` Golden Rule 6.

Subdirectories (design doc §22.3):
`math/` (Eigen, quaternions, typed vectors) · `frames/` (transforms + EOP) ·
`time/` (TAI/UTC/GPS) · `state/` (`EstimatedState`/`TruthState`) ·
`constants/` (registry incl. WGS84) · `environment/` (gravity, drag, SRP, IGRF, 3-body) ·
`ephemeris/` (Chebyshev onboard evaluator) · `random/` (seeded per-source RNG streams, §3.5) ·
`models/`.

Each subdirectory carries its own `README.md` with a contents table.

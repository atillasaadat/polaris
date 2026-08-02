# `lib/` — Shared C++

Flight/sim-shared C++ used by both the FSW (`flight/`) and the truth sim (`sim/`),
and exposed to Python via `bindings/`. **Flight paths of `lib/` obey the flight
memory/exception rules** (no heap after init, fixed-size Eigen, no exceptions) —
see root `CLAUDE.md` Golden Rule 6.

Subdirectories (design doc §22.3):
`math/` (Eigen, quaternions, typed vectors) · `frames/` (transforms + EOP) ·
`time/` (TAI/UTC/GPS) · `state/` (`EstimatedState`/`TruthState`) ·
`constants/` (registry incl. WGS84) · `environment/` (IGRF geomagnetic field + its
IAGA coefficient loader — gravity, drag, SRP and third-body are truth-side only
and live in `sim/world/`) · `ephemeris/` (Chebyshev onboard evaluator) ·
`random/` (seeded per-source RNG streams, §3.5) ·
`gnc/` (estimation/guidance/control algorithms, §8) · `onboard/` (table store, §11.3) ·
`sitl/` (plant↔FSW wire format, §2.2).

Each subdirectory carries its own `README.md` with a contents table.

## Angle computations: `atan2` forms, never a bare `acos` of a dot product

Repo-wide convention, flight and sim alike. The angle between two vectors is
`atan2(a.cross(b).norm(), a.dot(b))`, not `acos(clamp(a.dot(b), -1, 1))`; the
angle of a quaternion is `2*atan2(dq.vec().norm(), abs(dq.scalar()))`, not
`2*acos(dq.scalar())`; a colatitude is `atan2(hypot(x, y), z)`. The reason is
conditioning, not taste: cosine is stationary where the angle is small (and
where it approaches π), so `acos` of a cosine that is itself accurate to
machine epsilon still returns an angle wrong by ~1e-8 rad — precisely the
near-parallel geometry that attitude errors, limb clearances and sensor
boresight angles live in. The sine term carries the information the cosine
lost, and `atan2` uses both. `tests/unit/angle_conditioning_test.cpp` pins the
difference; a clamp is not a fix, it hides the NaN and keeps the error.

Where the input genuinely is a scalar ratio with no vector pair in scope — the
law-of-cosines terms in `math/fov_overlap.hpp` and `sim/world/eclipse.cpp` —
write `atan2(sqrt(max(0, (1-t)*(1+t))), t)` anyway for uniformity and to make
the function total, but note in a comment that this buys robustness only: the
conditioning was already lost in forming `t`.

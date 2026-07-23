# `lib/frames/` — ECI↔ECEF Reduction & Earth Orientation

The single time-dependent frame transform in the repo (design doc §3.1,
REQ-CONV-001/002): "no ad-hoc rotations."

| File | Role |
|---|---|
| `eci_ecef.{hpp,cpp}` | IAU 2006/2000A reduction via ERFA's `eraC2t06a` (the SOFA reference implementation); state transforms carry the `ω⊕×r` transport term [iers2010][wallace2006][vallado2013]. The house documentation exemplar (§21.3). |
| `eop.hpp` | Fixed-capacity IERS EOP table (ΔUT1, polar motion) with bounded interpolation; no extrapolation, no throw |

Every GNSS fix crosses ECEF→ECI here before the inertial filter runs; every
geopotential/ground-track computation crosses the other way.

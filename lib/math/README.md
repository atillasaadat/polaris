# `lib/math/` — Frames, Vectors, Quaternions

The convention bedrock (design doc §3.1): mixing frames is a **compile error**,
and there is exactly one quaternion convention in the repo.

| File | Role |
|---|---|
| `frames.hpp` | Frame tag types (`ECI`, `ECEF`, `Body`, `LVLH`, `RIC`, …) |
| `typed_vector.hpp` | `Vec3<Frame>` — frame-tagged 3-vector; raw Eigen only at kernel boundaries via `.eigen()` |
| `quaternion.{hpp,cpp}` | **JPL scalar-first** `[q0,q1,q2,q3]`, `q0≥0`, passive rotations, Shepperd extraction [trawny2005][shepperd1978]; tagged `Quat<To,From>` boundary wrapper |
| `frame_geometry.{hpp,cpp}` | RIC and LVLH frames as pure functions of the ECI orbit state (no EOP/time) [vallado2013] |

Flight-safe: no heap, no exceptions; degenerate inputs return `false` and leave
outputs untouched.

# `lib/time/` — Time Scales & the TAI Master Clock

Strongly-typed time (design doc §3.2): mixing scales is a compile error, the
onboard master clock is **TAI** as int64 nanoseconds.

| File | Role |
|---|---|
| `duration.hpp` | Signed int64-ns interval, `constexpr` arithmetic |
| `timescales.hpp` | `Tai`/`Gps`/`Tt` instants + exact constant-offset conversions (`TAI−GPS=19 s`, `TT−TAI=32.184 s`) |
| `tdb.{hpp,cpp}` | TT↔TDB periodic term (two-harmonic series, ~30 µs) [vallado2013] |
| `leap_seconds.{hpp,cpp}` | Fixed-capacity IERS ΔAT table (historical + frozen/settable for reproducible MC) |
| `utc.{hpp,cpp}` | UTC derivation incl. the `hh:mm:60` leap second — ground-facing only |
| `civil.hpp` | Branch-free proleptic-Gregorian date algorithms [hinnant2016] |

Flight-safe: no heap, no exceptions, `constexpr` where possible.

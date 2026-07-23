# `lib/ephemeris/` — Onboard Chebyshev Ephemeris

The onboard Sun/Moon position source (design doc §11.3, REQ-CDH-002): uploaded
Chebyshev coefficient segments evaluated at TDB epochs — SPICE stays
ground-side only (§18.6) and is never linked into flight.

| File | Role |
|---|---|
| `chebyshev.{hpp,cpp}` | One fixed-capacity coefficient segment (degree ≤ 15), position + analytic-derivative velocity in ECI metres [newhall1989] |
| `ephemeris_table.hpp` | A body's consecutive segments; covering-interval lookup, no extrapolation across gaps |

Flight-safe: fixed capacity, no heap; out-of-coverage returns `false` with
outputs untouched. The DE440 ground fit lives in `sim/world/ephemeris_file` +
`tools/`.

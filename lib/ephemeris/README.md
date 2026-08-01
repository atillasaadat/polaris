# `lib/ephemeris/` — Onboard Chebyshev Ephemeris

The onboard Sun/Moon position source (design doc §11.3, REQ-CDH-002): uploaded
Chebyshev coefficient segments evaluated at TDB epochs — SPICE stays
ground-side only (§18.6) and is never linked into flight.

| File | Role |
|---|---|
| `chebyshev.{hpp,cpp}` | One fixed-capacity coefficient segment (degree ≤ 15), position + analytic-derivative velocity in ECI metres [newhall1989] |
| `ephemeris_table.hpp` | A body's consecutive segments; covering-interval lookup, no extrapolation across gaps |
| `analytic_sun.{hpp,cpp}` | Table-independent low-precision geocentric Sun position, Vallado §5.1 Algorithm 29 (~0.01° in-frame; mean-of-date≈J2000) [vallado2013] |
| `analytic_moon.{hpp,cpp}` | Table-independent low-precision geocentric Moon position, Vallado §5.3.2 (~0.3° in-frame) [vallado2013] |

Flight-safe: fixed capacity, no heap; out-of-coverage returns `false` with
outputs untouched. The DE440 ground fit lives in `sim/world/ephemeris_file` +
`tools/`.

The two `analytic_*` models are the **coarse fallback** the onboard table store
(`lib/onboard`) serves when the uploaded Chebyshev fit is missing or its
coverage has lapsed — pure functions of the clock, no data dependency, so the
Safe-mode coarse sun-pointing floor (§8.1, §10) stays table-independent. They
carry a mean-equator-of-date≈J2000 frame approximation (neglected precession,
≈0.4° at the mid-2020s epoch); see each header for the error budget.

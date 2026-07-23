# `lib/environment/` — Onboard-Shareable Environment Models

Environment models clean enough for both sides of the truth/onboard divide
(design doc §5.2/§8.1): the FSW needs its own IGRF for measured-vs-modelled
magnetometer checks and coarse attitude, so this lives in `lib/`, not `sim/`.

| File | Role |
|---|---|
| `igrf.{hpp,cpp}` | IGRF spherical-harmonic geomagnetic field (Schmidt semi-normalised Legendre recursion) [alken2021][langel1987][winch2005]; coefficients are loaded from the committed IAGA file by `sim/world/igrf_file` |

Flight-safe: fixed-size storage, no heap, refusal (`false`) over garbage.

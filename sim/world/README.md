# `sim/world/` — Environment Models & Reference-Data Loaders

The forces, torques, and fields of the orbital environment (design doc §5,
REQ-SIM-002), plus the loaders for the **committed-verbatim** reference data
they consume (§3.7: upstream files are committed byte-for-byte in their original
format and parsed as-is — never pre-digested).

| File | Model / loader | Reference data |
|---|---|---|
| `gravity_field.{hpp,cpp}` | Fully-normalized spherical-harmonic gravity (Holmes-Featherstone recursion, stable to 200×200) + gravity-gradient torque | — |
| `egm2008.{hpp,cpp}` | ICGEM `.gfc` coefficient loader (EGM2008) | `tests/golden/EGM2008_to200.gfc` |
| `third_body.{hpp,cpp}` | Point-mass differential gravity: Sun, Moon, and config-selectable planets (DE440 barycenters) | via ephemeris |
| `ephemeris_file.{hpp,cpp}` | DE440-fitted Chebyshev segment loader | `tests/golden/de440_bodies.cheb` |
| `body_position.{hpp,cpp?}` | Injected body-position resolver plumbing (one ephemeris, many consumers) | — |
| `drag.{hpp,cpp}` | Cannonball atmospheric drag | atmosphere below |
| `atmosphere.{hpp,cpp}` | Piecewise-exponential density (Vallado Table 8-4) | — |
| `nrlmsis.{hpp,cpp}` + `msis_shim.F90` | NRLMSIS 2.1 (optional build; non-commercial license — see `LICENSING.md`) | fetched upstream tree |
| `space_weather_file.{hpp,cpp}` | CelesTrak SW-All loader (F10.7, ap) driving NRLMSIS | `tests/golden/SW-All.csv` |
| `srp.{hpp,cpp}` | Cannonball solar radiation pressure | via ephemeris |
| `eclipse.{hpp,cpp}` | Conical umbra/penumbra shadow factor | — |
| `magnetic_field.{hpp,cpp}` | IGRF field at the vehicle + residual-dipole torque | via `igrf_file` |
| `igrf_file.{hpp,cpp}` | IAGA IGRF-14 coefficient loader | `tests/golden/igrf14coeffs.txt` |
| `eop_file.{hpp,cpp}` | IERS `finals.all` (Bulletin A) loader, windowed | `tests/golden/finals.all.iau2000.txt` |

## Conventions

- **Validation is golden-first:** every model here is cross-checked against
  GMAT fixtures (`tests/golden/`) or the upstream model's own reference output,
  within documented tolerance bands (REQ-VV-002).
- **Injected resolvers, not global state:** eclipse, body positions, and the
  field are passed as functions into consumers (SRP, sun sensor, MTQ torque),
  so there is exactly one implementation of each physical fact in a run.
- Truth fidelity is deliberately **higher than the onboard models** the FSW
  will fly (§2.3) — don't hand the FSW a truth-grade quantity.

New environment model? Follow an existing pair (`srp` is the smallest complete
exemplar): model file + loader if it consumes external data (committed verbatim,
with the fetch tool recording the source URL), golden or analytic validation
test, and an `EnvironmentConfig` toggle wired through the schema → loader →
`sim_runner`.

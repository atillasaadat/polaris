# `sim/sensors/` — Sensor Truth Models

Truth-in → measurement-out models of every sensor the vehicle can fly (design
doc §6). Each takes the true state (already rotated into its frame by the
caller), applies its full datasheet error stack, and returns what the real unit
would report — including *when it can't* (occlusion, eclipse, slew limits,
faults), because availability drives the estimator design as much as noise does.

| File | Model | Output contract |
|---|---|---|
| `sensor_error.hpp` | Shared §6.1 error stack (scale/misalignment → bias → noise → quantize → saturate) | used by IMU + magnetometer |
| `occlusion.{hpp,cpp}` | Shared line-of-sight model: Earth(+atmosphere limb)/Sun/Moon keep-out verdict + fractional FOV coverage | used by every optical sensor |
| `imu.{hpp,cpp}` | Gyro + accelerometer triads: ARW/VRW, Gauss-Markov bias drift, turn-on bias, scale/misalignment, g-sensitivity | Δθ/Δv + rate/specific force |
| `magnetometer.{hpp,cpp}` | Hard-iron, soft-iron/misalignment, noise, quantization | body-frame field [T] |
| `star_tracker.{hpp,cpp}` | Anisotropic bias/LF/HF-spatial/temporal/thermo-elastic errors; acquisition↔tracking state machine with separate rate *and* accel envelopes | attitude quaternion + validity |
| `sun_sensor.{hpp,cpp}` | **Two contracts:** analogue → per-diode cosine-law counts; digital → sun vector with incidence-dependent accuracy; albedo as a **directed pull toward the sunlit Earth** (not noise — it is correctable onboard, §8.1) with a per-unit dispersion about it | counts *or* unit vector |
| `gnss.{hpp,cpp}` | PVT fix: H/V-split position σ, velocity σ, clock bias; sample-rate gating, cold-start/reacquisition; outage/spoof/clock-jump + geographic jamming | ECEF position/velocity @ **GPS time** |
| `gnss_jamming.{hpp,cpp}` | KML polygon regions → "is the sub-satellite point jammed" | region name / none |

Implements **REQ-SIM-003** (truth models with full error stacks + shared
occlusion) and **REQ-SIM-005** (scriptable fault injection).

## How to add a new sensor

The contract, end to end. Use `magnetometer` as the minimal exemplar and
`star_tracker` as the full-featured one.

1. **Model the interface the unit actually presents** (§6.2). An analogue part
   hands over raw signals (counts, currents) — the FSW reconstructs; a digital
   part (star tracker, GomSpace FSS, GNSS) runs proprietary processing inside,
   so you reproduce its *specified output accuracy*, never an invented internal
   signal chain.
2. **Write the `Spec` + `fromParams`.** A plain struct of SI values, and
   `static Spec fromParams(const std::map<std::string, double>&)` converting
   datasheet-native keys (`*_deg`, `*_arcsec_3sigma`, `*_mrad`, `*_rpm`, …) to
   SI in exactly one place. Missing key ⇒ 0 ⇒ that term disabled. Vendors quote
   3σ — divide here, not in consumers.
3. **Constructor** takes `(spec, mounting_dcm, master_seed, stream_id,
   noise_enabled = true)`. Realise fixed per-unit errors (turn-on bias,
   misalignment draws) **at construction, in a fixed order**, from
   `random::streamRng(master_seed, stream_id)`. `noise_enabled=false` must
   build the *ideal* unit (measurement = truth) — geometry still applies.
4. **`sample(epoch, …)` draws a fixed number of RNG samples per call**,
   regardless of geometry, mode, or faults — a conditional draw desyncs the
   stream and silently breaks `{config, seed}` reproducibility (§3.5). Draw
   into locals first if a branch decides whether to *use* them.
5. **Any line of sight goes through `evaluateLineOfSight`** — never private
   geometry. Two sensors disagreeing about where the Earth is produces an
   estimator that works in sim and fails in flight.
6. **Return a measurement struct** carrying: the measurement, the realised σ it
   was drawn with (estimators need actual noise, not headlines), `valid` /
   availability flags (invalid ≠ zero — a zero reading is a *measurement*),
   staleness (`fresh`) if the part has a native rate, and the TAI time tag.
7. **Fault hooks** (§9): at minimum a dropout and a bias-jump/offset, plus
   whatever failure modes the FDIR spec (§9.2) names for the class. `clearFaults()`.
8. **Wire it**: add the `kind` to `HardwareKind` in `tools/configc/schema.py`;
   add a branch in `sim/scenario/vehicle.cpp` (reject configs that would build a
   silently-perfect unit); add the source to `sim/CMakeLists.txt`.
9. **Catalog entry**: `config/hardware/<kind>/<part>.yaml` per the §21.3 YAML
   standard — datasheet `Source:` URL, per-key unit comments, explicit
   "modelling choice" callouts. Pin the datasheet numbers in
   `tests/tools/test_config_compiler.py`.
10. **Tests** (`tests/unit/sim_sensors_<kind>_test.cpp`): the spec conversion
    (SI + 3σ→1σ), the defining physics (Monte-Carlo the σ if stochastic),
    availability/geometry edges, bit-reproducibility from `{seed, stream}`,
    the ideal (`noise_enabled=false`) contract, and every fault hook.

## Field conventions

- Frames: measurement vectors are **body-frame** (`math::Vec3<frames::Body>`)
  except GNSS (ECEF, by the real interface). Mounting DCM is **unit→body**.
- Time tags are **TAI** (`time::Tai`) except GNSS (`time::Gps`, by the real
  interface — the FSW converts on ingest, REQ-CONV-001).
- All internal quantities SI; conversions live in `fromParams` only.

# `config/hardware/` — Hardware Model Library

The **only** place hardware performance numbers live (design doc §19.2/§19.4).
One YAML file per part, keyed by a unique `model_id`; a spacecraft config
references parts by that string, and the config compiler inlines the params into
the emitted artifacts. Swapping the string swaps the flown unit — there is no
in-code catalog to fall back on, and C++ source may not contain hardware
constants (test fixtures excepted).

```
hardware/
├── imu/            STIM300 · STIM377H · IMU-GENERIC
├── star_tracker/   AURIGA (Sodern) · ST-16 (Sinclair) · ST-GENERIC
├── sun_sensor/     GS-NANOSENSE-FSS (GomSpace) · CSS-GENERIC · FSS-GENERIC
├── magnetometer/   MAG-GENERIC
├── gnss/           NOVATEL-OEM7600 · GNSS-GENERIC
├── reaction_wheel/ RW-0.4 (Rocket Lab) · RW-X (generic)
└── magnetorquer/   NSS Taurus · MTQ800 (AAC Clyde) · MTQ-GENERIC
```

## How to add a catalog entry

1. **File + identity.** `config/hardware/<kind>/<vendor_part>.yaml` with
   `model_id` (unique across the whole library), `kind` (must match the
   directory and a `HardwareKind` in `tools/configc/schema.py`), `description`,
   and `params` (flat map, numbers only).
2. **Keys are what the C++ `fromParams` reads** — see the key list in
   `sim/sensors/<kind>.cpp` / `sim/actuators/<kind>.cpp`. A misspelled key is
   *silently zero* (= that error term disabled), which is why step 4 exists.
   Keys carry their unit as a suffix: `_deg`, `_deg_s`, `_arcsec_3sigma`,
   `_mrad`, `_ppm`, `_ut`, `_nt`, `_rpm`, `_nms`, `_kg_m2`, `_ms`, `_hz`.
   Vendors quote 3σ; the key says so and the C++ divides — never pre-convert.
3. **Comment to the §21.3 standard** (exemplars:
   `star_tracker/sodern_auriga.yaml`, `sun_sensor/gomspace_nanosense_fss.yaml`):
   - Header block: vendor, **`Source:` datasheet URL + revision/date**, and
     which C++ `fromParams` consumes it.
   - `# --- Section (datasheet ref) ---` banners grouping the params.
   - A unit/provenance comment on **every** key.
   - Explicit **"NOT in datasheet / modelling choice"** callouts for anything
     that isn't the vendor's number (correlation times, defaults, estimates) —
     the difference between a datasheet value and a guess must be visible.
   - Generic templates omit `Source:` (nothing to cite) but keep the rest.
4. **Pin the datasheet numbers** in `tests/tools/test_config_compiler.py`: a
   key-completeness check against the C++ spec (so a silent zero fails loudly)
   and exact-value asserts for the figures an analysis would close against.
5. **Reference it** from a spacecraft config (`config/spacecraft/*.yaml`) under
   `sensors:`/`actuators:` with an instance `name` (unique on the vehicle — it
   keys the unit's RNG stream) and optionally `mounting_dcm_row_major`,
   `spin_axis` (wheels), or a per-unit `noise_enabled` override.

**Adding a new *kind* of hardware** (not just a new part) additionally needs a
truth model — see the walkthroughs in
[`sim/sensors/README.md`](../../sim/sensors/README.md) /
[`sim/actuators/README.md`](../../sim/actuators/README.md).

## Ground rules

- Never edit a datasheet-pinned number without updating the pin test **and** the
  `Source:` line — the pins exist so a silent edit fails CI, not to be chased.
- Imbalance/calibration values that are per-unit measurement data (not
  datasheet) default to zero and say so.
- Licensing: datasheet numbers are facts; do not paste datasheet *text*.

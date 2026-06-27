# `config/` — Spacecraft, scenario & hardware configuration

The **single source of truth** for spacecraft/scenario definition (design doc §19).
Compiled by the config compiler (`tools/configc/`) into F´ params, sim setup, and
analysis inputs — don't hand-edit derived params.

- `hardware/` — parameterized hardware model library, keyed by model ID (IMU, star
  tracker, reaction wheel, …).
- `spacecraft/` — vehicle definitions (mass/inertia, sensor & actuator suite, gains).
- `scenarios/` — scenario/epoch/environment + MC dispersions.

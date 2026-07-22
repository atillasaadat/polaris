# `config/` — Spacecraft, scenario & hardware configuration

The **single source of truth** for spacecraft/scenario definition (design doc §19).
Compiled by the config compiler (`tools/configc/`) into F´ params, sim setup, and
analysis inputs — don't hand-edit derived params.

- `hardware/` — parameterized hardware-model catalog, keyed by `model_id`. Organized
  into one subdirectory per device kind; the loader searches recursively, so a unit
  is found by its `model_id`, never its path. Add a COTS part by dropping a `.yaml`
  in the matching kind folder (copy `imu/generic.yaml` as a template).

  ```
  hardware/
  ├── imu/              # STIM300, STIM377H, generic template
  ├── star_tracker/     # ST-16
  └── reaction_wheel/   # RW-X
  ```
- `spacecraft/` — vehicle definitions (mass/inertia, sensor & actuator suite, gains).
- `scenarios/` — scenario/epoch/environment + MC dispersions.

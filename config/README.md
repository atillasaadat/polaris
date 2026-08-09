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
  ├── star_tracker/     # Sodern AURIGA, ST-16, generic template
  ├── sun_sensor/       # GomSpace NanoSense FSS, coarse + fine templates
  ├── magnetometer/     # generic three-axis
  ├── gnss/             # NovAtel OEM7600, generic
  ├── reaction_wheel/   # RW-0.4, RW-X generic (30 mN·m·s)
  ├── magnetorquer/     # NSS Taurus, MTQ800, generic
  └── payload_sensor/   # generic imager
  ```

  These entries are the **only** place a unit's parameters are written down: the
  sim builds every model from the compiled params and carries no in-code hardware
  catalog (design doc §19.4). A unit's `params` keys are the datasheet-native keys
  the matching C++ `fromParams` reads, so a new key means editing the model spec
  and the YAML — nowhere else. Entry format, comment standard, and the full
  add-a-part walkthrough: [`hardware/README.md`](hardware/README.md).
- `spacecraft/` — vehicle definitions (mass/inertia, sensor & actuator suite, gains,
  scenario/environment block). `leo_smallsat.yaml` is the ready-to-compile template.
- `scenarios/` — scenario side-data: GNSS-jamming region KMLs (`jamming/`), and
  future scenario/dispersion sets (§13).
- `claude/` — committed snapshot of the global `~/.claude` Claude Code development
  environment (rules, agents, commands, hooks, settings), with an `install.sh` to
  reproduce it on another machine. Not spacecraft configuration; see
  [`claude/README.md`](claude/README.md).

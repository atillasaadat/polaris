# `lib/onboard/` — onboard table store

Flight-safe holder for the three onboard reference tables the Phase-4 GNC stack
consumes (design doc §11.3, §22; REQ-CDH-002):

- **leap seconds** (ΔAT) — `time::LeapSecondTable` (the committed in-code
  `historical()` record; no uploaded leap file exists yet),
- **IERS EOP** — `frames::EopTable`, loaded from the verbatim `finals.all`
  product and windowed to the ephemeris span,
- **Chebyshev ephemeris** of the Sun and Moon — `ephemeris::EphemerisTable`,
  loaded from the committed `.cheb` fixture (planets are skipped).

`TableStore` (`tables.hpp`) loads, validates, and answers point queries
(`eopAt`, `bodyPositionEci`, `taiUtcOffset`, `coverageAt`). It only *wraps* the
`lib/` tables and evaluators — the parsing fills them through their public
`addEntry`/`addSegment` contracts.

Flight discipline (§3.6): no heap or exceptions in steady state; fixed-capacity
tables and a fixed line buffer; `<cstdio>` reads at load/reload only. Two
`TableSet` slots back an atomic active index, so a load stages into the inactive
slot and flips only on full success — a failed reload leaves the previous tables
in service and a query never sees a half-loaded table.

The `flight.OnboardTables` F´ component
(`flight/PolarisFsw/OnboardTables/`) wraps this store; unit tests
(`tests/unit/onboard_tables_test.cpp`) exercise it directly against the
committed fixtures.

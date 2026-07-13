# `tools/gmat/` — GMAT golden-data harness

GMAT (NASA GSFC) is the V&V reference tool (design doc §23.1, REQ-SYS-010 /
REQ-VV-002): it generates the versioned reference ("golden") datasets in
[`tests/golden/`](../../tests/golden/) that Polaris numerical functions are
checked against, within documented per-quantity tolerance bands.

**GMAT is not a CI dependency.** The golden fixtures are committed data; CI only
compares Polaris output against them. GMAT runs here *occasionally*, to
regenerate a fixture when the reference case changes — never in the test path.

## Regenerating a fixture (requires GMAT on PATH)

```bash
# 1. Emit the GMAT script (already committed under scripts/):
PYTHONPATH=tools uv run python -c "
from pathlib import Path
from gmat import write_time_scales_script
script = write_time_scales_script(
    ['06 Jan 1980 00:00:00.000', '01 Jan 2020 00:00:00.000'],
    'tools/gmat/scripts/time_scales_report.txt')
Path('tools/gmat/scripts/time_scales.script').write_text(script)
"

# 2. Run GMAT headless to produce the report:
GmatConsole tools/gmat/scripts/time_scales.script

# 3. Parse the report into the fixture and write it:
PYTHONPATH=tools uv run python -c "
import json
from pathlib import Path
from gmat import parse_report, build_time_scales_fixture
rows = parse_report(Path('tools/gmat/scripts/time_scales_report.txt').read_text())
# supply case_names + utc_fields matching the epochs, then:
# Path('tests/golden/time_scales.json').write_text(
#     json.dumps(build_time_scales_fixture(case_names, utc_fields, rows), indent=2))
"
```

A GMAT-regenerated fixture uses the harness default tolerance (`GMAT_MJD_TOL_S`,
~1 µs) — the floor of differencing float64 ModJulian columns. The committed
`time_scales.json` is instead seeded from exact published constants, so it can
declare (and lib/time meets) a far tighter 1 ns band.

`golden.py` is split so the parser and offset math are unit-tested without GMAT
(`tests/tools/test_gmat_golden.py`); only step 2 needs the GMAT binary.

## Provenance

The current `tests/golden/time_scales.json` was seeded from **published
constants** (IERS leap seconds; the IAU definitional TT−TAI = 32.184 s and
TAI−GPS = 19 s), which are independent of the Polaris code they verify and which
GMAT reproduces exactly. It is marked `gmat_regeneratable: true`; a GMAT run
replaces the seed values in place (they will agree to the fixture tolerance).

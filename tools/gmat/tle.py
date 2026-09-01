"""GMAT SGP4/TLE golden harness — the third independent SGP4 lineage (§23.1).

Polaris implements the Vallado formulation of SGP4 (AIAA 2006-6753) and is
accepted against that paper's own verification vectors. That proves Polaris
reproduces *the reference*. It does not, on its own, prove the whole chain a
flight user actually calls — parse a TLE, propagate, convert TEME to ECI — lands
where other operational tools put the satellite.

Two external tools answer that, and the point is that they share nothing with
Polaris or with each other:

===================  ===============================  ==================
tool                 SGP4 implementation              reported frame
===================  ===============================  ==================
Polaris              Vallado / AIAA 2006-6753         TEME -> ECI (GCRS)
FreeFlyer 7.10.1     US Space Force AstroStds v9.5    ICRF
GMAT R2026a          NAIF SPICE (``SPICESGP4``)       ICRF
===================  ===============================  ==================

Three lineages, three codebases, three frame chains. Agreement across them is
evidence about the algorithm; agreement with only one could still be a shared
convention.

**Why the report asks for two coordinate systems.** Each case reports the state
in *both* ``ICRF`` and ``EarthMJ2000Eq``. ICRF is what Polaris is compared
against — it is the GCRS-realisation matching this repository's ECI. MJ2000Eq is
FK5-based, and the difference between the two columns is precisely the ~23 mas
frame bias that ``lib/frames/teme_eci.cpp`` composes via ``eraBp00``. Carrying
both turns a design decision into a *measurement*: the fixture records GMAT's own
number for the bias (~0.69 m at LEO, almost entirely on Z), so
``sgp4_external_golden_test.cpp`` can assert that Polaris's bias term matches an
independent tool's rather than asserting only that the code does what it does.

Regenerate (needs GMAT — see ``tools/gmat/install_gmat.sh``)::

    GMAT_CONSOLE="$(bash tools/gmat/install_gmat.sh)" \
        uv run python -m gmat tle --write

CI never runs GMAT. The weekly ``golden`` lane drift-checks the committed
fixture (``tests/tools/test_gmat_drift.py``); every PR just compares against it.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = _REPO_ROOT / "tests" / "golden" / "gmat_sgp4.json"
VERIFICATION_TLE = _REPO_ROOT / "tests" / "golden" / "SGP4-VER.TLE"

#: GMAT reports UTCModJulian relative to its own reference epoch, not the
#: standard MJD one; only *differences* are used here, so the offset cancels.
SECONDS_PER_DAY = 86400.0

#: The same five regimes the FreeFlyer fixture uses, drawn from the same
#: committed AIAA verification set. Keeping the case list and sample schedule
#: identical is what makes the two fixtures directly comparable — and lets one
#: test difference the two external tools against each other.
TLE_CASES = (
    {
        "name": "leo_near_earth",
        "norad": "00005",
        "note": "LEO, e=0.186 — the reference set's first case",
        "sample_times_s": (0.0, 600.0, 1800.0, 3600.0, 5400.0),
    },
    {
        "name": "leo_sso",
        "norad": "28057",
        "note": "sun-synchronous LEO, near-circular",
        "sample_times_s": (0.0, 600.0, 1800.0, 3600.0, 5400.0),
    },
    {
        "name": "molniya_12h_resonant",
        "norad": "22674",
        "note": "Molniya, e=0.754 — exercises the 2:1 deep-space resonance",
        "sample_times_s": (0.0, 3600.0, 10800.0, 21600.0, 43200.0),
    },
    {
        "name": "geo_24h_synchronous",
        "norad": "26900",
        "note": "GEO — exercises the 1:1 deep-space resonance",
        "sample_times_s": (0.0, 3600.0, 21600.0, 43200.0, 86400.0),
    },
    {
        "name": "deep_space_non_resonant",
        "norad": "04632",
        "note": (
            "deep space, 19.96 h period, e=0.145 — SDP4 lunar-solar periodics "
            "with neither the 12 h nor the 24 h resonance active"
        ),
        "sample_times_s": (0.0, 3600.0, 10800.0, 21600.0, 43200.0),
    },
)

#: Cases the FreeFlyer fixture carries that GMAT will not run, recorded here so
#: the shorter GMAT case list is *data* rather than a silent gap.
#:
#: GMAT's ``SPICESGP4`` aborts on element set 11801 — the AIAA set's synthetic
#: decaying deep-space case (e=0.732, 151 km perigee altitude, B* = 0.143, three
#: orders above a typical value). The failure surfaces as SPICE returning
#: uninitialised memory to GMAT's epoch formatter ("Gregorian date '0H..' is not
#: valid"), not as a diagnosed rejection.
#:
#: The responsible field was **not** isolated, and this says so rather than
#: guessing: bisecting one field at a time — zeroing ndot, zeroing B*, zeroing
#: both, dropping eccentricity to 0.10, raising mean motion out of the
#: deep-space band, filling the blank international designator and ephemeris-type
#: columns, rewriting line 1 in canonical form with a valid checksum, and
#: changing the catalogue number — left it failing in every combination, while
#: known-good elements at the same 1980 epoch, and deep-space elements moved to
#: that epoch, both run fine. So it is neither the epoch, nor deep space, nor
#: drag, nor formatting on its own.
#:
#: This costs no coverage that matters. 11801 is checked against the AIAA
#: reference vectors in ``sgp4_verification_golden_test.cpp`` (which is the
#: authority for SGP4 itself) and cross-checked against FreeFlyer, so it keeps
#: two independent implementations; only the third declines it.
GMAT_UNSUPPORTED = {
    "11801": (
        "GMAT R2026a SPICESGP4 aborts on this element set; responsible field not "
        "isolated (see module docstring). Covered by the AIAA reference vectors "
        "and by FreeFlyer."
    ),
}


def load_fixture_tle(norad: str, path: Path = VERIFICATION_TLE) -> tuple[str, str]:
    """The two lines for *norad* from the committed AIAA verification fixture.

    Truncated to 69 columns: the fixture appends the reference driver's
    start/stop/step to line 2, which is not part of the TLE format and which no
    external parser has any reason to tolerate.
    """
    lines = [
        line.rstrip("\n")
        for line in path.read_text(encoding="ascii").splitlines()
        if not line.startswith("#")
    ]
    for i in range(len(lines) - 1):
        if lines[i].startswith(f"1 {norad}"):
            return lines[i][:69], lines[i + 1][:69]
    raise ValueError(f"NORAD {norad} not found in {path}")


def tle_epoch_gmat_mjd(line1: str) -> float:
    """The epoch encoded in TLE line 1 as GMAT's ModJulian day number.

    **Not** ``UTCGregorian``, and that is the whole point. GMAT's Gregorian
    spelling carries three decimal places, so a spacecraft epoch set that way is
    rounded to the nearest millisecond — and at LEO orbital speed 0.5 ms is
    **4 metres** of along-track position. Setting the epoch that way inflated the
    measured Polaris-vs-GMAT disagreement to 0.16-0.21 arcsec (5.7-7.2 m) against
    FreeFlyer's 0.027-0.035 arcsec on the same element sets: an artefact of the
    harness that would have been silently absorbed into a widened band.

    GMAT's ModJulian field is a double with its reference at JD 2430000.0
    (05 Jan 1941 12:00). Near 21723 its ULP is ~4e-12 day, about 0.3 microseconds
    or 2 mm at LEO — four orders below the quantity being measured, so the epoch
    stops being part of the error budget.
    """
    import datetime as _dt

    two_digit_year = int(line1[18:20])
    # The TLE two-digit year window: 57-99 -> 1957-1999, 00-56 -> 2000-2056.
    year = 1900 + two_digit_year if two_digit_year >= 57 else 2000 + two_digit_year
    day_of_year = float(line1[20:32])
    whole_days = int(day_of_year)
    # Kept as (integer days) + (fraction of a day) rather than one big float:
    # the integer part is exact and the fraction carries full double precision,
    # where forming a Julian date near 2.45e6 first would throw away ~8 us.
    days_to_jan1 = (_dt.date(year, 1, 1) - _dt.date(1941, 1, 5)).days - 0.5
    return days_to_jan1 + (whole_days - 1) + (day_of_year - whole_days)


def write_tle_file(line1: str, line2: str, path: Path, name: str = "POLARISVV") -> str:
    """Write a 3-line TLE and return the "line 0" identifier GMAT matches on.

    GMAT's ``Spacecraft.Id`` is matched against the line-0 name or the five-digit
    catalogue number **as a case-sensitive string**. The bare integer form is
    brittle across zero-padding, so a line-0 name is written and used.
    """
    path.write_text(f"{name}\n{line1}\n{line2}\n", encoding="ascii")
    return name


def build_tle_script(
    tle_path: Path, sat_id: str, epoch_mjd: float, sample_times_s, report_path: Path
) -> str:
    """A GMAT script propagating one TLE with ``SPICESGP4`` to each sample time.

    Each sample is a separate ``Propagate ... {ElapsedSecs = t}`` measured from
    the *epoch*, not from the previous sample, so a sample schedule is not a
    chain of accumulating steps.
    """
    steps = "\n".join(
        f"Propagate TLEProp(Sat) {{Sat.ElapsedSecs = {t:.6f}}}" for t in sample_times_s
    )
    columns = ", ".join(
        [
            "Sat.UTCModJulian",
            *(f"Sat.ICRFSys.{c}" for c in ("X", "Y", "Z", "VX", "VY", "VZ")),
            *(f"Sat.EarthMJ2000Eq.{c}" for c in ("X", "Y", "Z")),
        ]
    )
    return f"""% Polaris SGP4/TLE cross-validation - generated by tools/gmat/tle.py.
Create Spacecraft Sat
% ModJulian, not Gregorian: Gregorian rounds to the millisecond, which is metres
% of along-track position at LEO speed (see tle_epoch_gmat_mjd).
Sat.DateFormat = UTCModJulian
Sat.Epoch = '{epoch_mjd:.12f}'
Sat.EphemerisName = '{tle_path}'
Sat.Id = '{sat_id}'

% ICRF is the GCRS realisation Polaris's ECI matches. EarthMJ2000Eq is FK5-based;
% carrying both makes GMAT report the frame bias itself (see module docstring).
Create CoordinateSystem ICRFSys
ICRFSys.Origin = Earth
ICRFSys.Axes = ICRF

Create Propagator TLEProp
TLEProp.Type = SPICESGP4
TLEProp.InitialStepSize = 300

Create ReportFile RF
RF.Filename = '{report_path}'
RF.Precision = 16
RF.Add = {{ {columns} }}
RF.WriteHeaders = True
RF.FixedWidth = True
RF.Delimiter = ' '
RF.ColumnWidth = 25
RF.WriteReport = True

BeginMissionSequence
{steps}
"""


def parse_tle_report(text: str, sample_times_s) -> list[dict]:
    """Turn a GMAT ReportFile into one record per sample.

    GMAT emits a row per propagation step, so the rows are matched back to the
    requested sample times by elapsed time from the first row rather than by
    position — a step-size change must not silently re-label the samples.
    """
    rows = [line.split() for line in text.splitlines()[1:] if line.strip()]
    if not rows:
        raise ValueError("GMAT report contained no data rows")
    values = [[float(v) for v in row] for row in rows]
    t0 = values[0][0]
    out = []
    for wanted in sample_times_s:
        best = min(values, key=lambda v: abs((v[0] - t0) * SECONDS_PER_DAY - wanted))
        actual = (best[0] - t0) * SECONDS_PER_DAY
        if abs(actual - wanted) > 1.0e-3:
            raise ValueError(
                f"GMAT has no sample within 1 ms of t={wanted} s "
                f"(closest {actual} s) — the report and the requested schedule "
                f"have diverged; do not silently accept the nearest row."
            )
        out.append(
            {
                "t_s": wanted,
                "position_km": best[1:4],
                "velocity_km_s": best[4:7],
                "position_mj2000eq_km": best[7:10],
            }
        )
    return out


def run_tle_case(gmat_console: str, case: dict, work_dir: Path) -> list[dict]:
    """Run one case through GmatConsole and return its parsed samples."""
    line1, line2 = load_fixture_tle(case["norad"])
    work_dir.mkdir(parents=True, exist_ok=True)
    tle_path = work_dir / f"{case['name']}.tle"
    sat_id = write_tle_file(line1, line2, tle_path)
    report_path = work_dir / f"{case['name']}_report.txt"
    script_path = work_dir / f"{case['name']}.script"
    script_path.write_text(
        build_tle_script(
            tle_path,
            sat_id,
            tle_epoch_gmat_mjd(line1),
            case["sample_times_s"],
            report_path,
        ),
        encoding="ascii",
    )
    result = subprocess.run(  # noqa: S603 — console path is operator-supplied, not user input
        [gmat_console, "--run", str(script_path.resolve())],
        cwd=str(Path(gmat_console).resolve().parent),
        capture_output=True,
        text=True,
        # GMAT's console banner carries bytes that are not valid UTF-8; this
        # stream is only ever quoted back in an error message, so decode
        # leniently rather than letting an unrelated banner byte mask a real
        # propagation failure.
        errors="replace",
        timeout=600,
        check=False,
    )
    if not report_path.exists():
        raise RuntimeError(
            f"GMAT produced no report for {case['name']} (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return parse_tle_report(report_path.read_text(), case["sample_times_s"])


def regenerate_tle_fixture(gmat_console: str, work_dir: str | Path) -> dict:
    """Rebuild the whole fixture from GMAT. Returns the fixture dict."""
    work = Path(work_dir)
    cases = []
    for case in TLE_CASES:
        line1, line2 = load_fixture_tle(case["norad"])
        samples = run_tle_case(gmat_console, case, work / case["name"])
        cases.append(
            {
                "name": case["name"],
                "norad": case["norad"],
                "note": case["note"],
                "tle_line1": line1,
                "tle_line2": line2,
                "epoch_gmat_mjd": tle_epoch_gmat_mjd(line1),
                "samples": samples,
            }
        )
    return {
        "_comment": (
            "GMAT SGP4/TLE cross-validation fixture (REQ-ODP-003, REQ-VV-006). "
            "Generated by tools/gmat/tle.py; CI never runs GMAT. GMAT's SGP4 is "
            "NAIF SPICE's (propagator type SPICESGP4) -- a lineage independent of "
            "both Polaris (Vallado) and FreeFlyer (USSF AstroStds). position_km "
            "and velocity_km_s are ICRF, the GCRS realisation matching Polaris's "
            "ECI; position_mj2000eq_km is the same state in FK5-based "
            "EarthMJ2000Eq, so their difference is GMAT's own measurement of the "
            "frame bias that lib/frames/teme_eci.cpp applies."
        ),
        "gmat_version": "R2026a",
        "unsupported_cases": GMAT_UNSUPPORTED,
        "propagator": "SPICESGP4",
        "frame": "ICRF",
        "cases": cases,
    }


def compare_tle(
    committed: dict, regenerated: dict, tol_km: float = 1.0e-6
) -> list[str]:
    """Drift between a committed fixture and a freshly regenerated one.

    The band is tight on purpose: this compares GMAT against *itself*, so any
    difference beyond report round-off means the tool, its kernels or the script
    changed — which is the whole point of the weekly lane.
    """
    drift: list[str] = []
    by_name = {c["name"]: c for c in regenerated["cases"]}
    for case in committed["cases"]:
        fresh = by_name.get(case["name"])
        if fresh is None:
            drift.append(f"{case['name']}: absent from regenerated fixture")
            continue
        for old, new in zip(case["samples"], fresh["samples"]):
            d = math.dist(old["position_km"], new["position_km"])
            if d > tol_km:
                drift.append(
                    f"{case['name']} t={old['t_s']}s: position drifted {d:.3e} km "
                    f"(band {tol_km:.1e} km)"
                )
    return drift

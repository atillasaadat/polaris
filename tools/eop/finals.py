"""IERS ``finals2000A.all`` fetch + parse into the committed EOP fixture.

Ground-side only (design doc §23.1; REQ-CONV-002). IERS publishes the Earth
Orientation Parameters daily in the fixed-width ``finals2000A.all`` product; the
onboard/sim ``polaris::frames::EopTable`` consumes UT1-UTC and polar motion.
This trims the full product to a JSON fixture that is **committed data** — CI
never downloads (same policy as ``tests/golden/``); regenerate deliberately with
``python -m eop``.

The Bulletin A columns (always present, predictions included) are parsed, per the
IERS ``finals2000A.all`` format:

    8-15   F8.2   fractional MJD (UTC)
    19-27  F9.6   Bull. A PM-x [arcsec]
    38-46  F9.6   Bull. A PM-y [arcsec]
    59-68  F10.7  Bull. A UT1-UTC [s]

Rows past the prediction span leave these blank; parsing stops at the first row
with no UT1-UTC value.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# IERS Rapid Service/Prediction Center; the maia.usno mirror is the fallback.
DEFAULT_URL = "https://datacenter.iers.org/data/9/finals2000A.all"

# 0-indexed [start, stop) slices of the Bulletin A fixed-width columns above.
_MJD = slice(7, 15)
_PM_X = slice(18, 27)
_PM_Y = slice(37, 46)
_DUT1 = slice(58, 68)


@dataclass(frozen=True)
class EopRow:
    """One daily record, in the units the reduction consumes."""

    mjd_utc: float  # UTC Modified Julian Date (integral, 00:00 UTC)
    xp_arcsec: float  # polar motion x [arcsec]
    yp_arcsec: float  # polar motion y [arcsec]
    dut1: float  # UT1 - UTC [s]


def parse_finals2000a(text: str) -> list[EopRow]:
    """Parse the Bulletin A columns; stop at the first row lacking UT1-UTC."""
    rows: list[EopRow] = []
    for line in text.splitlines():
        if len(line) < _DUT1.stop:
            break
        dut1_field = line[_DUT1].strip()
        if not dut1_field:  # past the prediction span
            break
        rows.append(
            EopRow(
                mjd_utc=float(line[_MJD].strip()),
                xp_arcsec=float(line[_PM_X].strip()),
                yp_arcsec=float(line[_PM_Y].strip()),
                dut1=float(dut1_field),
            )
        )
    return rows


def fetch_finals2000a(url: str = DEFAULT_URL) -> str:
    """Download the raw product. Ground-side only; never called from CI."""
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted IERS host)
        return resp.read().decode("ascii", errors="replace")


def trim(rows: list[EopRow], start_mjd: float, end_mjd: float) -> list[EopRow]:
    """Keep the inclusive [start_mjd, end_mjd] window, ascending."""
    kept = [r for r in rows if start_mjd <= r.mjd_utc <= end_mjd]
    return sorted(kept, key=lambda r: r.mjd_utc)


def build_fixture(rows: list[EopRow], url: str) -> dict:
    """The committed-fixture shape: provenance + ascending daily entries."""
    return {
        "source": url,
        "product": "IERS finals2000A.all (Bulletin A)",
        "units": {
            "mjd_utc": "day",
            "dut1": "s (UT1-UTC)",
            "xp_arcsec": "arcsec",
            "yp_arcsec": "arcsec",
        },
        "entries": [
            {
                "mjd_utc": r.mjd_utc,
                "dut1": r.dut1,
                "xp_arcsec": r.xp_arcsec,
                "yp_arcsec": r.yp_arcsec,
            }
            for r in rows
        ],
    }


def write_fixture(path: Path, fixture: dict) -> None:
    path.write_text(json.dumps(fixture, indent=2) + "\n")


def _self_check() -> None:
    """Parse a synthetic finals2000A row and a blank-tail row (assert-based)."""
    # A real 2020-06-01 line (MJD 59001), truncated after the Bull. A UT1-UTC field.
    line = (
        "20 6 1 59001.00 I  0.073000 0.000100  0.285000 0.000100  "
        "I-0.1770000 0.0000100"
    )
    rows = parse_finals2000a(line + "\n" + " " * 70)
    assert len(rows) == 1, rows
    (r,) = rows
    assert r.mjd_utc == 59001.00, r
    assert abs(r.xp_arcsec - 0.073) < 1e-9, r
    assert abs(r.yp_arcsec - 0.285) < 1e-9, r
    assert abs(r.dut1 - (-0.177)) < 1e-9, r
    assert trim(rows, 59000, 59000) == [], "out-of-window row must be dropped"
    print("eop.finals self-check: ok")


if __name__ == "__main__":
    _self_check()

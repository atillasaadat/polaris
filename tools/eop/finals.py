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

# IERS *produces* EOP; JPL/NAIF and everyone else derive Earth orientation from
# it, so these are the same numbers served by different hosts. Tried in order,
# first success wins — so a manual regeneration never hangs on one flaky host.
# (CI never fetches: the fixture is committed static data.) JPL's own product is
# the binary SPICE kernel earth_latest_high_prec.bpc, which needs SPICE to read
# and yields the same EOP — not worth the dependency for a text table.
MIRRORS = (
    "https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt",
    "https://datacenter.iers.org/data/9/finals2000A.all",
    # NASA CDDIS mirror (may require an Earthdata login; skipped on auth failure).
    "https://cddis.nasa.gov/archive/products/iers/finals2000A.all",
    # USNO / maia classic mirror.
    "https://maia.usno.navy.mil/ser7/finals2000A.all",
)

# Back-compat alias for the primary endpoint.
DEFAULT_URL = MIRRORS[0]

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


def fetch_finals2000a(url: str | None = None) -> str:
    """Download the raw product, trying mirrors in order until one succeeds.

    Ground-side only; never called from CI. Pass @p url to force a single
    endpoint; otherwise every entry in ``MIRRORS`` is tried and the first
    success wins. Raises the last error if all mirrors fail.
    """
    urls = [url] if url else list(MIRRORS)
    last_err: Exception | None = None
    for candidate in urls:
        try:
            with urllib.request.urlopen(candidate, timeout=60) as resp:  # noqa: S310 (trusted IERS/NASA hosts)
                return resp.read().decode("ascii", errors="replace")
        except OSError as exc:  # DNS, timeout, HTTP error, auth failure
            last_err = exc
    raise OSError(f"all EOP mirrors failed; last error: {last_err}")


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

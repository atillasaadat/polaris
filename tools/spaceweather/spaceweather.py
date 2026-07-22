"""Fetch CelesTrak ``SW-All.csv`` for the committed space-weather fixture.

Ground-side only (design doc §23.1; REQ-SIM-002). NRLMSIS is driven by solar
radio flux (F10.7) and geomagnetic activity (Ap); CelesTrak publishes both daily
in this CSV, combining the observed record with near-term and monthly
predictions. We commit the file **verbatim** as CelesTrak serves it
(``tests/golden/SW-All.csv``) — so updating is just re-downloading and
overwriting, no bespoke format in between — and parse the columns on the C++ side
(``sim/world/space_weather_file.cpp``). CI never downloads.

The columns NRLMSIS 2.1 actually consumes (the model uses only the *daily* Ap
unless its storm-time switches are enabled, which Polaris leaves off — so the
eight 3-hourly Ap values are not read), 0-indexed:

    [0]   DATE               (YYYY-MM-DD, 00:00 UTC)
    [20]  AP_AVG             daily Ap index
    [24]  F10.7_OBS          observed daily F10.7 [sfu]
    [26]  F10.7_DATA_TYPE    OBS / INT / PRD / PRM
    [27]  F10.7_OBS_CENTER81 centered 81-day average of observed F10.7 [sfu]

`parse_sw_all` is kept only to sanity-check a fetch (the authoritative parse is
the C++ one). Rows past the daily record carry a blank AP_AVG (monthly
predictions give F10.7 only); parsing stops at the first such row.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass

# CelesTrak is the canonical redistribution of NOAA SWPC / GFZ Potsdam space
# weather for orbit work; a single authoritative host, no mirror chain.
DEFAULT_URL = "https://celestrak.org/SpaceData/SW-All.csv"

# 0-indexed CSV field positions (see module docstring).
_DATE = 0
_AP_AVG = 20
_F107_OBS = 24
_F107_OBS_CENTER81 = 27
# Require the full 31-column schema (not just the 28 we read) so a truncated row
# is skipped rather than parsed with a shifted last column. See the C++ loader.
_MIN_FIELDS = 31


@dataclass(frozen=True)
class SpaceWeatherRow:
    """One daily record, in the units NRLMSIS consumes."""

    date: str  # YYYY-MM-DD (00:00 UTC)
    ap_daily: float  # daily Ap index
    f107_obs: float  # observed daily F10.7 [sfu]
    f107_center81: float  # centered 81-day average of observed F10.7 [sfu]


def parse_sw_all(text: str) -> list[SpaceWeatherRow]:
    """Parse the daily records; stop at the first row lacking a daily Ap."""
    rows: list[SpaceWeatherRow] = []
    lines = text.splitlines()
    for line in lines[1:]:  # skip the header row
        fields = line.split(",")
        if len(fields) < _MIN_FIELDS:
            continue
        ap = fields[_AP_AVG].strip()
        if not ap:  # past the daily record (monthly predictions have no Ap)
            break
        rows.append(
            SpaceWeatherRow(
                date=fields[_DATE].strip(),
                ap_daily=float(ap),
                f107_obs=float(fields[_F107_OBS].strip()),
                f107_center81=float(fields[_F107_OBS_CENTER81].strip()),
            )
        )
    return rows


def fetch_sw_all(url: str | None = None) -> str:
    """Download the raw CSV. Ground-side only; never called from CI."""
    endpoint = url or DEFAULT_URL
    with urllib.request.urlopen(endpoint, timeout=60) as resp:  # noqa: S310 (trusted CelesTrak host)
        return resp.read().decode("ascii", errors="replace")


def _self_check() -> None:
    """Parse a synthetic header + row + blank-Ap tail (assert-based)."""
    header = "DATE,BSRN,ND," + ",".join(f"C{i}" for i in range(28))
    # AP_AVG at idx 20, F10.7_OBS at 24, F10.7_OBS_CENTER81 at 27.
    good = ["1957-10-01"] + ["x"] * 30
    good[_AP_AVG] = "21"
    good[_F107_OBS] = "269.3"
    good[_F107_OBS_CENTER81] = "266.6"
    tail = ["2041-09-01"] + [""] * 30  # monthly prediction: blank Ap
    tail[_F107_OBS] = "68.9"
    text = "\n".join([header, ",".join(good), ",".join(tail)])
    rows = parse_sw_all(text)
    assert len(rows) == 1, rows
    (r,) = rows
    assert r.date == "1957-10-01", r
    assert r.ap_daily == 21.0, r
    assert abs(r.f107_obs - 269.3) < 1e-9, r
    assert abs(r.f107_center81 - 266.6) < 1e-9, r
    print("spaceweather self-check: ok")


if __name__ == "__main__":
    _self_check()

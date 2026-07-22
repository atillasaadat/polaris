"""Tests for the CelesTrak space-weather fetch/parse tool (design doc §3.7).

CI never downloads, so these exercise the parse — the column convention and the
stop-at-the-daily-record boundary — against synthetic CSV text, plus a schema
check on the committed fixture.
"""

from __future__ import annotations

from pathlib import Path

from spaceweather import parse_sw_all

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "golden" / "SW-All.csv"

_HEADER = (
    "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,"
    "AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
    "F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81"
)


def _row(date: str, ap: str, f107: str, center81: str) -> str:
    fields = [date] + ["0"] * 30
    fields[20] = ap
    fields[24] = f107
    fields[27] = center81
    return ",".join(fields)


def test_parses_columns_by_position() -> None:
    text = "\n".join([_HEADER, _row("2024-06-15", "20", "200.5", "121.0")])
    rows = parse_sw_all(text)
    assert len(rows) == 1
    (r,) = rows
    assert r.date == "2024-06-15"
    assert r.ap_daily == 20.0
    assert r.f107_obs == 200.5
    assert r.f107_center81 == 121.0


def test_stops_at_blank_daily_ap() -> None:
    # A monthly-prediction tail row has a blank AP_AVG (idx 20) and must end parse.
    tail = ["2041-09-01"] + [""] * 30
    tail[24] = "69"
    text = "\n".join([_HEADER, _row("2024-06-15", "20", "200", "120"), ",".join(tail)])
    rows = parse_sw_all(text)
    assert len(rows) == 1


def test_committed_fixture_is_well_formed() -> None:
    rows = parse_sw_all(_FIXTURE.read_text())
    assert len(rows) > 20000  # decades of daily records
    # The first row is the fixed historical 1957-10-01 record.
    assert rows[0].date == "1957-10-01"
    assert rows[0].ap_daily == 21.0
    assert rows[0].f107_obs == 269.3

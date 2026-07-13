"""Tests for the GMAT golden-data harness (design doc §23.1, REQ-VV-002).

GMAT cannot run in CI, so these exercise the two pieces that must be correct
without it: parsing a GMAT ``ReportFile`` and turning its ModJulian columns into
scale offsets. The sample report is internally consistent with TAI−UTC = 19 s and
TT−TAI = 32.184 s at the GPS epoch.
"""

from __future__ import annotations

import pytest

from gmat import (
    build_time_scales_fixture,
    offsets_from_rows,
    parse_report,
    write_time_scales_script,
)
from gmat.golden import GMAT_MJD_TOL_S, SECONDS_PER_DAY

_UTC_1980 = {
    "year": 1980,
    "month": 1,
    "day": 6,
    "hour": 0,
    "minute": 0,
    "second": 0,
    "nanosecond": 0,
}

# utc = 44244.0; tai = utc + 19/86400; tt = utc + 51.184/86400.
_SAMPLE_REPORT = """\
sat.TAIModJulian sat.UTCModJulian sat.TTModJulian
44244.0002199074074 44244.0000000000000 44244.0005924074074
"""


@pytest.mark.verifies("REQ-VV-002")
def test_parse_report_reduces_columns_and_reads_values():
    rows = parse_report(_SAMPLE_REPORT)
    assert len(rows) == 1
    assert set(rows[0]) == {"TAIModJulian", "UTCModJulian", "TTModJulian"}
    assert rows[0]["UTCModJulian"] == pytest.approx(44244.0)


@pytest.mark.verifies("REQ-VV-002")
def test_offsets_recover_known_scale_differences():
    off = offsets_from_rows(parse_report(_SAMPLE_REPORT))[0]
    assert off["tai_minus_utc_s"] == pytest.approx(19.0, abs=1e-4)
    assert off["tt_minus_tai_s"] == pytest.approx(32.184, abs=1e-4)
    assert off["tai_minus_gps_s"] == 19.0  # definitional constant


@pytest.mark.verifies("REQ-VV-002")
def test_parse_report_rejects_ragged_rows():
    with pytest.raises(ValueError, match="cols, header has"):
        parse_report("a b c\n1.0 2.0\n")


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_matches_loader_schema():
    rows = parse_report(_SAMPLE_REPORT)
    fixture = build_time_scales_fixture(
        case_names=["gps_epoch_1980"],
        utc_fields=[
            {
                "year": 1980,
                "month": 1,
                "day": 6,
                "hour": 0,
                "minute": 0,
                "second": 0,
                "nanosecond": 0,
            }
        ],
        rows=rows,
    )
    assert fixture["schema_version"] == "1.0"
    case = fixture["cases"][0]
    assert case["name"] == "gps_epoch_1980"
    q = case["quantities"]["tai_minus_utc_s"]
    assert q["expected"] == pytest.approx(19.0, abs=1e-4)
    assert q["tol_abs"] == GMAT_MJD_TOL_S and q["unit"] == "s"


@pytest.mark.verifies("REQ-VV-002")
def test_write_script_emits_epoch_and_report_lines():
    script = write_time_scales_script(
        ["06 Jan 1980 00:00:00.000"], report_path="/tmp/rf.txt"
    )
    assert "sat.Epoch = '06 Jan 1980 00:00:00.000';" in script
    assert "Report rf sat.TAIModJulian sat.UTCModJulian sat.TTModJulian;" in script
    assert "BeginMissionSequence;" in script


@pytest.mark.verifies("REQ-VV-002")
def test_write_script_rejects_quote_injection():
    with pytest.raises(ValueError, match="single quote"):
        write_time_scales_script(["06 Jan 1980'; Stop;"], report_path="/tmp/rf.txt")


@pytest.mark.verifies("REQ-VV-002")
def test_parse_report_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        parse_report("a b c\n1.0 nan 3.0\n")


@pytest.mark.verifies("REQ-VV-002")
def test_offsets_reject_missing_column():
    with pytest.raises(ValueError, match="missing column"):
        offsets_from_rows([{"TAIModJulian": 44244.0, "UTCModJulian": 44244.0}])


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_rejects_length_mismatch():
    rows = parse_report(_SAMPLE_REPORT)  # one row
    with pytest.raises(ValueError, match="equal length"):
        build_time_scales_fixture(
            case_names=["a", "b"], utc_fields=[_UTC_1980, _UTC_1980], rows=rows
        )


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_rejects_missing_utc_keys():
    rows = parse_report(_SAMPLE_REPORT)
    with pytest.raises(ValueError, match="missing required keys"):
        build_time_scales_fixture(
            case_names=["gps_epoch_1980"], utc_fields=[{"year": 1980}], rows=rows
        )


@pytest.mark.verifies("REQ-VV-002")
def test_build_fixture_default_tol_is_the_mjd_precision_floor():
    # A GMAT-regenerated fixture cannot claim tighter than ~1 µs because it comes
    # from differencing float64 ModJulian columns (see golden.py). Guards the HIGH
    # review finding: the default must be the documented floor, not 1e-9.
    rows = parse_report(_SAMPLE_REPORT)
    fixture = build_time_scales_fixture(
        case_names=["gps_epoch_1980"], utc_fields=[_UTC_1980], rows=rows
    )
    tol = fixture["cases"][0]["quantities"]["tai_minus_utc_s"]["tol_abs"]
    assert tol == GMAT_MJD_TOL_S == pytest.approx(1.0e-6)
    assert 1.0 * SECONDS_PER_DAY == pytest.approx(86400.0)

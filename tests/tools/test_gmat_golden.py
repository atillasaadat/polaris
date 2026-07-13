"""Tests for the GMAT golden-data harness (design doc §23.1, REQ-VV-002).

GMAT cannot run in CI, so these exercise the two pieces that must be correct
without it: parsing a GMAT ``ReportFile`` and turning its ModJulian columns into
scale offsets. The sample report is internally consistent with TAI−UTC = 19 s and
TT−TAI = 32.184 s at the GPS epoch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmat import (
    build_time_scales_fixture,
    compare_time_scales,
    offsets_from_rows,
    parse_report,
    regenerate_time_scales_fixture,
    write_time_scales_script,
)
from gmat import golden
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
def test_parse_report_skips_repeated_headers():
    # GMAT (WriteHeaders=true) re-emits the header before each Report command.
    interleaved = (
        "sat.TAIModJulian sat.UTCModJulian sat.TTModJulian\n"
        "14244.50021990741 14244.5 14244.50059240741\n"
        "sat.TAIModJulian sat.UTCModJulian sat.TTModJulian\n"
        "28849.50042824074 28849.5 28849.50080074074\n"
    )
    rows = parse_report(interleaved)
    assert len(rows) == 2
    off = offsets_from_rows(rows)
    assert off[0]["tai_minus_utc_s"] == pytest.approx(19.0, abs=1e-4)
    assert off[1]["tai_minus_utc_s"] == pytest.approx(37.0, abs=1e-4)


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
def test_write_script_is_ascii_only():
    # GMAT R2026a rejects any non-ASCII byte in a script file (em-dashes, etc.).
    script = write_time_scales_script(
        ["06 Jan 1980 00:00:00.000", "01 Jan 2020 00:00:00.000"], report_path="rf.txt"
    )
    assert script.isascii(), "GMAT script must be pure ASCII"


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


# Report for the two canonical TIME_SCALES_CASES: 1980-01-06 (TAI−UTC=19) and
# 2020-01-01 (TAI−UTC=37); TT−TAI=32.184 s (=0.0003725 day) in both.
_TWO_CASE_REPORT = (
    "sat.TAIModJulian sat.UTCModJulian sat.TTModJulian\n"
    "44244.0002199074074 44244.0000000000000 44244.0005924074074\n"
    "51544.0004282407407 51544.0000000000000 51544.0008007407407\n"
)


@pytest.mark.verifies("REQ-VV-002")
def test_regenerate_runs_gmat_and_builds_two_case_fixture(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        # regenerate writes the report next to the script it hands GMAT
        (Path(cmd[1]).parent / "time_scales_report.txt").write_text(_TWO_CASE_REPORT)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(golden.subprocess, "run", fake_run)
    fixture = regenerate_time_scales_fixture("GmatConsole", str(tmp_path))
    assert [c["name"] for c in fixture["cases"]] == [
        "gps_epoch_1980",
        "post_2017_epoch_2020",
    ]
    assert fixture["cases"][1]["quantities"]["tai_minus_utc_s"]["expected"] == (
        pytest.approx(37.0, abs=1e-4)
    )


@pytest.mark.verifies("REQ-VV-002")
def test_regenerate_raises_when_gmat_writes_no_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        golden.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="no report"):
        regenerate_time_scales_fixture("GmatConsole", str(tmp_path))


@pytest.mark.verifies("REQ-VV-002")
def test_compare_passes_on_agreement_and_flags_drift():
    committed = build_time_scales_fixture(
        ["gps_epoch_1980"], [_UTC_1980], parse_report(_SAMPLE_REPORT)
    )
    assert compare_time_scales(committed, committed) == []

    drifted = json.loads(json.dumps(committed))  # deep copy
    drifted["cases"][0]["quantities"]["tai_minus_utc_s"]["expected"] += 1.0
    messages = compare_time_scales(committed, drifted)
    assert messages and "tai_minus_utc_s" in messages[0]


@pytest.mark.verifies("REQ-VV-002")
def test_compare_flags_missing_quantity():
    committed = build_time_scales_fixture(
        ["gps_epoch_1980"], [_UTC_1980], parse_report(_SAMPLE_REPORT)
    )
    regenerated = json.loads(json.dumps(committed))
    del regenerated["cases"][0]["quantities"]["tt_minus_tai_s"]
    messages = compare_time_scales(committed, regenerated)
    assert any("missing" in m for m in messages)

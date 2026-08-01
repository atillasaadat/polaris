"""Tests for the ``Svc::PrmDb`` parameter-file emitter (design doc §19.3).

The emitter writes a binary the flight software parses at startup, so these
tests pin the **byte layout** against the F´ v4.2.2 reader
(``fprime/Svc/PrmDb/PrmDbImpl.cpp``) rather than round-tripping through the
emitter's own helpers, which would agree with themselves no matter what: a
decoder written here from the reader's field order and widths is what catches an
endianness or record-size mistake. The end-to-end check that a real ``Svc::PrmDb``
accepts the file lives in tests/integration/sitl_attitude_tuning_test.cpp.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from configc import (
    PRMDB_FILENAME,
    ConfigError,
    ParamSpec,
    PrmDbError,
    build_param_file,
    compile_config,
    encode_records,
    load_dictionary,
)
from configc.prmdb import ENTRY_DELIMITER, MAX_ENTRIES

_REPO = Path(__file__).resolve().parents[2]
_HARDWARE = _REPO / "config" / "hardware"
_TEMPLATE = _REPO / "config" / "spacecraft" / "leo_smallsat.yaml"
_DICTIONARY = (
    _REPO
    / "build-artifacts"
    / "Linux"
    / "flight_PolarisFsw"
    / "dict"
    / "PolarisFswTopologyDictionary.json"
)


def _decode(image: bytes) -> list[tuple[int, bytes]]:
    """Parse a PrmDb file exactly as ``PrmDbImpl::readParamFileImpl`` does.

    Written from the reader, not from the emitter: CRC header in host byte order,
    then delimiter / big-endian record size / big-endian U32 ID / value bytes.
    Raises AssertionError on anything the flight reader would reject.
    """
    assert len(image) >= 4, "file is shorter than the CRC header"
    file_crc = int.from_bytes(image[:4], "little")
    body = image[4:]
    assert (
        file_crc == zlib.crc32(body) ^ 0xFFFFFFFF
    ), "CRC does not cover the record body"

    records: list[tuple[int, bytes]] = []
    offset = 0
    while offset < len(body):
        assert body[offset] == ENTRY_DELIMITER, f"bad delimiter at byte {offset}"
        offset += 1
        (record_size,) = struct.unpack_from(">I", body, offset)
        offset += 4
        assert record_size >= 4, "record smaller than the parameter ID"
        (param_id,) = struct.unpack_from(">I", body, offset)
        offset += 4
        value_len = record_size - 4
        records.append((param_id, body[offset : offset + value_len]))
        offset += value_len
    assert offset == len(body), "trailing bytes after the last record"
    return records


def _spec(name: str, param_id: int, type_name: str = "F64") -> ParamSpec:
    return ParamSpec(name=name, param_id=param_id, type_name=type_name)


# --- Byte layout --------------------------------------------------------------


def test_record_layout_matches_the_flight_reader():
    image = encode_records([(0x10030000, struct.pack(">d", 0.0116))])
    # 4 CRC + 1 delimiter + 4 record size + 4 ID + 8 value
    assert len(image) == 21
    assert _decode(image) == [(0x10030000, struct.pack(">d", 0.0116))]


def test_values_are_serialized_big_endian_per_type():
    dictionary = {
        "a": _spec("a", 1, "F64"),
        "b": _spec("b", 2, "U32"),
        "c": _spec("c", 3, "bool"),
    }
    records = dict(_decode(build_param_file({"a": 1.5, "b": 7, "c": True}, dictionary)))
    assert records[1] == struct.pack(">d", 1.5)
    assert records[2] == b"\x00\x00\x00\x07"
    assert records[3] == b"\x01"


def test_encoding_is_deterministic():
    dictionary = {"a": _spec("a", 1), "b": _spec("b", 2)}
    values = {"b": 2.0, "a": 1.0}
    assert build_param_file(values, dictionary) == build_param_file(
        dict(reversed(list(values.items()))), dictionary
    )


def test_too_many_records_is_refused():
    # Past PRMDB_NUM_DB_ENTRIES the flight database silently drops records, so
    # the emitter must refuse rather than ship a file that loads partially.
    with pytest.raises(PrmDbError, match="PRMDB_NUM_DB_ENTRIES"):
        encode_records([(i, b"\x00") for i in range(MAX_ENTRIES + 1)])


# --- Validation against the flight build --------------------------------------


def test_unknown_parameter_name_is_refused():
    with pytest.raises(PrmDbError, match="does not declare"):
        build_param_file({"typo": 1.0}, {"a": _spec("a", 1)})


def test_unset_declared_parameter_is_refused():
    # §19.3: no flight defaults, so a parameter the build declares and the config
    # omits would ship a component that refuses to run.
    with pytest.raises(PrmDbError, match="no flight defaults"):
        build_param_file({"a": 1.0}, {"a": _spec("a", 1), "b": _spec("b", 2)})


def test_unsupported_type_is_refused():
    with pytest.raises(PrmDbError, match="not a scalar"):
        build_param_file({"a": 1.0}, {"a": _spec("a", 1, "SomeStructType")})


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (70000, "U16"),  # struct.error
        (1e40, "F32"),  # OverflowError, a different exception type
    ],
)
def test_out_of_range_value_is_refused(value, type_name):
    with pytest.raises(PrmDbError, match="not representable"):
        build_param_file({"a": value}, {"a": _spec("a", 1, type_name)})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_value_is_refused(value):
    # These pack into IEEE-754 without complaint, so nothing downstream would
    # catch them until the vehicle's own finiteness check refused the cycle.
    with pytest.raises(PrmDbError, match="not finite"):
        build_param_file({"a": value}, {"a": _spec("a", 1, "F64")})


def test_integer_and_bool_values_survive_the_schema(tmp_path):
    # The config schema must not coerce these to float, or struct.pack fails on
    # an integer- or bool-typed F´ parameter with a confusing error.
    from configc.schema import Spacecraft

    sc = Spacecraft.model_validate(
        {
            "name": "T",
            "mass_kg": 10.0,
            "com_m": [0.0, 0.0, 0.0],
            "inertia_kgm2": {"ixx": 0.1, "iyy": 0.1, "izz": 0.1},
            "fsw_parameters": {"count": 7, "flag": True, "sigma": 0.5},
        }
    )
    dictionary = {
        "count": _spec("count", 1, "U32"),
        "flag": _spec("flag", 2, "bool"),
        "sigma": _spec("sigma", 3, "F64"),
    }
    records = dict(_decode(build_param_file(sc.fsw_parameters, dictionary)))
    assert records[1] == b"\x00\x00\x00\x07"
    assert records[2] == b"\x01"


def test_malformed_dictionary_is_refused(tmp_path):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps({"commands": []}), encoding="utf-8")
    with pytest.raises(PrmDbError, match="not an F´ topology dictionary"):
        load_dictionary(path)


# --- Against the real flight dictionary and vehicle config --------------------

_needs_build = pytest.mark.skipif(
    not _DICTIONARY.is_file(),
    reason="flight dictionary not built (run `uv run fprime-util build`)",
)


@_needs_build
def test_dictionary_ids_are_read_not_invented():
    dictionary = load_dictionary(_DICTIONARY)
    raw = json.loads(_DICTIONARY.read_text(encoding="utf-8"))
    assert {name: spec.param_id for name, spec in dictionary.items()} == {
        entry["name"]: entry["id"] for entry in raw["parameters"]
    }


@_needs_build
@pytest.mark.verifies("REQ-CFG-001")
def test_reference_vehicle_emits_a_loadable_parameter_file(tmp_path):
    compile_config(_TEMPLATE, _HARDWARE, tmp_path, dictionary_path=_DICTIONARY)
    image = (tmp_path / PRMDB_FILENAME).read_bytes()

    dictionary = load_dictionary(_DICTIONARY)
    records = dict(_decode(image))
    assert set(records) == {spec.param_id for spec in dictionary.values()}
    assert len(records) <= MAX_ENTRIES

    # Spot-check one value against the config, decoded independently: this is
    # the link that would break if IDs and values were ever paired up wrongly.
    sigma = dictionary["flight.attitudeEstimator.SigmaSunWhiteRad"]
    assert struct.unpack(">d", records[sigma.param_id])[0] == pytest.approx(0.0116)


@_needs_build
def test_config_naming_an_unknown_parameter_fails_the_compile(tmp_path):
    import yaml

    config = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    config["spacecraft"]["fsw_parameters"]["flight.attitudeEstimator.NoSuchParam"] = 1.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="NoSuchParam"):
        compile_config(path, _HARDWARE, tmp_path, dictionary_path=_DICTIONARY)

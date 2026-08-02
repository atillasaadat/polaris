"""F´ ``Svc::PrmDb`` parameter-file emitter (design doc §19.3, REQ-CFG-001).

The last link in the tuning chain: the config compiler resolves mission tuning
values, and this module encodes them in the on-disk format ``Svc::PrmDb`` reads
at startup (``readParamFile``, the topology's ``readParameters`` phase). Without
it, a component with no-default parameters — ``flight::AttitudeEstimator`` is the
first — refuses to run because nothing ever populated the database.

Parameter **IDs are never written by hand**. They come from the FPP-generated
topology dictionary (``build-artifacts/Linux/flight_PolarisFsw/dict/
PolarisFswTopologyDictionary.json``), which is the same artifact the ground
system reads, so an ID renumbering (a base-ID move, a reordered ``param``
declaration) can never silently desynchronise the file from the flight build:
the config names the parameter, the dictionary supplies the number.

File format (F´ v4.2.2 ``Svc/PrmDb/PrmDbImpl.cpp``; the component SDD documents
only the record layout and predates the CRC header)::

    U32 CRC                                   host byte order, see note below
    repeated, one per parameter:
      U8   0xA5                               entry delimiter
      U32  record size (big-endian)           sizeof(FwPrmIdType) + value bytes
      U32  parameter ID (big-endian)          FwPrmIdType = FwIdType = U32
      ...  serialized parameter value         F´ serialization (big-endian)

The CRC is a reflected CRC-32 (``Utils/Hash/libcrc``, polynomial 0xEDB88320)
seeded 0xFFFFFFFF **without** the final complement that ``zlib.crc32`` applies,
accumulated over every byte after the header. Unlike every other field it is
written with a raw memcpy of the ``U32``, so it takes the *host's* byte order
rather than F´'s big-endian serialization; this emitter writes little-endian,
which covers x86-64 and every ARM flight target in practice. A big-endian target
would need the byte order flipped here — hence :data:`CRC_BYTE_ORDER`.
"""

from __future__ import annotations

import json
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

#: Byte order of the CRC header field: the target CPU's, not F´'s wire order.
CRC_BYTE_ORDER = "little"

#: ``PRMDB_ENTRY_DELIMITER`` (``fprime/default/config/PrmDbImplCfg.hpp``).
ENTRY_DELIMITER = 0xA5

#: ``sizeof(FwPrmIdType)``: ``FwPrmIdType = FwIdType = U32``
#: (``fprime/default/config/FpConfig.fpp``). Counted into each record's size.
_PRM_ID_BYTES = 4

#: Polaris's override of the F´ default configuration, where the flight value of
#: ``PRMDB_NUM_DB_ENTRIES`` lives.
PRMDB_CONFIG_HEADER = (
    Path(__file__).parents[2] / "flight" / "config" / "PrmDbImplCfg.hpp"
)


def _read_max_entries(header: Path = PRMDB_CONFIG_HEADER) -> int:
    """``PRMDB_NUM_DB_ENTRIES`` as the flight build sees it.

    Read out of the header rather than duplicated here. The database holds no
    more records than this and a longer file loads *partially*, leaving later
    parameters invalid — so a copy of the number that drifted low would refuse
    valid configs, and one that drifted high would ship exactly the silently
    broken file this limit exists to prevent. Neither is a failure a comment
    saying "keep these in step" reliably catches.
    """
    match = re.search(
        r"PRMDB_NUM_DB_ENTRIES\s*=\s*(\d+)", header.read_text(encoding="utf-8")
    )
    if match is None:
        # Plain ValueError, not PrmDbError: this runs at import time, before
        # that subclass is defined. PrmDbError is a ValueError anyway.
        raise ValueError(f"{header}: no PRMDB_NUM_DB_ENTRIES definition found")
    return int(match.group(1))


#: The database holds no more records than this (see :func:`_read_max_entries`).
MAX_ENTRIES = _read_max_entries()

#: F´ scalar type name -> ``struct`` format, big-endian (F´ serialization order).
_VALUE_FORMATS: Mapping[str, str] = {
    "bool": ">?",
    "U8": ">B",
    "I8": ">b",
    "U16": ">H",
    "I16": ">h",
    "U32": ">I",
    "I32": ">i",
    "U64": ">Q",
    "I64": ">q",
    "F32": ">f",
    "F64": ">d",
}


class PrmDbError(ValueError):
    """A parameter set that cannot be encoded into a ``Svc::PrmDb`` file."""


class ParamSpec(NamedTuple):
    """One parameter as the FPP-generated dictionary declares it.

    ``array_size`` is 0 for a scalar and the element count for an FPP array
    type (``Vec3F64`` and friends), in which case ``type_name`` is the *element*
    type. Arrays are supported because some tuning genuinely is a vector — a
    sensor boresight, say — and splitting one into three scalar parameters is
    three chances for them to disagree about which vector they describe.
    """

    name: str
    param_id: int
    type_name: str
    array_size: int = 0


def load_dictionary(path: Path) -> dict[str, ParamSpec]:
    """Read the FPP topology dictionary's ``parameters`` block, keyed by name.

    @p path is the JSON dictionary ``fprime-util build`` emits. Raises
    PrmDbError on a missing/malformed file, so a stale ``--dictionary`` argument
    fails here rather than producing a file of plausible-looking wrong IDs.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PrmDbError(f"{path}: cannot read topology dictionary\n{exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrmDbError(f"{path}: not valid JSON\n{exc}") from exc
    if not isinstance(raw, dict) or "parameters" not in raw:
        raise PrmDbError(f"{path}: not an F´ topology dictionary (no 'parameters')")

    # FPP names an array-typed parameter by qualified identifier and puts its
    # shape in `typeDefinitions`, so the two have to be read together.
    arrays = {
        str(t["qualifiedName"]): t
        for t in raw.get("typeDefinitions", [])
        if isinstance(t, dict) and t.get("kind") == "array"
    }

    specs: dict[str, ParamSpec] = {}
    for entry in raw["parameters"]:
        try:
            name = str(entry["name"])
            param_id = int(entry["id"])
            type_name = str(entry["type"]["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PrmDbError(f"{path}: malformed parameter entry {entry!r}") from exc
        if name in specs:
            raise PrmDbError(f"{path}: duplicate parameter name '{name}'")
        array_size = 0
        if type_name in arrays:
            definition = arrays[type_name]
            try:
                array_size = int(definition["size"])
                type_name = str(definition["elementType"]["name"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PrmDbError(
                    f"{path}: malformed array type {definition!r}"
                ) from exc
        specs[name] = ParamSpec(
            name=name,
            param_id=param_id,
            type_name=type_name,
            array_size=array_size,
        )
    return specs


def serialize_value(spec: ParamSpec, value: Any) -> bytes:
    """F´-serialize @p value as @p spec's declared type (big-endian).

    An array parameter takes a sequence of exactly its declared length and is
    encoded as its elements back to back, which is F´'s own array serialization
    — no length prefix, because the length is part of the type.
    """
    if spec.array_size:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise PrmDbError(
                f"parameter '{spec.name}': {spec.type_name}[{spec.array_size}] "
                f"needs a sequence of {spec.array_size} values, got {value!r}"
            )
        if len(value) != spec.array_size:
            raise PrmDbError(
                f"parameter '{spec.name}': expected {spec.array_size} values, "
                f"got {len(value)}"
            )
        element = spec._replace(array_size=0)
        return b"".join(serialize_value(element, item) for item in value)

    fmt = _VALUE_FORMATS.get(spec.type_name)
    if fmt is None:
        raise PrmDbError(
            f"parameter '{spec.name}': type {spec.type_name} is not a scalar this "
            f"emitter can encode (supported: {', '.join(sorted(_VALUE_FORMATS))})"
        )
    if isinstance(value, bool) and spec.type_name != "bool":
        raise PrmDbError(f"parameter '{spec.name}': bool given for {spec.type_name}")
    # NaN and +/-inf pack happily into IEEE-754 and would reach the vehicle as a
    # tuning value every consumer's finiteness check then rejects at runtime.
    # A config that says `.nan` is wrong on the ground, so fail on the ground.
    if spec.type_name in ("F32", "F64") and not math.isfinite(value):
        raise PrmDbError(f"parameter '{spec.name}': {value!r} is not finite")
    try:
        return struct.pack(fmt, value)
    # struct raises OverflowError, not struct.error, for an out-of-range float.
    except (struct.error, OverflowError) as exc:
        raise PrmDbError(
            f"parameter '{spec.name}': {value!r} is not representable as "
            f"{spec.type_name}\n{exc}"
        ) from exc


def encode_records(records: list[tuple[int, bytes]]) -> bytes:
    """Encode ``(parameter id, serialized value)`` pairs into a PrmDb file image.

    Records are written in the order given; ``Svc::PrmDb`` keys on the ID, so the
    order only has to be stable (it is, so the artifact is reproducible).
    """
    if len(records) > MAX_ENTRIES:
        raise PrmDbError(
            f"{len(records)} parameters exceeds PRMDB_NUM_DB_ENTRIES ({MAX_ENTRIES}) "
            f"— records past the limit would never load"
        )
    body = bytearray()
    for param_id, value in records:
        if not 0 <= param_id <= 0xFFFFFFFF:
            raise PrmDbError(f"parameter id {param_id} does not fit FwPrmIdType (U32)")
        # Record size counts the ID as well as the value, per PrmDbImpl's
        # sanity check (`recordSize >= sizeof(U32)`).
        body.append(ENTRY_DELIMITER)
        body += struct.pack(">I", _PRM_ID_BYTES + len(value))
        body += struct.pack(">I", param_id)
        body += value
    crc = zlib.crc32(bytes(body)) ^ 0xFFFFFFFF
    return crc.to_bytes(4, CRC_BYTE_ORDER) + bytes(body)


def build_param_file(
    values: Mapping[str, Any], dictionary: Mapping[str, ParamSpec]
) -> bytes:
    """Encode every parameter the flight build declares, from @p values.

    Both directions are errors, deliberately (§19.3, "no defaults"): a config
    value naming a parameter the build does not have is a typo or a stale config,
    and a declared parameter the config does not set would ship a component that
    refuses to run — the failure this whole path exists to remove. Failing at
    compile time is what makes the parameter file trustworthy on orbit.
    """
    unknown = sorted(set(values) - set(dictionary))
    if unknown:
        raise PrmDbError(
            "config sets parameters the flight build does not declare: "
            + ", ".join(unknown)
        )
    missing = sorted(set(dictionary) - set(values))
    if missing:
        raise PrmDbError(
            "flight build declares parameters the config does not set (there are "
            "no flight defaults, design doc §19.3): " + ", ".join(missing)
        )
    records = [
        (dictionary[name].param_id, serialize_value(dictionary[name], values[name]))
        for name in sorted(values)
    ]
    return encode_records(records)

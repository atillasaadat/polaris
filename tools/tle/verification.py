"""Fetch the official SGP4 verification fixtures (design doc §3.7, REQ-ODP-003).

Ground-side only; CI never downloads. SGP4 is unusual among the algorithms in
this repository: it is defined **by its reference implementation**, not by a
closed-form description one can re-derive. Vallado's AIAA 2006-6753 paper says
so explicitly, and it ships a verification set — a TLE file exercising every
branch (near-Earth, deep-space resonance, Lyddane choice, decayed orbits) and
the state vectors the reference implementations produce for it.

That set is the acceptance test for :cpp:class:`polaris::gnc::Sgp4`. Matching it
is not evidence of correctness by analogy; for this algorithm it *is*
correctness, which is why the tolerance below is a printing-precision floor
rather than an engineering allowance.

Two files are committed **verbatim** under ``tests/golden/``, upstream names
kept so a refresh is a re-download and overwrite with no bespoke format in
between:

``SGP4-VER.TLE``
    33 verification cases. Line 2 of each carries three non-standard trailing
    fields — start, stop and step in minutes — which the reference drivers read
    and an ordinary TLE parser must ignore. Real TLEs have no such fields.

``tforverf.out``
    Expected state vectors from the **Fortran** reference: time from epoch
    [min], TEME position [km], TEME velocity [km/s].

**Why the Fortran output and not the C++ one.** The package ships no
``tcppver.out``; the C++ project builds it. It ships Fortran and MATLAB outputs,
and those two independent implementations agree to **7e-8 km (0.07 mm)** over
all 665 comparable rows — the last printed digit. So either is the reference and
neither is privileged; the Fortran file is the more recent of the two. That
measured agreement is where ``kVerificationToleranceKm`` comes from: a tolerance
tighter than the references agree with each other would fail on their round-off
rather than on ours.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import urllib.request

#: The AIAA 2006-6753 companion package (Vallado, Crawford, Hujsak & Kelso,
#: "Revisiting Spacetrack Report #3"). CelesTrak is the authoritative host — it
#: is Kelso's site, and he is an author. One URL, not a mirror list: unlike the
#: IERS EOP product this is a versioned publication artifact, so a "mirror"
#: serving a different revision would be a different fixture wearing the same
#: name, which is worse than a failed download.
PACKAGE_URL = "https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753.zip"

#: Paths inside the archive -> committed filename under tests/golden/.
MEMBERS = {
    "sgp4/cpp/testsgp4/SGP4-VER.TLE": "SGP4-VER.TLE",
    "sgp4/for/tforverf.out": "tforverf.out",
}


@dataclass(frozen=True)
class Fetched:
    """One extracted member and the bytes committed for it."""

    member: str
    filename: str
    data: bytes


def fetch(url: str = PACKAGE_URL, timeout_s: float = 120.0) -> list[Fetched]:
    """Download the package and extract the verification members.

    Returns them in the order of :data:`MEMBERS`. Raises on a missing member
    rather than writing a short set — a verification fixture that silently lost
    a file would weaken the acceptance test without failing it, which is the
    failure mode this repository keeps rediscovering.
    """
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        blob = response.read()
    out: list[Fetched] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = set(archive.namelist())
        for member, filename in MEMBERS.items():
            if member not in names:
                raise KeyError(
                    f"{member!r} is not in {url} — the package layout changed. "
                    f"Do not substitute a similarly named file: the fixture is "
                    f"only meaningful as the set the reference produced."
                )
            out.append(
                Fetched(member=member, filename=filename, data=archive.read(member))
            )
    return out


def write(destination: Path, fetched: list[Fetched]) -> list[Path]:
    """Write extracted members into @p destination, verbatim."""
    written = []
    for item in fetched:
        path = destination / item.filename
        path.write_bytes(item.data)
        written.append(path)
    return written

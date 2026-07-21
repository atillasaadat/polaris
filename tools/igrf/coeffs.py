"""Fetch the IAGA ``igrf14coeffs.txt`` product for the committed IGRF fixture.

Ground-side only (design doc §3.7). IAGA publishes the 14th-generation
International Geomagnetic Reference Field as a single fixed-column text table;
`lib/environment/igrf.*` consumes the Schmidt semi-normalised Gauss coefficients
it carries. We commit the file **verbatim** as upstream serves it
(``tests/golden/igrf14coeffs.txt``) — updating is re-downloading and overwriting,
with no bespoke format in between — and parse it on the C++ side. CI never
downloads.

Layout: two ``#`` comment lines, a model-type row (``c/s deg ord IGRF DGRF ...``),
a header row naming the epochs, then one row per Gauss coefficient::

    g/h  n  m  <value at 1900.0> ... <value at 2025.0>  <secular variation>

The 26 main-field columns are nT at 5-year epochs 1900.0 … 2025.0; the trailing
column is the 2025-2030 secular variation in nT/yr, which is how the model is
evaluated past the last definitive epoch. Rows are in the conventional spherical
harmonic order (g10, g11, h11, g20, …) up to degree 13.

`parse_igrf_coeffs` is kept only to sanity-check a fetch and to cross-check the
committed file against the IAGA reference implementation's own ``IGRF14.SHC``
(see ``tools/igrf/reference.py``); the authoritative parse is the C++ one.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass

# IAGA Working Group V-MOD coefficient endpoint, hosted by NOAA NCEI.
DEFAULT_URL = "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt"

# The row naming the epochs is the last header row; data rows start with g or h.
_KINDS = ("g", "h")


@dataclass(frozen=True)
class CoeffRow:
    """One Gauss coefficient across every epoch, in the units IAGA publishes."""

    kind: str  # "g" or "h"
    n: int  # spherical harmonic degree
    m: int  # spherical harmonic order
    values: tuple[float, ...]  # main field at each epoch [nT]
    sv: float  # secular variation over the trailing 5-year window [nT/yr]


@dataclass(frozen=True)
class IgrfCoeffs:
    """The parsed table: epoch list plus the g/h rows in file order."""

    epochs: tuple[float, ...]  # decimal years, 1900.0 .. 2025.0
    rows: tuple[CoeffRow, ...]

    @property
    def nmax(self) -> int:
        """Highest degree present."""
        return max(row.n for row in self.rows)


def parse_igrf_coeffs(text: str) -> IgrfCoeffs:
    """Parse the IAGA coefficient table.

    Raises
    ------
    ValueError
        If no epoch header row is found, or a data row's column count does not
        match the epoch count (which is what a truncated download looks like).
    """
    epochs: tuple[float, ...] | None = None
    rows: list[CoeffRow] = []

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "g/h":
            # "g/h n m 1900.0 ... 2025.0 2025-30" — the trailing SV label is not
            # an epoch, so it is dropped here and carried on CoeffRow.sv instead.
            epochs = tuple(float(f) for f in fields[3:-1])
            continue
        if fields[0] not in _KINDS:
            continue  # the model-type row, or any other annotation
        if epochs is None:
            raise ValueError("data row encountered before the epoch header row")
        values = [float(f) for f in fields[3:]]
        if len(values) != len(epochs) + 1:
            raise ValueError(
                f"row {fields[:3]} has {len(values)} columns, "
                f"expected {len(epochs)} epochs + 1 secular-variation column"
            )
        rows.append(
            CoeffRow(
                kind=fields[0],
                n=int(fields[1]),
                m=int(fields[2]),
                values=tuple(values[:-1]),
                sv=values[-1],
            )
        )

    if epochs is None:
        raise ValueError("no 'g/h' epoch header row found — not an IGRF coeff file?")
    return IgrfCoeffs(epochs=epochs, rows=tuple(rows))


def fetch(url: str = DEFAULT_URL) -> str:
    """Download the raw coefficient table. Ground-side only; never called from CI."""
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted NOAA host)
        return resp.read().decode("ascii", errors="replace")


def _self_check() -> None:
    """Parse a synthetic two-epoch table (assert-based)."""
    text = (
        "# 14th Generation International Geomagnetic Reference Field\n"
        "c/s deg ord DGRF IGRF SV\n"
        "g/h n m 2020.0 2025.0 2025-30\n"
        "g  1  0 -29403.41 -29350.0 12.0\n"
        "g  1  1  -1451.37  -1410.5  9.7\n"
        "h  1  1   4653.35   4545.5 -21.5\n"
    )
    parsed = parse_igrf_coeffs(text)
    assert parsed.epochs == (2020.0, 2025.0), parsed.epochs
    assert parsed.nmax == 1, parsed.nmax
    assert len(parsed.rows) == 3, parsed.rows
    assert parsed.rows[2] == CoeffRow("h", 1, 1, (4653.35, 4545.5), -21.5), parsed.rows

    ragged = text + "g  2  0  -2499.78 12.0\n"
    try:
        parse_igrf_coeffs(ragged)
    except ValueError:
        pass
    else:
        raise AssertionError("a short data row should have been rejected")
    print("igrf.coeffs self-check: ok")


if __name__ == "__main__":
    _self_check()

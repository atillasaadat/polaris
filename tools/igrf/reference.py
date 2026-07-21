"""Generate the committed IGRF-14 golden fixture from the IAGA reference code.

Ground-side only (design doc §3.7). The fixture is a **derived product**:
field components evaluated by ``pyIGRF14``, the official IAGA reference
implementation (Ciaran Beggan, BGS; MIT). ``pyIGRF14`` is a **fetch input**, not
vendored — exactly like ``de440s.bsp`` for ``tools/ephem`` — so nothing of BGS's
enters git history and the fixture stays reproducible by re-download.

What is committed alongside it is the IAGA coefficient table itself, verbatim
(``tests/golden/igrf14coeffs.txt``, see ``tools/igrf/coeffs.py``).
``pyIGRF14`` ships the same coefficients in its own ``SHC_files/IGRF14.SHC``;
`cross_check` proves the two agree before any field value is written, so the
fixture cannot silently be a golden for a *different* model than the one the
flight code parses.

Field synthesis is ``igrf_utils.synth_values`` verbatim. Time interpolation is
linear between epoch snapshots — what ``pyIGRF.py`` does with
``scipy.interpolate.interp1d`` (default ``kind='linear'``), reproduced here with
`numpy.interp` so the tool needs no SciPy. The SHC file carries a synthetic
2030.0 snapshot equal to ``2025.0 + 5 * SV``, so dates in the 2025-2030
secular-variation window fall inside the interpolation range and need no special
case; `cross_check` verifies that relation too.

Outputs are geocentric spherical components in nT at a geocentric radius — no
geodetic conversion — because that is the frame the onboard model works in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from .coeffs import IgrfCoeffs

#: Where to get the reference implementation, if `--pyigrf` points at nothing.
PYIGRF_URL = "https://www.ngdc.noaa.gov/IAGA/vmod/pyIGRF14.zip"

#: IGRF's spherical-harmonic reference radius [km]. Not the WGS-84 semi-major
#: axis: it is the mean Earth radius the Gauss coefficients are defined against,
#: and using anything else silently rescales the whole field.
REFERENCE_RADIUS_KM = 6371.2

#: Decimal years spanning IGRF-14 validity. Chosen to exercise, in order: an
#: early DGRF interval, a mid-century interval, an exact epoch boundary, a
#: mid-interval date, a second exact boundary, two more mid-interval dates, the
#: last definitive epoch, and two dates inside the 2025-2030 SV-extrapolation
#: window.
SAMPLE_YEARS = (
    1905.0,
    1957.3,
    2000.0,
    2012.5,
    2020.0,
    2023.7,
    2025.0,
    2026.5,
    2029.0,
)

#: ``(geocentric radius [km], colatitude [deg], longitude [deg])``. Colatitudes
#: 0.001 and 179.999 sit just inside the poles, where ``B_phi`` carries a
#: ``1/sin(theta)`` factor — the one place a re-implementation blows up. The rest
#: spread over both hemispheres, the equator, the full longitude circle
#: (including 0 and values past 180), and radii from the reference sphere
#: through LEO to GEO.
SAMPLE_POINTS = (
    (REFERENCE_RADIUS_KM, 0.001, 0.0),
    (REFERENCE_RADIUS_KM, 179.999, 0.0),
    (6771.0, 0.001, 275.0),
    (6771.0, 179.999, 190.0),
    (REFERENCE_RADIUS_KM, 90.0, 0.0),
    (6771.0, 90.0, 123.456),
    (REFERENCE_RADIUS_KM, 45.0, 210.0),
    (7171.0, 135.0, 359.9),
    (42164.0, 23.5, 180.0),
    (42164.0, 156.5, 300.0),
    (6871.0, 60.0, 45.0),
    (REFERENCE_RADIUS_KM, 10.0, 100.0),
)


def load_igrf_utils(pyigrf_dir: Path) -> ModuleType:
    """Import ``igrf_utils`` out of an extracted pyIGRF14 tree.

    Parameters
    ----------
    pyigrf_dir : Path
        Directory holding ``igrf_utils.py`` and ``SHC_files/``.

    Raises
    ------
    FileNotFoundError
        If the module is absent, with the download URL in the message.
    """
    module_path = pyigrf_dir / "igrf_utils.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{module_path} not found. pyIGRF14 is a fetch input and is not "
            f"committed; download and extract it from {PYIGRF_URL}, then pass "
            "--pyigrf <extracted>/pyIGRF14"
        )
    sys.path.insert(0, str(pyigrf_dir))
    import igrf_utils  # noqa: PLC0415 (path-dependent import, by design)

    return igrf_utils


def cross_check(model, committed: IgrfCoeffs, tolerance: float = 5e-9) -> None:
    """Verify pyIGRF14's ``IGRF14.SHC`` matches the committed IAGA table.

    Both are the same model published in two formats, so the main-field columns
    must agree exactly up to the decimal places the SHC file prints. The SHC's
    extra trailing snapshot must equal ``2025.0 + 5 * SV``.

    Parameters
    ----------
    model : igrf_utils.igrf
        Result of ``igrf_utils.load_shcfile``.
    committed : IgrfCoeffs
        Result of parsing ``tests/golden/igrf14coeffs.txt``.
    tolerance : float, optional
        Absolute tolerance in nT.

    Raises
    ------
    ValueError
        On any shape or value mismatch.
    """
    shc_epochs = np.asarray(model.time, dtype=float)
    shc_coeffs = np.asarray(model.coeffs, dtype=float)
    iaga_coeffs = np.array([row.values for row in committed.rows], dtype=float)
    iaga_sv = np.array([row.sv for row in committed.rows], dtype=float)

    n_epochs = len(committed.epochs)
    if shc_coeffs.shape[0] != iaga_coeffs.shape[0]:
        raise ValueError(
            f"coefficient count differs: SHC {shc_coeffs.shape[0]}, "
            f"IAGA {iaga_coeffs.shape[0]}"
        )
    if not np.array_equal(shc_epochs[:n_epochs], np.asarray(committed.epochs)):
        raise ValueError("epoch lists differ between the SHC file and the IAGA table")

    worst = float(np.max(np.abs(shc_coeffs[:, :n_epochs] - iaga_coeffs)))
    if worst > tolerance:
        raise ValueError(f"main-field coefficients differ by up to {worst:g} nT")

    extrapolated = iaga_coeffs[:, -1] + 5.0 * iaga_sv
    worst_sv = float(np.max(np.abs(shc_coeffs[:, n_epochs:].ravel() - extrapolated)))
    if worst_sv > tolerance:
        raise ValueError(
            f"SHC trailing snapshot is not 2025.0 + 5*SV (off by {worst_sv:g} nT)"
        )


def _coeffs_at(model, year: float) -> np.ndarray:
    """Linearly interpolate the Gauss coefficients to a decimal year."""
    time = np.asarray(model.time, dtype=float)
    coeffs = np.asarray(model.coeffs, dtype=float)
    if not time[0] <= year <= time[-1]:
        raise ValueError(f"{year} is outside IGRF-14 validity {time[0]}..{time[-1]}")
    return np.array([np.interp(year, time, row) for row in coeffs])


def synthesize(
    igrf_utils: ModuleType,
    model,
    years: tuple[float, ...] = SAMPLE_YEARS,
    points: tuple[tuple[float, float, float], ...] = SAMPLE_POINTS,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Evaluate every ``(point, year)`` pair; returns fixture rows in file order.

    Each row is ``(radius_km, colatitude_deg, longitude_deg, decimal_year,
    br_nt, btheta_nt, bphi_nt)``, the last three straight out of
    ``synth_values`` with no frame conversion.
    """
    nmax = model.parameters["nmax"]
    rows = []
    for year in years:
        coeffs = _coeffs_at(model, year)
        for radius, colatitude, longitude in points:
            br, btheta, bphi = igrf_utils.synth_values(
                coeffs.T, radius, colatitude, longitude, nmax
            )
            rows.append(
                (
                    radius,
                    colatitude,
                    longitude,
                    year,
                    float(br),
                    float(btheta),
                    float(bphi),
                )
            )
    return rows


def write_fixture(
    path: Path,
    rows: list[tuple[float, float, float, float, float, float, float]],
    *,
    coeffs_name: str,
    coeffs_sha256: str,
    shc_name: str,
    shc_sha256: str,
    nmax: int,
) -> None:
    """Write the CSV fixture, provenance header first."""
    lines = [
        "# Polaris IGRF-14 geocentric field golden fixture.",
        "#",
        "# DERIVED PRODUCT (design doc SS3.7): computed with pyIGRF14, the official",
        "# IAGA reference implementation, which is a fetch input and is NOT committed.",
        "# Regenerate with:",
        "#   PYTHONPATH=tools uv run --group igrf python -m igrf golden \\",
        "#       --pyigrf <extracted>/pyIGRF14 --out <this file>",
        "#",
        f"# pyigrf_url:     {PYIGRF_URL}",
        "# pyigrf_licence: MIT, Copyright (c) 2024 Ciaran Beggan (British Geological Survey)",
        f"# shc_file:       {shc_name}",
        f"# shc_sha256:     {shc_sha256}",
        "#",
        "# Coefficients cross-checked against the committed verbatim IAGA table:",
        f"# coeffs_file:    {coeffs_name}",
        f"# coeffs_sha256:  {coeffs_sha256}",
        "# coeffs_url:     https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt",
        "#",
        f"# Schmidt semi-normalised Gauss coefficients to degree {nmax}; IGRF reference",
        "# radius 6371.2 km. Columns are GEOCENTRIC spherical components in nT as",
        "# returned by igrf_utils.synth_values -- no geodetic conversion is applied.",
        "# radius is geocentric [km], colatitude = 90 - geocentric latitude [deg].",
        "#",
        "radius_km,colatitude_deg,longitude_deg,decimal_year,br_nt,btheta_nt,bphi_nt",
    ]
    # `repr` emits the shortest decimal that round-trips an IEEE-754 double
    # exactly, so reloading the fixture reproduces these values bit-for-bit
    # while the round inputs still read as 6371.2 rather than 6371.19999...
    lines.extend(",".join(repr(value) for value in row) for row in rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

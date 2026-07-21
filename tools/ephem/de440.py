"""Fit geocentric Sun/Moon Chebyshev segments from a JPL DE440 SPK kernel.

Ground-side only (design doc §11.3, §3.7). NAIF publishes DE440 as a binary SPK
(`.bsp`) covering 1550–2650 (`de440.bsp`, ~114 MB) or 1849–2150 (`de440s.bsp`,
~32 MB). Neither is committed: they are **fetch inputs**, exactly like the full
EGM2008 `.gfc`. What we commit is the derived *uploadable table* — the
Chebyshev coefficient sets the onboard `ephemeris::EphemerisTable` consumes
(REQ-CDH-002) — which §3.7 explicitly sanctions as a derivation computed *from*
the original rather than a substitute for it. The source URL is recorded in the
fixture's provenance block so it is reproducible by re-download.

**Geocentric, not barycentric.** DE440 stores Solar-System-barycentric positions
for the Sun and Earth-Moon barycenter, and EMB-relative positions for the Earth
and Moon. Polaris's environment models want positions relative to the *Earth*
(`math::frames::ECI`), so:

    r_sun_geocentric  = SSB->Sun  -  (SSB->EMB + EMB->Earth)
    r_moon_geocentric = EMB->Moon -  EMB->Earth

Fitting the geocentric difference directly — rather than fitting each leg and
subtracting onboard — is what keeps the onboard table small: the geocentric
vectors are far smoother than their barycentric parts, so the same accuracy needs
far fewer coefficients.

The fit is a least-squares Chebyshev approximation per interval, sampled well
above the coefficient count so the residual reported by ``fit_body`` is a real
accuracy estimate and not an interpolation artifact.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# NAIF generic-kernel endpoint. `de440s.bsp` ("short") covers 1849-2150, which
# spans any plausible mission epoch at a quarter the size of the full kernel.
DEFAULT_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
)

# NAIF body codes.
_SSB = 0
_EMB = 3
_SUN = 10
_MOON = 301
_EARTH = 399

# Julian Date of 1970-01-01T00:00:00, the uniform-scale epoch `time::Instant`
# counts from (constants::time::kJulianDate1970).
_JD_1970 = 2_440_587.5
_SECONDS_PER_DAY = 86_400.0

#: Highest coefficient index the flight container can hold
#: (`ephemeris::kMaxChebyshevDegree`). Fits must not exceed it.
MAX_DEGREE = 15


def fetch(url: str = DEFAULT_URL, out: Path | None = None) -> Path:
    """Download an SPK kernel. Ground-side only; never called from CI."""
    destination = out or Path(url.rsplit("/", 1)[-1])
    with urllib.request.urlopen(url, timeout=600) as resp:  # noqa: S310 (trusted NAIF host)
        destination.write_bytes(resp.read())
    return destination


@dataclass(frozen=True)
class Segment:
    """One fitted interval: the committed form of a `ChebyshevSegment`."""

    mid_ns: int
    """Interval midpoint, TDB nanoseconds since 1970-01-01T00:00:00."""
    radius_seconds: float
    """Interval half-width [s]."""
    degree: int
    """Highest coefficient index; ``degree + 1`` coefficients per component."""
    cx: list[float] = field(default_factory=list)
    cy: list[float] = field(default_factory=list)
    cz: list[float] = field(default_factory=list)


def _geocentric_km(kernel, body: int, jd: np.ndarray) -> np.ndarray:
    """Position of `body` relative to the Earth [km], shape (3, N).

    See the module docstring for why the difference is taken here rather than
    onboard. `jd` is a TDB Julian Date array.
    """
    if body == _MOON:
        # Both legs are already EMB-relative, so the barycentre cancels exactly
        # and never enters the arithmetic.
        return kernel[_EMB, _MOON].compute(jd) - kernel[_EMB, _EARTH].compute(jd)
    if body == _SUN:
        earth = kernel[_SSB, _EMB].compute(jd) + kernel[_EMB, _EARTH].compute(jd)
        return kernel[_SSB, _SUN].compute(jd) - earth
    raise ValueError(f"unsupported body code {body}")


def _fit_interval(
    kernel, body: int, jd_mid: float, radius_days: float, degree: int, samples: int
) -> tuple[Segment, float]:
    """Least-squares Chebyshev fit over one interval; returns (segment, max residual [m])."""
    # Sample on Chebyshev nodes: they minimize the worst-case error of the fit,
    # unlike uniform sampling which lets the endpoints drift (Runge).
    k = np.arange(samples)
    tau = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * samples))
    jd = jd_mid + tau * radius_days

    # (3, N) km -> metres. The design vector is T_n(tau) evaluated at the nodes.
    xyz = _geocentric_km(kernel, body, jd) * 1000.0

    design = np.polynomial.chebyshev.chebvander(tau, degree)
    coeffs, *_ = np.linalg.lstsq(design, xyz.T, rcond=None)  # (degree+1, 3)

    residual = float(np.max(np.abs(design @ coeffs - xyz.T)))

    mid_ns = int(round((jd_mid - _JD_1970) * _SECONDS_PER_DAY * 1e9))
    return (
        Segment(
            mid_ns=mid_ns,
            radius_seconds=radius_days * _SECONDS_PER_DAY,
            degree=degree,
            cx=coeffs[:, 0].tolist(),
            cy=coeffs[:, 1].tolist(),
            cz=coeffs[:, 2].tolist(),
        ),
        residual,
    )


def fit_body(
    kernel,
    body: int,
    jd_start: float,
    jd_end: float,
    interval_days: float,
    degree: int,
    samples: int = 64,
) -> tuple[list[Segment], float]:
    """Fit contiguous Chebyshev segments covering ``[jd_start, jd_end]``.

    Returns the segments and the worst residual [m] across all of them — the
    number to check before trusting a fixture, since a too-low ``degree`` or
    too-long ``interval_days`` shows up here and nowhere else.
    """
    if degree > MAX_DEGREE:
        raise ValueError(f"degree {degree} exceeds the flight container's {MAX_DEGREE}")

    radius_days = interval_days / 2.0
    segments: list[Segment] = []
    worst = 0.0

    jd_mid = jd_start + radius_days
    # Strict `<` would drop the interval containing jd_end whenever the span is
    # not an exact multiple of interval_days; the epsilon keeps coverage inclusive.
    while jd_mid - radius_days < jd_end - 1e-9:
        segment, residual = _fit_interval(
            kernel, body, jd_mid, radius_days, degree, samples
        )
        segments.append(segment)
        worst = max(worst, residual)
        jd_mid += interval_days

    return segments, worst

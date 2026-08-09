"""The §5.3 environmental disturbance-torque budget, analytic and worst-case.

Four terms, each in its standard closed form, each evaluated at the config's own
orbit and the config's own vehicle properties — areas, coefficients, CP-CM lever
arms and residual dipole all come from the same YAML the sim builds its truth
models from, so this budget and the plant's are the same vehicle.

The formulas
------------
**Gravity gradient** [wertz1978] §17.2, [markley2014] §3.4:

.. math:: \\tau_{gg} = \\frac{3\\mu}{2R^3}\\,\\lvert I_{\\max}-I_{\\min}\\rvert ,

the worst case over attitude — the :math:`\\sin 2\\theta` in
:math:`\\tfrac32 n^2 \\Delta I \\sin 2\\theta` reaches one at 45° off local
vertical. (Dropping the :math:`\\tfrac12` is a common and doubling error: the
maximum of :math:`\\sin 2\\theta` is 1, not 2.)

**Aerodynamic** [vallado2013] §8.6, [wertz2011] §19.2:

.. math:: \\tau_{aero} = \\tfrac12\\rho V^2 C_d A \\,\\lvert\\mathbf d_{cp}\\rvert ,

a flat plate at full broadside incidence with the CP-CM offset as the lever arm.

**Solar radiation pressure** [montenbruck2000] §3.4, [wertz2011] §19.2:

.. math:: \\tau_{srp} = \\frac{\\Phi}{c} A (1+q)\\,\\lvert\\mathbf d_{cp}\\rvert ,

with :math:`1+q` the config's ``srp_cr`` — the cannonball reflectivity
coefficient, 1 for a perfect absorber and 2 for a perfect specular reflector.

**Residual magnetic** [wertz1978] §17.1:

.. math:: \\tau_{mag} = \\lvert\\mathbf m_{res}\\rvert\\,\\lvert\\mathbf B\\rvert ,

taken at the **minimum** field magnitude for actuation sizing but at the
**maximum** here — a disturbance is worst where the field is strongest, and the
same field that makes the rods effective makes this term large.

Secular and cyclic, and why the split decides everything
--------------------------------------------------------
A **cyclic** torque stores momentum and gives it back within an orbit: it sizes
*storage*. A **secular** torque accumulates without bound: it sizes
*desaturation authority*, and no amount of wheel momentum fixes it. Which term
is which follows from the reference attitude, not from the torque, and that is
an assumption — see :class:`analysis.sizing.assumptions.SizingAssumptions`,
whose ``secular_fraction_*`` fields carry it and whose defaults are the standard
LVLH treatment. The budget below splits every term by those fractions rather
than hard-coding a decision the reader cannot see.

Fidelity, stated
----------------
These are **analytic worst-case magnitudes**, deliberately: a sizing budget wants
an upper bound in closed form, not a time history. The sim's §5.3 providers are
the truth models and they are more faithful in every term (real attitude, real
NRLMSIS density, conical eclipse, IGRF-14). Where a driver is close, the answer
is a simulation, not a bigger analytic margin.

Units and frames
----------------
SI: torques [N·m], densities [kg/m³], fields [T], lengths [m], areas [m²].
Lever arms and dipoles are body-frame vectors from the config; only their
magnitudes enter, which is what makes these worst-case rather than attitude-
dependent.

References
----------
Wertz, *Spacecraft Attitude Determination and Control*, §17 [wertz1978] — the
environmental-torque set and its magnitudes.
Wertz, Everett & Puschell, *The New SMAD*, §19.2 [wertz2011] — the sizing forms
of all four terms and the cyclic/secular distinction.
Vallado, *Fundamentals of Astrodynamics and Applications*, §8.6.2 and Table 8-4
[vallado2013] — the piecewise-exponential atmosphere parsed below.
Montenbruck & Gill, *Satellite Orbits*, §3.4 [montenbruck2000] — solar radiation
pressure.
Design doc §5.3 (disturbance torques), §12 (analysis tools).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.control.field import (
    circular_orbit_eci_m,
    field_eci_t,
    load_dipole,
)
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions

#: WGS84 Earth gravitational parameter [m³/s²].
EARTH_GM_M3_S2 = 3.986004418e14

#: Total solar irradiance at 1 AU [W/m²] and the speed of light [m/s] — the
#: modern TSI value, the same pair ``lib/constants`` carries for the flight and
#: sim sides. Their ratio is the radiation pressure, 4.54 µPa.
SOLAR_CONSTANT_W_M2 = 1361.0
SPEED_OF_LIGHT_M_S = 2.99792458e8

#: Mean equatorial Earth radius [m], for the geodetic-free altitude a circular
#: orbit's semi-major axis implies.
EARTH_RADIUS_M = 6378137.0

#: The flown piecewise-exponential atmosphere's band table, **parsed from the
#: sim implementation** rather than transcribed (``sim/world/atmosphere.cpp``,
#: Vallado Table 8-4). A transcription is a copy, and every copy is a place for
#: the analysis and the plant to disagree about the vehicle's own environment.
ATMOSPHERE_SOURCE = Path("sim/world/atmosphere.cpp")

_BAND_PATTERN = re.compile(
    r"\{\s*([0-9'.]+e?[-+]?[0-9]*)\s*,\s*([0-9'.]+e?[-+]?[0-9]*)\s*,"
    r"\s*([0-9'.]+e?[-+]?[0-9]*)\s*\}"
)


def exponential_density(
    altitude_m: float, source: str | Path = ATMOSPHERE_SOURCE
) -> float:
    """Neutral density [kg/m³] from the same band table the plant flies.

    :math:`\\rho = \\rho_0 \\exp[-(h - h_0)/H]` within the band containing
    @p altitude_m, with :math:`(h_0, \\rho_0, H)` read from
    ``sim/world/atmosphere.cpp``.

    **This is a static model and the honest caveat is large**: the real
    thermospheric density at 500 km swings more than an order of magnitude
    between solar minimum and maximum, which dwarfs every other uncertainty in
    the aerodynamic term. The sim's NRLMSIS 2.1 path resolves that; a closed-form
    sizing budget cannot, so the aero criterion is reported as
    assumption-dominated rather than as a measurement.

    Parameters
    ----------
    altitude_m : float
        Height above the mean equatorial radius [m].
    source : str or pathlib.Path, optional
        The C++ file the band table is parsed from.

    Returns
    -------
    float
        Density [kg/m³].

    Raises
    ------
    ValueError
        If the table cannot be parsed, or if the altitude is below its first
        band. A silently defaulted density would give a plausible aerodynamic
        torque with no provenance.
    """
    text = Path(source).read_text()
    bands = [
        tuple(float(g.replace("'", "")) for g in match.groups())
        for match in _BAND_PATTERN.finditer(text)
    ]
    bands = [b for b in bands if b[2] > 100.0]  # scale heights only; skip stray triples
    if not bands:
        raise ValueError(f"{source}: no exponential-atmosphere band table found")
    bands.sort()
    if altitude_m < bands[0][0]:
        raise ValueError(f"altitude {altitude_m} m is below the band table's floor")
    base_alt, base_density, scale_height = next(
        b for b in reversed(bands) if altitude_m >= b[0]
    )
    return float(base_density * np.exp(-(altitude_m - base_alt) / scale_height))


@dataclass(frozen=True)
class FieldStatistics:
    """Geomagnetic field magnitude over one orbit, from the tilted dipole.

    Sizing uses the **minimum**: a magnetorquer's worst moment is where the
    field is weakest, and an authority claim made at the mean is a claim the
    vehicle cannot honour for part of every lap. The maximum is what the
    residual-dipole *disturbance* is worst at, so both ends are carried.

    Attributes
    ----------
    min_t, mean_t, max_t : float
        Field magnitude over one orbit [T].
    samples : int
        Number of points the orbit was sampled at.
    """

    min_t: float
    mean_t: float
    max_t: float
    samples: int


def field_statistics(vehicle: Vehicle, samples: int = 721) -> FieldStatistics:
    """Field magnitude statistics over one orbit of the config's orbit.

    Reuses :mod:`analysis.control.field` — the degree-1 IGRF truncation parsed
    from the committed IAGA table — rather than a second field model. The
    truncation costs a few percent in magnitude at LEO, which is inside the
    30 % sizing margin and far inside the solar-cycle variation in the aero
    term; ``lib/environment/igrf`` is the authority and reaches Python when
    ``bindings/`` does.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies the orbit.
    samples : int, optional
        Points around the orbit.

    Returns
    -------
    FieldStatistics
    """
    times = np.linspace(0.0, vehicle.orbit.period_s, samples)
    positions = circular_orbit_eci_m(
        vehicle.orbit.sma_m,
        vehicle.orbit.inc_rad,
        vehicle.orbit.raan_rad,
        vehicle.orbit.mean_motion_rad_s,
        times,
    )
    magnitudes = np.linalg.norm(field_eci_t(load_dipole(), positions, times), axis=1)
    return FieldStatistics(
        min_t=float(np.min(magnitudes)),
        mean_t=float(np.mean(magnitudes)),
        max_t=float(np.max(magnitudes)),
        samples=samples,
    )


@dataclass(frozen=True)
class DisturbanceTerm:
    """One environmental torque, split into its secular and cyclic parts.

    Attributes
    ----------
    name : str
        Term name, e.g. ``"gravity gradient"``.
    torque_nm : float
        Worst-case magnitude [N·m].
    secular_fraction : float
        Fraction treated as secular [-]; the rest is cyclic.
    formula : str
        The closed form evaluated, for the report.
    formula_tex : str
        The same closed form as LaTeX, written here beside the ASCII rather than
        recovered from it downstream. Empty means the page sets the ASCII as
        plain text, deliberately and visibly.
    inputs : str
        The values it was evaluated at, for the report.
    """

    name: str
    torque_nm: float
    secular_fraction: float
    formula: str
    inputs: str
    formula_tex: str = ""

    @property
    def secular_nm(self) -> float:
        """Secular part [N·m] — what desaturation authority is sized against."""
        return self.torque_nm * self.secular_fraction

    @property
    def cyclic_nm(self) -> float:
        """Cyclic part [N·m] — what momentum storage is sized against."""
        return self.torque_nm * (1.0 - self.secular_fraction)


@dataclass(frozen=True)
class DisturbanceBudget:
    """The four §5.3 terms and their totals.

    Totals are **sums, not RSS**: the terms are not independent random variables
    but deterministic torques whose worst-case attitudes can coincide, and a
    sizing budget takes the coincidence. RSS would understate the total by ~30 %
    on a budget with two comparable terms.

    Attributes
    ----------
    terms : tuple of DisturbanceTerm
        In report order.
    field : FieldStatistics
        The field the magnetic term was evaluated in.
    density_kg_m3 : float
        The density the aerodynamic term was evaluated at.
    altitude_m : float
        The altitude both were evaluated at.
    """

    terms: tuple[DisturbanceTerm, ...]
    field: FieldStatistics
    density_kg_m3: float
    altitude_m: float

    @property
    def total_nm(self) -> float:
        """Worst-case total disturbance torque [N·m]."""
        return float(sum(t.torque_nm for t in self.terms))

    @property
    def secular_nm(self) -> float:
        """Total secular torque [N·m]."""
        return float(sum(t.secular_nm for t in self.terms))

    @property
    def cyclic_nm(self) -> float:
        """Total cyclic peak torque [N·m]."""
        return float(sum(t.cyclic_nm for t in self.terms))


def gravity_gradient_torque(vehicle: Vehicle) -> float:
    """Worst-case gravity-gradient torque [N·m].

    :math:`\\tau = \\tfrac{3\\mu}{2R^3}\\lvert I_{\\max}-I_{\\min}\\rvert`, at 45°
    off local vertical.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.

    Returns
    -------
    float
    """
    moments = vehicle.principal_moments_kgm2
    spread = float(np.max(moments) - np.min(moments))
    return 1.5 * EARTH_GM_M3_S2 / vehicle.orbit.sma_m**3 * spread


def aerodynamic_torque(vehicle: Vehicle, density_kg_m3: float) -> float:
    """Worst-case aerodynamic torque [N·m].

    :math:`\\tau = \\tfrac12\\rho V^2 C_d A \\lvert\\mathbf d_{cp}\\rvert`, with
    :math:`V` the circular orbital speed and the vehicle broadside to the ram.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    density_kg_m3 : float
        Neutral density at the orbit altitude [kg/m³].

    Returns
    -------
    float
    """
    speed = float(np.sqrt(EARTH_GM_M3_S2 / vehicle.orbit.sma_m))
    arm = float(np.linalg.norm(vehicle.cp_offset_aero_m))
    return 0.5 * density_kg_m3 * speed**2 * vehicle.drag_cd * vehicle.drag_area_m2 * arm


def srp_torque(vehicle: Vehicle) -> float:
    """Worst-case solar-radiation-pressure torque [N·m].

    :math:`\\tau = (\\Phi/c) A C_r \\lvert\\mathbf d_{cp}\\rvert`, at normal
    incidence and outside eclipse.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.

    Returns
    -------
    float
    """
    pressure = SOLAR_CONSTANT_W_M2 / SPEED_OF_LIGHT_M_S
    arm = float(np.linalg.norm(vehicle.cp_offset_srp_m))
    return pressure * vehicle.srp_area_m2 * vehicle.srp_cr * arm


def residual_dipole_torque(vehicle: Vehicle, field_t: float) -> float:
    """Residual-magnetic torque :math:`\\lvert\\mathbf m_{res}\\rvert
    \\lvert\\mathbf B\\rvert` [N·m].

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    field_t : float
        Field magnitude [T]; the orbit **maximum** for a worst-case disturbance.

    Returns
    -------
    float
    """
    return float(np.linalg.norm(vehicle.residual_dipole_am2)) * field_t


def disturbance_budget(
    vehicle: Vehicle, assumptions: SizingAssumptions | None = None
) -> DisturbanceBudget:
    """The full four-term budget at the config's orbit.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    assumptions : SizingAssumptions, optional
        Supplies the secular fractions; defaults to the LVLH treatment.

    Returns
    -------
    DisturbanceBudget
    """
    assumptions = assumptions or SizingAssumptions()
    altitude = vehicle.orbit.sma_m - EARTH_RADIUS_M
    density = exponential_density(altitude)
    field = field_statistics(vehicle)
    moments = vehicle.principal_moments_kgm2

    terms = (
        DisturbanceTerm(
            name="gravity gradient",
            torque_nm=gravity_gradient_torque(vehicle),
            secular_fraction=assumptions.secular_fraction_gg,
            formula="3*mu/(2*R^3) * |I_max - I_min|",
            formula_tex=(
                r"\tau_{gg} = \frac{3\mu}{2R^{3}}" r"\left|I_{\max}-I_{\min}\right|"
            ),
            inputs=(
                f"R = {vehicle.orbit.sma_m / 1000.0:.1f} km, "
                f"dI = {np.max(moments) - np.min(moments):.4g} kg.m^2"
            ),
        ),
        DisturbanceTerm(
            name="aerodynamic",
            torque_nm=aerodynamic_torque(vehicle, density),
            secular_fraction=assumptions.secular_fraction_aero,
            formula="0.5 * rho * V^2 * Cd * A * |d_cp|",
            formula_tex=(
                r"\tau_{a} = \tfrac{1}{2}\,\rho\,V^{2}\,C_d\,A\," r"\left|d_{cp}\right|"
            ),
            inputs=(
                f"rho = {density:.3g} kg/m^3 at {altitude / 1000.0:.0f} km, "
                f"Cd = {vehicle.drag_cd:g}, A = {vehicle.drag_area_m2:g} m^2, "
                f"|d| = {np.linalg.norm(vehicle.cp_offset_aero_m):.4g} m"
            ),
        ),
        DisturbanceTerm(
            name="solar radiation pressure",
            torque_nm=srp_torque(vehicle),
            secular_fraction=assumptions.secular_fraction_srp,
            formula="(Phi/c) * A * Cr * |d_cp|",
            formula_tex=(
                r"\tau_{srp} = \frac{\Phi}{c}\,A\,C_r\," r"\left|d_{cp}\right|"
            ),
            inputs=(
                f"Phi/c = {SOLAR_CONSTANT_W_M2 / SPEED_OF_LIGHT_M_S:.3g} Pa, "
                f"Cr = {vehicle.srp_cr:g}, A = {vehicle.srp_area_m2:g} m^2, "
                f"|d| = {np.linalg.norm(vehicle.cp_offset_srp_m):.4g} m"
            ),
        ),
        DisturbanceTerm(
            name="residual magnetic",
            torque_nm=residual_dipole_torque(vehicle, field.max_t),
            secular_fraction=assumptions.secular_fraction_mag,
            formula="|m_res| * |B|_max",
            formula_tex=(r"\tau_{m} = \left|m_{\mathrm{res}}\right|\,|B|_{\max}"),
            inputs=(
                f"|m_res| = {np.linalg.norm(vehicle.residual_dipole_am2):.4g} A.m^2, "
                f"|B|_max = {field.max_t * 1e6:.1f} uT"
            ),
        ),
    )
    return DisturbanceBudget(
        terms=terms,
        field=field,
        density_kg_m3=density,
        altitude_m=altitude,
    )

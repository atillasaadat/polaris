"""ADCS actuator sizing and design validation (§7, §8.5, §12).

Three things in one package, for a **user-supplied** vehicle config:

1. **Size the actuators.** The exact achievable set of a wheel array or rod set
   — the zonotope, its inscribed and circumscribed radii, and the L2 ellipsoid
   inscribed in it (:mod:`analysis.sizing.envelope`). Sizing is done against the
   inscribed radius, the capability guaranteed in *every* direction, never
   against the best-direction figure.
2. **Validate the design against its drivers, with 30 % margin.** Tip-off
   absorption, cyclic storage, secular accumulation and slew agility for the
   wheels (:mod:`analysis.sizing.wheels`); desaturation authority, detumble
   authority and the B-dot measurement floor for the rods
   (:mod:`analysis.sizing.magnetorquers`) — all against the §5.3 analytic
   disturbance-torque budget evaluated at the config's own orbit
   (:mod:`analysis.sizing.disturbances`).
3. **Derive and justify the flight tuning the design implies.** PID gains, the
   momentum envelope, the desaturation hysteresis, the detumble exit threshold
   and the B-dot gain — each returned with its formula, its inputs and the
   reasoning (:mod:`analysis.sizing.parameters`), because a recommended number
   without its argument is not a recommendation.

This is the package ``analysis/CLAUDE.md`` listed as ``momentum/``; it is broader
than momentum budgeting, so it ships under a name that says what it does.

Reuse, and the boundary it respects
-----------------------------------
The vehicle comes from :func:`analysis.control.vehicle.load_vehicle` — the same
loader :mod:`analysis.control` uses, extended with the mass properties, lever
arms and thresholds sizing needs, rather than a second parser. The SISO
momentum boundary comes from :func:`analysis.control.plant.siso_coupling` and
the loop crossover from :func:`analysis.control.margins.axis_margins`; the
geomagnetic field from :mod:`analysis.control.field`; the atmosphere band table
is parsed from the plant's own ``sim/world/atmosphere.cpp``. Nothing is
transcribed.

Like :mod:`analysis.control`, this package computes on **matrices and scalars
read from config** — no quaternion, no frame transform, no propagation, no
filter update. That is the stated boundary of the ``control/`` exception in
``analysis/CLAUDE.md``, and it applies here unchanged: the moment a tool in this
package needs a rotation or a trajectory, it needs ``bindings/``.

Reusability
-----------
No spacecraft number appears anywhere in this package. Anything the config does
not carry — tip-off rate, desaturation interval, the secular/cyclic split — is a
field of :class:`analysis.sizing.assumptions.SizingAssumptions` with a
documented default, and every one of them is rendered into the report's
assumptions block. See ``analysis/sizing/README.md``.

Run the gate::

    PYTHONPATH=tools uv run --group analysis python -m analysis.sizing \\
        config/spacecraft/leo_smallsat.yaml
"""

from analysis.sizing.assumptions import (
    MARGIN,
    MAX_OVERSIZING,
    MTQ_ORBIT_EFFICIENCY,
    SizingAssumptions,
)
from analysis.sizing.disturbances import (
    DisturbanceBudget,
    DisturbanceTerm,
    FieldStatistics,
    aerodynamic_torque,
    disturbance_budget,
    exponential_density,
    field_statistics,
    gravity_gradient_torque,
    residual_dipole_torque,
    srp_torque,
)
from analysis.sizing.envelope import (
    DegenerateArrayError,
    Envelope,
    check_axes,
    envelope,
    fibonacci_sphere,
    sampled_inscribed,
    support,
)
from analysis.sizing.magnetorquers import (
    BdotNoiseFloor,
    MtqSizing,
    bdot_noise_floor,
    mtq_sizing,
)
from analysis.sizing.parameters import (
    DerivedParameter,
    derived_parameters,
    implied_bandwidth,
)
from analysis.sizing.report import (
    SizingAnalysis,
    format_budget,
    format_derived,
    sizing_analysis,
    sizing_report,
)
from analysis.sizing.wheels import MomentumDriver, WheelSizing, wheel_sizing

__all__ = [
    "MARGIN",
    "MAX_OVERSIZING",
    "MTQ_ORBIT_EFFICIENCY",
    "BdotNoiseFloor",
    "DegenerateArrayError",
    "DerivedParameter",
    "DisturbanceBudget",
    "DisturbanceTerm",
    "Envelope",
    "FieldStatistics",
    "MomentumDriver",
    "MtqSizing",
    "SizingAnalysis",
    "SizingAssumptions",
    "WheelSizing",
    "aerodynamic_torque",
    "bdot_noise_floor",
    "check_axes",
    "derived_parameters",
    "disturbance_budget",
    "envelope",
    "exponential_density",
    "fibonacci_sphere",
    "field_statistics",
    "format_budget",
    "format_derived",
    "gravity_gradient_torque",
    "implied_bandwidth",
    "mtq_sizing",
    "residual_dipole_torque",
    "sampled_inscribed",
    "sizing_analysis",
    "sizing_report",
    "srp_torque",
    "support",
    "wheel_sizing",
]

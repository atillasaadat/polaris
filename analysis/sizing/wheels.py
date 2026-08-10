"""Reaction-wheel sizing: the momentum and torque a design must supply.

Four momentum drivers and one torque driver, each compared against the wheel
array's **inscribed** envelope radius (:mod:`analysis.sizing.envelope`) — the
capability guaranteed in every direction, never the best-direction figure.

The four momentum drivers
-------------------------
**D1 — tip-off absorption.** :math:`h = \\lvert J\\boldsymbol\\omega\\rvert` at
the separation rate. Reported twice: at the raw tip-off, and at the rate B-dot
hands over at (``DetumbleExitRadps``). Which one the design is *sized on* is a
concept-of-operations decision — a vehicle whose wheels catch the raw tip-off
needs no detumble phase at all — so both appear and the report says which
criterion is which.

**D2 — cyclic storage.** A disturbance oscillating at orbit rate stores momentum
for half a cycle and gives it back. The standard sizing result [wertz2011] §19.2
is

.. math:: h_{cyc} = 0.707\\,\\tau_{cyc}\\,\\frac{T_{orbit}}{4},

the quarter-period integral with the 0.707 peak-to-RMS factor for a sinusoid.
(The exact quarter-period integral of :math:`\\tau\\sin nt` is
:math:`\\tau/n = 0.159\\,\\tau T`; the SMAD form is :math:`0.177\\,\\tau T`, i.e.
11 % conservative. The conservative one is used.)

**D3 — secular accumulation.** :math:`h = \\tau_{sec}T_{desat}`. Unbounded by
construction: this is the driver that no wheel can solve, only desaturation can,
which is why the same :math:`\\tau_{sec}` reappears as the magnetorquer's M1
criterion.

**D4 — slew agility.** :math:`h = \\lvert J\\boldsymbol\\omega_{slew}\\rvert`.
Judged only when the config declares a slew rate; otherwise the slew rate the
design *supports* is reported as a diagnostic with no verdict word, because a
PASS that cannot fail is not a verdict.

Two capabilities, and the gap between them
------------------------------------------
The wheels' **hardware** capability is the zonotope's inscribed radius. The
vehicle's **usable** capability is the smaller of that and
``MomentumEnvelopeNms`` — the flight momentum-management ceiling, which on a
vehicle whose margins come from a per-axis SISO analysis is set by that
analysis's validity boundary and not by the hardware (design doc §8.5). Every
momentum criterion is evaluated against the **usable** figure, because momentum
the vehicle will raise an envelope event over is momentum it does not have. The
hardware figure is reported beside it, and where the two differ by a large
factor that difference *is* the finding.

Units and frames
----------------
Momentum [N·m·s], torque [N·m], rates [rad/s], inertia [kg·m²], all body frame.

References
----------
Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*, §7.3 [markley2014] — wheel arrays and momentum envelopes.
Wertz, Everett & Puschell, *The New SMAD*, §19.2 [wertz2011] — the four sizing
drivers and the cyclic-storage form.
Sidi, *Spacecraft Dynamics and Control*, §7.3 [sidi1997] — momentum storage
against secular and cyclic disturbance.
Design doc §7 (actuators), §8.5 (control and the SISO envelope), §12 (analysis tools).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.common.report import Criterion
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import DisturbanceBudget
from analysis.sizing.envelope import Envelope, envelope

#: Peak-to-RMS factor for a sinusoidal cyclic disturbance [-] ([wertz2011]).
CYCLIC_RMS_FACTOR = 0.707


@dataclass(frozen=True)
class MomentumDriver:
    """One required-momentum driver.

    Attributes
    ----------
    name : str
        Driver name, e.g. ``"D1 tip-off absorption"``.
    required_nms : float
        Momentum the design must be able to hold [N·m·s].
    formula : str
        The closed form, for the report.
    formula_tex : str
        The same closed form as LaTeX, written here beside the ASCII rather than
        recovered from it downstream. Empty means the page sets the ASCII as
        plain text, deliberately and visibly.
    inputs : str
        The values it was evaluated at.
    inputs_tex : str
        The evaluated inputs as LaTeX, where setting them as mathematics adds
        something. Empty is the ordinary case: a list of numbers reads fine.
    judged : bool
        Whether a criterion is written on it. ``False`` for a driver the config
        does not declare an input for — it is reported, not judged.
    """

    name: str
    required_nms: float
    formula: str
    inputs: str
    formula_tex: str = ""
    inputs_tex: str = ""
    judged: bool = True


@dataclass(frozen=True)
class WheelSizing:
    """The wheel array's capability and the drivers it is judged against.

    Attributes
    ----------
    momentum : Envelope
        Momentum envelope at the per-wheel capacity [N·m·s].
    torque : Envelope
        Torque envelope at the per-wheel torque limit [N·m].
    usable_momentum_nms : float
        ``min(momentum.inscribed, MomentumEnvelopeNms)`` [N·m·s] — the momentum
        the vehicle may actually use.
    envelope_limited : bool
        True when the flight envelope, not the hardware, is the binding limit.
    drivers : tuple of MomentumDriver
        The momentum drivers, in report order.
    required_torque_nm : float
        Body torque the array must deliver in every direction [N·m].
    commanded_torque_nm : float
        The flight parameter ``WheelMaxTorqueNm`` [N·m] — what the allocator is
        allowed to ask one wheel for, and what the whole torque envelope above
        is built on.
    catalog_torque_nm : float
        The installed wheel's catalog ``max_torque_nm`` [N·m] — what it can
        actually deliver. Kept separate from ``commanded_torque_nm`` because
        the pair is the criterion.
    supported_slew_radps : float
        The slew rate the usable momentum supports about the worst body axis
        [rad/s] — a diagnostic, always reported.
    """

    momentum: Envelope
    torque: Envelope
    usable_momentum_nms: float
    envelope_limited: bool
    drivers: tuple[MomentumDriver, ...]
    required_torque_nm: float
    commanded_torque_nm: float
    catalog_torque_nm: float
    supported_slew_radps: float

    @property
    def largest_driver(self) -> MomentumDriver:
        """The judged driver demanding the most momentum."""
        return max((d for d in self.drivers if d.judged), key=lambda d: d.required_nms)

    @property
    def largest_demand(self) -> "MomentumDriver":
        """The largest driver of any kind, judged or merely reported.

        Oversizing asks about the *unit class* — is this the wrong wheel for
        this vehicle? — so it weighs every real demand, including one that is
        reported rather than gated. Using only the judged set would call a
        wheel oversized because the criterion that justified its size stopped
        gating (D1, once the rods are shown to remove the tip-off), which is an
        artefact of the verdict logic rather than a fact about the hardware.
        """
        return max(self.drivers, key=lambda d: d.required_nms)

    @property
    def oversizing(self) -> float:
        """Hardware momentum capability over the largest demand of any kind [-]."""
        return self.momentum.inscribed / self.largest_demand.required_nms


def wheel_sizing(
    vehicle: Vehicle,
    budget: DisturbanceBudget,
    assumptions: SizingAssumptions | None = None,
    magnetic_detumble: bool = False,
) -> WheelSizing:
    """Compute the wheel array's envelopes and its required momentum.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    budget : DisturbanceBudget
        The disturbance budget feeding drivers D2 and D3.
    assumptions : SizingAssumptions, optional
        Defaults to :class:`~analysis.sizing.assumptions.SizingAssumptions`.

    Returns
    -------
    WheelSizing

    Raises
    ------
    analysis.sizing.envelope.DegenerateArrayError
        If the wheel axes do not span three dimensions.
    """
    assumptions = assumptions or SizingAssumptions()
    moments = vehicle.principal_moments_kgm2
    worst_inertia = float(np.max(moments))
    period = vehicle.orbit.period_s

    momentum = envelope(vehicle.wheel_spin_axes, vehicle.wheel_max_momentum_nms)
    torque = envelope(vehicle.wheel_spin_axes, vehicle.wheel_max_torque_nm)
    usable = min(momentum.inscribed, vehicle.momentum_envelope_nms)

    desat_interval = assumptions.desat_interval(period)
    drivers = [
        MomentumDriver(
            name="D1 tip-off absorption",
            required_nms=worst_inertia * assumptions.tipoff_rate_radps,
            formula="|J * omega_tipoff|",
            formula_tex=r"h_{D1} = \left|J\,\omega_{\mathrm{tipoff}}\right|",
            inputs=(
                f"J_max = {worst_inertia:g} kg.m^2, "
                f"omega = {np.degrees(assumptions.tipoff_rate_radps):.3g} deg/s"
                + (
                    "; reported only — the rods remove the tip-off (M2), so the "
                    "binding wheel requirement is the D1b handover"
                    if magnetic_detumble
                    else ""
                )
            ),
            # Absorbing the raw tip-off on the wheels alone is a requirement only
            # for a vehicle that cannot detumble magnetically. Where M2 passes,
            # the CONOPS is rods-then-wheels and D1b is the handover that binds;
            # judging D1 as well would size the wheels for a mode the vehicle
            # does not fly. It stays in the table either way, because "what if
            # the rods are lost" is a real question and the number answers it.
            judged=not magnetic_detumble,
        ),
        MomentumDriver(
            name="D1b post-B-dot handover",
            required_nms=worst_inertia * vehicle.detumble_exit_radps,
            formula="|J * DetumbleExitRadps|",
            formula_tex=r"h_{D1b} = \left|J\,\omega_{\mathrm{exit}}\right|",
            inputs=(
                f"J_max = {worst_inertia:g} kg.m^2, "
                f"omega = {np.degrees(vehicle.detumble_exit_radps):.3g} deg/s"
            ),
        ),
        MomentumDriver(
            name="D2 cyclic storage",
            required_nms=CYCLIC_RMS_FACTOR * budget.cyclic_nm * period / 4.0,
            formula="0.707 * tau_cyclic * T_orbit / 4",
            formula_tex=r"h_{D2} = 0.707\,\tau_{\mathrm{cyc}}\,\frac{T_{\mathrm{orbit}}}{4}",
            inputs=(f"tau_cyclic = {budget.cyclic_nm:.3g} N.m, T = {period:.0f} s"),
        ),
        MomentumDriver(
            name="D3 secular accumulation",
            required_nms=budget.secular_nm * desat_interval,
            formula="tau_secular * T_desat",
            formula_tex=r"h_{D3} = \tau_{\mathrm{sec}}\,T_{\mathrm{desat}}",
            inputs=(
                f"tau_secular = {budget.secular_nm:.3g} N.m, "
                f"T_desat = {desat_interval:.0f} s"
            ),
        ),
        MomentumDriver(
            name="D4 slew agility",
            required_nms=worst_inertia * (assumptions.slew_rate_radps or 0.0),
            formula="|J * omega_slew|",
            formula_tex=r"h_{D4} = \left|J\,\omega_{\mathrm{slew}}\right|",
            inputs=(
                "no commanded slew rate in the config"
                if assumptions.slew_rate_radps is None
                else f"omega_slew = {np.degrees(assumptions.slew_rate_radps):.3g} deg/s"
            ),
            judged=assumptions.slew_rate_radps is not None,
        ),
    ]

    return WheelSizing(
        momentum=momentum,
        torque=torque,
        usable_momentum_nms=usable,
        envelope_limited=vehicle.momentum_envelope_nms < momentum.inscribed,
        drivers=tuple(drivers),
        required_torque_nm=vehicle.pid.max_torque_nm + budget.total_nm,
        commanded_torque_nm=vehicle.wheel_max_torque_nm,
        catalog_torque_nm=vehicle.wheel_catalog_torque_nm,
        supported_slew_radps=usable / worst_inertia,
    )


def criteria(
    sizing: WheelSizing, assumptions: SizingAssumptions | None = None
) -> list[Criterion]:
    """Pass/fail criteria for the wheel design.

    One criterion per judged momentum driver against the **usable** envelope,
    one for the tip-off driver against the **hardware** envelope (so a design
    whose only problem is its certified ceiling is distinguishable from one whose
    wheels are genuinely too small), the torque criterion, the commanded-torque
    consistency check, and the oversizing check.

    The commanded-torque check, and why only one side of it fails
    -------------------------------------------------------------
    ``WheelMaxTorqueNm`` is a flight parameter; ``max_torque_nm`` is a property
    of the unit bolted to the deck. Nothing links them, so swapping the wheel
    and leaving the parameter behind is a silent change — and on this vehicle it
    happened, leaving the FSW authorised to command 12.5× what the new wheel can
    produce. The criterion is therefore ``commanded ≤ catalog``.

    A commanded limit *below* the catalog value is legitimate — derating for
    thermal margin, bearing life or a stability argument is a real design choice,
    and one the analysis has no basis to overrule — so it is not a failure. A
    *large* gap is still a finding, because the wheel is then paying mass and
    power for authority the vehicle never uses, so
    :func:`analysis.sizing.report.sizing_report` raises it as a report warning.
    Warnings qualify a report; they do not fail it (``analysis/CLAUDE.md``).

    **No requirement ID is attached to any of these**, deliberately. The
    repo's requirement baseline has nothing written on actuator *sizing*:
    REQ-ACTL-009 states that the FSW must compute stored momentum and alarm on
    the envelope, which is a behaviour, not a sizing bound. Borrowing that ID
    here would claim verification evidence for a requirement these criteria do
    not test. :func:`analysis.sizing.report.sizing_report` raises the gap as a
    report warning instead.

    Parameters
    ----------
    sizing : WheelSizing
        The computed capability and drivers.
    assumptions : SizingAssumptions, optional
        Supplies the margin factor and the oversizing ceiling.

    Returns
    -------
    list of analysis.common.report.Criterion
    """
    assumptions = assumptions or SizingAssumptions()
    margin = assumptions.margin
    out: list[Criterion] = []

    for driver in sizing.drivers:
        if not driver.judged:
            continue
        out.append(
            Criterion(
                name=f"usable momentum vs {driver.name}",
                requirement="",
                threshold=margin * driver.required_nms,
                measured=sizing.usable_momentum_nms,
                units="N.m.s",
                sense="min",
                note=(
                    f"{driver.formula} = {driver.required_nms:.3g} N.m.s "
                    f"({driver.inputs}); capability is min(zonotope r_in, "
                    "MomentumEnvelopeNms)"
                ),
                # The note opens with the driver's own formula, so it carries the
                # driver's own LaTeX with it. Nothing downstream re-derives it.
                formula=driver.formula,
                formula_tex=driver.formula_tex,
            )
        )

    tipoff = sizing.drivers[0]
    out.append(
        Criterion(
            name="hardware momentum vs D1 tip-off absorption",
            requirement="",
            threshold=margin * tipoff.required_nms,
            measured=sizing.momentum.inscribed,
            units="N.m.s",
            sense="min",
            note=(
                "the wheels themselves, ignoring the flight envelope: this "
                "passing while the usable criterion fails means the ceiling is "
                "the certified linear regime, not the hardware"
            ),
        )
    )
    out.append(
        Criterion(
            name="wheel torque, guaranteed in every direction",
            requirement="",
            threshold=margin * sizing.required_torque_nm,
            measured=sizing.torque.inscribed,
            units="N.m",
            sense="min",
            note=(
                f"demand = PidMaxTorqueNm + total disturbance = "
                f"{sizing.required_torque_nm:.3g} N.m; per-body-axis reach is "
                f"{sizing.torque.per_body_axis[0]:.3g} N.m, which is the number "
                "an axis-only check would have used"
            ),
        )
    )
    out.append(
        Criterion(
            name="commanded wheel torque within hardware capability",
            requirement="",
            threshold=sizing.catalog_torque_nm,
            measured=sizing.commanded_torque_nm,
            units="N.m",
            sense="max",
            note=(
                "A margin of 0% is the intended state for this row: equality "
                "means the flight parameter commands exactly the wheel that is "
                "installed, so the 30% sizing convention does not apply here. "
                f"WheelMaxTorqueNm = {sizing.commanded_torque_nm:.3g} N.m against "
                f"the catalog max_torque_nm = {sizing.catalog_torque_nm:.3g} N.m of "
                "the installed unit. Commanding past the catalog value is not "
                "conservative in the safe direction: the wheel simply does not "
                "deliver it, so the allocator's authority assumption is wrong and "
                "every torque margin computed on it — including the guaranteed-"
                "radius criterion above, which is built from this same parameter "
                "— is optimistic by the same factor. Commanding below it is a "
                "derate: legitimate, and not a failure, but the vehicle then "
                "carries the mass and power of authority it never uses, which is "
                "raised as a warning past 2x"
            ),
        )
    )
    out.append(
        Criterion(
            name="wheel momentum oversizing factor",
            requirement="",
            threshold=assumptions.max_oversizing,
            measured=sizing.oversizing,
            units="x",
            sense="max",
            note=(
                f"hardware r_in {sizing.momentum.inscribed:.3g} N.m.s over the "
                f"largest demand ({sizing.largest_demand.name}, "
                f"{sizing.largest_demand.required_nms:.3g} N.m.s)"
            ),
        )
    )
    return out

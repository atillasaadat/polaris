"""The flight tuning that follows from a design — each with its justification.

Sizing answers "is this hardware big enough". This module answers the question
that comes next and is usually left to folklore: **given this hardware, what
should the flight parameters be, and why?** Every entry returns the formula, the
inputs it was evaluated at, the reasoning, and the committed value it is being
compared against, so the output is an argument a reviewer can check rather than
a number to be trusted.

What is derived here
--------------------
* **PID gains** — :math:`K_p = J\\omega_n^2`, :math:`K_d = 2\\zeta\\omega_n J`
  for the design bandwidth and damping ([wie2008] §7.3, the standard
  second-order placement on a rigid double integrator). The committed pair
  implies an :math:`(\\omega_n, \\zeta)` and the two checks written on them are
  the ones that are *not* circular: the bandwidth must sit well below the sample
  rate, and the damping the two gains jointly imply must be sane. Whether the
  bandwidth should be higher is a design choice about noise and actuator
  authority, not something a formula settles, so no criterion pretends to.
* **Momentum envelope** — from the SISO validity boundary, **reused** from
  :func:`analysis.control.plant.siso_coupling` rather than re-derived. The bound
  is the stored momentum at which the gyroscopic coupling stops being negligible
  against the control torque at the loop crossover (design doc §8.5).
* **Desaturation thresholds** — the ordering invariant
  :math:`\\text{exit} < \\text{enter} < \\text{envelope}`: the vehicle must act
  before it alarms, and stop below where it started, or the action chatters at
  the threshold.
* **Detumble exit threshold** — momentum-based against the usable wheel
  envelope, and then **cross-checked against the B-dot noise floor**. A
  recommendation below the floor is refused and the floor is recommended
  instead, with the reason stated: a threshold the magnetometer cannot resolve
  declares completion on noise.
* **B-dot gain** — Avanzini & Giulietti's convergence floor
  :math:`k \\ge 2\\omega_o(1+\\sin\\xi)J_{\\min}` [avanzini2012], at the
  worst-case :math:`\\xi`.

Units
-----
SI in every field; the report converts at its own boundary.

References
----------
Wie, *Space Vehicle Dynamics and Control*, 2nd ed., §7.3 [wie2008] — the
second-order placement the PID gains come from.
Avanzini & Giulietti [avanzini2012] — the B-dot gain floor.
Franklin, Powell & Workman, *Digital Control of Dynamic Systems*, §3 and §4
[franklin1998] — the sampling-rate rule of thumb the bandwidth ceiling is.
Camillo & Markley [camillo1980] — magnetic unloading, the desaturation law the
thresholds gate.
Design doc §8.5 (control), §12 (analysis tools), §19.3 (the config pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.common.report import Criterion
from analysis.control.margins import axis_margins
from analysis.control.plant import siso_coupling
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.magnetorquers import MtqSizing
from analysis.sizing.wheels import WheelSizing

#: Largest closed-loop natural frequency, as a fraction of the sample rate
#: :math:`\omega_s = 2\pi/T_s` [-]. The classical digital-control rule of thumb
#: is 10–20 samples per closed-loop period ([franklin1998] §3.4); one tenth is
#: the loose end of it and is what this checks against, since the sampled-data
#: margins in :mod:`analysis.control.margins` are the authoritative check and
#: this is the sanity bound.
MAX_BANDWIDTH_FRACTION = 0.1

#: Smallest closed-loop damping ratio a pointing design should ship [-]. Below
#: it the step response overshoots more than ~16 % and the settling time stops
#: improving with bandwidth. Standard practice, not fitted to any measurement.
MIN_DAMPING = 0.5


@dataclass(frozen=True)
class DerivedParameter:
    """One flight parameter, derived, justified and compared.

    Attributes
    ----------
    name : str
        The FSW parameter name, e.g. ``"PidKpNmPerRad"``.
    derived : float
        The value this analysis recommends, in :attr:`units`.
    committed : float
        The value the config currently carries. ``nan`` when the parameter is a
        derived quantity with no config entry.
    units : str
        Display units.
    formula : str
        The closed form used.
    inputs : str
        The values it was evaluated at.
    reasoning : str
        Why this is the right form and what it trades — the part that makes the
        number an argument rather than an assertion.
    """

    name: str
    derived: float
    committed: float
    units: str
    formula: str
    inputs: str
    reasoning: str

    @property
    def ratio(self) -> float:
        """Committed over derived [-]; ``nan`` when either is unavailable."""
        if self.derived == 0.0 or np.isnan(self.committed):
            return float("nan")
        return self.committed / self.derived


def implied_bandwidth(vehicle: Vehicle) -> tuple[float, float]:
    """The :math:`(\\omega_n, \\zeta)` the committed PID gains imply.

    Inverting :math:`K_p = J\\omega_n^2` and :math:`K_d = 2\\zeta\\omega_n J` on
    the **mean** principal moment, which is the tensor the shipped scalar gains
    were placed against (the config says so; a per-axis set would need three
    gain pairs and the flight law carries one).

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.

    Returns
    -------
    tuple of float
        ``(omega_n [rad/s], zeta [-])``.
    """
    inertia = float(np.mean(vehicle.principal_moments_kgm2))
    wn = float(np.sqrt(vehicle.pid.kp_nm_per_rad / inertia))
    zeta = (
        float(vehicle.pid.kd_nm_per_radps / (2.0 * wn * inertia)) if wn > 0.0 else 0.0
    )
    return wn, zeta


def siso_momentum_bound(vehicle: Vehicle) -> tuple[float, float]:
    """The SISO validity bound and the crossover it was taken at.

    Reuses :func:`analysis.control.margins.axis_margins` and
    :func:`analysis.control.plant.siso_coupling` — one implementation of the
    boundary, shared with the package that owns it.

    Returns ``(nan, nan)`` rather than a plausible number when the loop has no
    findable gain crossover, which is what a badly tuned candidate design does.
    A criterion measured at ``nan`` fails, which is the conservative answer: the
    bound is unknown, so the envelope cannot be certified against it.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.

    Returns
    -------
    tuple of float
        ``(momentum_bound [N.m.s], crossover [rad/s])``.
    """
    crossover = axis_margins(vehicle, 0).gain_crossover_rad_s
    if not np.isfinite(crossover) or crossover <= 0.0:
        return float("nan"), float("nan")
    _, bound = siso_coupling(vehicle, crossover)
    return (bound if bound > 0.0 else float("nan")), crossover


def _settling_time_s(wn: float, zeta: float) -> float:
    """2 % settling time :math:`4/(\\zeta\\omega_n)` [s]; ``inf`` when undamped."""
    product = zeta * wn
    return 4.0 / product if product > 0.0 else float("inf")


def derived_parameters(
    vehicle: Vehicle,
    wheels: WheelSizing,
    mtq: MtqSizing,
    assumptions: SizingAssumptions | None = None,
) -> tuple[DerivedParameter, ...]:
    """Every parameter this design determines, with its justification.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    wheels : WheelSizing
        Supplies the usable momentum envelope.
    mtq : MtqSizing
        Supplies the B-dot noise floor.
    assumptions : SizingAssumptions, optional
        Supplies the detumble handover fraction.

    Returns
    -------
    tuple of DerivedParameter
    """
    assumptions = assumptions or SizingAssumptions()
    inertia_mean = float(np.mean(vehicle.principal_moments_kgm2))
    inertia_min = float(np.min(vehicle.principal_moments_kgm2))
    inertia_max = float(np.max(vehicle.principal_moments_kgm2))
    wn, zeta = implied_bandwidth(vehicle)
    sample_rate = 2.0 * np.pi / vehicle.control_period_s

    siso_limit, crossover = siso_momentum_bound(vehicle)

    momentum_exit = (
        assumptions.detumble_exit_fraction * wheels.usable_momentum_nms / inertia_max
    )
    floor = mtq.noise_floor.rate_worst_radps
    recommended_exit = max(momentum_exit, floor)

    return (
        DerivedParameter(
            name="PidKpNmPerRad",
            derived=inertia_mean * wn**2,
            committed=vehicle.pid.kp_nm_per_rad,
            units="N.m/rad",
            formula="Kp = J * wn^2",
            inputs=f"J = {inertia_mean:g} kg.m^2, wn = {wn:.4g} rad/s",
            reasoning=(
                f"wn = {wn:.4g} rad/s is {sample_rate / wn:.0f}x below the "
                f"{sample_rate:.3g} rad/s sample rate, so the zero-order hold "
                "costs a few degrees of phase rather than the design's stability; "
                f"the sampled-data loop crosses over at {crossover:.4g} rad/s and "
                "analysis.control.margins is the authority on what that is worth. "
                "Raising wn is bounded above by the sample rate, by the "
                "estimator's noise (a wider loop passes more of it to the wheels) "
                "and by the wheel torque the transient demands, none of which a "
                "gain formula settles — which is why no criterion here judges the "
                "bandwidth choice, only that it is inside the sampling bound."
            ),
        ),
        DerivedParameter(
            name="PidKdNmPerRadps",
            derived=2.0 * zeta * wn * inertia_mean,
            committed=vehicle.pid.kd_nm_per_radps,
            units="N.m/(rad/s)",
            formula="Kd = 2 * zeta * wn * J",
            inputs=f"J = {inertia_mean:g} kg.m^2, wn = {wn:.4g} rad/s, zeta = {zeta:.4g}",
            reasoning=(
                f"zeta = {zeta:.4g} is what the committed Kp and Kd jointly imply; "
                "it is not an independent recommendation, so this row is a "
                "consistency check that the two gains describe one second-order "
                f"design. Settling to 2% takes about 4/(zeta*wn) = "
                f"{_settling_time_s(wn, zeta):.0f} s. The derivative term acts on the "
                "measured body rate rather than a differenced error, so there is "
                "no derivative-filter pole to trade against it."
            ),
        ),
        DerivedParameter(
            name="MomentumEnvelopeNms",
            derived=siso_limit,
            committed=vehicle.momentum_envelope_nms,
            units="N.m.s",
            formula="h_limit = MAX_SISO_COUPLING_RATIO * J_min * w_crossover",
            inputs=(
                f"J_min = {inertia_min:g} kg.m^2, "
                f"w_c = {crossover:.4g} rad/s (axis x)"
            ),
            reasoning=(
                "Not a wheel-capacity number. The certified pointing margins come "
                "from a per-axis analysis that assumes the gyroscopic term "
                "omega x (J omega + h) is negligible at crossover; past this "
                "momentum it is not, and the margins describe a different "
                f"vehicle. The wheels themselves hold "
                f"{wheels.momentum.inscribed:.3g} N.m.s in every direction — "
                f"{wheels.momentum.inscribed / siso_limit:.0f}x more than the "
                "analysis covers. Raising the envelope needs MIMO analysis, not a "
                "bigger wheel (design doc SS8.5)."
                if np.isfinite(siso_limit)
                else "vehicle. This loop has no findable gain crossover, so the "
                "bound is unknown and no envelope can be certified against it — "
                "fix the tuning before reading any momentum criterion here."
            ),
        ),
        DerivedParameter(
            name="MomentumDesatEnterNms",
            derived=0.5 * vehicle.momentum_envelope_nms,
            committed=vehicle.momentum_desat_enter_nms,
            units="N.m.s",
            formula="enter = 0.5 * MomentumEnvelopeNms",
            inputs=f"envelope = {vehicle.momentum_envelope_nms:g} N.m.s",
            reasoning=(
                "The vehicle must act before it alarms, so the entry threshold "
                "sits inside the envelope; half of it leaves room for the "
                "momentum to keep rising during the confirmation count and the "
                "unloading transient without crossing the envelope the FDIR event "
                "is written on. The exact fraction is a margin choice, not a "
                "derivation — what the tool enforces is the ordering."
            ),
        ),
        DerivedParameter(
            name="MomentumDesatExitNms",
            derived=0.3 * vehicle.momentum_desat_enter_nms,
            committed=vehicle.momentum_desat_exit_nms,
            units="N.m.s",
            formula="exit = 0.3 * enter",
            inputs=f"enter = {vehicle.momentum_desat_enter_nms:g} N.m.s",
            reasoning=(
                "Hysteresis: unloading must stop well below where it started or "
                "the vehicle re-engages on the next disturbance cycle and the rods "
                "run continuously. A 3.3x band is wide enough that the observer's "
                "noise on the momentum estimate cannot walk the state across it."
            ),
        ),
        DerivedParameter(
            name="DetumbleExitRadps",
            derived=recommended_exit,
            committed=vehicle.detumble_exit_radps,
            units="rad/s",
            formula="max(f * h_usable / J_max, sigma*sqrt(2)/(dt*|B|_min))",
            inputs=(
                f"f = {assumptions.detumble_exit_fraction:g}, "
                f"h_usable = {wheels.usable_momentum_nms:.3g} N.m.s, "
                f"J_max = {inertia_max:g} kg.m^2 -> "
                f"{np.degrees(momentum_exit):.3g} deg/s; "
                f"noise floor -> {np.degrees(floor):.3g} deg/s"
            ),
            reasoning=(
                "Two independent bounds and the threshold must clear both. From "
                "below, the wheels have to absorb whatever body momentum survives "
                "detumble and still have authority left, which puts the handover "
                f"at {np.degrees(momentum_exit):.3g} deg/s. From below also, B-dot "
                "cannot resolve a rate under its own measurement floor, which is "
                f"{np.degrees(floor):.3g} deg/s here — so the momentum-based value "
                + (
                    "is refused and the floor is recommended instead. **These two "
                    "bounds are incompatible on this vehicle**: handing over at "
                    "the floor leaves more body momentum than the usable envelope "
                    "can hold, so no threshold satisfies both and the design has "
                    "to change (a quieter magnetometer, a longer B-dot "
                    "differencing interval, or a larger certified envelope)."
                    if floor > momentum_exit
                    else "governs and is recommended."
                )
            ),
        ),
        DerivedParameter(
            name="BdotGainNms",
            derived=2.0 * vehicle.orbit.mean_motion_rad_s * 2.0 * inertia_min,
            committed=vehicle.bdot_gain_nms,
            units="N.m.s",
            formula="k >= 2 * omega_o * (1 + sin xi) * J_min",
            inputs=(
                f"omega_o = {vehicle.orbit.mean_motion_rad_s:.4g} rad/s, "
                f"sin xi = 1 (worst case), J_min = {inertia_min:g} kg.m^2"
            ),
            reasoning=(
                "Avanzini & Giulietti's floor guarantees exponential convergence; "
                "xi is the field's inclination to the orbit plane and 90 deg is "
                "taken, the worst case and the right one for a near-polar orbit. "
                "Everything above the floor buys decay rate until the rods "
                "saturate, so a gain well above it is not an error — but a gain "
                "below it does not converge, which is why this one has a verdict."
            ),
        ),
    )


def criteria(
    vehicle: Vehicle,
    wheels: WheelSizing,
    mtq: MtqSizing,
    assumptions: SizingAssumptions | None = None,
) -> list[Criterion]:
    """Pass/fail criteria on the derived tuning.

    Only the checks that can genuinely fail appear: the bandwidth against the
    sampling bound, the damping the gains imply, the envelope against the SISO
    boundary, both desaturation ordering invariants, and the handover momentum
    the B-dot noise floor forces the wheels to absorb.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    wheels : WheelSizing
        Supplies the usable envelope.
    mtq : MtqSizing
        Supplies the noise floor.
    assumptions : SizingAssumptions, optional
        Supplies the margin factor.

    Returns
    -------
    list of analysis.common.report.Criterion
    """
    assumptions = assumptions or SizingAssumptions()
    wn, zeta = implied_bandwidth(vehicle)
    sample_rate = 2.0 * np.pi / vehicle.control_period_s
    siso_limit, crossover = siso_momentum_bound(vehicle)
    inertia_max = float(np.max(vehicle.principal_moments_kgm2))
    inertia_min = float(np.min(vehicle.principal_moments_kgm2))
    handover_momentum = inertia_max * mtq.noise_floor.rate_worst_radps

    return [
        Criterion(
            name="closed-loop bandwidth below the sampling bound",
            requirement="",
            threshold=MAX_BANDWIDTH_FRACTION * sample_rate,
            measured=wn,
            units="rad/s",
            sense="max",
            note=(
                f"wn implied by Kp = {vehicle.pid.kp_nm_per_rad:g} N.m/rad on the "
                f"mean principal moment; sample rate {sample_rate:.4g} rad/s"
            ),
        ),
        Criterion(
            name="damping ratio implied by Kp and Kd together",
            requirement="",
            threshold=MIN_DAMPING,
            measured=zeta,
            units="-",
            sense="min",
            note=f"settling to 2% in about {_settling_time_s(wn, zeta):.0f} s",
        ),
        Criterion(
            name="MomentumEnvelopeNms within the SISO validity bound",
            requirement="REQ-ACTL-009",
            threshold=siso_limit / 1.10,
            measured=vehicle.momentum_envelope_nms,
            units="N.m.s",
            sense="max",
            note=(
                f"bound {siso_limit:.3g} N.m.s at the {crossover:.4g} rad/s "
                "crossover, with the 10% margin REQ-ACTL-009 requires; recomputed "
                "here through analysis.control.plant.siso_coupling, the same "
                "function tests/analysis/test_control_momentum_envelope.py uses"
            ),
        ),
        Criterion(
            name="desaturation entry inside the envelope",
            requirement="",
            threshold=1.0,
            measured=vehicle.momentum_desat_enter_nms / vehicle.momentum_envelope_nms,
            units="of envelope",
            sense="max",
            note="ordering invariant: exit < enter < envelope — act before alarming",
        ),
        Criterion(
            name="desaturation exit below entry",
            requirement="",
            threshold=1.0,
            measured=(
                vehicle.momentum_desat_exit_nms / vehicle.momentum_desat_enter_nms
            ),
            units="of entry",
            sense="max",
            note="the other half of the ordering invariant; the gap is the hysteresis",
        ),
        Criterion(
            name="BdotGainNms above the Avanzini convergence floor",
            requirement="",
            threshold=2.0 * vehicle.orbit.mean_motion_rad_s * 2.0 * inertia_min,
            measured=vehicle.bdot_gain_nms,
            units="N.m.s",
            sense="min",
            note=(
                "k >= 2*omega_o*(1 + sin xi)*J_min at sin xi = 1, the worst case "
                "and the right one for a near-polar orbit [avanzini2012]; above "
                "the floor buys decay rate, below it does not converge"
            ),
        ),
        Criterion(
            name="usable momentum vs handover at the B-dot noise floor",
            requirement="",
            threshold=assumptions.margin * handover_momentum,
            measured=wheels.usable_momentum_nms,
            units="N.m.s",
            sense="min",
            note=(
                f"B-dot cannot certify a rate below "
                f"{np.degrees(mtq.noise_floor.rate_worst_radps):.2f} deg/s, so the "
                f"wheels must be able to take {handover_momentum:.3g} N.m.s at "
                "handover however the exit threshold is set"
            ),
        ),
    ]

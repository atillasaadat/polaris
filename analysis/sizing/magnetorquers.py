"""Magnetorquer sizing: desaturation authority, detumble authority, noise floor.

A rod set produces :math:`\\boldsymbol\\tau = \\mathbf m\\times\\mathbf B`. Two
consequences shape everything here:

1. **Only the dipole perpendicular to B does work.** The component along
   :math:`\\hat{\\mathbf b}` produces no torque at all, and of the torque
   produced, only the part along the momentum being removed unloads anything.
   The cross-product law delivers :math:`-k(I-\\hat{\\mathbf b}\\hat{\\mathbf
   b}^\\top)\\Delta\\mathbf h`, whose orbit-average is :math:`2/3` of the ideal
   because :math:`\\langle\\hat{\\mathbf b}\\hat{\\mathbf b}^\\top\\rangle=I/3`
   for a field direction that samples the sphere (see
   :data:`analysis.sizing.assumptions.MTQ_ORBIT_EFFICIENCY`; a near-equatorial
   orbit does worse and the assumption must be lowered for one).
2. **Size against the minimum field over the orbit, not the mean.** The rods'
   worst moment is where :math:`\\lvert\\mathbf B\\rvert` is smallest, and an
   authority claim made at the orbit mean is one the vehicle cannot honour for
   part of every lap. :func:`analysis.sizing.disturbances.field_statistics`
   supplies the minimum from the tilted-dipole model over the config's own orbit.

The available average torque is therefore

.. math::

   \\bar\\tau_{mtq} = \\eta\\;m_{\\mathrm{in}}\\;\\lvert\\mathbf B\\rvert_{\\min}
   \\;d,

with :math:`m_{\\mathrm{in}}` the **inscribed** radius of the rod set's dipole
zonotope (the dipole guaranteed in every direction — an orthogonal triad at 15
A·m² per rod guarantees 15, not the :math:`15\\sqrt3` a corner of the cube
reaches), :math:`\\eta` the efficiency above and :math:`d` the §7 duty factor.

The three criteria
------------------
**M1 — desaturation authority.** :math:`\\bar\\tau_{mtq} \\ge 1.3\\,
\\tau_{secular}`. *This is where the disturbance-torque budget lands*: the rods
must dump secular momentum faster than the environment accumulates it, or the
wheels saturate no matter how large they are. The whole
:mod:`analysis.sizing.disturbances` budget exists to feed this one number.

**M2 — detumble authority.** The rods must remove the tip-off momentum inside
the assumed budget: :math:`\\bar\\tau_{mtq}\\,T_{budget} \\ge 1.3\\,\\lvert
J\\boldsymbol\\omega_{tipoff}\\rvert`. The implied duration is reported, with
the standing caveat that B-dot damps only the rate perpendicular to the field
and the residual spin about the field line unwinds over *orbits* (REQ-ACTL-001's
recorded finding), so this is a bound on the fast phase and not a detumble time.

**M3 — the B-dot noise floor.** B-dot differentiates the measured field over the
control period. Two samples each carrying :math:`\\sigma` of noise give a
derivative noise :math:`\\sigma\\sqrt2/\\Delta t`, and the true signal is
:math:`\\lvert\\mathrm dB/\\mathrm dt\\rvert \\approx \\lvert\\boldsymbol\\omega
\\times\\mathbf B\\rvert`. The two are equal at

.. math::

   \\omega_{floor} = \\frac{\\sigma\\sqrt2}{\\Delta t\\,\\lvert\\mathbf B\\rvert}.

**Below that rate B-dot is commanding on its own noise.** It is not a failure of
the law — it is the measurement floor — but it means any detumble exit threshold
placed below :math:`\\omega_{floor}` declares completion at a rate the vehicle
cannot tell from noise, and a controller left running there dissipates nothing
while dithering the rods. The floor is a **design output** of this tool, and the
criterion is that the committed ``DetumbleExitRadps`` sits above it.

Units and frames
----------------
Dipole [A·m²], field [T], torque [N·m], rates [rad/s], momentum [N·m·s]. Rod
axes are body-frame unit vectors.

References
----------
Sidi, *Spacecraft Dynamics and Control*, §7.5 [sidi1997] — magnetic unloading,
the cross-product law and its efficiency.
Camillo & Markley, "Orbit-averaged behavior of magnetic control laws for
momentum unloading" [camillo1980] — the orbit-averaged authority result.
Avanzini & Giulietti, "Magnetic Detumbling of a Rigid Spacecraft"
[avanzini2012] — the B-dot law and its gain floor.
Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*, §7.4 [markley2014] — magnetic actuation and its rank deficiency.
Design doc §7 (MTQ/MAG interlock), §8.5 (control), §12 (analysis tools).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.common.report import Criterion
from analysis.control.vehicle import Vehicle
from analysis.sizing.assumptions import SizingAssumptions
from analysis.sizing.disturbances import DisturbanceBudget
from analysis.sizing.envelope import Envelope, envelope


@dataclass(frozen=True)
class BdotNoiseFloor:
    """The body rate at which B-dot's signal equals its own measurement noise.

    Attributes
    ----------
    rate_worst_radps : float
        Floor at the orbit's **minimum** field [rad/s] — the worst case, and the
        one a threshold must clear.
    rate_mean_radps : float
        Floor at the orbit-mean field [rad/s], reported so a typical-case number
        is available beside the bounding one.
    derivative_noise_t_s : float
        :math:`\\sigma\\sqrt2/\\Delta t` [T/s], the noise on the differenced
        field.
    sigma_t : float
        Per-sample magnetometer noise, 1σ per axis [T].
    sample_period_s : float
        The differencing interval [s].
    """

    rate_worst_radps: float
    rate_mean_radps: float
    derivative_noise_t_s: float
    sigma_t: float
    sample_period_s: float


def bdot_noise_floor(vehicle: Vehicle, budget: DisturbanceBudget) -> BdotNoiseFloor:
    """The B-dot rate floor implied by the magnetometer and the control period.

    The sample period is the **control period**: the flight law differences
    successive samples with their own time tags inside the band
    ``BdotMinSampleDtSec``–``BdotMaxSampleDtSec``, and at the 10 Hz GNC rate the
    nominal spacing is the control period. Differencing over a longer interval
    lowers the floor proportionally, which is the design lever if the floor is
    the binding constraint — bounded above by ``BdotMaxSampleDtSec``, past which
    the secant stops being the tangent.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies the magnetometer noise and control period.
    budget : DisturbanceBudget
        Supplies the field statistics.

    Returns
    -------
    BdotNoiseFloor
    """
    derivative_noise = vehicle.mag_noise_t * np.sqrt(2.0) / vehicle.control_period_s
    return BdotNoiseFloor(
        rate_worst_radps=float(derivative_noise / budget.field.min_t),
        rate_mean_radps=float(derivative_noise / budget.field.mean_t),
        derivative_noise_t_s=float(derivative_noise),
        sigma_t=vehicle.mag_noise_t,
        sample_period_s=vehicle.control_period_s,
    )


@dataclass(frozen=True)
class MtqSizing:
    """Rod-set capability and the demands placed on it.

    Attributes
    ----------
    dipole : Envelope
        Dipole envelope at the per-rod rating [A·m²].
    average_torque_nm : float
        :math:`\\eta\\,m_{\\mathrm{in}}\\,\\lvert B\\rvert_{\\min}\\,d` [N·m] —
        what the set delivers on average, in the worst direction, at the weakest
        field.
    peak_torque_nm : float
        The same without the efficiency factor or the duty factor, i.e. the
        instantaneous torque with the dipole perpendicular to the field [N·m].
        Reported, never sized against.
    field_min_t : float
        The minimum field over the orbit the authority was evaluated at [T].
    secular_torque_nm : float
        The secular disturbance it must beat [N·m].
    tipoff_momentum_nms : float
        The tip-off momentum it must remove [N·m·s].
    detumble_budget_s : float
        The time allowed to remove it [s].
    noise_floor : BdotNoiseFloor
        The B-dot measurement floor.
    """

    dipole: Envelope
    average_torque_nm: float
    peak_torque_nm: float
    field_min_t: float
    secular_torque_nm: float
    tipoff_momentum_nms: float
    detumble_budget_s: float
    noise_floor: BdotNoiseFloor

    @property
    def implied_detumble_s(self) -> float:
        """Time to remove the tip-off momentum at the average torque [s]."""
        if self.average_torque_nm <= 0.0:
            return float("inf")
        return self.tipoff_momentum_nms / self.average_torque_nm

    @property
    def removable_momentum_nms(self) -> float:
        """Momentum removable inside the detumble budget [N·m·s]."""
        return self.average_torque_nm * self.detumble_budget_s


def mtq_sizing(
    vehicle: Vehicle,
    budget: DisturbanceBudget,
    assumptions: SizingAssumptions | None = None,
) -> MtqSizing:
    """Compute the rod set's authority and the demands on it.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    budget : DisturbanceBudget
        Supplies the secular torque and the field statistics.
    assumptions : SizingAssumptions, optional
        Supplies the efficiency, the tip-off rate and the detumble budget.

    Returns
    -------
    MtqSizing

    Raises
    ------
    analysis.sizing.envelope.DegenerateArrayError
        If the rod axes do not span three dimensions.
    """
    assumptions = assumptions or SizingAssumptions()
    dipole = envelope(vehicle.mtq_axes, vehicle.mtq_max_dipole_am2)
    worst_inertia = float(np.max(vehicle.principal_moments_kgm2))
    average = (
        assumptions.mtq_efficiency
        * dipole.inscribed
        * budget.field.min_t
        * vehicle.mtq_duty_factor
    )
    return MtqSizing(
        dipole=dipole,
        average_torque_nm=float(average),
        peak_torque_nm=float(dipole.inscribed * budget.field.min_t),
        field_min_t=budget.field.min_t,
        secular_torque_nm=budget.secular_nm,
        tipoff_momentum_nms=worst_inertia * assumptions.tipoff_rate_radps,
        detumble_budget_s=assumptions.detumble_budget(vehicle.orbit.period_s),
        noise_floor=bdot_noise_floor(vehicle, budget),
    )


def criteria(
    vehicle: Vehicle,
    sizing: MtqSizing,
    assumptions: SizingAssumptions | None = None,
) -> list[Criterion]:
    """Pass/fail criteria M1, M2 and M3 for the rod set.

    Carries no requirement IDs for the same reason
    :func:`analysis.sizing.wheels.criteria` does not: nothing in the baseline is
    written on actuator sizing. REQ-ACTL-010 governs *how* desaturation runs, not
    whether the rods are big enough to finish it.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    sizing : MtqSizing
        The computed authority.
    assumptions : SizingAssumptions, optional
        Supplies the margin factor.

    Returns
    -------
    list of analysis.common.report.Criterion
    """
    assumptions = assumptions or SizingAssumptions()
    margin = assumptions.margin
    floor = sizing.noise_floor
    return [
        Criterion(
            name="M1 desaturation authority vs secular disturbance",
            requirement="",
            threshold=margin * sizing.secular_torque_nm,
            measured=sizing.average_torque_nm,
            units="N.m",
            sense="min",
            note=(
                f"eta * m_in * |B|_min * duty = {assumptions.mtq_efficiency:.3g} * "
                f"{sizing.dipole.inscribed:g} A.m^2 * "
                f"{sizing.field_min_t * 1e6:.1f} uT * {vehicle.mtq_duty_factor:g}; the "
                "secular half of the disturbance budget is what it must beat"
            ),
            formula="eta * m_in * |B|_min * duty",
            formula_tex=(
                r"\bar\tau_{\mathrm{mtq}} = \eta\,m_{\mathrm{in}}\,"
                r"|B|_{\min}\,d_{\mathrm{duty}}"
            ),
        ),
        Criterion(
            name="M2 detumble authority (momentum removable in budget)",
            requirement="",
            threshold=margin * sizing.tipoff_momentum_nms,
            measured=sizing.removable_momentum_nms,
            units="N.m.s",
            sense="min",
            note=(
                f"implied fast-phase duration {sizing.implied_detumble_s:.0f} s "
                f"against a {sizing.detumble_budget_s:.0f} s budget; the residual "
                "spin about the field line unwinds over orbits and is not covered "
                "by this bound (REQ-ACTL-001)"
            ),
        ),
        Criterion(
            name="M3 DetumbleExitRadps above the B-dot noise floor",
            requirement="",
            threshold=float(np.degrees(floor.rate_worst_radps)),
            measured=float(np.degrees(vehicle.detumble_exit_radps)),
            units="deg/s",
            sense="min",
            note=(
                f"floor = sigma*sqrt(2)/(dt*|B|) with sigma = {floor.sigma_t * 1e9:.0f} "
                f"nT, dt = {floor.sample_period_s:g} s; "
                f"{np.degrees(floor.rate_worst_radps):.2f} deg/s at the orbit's "
                f"weakest field, {np.degrees(floor.rate_mean_radps):.2f} deg/s at "
                "the mean. Below the floor B-dot commands on noise"
            ),
            formula="floor = sigma*sqrt(2)/(dt*|B|)",
            formula_tex=(
                r"\omega_{\mathrm{floor}} = " r"\frac{\sigma\sqrt{2}}{\Delta t\,|B|}"
            ),
        ),
    ]

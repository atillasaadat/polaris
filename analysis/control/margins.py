"""Classical and disk stability margins, extracted from our own frequency response.

This module owns its margin arithmetic — there is no control-systems library
behind it, only ``numpy`` and ``scipy.optimize.brentq`` for root refinement.
That makes it the piece of the toolkit most deserving of analytic validation,
and ``tests/analysis/test_control_margins.py`` accordingly checks it against
three cases with closed-form answers before it is ever pointed at the vehicle:
a PD-controlled double integrator, a textbook third-order loop with an exact
gain margin of 6 (15.563 dB) [ogata2010], and the vehicle's own continuous loop
against its hand-derived polynomial.

The margin requirement, and where the numbers come from
-------------------------------------------------------
A rigid-body attitude loop is conventionally designed to at least **6 dB of
gain margin and 30° of phase margin**, with 45° the preferred design target;
these are the long-standing values of aerospace control practice, stated for
spacecraft attitude loops by Sidi [sidi1997] and as the classical rule of thumb
(phase margin 30–60°, gain margin 2–5, sensitivity peak :math:`M_s<2`) by
Åström & Murray [astrom2008]. They are declared here as
:data:`MIN_GAIN_MARGIN_DB`, :data:`MIN_PHASE_MARGIN_DEG` and
:data:`MAX_SENSITIVITY_PEAK`, enforced by
:func:`analysis.control.report.margin_report`, and are the content of
REQ-ACTL-006.

How the margins are extracted
-----------------------------
One logarithmic frequency sweep, then:

* **Gain crossovers** are the sign changes of :math:`|L|-1`, refined by
  bisection on the exact magnitude — not on an interpolant, which is accurate
  only to the grid spacing.
* **Phase crossovers** are the crossings of every odd multiple of −180° in the
  **unwrapped** phase, so a type-3 loop's several crossings are all found. The
  phase at a refined frequency is the exact wrapped angle put back on the
  branch the grid establishes, which is right to full precision *and* right
  about which multiple of 360° it belongs to.
* Multiple crossings of either kind are handled by reporting the **worst**: the
  smallest phase margin over all unity-gain crossings, and the tightest gain
  scaling in each direction over all phase crossings.

Gain margin has two directions, and this loop needs both
--------------------------------------------------------
The pointing loop is **type 3** — an integrator on top of a double integrator
(:mod:`analysis.control.plant`) — so its phase begins at −270° and rises
through −180° at the low frequency :math:`\\sqrt{K_i/K_d}` before settling at
−90°. The loop is *conditionally stable*: the −180° crossing sits at
:math:`|L|>1`, so **reducing** the loop gain destabilises it and increasing it
does not. A single signed "gain margin" — what the textbook
:math:`1/|L(j\\omega_{180})|` yields, and what every library margin routine
returns — reads −15 dB on a perfectly healthy design and would fail a
requirement written for the ordinary case.

This module therefore reports the margin as the interval of loop-gain scalings
that preserve stability: :attr:`AxisMargins.gain_margin_up_db` (how much the
gain may rise) and :attr:`AxisMargins.gain_margin_down_db` (how much it may
fall), with :attr:`AxisMargins.gain_margin_db` the smaller of the two — which
is the quantity the 6 dB requirement is about, and the honest one for a
conditionally stable loop.

Disk margins
------------
A gain margin holds the phase fixed and a phase margin holds the gain fixed;
neither describes a simultaneous perturbation, and a loop can hold 6 dB and 45°
while a small combined error destabilises it. The **symmetric (skew-0) disk
margin** closes that gap [seiler2020]:

.. math::

   \\alpha = \\frac{1}{\\big\\|S - \\tfrac12\\big\\|_\\infty}
   \\quad\\text{with}\\quad S = \\frac{1}{1+L},

whose guaranteed simultaneous variations are :math:`(2+\\alpha)/(2-\\alpha)` in
gain and :math:`2\\arctan(\\alpha/2)` in phase. Computed here from the same
frequency-response array as everything else; note that
:math:`S-\\tfrac12 = \\tfrac12(S-T)` with :math:`T = 1-S`, which is where the
"peak of :math:`S` and :math:`T`" spelling of the same quantity comes from.

What is analysed
----------------
The **sampled-data** loop at ``ControlPeriodSec`` by default — the zero-order
hold is part of the design, and a continuous analysis of a 10 Hz loop flatters
it. :func:`axis_margins` takes ``sampled=False`` for the continuous comparison,
and the difference between the two is small here only because the crossover
sits two decades below the sample rate.

All assumptions of :mod:`analysis.control.plant` apply: small angle,
unsaturated torque, integrator unfrozen, and — the one that is *measured*
rather than asserted — a near-diagonal inertia with small stored wheel
momentum. Per-axis analysis assumes both; see design doc §8.5, "SISO validity
boundary and MIMO roadmap", and :attr:`AxisMargins.siso_coupling_ratio`, which
carries the number for the analysed configuration.

References
----------
Sidi, *Spacecraft Dynamics and Control: A Practical Engineering Approach*,
Cambridge, 1997 [sidi1997].
Åström & Murray, *Feedback Systems*, §12 [astrom2008].
Ogata, *Modern Control Engineering*, 5th ed., Ch. 7 [ogata2010] — the
gain/phase-margin definitions and the third-order validation example.
Seiler, Packard & Gahinet, "An Introduction to Disk Margins," *IEEE Control
Systems Magazine* 40(5), 2020 [seiler2020].
Franklin, Powell & Workman, *Digital Control of Dynamic Systems*, 3rd ed.
[franklin1998].
Design doc §8.5, §13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from analysis.control.plant import (
    AXES,
    MAX_SISO_COUPLING_RATIO,
    Loop,
    discrete_open_loop,
    open_loop,
    siso_coupling,
)
from analysis.control.vehicle import Vehicle

#: Smallest acceptable gain margin [dB], in either direction (REQ-ACTL-006).
MIN_GAIN_MARGIN_DB = 6.0

#: Smallest acceptable phase margin [deg] (REQ-ACTL-006).
MIN_PHASE_MARGIN_DEG = 30.0

#: Preferred phase-margin design target [deg]. Not a pass/fail threshold — it is
#: reported so a design that merely clears the floor is visible as such.
PREFERRED_PHASE_MARGIN_DEG = 45.0

#: Largest acceptable peak of the sensitivity function :math:`\|S\|_\infty` [-].
#: The classical robustness companion to the two margins above: :math:`M_s<2`
#: alone implies at least 6 dB of gain margin and 29° of phase margin
#: [astrom2008], so it is checked rather than merely printed.
MAX_SENSITIVITY_PEAK = 2.0

#: Decades of clearance a detected crossing must keep from a grid edge. A
#: crossing sitting on the edge means the sweep may be hiding a second one just
#: outside it — and a crossing the sweep never sees is reported as ``inf``,
#: which is indistinguishable from "no crossing exists". That is a silently
#: *inflated* margin, so the guard fails loudly instead (see
#: :func:`_check_grid_window`).
GRID_EDGE_CLEARANCE_DECADES = 1.0


class GridWindowError(RuntimeError):
    """The frequency sweep does not comfortably bracket the loop's crossings.

    Raised rather than returning an inflated margin: an unbounded gain margin
    and a gain margin whose crossing fell off the end of the sweep are the same
    number, and only one of them is true.
    """


def frequency_grid(loop: Loop, points: int = 4000) -> np.ndarray:
    """Logarithmic frequency grid [rad/s], up to Nyquist for a discrete loop.

    Parameters
    ----------
    loop : analysis.control.plant.Loop
        The loop the grid is for; its time base sets the upper limit.
    points : int, optional
        Number of grid points.

    Returns
    -------
    numpy.ndarray
        Frequencies [rad/s], ascending.
    """
    # Up to and *including* Nyquist: a ZOH loop can meet the negative real axis
    # exactly at z = -1, and a grid stopping just short of it reports an
    # infinite upward gain margin where a finite one exists.
    top = math.pi / loop.dt if loop.discrete else 1.0e4
    return np.logspace(math.log10(1.0e-5), math.log10(top), points)


def _refine(fun, lo: float, hi: float) -> float:
    """Bracketed root of a scalar frequency function, or the midpoint on failure."""
    try:
        return float(brentq(fun, lo, hi, xtol=1.0e-12, rtol=1.0e-12))
    except ValueError:  # pragma: no cover - the caller only brackets sign changes
        return 0.5 * (lo + hi)


def _crossings(
    omega: np.ndarray, values: np.ndarray, target: float, fun
) -> list[float]:
    """Frequencies where ``values`` crosses ``target``, refined by bisection."""
    shifted = values - target
    idx = np.nonzero(np.sign(shifted[:-1]) * np.sign(shifted[1:]) < 0.0)[0]
    return [
        _refine(lambda w: fun(w) - target, float(omega[i]), float(omega[i + 1]))
        for i in idx
    ]


def _check_grid_window(
    loop: Loop,
    omega: np.ndarray,
    gain_crossings: list[float],
    phase_crossings: list[float],
) -> None:
    """Fail loudly when the sweep does not comfortably bracket the crossings.

    The grid bounds are constants, not derived from the gains, so a
    configuration whose characteristic frequencies sit outside them would be
    scored on a window that never saw its crossings — and an unseen crossing is
    reported as an *unbounded* margin, which reads as a better design rather
    than an unmeasured one.

    Two conditions, both cheap:

    * a loop with no unity-gain crossing at all inside the sweep has no
      measurable phase margin, whatever the arithmetic returns;
    * any detected crossing within
      :data:`GRID_EDGE_CLEARANCE_DECADES` of an edge means the sweep is only
      just containing this loop, so the next configuration probably will not be.

    The **top edge of a discrete sweep is exempt**: it is the Nyquist frequency,
    a physical boundary rather than a chosen one. There is no response above it
    for the sweep to be hiding, and the reference vehicle's sampled loop
    legitimately meets the negative real axis exactly there.

    Parameters
    ----------
    loop : analysis.control.plant.Loop
        The loop being scored.
    omega : numpy.ndarray
        The sweep the crossings were found on [rad/s].
    gain_crossings, phase_crossings : list of float
        Refined crossing frequencies [rad/s].

    Raises
    ------
    GridWindowError
        If either condition fires.
    """
    lo, hi = float(omega[0]), float(omega[-1])
    if not gain_crossings:
        raise GridWindowError(
            f"no unity-gain crossing in [{lo:.3g}, {hi:.3g}] rad/s: this loop's "
            "crossover is outside the sweep, so its phase margin was not measured "
            "and its gain margin would be reported unbounded. Widen the grid or "
            "check the gains."
        )
    margin = 10.0**GRID_EDGE_CLEARANCE_DECADES
    for what, crossings in (("gain", gain_crossings), ("phase", phase_crossings)):
        for w in crossings:
            near_top = not loop.discrete and w > hi / margin
            if w < lo * margin or near_top:
                raise GridWindowError(
                    f"{what} crossing at {w:.3g} rad/s is within "
                    f"{GRID_EDGE_CLEARANCE_DECADES:g} decade of the sweep edge "
                    f"[{lo:.3g}, {hi:.3g}] rad/s: a crossing just outside would be "
                    "invisible and would inflate the reported margin."
                )


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle into (−180, 180]."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass(frozen=True)
class AxisMargins:
    """Margins of one body axis's pointing loop.

    Attributes
    ----------
    axis : str
        Body axis name, ``"x"``/``"y"``/``"z"``.
    sampled : bool
        The loop analysed was the discrete sampled-data one.
    sample_period_s : float
        Sampling period [s]; ``0.0`` for a continuous analysis.
    stable : bool
        The nominal closed loop is stable. Every other field is meaningless
        when this is ``False``.
    gain_crossover_rad_s : float
        Frequency where :math:`|L| = 1` [rad/s] — the one whose phase margin is
        reported. ``nan`` if the loop never crosses unity gain.
    phase_margin_deg : float
        Smallest phase margin over all unity-gain crossings [deg].
    gain_margin_up_db : float
        Loop-gain **increase** tolerated before instability [dB]; ``inf`` when
        unbounded.
    gain_margin_down_db : float
        Loop-gain **decrease** tolerated [dB]; ``inf`` when unbounded.
    gain_margin_db : float
        The smaller of the two — the quantity the requirement is written on.
    phase_crossover_rad_s : tuple of float
        Every frequency where the phase passes an odd multiple of −180° [rad/s].
    sensitivity_peak : float
        :math:`\\|S\\|_\\infty` with :math:`S = 1/(1+L)` [-].
    complementary_peak : float
        :math:`\\|T\\|_\\infty` with :math:`T = L/(1+L)` [-].
    disk_margin : float
        Symmetric (skew-0) disk margin :math:`\\alpha` [-] [seiler2020].
    disk_gain_margin_db : float
        Gain variation guaranteed by the disk margin [dB].
    disk_phase_margin_deg : float
        Phase variation guaranteed by the disk margin [deg].
    siso_coupling_ratio : float
        Gyroscopic cross-coupling at this crossover with the wheel array at its
        momentum capacity [-]; see :func:`analysis.control.plant.siso_coupling`.
        ``nan`` when the loop did not come from a vehicle.
    siso_momentum_limit_nms : float
        Stored momentum [N·m·s] at which that ratio reaches
        :data:`analysis.control.plant.MAX_SISO_COUPLING_RATIO` — the largest
        momentum bias this per-axis analysis covers. ``nan`` as above.
    """

    axis: str
    sampled: bool
    sample_period_s: float
    stable: bool
    gain_crossover_rad_s: float
    phase_margin_deg: float
    gain_margin_up_db: float
    gain_margin_down_db: float
    gain_margin_db: float
    phase_crossover_rad_s: tuple[float, ...]
    sensitivity_peak: float
    complementary_peak: float
    disk_margin: float
    disk_gain_margin_db: float
    disk_phase_margin_deg: float
    siso_coupling_ratio: float = float("nan")
    siso_momentum_limit_nms: float = float("nan")

    @property
    def conditionally_stable(self) -> bool:
        """The loop is destabilised by a gain *decrease* (finite down-margin)."""
        return math.isfinite(self.gain_margin_down_db)

    @property
    def siso_assumption_holds(self) -> bool:
        """The per-axis model is trustworthy for this configuration.

        ``True`` when the coupling ratio was not evaluated (no vehicle) or is at
        or below :data:`analysis.control.plant.MAX_SISO_COUPLING_RATIO`. This is
        a **warning** condition, not a pass/fail one: the margins are still
        those of the loop as analysed, and what becomes doubtful is the claim
        that the loop as analysed is the vehicle.
        """
        if math.isnan(self.siso_coupling_ratio):
            return True
        return self.siso_coupling_ratio <= MAX_SISO_COUPLING_RATIO

    @property
    def passes(self) -> bool:
        """Meets the gain, phase and sensitivity-peak thresholds."""
        return (
            self.stable
            and self.gain_margin_db >= MIN_GAIN_MARGIN_DB
            and self.phase_margin_deg >= MIN_PHASE_MARGIN_DEG
            and self.sensitivity_peak <= MAX_SENSITIVITY_PEAK
        )


def _unstable(axis: str, sampled: bool, ts: float) -> AxisMargins:
    """The all-``nan`` result for a loop that is not stable to begin with."""
    nan = float("nan")
    return AxisMargins(
        axis=axis,
        sampled=sampled,
        sample_period_s=ts,
        stable=False,
        gain_crossover_rad_s=nan,
        phase_margin_deg=nan,
        gain_margin_up_db=nan,
        gain_margin_down_db=nan,
        gain_margin_db=nan,
        phase_crossover_rad_s=(),
        sensitivity_peak=nan,
        complementary_peak=nan,
        disk_margin=nan,
        disk_gain_margin_db=nan,
        disk_phase_margin_deg=nan,
    )


def disk_margin(response: np.ndarray) -> tuple[float, float, float]:
    """Symmetric disk margin and the simultaneous variation it guarantees.

    The skew-0 form :math:`\\alpha = 1/\\|S-\\tfrac12\\|_\\infty` with
    :math:`S = 1/(1+L)`, whose guaranteed margins are
    :math:`(2+\\alpha)/(2-\\alpha)` in gain and :math:`2\\arctan(\\alpha/2)` in
    phase [seiler2020]. Computed directly from a frequency-response array, so it
    costs one pass over data the caller already has.

    Parameters
    ----------
    response : numpy.ndarray
        :math:`L` evaluated on a frequency grid.

    Returns
    -------
    tuple of float
        ``(alpha, gain_variation_db, phase_variation_deg)``. The gain figure is
        ``inf`` for :math:`\\alpha\\ge2`, where the guaranteed gain variation is
        unbounded.
    """
    peak = float(np.max(np.abs(1.0 / (1.0 + response) - 0.5)))
    if peak <= 0.0 or not math.isfinite(peak):  # pragma: no cover - degenerate loop
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 / peak
    phase_deg = math.degrees(2.0 * math.atan(alpha / 2.0))
    if alpha >= 2.0:  # pragma: no cover - unbounded gain variation
        return alpha, float("inf"), phase_deg
    return alpha, 20.0 * math.log10((2.0 + alpha) / (2.0 - alpha)), phase_deg


def loop_margins(loop: Loop, axis: str = "-") -> AxisMargins:
    """Margins of an arbitrary SISO loop transfer function.

    Parameters
    ----------
    loop : analysis.control.plant.Loop
        Open-loop transfer function of a negative-unity-feedback loop,
        continuous or discrete.
    axis : str, optional
        Label carried into the result.

    Returns
    -------
    AxisMargins
        Populated result; ``stable=False`` short-circuits the rest to ``nan``.

    Raises
    ------
    GridWindowError
        If the frequency sweep does not comfortably bracket this loop's
        crossings — see :func:`_check_grid_window`. Preferred over returning a
        margin that is unbounded only because the crossing was off-window.
    """
    ts = loop.dt if loop.discrete else 0.0
    if not loop.is_stable():
        return _unstable(axis, loop.discrete, ts)

    omega = frequency_grid(loop)
    response = loop.response(omega)
    magnitude = np.abs(response)
    phase_deg = np.degrees(np.unwrap(np.angle(response)))

    def mag_at(w: float) -> float:
        return float(np.abs(loop.response(w)))

    def phase_at(w: float) -> float:
        """Unwrapped phase [deg] at one frequency, to full precision.

        A single-point evaluation gives a *wrapped* angle, which loses the
        branch a type-3 loop needs; the grid's unwrapped phase gives the branch
        but only to interpolation accuracy. Take the exact wrapped value and the
        branch from the grid, which is right about both.
        """
        exact = math.degrees(float(np.angle(loop.response(w))))
        approx = float(np.interp(w, omega, phase_deg))
        return exact + 360.0 * round((approx - exact) / 360.0)

    # Gain crossovers: |L| = 1. The phase margin is the smallest over all of
    # them, not the one at the first — a loop with several crossings is exactly
    # the case where taking the first quietly reports the wrong number.
    crossover = float("nan")
    phase_margin = float("inf")
    gain_crossings = _crossings(omega, magnitude, 1.0, mag_at)
    for wc in gain_crossings:
        pm = _wrap_deg(180.0 + phase_at(wc))
        if abs(pm) < abs(phase_margin):
            phase_margin = pm
            crossover = wc

    # Phase crossovers: every odd multiple of -180 inside the unwrapped range,
    # which is what catches the type-3 loop's low-frequency crossing.
    lo, hi = float(phase_deg.min()), float(phase_deg.max())
    phase_crossings: list[float] = []
    for k in range(
        int(math.floor((lo + 180.0) / 360.0)), int(math.ceil((hi + 180.0) / 360.0)) + 1
    ):
        target = -180.0 + 360.0 * k
        if lo < target < hi:
            phase_crossings.extend(_crossings(omega, phase_deg, target, phase_at))
    # A sampled loop can *touch* the negative real axis at Nyquist rather than
    # cross it; a sign-change search never sees that, and the loop nonetheless
    # has a finite upward gain margin there. Only add it when the sweep above did
    # not already find it: a loop whose phase reaches -180 + 360k *at* the last
    # sample is both a crossing and a touch, and reporting it twice would claim a
    # crossover the loop does not have.
    end = float(omega[-1])
    if abs(_wrap_deg(phase_deg[-1] + 180.0)) < 1.0e-6 and not any(
        math.isclose(w, end, rel_tol=1.0e-9) for w in phase_crossings
    ):
        phase_crossings.append(end)
    phase_crossings.sort()

    _check_grid_window(loop, omega, gain_crossings, phase_crossings)

    up = float("inf")
    down = float("inf")
    for wp in phase_crossings:
        gain = mag_at(wp)
        if gain <= 0.0:
            continue
        if gain < 1.0:
            up = min(up, 1.0 / gain)
        else:
            down = min(down, gain)

    alpha, disk_gm_db, disk_pm_deg = disk_margin(response)
    up_db = float("inf") if math.isinf(up) else 20.0 * math.log10(up)
    down_db = float("inf") if math.isinf(down) else 20.0 * math.log10(down)
    return AxisMargins(
        axis=axis,
        sampled=loop.discrete,
        sample_period_s=ts,
        stable=True,
        gain_crossover_rad_s=crossover,
        phase_margin_deg=phase_margin,
        gain_margin_up_db=up_db,
        gain_margin_down_db=down_db,
        gain_margin_db=min(up_db, down_db),
        phase_crossover_rad_s=tuple(phase_crossings),
        sensitivity_peak=float(np.max(np.abs(1.0 / (1.0 + response)))),
        complementary_peak=float(np.max(np.abs(response / (1.0 + response)))),
        disk_margin=alpha,
        disk_gain_margin_db=disk_gm_db,
        disk_phase_margin_deg=disk_pm_deg,
    )


def axis_margins(
    vehicle: Vehicle,
    axis: int,
    *,
    sampled: bool = True,
    integrator: bool = True,
    computation_delay_cycles: int = 0,
) -> AxisMargins:
    """Margins of one body axis of the as-flown pointing loop.

    Also evaluates the SISO validity boundary at the measured crossover, so a
    configuration whose stored momentum would invalidate the per-axis model
    carries that warning *with* its margins rather than beside them.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model, gains included.
    axis : int
        Body axis index, 0=x, 1=y, 2=z.
    sampled : bool, optional
        Analyse the discrete sampled-data loop at ``ControlPeriodSec``
        (default) rather than the continuous idealisation.
    integrator : bool, optional
        ``False`` analyses the saturated-loop case (:math:`K_i` frozen).
    computation_delay_cycles : int, optional
        Extra whole cycles of transport delay; only meaningful when ``sampled``.

    Returns
    -------
    AxisMargins
        The measured margins, with the coupling-ratio fields populated.
    """
    if sampled:
        loop = discrete_open_loop(
            vehicle,
            axis,
            integrator=integrator,
            computation_delay_cycles=computation_delay_cycles,
        )
    else:
        loop = open_loop(vehicle, axis, integrator=integrator)

    result = loop_margins(loop, AXES[axis])
    if not result.stable or not math.isfinite(result.gain_crossover_rad_s):
        return result
    ratio, limit = siso_coupling(vehicle, result.gain_crossover_rad_s)
    return replace(result, siso_coupling_ratio=ratio, siso_momentum_limit_nms=limit)

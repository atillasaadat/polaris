"""Everything the sizing analysis must assume, in one place, with its defaults.

A sizing result is a margin, and *a margin without its assumptions is not a
result* (``analysis/CLAUDE.md``). Some of what actuator sizing needs is not in
any config file — nobody writes down the launch vehicle's tip-off rate or how
often the operator intends to desaturate — so rather than burying those numbers
at their point of use, every one of them is a field here with a documented
default and a stated source. :meth:`SizingAssumptions.describe` renders them into
the report's assumptions block, so a reader of the output never has to open this
file to find out what was assumed.

The rule this module exists to enforce: **anything the config does not carry is
an explicit, surfaced assumption.** Nothing here is vehicle-specific — the
defaults are engineering practice and requirement values, and each is
overridable per analysis, which is what lets the same tool size a different
spacecraft.

Units
-----
SI throughout: rates [rad/s], times [s], torques [N·m], fractions [-].

References
----------
Wertz, Everett & Puschell, *Space Mission Engineering: The New SMAD*, §19.2
[wertz2011] — actuator sizing practice and the disturbance-torque budget.
Sidi, *Spacecraft Dynamics and Control*, §7.3–§7.5 [sidi1997] — momentum
storage, secular accumulation and magnetic unloading.
Design doc §5.3 (disturbance torques), §7 (actuators), §12 (analysis tools).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Required margin factor: capability must be at least this times the
#: requirement for a criterion to pass [-]. 30 % is the repo's standing sizing
#: margin (design doc §22.2 requires margin reporting; 1.3 is the actuator-sizing
#: convention in [wertz2011] §19.2 for a design that must still close after mass
#: growth and end-of-life degradation).
MARGIN = 1.30

#: Largest capability-to-requirement ratio that still counts as *sized* [-].
#: Above it the array is not conservatively sized, it is the wrong unit class:
#: an order of magnitude is beyond any growth allowance, and the mass, power and
#: cost bought are not buying capability the vehicle can use. Deliberately a
#: round engineering line rather than a value fitted to any measurement, per the
#: repo's rule that a threshold is never tuned to the number it judges.
MAX_OVERSIZING = 10.0

#: Orbit-average efficiency of the cross-product magnetic control law [-].
#: Only the dipole component perpendicular to **B** produces torque, and only
#: the torque component along the desired direction unloads momentum, so the law
#: :math:`\mathbf m = k\,\Delta\mathbf h\times\mathbf B/\lVert\mathbf B\rVert^2`
#: delivers :math:`-k(I - \hat{\mathbf b}\hat{\mathbf b}^\top)\Delta\mathbf h`.
#: Averaged over a field direction that samples the sphere reasonably —
#: :math:`\langle\hat{\mathbf b}\hat{\mathbf b}^\top\rangle = I/3` — that is
#: :math:`2/3` of the ideal. Cited as the standard result in [sidi1997] §7.5 and
#: [camillo1980]. A near-equatorial orbit, where **B** stays close to one
#: direction, does worse and this should be lowered for one.
MTQ_ORBIT_EFFICIENCY = 2.0 / 3.0


@dataclass(frozen=True)
class SizingAssumptions:
    """The assumed inputs, with defaults that are practice rather than vehicle.

    Attributes
    ----------
    tipoff_rate_radps : float
        Body-rate magnitude at separation [rad/s]. Default 5 deg/s, the value
        REQ-ACTL-001 is written on and the campaign flies; the config carries no
        tip-off rate because it is a property of the launch vehicle.
    desat_interval_s : float or None
        Time between momentum desaturations [s]. ``None`` — the default — uses
        one orbital period, i.e. the standard "unload once a lap" concept of
        operations. This is the single biggest lever on required wheel momentum
        and it is an *operational* choice, so it is stated rather than derived.
    detumble_budget_s : float or None
        Time allowed to remove the tip-off momentum magnetically [s]. ``None``
        uses one orbital period. Detumble is field-geometry limited, so a budget
        shorter than an orbit is not something rod sizing alone can buy.
    slew_rate_radps : float or None
        Commanded slew rate the wheels must supply momentum for [rad/s].
        ``None`` — the default — means the config declares no slew requirement,
        and the tool reports the slew rate the design *supports* as a diagnostic
        instead of judging one it was never given.
    detumble_enter_fraction, detumble_exit_fraction : float
        Fractions of the usable wheel momentum envelope that bracket the B-dot
        detumble mode, as body momentum :math:`|J\omega|` [-].

        Entry defaults to **1.0**: the vehicle is tumbling, for the purposes of
        this control system, exactly when its body momentum exceeds what the
        wheel array is *certified* to hold. Below that the wheels are inside the
        regime the pointing loop's margins were computed for and there is
        nothing for the rods to do.

        Exit defaults to **0.75**: the handover leaves the wheels a quarter of
        the envelope for the disturbance environment and the pointing transient
        that follows. Note which envelope — ``usable_momentum_nms`` is
        ``min(inscribed hardware capacity, MomentumEnvelopeNms)``, and on the
        reference vehicle the second is 15 % of the first. Handing over at 75 %
        of the *hardware* capacity would be 3.2x outside the SISO validity
        boundary: the array would physically hold the momentum, the certified
        margins would describe a different vehicle, and the first thing the
        wheels would have to do is desaturate rather than point.

        The 33 % band between them is the hysteresis. It is wide because the
        alternative is worse in both directions: too narrow and the momentum
        estimate's own noise walks the vehicle across it, and a mode that
        re-engages the rods on noise runs them continuously.
    secular_fraction_gg, secular_fraction_aero, secular_fraction_srp,
    secular_fraction_mag : float
        Fraction of each disturbance torque treated as **secular** (accumulating
        without bound) rather than **cyclic** (averaging to zero over an orbit)
        [-]. The defaults are the standard treatment for an LVLH (nadir-pointing)
        reference attitude: a torque fixed in the *orbiting* frame — gravity
        gradient, aerodynamic ram, and the magnetic term whose field reverses
        twice a lap — accumulates no net inertial momentum over a full orbit and
        is therefore cyclic, while SRP acts along the essentially
        inertially-fixed Sun direction and is therefore secular. **Change these
        for an inertially-pointing vehicle**, where the aerodynamic term becomes
        the secular one.
    mtq_efficiency : float
        See :data:`MTQ_ORBIT_EFFICIENCY`.
    margin : float
        See :data:`MARGIN`.
    max_oversizing : float
        See :data:`MAX_OVERSIZING`.
    """

    tipoff_rate_radps: float = float(np.deg2rad(5.0))
    desat_interval_s: float | None = None
    detumble_budget_s: float | None = None
    slew_rate_radps: float | None = None
    detumble_enter_fraction: float = 1.0
    detumble_exit_fraction: float = 0.75
    secular_fraction_gg: float = 0.0
    secular_fraction_aero: float = 0.0
    secular_fraction_srp: float = 1.0
    secular_fraction_mag: float = 0.0
    mtq_efficiency: float = MTQ_ORBIT_EFFICIENCY
    margin: float = MARGIN
    max_oversizing: float = MAX_OVERSIZING

    def desat_interval(self, orbit_period_s: float) -> float:
        """Desaturation interval [s], defaulting to one orbit.

        Parameters
        ----------
        orbit_period_s : float
            The vehicle's orbital period [s].

        Returns
        -------
        float
        """
        return (
            orbit_period_s if self.desat_interval_s is None else self.desat_interval_s
        )

    def detumble_budget(self, orbit_period_s: float) -> float:
        """Detumble time budget [s], defaulting to one orbit.

        Parameters
        ----------
        orbit_period_s : float
            The vehicle's orbital period [s].

        Returns
        -------
        float
        """
        return (
            orbit_period_s if self.detumble_budget_s is None else self.detumble_budget_s
        )

    def describe(self, orbit_period_s: float) -> tuple[str, ...]:
        """Render the assumptions in force, one line each, for the report.

        Parameters
        ----------
        orbit_period_s : float
            The vehicle's orbital period [s], used to resolve the ``None``
            defaults so the reader sees the number actually applied.

        Returns
        -------
        tuple of str
        """
        slew = (
            "no commanded slew rate in the config: slew agility is reported "
            "parametrically, not judged"
            if self.slew_rate_radps is None
            else f"commanded slew rate {np.degrees(self.slew_rate_radps):.3g} deg/s"
        )
        return (
            f"Margin convention: capability must exceed requirement by "
            f"{100.0 * (self.margin - 1.0):.0f}% for a criterion to pass.",
            f"Tip-off rate {np.degrees(self.tipoff_rate_radps):.3g} deg/s "
            "(REQ-ACTL-001's value; a launch-vehicle property, not a config one).",
            f"Desaturation interval {self.desat_interval(orbit_period_s):.0f} s "
            f"({self.desat_interval(orbit_period_s) / orbit_period_s:.2g} orbits) — "
            "an operational choice, and the largest single lever on required "
            "wheel momentum.",
            f"Detumble budget {self.detumble_budget(orbit_period_s):.0f} s.",
            slew,
            "Secular/cyclic split assumes an LVLH (nadir-pointing) reference "
            "attitude: gravity-gradient, aerodynamic and magnetic torques are "
            "fixed in the orbiting frame and accumulate no net inertial momentum "
            "over a lap (cyclic); SRP acts along the inertially-fixed Sun "
            "direction (secular). An inertially-pointing vehicle inverts this.",
            f"Magnetic control efficiency {self.mtq_efficiency:.3g} — the "
            "orbit-average of the cross-product law's (I - b b^T) projection.",
            f"Detumble engages at {100.0 * self.detumble_enter_fraction:.0f}% and "
            f"hands over at {100.0 * self.detumble_exit_fraction:.0f}% of the usable "
            "wheel momentum envelope, as body momentum |J*omega|.",
            f"An actuator more than {self.max_oversizing:g}x its largest driver is "
            "reported as oversized: past that it is the wrong unit class rather "
            "than a conservative choice.",
        )

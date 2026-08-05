"""Linearised attitude plant and the shipped control law, as transfer functions.

Built on **numpy and scipy only** — a loop here is a pair of polynomials plus a
sample period (:class:`Loop`), evaluated with ``numpy.polyval`` and discretised
with ``scipy.signal.cont2discrete``. Everything the margin work needs is
polynomial arithmetic and root finding, so there is no control-systems library
in the dependency graph.

What is modelled, and what is not
---------------------------------
The rigid-body attitude kinematics/dynamics are linearised about **zero body
rate** at an inertial hold, which is the regime the pointing law is designed
for (``lib/gnc/attitude_pid.hpp``). With :math:`\\delta\\boldsymbol\\theta` the
small error rotation and :math:`\\delta\\boldsymbol\\omega` the rate deviation,

.. math::

   \\dot{\\delta\\boldsymbol\\theta} = \\delta\\boldsymbol\\omega, \\qquad
   J\\,\\dot{\\delta\\boldsymbol\\omega} = \\boldsymbol\\tau ,

so the gyroscopic term :math:`\\boldsymbol\\omega\\times J\\boldsymbol\\omega`
and the §5.3 environmental torques drop out at first order and each body axis
of a diagonal :math:`J` is an independent double integrator
:math:`1/(J_{ii}s^2)`. Every assumption this package makes is one of:

* **Small angle.** Errors are radians-small; the ``sgn(δq₀)`` short-way-round
  branch of the flight law is inactive and irrelevant here.
* **Unsaturated.** Neither ``PidMaxTorqueNm`` nor the per-wheel torque box is
  reached, so the direction-preserving scale in the flight law and in the
  allocator is the identity. Margins are a property of the linear regime; a
  saturated loop has no gain margin, it has a describing function.
* **Integrator active.** The conditional integrator freezes on saturation
  (Åström & Murray §11.4 [astrom2008]); the loop shape below is the
  *unfrozen* one. The frozen loop is the same expression with
  :math:`K_i = 0`, which :func:`open_loop` will produce if asked, and which is
  strictly better behaved at low frequency — so analysing the unfrozen loop is
  the conservative choice, not a convenient one.
* **Wheels only.** The MTQ/MAG duty-cycle interlock (§7) does not appear: it
  gates *magnetometer samples*, not the wheel torque path, and the pointing
  loop closes on wheels.
* **Ideal actuator and sensor.** No wheel-motor lag, no gyro dynamics, no
  flexible modes. The reference vehicle is a rigid 6U with no deployables, and
  the RW jitter tones sit two decades above the loop bandwidth (§8.5).

**SISO validity boundary.** Per-axis analysis assumes a **near-diagonal**
inertia tensor and **small stored wheel momentum**. Both are checked rather
than assumed: :func:`analysis.control.vehicle.load_vehicle` refuses a config
with products of inertia, and :func:`siso_coupling` measures the gyroscopic
cross-coupling the stored momentum would produce at the loop crossover. See
design doc §8.5, "SISO validity boundary and MIMO roadmap", for the committed
triggers — CMGs, flexible appendages, momentum-biased operation, LQR/MPC —
each of which makes the plant genuinely multivariable and needs the MIMO
tooling the push that hits one of them will build.

The controller
--------------
``AttitudePid`` computes
:math:`\\boldsymbol\\tau = K_p\\,\\delta\\boldsymbol\\theta + K_i\\!\\int\\!
\\delta\\boldsymbol\\theta\\,dt + K_d(\\boldsymbol\\omega_\\mathrm{ref} -
\\hat{\\boldsymbol\\omega})`. At an inertial hold
(:math:`\\boldsymbol\\omega_\\mathrm{ref}=0`) the derivative term is
:math:`-K_d\\hat{\\boldsymbol\\omega}` — the **measured** rate, not a numerical
differentiation of the error — so there is no derivative filter pole to model
and the ideal :math:`K_d s` term is exact rather than an idealisation. Breaking
the loop at the actuator gives, per axis,

.. math::

   L(s) = \\frac{K_d s^2 + K_p s + K_i}{J\\,s^3},

a negative-unity-feedback loop of relative degree one and **type 3**. The
integrator on a double integrator is what makes it *conditionally stable*: the
phase starts at −270° and rises through −180°, so the loop tolerates an
unbounded gain increase and only a bounded gain **decrease**. See
:mod:`analysis.control.margins`, which reports both directions rather than the
single signed number a naive margin routine returns.

References
----------
Wie, *Space Vehicle Dynamics and Control*, 2nd ed., §7 [wie2008] — rigid-body
attitude dynamics, the linearised regulator, and the momentum-bias coupling the
SISO boundary is drawn against.
Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
Control*, §7.2 [markley2014] — the quaternion-feedback law this discretises.
Åström & Murray, *Feedback Systems*, §11.4 [astrom2008] — conditional
integration, the assumption behind the "integrator active" caveat.
Franklin, Powell & Workman, *Digital Control of Dynamic Systems*, 3rd ed.,
§4–§6 [franklin1998] — zero-order-hold equivalence and the sampled-data loop.
Design doc §8.5 (control), §13 (analysis).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete, ss2tf

from analysis.control.vehicle import PidGains, Vehicle

#: Body-axis names, in index order.
AXES = ("x", "y", "z")

#: Largest gyroscopic coupling ratio at which the per-axis (SISO) analysis is
#: still trusted [-]. Above it the cross-axis term dominates the diagonal
#: dynamics at the loop crossover and a per-axis Bode plot describes a system
#: the vehicle is not. Ten percent is the usual "an order of magnitude down"
#: engineering line; there is no standard for it, which is why it is named here
#: rather than buried inside a comparison.
MAX_SISO_COUPLING_RATIO = 0.1


@dataclass(frozen=True)
class Loop:
    """A SISO loop transfer function as a pair of polynomials.

    Deliberately minimal: the margin work needs a frequency response, poles and
    a gain scaling, and each is three lines of numpy. Coefficients are in
    **descending** powers, the ``numpy.poly1d`` convention.

    Attributes
    ----------
    num : numpy.ndarray
        Numerator coefficients, descending powers of :math:`s` (or :math:`z`).
    den : numpy.ndarray
        Denominator coefficients, same convention.
    dt : float
        Sample period [s]. ``0.0`` means continuous time; anything positive
        makes this a discrete loop evaluated at :math:`z=e^{j\\omega T_s}`.
    """

    num: np.ndarray
    den: np.ndarray
    dt: float = 0.0

    @property
    def discrete(self) -> bool:
        """This is a discrete-time loop."""
        return self.dt > 0.0

    def response(self, omega: np.ndarray | float) -> np.ndarray:
        """Frequency response :math:`L(j\\omega)` or :math:`L(e^{j\\omega T_s})`.

        Parameters
        ----------
        omega : numpy.ndarray or float
            Frequencies [rad/s].

        Returns
        -------
        numpy.ndarray
            Complex response, same shape as ``omega``.
        """
        w = np.asarray(omega, dtype=float)
        arg = np.exp(1j * w * self.dt) if self.discrete else 1j * w
        return np.polyval(self.num, arg) / np.polyval(self.den, arg)

    def closed_loop_poles(self) -> np.ndarray:
        """Poles of :math:`L/(1+L)` — the negative-unity-feedback closed loop.

        Returns
        -------
        numpy.ndarray
            Complex roots of ``den + num``.
        """
        return np.roots(np.polyadd(self.den, self.num))

    def is_stable(self) -> bool:
        """Closed-loop stability, judged in the right region for the time base.

        The open left half-plane for a continuous loop, the open unit disc for a
        discrete one. Every margin in this package presupposes this is true at
        the nominal gains — a "margin" around an unstable design is meaningless,
        so :func:`analysis.control.margins.loop_margins` checks it first.

        Returns
        -------
        bool
            True when every closed-loop pole is strictly inside the stable
            region.
        """
        poles = self.closed_loop_poles()
        if self.discrete:
            return bool(np.all(np.abs(poles) < 1.0))
        return bool(np.all(poles.real < 0.0))

    def scaled(self, gain: float) -> Loop:
        """This loop with its gain multiplied by @p gain.

        Parameters
        ----------
        gain : float
            Multiplicative loop-gain change [-].

        Returns
        -------
        Loop
            A new loop; this one is unchanged.
        """
        return Loop(self.num * gain, self.den, self.dt)

    def delayed(self, cycles: int) -> Loop:
        """This loop with @p cycles whole sample periods of transport delay.

        Parameters
        ----------
        cycles : int
            Whole cycles of :math:`z^{-1}`; must be non-negative, and this loop
            must be discrete.

        Returns
        -------
        Loop
            A new loop with ``cycles`` extra poles at the origin.

        Raises
        ------
        ValueError
            On a negative count, or on a continuous loop — a delay in whole
            cycles has no meaning without a sample period.
        """
        if cycles < 0:
            raise ValueError("delay cycles must be non-negative")
        if cycles == 0:
            return self
        if not self.discrete:
            raise ValueError("a delay in whole cycles needs a discrete loop")
        return Loop(self.num, np.polymul(self.den, [1.0] + [0.0] * cycles), self.dt)


def attitude_state_space(
    vehicle: Vehicle, wheels: tuple[int, ...] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled six-state attitude model driven by the wheel array.

    State is :math:`x = [\\delta\\boldsymbol\\theta;\\,
    \\delta\\boldsymbol\\omega]` (body frame, rad and rad/s), input is the
    per-wheel commanded torque [N·m]:

    .. math::

       A = \\begin{bmatrix} 0 & I \\\\ 0 & 0\\end{bmatrix}, \\qquad
       B = \\begin{bmatrix} 0 \\\\ J^{-1}A_w \\end{bmatrix},

    with :math:`A_w` the wheel torque-authority axes — the *negated* spin axes,
    per :meth:`analysis.control.vehicle.Vehicle.wheel_torque_axes`.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    wheels : tuple of int, optional
        Wheel indices to keep, for failure subsets. ``None`` keeps all.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B)`` with shapes ``(6, 6)`` and ``(6, k)``.
    """
    axes = vehicle.wheel_torque_axes(wheels)
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    b = np.vstack((np.zeros((3, axes.shape[1])), vehicle.inertia_inverse() @ axes))
    return a, b


def torque_state_space(vehicle: Vehicle) -> tuple[np.ndarray, np.ndarray]:
    """Six-state attitude model driven by the **body torque** [N·m], 3 inputs.

    The actuator-agnostic form (REQ-ACTL-003): control emits a body torque and
    the allocation layer is downstream, so this is the plant the control law
    sees and the one the per-axis loops decompose.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A, B)`` with shapes ``(6, 6)`` and ``(6, 3)``.
    """
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    b = np.vstack((np.zeros((3, 3)), vehicle.inertia_inverse()))
    return a, b


def axis_plant(inertia_kgm2: float) -> Loop:
    """Per-axis rigid-body plant :math:`1/(J s^2)`, torque [N·m] to angle [rad].

    Parameters
    ----------
    inertia_kgm2 : float
        The axis's principal moment of inertia [kg·m²].

    Returns
    -------
    Loop
        Continuous-time.
    """
    return Loop(np.array([1.0]), np.array([inertia_kgm2, 0.0, 0.0]))


def pid_numerator(gains: PidGains) -> np.ndarray:
    """The shipped PID as :math:`(K_d s^2 + K_p s + K_i)/s`, numerator only.

    Exact for the flight law at an inertial hold: the derivative term acts on
    the *measured* body rate, which for a fixed reference is the exact
    derivative of the error rotation, so no derivative filter pole appears.

    Parameters
    ----------
    gains : PidGains
        The committed tuning.

    Returns
    -------
    numpy.ndarray
        ``[Kd, Kp, Ki]``; the matching denominator is ``[1, 0]``.
    """
    return np.array([gains.kd_nm_per_radps, gains.kp_nm_per_rad, gains.ki_nm_per_rad_s])


def open_loop(vehicle: Vehicle, axis: int, *, integrator: bool = True) -> Loop:
    """Continuous-time open loop :math:`L(s)=C(s)P(s)` for one body axis.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    axis : int
        Body axis index, 0=x, 1=y, 2=z.
    integrator : bool, optional
        ``False`` drops :math:`K_i`, which is the loop the flight law runs while
        the torque command is saturated (conditional integration).

    Returns
    -------
    Loop
        Continuous-time loop transfer function, negative unity feedback.
    """
    num = pid_numerator(vehicle.pid)
    if not integrator:
        num = np.array([num[0], num[1], 0.0])
    inertia = vehicle.principal_moments_kgm2[axis]
    return Loop(num, np.array([inertia, 0.0, 0.0, 0.0]))


def discrete_open_loop(
    vehicle: Vehicle,
    axis: int,
    *,
    integrator: bool = True,
    computation_delay_cycles: int = 0,
) -> Loop:
    """Sampled-data open loop for one body axis, at the GNC period.

    The loop runs at ``ControlPeriodSec`` (0.1 s on the reference vehicle) and
    the wheel command is held across the period, so the honest loop is a
    discrete one rather than the continuous shape with a hand-waved delay. It
    is built in **state space and converted once**, because the three feedback
    paths are not the same signal and do not combine cleanly as separate
    transfer functions:

    * :math:`\\theta` and :math:`\\omega` are both **measured** (the estimator
      publishes attitude and rate), so they are two sampled outputs of one
      zero-order-held plant — the derivative path is never a backward difference
      of the first;
    * the integral path is the flight code's forward-Euler accumulator, whose
      state is advanced **after** the demand is formed
      (``lib/gnc/attitude_pid.cpp``), so it is one extra state read before it is
      updated.

    Summing the three as separate transfer functions leaves an uncancelled
    pole-zero pair at :math:`z=1` — they share the plant's :math:`(z-1)` factors
    — which reads as a marginally unstable loop. Realising the whole thing as
    one three-state system and taking its characteristic polynomial once avoids
    that entirely [franklin1998].

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model; supplies both the gains and the sampling period.
    axis : int
        Body axis index, 0=x, 1=y, 2=z.
    integrator : bool, optional
        ``False`` drops :math:`K_i` (the saturated-loop case).
    computation_delay_cycles : int, optional
        Extra whole cycles of transport delay between sampling and actuation,
        as :math:`z^{-n}`. The flight topology samples and commands inside one
        10 Hz cycle, so the as-built value is ``0``; ``1`` is the pessimistic
        case worth checking against, and the margin report records which was
        used.

    Returns
    -------
    Loop
        Discrete-time loop transfer function with ``dt`` = the control period.
    """
    ts = vehicle.control_period_s
    inertia = vehicle.principal_moments_kgm2[axis]
    gains = vehicle.pid

    a_d, b_d, _, _, _ = cont2discrete(
        (
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0 / inertia]]),
            np.eye(2),
            np.zeros((2, 1)),
        ),
        ts,
        method="zoh",
    )

    if integrator:
        a = np.block(
            [[a_d, np.zeros((2, 1))], [np.array([[ts, 0.0]]), np.ones((1, 1))]]
        )
        b = np.vstack((b_d, np.zeros((1, 1))))
        c = np.array(
            [[gains.kp_nm_per_rad, gains.kd_nm_per_radps, gains.ki_nm_per_rad_s]]
        )
    else:
        a, b = a_d, b_d
        c = np.array([[gains.kp_nm_per_rad, gains.kd_nm_per_radps]])

    num, den = ss2tf(a, b, c, np.zeros((1, 1)))
    loop = Loop(np.asarray(num[0], dtype=float), np.asarray(den, dtype=float), ts)
    return loop.delayed(computation_delay_cycles)


def siso_coupling(vehicle: Vehicle, crossover_rad_s: float) -> tuple[float, float]:
    """Gyroscopic cross-coupling the stored wheel momentum causes at crossover.

    Euler's equation with a wheel array carrying stored momentum
    :math:`\\bar{\\mathbf h}` is :math:`J\\dot{\\boldsymbol\\omega} +
    \\boldsymbol\\omega\\times(J\\boldsymbol\\omega + \\bar{\\mathbf h}) =
    \\boldsymbol\\tau`; linearised about zero body rate the cross term is
    :math:`-[\\bar{\\mathbf h}\\times]\\delta\\boldsymbol\\omega`, which couples
    all three axes and which the per-axis models here do **not** contain.
    Comparing it against the diagonal term :math:`J\\dot{\\delta\\boldsymbol
    \\omega}` at the loop crossover gives the dimensionless ratio

    .. math::

       \\rho = \\frac{\\|\\bar{\\mathbf h}\\|}{J_{\\min}\\,\\omega_c},

    i.e. the reciprocal of how far the crossover sits above the momentum corner.
    Below :data:`MAX_SISO_COUPLING_RATIO` the per-axis analysis describes the
    vehicle; above it the roll/yaw pair is a coupled nutation system and
    per-axis Bode gets it wrong (design doc §8.5, MIMO roadmap).

    The momentum used is the array's **capacity**, not an operating point: the
    analysis produces a bound, and what a wheel is *allowed* to store is what
    the config states. The bound is the largest body-axis momentum the array can
    hold, :math:`\\max_j \\sum_i |W_{ji}|\\,h_{\\max}`.

    Parameters
    ----------
    vehicle : Vehicle
        The as-flown model.
    crossover_rad_s : float
        The loop's gain-crossover frequency [rad/s].

    Returns
    -------
    tuple of float
        ``(ratio, momentum_limit_nms)`` — the coupling ratio at full stored
        momentum [-], and the stored momentum [N·m·s] at which the ratio would
        reach :data:`MAX_SISO_COUPLING_RATIO`, i.e. the largest momentum bias
        the per-axis analysis covers.
    """
    capacity = float(
        np.max(np.sum(np.abs(vehicle.wheel_spin_axes), axis=1))
        * vehicle.wheel_max_momentum_nms
    )
    diagonal = float(np.min(vehicle.principal_moments_kgm2)) * crossover_rad_s
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        return float("inf"), 0.0
    return capacity / diagonal, MAX_SISO_COUPLING_RATIO * diagonal

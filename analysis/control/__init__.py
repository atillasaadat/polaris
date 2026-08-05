"""Linear control analysis of the as-flown Polaris attitude loop (§8.5, §13).

Post-processing tools that answer three questions about the configuration the
vehicle actually ships, driven from the same ``config/spacecraft/*.yaml`` the
flight software is tuned with:

* **Can it be controlled?** — Kalman rank and finite-horizon controllability
  Gramians for the four-wheel pyramid, every 3-of-4 wheel-failure subset, and
  the magnetorquer-only case, whose instantaneous rank-2 actuation is only
  redeemed by the field turning over an orbit
  (:mod:`analysis.control.controllability`, REQ-ACTL-007).
* **Can it be estimated?** — observability of the (attitude error, gyro bias)
  model under two vector measurements, including the near-parallel geometry the
  §8.1 TRIAD gate refuses and the eclipse case
  (:mod:`analysis.control.observability`, REQ-ACTL-008).
* **How much margin does the loop have?** — classical and disk margins of the
  sampled-data pointing loop at the committed gains
  (:mod:`analysis.control.margins`, REQ-ACTL-006).

Built on **numpy and scipy only**. The loop is a pair of polynomials
(:class:`analysis.control.plant.Loop`), discretised with
``scipy.signal.cont2discrete`` and evaluated with ``numpy.polyval``; the margin
extraction, the disk margin and the Gramians are all ours, which is why
``tests/analysis/`` validates the margin machinery against closed-form cases
before it is pointed at the vehicle.

This is linear-systems analysis on matrices read from config. It deliberately
computes **no** quaternion, frame or propagation math: per ``analysis/CLAUDE.md``
that belongs to the flight/sim C++ and reaches Python through ``bindings/``.

**Validity boundary.** The per-axis (SISO) models assume a near-diagonal inertia
tensor and small stored wheel momentum; both are checked rather than assumed
(:func:`analysis.control.plant.siso_coupling`). See design doc §8.5, "SISO
validity boundary and MIMO roadmap", for the committed triggers that require
multivariable analysis instead.

Everything here is importable and side-effect-free; the only writes are the
figures and the rendered report :mod:`analysis.control.plots` emits into a
caller-supplied directory.
"""

from analysis.control.controllability import (
    ControllabilityResult,
    mtq_controllability,
    orbit_averaged_mtq_controllability,
    wheel_controllability,
    wheel_failure_subsets,
)
from analysis.control.margins import (
    MAX_SENSITIVITY_PEAK,
    MIN_GAIN_MARGIN_DB,
    MIN_PHASE_MARGIN_DEG,
    PREFERRED_PHASE_MARGIN_DEG,
    AxisMargins,
    GridWindowError,
    axis_margins,
    disk_margin,
    frequency_grid,
    loop_margins,
)
from analysis.control.observability import (
    ObservabilityResult,
    information_ratio,
    two_vector_observability,
    vector_geometry_sweep,
)
from analysis.control.plant import (
    MAX_SISO_COUPLING_RATIO,
    Loop,
    attitude_state_space,
    axis_plant,
    discrete_open_loop,
    open_loop,
    pid_numerator,
    siso_coupling,
    torque_state_space,
)
from analysis.control.report import (
    MarginReport,
    control_analysis_report,
    margin_report,
)
from analysis.control.vehicle import Vehicle, load_vehicle

__all__ = [
    "MAX_SENSITIVITY_PEAK",
    "MAX_SISO_COUPLING_RATIO",
    "MIN_GAIN_MARGIN_DB",
    "MIN_PHASE_MARGIN_DEG",
    "PREFERRED_PHASE_MARGIN_DEG",
    "AxisMargins",
    "ControllabilityResult",
    "GridWindowError",
    "Loop",
    "MarginReport",
    "ObservabilityResult",
    "Vehicle",
    "attitude_state_space",
    "axis_margins",
    "axis_plant",
    "control_analysis_report",
    "discrete_open_loop",
    "disk_margin",
    "frequency_grid",
    "information_ratio",
    "load_vehicle",
    "loop_margins",
    "margin_report",
    "mtq_controllability",
    "open_loop",
    "orbit_averaged_mtq_controllability",
    "pid_numerator",
    "siso_coupling",
    "torque_state_space",
    "two_vector_observability",
    "vector_geometry_sweep",
    "wheel_controllability",
    "wheel_failure_subsets",
]

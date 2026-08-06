"""The committed momentum envelope must sit inside the SISO-validity bound.

REQ-ACTL-009. This is the one test that closes the loop between Push 55's linear
analysis and Push 56's flight momentum management: the pointing loop's certified
margins (REQ-ACTL-006) are computed per axis, and that model is only valid while
the stored wheel momentum is small enough for the gyroscopic coupling
:math:`\\omega\\times(J\\omega + h)` to stay negligible at the loop crossover
(design doc §8.5, "SISO validity boundary"). The flight parameter
``MomentumEnvelopeNms`` is what keeps the vehicle inside that range, so a config
change that raised it past the bound would silently invalidate the margin
evidence the same config ships.

Nothing here transcribes a number. The bound is recomputed from the committed
YAML through :func:`analysis.control.plant.siso_coupling` — the same function the
margin report warns from — so the analysis and the vehicle cannot drift apart in
the direction that passes.

Also covers the module's command-line gate (``python -m analysis.control``),
which is the pre-simulation design check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.control import margin_report
from analysis.control.plant import MAX_SISO_COUPLING_RATIO

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Fraction of the SISO-validity bound the committed envelope must stay under.
#: A threshold *at* the bound would be a design with no margin at all against the
#: assumption its own margins are computed under, and the bound moves whenever
#: the gains or the inertia move.
ENVELOPE_MARGIN_FRACTION = 0.9


def _siso_momentum_bound_nms(vehicle) -> float:
    """Smallest per-axis stored momentum the SISO analysis is valid to [N·m·s].

    The margin report evaluates the coupling ratio at each axis's own gain
    crossover; the binding bound is the smallest of the three, because the
    envelope is a single vehicle-level number and the analysis has to hold on
    every axis at once.
    """
    return min(axis.siso_momentum_limit_nms for axis in margin_report(vehicle).axes)


def test_committed_envelope_is_inside_the_siso_validity_bound(vehicle):
    """``MomentumEnvelopeNms`` must sit inside the momentum range the margins cover."""
    bound = _siso_momentum_bound_nms(vehicle)
    assert bound > 0.0
    assert vehicle.momentum_envelope_nms <= ENVELOPE_MARGIN_FRACTION * bound, (
        f"the committed momentum envelope {vehicle.momentum_envelope_nms:.3e} N.m.s "
        f"is not inside the {bound:.3e} N.m.s the per-axis margin analysis is valid "
        "over (design doc SS8.5, SISO validity boundary): either lower the envelope, "
        "or the pointing margins REQ-ACTL-006 states no longer describe this vehicle"
    )


def test_desaturation_acts_before_the_envelope_alarms(vehicle):
    """The action must sit inside the alarm, or the alarm names an unpreventable state."""
    assert vehicle.momentum_desat_enter_nms < vehicle.momentum_envelope_nms


def test_the_bound_is_the_coupling_ratio_it_claims_to_be(vehicle):
    """The bound is where the coupling ratio reaches its limit — checked, not assumed.

    Guards the test above against the bound quietly becoming some other quantity:
    at exactly the returned momentum the ratio must equal
    :data:`analysis.control.plant.MAX_SISO_COUPLING_RATIO`, since the ratio is
    linear in the stored momentum.
    """
    report = margin_report(vehicle)
    for axis in report.axes:
        implied = axis.siso_coupling_ratio * (
            axis.siso_momentum_limit_nms
            / (
                vehicle.wheel_max_momentum_nms
                * max(abs(vehicle.wheel_spin_axes).sum(axis=1))
            )
        )
        assert implied == pytest.approx(MAX_SISO_COUPLING_RATIO, rel=1e-9)


def test_command_line_gate_passes_on_the_committed_config(reference_config, tmp_path):
    """``python -m analysis.control <config>`` exits 0 and renders the report.

    A subprocess on purpose: the exit status *is* the interface, and asserting on
    it through :func:`main` in-process would not catch a module that cannot be
    imported as ``__main__``.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.control",
            str(reference_config),
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        # `tools/` on the path is how the config compiler's loader is reachable
        # outside pytest (see analysis/control/vehicle.py) — the same thing a
        # hand-run of this gate needs, so the test runs it the documented way.
        env={**os.environ, "PYTHONPATH": f"tools{os.pathsep}{REPO_ROOT}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Overall: PASS" in result.stdout
    assert (tmp_path / "control_analysis_report.txt").is_file()
    # The figures are written too — the standing convention is report *and* plots.
    assert list(tmp_path.glob("*.png"))

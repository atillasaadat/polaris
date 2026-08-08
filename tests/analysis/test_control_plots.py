"""Figure generation and config loading for the linear analysis (design doc §13).

A smoke suite: the figures are derived artifacts and nothing asserts on their
pixels, but a plotting call that raises would take the whole toolkit down at
the moment someone needed a picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from analysis.control import load_vehicle
from analysis.control.plots import write_all

REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "spacecraft" / "leo_smallsat.yaml"
)


def test_every_figure_and_the_text_report_are_written(vehicle, tmp_path):
    """Five figures plus the rendered report, into a caller-supplied directory.

    Both, always: the convention is that an analysis with a pass/fail criterion
    produces plots *and* a report. Nothing here asserts on pixels or parses the
    text for a verdict — the structured report is what the requirement rows
    assert on.
    """
    paths = write_all(vehicle, tmp_path, REFERENCE_CONFIG)
    assert len(paths) == 6
    for path in paths:
        assert path.parent == tmp_path
        assert path.stat().st_size > 0
    assert [p.suffix for p in paths] == [".png"] * 5 + [".txt"]
    assert str(REFERENCE_CONFIG) in paths[-1].read_text()


def test_the_vehicle_is_loaded_from_the_committed_config(vehicle):
    """Spot-check that config values arrive unaltered and in SI."""
    assert vehicle.name == "LEO-Smallsat-Ref"
    assert vehicle.wheel_spin_axes.shape == (3, 4)
    assert vehicle.mtq_axes.shape == (3, 3)
    # Momentum capacity is read from the hardware catalog the model_id resolves
    # to, read from the catalog rather than restated — the SISO boundary depends
    # on it, and the part changed (RW-X -> RW-S) in Push 60. What this asserts is
    # that the loader resolves the model_id through the catalog at all, which a
    # transcribed number could not distinguish from a hard-coded default.
    catalog = yaml.safe_load(
        (
            REFERENCE_CONFIG.parent.parent / "hardware" / "reaction_wheel" / "rws.yaml"
        ).read_text()
    )
    assert vehicle.wheel_max_momentum_nms == pytest.approx(
        catalog["params"]["max_momentum_nms"]
    )
    # Spin axes are unit vectors on the body diagonals; the torque authority is
    # their negation, which is the one sign the controller applies.
    assert np.linalg.norm(vehicle.wheel_spin_axes, axis=0) == pytest.approx(
        np.ones(4), rel=1e-4
    )
    assert vehicle.wheel_torque_axes() == pytest.approx(-vehicle.wheel_spin_axes)
    assert vehicle.wheel_torque_axes((0, 2)).shape == (3, 2)
    # Orbit: a 500 km circular orbit has a ~94.6 min period.
    assert vehicle.orbit.period_s == pytest.approx(5677.0, rel=1e-3)


def test_products_of_inertia_are_refused(tmp_path):
    """The per-axis models are stated for a diagonal tensor; enforce the premise."""
    text = REFERENCE_CONFIG.read_text().replace(
        "    izz: 0.10", "    izz: 0.10\n    ixy: 0.01", 1
    )
    skewed = tmp_path / "skewed.yaml"
    skewed.write_text(text)
    with pytest.raises(ValueError, match="products of inertia"):
        load_vehicle(skewed)


def test_an_axes_parameter_disagreeing_with_its_unit_count_is_refused():
    """A zero column inside the installed count, or a live one past it, is a defect."""
    from analysis.control.vehicle import _axes_from_flat

    with pytest.raises(ValueError, match="zero column"):
        _axes_from_flat([0.0] * 24, 4, "WheelAxesBody")
    with pytest.raises(ValueError, match="past the installed count"):
        _axes_from_flat([1.0, 0.0, 0.0] * 8, 4, "WheelAxesBody")
    with pytest.raises(ValueError, match="expected 24 values"):
        _axes_from_flat([1.0, 0.0, 0.0], 1, "WheelAxesBody")

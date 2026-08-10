"""The sizing tool works on a vehicle that is not the committed one (§12, §19.4).

Three properties, each of which broke or could break silently:

* **Every number tracks the config.** The tool exists to size *a* vehicle, not
  the one in this repository. :func:`test_a_variant_vehicle_carries_its_own_numbers`
  builds a config with a clearly different name, mass and inertia and asserts the
  report's provenance and criteria carry the variant's figures — and that no
  constant of the committed vehicle appears anywhere in the rendering.
* **The hardware catalog is found off-tree.** The default used to be
  ``<config>/../../hardware``, so a config at ``/tmp/alt.yaml`` resolved to
  ``/hardware`` and the tool died on a path the user never named.
* **Both output files are written under every flag combination.** The text
  report is the record; ``--no-plots`` skipped the figures and took it with them,
  contradicting the module docstring and ``analysis/CLAUDE.md``.

References
----------
Design doc §12, §19.3, §19.4; ``analysis/CLAUDE.md`` (the reporting convention).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from analysis.control.vehicle import (
    REPO_HARDWARE_DIR,
    load_vehicle,
    resolve_hardware_dir,
)
from analysis.sizing.html import write_html
from analysis.sizing.report import sizing_analysis, sizing_report

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed vehicle's identity and mass properties. If any of these leaks
#: into a report generated from a different config, the tool is describing the
#: wrong spacecraft.
COMMITTED_MARKERS = ("LEO-Smallsat-Ref", "12 kg", "diag(0.12, 0.12, 0.1)")

#: What the variant is given instead. Chosen far from the committed values so a
#: leak cannot hide inside rounding.
VARIANT_NAME = "Variant-Sizing-Probe"
VARIANT_MASS_KG = 48.0
VARIANT_INERTIA = {"ixx": 0.95, "iyy": 0.80, "izz": 0.55}


@pytest.fixture
def variant_config(tmp_path: Path, reference_config: Path) -> Path:
    """A config outside the repository tree, with a different vehicle in it.

    Written to ``tmp_path`` deliberately: that is off the ``config/spacecraft``,
    ``config/hardware`` sibling layout, so loading it also exercises the
    fallback in :func:`analysis.control.vehicle.resolve_hardware_dir`.
    """
    document = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    document["spacecraft"]["name"] = VARIANT_NAME
    document["spacecraft"]["mass_kg"] = VARIANT_MASS_KG
    document["spacecraft"]["inertia_kgm2"].update(VARIANT_INERTIA)
    target = tmp_path / "variant_vehicle.yaml"
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# Every number tracks the config
# --------------------------------------------------------------------------


def test_a_variant_vehicle_carries_its_own_numbers(variant_config, tmp_path):
    """The report and the page describe the config given, and nothing else.

    This is the test that keeps the tool reusable. A sizing tool that quietly
    mixes in the reference vehicle's inertia is worse than no tool: it produces
    a plausible number for a spacecraft nobody is building.
    """
    vehicle = load_vehicle(variant_config)
    assert vehicle.name == VARIANT_NAME
    assert vehicle.mass_kg == pytest.approx(VARIANT_MASS_KG)

    analysis = sizing_analysis(vehicle)
    report = sizing_report(vehicle, variant_config, analysis.assumptions, analysis)

    # Provenance: the variant's identity and mass properties, verbatim.
    rendered = report.format_text()
    assert VARIANT_NAME in report.title
    assert str(variant_config) == report.config_path
    assert "48 kg" in rendered
    assert "diag(0.95, 0.8, 0.55)" in rendered

    # Criteria: the drivers scale with the variant's inertia, which is ~8x the
    # committed vehicle's, so the tip-off driver cannot have kept its old value.
    tipoff = next(d for d in analysis.wheels.drivers if d.name.startswith("D1 "))
    assert tipoff.required_nms == pytest.approx(
        max(VARIANT_INERTIA.values()) * analysis.assumptions.tipoff_rate_radps
    )

    # And nothing of the committed vehicle survives into either rendering. The
    # inlined plotly bundle is stripped first: it is a third-party payload whose
    # SVG path data contains every short numeric string by coincidence, and
    # searching it would be searching a library, not this report.
    page = write_html(analysis, report, tmp_path / "page").read_text(encoding="utf-8")
    # IGNORECASE is load-bearing, not defensive style: a tag-stripping regex
    # that misses <SCRIPT> leaves the bundle in the text, and the leak search
    # below would then be scanning a third-party library and failing on its
    # coincidental digits rather than on this report.
    page = re.sub(r"<script\b.*?</script>", " ", page, flags=re.S | re.IGNORECASE)
    for marker in COMMITTED_MARKERS:
        assert marker not in rendered, f"committed-vehicle value {marker!r} leaked"
        assert marker not in page, f"committed-vehicle value {marker!r} leaked"


# --------------------------------------------------------------------------
# Hardware catalog resolution
# --------------------------------------------------------------------------


def test_the_hardware_catalog_is_found_for_a_config_outside_the_repo(variant_config):
    """A config anywhere on disk loads without a ``--hardware`` flag.

    The sibling layout does not exist under ``tmp_path``, so this can only
    succeed through the repository fallback, which is located relative to the
    package rather than the working directory.
    """
    assert not (variant_config.parent.parent / "hardware").exists()
    assert resolve_hardware_dir(variant_config) == REPO_HARDWARE_DIR
    assert load_vehicle(variant_config).name == VARIANT_NAME


def test_the_sibling_catalog_wins_over_the_repository_one(tmp_path):
    """A checkout carrying its own catalog is used in preference to this one."""
    sibling = tmp_path / "config" / "hardware"
    sibling.mkdir(parents=True)
    config = tmp_path / "config" / "spacecraft" / "vehicle.yaml"
    config.parent.mkdir(parents=True)
    config.touch()
    assert resolve_hardware_dir(config) == sibling
    # An explicit directory beats both.
    assert resolve_hardware_dir(config, tmp_path) == tmp_path


def test_a_missing_catalog_names_both_paths_and_the_flag(tmp_path, monkeypatch):
    """The failure says what was tried and what to do, not just "not found"."""
    monkeypatch.setattr(
        "analysis.control.vehicle.REPO_HARDWARE_DIR", tmp_path / "absent"
    )
    config = tmp_path / "loose.yaml"
    config.touch()
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_hardware_dir(config)
    message = str(excinfo.value)
    assert str(config.parent.parent / "hardware") in message
    assert str(tmp_path / "absent") in message
    assert "--hardware" in message


# --------------------------------------------------------------------------
# Both output files, under both flag combinations
# --------------------------------------------------------------------------


def _run(config: Path, out: Path, *flags: str) -> subprocess.CompletedProcess:
    """Run the CLI the way a user does, in a subprocess, and return the result."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.sizing",
            str(config),
            "--no-browser",
            "--out",
            str(out),
            *flags,
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'tools'}",
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("flags", [(), ("--no-plots",)])
def test_the_text_report_and_the_page_are_written_either_way(
    tmp_path, reference_config, flags
):
    """``--no-plots`` drops the figures and nothing else.

    The text rendering is the record — the module docstring and
    ``analysis/CLAUDE.md`` both say it is written either way — and it used to
    disappear with the figures because it was produced by the figure writer.
    """
    out = tmp_path / "out"
    result = _run(reference_config, out, *flags)
    assert result.returncode == 0, result.stderr
    assert (out / "index.html").is_file()
    assert (out / "sizing_report.txt").is_file()
    assert "ADCS actuator sizing" in (out / "sizing_report.txt").read_text()
    figures = sorted(path.name for path in out.glob("*.png"))
    assert figures == [] if flags else figures


def test_the_cli_runs_on_a_config_outside_the_repository(tmp_path, variant_config):
    """End to end, with no ``--hardware`` flag: the defect a user hit first."""
    out = tmp_path / "out"
    result = _run(variant_config, out)
    assert result.returncode in (0, 1), result.stderr
    assert "hardware library directory not found" not in result.stderr
    assert VARIANT_NAME in (out / "sizing_report.txt").read_text(encoding="utf-8")

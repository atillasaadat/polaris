Tools & Python
==============

The **config compiler** (``tools/configc/``, design doc §19.3) is the first
Python component: it validates the single-source-of-truth spacecraft/scenario
YAML against a Pydantic schema, resolves hardware ``model_id`` references against
the library in ``config/hardware/``, and emits provenance-stamped
F´-param / sim / analysis artifacts from **one resolved object**
(REQ-CFG-001/002/003). Run it with::

   PYTHONPATH=tools uv run python -m configc --help

The **GMAT golden harness** (``tools/gmat/``) and the **ephemeris fitter**
(``tools/ephem/``) generate committed reference fixtures; neither runs in the
normal test path (they need the GMAT binary / a NAIF kernel). See
:doc:`/guides/verification`.

The **fetch tools** refresh the committed external reference data under the
verbatim-data rule (design doc §3.7). Each downloads from the authoritative source, sanity-checks that the file
parses, and writes it **verbatim** — the parsing that flies happens on the C++
side, so these never emit a bespoke intermediate. They are ground-side and never
run in CI:

- ``tools/eop/`` — IERS ``finals.all.iau2000`` Earth-orientation product
  (mirrors tried in order); ``finals.py`` also exposes the column parser used to
  validate the download.
- ``tools/gravity/`` — the EGM2008 ``.gfc`` spherical-harmonic model, truncated
  to a requested maximum degree in its native format (the full model is ~100 MB).
- ``tools/igrf/`` — the IAGA IGRF-14 coefficient table (committed verbatim) plus
  ``golden``, which regenerates the derived field fixture with ``pyIGRF14``, the
  official IAGA reference implementation — a fetch input, never vendored.
- ``tools/spaceweather/`` — CelesTrak ``SW-All.csv``, the solar radio flux
  (F10.7) and geomagnetic (Ap) record NRLMSIS is driven by, parsed by
  ``sim/world/space_weather_file.cpp``.

``tools/dev/collect_gtest_trace.py`` is the C++ half of the traceability gate:
it converts GoogleTest JSON output (``--gtest_output=json:…``) into
``docs/_generated/verif_gtest.json``, the sphinx-needs external-needs file that
turns each test's ``RecordProperty("verifies", "REQ-…")`` into a ``verified by``
back-link in the RVTM. With no input files it writes an empty-but-valid file so
the docs build still resolves.

.. note::

   ``analysis/`` and ``bindings/`` (pybind11) have no modules yet; numpydoc /
   autodoc directives land here with the first analysis code, at which point
   this page documents the Python-facing API of the same C++ that flies.

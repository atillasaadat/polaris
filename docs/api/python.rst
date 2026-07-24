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

.. note::

   ``analysis/`` and ``bindings/`` (pybind11) have no modules yet; numpydoc /
   autodoc directives land here with the first analysis code, at which point
   this page documents the Python-facing API of the same C++ that flies.

API Reference
=============

Generated from source as code lands (design doc §21.3):

- **C++** (``lib/``, ``flight/``, ``sim/``) via Doxygen → Breathe.
- **Python** (``analysis/``, ``bindings/``) via numpydoc/autodoc.

Every documented symbol states its **units and frames** for physical quantities
and cites its source. This page currently covers the ``lib/`` foundations
(Phase 0, Push 2); the flight, sim, and Python APIs fill in as those layers
land.

.. note::

   The C++ API below is extracted from the in-source Doxygen comments by
   ``docs/Doxyfile`` (XML) and rendered by Breathe. Run ``tools/dev/build_docs.sh``
   with ``doxygen`` installed to populate it — the doc build is a ``-W`` gate, so
   a documented symbol missing its units/frames or citation fails CI.

C++ — ``lib/`` foundations
--------------------------

Constants registry — ``polaris::constants``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shared physical-constants registry (WGS84, time offsets, *c*); every value is
sourced (design doc §3.6).

.. doxygennamespace:: polaris::constants
   :members:

Math — ``polaris::math``
^^^^^^^^^^^^^^^^^^^^^^^^^

Frame-tagged ``Vec3<Frame>`` vectors, the JPL scalar-first ``Quaternion``, and
the geometric frame tags. Mixing frames across a boundary is a compile error
(Golden Rule 4).

.. doxygennamespace:: polaris::math
   :members:

Time — ``polaris::time``
^^^^^^^^^^^^^^^^^^^^^^^^^

The onboard time foundation (design doc §3.2): the TAI master clock and the
strongly-typed uniform scales (GPS/TT/TDB), the signed-nanosecond ``Duration``,
the fixed-capacity leap-second table, ground-only UTC derivation, and the
TT↔TDB periodic-term conversion that feeds the planetary ephemerides.

.. doxygennamespace:: polaris::time
   :members:

Ephemeris — ``polaris::ephemeris``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Onboard Sun/Moon/planet positions (design doc §11.3): the flight-side Chebyshev
evaluator (``ChebyshevSegment`` + ``evaluate``) that reads ground-generated
coefficient sets, and the fixed-capacity ``EphemerisTable`` that selects the
covering interval. Queried at **TDB**; positions are ECI/J2000 metres. SPICE is
ground/sim-only and never linked into flight (REQ-CDH-002).

.. doxygennamespace:: polaris::ephemeris
   :members:

Canonical state — ``polaris::state``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The single navigation product (design doc §8.0): ``EstimatedState`` (onboard —
attitude/rates/position/velocity with frame tags, biases, 15-state covariance,
per-field validity, estimation mode) and ``TruthState`` (sim-only). The
truth-vs-onboard split is a type distinction, so flight code cannot consume
truth.

.. doxygennamespace:: polaris::state
   :members:

Python
------

No Python modules under ``analysis/`` or ``bindings/`` yet; numpydoc/autodoc
directives land here with the first analysis code.

The **config compiler** (``tools/configc/``, design doc §19.3) is the first
Python tool: it validates the single-source-of-truth spacecraft/scenario YAML
against a Pydantic schema, resolves hardware model-IDs against the library in
``config/hardware/``, and emits provenance-stamped F´-param / sim / analysis
artifacts from one resolved object (REQ-CFG-001/002/003). Run it with
``PYTHONPATH=tools uv run python -m configc --help``.

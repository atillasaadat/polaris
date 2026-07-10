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
strongly-typed uniform scales (GPS/TT), the signed-nanosecond ``Duration``, the
fixed-capacity leap-second table, and ground-only UTC derivation.

.. doxygennamespace:: polaris::time
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

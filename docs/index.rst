Polaris GNC — Documentation
===========================

**Polaris** is a from-scratch spacecraft Guidance, Navigation & Control software
suite: flight software in F´ (F Prime) / C++, a high-fidelity 6DOF truth
simulation, a shared C++ library, and Python analysis tooling — verified against
GMAT golden data and Monte Carlo campaigns.

This site is generated from the code and the requirements set. It is a CI gate:
broken docstrings, missing citations, or unverified baselined requirements fail
the build (``sphinx-build -W``).

.. note::

   The authoritative design baseline is
   ``docs/design/Polaris_Design_Document.md`` (read it before non-trivial work).
   It is intentionally not rendered into this site yet.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   requirements/index
   glossary
   references
   api/index

Conventions at a glance
-----------------------

- **Time:** onboard master clock is **TAI** (int64 ns); GNSS GPS time + ECEF
  converted on ingest (``TAI = GPS + 19 s``).
- **Attitude:** quaternions are **JPL, scalar-first** ``[q0, q1, q2, q3]``,
  ``q0 >= 0``.
- **Units:** **SI everywhere** inside FSW and ``lib/``.
- **Frames:** every vector/quaternion is frame-tagged; mixing frames across a
  boundary is a compile error.
- **Provenance:** every algorithm/model cites a source in :doc:`references`.

Indices
-------

* :ref:`genindex`
* :ref:`search`

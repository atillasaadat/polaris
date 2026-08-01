API Reference
=============

Generated from the in-source documentation comments (design doc §21.3):

- **C++** (``lib/``, ``sim/``) via Doxygen → Breathe. Every documented symbol
  states its **units and frames** for physical quantities and cites its source;
  a symbol missing either fails the ``-W`` docs build.
- **Python** (``tools/``, and ``analysis/`` / ``bindings/`` as they land) via
  numpydoc / autodoc.

The reference is grouped by layer. For task-oriented walkthroughs — configuring
a vehicle, adding a sensor, running the closed loop — start with the
:doc:`/guides/index`.

.. note::

   Run ``tools/dev/build_docs.sh`` with ``doxygen`` installed to populate the
   C++ pages; the doc build extracts them from ``docs/Doxyfile`` (XML) and
   renders them with Breathe.

Shared library — ``lib/``
-------------------------

The flight/sim-shared foundations: math, frames, time, canonical state,
constants, the onboard environment/ephemeris/RNG, and the GNC algorithms.

.. toctree::
   :maxdepth: 1

   constants
   math
   frames
   time
   state
   ephemeris
   environment
   gnc
   random

Truth simulation — ``sim/``
---------------------------

The plant the flight software flies against: 6DOF dynamics, environment models,
sensor and actuator truth models, the config→models bridge, and the closed loop.

.. toctree::
   :maxdepth: 1

   sim_dynamics
   sim_world
   sim_sensors
   sim_actuators
   sim_scenario
   sim_io

Tools & Python
--------------

.. toctree::
   :maxdepth: 1

   python

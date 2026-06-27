API Reference
=============

The API reference is generated from source as code lands:

- **C++** (``lib/``, ``flight/``, ``sim/``) via Doxygen → Breathe.
- **Python** (``analysis/``, ``bindings/``) via numpydoc/autodoc.

Both are configured (see ``docs/conf.py`` and ``docs/Doxyfile``) but not yet
populated — there is no code in Phase 0, Push 1. This page fills in starting with
the ``lib/`` foundations (Push 2): typed vectors, the constants registry, the
canonical state structs, and the quaternion/frames/time libraries.

.. note::

   Every documented symbol states **units and frames** for physical quantities
   and cites its source (design doc §21.3).

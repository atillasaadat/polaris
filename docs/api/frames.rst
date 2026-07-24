Frames — ``polaris::frames``
============================

The one time-dependent frame transform in the repo: the IAU 2006/2000A **ECI↔ECEF** reduction (via ERFA) and the IERS Earth-orientation table it reads. Every GNSS fix and every ground-track/geopotential computation routes through here — no ad-hoc rotations (REQ-CONV-001/002).

.. doxygennamespace:: polaris::frames
   :members:

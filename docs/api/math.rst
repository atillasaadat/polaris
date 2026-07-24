Math — ``polaris::math``
========================

The convention bedrock (design doc §3.1). Frame-tagged ``Vec3<Frame>`` vectors and ``Quat<To, From>`` rotations make **mixing frames across a boundary a compile error**; the single JPL scalar-first ``Quaternion`` is the only attitude representation in the repo; and ``frame_geometry`` builds the LVLH/RIC orbit frames as pure functions of the ECI state.

.. doxygennamespace:: polaris::math
   :members:

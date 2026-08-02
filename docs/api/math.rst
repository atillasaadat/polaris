Math — ``polaris::math``
========================

The convention bedrock (design doc §3.1). Frame-tagged ``Vec3<Frame>`` vectors and ``Quat<To, From>`` rotations make **mixing frames across a boundary a compile error**; the single JPL scalar-first ``Quaternion`` is the only attitude representation in the repo; ``frame_geometry`` builds the LVLH/RIC orbit frames as pure functions of the ECI state; and ``fov_overlap`` gives the one field-of-view/body overlap fraction that both the sim's optical sensors and the flight albedo correction weight by, because two implementations of it would mean correcting for an error the sensor never had.

.. doxygennamespace:: polaris::math
   :members:

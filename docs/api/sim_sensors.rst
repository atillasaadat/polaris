Sensors — ``polaris::sim::sensors``
===================================

Truth-in → measurement-out models of every sensor (design doc §6): IMU, magnetometer, star tracker, sun sensor (analogue + digital), and GNSS, plus the shared §6.1 error stack and the one line-of-sight occlusion model every optical sensor routes through. See the :doc:`sensor guide </guides/adding_a_sensor>` to add one.

.. doxygennamespace:: polaris::sim::sensors
   :members:

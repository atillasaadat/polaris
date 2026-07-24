Actuators — ``polaris::sim::actuators``
=======================================

Command-in → delivered-effect-out models (design doc §7): the reaction wheel (torque + wheel-local speed modes, RW-0.4 friction, imbalance) and its **W-matrix assembly**, and the magnetorquer (dipole limit + hysteresis). Deterministic by design — no stochastic noise.

.. doxygennamespace:: polaris::sim::actuators
   :members:

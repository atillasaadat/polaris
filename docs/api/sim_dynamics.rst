Dynamics — ``polaris::sim::dynamics``
=====================================

The 6DOF plant: the coupled rigid-body state derivative (Euler's equation + quaternion kinematics with per-step renormalization), the composable ``ForceTorqueModel`` interface the environment models implement, and the RK8(9) adaptive integrator (design doc §5.1).

.. doxygennamespace:: polaris::sim::dynamics
   :members:

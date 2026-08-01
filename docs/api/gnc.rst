GNC — ``polaris::gnc``
======================

Estimation, guidance, and control algorithms shared by the FSW and the analysis side (design doc §8). Phase 4 starts with the **coarse attitude chain**: the deterministic **TRIAD** single-frame initializer and the **SS+MAG+IMU coarse estimator** that stands behind the Safe-mode floor (§8.1, §10) — star-tracker- and table-independent, since the analytic Sun ephemeris (§11.3) always answers. Pure math on the canonical ``EstimatedState``: no F´ types and no I/O, so the F´ ``AttitudeEstimator`` component is a thin wrapper.

.. doxygennamespace:: polaris::gnc
   :members:

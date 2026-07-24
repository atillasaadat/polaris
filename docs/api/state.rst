Canonical state — ``polaris::state``
====================================

The single navigation product (design doc §8.0). ``EstimatedState`` is the onboard estimate (attitude/rates/position/velocity with frame tags, biases, 15-state covariance, per-field validity, estimation mode); ``TruthState`` is the sim-only analogue. The truth-vs-onboard split is a **type distinction** — flight code cannot consume truth.

.. doxygennamespace:: polaris::state
   :members:

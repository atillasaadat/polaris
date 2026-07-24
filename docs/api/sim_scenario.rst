Scenario — ``polaris::sim::scenario``
=====================================

The bridge from compiled configuration to running models (design doc §19.3/§19.4): the ``sim_setup.json`` loader, the vehicle builder (the only place a catalog number becomes a model), the GNSS fault schedule, and the ``SimRunner`` that composes the force stack and steps the plant.

.. doxygennamespace:: polaris::sim::scenario
   :members:

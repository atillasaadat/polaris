Power (PWR)
===========

Source: design doc §14. Fully populated in Phase 9; firm seed below.

.. req:: Solar array + battery SoC with low-SoC safing
   :id: REQ-PWR-001
   :status: reviewed
   :level: L2
   :tags: power, fdir
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/components/Power

   The FSW/sim **shall** model attitude-dependent solar-array generation (gated by
   eclipse) and battery state-of-charge, and a low-SoC monitor **shall** be able to
   trigger Safe mode.

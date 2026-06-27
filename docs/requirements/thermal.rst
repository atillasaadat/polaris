Thermal (THRM)
==============

Source: design doc §15. Fully populated in Phase 9; firm seed below.

.. req:: Lumped-node thermal with limit safing
   :id: REQ-THRM-001
   :status: reviewed
   :level: L2
   :tags: thermal, fdir
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/components/Thermal

   The FSW/sim **shall** model lumped-node temperatures (absorbed solar/albedo/IR +
   internal dissipation − radiated) with simple heater control, and a thermal-limit
   monitor **shall** be able to trigger Safe mode.

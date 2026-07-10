Guidance & Modes (GDM)
======================

Source: design doc §8.4, §10. Fully populated in Phase 7; firm seeds below.

.. req:: Pointing modes generate reference attitude/rate
   :id: REQ-GDM-001
   :status: reviewed
   :level: L2
   :tags: guidance, pointing
   :method: Test
   :derived_from: REQ-MIS-001
   :allocation: flight/components/Guidance

   Guidance **shall** generate reference attitude and rate for sun-point, nadir,
   LVLH, inertial-hold, star-track, and ground-track pointing modes.

.. req:: Constrained slew planning
   :id: REQ-GDM-002
   :status: reviewed
   :level: L2
   :tags: guidance, slew, constraints
   :method: Test
   :derived_from: REQ-GDM-001
   :allocation: flight/components/Guidance
   :refs: markley2014

   Guidance **shall** plan eigenaxis slews with rate/acceleration limits and honor
   keep-out cones (star-tracker vs Sun/Earth/Moon) and keep-in cones (comms,
   solar-array sun-pointing).

.. req:: Maneuver targeting
   :id: REQ-GDM-003
   :status: reviewed
   :level: L2
   :tags: guidance, maneuver
   :method: Analysis
   :derived_from: REQ-MIS-001
   :allocation: flight/components/Guidance

   Guidance **shall** compute burns to reach target Keplerian elements / altitude,
   generating impulsive and finite-burn references.

.. req:: Mode state machine with guards
   :id: REQ-GDM-004
   :status: reviewed
   :level: L2
   :tags: modes, fdir
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/components/ModeManager

   The mode manager **shall** implement a documented state machine (Safe, Detumble,
   Sun Point, Nadir Track, Star Track, LVLH, Inertial Hold, Ground-Station Tracking,
   Delta-V) with explicit entry/exit guards that prevent illegal/unsafe transitions.

.. req:: Onboard ground-station list
   :id: REQ-GDM-005
   :status: reviewed
   :level: L2
   :tags: modes, comms
   :method: Inspection
   :derived_from: REQ-SYS-008
   :allocation: flight/config

   The FSW **shall** carry a configurable onboard ground-station list (lat/lon/alt,
   mask angles) used for contact prediction, tracking, and link analysis.

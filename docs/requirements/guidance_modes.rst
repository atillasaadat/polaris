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
   :allocation: flight/PolarisFsw/Guidance

   Guidance **shall** generate reference attitude and rate for sun-point, nadir,
   LVLH, inertial-hold, star-track, and ground-track pointing modes.

.. req:: Constrained slew planning
   :id: REQ-GDM-002
   :status: reviewed
   :level: L2
   :tags: guidance, slew, constraints
   :method: Test
   :derived_from: REQ-GDM-001
   :allocation: flight/PolarisFsw/Guidance
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
   :allocation: flight/PolarisFsw/Guidance

   Guidance **shall** compute burns to reach target Keplerian elements / altitude,
   generating impulsive and finite-burn references.

.. req:: Mode state machine with guards
   :id: REQ-GDM-004
   :status: reviewed
   :level: L2
   :tags: modes, fdir
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/PolarisFsw/ModeManager

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

.. req:: Finite-burn executor and thrust knowledge to the orbit filter
   :id: REQ-MAN-001
   :status: reviewed
   :level: L2
   :tags: maneuver, propulsion, od
   :method: Test
   :derived_from: REQ-MIS-001, REQ-ODP-001
   :allocation: flight/PolarisFsw/BurnExecutor, flight/PolarisFsw/OrbitEstimator, sim/actuators
   :value_required: commanded acceleration = throttle * F / m along the mounted axis in ECI, sigma = knowledge fraction * |a|

   The FSW **shall** execute every burn as a **finite burn** (design doc §17): a
   commanded throttle held on the configured thrusters for a commanded duration
   through a single executor, started and aborted by command, refused by name
   when the duration or throttle is out of range, the attitude estimate is
   invalid or stale, or a burn is already in progress, and aborted when the
   attitude estimate goes stale during it. While burning the executor **shall**
   publish the commanded non-gravitational acceleration — thrust over its own
   depleting mass estimate, rotated to ECI with the current attitude — with a
   1-sigma knowledge fraction, to the orbit filter; while idle it **shall**
   publish an explicit "no thrust" every cycle. Per-unit thrust, specific
   impulse and thrust axis **shall** be configuration cross-checked against the
   installed thruster catalog entries.

   Rationale: a solution-domain orbit filter that is not told about a burn
   rejects every fix under it and, in an outage, coasts a kilometre-class error
   the paper it is compared against measured at 9 km against 5 km with the
   thrust known (Ceresoli et al. 2025). Steering (velocity tracking, Delta-V
   mode) and targeting are Phase 8; pointing during a burn is REQ-ACTL-002's.

   Verified by the ``BurnExecutor`` component tests (refusal paths, the
   hand-computed ECI acceleration for a known attitude with mass depletion at
   F/(Isp g0), abort mid-burn, stale-attitude abort, the armed bench hook), the
   config compiler's thruster cross-checks (``tests/tools/test_config_compiler.py``)
   and the SITL burn rows (``tests/integration/sitl_od_fault_test.cpp``).

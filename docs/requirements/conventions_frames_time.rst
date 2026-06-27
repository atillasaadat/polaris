Conventions, Frames & Time (CONV)
=================================

Subsystem-level requirements on coordinate frames, time systems, and
quaternion/unit conventions. Source: design doc §3.

.. req:: GPS-to-TAI / ECEF-to-ECI ingest conversion
   :id: REQ-CONV-001
   :status: reviewed
   :level: L2
   :tags: time, frames, gnss
   :method: Test
   :derived_from: REQ-SYS-001
   :allocation: lib/time, lib/frames
   :value_required: TAI = GPS + 19 s
   :refs: vallado2013

   On every GNSS fix the GNC/OD loops **shall** convert the reported GPS time to
   TAI via the constant ``TAI = GPS + 19 s`` and transform the ECEF state to
   ECI/J2000 (onboard EOP) before the inertial-frame filter and propagator run.

.. req:: Single transform library, IAU 2006/2000A
   :id: REQ-CONV-002
   :status: reviewed
   :level: L2
   :tags: frames
   :method: Analysis
   :derived_from: REQ-SYS-004
   :allocation: lib/frames
   :refs: iers2010

   All frame transformations (precession/nutation/polar motion/sidereal per IAU
   2006/2000A; ECI↔ECEF with latest IERS EOP) **shall** route through a single
   transform library; no ad-hoc rotations.

.. req:: WGS84 Earth model from shared constants
   :id: REQ-CONV-003
   :status: reviewed
   :level: L2
   :tags: frames, constants
   :method: Inspection
   :derived_from: REQ-SYS-002
   :allocation: lib/constants, lib/frames

   Geodetic latitude/longitude/altitude, ground-station coordinates, and
   ECEF↔geodetic conversions **shall** use WGS84 (latest realization) constants
   drawn from the shared physical-constants registry, so truth and onboard cannot
   disagree.

.. req:: Quaternion safety operations
   :id: REQ-CONV-004
   :status: reviewed
   :level: L2
   :tags: attitude
   :method: Test
   :derived_from: REQ-SYS-003
   :allocation: lib/math
   :refs: markley2014

   The quaternion library **shall** enforce canonical form (``q0 >= 0``), provide
   normalization checks and a documented safe-renormalization policy, and provide
   numerically safe DCM↔quaternion↔Euler conversions with documented sequences.

.. req:: Two-part high-precision time for long arcs
   :id: REQ-CONV-005
   :status: reviewed
   :level: L2
   :tags: time
   :method: Analysis
   :derived_from: REQ-SYS-001
   :allocation: lib/time

   For long-arc work (orbit propagation, ephemeris evaluation) the time library
   **shall** provide a two-part high-precision representation (int64 seconds +
   double fraction) in addition to the int64-nanosecond master-clock form.

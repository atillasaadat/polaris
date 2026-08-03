Fault Management (FDIR)
=======================

Source: design doc §9, §23.1.1. Fully populated in Phase 10; firm seeds below.

.. req:: Tiered FDIR with safing escalation
   :id: REQ-FDIR-001
   :status: reviewed
   :level: L2
   :tags: fdir, safety
   :method: Test
   :derived_from: REQ-MIS-004
   :allocation: flight/PolarisFsw/FDIR

   The FSW **shall** implement monitors → isolation → response → safing escalation;
   faults are F´ events with severity, and every alert maps to an autonomous action
   or an operator alarm with a documented (possibly null) action. FDIR state and
   triggers are telemetered.

.. req:: Validity-flag gating
   :id: REQ-FDIR-002
   :status: reviewed
   :level: L2
   :tags: fdir, sensors
   :method: Test
   :derived_from: REQ-SYS-005
   :allocation: flight/PolarisFsw/SensorProcessing

   Each measurement **shall** carry a validity flag (range, rate-of-change,
   staleness/timeout, cross-sensor consistency, solution-quality), and downstream
   consumers **shall** exclude invalid measurements — never silently use them.

.. req:: Estimator and GNSS fault response
   :id: REQ-FDIR-003
   :status: reviewed
   :level: L2
   :tags: fdir, gnss, estimation
   :method: Test
   :derived_from: REQ-FDIR-001
   :allocation: flight/PolarisFsw/FDIR

   The FSW **shall** trigger fine→coarse attitude fallback on star-tracker
   loss/occlusion, and detect GNSS outage (coasting on propagation) and
   spoofing/meaconing (innovation/consistency + reasonableness bounds, with
   measurement rejection).

.. req:: Fault-injection verification
   :id: REQ-FDIR-004
   :status: reviewed
   :level: L2
   :tags: fdir, vv
   :method: Test
   :derived_from: REQ-MIS-005
   :allocation: tests/integration

   Every FDIR monitor/response **shall** be exercised by the fault-injection
   integration suite, asserting correct detection (no false positives on nominal
   runs) and response within a required detection latency.

Attitude-source mode transitions under fault
--------------------------------------------

Source: design doc §8.2, §9; user decision 2026-08-03 (open the §9 FDIR work with
a systematic SITL fault matrix). Push 53.

The requirements below are the **behavioural** half of the attitude-determination
FDIR story, and they are deliberately separate from the REQ-ADET requirements
that specify the combination rules themselves. REQ-ADET-008/010/011 say what a
vote or a fusion layer must compute; these say what the **mode machine** must do
about it — which rung of the §8.2 ladder the vehicle ends on, which unit is
blamed, and what the response costs. The two fail independently: a vote can
identify the right unit while the mode machine demotes the filter that identified
it, and that combination is a defect neither requirement alone would catch.

REQ-FDIR-013 is the one of them that was not authored in advance: the matrix
found the defect it describes on a nominal run, and the requirement, the fix and
its verification landed together in Push 53. It is written here as the property
the vehicle must have rather than as a change log — the history lives in the
design doc (§23.1.1) and in the verifying case.

Each is verified by one row of ``tests/integration/sitl_fault_matrix_test.cpp``,
which drives the sim's fault hooks against the deployed flight binary over the
SITL wire and asserts on the deployment's own event stream — the same channel an
operator would read.

.. req:: Fine-mode source is reported as a transition, not inferred
   :id: REQ-FDIR-005
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs
   :method: Test
   :derived_from: REQ-FDIR-001, REQ-ADET-004
   :allocation: flight/PolarisFsw/AttitudeEstimator

   Every change of the measurement source the fine solution is being built from
   **shall** be reported as a distinct event naming the source before and after,
   emitted on the **transition** and not once per cycle, and the fall to no fine
   solution at all **shall** be reported through the same event rather than left
   to be inferred from the demotion that accompanies it.

   A source change **shall not** be a demotion: the filter **shall** keep its
   state and covariance across it, the published solution **shall** stay valid,
   and no re-seed **shall** occur.

   **Carve-out.** The arbitration of REQ-FDIR-013 is the one source change that
   *does* replace the filter's state, by construction — it exists precisely to
   move the solution onto a source the filter was rejecting. It is still not a
   demotion (the published attitude stays valid throughout and no
   ``FineModeDemoted`` accompanies it) and it announces itself with its own
   event, so a consumer can tell the two apart. Nothing else may re-seed across a
   source change.

.. req:: Star-tracker redundancy holds the finest rung
   :id: REQ-FDIR-006
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs, redundancy
   :method: Test
   :derived_from: REQ-FDIR-003, REQ-ADET-012
   :allocation: flight/PolarisFsw/AttitudeEstimator

   Loss of **one** star tracker — including the king, whose mounting defines the
   body frame — **shall not** move the fine solution off its star-tracker source,
   provided a second calibrated unit is delivering solutions. Loss of **all**
   star trackers **shall** move it to the sun/magnetometer source without an
   attitude loss and without a demotion.

   A unit that stops delivering solutions **shall not** be latched out or
   otherwise blamed: absence is not evidence of a fault.

.. req:: The finest rung is reachable without a coarse floor
   :id: REQ-FDIR-007
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs
   :method: Test
   :derived_from: REQ-ADET-012
   :allocation: flight/PolarisFsw/AttitudeEstimator

   With no sun measurement available — eclipse, or a cold start pointed away from
   the Sun — the fine mode **shall** be seedable from a star tracker's own
   solution and covariance, with no coarse solution and no vector pair
   underneath it. The subsequent return of the sun measurement **shall** produce
   no source change, no re-seed and no demotion.

.. req:: Fault attribution requires a reference independent of the units judged
   :id: REQ-FDIR-008
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs, redundancy
   :method: Test
   :derived_from: REQ-ADET-011, REQ-FDIR-002
   :allocation: lib/gnc, flight/PolarisFsw/AttitudeEstimator

   A two-unit magnetometer disagreement **shall** be attributed only while the
   attitude used to rotate the identification reference is independent of the
   magnetometers — i.e. while the fine solution is star-tracker-sourced. In any
   other mode the disagreement **shall** be reported as unattributable, with no
   unit latched out, and the magnetic pair dropped for the affected cycles.

   Where the reference **is** independent, the offending unit **shall** be
   excluded exactly once for the duration of the fault, the healthy unit
   **shall not** be excluded, the fine solution **shall not** be demoted by the
   identification, and the unit **shall** be re-admitted automatically on
   recovery.

   .. note::

      The second clause is the fault-tolerance *inversion* the gate exists to
      prevent, and it is asserted directly rather than left to follow from the
      first: a drifted unit drags the solution, which drags the rotated reference
      toward the drifted unit, so an ungated identification does not merely fail —
      it preferentially latches out the **healthy** magnetometer.

.. req:: Sun-sensor cross-check detects without an attitude and resolves with one
   :id: REQ-FDIR-009
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs, redundancy
   :method: Test
   :derived_from: REQ-ADET-010, REQ-FDIR-002
   :allocation: flight/PolarisFsw/AttitudeEstimator

   A sun sensor whose reported direction has shifted while it continues to report
   valid, sun-present and its nominal σ **shall** be detected by comparison with
   the second-best unit in view, after a configured persistence, in **any** mode —
   the comparison requires no attitude.

   The selection **shall** move to the runner-up only when a sun-independent
   attitude (a star-tracker-sourced fine solution) agrees with the runner-up and
   disagrees with the incumbent, both against the same threshold. Nothing
   **shall** be latched by the override, so a unit that recovers simply stops
   being overridden.

.. req:: Loss of a vector source costs accuracy, not the attitude
   :id: REQ-FDIR-010
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs
   :method: Test
   :derived_from: REQ-FDIR-003, REQ-ADET-004
   :allocation: flight/PolarisFsw/AttitudeEstimator

   Loss of the sun measurement from **every** sun sensor **shall not** demote the
   fine mode or invalidate the published attitude while the magnetic pair is
   still updating the filter and the fine coast horizon has not expired. A sun
   sensor reporting no sun **shall not** raise any fault event or residual-monitor
   alert: it is the nominal eclipse condition, and an FDIR response to it would
   fire on every orbit.

.. req:: An unattributable rate disagreement withholds the rate and latches nothing
   :id: REQ-FDIR-011
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs, redundancy
   :method: Test
   :derived_from: REQ-ADET-008, REQ-ADET-009
   :allocation: flight/PolarisFsw/AttitudeEstimator

   With a two-IMU disagreement present and no independent rate reference
   available, the vehicle **shall** publish no body rate, **shall not** latch an
   exclusion on either unit, and **shall** continue to acquire and hold an
   attitude from the vector pairs, which are independent of the gyro fault. The
   condition **shall** be reported once on its edge and escalated at a bounded
   cadence while it persists.

   .. note::

      **A fine solution does not resolve this**, and the SITL matrix pins that
      rather than implying otherwise. The vote's only tie-break is the filter's
      *propagated gyro* rate, which is reported invalid precisely when no usable
      gyro has been seen — so the evidence the vote needs is the evidence the
      fault removed, and the disagreement stands until the ground acts. This is
      self-consistent and conservative, not a defect; the upgrade path, if the
      cost is ever judged too high, is to differentiate successive star-tracker
      attitudes into an independent rate. The design does not do that today.

.. req:: A fault present before any reference is blamed only once one exists
   :id: REQ-FDIR-012
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs, redundancy
   :method: Test
   :derived_from: REQ-FDIR-008
   :allocation: flight/PolarisFsw/AttitudeEstimator

   A sensor fault that is already present at cold start, before the vehicle has
   any attitude solution, **shall** be reported as unattributable and **shall
   not** cause any unit to be latched out. Once an independent reference becomes
   available, the same fault **shall** be attributed to the unit that carries it,
   with the ambiguity report preceding the exclusion in the event stream.

.. req:: A better source is adopted, not blamed, for disagreeing with a worse one
   :id: REQ-FDIR-013
   :status: reviewed
   :level: L2
   :tags: fdir, estimation, adcs
   :method: Test
   :derived_from: REQ-FDIR-006, REQ-ADET-012
   :allocation: flight/PolarisFsw/AttitudeEstimator

   A star tracker whose measurements the fine filter rejects **while the fine
   solution is not itself star-tracker-sourced** **shall not** be excluded on
   that evidence: the rejection is a statement about the filter's covariance and
   not about the unit.

   The verdict **shall** instead be taken against the coarse solution, whose
   covariance carries a systematic floor, in the **Mahalanobis** metric on that
   covariance — the coarse attitude uncertainty is strongly anisotropic, so an
   isotropic bound is simultaneously too tight across its stiff axes and too loose
   along its weak one. The covariance **shall** be verified symmetric
   positive-definite before use, and a tracker with ``d² ≤ χ²₃(0.999)`` **shall**
   cause the filter to be **re-seeded from the tracker**, reported by a distinct
   event, with no demotion and no loss of the published attitude.

   A tracker outside that containment **shall** be refused — not adopted, and
   reported at a bounded cadence — and **shall not** be latched out. The
   re-admission criterion for an excluded tracker is agreement with the *fine*
   solution, so an exclusion decided on *coarse* agreement could never be served;
   the unit therefore stays a candidate, and a later cycle with a better reference
   may still adopt it. The refusal report **shall** additionally require a fresh
   coarse fix, since coast growth is a stated lower bound on the true uncertainty
   and an accusation must not rest on a number known to be optimistic; adoption
   carries no such condition, erring the permissive way.

   Neither outcome **shall** be repeatable without an intervening change of
   state: the mode **shall** settle in one transition rather than oscillate.

   .. note::

      **Found by the Push 53 fault matrix, on a nominal sunlit run with no
      injected fault at all.** A vehicle booting in sunlight promotes off a
      Davenport seed within one cycle, well before a star tracker's cold
      acquisition; the SS+MAG solution's error is then dominated by the
      magnetometer's ~34 mrad systematic, which the filter's white-``R`` model
      averages its covariance down through. Within seconds ``S = HPHᵀ + R`` is a
      few mrad² against a residual that is the whole systematic, so every
      arriving tracker update fails the χ²₃ gate — and the pre-fix response
      latched the *tracker* out after ``MekfNisStreak`` cycles and demoted the
      mode. With only the king fused (the launch state, before the inter-tracker
      alignment has run) the vehicle then stayed on the sun/magnetic rung for the
      rest of the flight: REQ-ADET-007 accuracy unreachable from a sunlit boot,
      and unrecoverable after any tracker outage, with one event to say so.

      The second clause is the failure mode the fix creates. Adopting a rejected
      tracker is right only when the tracker is the better source; without the
      coarse-covariance guard a faulted unit could hijack the solution by being
      rejected persistently enough, and the vehicle would fly a wrong attitude at
      arcsecond-class reported covariance — worse than the defect. The third
      clause needs no explicit rate limiter and deliberately has none: a re-seed
      succeeds by construction (the filter is initialised *to* the tracker, so
      the next cycle accepts it) and a refusal changes no state at all, so neither
      branch can oscillate.

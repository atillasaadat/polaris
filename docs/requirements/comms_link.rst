Communications & Link (COMM)
============================

Source: design doc §16, §11.1. Fully populated in Phase 9; firm seeds below.

.. req:: Ground-station contact prediction
   :id: REQ-COMM-001
   :status: reviewed
   :level: L2
   :tags: comms, contacts
   :method: Analysis
   :derived_from: REQ-GDM-005
   :allocation: analysis/contacts
   :refs: vallado2013

   The suite **shall** predict ground-station contact windows (elevation mask,
   horizon geometry) across the GS network, validated against GMAT contact-geometry
   golden fixtures.

.. req:: Link budget with margin
   :id: REQ-COMM-002
   :status: reviewed
   :level: L2
   :tags: comms, linkbudget
   :method: Analysis
   :derived_from: REQ-MIS-003
   :allocation: analysis/linkbudget

   The suite **shall** compute a per-pass link budget (EIRP, free-space path loss,
   pointing loss from attitude + pass geometry, ground-station G/T) yielding C/N0,
   Eb/N0, and link margin against a required threshold.

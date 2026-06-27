Traceability Matrix (RVTM)
==========================

The live Requirements Verification & Traceability Matrix, generated from the
requirement set and the verification results collected from the test suites
(``docs/_generated/verif_*.json``). Read left-to-right for forward trace
(requirement → method → verifying artifact); the *Verified By* column gives the
backward trace.

.. note::

   In Phase 0 there are no tests yet, so the *Verified By* column is empty and all
   requirements sit at status ``reviewed``. As tests land (Push 2+) they annotate
   the requirement IDs they verify and populate this matrix automatically; the CI
   gate then enforces that every ``approved`` requirement is covered with margin.

Full matrix
-----------

.. needtable::
   :types: req
   :columns: id, title, level, method, status, value_required, margin_required, verifies_back as "Verified By"
   :style: datatables

Requirements by subsystem
-------------------------

.. needtable::
   :types: req
   :columns: id, title, status, allocation
   :sort: id
   :style: datatables

Verification artifacts
-----------------------

.. needtable::
   :types: test, mc
   :columns: id, title, status, verifies as "Verifies"
   :style: datatables

Requirements
============

The Polaris requirements baseline. Authored before development of each capability
and kept in **bidirectional traceability** with the tests that verify it
(design doc §22.2; NASA NPR 7150.2 SWE-052/059/072). Requirements are first-class
objects in this site via `sphinx-needs`; the :doc:`matrix` page renders the live
Requirements Verification & Traceability Matrix (RVTM).

How requirements are written
----------------------------

**ID scheme.** ``REQ-<SUBSYS>-NNN`` — zero-padded 3-digit, **immutable and never
renumbered or reused** (deprecate instead). ``<SUBSYS>`` codes:

``MIS`` mission · ``SYS`` system/program · ``CONV`` conventions/frames/time ·
``ADET`` attitude determination · ``ACTL`` attitude control · ``ACT``
actuators/allocation · ``ODP`` orbit determination & propagation · ``GDM``
guidance & modes · ``FDIR`` fault management · ``COMM`` comms/link · ``PWR``
power · ``THRM`` thermal · ``CDH`` C&DH / FSW infra · ``SIM`` simulation/env ·
``PAY`` payload · ``CFG`` config/data-products · ``VV`` verification process.

**Levels.** ``L0`` mission objective/constraint · ``L1`` system/program
(allocated to FSW/sim/ground) · ``L2`` subsystem. Every L2 requirement either
``derived_from`` an L1 parent **or** is explicitly *derived* (no single parent —
then ``rationale`` is mandatory).

**"Shall" style** (INCOSE): one binding, singular, verifiable claim per
requirement. "shall" = binding; "should" = goal; "will" = fact/context. State
**units and frame** for every physical quantity.

**Verification method** (``method``): **Test** / **Analysis** / **Inspection** /
**Demonstration** (T/A/I/D).

**Status workflow:** ``draft`` → ``reviewed`` → ``approved`` → ``verified`` →
(``deprecated``). The CI gate (``needs_warnings`` + ``sphinx-build -W``) fails the
build if an **approved** or **verified** requirement has no incoming ``verifies``
link. That coverage check is currently the *only* gate: ``margin_achieved`` is
collected and reported in the RVTM, but no build-time check compares it against
``margin_required`` yet — a shortfall shows up in the matrix, it does not fail
the build. Phase-0 seed requirements are kept at ``reviewed`` until a verifying
test exists, so the gate is meaningful without blocking early work.

Attributes
----------

``id`` · ``status`` · ``level`` · ``method`` · ``rationale`` ·
``allocation`` (owning component / sim model / lib module) · ``derived_from`` ·
``value_required`` + ``margin_required`` (quantitative) ·
``value_demonstrated`` + ``margin_achieved`` (written back by the verifying
artifact) · ``refs`` (``docs/refs.bib`` bibkey) · ``tags``.

Template
--------

.. code-block:: rst

   .. req:: <short imperative title>
      :id: REQ-ACTL-014
      :status: reviewed
      :level: L2
      :tags: adcs, control, momentum
      :method: Test
      :derived_from: REQ-SYS-012
      :allocation: flight/PolarisFsw/MomentumManager
      :value_required: 0.8 * h_max
      :margin_required: 20 %
      :refs: markley2014

      The ADCS **shall** command reaction-wheel desaturation when total stored
      body-frame momentum exceeds 0.8 of the per-wheel saturation limit
      ``h_max`` (N·m·s), retaining >= 20 % margin to saturation.

      :rationale: Desaturate before saturation to retain control authority.

How a test claims a requirement
-------------------------------

- **Python:** ``@pytest.mark.verifies("REQ-…")`` + ``record_property("margin_pct", …)``.
- **C++ (GoogleTest):** ``RecordProperty("verifies", "REQ-…")`` + ``RecordProperty("margin_pct", …)``.
- **Monte Carlo:** an ``.. mc::`` need with ``:verifies:`` and a worst-case margin.

Collectors emit ``docs/_generated/verif_*.json`` (Sphinx-Needs external needs), so
the verifying tests appear as ``verified by`` back-links and flow into the RVTM.

.. toctree::
   :maxdepth: 1
   :caption: Requirement sets

   l0_mission
   l1_system
   conventions_frames_time
   adcs_determination
   adcs_control
   adcs_actuators
   orbit_od_propagation
   guidance_modes
   fdir
   comms_link
   power
   thermal
   cdh_fsw_infra
   payload
   simulation_env
   config_dataproducts
   vv_process
   matrix

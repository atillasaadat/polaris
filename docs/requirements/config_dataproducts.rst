Configuration & Data Products (CFG)
===================================

Source: design doc §19, §20. Fully populated in Phases 0–8; firm seeds below.

.. req:: One config compiler, one source of truth
   :id: REQ-CFG-001
   :status: reviewed
   :level: L2
   :tags: config, architecture
   :method: Test
   :derived_from: REQ-SYS-008
   :allocation: tools/configc

   The config compiler **shall** resolve hardware model-IDs, validate against the
   schema, and emit F´ parameter sets, sim/truth setup, and analysis inputs from a
   single resolved/validated config object; no consumer re-parses raw YAML
   independently.

.. req:: Hardware model library by model ID
   :id: REQ-CFG-002
   :status: reviewed
   :level: L2
   :tags: config, hardware
   :method: Inspection
   :derived_from: REQ-CFG-001
   :allocation: config/hardware

   Hardware **shall** be defined as parameterized model definitions keyed by model
   ID; a spacecraft config references hardware by model-ID string, so swapping the
   string swaps the modeled unit.

.. req:: Config provenance
   :id: REQ-CFG-003
   :status: reviewed
   :level: L2
   :tags: config, provenance
   :method: Inspection
   :derived_from: REQ-CFG-001
   :allocation: tools/configc

   Each emitted artifact **shall** record the source config hash/version, so any
   FSW param, sim run, golden fixture, or MC campaign is traceable to the exact
   config that produced it.

.. req:: Interoperability ephemeris exports
   :id: REQ-CFG-004
   :status: reviewed
   :level: L2
   :tags: dataproducts, interop
   :method: Analysis
   :derived_from: REQ-MIS-003
   :allocation: lib/state, analysis/postproc

   The suite **shall** export CCSDS OEM and OMM/TLE (pre- and post-burn) and STK
   (.e) and FreeFlyer ephemeris for visualization, as standardized, frame-tagged,
   unit-declared data products.

.. req:: Uplink overlay reconciliation
   :id: REQ-CFG-005
   :status: reviewed
   :level: L2
   :tags: config, ops
   :method: Test
   :derived_from: REQ-SYS-008
   :allocation: flight/PolarisFsw/ParameterManager

   On-orbit parameter uplinks **shall** be captured as a versioned overlay on the
   authoritative ground baseline (never silent divergence); the effective onboard
   parameter set is telemetered and diffable against the baseline.

---
description: Summarize progress against the Polaris development phasing and requirements
argument-hint: [phase number]
allowed-tools: Read, Grep, Glob, Bash
---

Report Polaris development status$ARGUMENTS against the phasing in `docs/design/Polaris_Design_Document.md` §24 and the Requirements ICD in `docs/requirements/`.

Do:
1. Identify the current phase (or the phase given in the argument) and its prerequisites. Phases are dependency-ordered — flag if work is happening out of order.
2. Inspect the repo to assess what exists vs. what the phase requires (components, lib modules, tests, fixtures, docs). Use `git log`/`Glob`/`Grep`; don't assume.
3. Check the Phase-0 / pre-kickoff gates where relevant: conventions + typed vectors, canonical state structs, config compiler, Requirements ICD, docs site, GMAT harness, license.
4. Report:
   - **Done** (with evidence: files/tests present),
   - **In progress / partial**,
   - **Not started / blocked** (and on what),
   - **Requirements coverage** for the phase: which `REQ-###` are verified (with margin) vs untested,
   - **GMAT golden coverage** for the phase's numerical functions.
5. Recommend the next 3–5 concrete tasks to advance the phase, in order.

Be concrete and evidence-based; this is a status read, not a plan to execute. Make no edits.

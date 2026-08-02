# `bindings/` — pybind11

> **Status: planned, not yet implemented** — this directory is a placeholder; the
> bindings below are the design intent (§18.7, Phase 11), built once `analysis/`
> and the live web tools (§21.4) need them.

pybind11 bindings exposing `lib/` (and FSW algorithms) to Python so `analysis/`
and the future live web tools exercise the **same** C++ that flies. If a needed
function isn't bound, add the binding here — do not reimplement GNC math in Python
(see `analysis/CLAUDE.md`).

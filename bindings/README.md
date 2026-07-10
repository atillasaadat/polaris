# `bindings/` — pybind11

pybind11 bindings exposing `lib/` (and FSW algorithms) to Python so `analysis/`
and the future live web tools exercise the **same** C++ that flies. If a needed
function isn't bound, add the binding here — do not reimplement GNC math in Python
(see `analysis/CLAUDE.md`).

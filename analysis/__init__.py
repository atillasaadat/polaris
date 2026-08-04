"""Polaris ground-side analysis tools (design doc §12, §13, §21.4).

Namespace package only. The rule that governs everything under it is in
``analysis/CLAUDE.md``: GNC math is reused from the flight/sim C++ through
``bindings/``, never reimplemented here. Linear-systems analysis on matrices
read from ``config/`` (``analysis.control``) is the documented exception — it
analyses the flight design rather than duplicating any flight computation.
"""

Onboard tables — ``polaris::onboard``
=====================================

The flight-safe holder for the three onboard reference tables the GNC stack consumes (design doc §11.3, §22; REQ-CDH-002): the in-code leap-second (ΔAT) record, the IERS EOP table parsed from the verbatim ``finals.all`` product, and the Chebyshev ephemeris of the Sun and Moon. ``TableStore`` loads, validates, and answers point queries: ``eopAt``, ``bodyPositionEci`` and ``taiUtcOffset`` each return a ``Quality`` — ``kPrecise`` from the uploaded table, ``kCoarse`` from the table-independent fallback (analytic Sun/Moon, zero-EOP) that keeps the Safe-mode floor answerable when no table is loaded. ``coverageAt`` is the exception: it reports whether the EOP and ephemeris spans cover an epoch at all, so it answers with a plain ``bool`` per domain rather than a grade. Two ``TableSet`` slots behind an atomic active index, each seqlock-guarded, give the "stage then swap" the upload→activate path needs: a failed reload leaves the previous tables in service and a query never observes a half-loaded table. No heap and no exceptions in steady state; the ``flight.OnboardTables`` F´ component is a thin wrapper.

.. doxygennamespace:: polaris::onboard
   :members:

Time — ``polaris::time``
========================

The onboard time foundation (design doc §3.2): the **TAI** master clock and the strongly-typed uniform scales (GPS/TT/TDB) — mixing scales is a compile error — plus the signed-nanosecond ``Duration``, the fixed-capacity leap-second table, ground-only UTC derivation, and the TT↔TDB periodic term that feeds the ephemerides.

.. doxygennamespace:: polaris::time
   :members:

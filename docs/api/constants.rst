Constants — ``polaris::constants``
==================================

Single source of truth for physical constants shared by the flight software and the truth simulation, so the two can never silently disagree. SI throughout; every value cites its source. Grouped into ``wgs84``, ``bodies`` (Sun/Moon/planet GMs), ``time``, ``iau``, ``tdb``, ``physical``, and ``srp`` (design doc §3.4).

.. doxygennamespace:: polaris::constants
   :members:

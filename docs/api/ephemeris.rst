Ephemeris — ``polaris::ephemeris``
==================================

Onboard Sun/Moon/planet positions (design doc §11.3): the flight-side Chebyshev evaluator (``ChebyshevSegment`` + ``evaluate``) reading ground-generated coefficient sets, and the fixed-capacity ``EphemerisTable`` that selects the covering interval. Queried at **TDB**, positions in ECI/J2000 metres. SPICE is ground-only and never linked into flight (REQ-CDH-002).

.. doxygennamespace:: polaris::ephemeris
   :members:

Environment — ``polaris::environment``
======================================

Environment models clean enough for both sides of the truth/onboard divide: the **IGRF** geomagnetic field the FSW needs for measured-vs-modelled magnetometer checks and coarse attitude (§5.2/§8.1). Coefficients are loaded from the committed IAGA file by ``sim/world``.

.. doxygennamespace:: polaris::environment
   :members:

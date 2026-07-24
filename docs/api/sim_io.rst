Closed loop — ``polaris::sim::io``
==================================

The §2.4 execution model, sim side: ``ClosedLoop`` marches sim time on an exact integer-ns event grid, samples each sensor at its native rate into the §2.4 buffers, fires an ``FswCallback`` at each macro boundary, and feeds actuator commands back into the plant. See the :doc:`closed-loop guide </guides/closed_loop>`.

.. doxygennamespace:: polaris::sim::io
   :members:

Payload (PAY)
=============

Source: design doc §6.3, §8.1. Requirements a payload instrument brings with it —
the ones stated about *its* boresight rather than about the vehicle's body axes.

.. _pay-cross-boresight-metric:

The cross-boresight metric
--------------------------

A payload does not care about the whole attitude error. It cares about the part
that moves its target across the focal plane. Split the knowledge error at the
sensor's own axes — **+Z is the boresight, always** (design doc §6.3) — and the
two halves mean different things:

- **Cross-boresight** (the sensor X–Y plane): the boresight is not where the
  solution says it is, so the scene is displaced. This is what mis-registers an
  image, misses a target, or walks a laser off its receiver.
- **About-boresight** (roll about +Z): the image is *rotated* about its own
  centre and the boresight still points exactly where it was commanded. For a
  circularly symmetric instrument this costs nothing at all; for a framing or
  pushbroom imager it is a de-rotation, not a miss.

So the requirement below is written on the first, defined as the **great-circle
angle between the true and the estimated boresight directions in inertial
space**:

.. math::

   \theta_\mathrm{bore} = \angle\left( R(\hat{q})^\top \mathbf{b},\ R(q)^\top
   \mathbf{b} \right),

with :math:`\mathbf{b}` the mounted boresight in body axes and :math:`R(q)` the
true Body←ECI attitude matrix. Equivalently, for an attitude-error rotation
vector :math:`\boldsymbol{\theta}` at angle :math:`\psi` to the boresight, it is
:math:`|\boldsymbol{\theta}| \sin\psi` — the projection of the error onto the
cross-boresight plane. Three properties follow, and they are why this is stated
separately from :ref:`the vehicle-level error norm <adet-knowledge-metric>`:

#. It is **invariant to rotation about the boresight** — an error purely about
   +Z scores zero, correctly.
#. It is **never larger than the error norm**, so a payload requirement can
   never be the binding one by accident.
#. It is **independent of where the payload is mounted**, because the knowledge
   error has no preferred body axis. That is asserted, not assumed: the
   verifying campaign measures a canted mounting alongside the reference one.

.. req:: Payload cross-boresight knowledge accuracy
   :id: REQ-PAY-001
   :status: reviewed
   :level: L2
   :tags: payload, adcs, estimation, accuracy, pointing
   :method: Test
   :derived_from: REQ-ADET-006
   :allocation: sim/sensors/payload_sensor, lib/gnc
   :value_required: <= 14 deg cross-boresight error (3-sigma)
   :margin_required: 20 %
   :refs: wertz2011

   For a payload sensor mounted on the vehicle, in fine mode on the same
   SS+MAG+IMU suite and with no star tracker, under the conditions of
   REQ-ADET-006, the attitude-knowledge error projected onto the sensor's X–Y
   plane — the cross-boresight error norm (:ref:`metric
   <pay-cross-boresight-metric>`) — **shall** be ≤ **14°** (3σ).

   .. note::

      **Where the number comes from.** The same 800-run campaign that sets the
      two vehicle-level thresholds
      (``tests/unit/attitude_accuracy_mc_test.cpp``), evaluated on the *same*
      runs so the metrics are directly comparable rather than separately
      sampled. It measures a median of **2.2°** and a 3σ bound of **7.9°**,
      against **2.9°** and **8.1°** for the total error norm on those runs.
      The ratio of the medians, 0.74, is the expected one: for an isotropically
      directed error the mean of :math:`\sin\psi` is :math:`\pi/4 = 0.785`, so a
      payload sees about three quarters of the vehicle's knowledge error and the
      rest is roll it does not pay for. 14° therefore carries 44% margin at the
      shipped seed, and the bound moves between 7.3° and 10.9° across the four
      master seeds tried while setting it — never less than 21% margin, so the
      threshold does not sit on a seed-specific tail.

      **This is a knowledge requirement, not a pointing one.** It bounds how
      well the vehicle *knows* where the boresight is, not how well it holds it
      there; control error adds to this in quadrature and belongs to the §8.4
      control requirements. Nor does it cover boresight **stability** (jitter
      over an integration time), which is a separate quantity a specific
      instrument's exposure sets — the reaction-wheel micro-vibration work
      (design doc §7) is where that gets a number.

      **Improving it is a sensor-budget action.** It inherits REQ-ADET-006's
      systematic floor exactly, so the levers are the same and in the same
      order: magnetometer hard-iron calibration, sun-sensor albedo correction,
      then a star tracker. Filtering harder buys nothing, for the reason given
      under REQ-ADET-006.

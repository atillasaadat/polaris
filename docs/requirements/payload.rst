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
   :derived_from: REQ-ADET-007
   :allocation: sim/sensors/payload_sensor, lib/gnc
   :value_required: <= 10 % of the sensor's smallest full FOV angle (3-sigma)
   :margin_required: 20 %
   :refs: wertz2011

   For each payload sensor the vehicle carries, in the estimation mode required
   for payload operations — **fine mode with one or more star trackers fused**,
   the mode of REQ-ADET-007 — the attitude-knowledge error projected onto that
   sensor's X–Y plane (the cross-boresight error norm, :ref:`metric
   <pay-cross-boresight-metric>`) **shall** be ≤ **10% of that sensor's smallest
   full field-of-view angle** (3σ).

   The smallest full FOV angle is ``2 × half_fov_rad`` for a conic field and
   ``2 × min(half_fov_x_rad, half_fov_y_rad)`` for a square or rectangular one:
   the narrow axis, because that is the axis a pointing miss leaves the field
   through first. For the generic imager of ``config/hardware/payload/`` (5° × 4°
   half-angles, so an 8° smallest full FOV) the threshold is **0.8°**.

   .. note::

      **Why a fraction of the FOV rather than an angle.** The number an
      instrument actually cares about is what fraction of its own field a
      pointing miss eats — 0.8° is a quarter of a narrow-field spectrometer's
      world and invisible to a wide-field imager. Stating it as a fraction makes
      the requirement scale with whatever instrument the vehicle ends up
      carrying instead of needing a re-derivation per payload, and 10% leaves
      the target inside the field with the rest of the budget available to
      control error and to the instrument's own alignment.

      **Why fine+ST and not the coarse-suite mode.** A knowledge requirement is
      only meaningful against the mode the mission actually images in. Anchoring
      it to the SS+MAG+IMU suite — as the first version of this requirement did,
      at an absolute 14° — measured a mode no payload would be operated in, and
      produced a threshold set by the magnetometer's hard-iron residual rather
      than by anything about the payload. With star trackers fused the
      cross-boresight budget is ~16 arcsec-class per tracker (REQ-ADET-007's
      note), so 10% of any realistic instrument's field is comfortably feasible
      — the requirement is a guard against a mis-specified suite or a
      mis-mounted instrument, not a stretch target.

      Not yet verifiable: the §8.2 fusion layer that makes fine+ST reachable is
      unbuilt, so this requirement is held at ``reviewed`` and carries no
      verifying artifact, exactly as REQ-ADET-007 is. The push that lands the
      fusion layer verifies both, per sensor, in the same campaign.

      **This is a knowledge requirement, not a pointing one.** It bounds how
      well the vehicle *knows* where the boresight is, not how well it holds it
      there; control error adds to this in quadrature and belongs to the §8.4
      control requirements. Nor does it cover boresight **stability** (jitter
      over an integration time), which is a separate quantity a specific
      instrument's exposure sets — the reaction-wheel micro-vibration work
      (design doc §7) is where that gets a number.

Random — ``polaris::random``
============================

The determinism substrate (design doc §3.5): a ``SplitMix64`` generator (uniform + Box-Muller Gaussian) and ``streamRng(master_seed, stream_id)`` — each source derives its own independent stream, so a run is bit-reproducible from ``{config, seed}`` and adding a noise source never perturbs the existing ones.

.. doxygennamespace:: polaris::random
   :members:

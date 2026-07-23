# `lib/random/` — Seeded Per-Source RNG Streams

The determinism substrate (design doc §3.5, `README` of every stochastic model):
a run is bit-reproducible from `{config, seed}`, and adding a noise source never
perturbs existing streams.

| File | Role |
|---|---|
| `rng.hpp` | `SplitMix64` generator [steele2014] (uniform + Box-Muller gaussian), and `streamRng(master_seed, stream_id)` — each source derives its own independent stream, keyed in practice by instance name (`sim/scenario/vehicle`) |

Never seed from wall clock or `std::random_device` in a run path; never share
one generator across sources.

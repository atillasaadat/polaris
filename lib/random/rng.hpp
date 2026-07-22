#ifndef POLARIS_RANDOM_RNG_HPP
#define POLARIS_RANDOM_RNG_HPP

/// @file
/// @brief Seeded, per-source-derived random streams (design doc §3.6 determinism;
/// §6.1 sensor noise).
///
/// Determinism is mandatory (sim/CLAUDE.md): a run must be bit-reproducible from
/// `{config, seed}`, including across the two-process boundary. That rules out
/// `std::random_device`, wall-clock seeding, and any global generator whose draw
/// order depends on evaluation order. Every stochastic source instead owns an
/// independent stream, seeded by deriving from one master seed and a stable
/// per-source stream id.
///
/// **Per-source derivation, not a shared sequence.** The design doc (§182) calls
/// for seed derivation such that *adding a new noise source does not perturb the
/// existing streams*. A single generator handed out to every source fails that:
/// insert one draw upstream and every downstream sample shifts, so yesterday's
/// golden run no longer reproduces. Here each source maps its stream id through
/// `streamSeed` to its own seed, so streams are independent and stable under
/// addition or reordering of sources.
///
/// The generator is **SplitMix64** (Steele et al., 2014): a single 64-bit state,
/// no tables, trivially seedable, with well-distributed output — more than enough
/// for sensor/actuator noise and Monte-Carlo dispersions, and simple enough to
/// reproduce bit-for-bit in any language on either side of the transport. This is
/// not a cryptographic generator and must not be used as one.
///
/// Flight-safe: no heap, no exceptions, `constexpr` integer core.
///
/// References:
///  - Steele, Lea & Flood, "Fast Splittable Pseudorandom Number Generators",
///    OOPSLA 2014 (SplitMix). [steele2014]

#include <cmath>
#include <cstdint>
#include <limits>

namespace polaris::random {

/// SplitMix64 PRNG. Deterministic from its seed; copy it to fork a replayable
/// sub-stream. Not cryptographically secure.
class SplitMix64 {
 public:
  explicit constexpr SplitMix64(std::uint64_t seed) : state_(seed) {}

  /// Next 64 uniformly-distributed bits.
  constexpr std::uint64_t nextU64() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /// Uniform double in [0, 1). Uses the top 53 bits, so every representable
  /// multiple of 2^-53 is equally likely and the result is never exactly 1.
  double uniform() { return static_cast<double>(nextU64() >> 11) * 0x1.0p-53; }

  /// One draw from the standard normal N(0, 1) via the Box–Muller transform.
  /// Consumes two uniforms per call (the sine companion is discarded — the small
  /// extra draw cost buys a stateless, trivially-reproducible sampler).
  double gaussian() {
    // Guard the log against a zero uniform (probability 2^-53, but it would
    // produce +inf); the smallest positive double keeps the transform finite.
    double u1 = uniform();
    if (u1 <= 0.0) {
      u1 = std::numeric_limits<double>::min();
    }
    const double u2 = uniform();
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
  }

 private:
  std::uint64_t state_;
};

/// Derive an independent stream seed from a master seed and a stable per-source
/// @p stream_id. Distinct ids yield well-separated seeds, and — because each id
/// is mixed independently — adding or removing a source never shifts any other
/// source's stream (design doc §182).
constexpr std::uint64_t streamSeed(std::uint64_t master_seed, std::uint64_t stream_id) {
  SplitMix64 mixer(master_seed + 0x9E3779B97F4A7C15ULL * stream_id);
  return mixer.nextU64();
}

/// An `Rng` seeded for source @p stream_id under @p master_seed.
constexpr SplitMix64 streamRng(std::uint64_t master_seed, std::uint64_t stream_id) {
  return SplitMix64(streamSeed(master_seed, stream_id));
}

}  // namespace polaris::random

#endif  // POLARIS_RANDOM_RNG_HPP

/// @file Unit tests for the seeded per-source RNG (design doc §3.6, §182).
///
/// The properties that matter here are not "is it random" — SplitMix64's
/// statistical quality is established upstream — but the ones the determinism
/// mandate rests on: identical seeds replay bit-for-bit, distinct streams are
/// independent, and (the §182 requirement) adding a stream never disturbs the
/// others. The distributional checks are coarse sanity, not a PRNG test suite.

#include "random/rng.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <vector>

namespace rnd = polaris::random;

TEST(Rng, IsBitReproducibleFromItsSeed) {
  rnd::SplitMix64 a(12345);
  rnd::SplitMix64 b(12345);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(a.nextU64(), b.nextU64()) << "draw " << i;
  }
}

TEST(Rng, DifferentSeedsDiverge) {
  rnd::SplitMix64 a(1);
  rnd::SplitMix64 b(2);
  // The very first outputs must already differ (SplitMix mixes the seed before
  // the first draw), so a one-bit seed change is not a one-bit output change.
  EXPECT_NE(a.nextU64(), b.nextU64());
}

TEST(Rng, StreamDerivationIsIndependentPerSource) {
  // The §182 property: each stream id derives its own seed, so introducing a new
  // source (a new id) leaves every existing stream's sequence untouched.
  constexpr std::uint64_t master = 0xDEADBEEF;
  rnd::SplitMix64 gyro = rnd::streamRng(master, 1);
  rnd::SplitMix64 mag = rnd::streamRng(master, 2);

  std::vector<std::uint64_t> gyro_first;
  for (int i = 0; i < 8; ++i) {
    gyro_first.push_back(gyro.nextU64());
    (void)mag.nextU64();  // "use" the other source; it must not affect gyro below
  }

  // Re-derive the gyro stream after "adding" the mag stream — its sequence is
  // identical, proving the streams do not share a sequence.
  rnd::SplitMix64 gyro_again = rnd::streamRng(master, 1);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(gyro_again.nextU64(), gyro_first[i]) << "draw " << i;
  }

  // And the two streams are not the same sequence.
  EXPECT_NE(rnd::streamSeed(master, 1), rnd::streamSeed(master, 2));
}

TEST(Rng, DistinctStreamIdsGiveDistinctSeeds) {
  constexpr std::uint64_t master = 42;
  std::set<std::uint64_t> seeds;
  for (std::uint64_t id = 0; id < 256; ++id) {
    seeds.insert(rnd::streamSeed(master, id));
  }
  EXPECT_EQ(seeds.size(), 256u) << "stream-seed collision within 256 sources";
}

TEST(Rng, UniformStaysInUnitInterval) {
  rnd::SplitMix64 rng(7);
  double sum = 0.0;
  constexpr int n = 100000;
  for (int i = 0; i < n; ++i) {
    const double u = rng.uniform();
    ASSERT_GE(u, 0.0);
    ASSERT_LT(u, 1.0);
    sum += u;
  }
  EXPECT_NEAR(sum / n, 0.5, 0.01) << "uniform mean far from 0.5";
}

TEST(Rng, GaussianHasApproxZeroMeanUnitVariance) {
  rnd::SplitMix64 rng(99);
  constexpr int n = 200000;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double x = rng.gaussian();
    sum += x;
    sum_sq += x * x;
  }
  const double mean = sum / n;
  const double var = sum_sq / n - mean * mean;
  EXPECT_NEAR(mean, 0.0, 0.02);
  EXPECT_NEAR(var, 1.0, 0.02);
}

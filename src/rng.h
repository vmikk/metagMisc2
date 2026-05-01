#pragma once

#include <cstdint>
#include <cstring>
#include <random>

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Deterministic RNG seed from double seed + column + depth index + replicate
// The IEEE 754 bit-pattern of seed_d is used as the initial key material,
// then mixed through three SplitMix64 rounds with col, d_idx, and rep
inline void seed_rep_rng(std::mt19937_64& rng, double seed_d, int col, int d_idx, int rep) {
  uint64_t seed = 0;
  std::memcpy(&seed, &seed_d, sizeof(double));
  seed ^= splitmix64(static_cast<uint64_t>(static_cast<uint32_t>(col)) + 1ULL);
  seed = splitmix64(seed);
  seed ^= static_cast<uint64_t>(static_cast<uint32_t>(d_idx));
  seed = splitmix64(seed);
  seed ^= static_cast<uint64_t>(static_cast<uint32_t>(rep));
  seed = splitmix64(seed);
  rng.seed(seed);
}

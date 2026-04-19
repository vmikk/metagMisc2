#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>

// Exact hypergeometric draw: population N, K successes, sample n (without replacement).
// Sequential urn (O(n)); suitable for moderate n. For very large n consider kernel="permutation".
inline int rhyper_mt(std::mt19937_64& rng, int64_t N, int64_t K, int n) {
  if (n <= 0 || K <= 0) {
    return 0;
  }
  if (N <= 0) {
    return 0;
  }
  if (K > N) {
    K = N;
  }
  if (n > N) {
    n = static_cast<int>(N);
  }
  if (K == N) {
    return n;
  }
  if (N > INT_MAX) {
    throw std::runtime_error(
        "rarefy: rarefaction population size exceeds INT_MAX; use subsampling or smaller counts");
  }
  int Ni = static_cast<int>(N);
  int Ki = static_cast<int>(K);
  int x = 0;
  for (int i = 0; i < n; ++i) {
    if (Ki <= 0) {
      break;
    }
    if (Ni <= Ki) {
      x += n - i;
      break;
    }
    const double u = std::generate_canonical<double, 53>(rng);
    if (u * static_cast<double>(Ni) < static_cast<double>(Ki)) {
      ++x;
      --Ki;
    }
    --Ni;
  }
  return x;
}

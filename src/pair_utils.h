#pragma once
#include <cmath>
#include <cstddef>

// Upper-triangular (i < j, 0-indexed) flat index for an n x n matrix
static inline size_t tri_index(int n, int i, int j) {
  return static_cast<size_t>(i) * static_cast<size_t>(2 * n - i - 1) / 2U +
         static_cast<size_t>(j - i - 1);
}

// Inverse of tri_index: given flat linear index lin, return row i and column j
// O(1) closed-form using the quadratic formula for triangular numbers
static inline void linear_to_pair(int n, size_t lin, int& i, int& j) {
  const double fn   = static_cast<double>(n);
  const double flin = static_cast<double>(lin);
  const double disc = (2.0 * fn - 1.0) * (2.0 * fn - 1.0) - 8.0 * flin;
  i = static_cast<int>(std::floor((2.0 * fn - 1.0 - std::sqrt(disc)) / 2.0));
  j = static_cast<int>(static_cast<size_t>(lin) - tri_index(n, i, i + 1) +
                        static_cast<size_t>(i) + 1U);
}

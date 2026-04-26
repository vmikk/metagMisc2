#include "RcppSparse.h"
#include "online_stats.h"
#include "rng.h"
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

static size_t tri_index(int n, int i, int j) {
  return static_cast<size_t>(i) * static_cast<size_t>(2 * n - i - 1) / 2U +
         static_cast<size_t>(j - i - 1);
}

static void linear_to_pair(int n, size_t lin, int& i, int& j) {
  size_t acc = 0;
  for (i = 0; i < n - 1; ++i) {
    const int row_len = n - 1 - i;
    if (lin < acc + static_cast<size_t>(row_len)) {
      j = static_cast<int>(static_cast<size_t>(i) + 1U + (lin - acc));
      return;
    }
    acc += static_cast<size_t>(row_len);
  }
  i = 0;
  j = 0;
}

static void to_pa_inplace(std::vector<double>& v) {
  for (double& x : v) {
    if (x > 0.0) {
      x = 1.0;
    }
  }
}

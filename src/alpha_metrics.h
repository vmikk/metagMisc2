#pragma once

#include <cmath>
#include <limits>
#include <vector>

// out:
//   [0]=richness
//   [1]=shannon, 
//   [2]=hill1, 
//   [3]=hill2 (inverse Simpson), 
//   [4]=simpson_dom (1-sum p^2),
//   [5]=evenness (Shannon / log S)
inline void compute_alpha_metrics(const std::vector<int>& /*idx*/,
                                  const std::vector<double>& cnt,
                                  double depth_d,
                                  double* out) {
  const int nnz = static_cast<int>(cnt.size());
  out[0] = static_cast<double>(nnz);
  if (nnz == 0 || depth_d <= 0) {
    for (int k = 1; k < 6; ++k) {
      out[k] = std::numeric_limits<double>::quiet_NaN();
    }
    return;
  }
  double shannon = 0.0;
  double sumsq = 0.0;
  for (int t = 0; t < nnz; ++t) {
    const double c = cnt[t];
    if (c <= 0) continue;
    const double p = c / depth_d;
    shannon -= p * std::log(p);
    sumsq += p * p;
  }
  out[1] = shannon;
  out[2] = std::exp(shannon);
  out[4] = 1.0 - sumsq;
  out[3] = (sumsq > 0.0) ? (1.0 / sumsq) : std::numeric_limits<double>::quiet_NaN();
  if (out[0] > 1.0) {
    out[5] = shannon / std::log(out[0]);
  } else {
    out[5] = std::numeric_limits<double>::quiet_NaN();
  }
}

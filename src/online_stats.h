#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

struct OnlineStats {
  int64_t n;
  double mean;
  double M2;
  double vmin;
  double vmax;

  OnlineStats()
 : n(0),
        mean(0),
        M2(0),
        vmin(std::numeric_limits<double>::infinity()),
        vmax(-std::numeric_limits<double>::infinity()) {}

  void clear() {
    n = 0;
    mean = 0;
    M2 = 0;
    vmin = std::numeric_limits<double>::infinity();
    vmax = -std::numeric_limits<double>::infinity();
  }

  void update(double x) {
    if (!std::isfinite(x)) return;
    ++n;
    const double d = x - mean;
    mean += d / static_cast<double>(n);
    const double d2 = x - mean;
    M2 += d * d2;
    vmin = std::min(vmin, x);
    vmax = std::max(vmax, x);
  }

  double variance_sample() const {
    return n > 1 ? M2 / static_cast<double>(n - 1) : 0.0;
  }
};

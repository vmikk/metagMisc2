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

static double dist_bray(const std::vector<int>& i1, const std::vector<double>& v1,
                        const std::vector<int>& i2, const std::vector<double>& v2) {
  double sum1 = 0, sum2 = 0, smin = 0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      smin += std::min(v1[a], v2[b]);
      sum1 += v1[a];
      sum2 += v2[b];
      ++a;
      ++b;
    } else if (i1[a] < i2[b]) {
      sum1 += v1[a];
      ++a;
    } else {
      sum2 += v2[b];
      ++b;
    }
  }
  while (a < i1.size()) {
    sum1 += v1[a++];
  }
  while (b < i2.size()) {
    sum2 += v2[b++];
  }
  const double den = sum1 + sum2;
  if (den <= 0.0) {
    return NA_REAL;
  }
  return 1.0 - 2.0 * smin / den;
}

static double dist_euclidean(const std::vector<int>& i1, const std::vector<double>& v1,
                             const std::vector<int>& i2, const std::vector<double>& v2) {
  double s = 0.0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      const double d = v1[a] - v2[b];
      s += d * d;
      ++a;
      ++b;
    } else if (i1[a] < i2[b]) {
      s += v1[a] * v1[a];
      ++a;
    } else {
      s += v2[b] * v2[b];
      ++b;
    }
  }
  while (a < i1.size()) {
    s += v1[a] * v1[a];
    ++a;
  }
  while (b < i2.size()) {
    s += v2[b] * v2[b];
    ++b;
  }
  return std::sqrt(s);
}

static double dist_hellinger(const std::vector<int>& i1, std::vector<double> v1,
                             const std::vector<int>& i2, std::vector<double> v2) {
  double s1 = 0, s2 = 0;
  for (double x : v1) {
    s1 += x;
  }
  for (double x : v2) {
    s2 += x;
  }
  if (s1 <= 0.0 || s2 <= 0.0) {
    return NA_REAL;
  }
  const double is1 = 1.0 / s1;
  const double is2 = 1.0 / s2;
  for (double& x : v1) {
    x = std::sqrt(x * is1);
  }
  for (double& x : v2) {
    x = std::sqrt(x * is2);
  }
  return dist_euclidean(i1, v1, i2, v2);
}

static double dist_simpson(const std::vector<int>& i1, const std::vector<double>& v1,
                           const std::vector<int>& i2, const std::vector<double>& v2) {
  double s1 = 0, s2 = 0;
  for (double x : v1) {
    s1 += x;
  }
  for (double x : v2) {
    s2 += x;
  }
  if (s1 <= 0.0 || s2 <= 0.0) {
    return NA_REAL;
  }
  double uij = 0, uji = 0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      uij += v1[a] / s1;
      uji += v2[b] / s2;
      ++a;
      ++b;
    } else if (i1[a] < i2[b]) {
      ++a;
    } else {
      ++b;
    }
  }
  const double uv = uij * uji;
  const double u_diff = uij - uv;
  const double v_diff = uji - uv;
  const double eps = 1e-17;
  return 1.0 - uv / (uv + std::min(u_diff, v_diff) + eps);
}

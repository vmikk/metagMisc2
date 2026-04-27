#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hypergeometric.h"
#include "RcppSparse.h"

struct PreparedColumn {
  std::vector<int> row;
  std::vector<int64_t> count;
  int64_t total = 0;
};

inline int64_t col_sum_rounded(RcppSparse::Matrix& A, int col) {
  int64_t s = 0;
  for (RcppSparse::Matrix::InnerIterator it(A, col); it; ++it) {
    s += static_cast<int64_t>(std::llround(it.value()));
  }
  return s;
}

inline void prepare_col(RcppSparse::Matrix& A, int col, PreparedColumn& out) {
  out.row.clear();
  out.count.clear();
  out.total = 0;

  int* Ap = A.outerIndexPtr().begin();
  int* Ai = A.innerIndexPtr().begin();
  double* Ax = A.nonzeros().begin();
  const int p0 = Ap[col];
  const int p1 = Ap[col + 1];
  out.row.reserve(static_cast<size_t>(p1 - p0));
  out.count.reserve(static_cast<size_t>(p1 - p0));

  for (int t = p0; t < p1; ++t) {
    const int64_t c = static_cast<int64_t>(std::llround(Ax[t]));
    if (c <= 0) {
      continue;
    }
    out.row.push_back(Ai[t]);
    out.count.push_back(c);
    out.total += c;
  }
}

inline void copy_prepared_column(const PreparedColumn& src, std::vector<int>& idx,
                                 std::vector<double>& cnt) {
  idx = src.row;
  cnt.resize(src.count.size());
  for (size_t t = 0; t < src.count.size(); ++t) {
    cnt[t] = static_cast<double>(src.count[t]);
  }
}

inline void rarefy_col_hyper(const PreparedColumn& src, int target_depth, std::vector<int>& idx,
                             std::vector<double>& cnt, std::mt19937_64& rng) {
  idx.clear();
  cnt.clear();
  if (target_depth <= 0 || target_depth > src.total) {
    return;
  }
  if (target_depth == src.total) {
    copy_prepared_column(src, idx, cnt);
    return;
  }

  int remaining = target_depth;
  int64_t remaining_N = src.total;
  for (size_t t = 0; t < src.count.size() && remaining > 0; ++t) {
    const int row = src.row[t];
    const int64_t c = src.count[t];
    if (c <= 0) {
      continue;
    }
    const int drawn = rhyper_mt(rng, remaining_N, c, remaining);
    if (drawn > 0) {
      idx.push_back(row);
      cnt.push_back(static_cast<double>(drawn));
    }
    remaining -= drawn;
    remaining_N -= c;
  }
}


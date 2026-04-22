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

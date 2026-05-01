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

inline void rarefy_col_hyper(RcppSparse::Matrix& A, int col, int target_depth, int64_t total,
                             std::vector<int>& idx, std::vector<double>& cnt,
                             std::mt19937_64& rng) {
  idx.clear();
  cnt.clear();
  if (target_depth <= 0 || target_depth > total) {
    return;
  }

  int* Ap = A.outerIndexPtr().begin();
  int* Ai = A.innerIndexPtr().begin();
  double* Ax = A.nonzeros().begin();
  const int p0 = Ap[col];
  const int p1 = Ap[col + 1];

  if (target_depth == total) {
    for (int t = p0; t < p1; ++t) {
      const double v = std::round(Ax[t]);
      if (v > 0) {
        idx.push_back(Ai[t]);
        cnt.push_back(v);
      }
    }
    return;
  }

  int remaining = target_depth;
  int64_t remaining_N = total;
  for (int t = p0; t < p1 && remaining > 0; ++t) {
    const int64_t c = static_cast<int64_t>(std::llround(Ax[t]));
    if (c <= 0) {
      continue;
    }
    const int drawn = rhyper_mt(rng, remaining_N, c, remaining);
    if (drawn > 0) {
      idx.push_back(Ai[t]);
      cnt.push_back(static_cast<double>(drawn));
    }
    remaining -= drawn;
    remaining_N -= c;
  }
}

inline void rarefy_col_hyper(RcppSparse::Matrix& A, int col, int target_depth,
                             std::vector<int>& idx, std::vector<double>& cnt,
                             std::mt19937_64& rng) {
  rarefy_col_hyper(A, col, target_depth, col_sum_rounded(A, col), idx, cnt, rng);
}

inline void rarefy_col_perm(const PreparedColumn& src, int target_depth, std::vector<int>& idx,
                            std::vector<double>& cnt, std::vector<int>& pool,
                            std::mt19937_64& rng) {
  idx.clear();
  cnt.clear();
  pool.clear();
  if (target_depth <= 0) {
    return;
  }
  for (size_t t = 0; t < src.count.size(); ++t) {
    const int r = src.row[t];
    const int64_t c = src.count[t];
    for (int64_t u = 0; u < c; ++u) {
      pool.push_back(r);
    }
  }
  if (static_cast<int>(pool.size()) < target_depth) {
    return;
  }
  // Partial Fisher-Yates: O(depth) instead of O(N) full shuffle
  // Selects exactly target_depth elements uniformly without replacement
  const int sz = static_cast<int>(pool.size());
  for (int i = 0; i < target_depth; ++i) {
    std::uniform_int_distribution<int> dist(i, sz - 1);
    std::swap(pool[i], pool[dist(rng)]);
  }
  std::sort(pool.begin(), pool.begin() + target_depth);
  int cur = pool[0];
  int run = 1;
  for (int u = 1; u < target_depth; ++u) {
    if (pool[u] == cur) {
      ++run;
    } else {
      idx.push_back(cur);
      cnt.push_back(static_cast<double>(run));
      cur = pool[u];
      run = 1;
    }
  }
  idx.push_back(cur);
  cnt.push_back(static_cast<double>(run));
}

inline void rarefy_col_perm(RcppSparse::Matrix& A, int col, int target_depth,
                            std::vector<int>& idx, std::vector<double>& cnt,
                            std::vector<int>& pool, std::mt19937_64& rng) {
  idx.clear();
  cnt.clear();
  pool.clear();
  if (target_depth <= 0) {
    return;
  }
  int* Ap = A.outerIndexPtr().begin();
  int* Ai = A.innerIndexPtr().begin();
  double* Ax = A.nonzeros().begin();
  const int p0 = Ap[col];
  const int p1 = Ap[col + 1];
  for (int t = p0; t < p1; ++t) {
    const int r = Ai[t];
    const int c = static_cast<int>(std::llround(Ax[t]));
    for (int u = 0; u < c; ++u) {
      pool.push_back(r);
    }
  }
  if (static_cast<int>(pool.size()) < target_depth) {
    return;
  }
  // Partial Fisher-Yates: O(depth) instead of O(N) full shuffle
  // Selects exactly target_depth elements uniformly without replacement
  const int sz = static_cast<int>(pool.size());
  for (int i = 0; i < target_depth; ++i) {
    std::uniform_int_distribution<int> dist(i, sz - 1);
    std::swap(pool[i], pool[dist(rng)]);
  }
  std::sort(pool.begin(), pool.begin() + target_depth);
  int cur = pool[0];
  int run = 1;
  for (int u = 1; u < target_depth; ++u) {
    if (pool[u] == cur) {
      ++run;
    } else {
      idx.push_back(cur);
      cnt.push_back(static_cast<double>(run));
      cur = pool[u];
      run = 1;
    }
  }
  idx.push_back(cur);
  cnt.push_back(static_cast<double>(run));
}

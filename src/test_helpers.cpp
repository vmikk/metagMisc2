#include <Rcpp.h>

#include "rarefy_ops.h"
#include "rng.h"

// [[Rcpp::export(name = ".test_rhyper_mean_cpp")]]
Rcpp::NumericVector test_rhyper_mean_cpp_impl(Rcpp::IntegerVector N, Rcpp::IntegerVector K,
                                              Rcpp::IntegerVector n, int n_iter, double seed) {
  if (N.size() != K.size() || N.size() != n.size()) {
    Rcpp::stop("N, K, and n must have the same length");
  }
  if (n_iter < 1) {
    Rcpp::stop("n_iter must be >= 1");
  }

  const int m = N.size();
  Rcpp::NumericVector out(m);
  std::mt19937_64 rng;
  for (int i = 0; i < m; ++i) {
    seed_rep_rng(rng, seed, i, 0, 0);
    double acc = 0.0;
    for (int rep = 0; rep < n_iter; ++rep) {
      acc += rhyper_mt(rng, static_cast<int64_t>(N[i]), static_cast<int64_t>(K[i]), n[i]);
    }
    out[i] = acc / static_cast<double>(n_iter);
  }
  return out;
}

// [[Rcpp::export(name = ".test_mvhyper_mean_cpp")]]
Rcpp::NumericVector test_mvhyper_mean_cpp_impl(Rcpp::IntegerVector counts, int depth, int n_iter,
                                               double seed, int kernel = 0) {
  if (n_iter < 1) {
    Rcpp::stop("n_iter must be >= 1");
  }

  PreparedColumn prep;
  prep.row.reserve(static_cast<size_t>(counts.size()));
  prep.count.reserve(static_cast<size_t>(counts.size()));
  for (int i = 0; i < counts.size(); ++i) {
    const int64_t c = static_cast<int64_t>(counts[i]);
    if (c <= 0) {
      continue;
    }
    prep.row.push_back(i);
    prep.count.push_back(c);
    prep.total += c;
  }
  if (depth > prep.total) {
    Rcpp::stop("depth exceeds total count");
  }

  Rcpp::NumericVector out(counts.size());
  std::vector<int> idx;
  std::vector<double> cnt;
  std::vector<int> pool;
  std::mt19937_64 rng;

  for (int rep = 0; rep < n_iter; ++rep) {
    seed_rep_rng(rng, seed, 0, 0, rep);
    if (kernel == 0) {
      rarefy_col_hyper(prep, depth, idx, cnt, rng);
    } else {
      rarefy_col_perm(prep, depth, idx, cnt, pool, rng);
    }
    for (size_t j = 0; j < idx.size(); ++j) {
      out[idx[j]] += cnt[j];
    }
  }

  out = out / static_cast<double>(n_iter);
  return out;
}

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

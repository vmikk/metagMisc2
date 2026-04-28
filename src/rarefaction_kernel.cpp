#include "RcppSparse.h"
#include "alpha_metrics.h"
#include "online_stats.h"
#include "rng.h"
#include "rarefy_ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>

// [[Rcpp::export]]
Rcpp::S4 rarefy_single_matrix_cpp(Rcpp::S4 mat, int depth, double seed, int kernel) {
  RcppSparse::Matrix A(mat);
  const int nr = static_cast<int>(A.rows());
  const int nc = static_cast<int>(A.cols());
  std::vector<int> Ti, Tj;
  std::vector<double> Tx;
  Ti.reserve(A.n_nonzero());
  Tj.reserve(A.n_nonzero());
  Tx.reserve(A.n_nonzero());

  for (int col = 0; col < nc; ++col) {
    PreparedColumn prep;
    prepare_col(A, col, prep);
    if (depth > prep.total) {
      Rcpp::stop("depth exceeds column sum for at least one sample");
    }
    std::vector<int> idx;
    std::vector<double> cnt;
    std::vector<int> pool;
    std::mt19937_64 rng;
    seed_rep_rng(rng, seed, col, 0, 0);
    if (kernel == 0) {
      rarefy_col_hyper(prep, depth, idx, cnt, rng);
    } else {
      rarefy_col_perm(prep, depth, idx, cnt, pool, rng);
    }
    for (size_t t = 0; t < idx.size(); ++t) {
      Ti.push_back(idx[t]);
      Tj.push_back(col);
      Tx.push_back(cnt[t]);
    }
  }

  Rcpp::Environment menv = Rcpp::Environment::namespace_env("Matrix");
  Rcpp::Function sparseMatrix = menv["sparseMatrix"];
  Rcpp::IntegerVector dims = Rcpp::IntegerVector::create(nr, nc);
  return sparseMatrix(Rcpp::_["i"] = Ti, Rcpp::_["j"] = Tj, Rcpp::_["x"] = Tx,
                      Rcpp::_["dims"] = dims, Rcpp::_["index1"] = false,
                      Rcpp::_["repr"] = "C");
}

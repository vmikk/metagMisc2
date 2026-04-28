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
Rcpp::List rarefy_alpha_cpp(Rcpp::S4 mat, Rcpp::IntegerVector depths, int n_iter,
                            Rcpp::LogicalVector metric_mask, int n_threads, double seed,
                            int kernel) {
  RcppSparse::Matrix A(mat);
  const int ns = static_cast<int>(A.cols());
  const int nd = depths.size();
  if (metric_mask.size() != 6) {
    Rcpp::stop("metric_mask must have length 6");
  }
  if (n_iter < 1) {
    Rcpp::stop("n_iter must be >= 1");
  }

  std::vector<OnlineStats> acc(static_cast<size_t>(ns) * static_cast<size_t>(nd) * 6U);
  auto ix = [&](int s, int d, int m) -> size_t {
    return (static_cast<size_t>(s) * static_cast<size_t>(nd) + static_cast<size_t>(d)) * 6U +
           static_cast<size_t>(m);
  };

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < ns; ++s) {
    PreparedColumn prep;
    prepare_col(A, s, prep);
    std::vector<int> idx;
    std::vector<double> cnt;
    std::vector<int> pool;
    std::mt19937_64 rng;
    double alpha_out[6];
    for (int d = 0; d < nd; ++d) {
      const int dep = depths[d];
      if (dep > prep.total) {
        continue;
      }
      for (int rep = 0; rep < n_iter; ++rep) {
        seed_rep_rng(rng, seed, s, d, rep);
        if (kernel == 0) {
          rarefy_col_hyper(prep, dep, idx, cnt, rng);
        } else {
          rarefy_col_perm(prep, dep, idx, cnt, pool, rng);
        }
        compute_alpha_metrics(idx, cnt, static_cast<double>(dep), alpha_out);
        for (int m = 0; m < 6; ++m) {
          if (!metric_mask[m]) {
            continue;
          }
          acc[ix(s, d, m)].update(alpha_out[m]);
        }
      }
    }
  }

  Rcpp::List mean_l(6), sd_l(6), min_l(6), max_l(6);
  const Rcpp::CharacterVector mnames =
      Rcpp::CharacterVector::create("richness", "shannon", "hill1", "hill2", "simpson_dom",
                                    "evenness");
  mean_l.attr("names") = Rcpp::clone(mnames);
  sd_l.attr("names") = Rcpp::clone(mnames);
  min_l.attr("names") = Rcpp::clone(mnames);
  max_l.attr("names") = Rcpp::clone(mnames);

  for (int m = 0; m < 6; ++m) {
    if (!metric_mask[m]) {
      mean_l[m] = R_NilValue;
      sd_l[m] = R_NilValue;
      min_l[m] = R_NilValue;
      max_l[m] = R_NilValue;
      continue;
    }
    Rcpp::NumericMatrix Mmean(ns, nd), Msd(ns, nd), Mmin(ns, nd), Mmax(ns, nd);
    for (int s = 0; s < ns; ++s) {
      for (int d = 0; d < nd; ++d) {
        const OnlineStats& o = acc[ix(s, d, m)];
        if (o.n > 0) {
          Mmean(s, d) = o.mean;
          Msd(s, d) = o.n > 1 ? std::sqrt(o.variance_sample()) : NA_REAL;
          Mmin(s, d) = o.vmin;
          Mmax(s, d) = o.vmax;
        } else {
          Mmean(s, d) = NA_REAL;
          Msd(s, d) = NA_REAL;
          Mmin(s, d) = NA_REAL;
          Mmax(s, d) = NA_REAL;
        }
      }
    }
    mean_l[m] = Mmean;
    sd_l[m] = Msd;
    min_l[m] = Mmin;
    max_l[m] = Mmax;
  }

  return Rcpp::List::create(Rcpp::Named("mean") = mean_l, Rcpp::Named("sd") = sd_l,
                            Rcpp::Named("min") = min_l, Rcpp::Named("max") = max_l);
}

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

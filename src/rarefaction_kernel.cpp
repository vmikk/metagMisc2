#include "RcppSparse.h"
#include "alpha_metrics.h"
#include "online_stats.h"
#include "phylo_tree.h"
#include "rng.h"
#include "rarefy_ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>

namespace {

// Compute Faith's PD for a single rarefied sample
// taxa: row indices of non-zero taxa (from rarefy_col_hyper / rarefy_col_perm)
static double faith_pd_from_sample(const PhyloTree& tree,
                                   const Rcpp::IntegerVector& row_to_tip,
                                   const std::vector<int>& taxa) {
  std::vector<unsigned char> present(static_cast<size_t>(tree.nnodes), 0);
  for (int row : taxa) {
    if (row < 0 || row >= row_to_tip.size()) {
      Rcpp::stop("rarefied taxon index is outside the row-to-tip map");
    }
    const int tip = row_to_tip[row];
    if (tip < 0 || tip >= tree.ntips) {
      Rcpp::stop("row-to-tip map contains invalid tip indices");
    }
    present[static_cast<size_t>(tip)] = 1;
  }
  double pd = 0.0;
  for (int rank = 0; rank < tree.nedges; ++rank) {
    const int edge_id = tree.postorder_edges[static_cast<size_t>(rank)];
    const int child   = tree.child[static_cast<size_t>(edge_id)];
    const int parent  = tree.parent[static_cast<size_t>(edge_id)];
    if (present[static_cast<size_t>(child)]) {
      pd += tree.length[static_cast<size_t>(edge_id)];
      present[static_cast<size_t>(parent)] = 1;
    }
  }
  return pd;
}

}  // namespace

// Alpha-diversity kernel
//
// Computes up to 6 count-based metrics (richness, shannon, hill1, hill2,
// simpson_dom, evenness) and optionally Faith's PD, all in a single
// rarefaction pass per (sample, depth, replicate)
//
// count_mask : LogicalVector of length 6 — which count metrics to accumulate
// phylo_mask : LogicalVector of length 1 (faith_pd); pass an empty vector or
//              a single FALSE if no phylogenetic alpha is needed
// phylo      : the R phylo List (pass R_NilValue when phylo_mask is empty/FALSE)
// row_to_tip : integer map from matrix row -> 0-based tip index in the tree
//              (pass R_NilValue when phylo_mask is empty/FALSE)
//
// [[Rcpp::export]]
Rcpp::List rarefy_alpha_cpp(Rcpp::S4 mat, Rcpp::IntegerVector depths, int n_iter,
                            Rcpp::LogicalVector count_mask,
                            Rcpp::LogicalVector phylo_mask,
                            Rcpp::RObject phylo,
                            Rcpp::RObject row_to_tip_obj,
                            int n_threads, double seed, int kernel) {
  RcppSparse::Matrix A(mat);
  const int ns = static_cast<int>(A.cols());
  const int nd = depths.size();
  if (count_mask.size() != 6) {
    Rcpp::stop("count_mask must have length 6");
  }
  if (n_iter < 1) {
    Rcpp::stop("n_iter must be >= 1");
  }

  // Resolve optional phylo arguments.
  const bool want_faith = (phylo_mask.size() >= 1) && (bool)phylo_mask[0];
  PhyloTree tree;
  Rcpp::IntegerVector row_to_tip;
  if (want_faith) {
    if (phylo.isNULL()) {
      Rcpp::stop("phylo must be provided when faith_pd is requested");
    }
    if (row_to_tip_obj.isNULL()) {
      Rcpp::stop("row_to_tip must be provided when faith_pd is requested");
    }
    tree        = parse_phylo(Rcpp::as<Rcpp::List>(phylo));
    row_to_tip  = Rcpp::as<Rcpp::IntegerVector>(row_to_tip_obj);
    if (tree.ntips != static_cast<int>(A.rows())) {
      Rcpp::stop("phy_tree tip count must match the number of matrix rows after pruning");
    }
    if (row_to_tip.size() != static_cast<int>(A.rows())) {
      Rcpp::stop("row_to_tip must have one entry per matrix row");
    }
  }

  // Accumulators: [sample x depth x 6] for count metrics
  std::vector<OnlineStats> acc_count(
      static_cast<size_t>(ns) * static_cast<size_t>(nd) * 6U);
  auto ix_c = [&](int s, int d, int m) -> size_t {
    return (static_cast<size_t>(s) * static_cast<size_t>(nd) +
            static_cast<size_t>(d)) *
               6U +
           static_cast<size_t>(m);
  };

  // Accumulators: [sample x depth] for Faith PD
  std::vector<OnlineStats> acc_pd(
      want_faith ? static_cast<size_t>(ns) * static_cast<size_t>(nd) : 0U);
  auto ix_pd = [&](int s, int d) -> size_t {
    return static_cast<size_t>(s) * static_cast<size_t>(nd) +
           static_cast<size_t>(d);
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
    std::vector<int>    idx;
    std::vector<double> cnt;
    std::vector<int>    pool;
    std::mt19937_64     rng;
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
          if (count_mask[m]) {
            acc_count[ix_c(s, d, m)].update(alpha_out[m]);
          }
        }
        if (want_faith) {
          acc_pd[ix_pd(s, d)].update(
              faith_pd_from_sample(tree, row_to_tip, idx));
        }
      }
    }
  }

  // ---- build output for count metrics ----
  Rcpp::List mean_c(6), sd_c(6), min_c(6), max_c(6);
  const Rcpp::CharacterVector cnames =
      Rcpp::CharacterVector::create("richness", "shannon", "hill1", "hill2",
                                    "simpson_dom", "evenness");
  mean_c.attr("names") = Rcpp::clone(cnames);
  sd_c.attr("names")   = Rcpp::clone(cnames);
  min_c.attr("names")  = Rcpp::clone(cnames);
  max_c.attr("names")  = Rcpp::clone(cnames);

  for (int m = 0; m < 6; ++m) {
    if (!count_mask[m]) {
      mean_c[m] = R_NilValue;
      sd_c[m]   = R_NilValue;
      min_c[m]  = R_NilValue;
      max_c[m]  = R_NilValue;
      continue;
    }
    Rcpp::NumericMatrix Mmean(ns, nd), Msd(ns, nd), Mmin(ns, nd), Mmax(ns, nd);
    for (int s = 0; s < ns; ++s) {
      for (int d = 0; d < nd; ++d) {
        const OnlineStats& o = acc_count[ix_c(s, d, m)];
        if (o.n > 0) {
          Mmean(s, d) = o.mean;
          Msd(s, d)   = o.n > 1 ? std::sqrt(o.variance_sample()) : NA_REAL;
          Mmin(s, d)  = o.vmin;
          Mmax(s, d)  = o.vmax;
        } else {
          Mmean(s, d) = NA_REAL;
          Msd(s, d)   = NA_REAL;
          Mmin(s, d)  = NA_REAL;
          Mmax(s, d)  = NA_REAL;
        }
      }
    }
    mean_c[m] = Mmean;
    sd_c[m]   = Msd;
    min_c[m]  = Mmin;
    max_c[m]  = Mmax;
  }

  // ---- build output for Faith PD ----
  Rcpp::List mean_pd(1), sd_pd(1), min_pd(1), max_pd(1);
  const Rcpp::CharacterVector pnames = Rcpp::CharacterVector::create("faith_pd");
  mean_pd.attr("names") = Rcpp::clone(pnames);
  sd_pd.attr("names")   = Rcpp::clone(pnames);
  min_pd.attr("names")  = Rcpp::clone(pnames);
  max_pd.attr("names")  = Rcpp::clone(pnames);

  if (!want_faith) {
    mean_pd[0] = R_NilValue;
    sd_pd[0]   = R_NilValue;
    min_pd[0]  = R_NilValue;
    max_pd[0]  = R_NilValue;
  } else {
    Rcpp::NumericMatrix Mmean(ns, nd), Msd(ns, nd), Mmin(ns, nd), Mmax(ns, nd);
    for (int s = 0; s < ns; ++s) {
      for (int d = 0; d < nd; ++d) {
        const OnlineStats& o = acc_pd[ix_pd(s, d)];
        if (o.n > 0) {
          Mmean(s, d) = o.mean;
          Msd(s, d)   = o.n > 1 ? std::sqrt(o.variance_sample()) : NA_REAL;
          Mmin(s, d)  = o.vmin;
          Mmax(s, d)  = o.vmax;
        } else {
          Mmean(s, d) = NA_REAL;
          Msd(s, d)   = NA_REAL;
          Mmin(s, d)  = NA_REAL;
          Mmax(s, d)  = NA_REAL;
        }
      }
    }
    mean_pd[0] = Mmean;
    sd_pd[0]   = Msd;
    min_pd[0]  = Mmin;
    max_pd[0]  = Mmax;
  }

  return Rcpp::List::create(
      Rcpp::Named("count") =
          Rcpp::List::create(Rcpp::Named("mean") = mean_c,
                             Rcpp::Named("sd")   = sd_c,
                             Rcpp::Named("min")  = min_c,
                             Rcpp::Named("max")  = max_c),
      Rcpp::Named("phylo") =
          Rcpp::List::create(Rcpp::Named("mean") = mean_pd,
                             Rcpp::Named("sd")   = sd_pd,
                             Rcpp::Named("min")  = min_pd,
                             Rcpp::Named("max")  = max_pd));
}

// [[Rcpp::export]]
Rcpp::S4 rarefy_single_matrix_cpp(Rcpp::S4 mat, int depth, double seed, int kernel) {
  RcppSparse::Matrix A(mat);
  const int nr = static_cast<int>(A.rows());
  const int nc = static_cast<int>(A.cols());
  std::vector<int>    Ti, Tj;
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
    std::vector<int>    idx;
    std::vector<double> cnt;
    std::vector<int>    pool;
    std::mt19937_64     rng;
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

  Rcpp::Environment    menv        = Rcpp::Environment::namespace_env("Matrix");
  Rcpp::Function       sparseMatrix = menv["sparseMatrix"];
  Rcpp::IntegerVector  dims        = Rcpp::IntegerVector::create(nr, nc);
  return sparseMatrix(Rcpp::_["i"] = Ti, Rcpp::_["j"] = Tj,
                      Rcpp::_["x"] = Tx, Rcpp::_["dims"] = dims,
                      Rcpp::_["index1"] = false, Rcpp::_["repr"] = "C");
}

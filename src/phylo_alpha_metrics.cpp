#include "RcppSparse.h"
#include "online_stats.h"
#include "phylo_tree.h"
#include "rng.h"
#include "rarefy_ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

static double faith_pd_from_sample(const PhyloTree& tree, const Rcpp::IntegerVector& row_to_tip,
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
    const int child = tree.child[static_cast<size_t>(edge_id)];
    const int parent = tree.parent[static_cast<size_t>(edge_id)];
    if (present[static_cast<size_t>(child)]) {
      pd += tree.length[static_cast<size_t>(edge_id)];
      present[static_cast<size_t>(parent)] = 1;
    }
  }
  return pd;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List rarefy_phylo_alpha_cpp(Rcpp::S4 mat, Rcpp::IntegerVector depths, int n_iter,
                                  Rcpp::LogicalVector metric_mask, Rcpp::List phylo,
                                  Rcpp::IntegerVector row_to_tip, int n_threads,
                                  double seed, int kernel) {
  RcppSparse::Matrix A(mat);
  const int ns = static_cast<int>(A.cols());
  const int nd = depths.size();
  if (metric_mask.size() != 1) {
    Rcpp::stop("phylogenetic alpha metric_mask must have length 1");
  }
  if (n_iter < 1) {
    Rcpp::stop("n_iter must be >= 1");
  }
  if (row_to_tip.size() != static_cast<int>(A.rows())) {
    Rcpp::stop("row_to_tip must have one entry per matrix row");
  }
  const PhyloTree tree = parse_phylo(phylo);
  if (tree.ntips != static_cast<int>(A.rows())) {
    Rcpp::stop("phy_tree tip count must match the number of matrix rows after pruning");
  }

  std::vector<OnlineStats> acc(static_cast<size_t>(ns) * static_cast<size_t>(nd));
  auto ix = [&](int s, int d) -> size_t {
    return static_cast<size_t>(s) * static_cast<size_t>(nd) + static_cast<size_t>(d);
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
        if (metric_mask[0]) {
          acc[ix(s, d)].update(faith_pd_from_sample(tree, row_to_tip, idx));
        }
      }
    }
  }

  Rcpp::List mean_l(1), sd_l(1), min_l(1), max_l(1);
  const Rcpp::CharacterVector mnames = Rcpp::CharacterVector::create("faith_pd");
  mean_l.attr("names") = Rcpp::clone(mnames);
  sd_l.attr("names") = Rcpp::clone(mnames);
  min_l.attr("names") = Rcpp::clone(mnames);
  max_l.attr("names") = Rcpp::clone(mnames);

  if (!metric_mask[0]) {
    mean_l[0] = R_NilValue;
    sd_l[0] = R_NilValue;
    min_l[0] = R_NilValue;
    max_l[0] = R_NilValue;
  } else {
    Rcpp::NumericMatrix Mmean(ns, nd), Msd(ns, nd), Mmin(ns, nd), Mmax(ns, nd);
    for (int s = 0; s < ns; ++s) {
      for (int d = 0; d < nd; ++d) {
        const OnlineStats& o = acc[ix(s, d)];
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
    mean_l[0] = Mmean;
    sd_l[0] = Msd;
    min_l[0] = Mmin;
    max_l[0] = Mmax;
  }

  return Rcpp::List::create(Rcpp::Named("mean") = mean_l, Rcpp::Named("sd") = sd_l,
                            Rcpp::Named("min") = min_l, Rcpp::Named("max") = max_l);
}

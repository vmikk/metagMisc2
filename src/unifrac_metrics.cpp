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
#include <string>
#include <vector>

namespace {

struct SparseBranchMass {
  std::vector<int> idx;
  std::vector<double> val;
};

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

static bool is_unifrac_metric(const std::string& metric) {
  return metric == "unifrac_unweighted" || metric == "unifrac_weighted" ||
         metric == "unifrac_generalized" || metric == "unifrac_vaw";
}

static SparseBranchMass sample_to_branch_masses(const PhyloTree& tree, const Rcpp::IntegerVector& row_to_tip,
                                                const std::vector<int>& taxa,
                                                const std::vector<double>& counts, int depth) {
  std::vector<double> node_mass(static_cast<size_t>(tree.nnodes), 0.0);
  const double inv_depth = 1.0 / static_cast<double>(depth);
  for (size_t k = 0; k < taxa.size(); ++k) {
    const int row = taxa[k];
    if (row < 0 || row >= row_to_tip.size()) {
      Rcpp::stop("rarefied taxon index is outside the row-to-tip map");
    }
    const int tip = row_to_tip[row];
    if (tip < 0 || tip >= tree.ntips) {
      Rcpp::stop("row-to-tip map contains invalid tip indices");
    }
    node_mass[static_cast<size_t>(tip)] += counts[k] * inv_depth;
  }

  SparseBranchMass out;
  out.idx.reserve(taxa.size());
  out.val.reserve(taxa.size());
  for (int rank = 0; rank < tree.nedges; ++rank) {
    const int edge_id = tree.postorder_edges[static_cast<size_t>(rank)];
    const int child = tree.child[static_cast<size_t>(edge_id)];
    const int parent = tree.parent[static_cast<size_t>(edge_id)];
    const double mass = node_mass[static_cast<size_t>(child)];
    if (mass > 0.0) {
      out.idx.push_back(rank);
      out.val.push_back(mass);
    }
    node_mass[static_cast<size_t>(parent)] += mass;
  }
  return out;
}

static void accumulate_branch(const std::string& metric, const std::vector<double>& branch_length,
                              int rank, double p1, double p2, double alpha, double& num, double& den) {
  const double len = branch_length[static_cast<size_t>(rank)];
  if (len <= 0.0) {
    return;
  }
  const double psum = p1 + p2;
  if (psum <= 0.0) {
    return;
  }
  const double diff = std::fabs(p1 - p2);
  if (metric == "unifrac_unweighted") {
    den += len;
    if ((p1 > 0.0) != (p2 > 0.0)) {
      num += len;
    }
  } else if (metric == "unifrac_weighted") {
    num += len * diff;
    den += len * psum;
  } else if (metric == "unifrac_generalized") {
    const double weight = std::pow(psum, alpha);
    num += len * weight * diff / psum;
    den += len * weight;
  } else if (metric == "unifrac_vaw") {
    const double eps = 1e-15;
    const double variance = std::max(psum - diff * diff, eps);
    num += len * diff / std::sqrt(variance);
    den += len * std::sqrt(psum);
  }
}

static double dist_unifrac(const std::string& metric, const PhyloTree& tree,
                           const SparseBranchMass& a, const SparseBranchMass& b, double alpha) {
  double num = 0.0;
  double den = 0.0;
  size_t ia = 0;
  size_t ib = 0;
  while (ia < a.idx.size() || ib < b.idx.size()) {
    int rank;
    double p1 = 0.0;
    double p2 = 0.0;
    if (ib >= b.idx.size() || (ia < a.idx.size() && a.idx[ia] < b.idx[ib])) {
      rank = a.idx[ia];
      p1 = a.val[ia];
      ++ia;
    } else if (ia >= a.idx.size() || b.idx[ib] < a.idx[ia]) {
      rank = b.idx[ib];
      p2 = b.val[ib];
      ++ib;
    } else {
      rank = a.idx[ia];
      p1 = a.val[ia];
      p2 = b.val[ib];
      ++ia;
      ++ib;
    }
    accumulate_branch(metric, tree.postorder_length, rank, p1, p2, alpha, num, den);
  }
  if (den <= 0.0) {
    return NA_REAL;
  }
  return num / den;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::NumericMatrix rarefy_unifrac_mean_cpp(Rcpp::S4 mat, int depth, int n_iter,
                                            std::string metric, Rcpp::List phylo,
                                            Rcpp::IntegerVector row_to_tip, int n_threads,
                                            double seed, int kernel, double alpha) {
  if (!is_unifrac_metric(metric)) {
    Rcpp::stop(std::string("unknown UniFrac metric: ") + metric);
  }
  if (depth <= 0) {
    Rcpp::stop("depth must be positive for UniFrac");
  }
  if (metric == "unifrac_generalized" && (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0)) {
    Rcpp::stop("alpha must be in the 0-1 range for generalized UniFrac");
  }
  RcppSparse::Matrix A(mat);
  const int n = static_cast<int>(A.cols());
  if (n < 2) {
    Rcpp::stop("need at least two samples (columns) for beta diversity");
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

  const size_t npairs =
      static_cast<size_t>(n) * (static_cast<size_t>(n) - 1U) / 2U;
  std::vector<OnlineStats> w(npairs);
  std::vector<int64_t> col_totals(static_cast<size_t>(n));
  for (int col = 0; col < n; ++col) {
    col_totals[static_cast<size_t>(col)] = col_sum_rounded(A, col);
    if (depth > col_totals[static_cast<size_t>(col)]) {
      Rcpp::stop("depth exceeds column sum for at least one sample");
    }
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
#endif

  for (int rep = 0; rep < n_iter; ++rep) {
    std::vector<SparseBranchMass> branch_masses(static_cast<size_t>(n));
    for (int col = 0; col < n; ++col) {
      std::mt19937_64 rng;
      seed_rep_rng(rng, seed, col, 424242, rep);
      std::vector<int> idx;
      std::vector<double> cnt;
      std::vector<int> pool;
      if (kernel == 0) {
        rarefy_col_hyper(A, col, depth, col_totals[static_cast<size_t>(col)], idx, cnt, rng);
      } else {
        rarefy_col_perm(A, col, depth, idx, cnt, pool, rng);
      }
      branch_masses[static_cast<size_t>(col)] =
          sample_to_branch_masses(tree, row_to_tip, idx, cnt, depth);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t lin = 0; lin < npairs; ++lin) {
      int i = 0;
      int j = 0;
      linear_to_pair(n, lin, i, j);
      const double d = dist_unifrac(metric, tree, branch_masses[static_cast<size_t>(i)],
                                    branch_masses[static_cast<size_t>(j)], alpha);
      w[lin].update(d);
    }
  }

  Rcpp::NumericMatrix D(n, n);
  std::fill(D.begin(), D.end(), 0.0);
  for (int i = 0; i < n; ++i) {
    D(i, i) = 0.0;
  }
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const size_t k = tri_index(n, i, j);
      const OnlineStats& o = w[k];
      const double v = o.n > 0 ? o.mean : NA_REAL;
      D(i, j) = v;
      D(j, i) = v;
    }
  }
  return D;
}

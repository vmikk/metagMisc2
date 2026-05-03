#include "RcppSparse.h"
#include "online_stats.h"
#include "pair_utils.h"
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
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Sparse branch-mass representation for UniFrac
// ---------------------------------------------------------------------------

struct SparseBranchMass {
  std::vector<int>    idx;  // postorder rank of each branch with mass > 0
  std::vector<double> val;  // normalised mass (count / depth)
};

static SparseBranchMass
sample_to_branch_masses(const PhyloTree& tree,
                        const Rcpp::IntegerVector& row_to_tip,
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
    const int child   = tree.child[static_cast<size_t>(edge_id)];
    const int parent  = tree.parent[static_cast<size_t>(edge_id)];
    const double mass = node_mass[static_cast<size_t>(child)];
    if (mass > 0.0) {
      out.idx.push_back(rank);
      out.val.push_back(mass);
    }
    node_mass[static_cast<size_t>(parent)] += mass;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Metric enums — resolved once from strings before any inner loops
// ---------------------------------------------------------------------------

enum class BetaMetricId {
  bray_curtis,
  bray_curtis_pa,
  euclidean,
  hellinger,
  simpson,
  simpson_pa
};

enum class UniFracMetricId {
  unweighted,
  weighted,
  generalized,
  vaw
};

static BetaMetricId beta_metric_id(const std::string& s) {
  if (s == "bray_curtis")    return BetaMetricId::bray_curtis;
  if (s == "bray_curtis_pa") return BetaMetricId::bray_curtis_pa;
  if (s == "euclidean")      return BetaMetricId::euclidean;
  if (s == "hellinger")      return BetaMetricId::hellinger;
  if (s == "simpson")        return BetaMetricId::simpson;
  if (s == "simpson_pa")     return BetaMetricId::simpson_pa;
  Rcpp::stop("unknown beta metric: " + s);
}

static UniFracMetricId unifrac_metric_id(const std::string& s) {
  if (s == "unifrac_unweighted")  return UniFracMetricId::unweighted;
  if (s == "unifrac_weighted")    return UniFracMetricId::weighted;
  if (s == "unifrac_generalized") return UniFracMetricId::generalized;
  if (s == "unifrac_vaw")         return UniFracMetricId::vaw;
  Rcpp::stop("unknown UniFrac metric: " + s);
}

// ---------------------------------------------------------------------------
// UniFrac distance between two samples' branch-mass vectors
// ---------------------------------------------------------------------------

static void accumulate_branch(const std::string& metric,
                              const std::vector<double>& branch_length,
                              int rank, double p1, double p2, double alpha,
                              double& num, double& den) {
  const double len = branch_length[static_cast<size_t>(rank)];
  if (len <= 0.0) return;
  const double psum = p1 + p2;
  if (psum <= 0.0) return;
  const double diff = std::fabs(p1 - p2);
  if (metric == "unifrac_unweighted") {
    den += len;
    if ((p1 > 0.0) != (p2 > 0.0)) num += len;
  } else if (metric == "unifrac_weighted") {
    num += len * diff;
    den += len * psum;
  } else if (metric == "unifrac_generalized") {
    const double weight = std::pow(psum, alpha);
    num += len * weight * diff / psum;
    den += len * weight;
  } else if (metric == "unifrac_vaw") {
    const double eps      = 1e-15;
    const double variance = std::max(psum - diff * diff, eps);
    num += len * diff / std::sqrt(variance);
    den += len * std::sqrt(psum);
  }
}

static double dist_unifrac(const std::string& metric, const PhyloTree& tree,
                           const SparseBranchMass& a,
                           const SparseBranchMass& b, double alpha) {
  double num = 0.0, den = 0.0;
  size_t ia = 0, ib = 0;
  while (ia < a.idx.size() || ib < b.idx.size()) {
    int    rank;
    double p1 = 0.0, p2 = 0.0;
    if (ib >= b.idx.size() ||
        (ia < a.idx.size() && a.idx[ia] < b.idx[ib])) {
      rank = a.idx[ia]; p1 = a.val[ia]; ++ia;
    } else if (ia >= a.idx.size() || b.idx[ib] < a.idx[ia]) {
      rank = b.idx[ib]; p2 = b.val[ib]; ++ib;
    } else {
      rank = a.idx[ia]; p1 = a.val[ia]; p2 = b.val[ib]; ++ia; ++ib;
    }
    accumulate_branch(metric, tree.postorder_length, rank, p1, p2, alpha,
                      num, den);
  }
  return den <= 0.0 ? NA_REAL : num / den;
}

// ---------------------------------------------------------------------------
// Count-based beta metrics
// ---------------------------------------------------------------------------

static double dist_bray(const std::vector<int>&    i1,
                        const std::vector<double>& v1,
                        const std::vector<int>&    i2,
                        const std::vector<double>& v2) {
  double sum1 = 0, sum2 = 0, smin = 0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      smin += std::min(v1[a], v2[b]);
      sum1 += v1[a]; sum2 += v2[b];
      ++a; ++b;
    } else if (i1[a] < i2[b]) {
      sum1 += v1[a++];
    } else {
      sum2 += v2[b++];
    }
  }
  while (a < i1.size()) sum1 += v1[a++];
  while (b < i2.size()) sum2 += v2[b++];
  const double den = sum1 + sum2;
  return den <= 0.0 ? NA_REAL : 1.0 - 2.0 * smin / den;
}

static double dist_euclidean(const std::vector<int>&    i1,
                             const std::vector<double>& v1,
                             const std::vector<int>&    i2,
                             const std::vector<double>& v2) {
  double s = 0.0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      const double d = v1[a] - v2[b]; s += d * d; ++a; ++b;
    } else if (i1[a] < i2[b]) {
      s += v1[a] * v1[a]; ++a;
    } else {
      s += v2[b] * v2[b]; ++b;
    }
  }
  while (a < i1.size()) { s += v1[a] * v1[a]; ++a; }
  while (b < i2.size()) { s += v2[b] * v2[b]; ++b; }
  return std::sqrt(s);
}

// Hellinger: sqrt-normalise then Euclidean
// Takes copies because it mutates
static double dist_hellinger(const std::vector<int>& i1, std::vector<double> v1,
                             const std::vector<int>& i2, std::vector<double> v2) {
  double s1 = 0, s2 = 0;
  for (double x : v1) s1 += x;
  for (double x : v2) s2 += x;
  if (s1 <= 0.0 || s2 <= 0.0) return NA_REAL;
  const double is1 = 1.0 / s1, is2 = 1.0 / s2;
  for (double& x : v1) x = std::sqrt(x * is1);
  for (double& x : v2) x = std::sqrt(x * is2);
  return dist_euclidean(i1, v1, i2, v2);
}

static double dist_simpson(const std::vector<int>&    i1,
                           const std::vector<double>& v1,
                           const std::vector<int>&    i2,
                           const std::vector<double>& v2) {
  double s1 = 0, s2 = 0;
  for (double x : v1) s1 += x;
  for (double x : v2) s2 += x;
  if (s1 <= 0.0 || s2 <= 0.0) return NA_REAL;
  double uij = 0, uji = 0;
  size_t a = 0, b = 0;
  while (a < i1.size() && b < i2.size()) {
    if (i1[a] == i2[b]) {
      uij += v1[a] / s1; uji += v2[b] / s2; ++a; ++b;
    } else if (i1[a] < i2[b]) { ++a; } else { ++b; }
  }
  const double uv     = uij * uji;
  const double u_diff = uij - uv;
  const double v_diff = uji - uv;
  const double eps    = 1e-17;
  return 1.0 - uv / (uv + std::min(u_diff, v_diff) + eps);
}

// Presence-absence helper: converts a count vector to 0/1 in-place
// (copy provided by caller when needed to avoid aliasing with original data)
static void to_pa_inplace(std::vector<double>& v) {
  for (double& x : v)
    if (x > 0.0) x = 1.0;
}

// Dispatch a single count-based beta metric
// v1/v2 passed by value only when the metric requires mutation (PA variants, Hellinger)
// all others receive const refs via the overloads below
static double dispatch_beta(const std::string&         metric,
                            const std::vector<int>&    i1,
                            const std::vector<double>& v1,
                            const std::vector<int>&    i2,
                            const std::vector<double>& v2) {
  if (metric == "bray_curtis") {
    return dist_bray(i1, v1, i2, v2);
  }
  if (metric == "bray_curtis_pa") {
    std::vector<double> c1(v1), c2(v2);
    to_pa_inplace(c1); to_pa_inplace(c2);
    return dist_bray(i1, c1, i2, c2);
  }
  if (metric == "euclidean") {
    return dist_euclidean(i1, v1, i2, v2);
  }
  if (metric == "hellinger") {
    return dist_hellinger(i1, v1, i2, v2);  // copies internally
  }
  if (metric == "simpson") {
    return dist_simpson(i1, v1, i2, v2);
  }
  if (metric == "simpson_pa") {
    std::vector<double> c1(v1), c2(v2);
    to_pa_inplace(c1); to_pa_inplace(c2);
    return dist_simpson(i1, c1, i2, c2);
  }
  return NA_REAL;
}

static bool is_unifrac(const std::string& m) {
  return m == "unifrac_unweighted" || m == "unifrac_weighted" ||
         m == "unifrac_generalized" || m == "unifrac_vaw";
}

}  // namespace

// ---------------------------------------------------------------------------
// Unified exported function
// ---------------------------------------------------------------------------
//
// Rarefy each sample column once per replicate and compute all requested
// count-based and/or UniFrac dissimilarities in a single pair loop
//
// beta_metrics    : character vector of count-based metrics (may be empty)
// unifrac_metrics : character vector of UniFrac metrics   (may be empty)
// phylo           : R phylo List; R_NilValue when unifrac_metrics is empty
// row_to_tip_obj  : integer map matrix-row -> 0-based tip index; R_NilValue
//                   when unifrac_metrics is empty
// unifrac_alpha   : alpha parameter for generalised UniFrac (ignored otherwise)
//
// Returns a named Rcpp::List of n x n NumericMatrix, one per requested metric,
// in the order [ beta_metrics..., unifrac_metrics... ]
//
// [[Rcpp::export]]
Rcpp::List rarefy_beta_cpp(Rcpp::S4 mat, int depth, int n_iter,
                               Rcpp::CharacterVector beta_metrics,
                               Rcpp::CharacterVector unifrac_metrics,
                               Rcpp::RObject phylo,
                               Rcpp::RObject row_to_tip_obj,
                               double unifrac_alpha,
                               int n_threads, double seed, int kernel) {
  // Validate inputs
  if (depth <= 0) Rcpp::stop("depth must be positive");
  if (n_iter < 1)  Rcpp::stop("n_iter must be >= 1");

  const int nb = static_cast<int>(beta_metrics.size());
  const int nu = static_cast<int>(unifrac_metrics.size());
  if (nb + nu == 0) Rcpp::stop("at least one metric must be requested");

  for (int m = 0; m < nb; ++m) {
    const std::string& s = Rcpp::as<std::string>(beta_metrics[m]);
    if (s != "bray_curtis" && s != "bray_curtis_pa" && s != "euclidean" &&
        s != "hellinger"   && s != "simpson"        && s != "simpson_pa") {
      Rcpp::stop("unknown beta metric: " + s);
    }
  }
  for (int m = 0; m < nu; ++m) {
    if (!is_unifrac(Rcpp::as<std::string>(unifrac_metrics[m]))) {
      Rcpp::stop("unknown UniFrac metric: " +
                 Rcpp::as<std::string>(unifrac_metrics[m]));
    }
  }
  if (nu > 0 && (!std::isfinite(unifrac_alpha) ||
                  unifrac_alpha < 0.0 || unifrac_alpha > 1.0)) {
    // Only enforce range when generalized UniFrac is actually requested
    bool need_alpha = false;
    for (int m = 0; m < nu; ++m)
      if (Rcpp::as<std::string>(unifrac_metrics[m]) == "unifrac_generalized")
        need_alpha = true;
    if (need_alpha)
      Rcpp::stop("unifrac_alpha must be in [0,1] for generalized UniFrac");
  }

  RcppSparse::Matrix A(mat);
  const int n = static_cast<int>(A.cols());
  if (n < 2) Rcpp::stop("need at least two samples for beta diversity");

  // Parse tree when UniFrac is requested
  PhyloTree            tree;
  Rcpp::IntegerVector  row_to_tip;
  if (nu > 0) {
    if (phylo.isNULL())        Rcpp::stop("phylo required for UniFrac metrics");
    if (row_to_tip_obj.isNULL()) Rcpp::stop("row_to_tip required for UniFrac metrics");
    tree       = parse_phylo(Rcpp::as<Rcpp::List>(phylo));
    row_to_tip = Rcpp::as<Rcpp::IntegerVector>(row_to_tip_obj);
    if (tree.ntips != static_cast<int>(A.rows()))
      Rcpp::stop("phy_tree tip count must match the number of matrix rows after pruning");
    if (row_to_tip.size() != static_cast<int>(A.rows()))
      Rcpp::stop("row_to_tip must have one entry per matrix row");
  }

  // Pre-compute column totals (needed for hyper kernel)
  std::vector<int64_t> col_totals(static_cast<size_t>(n));
  for (int col = 0; col < n; ++col) {
    col_totals[static_cast<size_t>(col)] = col_sum_rounded(A, col);
    if (depth > col_totals[static_cast<size_t>(col)])
      Rcpp::stop("depth exceeds column sum for at least one sample");
  }

  const size_t npairs =
      static_cast<size_t>(n) * (static_cast<size_t>(n) - 1U) / 2U;

  // Accumulators: [metric_index][pair_index]
  std::vector<std::vector<OnlineStats>> acc_beta(
      static_cast<size_t>(nb),
      std::vector<OnlineStats>(npairs));
  std::vector<std::vector<OnlineStats>> acc_unifrac(
      static_cast<size_t>(nu),
      std::vector<OnlineStats>(npairs));

#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif

  for (int rep = 0; rep < n_iter; ++rep) {
    // Rarefy all columns exactly once for this replicate
    std::vector<std::vector<int>>    idc(static_cast<size_t>(n));
    std::vector<std::vector<double>> vvc(static_cast<size_t>(n));
    for (int col = 0; col < n; ++col) {
      std::mt19937_64 rng;
      seed_rep_rng(rng, seed, col, 424242, rep);
      std::vector<int>    pool;
      if (kernel == 0) {
        rarefy_col_hyper(A, col, depth,
                         col_totals[static_cast<size_t>(col)],
                         idc[static_cast<size_t>(col)],
                         vvc[static_cast<size_t>(col)], rng);
      } else {
        rarefy_col_perm(A, col, depth,
                        idc[static_cast<size_t>(col)],
                        vvc[static_cast<size_t>(col)], pool, rng);
      }
    }

    // Convert to branch masses for UniFrac (one pass per sample)
    std::vector<SparseBranchMass> bm;
    if (nu > 0) {
      bm.resize(static_cast<size_t>(n));
      for (int col = 0; col < n; ++col) {
        bm[static_cast<size_t>(col)] = sample_to_branch_masses(
            tree, row_to_tip,
            idc[static_cast<size_t>(col)],
            vvc[static_cast<size_t>(col)], depth);
      }
    }

    // Parallel pair loop: compute all metrics for each pair
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t lin = 0; lin < npairs; ++lin) {
      int i = 0, j = 0;
      linear_to_pair(n, lin, i, j);
      const size_t ci = static_cast<size_t>(i);
      const size_t cj = static_cast<size_t>(j);

      for (int m = 0; m < nb; ++m) {
        const double d = dispatch_beta(
            Rcpp::as<std::string>(beta_metrics[m]),
            idc[ci], vvc[ci], idc[cj], vvc[cj]);
        acc_beta[static_cast<size_t>(m)][lin].update(d);
      }
      for (int m = 0; m < nu; ++m) {
        const double d = dist_unifrac(
            Rcpp::as<std::string>(unifrac_metrics[m]),
            tree, bm[ci], bm[cj], unifrac_alpha);
        acc_unifrac[static_cast<size_t>(m)][lin].update(d);
      }
    }
  }

  // Build output: one n x n matrix per metric
  auto make_dist_matrix = [&](const std::vector<OnlineStats>& acc) {
    Rcpp::NumericMatrix D(n, n);
    std::fill(D.begin(), D.end(), 0.0);
    for (int i = 0; i < n - 1; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const OnlineStats& o = acc[tri_index(n, i, j)];
        const double v = o.n > 0 ? o.mean : NA_REAL;
        D(i, j) = v;
        D(j, i) = v;
      }
    }
    return D;
  };

  Rcpp::List out(nb + nu);
  Rcpp::CharacterVector out_names(nb + nu);
  for (int m = 0; m < nb; ++m) {
    out[m]       = make_dist_matrix(acc_beta[static_cast<size_t>(m)]);
    out_names[m] = beta_metrics[m];
  }
  for (int m = 0; m < nu; ++m) {
    out[nb + m]       = make_dist_matrix(acc_unifrac[static_cast<size_t>(m)]);
    out_names[nb + m] = unifrac_metrics[m];
  }
  out.attr("names") = out_names;
  return out;
}

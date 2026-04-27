#pragma once

#include <Rcpp.h>

#include <cmath>
#include <vector>

struct PhyloTree {
  int ntips = 0;
  int nnode = 0;
  int nnodes = 0;
  int nedges = 0;
  int root = -1;
  std::vector<int> parent;
  std::vector<int> child;
  std::vector<double> length;
  std::vector<int> postorder_edges;
  std::vector<double> postorder_length;
};

struct PhyloStackFrame {
  int node;
  int edge_from_parent;
  bool exiting;
};

inline PhyloTree parse_phylo(Rcpp::List phylo) {
  Rcpp::IntegerMatrix edge = phylo["edge"];
  Rcpp::NumericVector edge_length = phylo["edge.length"];
  Rcpp::CharacterVector tip_label = phylo["tip.label"];
  const int nnode = Rcpp::as<int>(phylo["Nnode"]);
  const int ntips = static_cast<int>(tip_label.size());
  const int nedges = edge.nrow();
  const int nnodes = ntips + nnode;
  if (edge.ncol() != 2) {
    Rcpp::stop("phy_tree$edge must have two columns");
  }
  if (ntips < 1 || nnode < 1 || nnodes < 2) {
    Rcpp::stop("phy_tree must contain at least one tip and one internal node");
  }
  if (edge_length.size() != nedges) {
    Rcpp::stop("phy_tree$edge.length must have one value per edge");
  }

  PhyloTree tree;
  tree.ntips = ntips;
  tree.nnode = nnode;
  tree.nnodes = nnodes;
  tree.nedges = nedges;
  tree.parent.resize(static_cast<size_t>(nedges));
  tree.child.resize(static_cast<size_t>(nedges));
  tree.length.resize(static_cast<size_t>(nedges));

  std::vector<int> child_counts(static_cast<size_t>(nnodes), 0);
  std::vector<unsigned char> has_parent(static_cast<size_t>(nnodes), 0);
  std::vector<unsigned char> has_child(static_cast<size_t>(nnodes), 0);
  for (int e = 0; e < nedges; ++e) {
    const int parent = edge(e, 0) - 1;
    const int child = edge(e, 1) - 1;
    const double len = edge_length[e];
    if (parent < 0 || parent >= nnodes || child < 0 || child >= nnodes) {
      Rcpp::stop("phy_tree$edge contains node indices outside the expected range");
    }
    if (!std::isfinite(len) || len < 0.0) {
      Rcpp::stop("phy_tree$edge.length must contain finite non-negative values");
    }
    tree.parent[static_cast<size_t>(e)] = parent;
    tree.child[static_cast<size_t>(e)] = child;
    tree.length[static_cast<size_t>(e)] = len;
    ++child_counts[static_cast<size_t>(parent)];
    has_parent[static_cast<size_t>(child)] = 1;
    has_child[static_cast<size_t>(parent)] = 1;
  }

  for (int node = ntips; node < nnodes; ++node) {
    if (!has_parent[static_cast<size_t>(node)] && has_child[static_cast<size_t>(node)]) {
      if (tree.root >= 0) {
        Rcpp::stop("phy_tree must have exactly one root");
      }
      tree.root = node;
    }
  }
  if (tree.root < 0) {
    Rcpp::stop("failed to identify phy_tree root");
  }

  std::vector<int> offsets(static_cast<size_t>(nnodes + 1), 0);
  for (int node = 0; node < nnodes; ++node) {
    offsets[static_cast<size_t>(node + 1)] =
        offsets[static_cast<size_t>(node)] + child_counts[static_cast<size_t>(node)];
  }
  std::vector<int> cursor = offsets;
  std::vector<int> child_edges(static_cast<size_t>(nedges));
  for (int e = 0; e < nedges; ++e) {
    const int parent = tree.parent[static_cast<size_t>(e)];
    child_edges[static_cast<size_t>(cursor[static_cast<size_t>(parent)]++)] = e;
  }

  tree.postorder_edges.reserve(static_cast<size_t>(nedges));
  std::vector<PhyloStackFrame> stack;
  stack.reserve(static_cast<size_t>(nnodes));
  stack.push_back({tree.root, -1, false});
  while (!stack.empty()) {
    const PhyloStackFrame frame = stack.back();
    stack.pop_back();
    if (frame.exiting) {
      if (frame.edge_from_parent >= 0) {
        tree.postorder_edges.push_back(frame.edge_from_parent);
      }
      continue;
    }
    stack.push_back({frame.node, frame.edge_from_parent, true});
    const int begin = offsets[static_cast<size_t>(frame.node)];
    const int end = offsets[static_cast<size_t>(frame.node + 1)];
    for (int k = end - 1; k >= begin; --k) {
      const int edge_id = child_edges[static_cast<size_t>(k)];
      stack.push_back({tree.child[static_cast<size_t>(edge_id)], edge_id, false});
    }
  }
  if (static_cast<int>(tree.postorder_edges.size()) != nedges) {
    Rcpp::stop("phy_tree appears to be disconnected or cyclic");
  }
  tree.postorder_length.resize(static_cast<size_t>(nedges));
  for (int rank = 0; rank < nedges; ++rank) {
    const int edge_id = tree.postorder_edges[static_cast<size_t>(rank)];
    tree.postorder_length[static_cast<size_t>(rank)] = tree.length[static_cast<size_t>(edge_id)];
  }
  return tree;
}

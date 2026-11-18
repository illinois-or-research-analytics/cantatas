#ifndef STRUCTS_H
#define STRUCTS_H

#include <span>

// Per-node preference components for the current year. The values are not
// normalized globally: each candidate set is renormalized when it is scored,
// so a global normalization would cancel out.
struct alignas(32) NodeScoreComponents {
  double pa = 0.0;  // preferential attachment (in-degree)
  double fit = 0.0; // fitness
  double na = 0.0;  // number of authors
  double ar = 0.0;  // author reputation (h-index)
};

struct NodeMetrics {
  std::span<const NodeScoreComponents> components;
};

struct AgentWeights {
  double pa_weight;
  double fit_weight;
  double num_authors_weight;
  double author_reputation_weight;
};

#endif

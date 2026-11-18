#ifndef VECTORIZED_SCORING_H
#define VECTORIZED_SCORING_H

#include "structs.h"
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace ladybug {

struct ScoringScratchpad {
  std::vector<double> raw_pa;
  std::vector<double> raw_fit;
  std::vector<double> raw_na;
  std::vector<double> raw_ar;
  std::vector<double> composite_weights;

  void EnsureCapacity(size_t capacity) {
    if (composite_weights.size() < capacity) {
      size_t new_cap = std::max(capacity, composite_weights.size() * 2 + 128);
      raw_pa.resize(new_cap);
      raw_fit.resize(new_cap);
      raw_na.resize(new_cap);
      raw_ar.resize(new_cap);
      composite_weights.resize(new_cap);
    }
  }
};

class VectorizedScoring {
public:
  // Weight of each candidate: sum over the four components of
  // agent_weight * component / (sum of that component over the candidates).
  static void ComputeCandidateWeights(std::span<const int> candidate_nodes,
                                      const NodeMetrics &metrics,
                                      const AgentWeights &weights,
                                      ScoringScratchpad &scratch,
                                      std::span<double> &out_weights);
};

} // namespace ladybug

#endif // VECTORIZED_SCORING_H

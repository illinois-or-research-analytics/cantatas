#ifndef VECTORIZED_SCORING_H
#define VECTORIZED_SCORING_H

#include "structs.h"
#include <cstddef>
#include <span>
#include <vector>

namespace ladybug {

/*
 * Scratchpad buffer for vectorized preference score calculations.
 */
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

/*
 * VectorizedScoring: AVX2 / OpenMP SIMD accelerated multi-component preference scoring.
 * Evaluates Preferential Attachment (PA), Temporal Fitness, Team Size, and Author Reputation
 * directly against contiguous columnar buffers with zero memory allocation.
 */
class VectorizedScoring {
public:
  // Computes normalized candidate weights in O(K) time using SIMD registers
  static void ComputeCandidateWeights(
      const std::vector<int> &candidate_nodes,
      const std::vector<int> &continuous_node_mapping,
      const NodeMetrics &metrics,
      const AgentWeights &weights,
      ScoringScratchpad &scratch,
      std::span<double> &out_weights);

  // SIMD batch score computation directly on contiguous node slices
  static void BatchComputeScores(
      std::span<const double> pa_span,
      std::span<const double> fit_span,
      std::span<const double> na_span,
      std::span<const double> ar_span,
      const AgentWeights &weights,
      std::span<double> out_scores);
};

} // namespace ladybug

#endif // VECTORIZED_SCORING_H

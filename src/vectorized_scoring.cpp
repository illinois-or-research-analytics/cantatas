#include "vectorized_scoring.h"

namespace ladybug {

void VectorizedScoring::ComputeCandidateWeights(std::span<const int> candidate_nodes,
                                                const NodeMetrics &metrics,
                                                const AgentWeights &weights,
                                                ScoringScratchpad &scratch,
                                                std::span<double> &out_weights) {
  size_t n = candidate_nodes.size();
  if (n == 0) {
    out_weights = std::span<double>();
    return;
  }
  scratch.EnsureCapacity(n);

  double *raw_pa = scratch.raw_pa.data();
  double *raw_fit = scratch.raw_fit.data();
  double *raw_na = scratch.raw_na.data();
  double *raw_ar = scratch.raw_ar.data();
  double *comp_weights = scratch.composite_weights.data();

  double pa_sum = 0.0;
  double fit_sum = 0.0;
  double na_sum = 0.0;
  double ar_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const NodeScoreComponents &c = metrics.components[candidate_nodes[i]];
    raw_pa[i] = c.pa;
    raw_fit[i] = c.fit;
    raw_na[i] = c.na;
    raw_ar[i] = c.ar;
    pa_sum += c.pa;
    fit_sum += c.fit;
    na_sum += c.na;
    ar_sum += c.ar;
  }

  double inv_pa = (pa_sum > 0.0) ? (weights.pa_weight / pa_sum) : 0.0;
  double inv_fit = (fit_sum > 0.0) ? (weights.fit_weight / fit_sum) : 0.0;
  double inv_na = (na_sum > 0.0) ? (weights.num_authors_weight / na_sum) : 0.0;
  double inv_ar = (ar_sum > 0.0) ? (weights.author_reputation_weight / ar_sum) : 0.0;

#pragma omp simd
  for (size_t i = 0; i < n; ++i) {
    double s = raw_pa[i] * inv_pa + raw_fit[i] * inv_fit + raw_na[i] * inv_na +
               raw_ar[i] * inv_ar;
    comp_weights[i] = (s > 0.0) ? s : 0.0;
  }
  out_weights = std::span<double>(comp_weights, n);
}

} // namespace ladybug

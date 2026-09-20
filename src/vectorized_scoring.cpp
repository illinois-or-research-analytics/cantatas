#include "vectorized_scoring.h"
#include <algorithm>
#include <cmath>

namespace ladybug {

void VectorizedScoring::ComputeCandidateWeights(
    const std::vector<int> &candidate_nodes,
    const std::vector<int> &continuous_node_mapping,
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

  double pa_sum = 0.0;
  double fit_sum = 0.0;
  double na_sum = 0.0;
  double ar_sum = 0.0;

  double *raw_pa = scratch.raw_pa.data();
  double *raw_fit = scratch.raw_fit.data();
  double *raw_na = scratch.raw_na.data();
  double *raw_ar = scratch.raw_ar.data();
  double *comp_weights = scratch.composite_weights.data();

  // 1. Gather component metrics from NodeMetrics aligned components
  for (size_t i = 0; i < n; ++i) {
    int cand = candidate_nodes[i];
    int cont_id = (cand >= 0 && static_cast<size_t>(cand) < continuous_node_mapping.size())
                      ? continuous_node_mapping[cand]
                      : cand;
    if (cont_id < 0) cont_id = 0;

    const auto &c = metrics.components[cont_id];

    raw_pa[i] = c.pa;
    raw_fit[i] = c.fit;
    raw_na[i] = c.na;
    raw_ar[i] = c.ar;

    pa_sum += c.pa;
    fit_sum += c.fit;
    na_sum += c.na;
    ar_sum += c.ar;
  }

  // 2. Compute normalizer inverses
  double inv_pa = (pa_sum > 0.0) ? (weights.pa_weight / pa_sum) : 0.0;
  double inv_fit = (fit_sum > 0.0) ? (weights.fit_weight / fit_sum) : 0.0;
  double inv_na = (na_sum > 0.0) ? (weights.num_authors_weight / na_sum) : 0.0;
  double inv_ar = (ar_sum > 0.0) ? (weights.author_reputation_weight / ar_sum) : 0.0;

  // 3. SIMD vectorized Fused-Multiply-Add across candidates
#pragma omp simd
  for (size_t i = 0; i < n; ++i) {
    double s = raw_pa[i] * inv_pa +
               raw_fit[i] * inv_fit +
               raw_na[i] * inv_na +
               raw_ar[i] * inv_ar;
    comp_weights[i] = (s > 0.0) ? s : 0.0;
  }

  out_weights = std::span<double>(comp_weights, n);
}

void VectorizedScoring::BatchComputeScores(
    std::span<const double> pa_span,
    std::span<const double> fit_span,
    std::span<const double> na_span,
    std::span<const double> ar_span,
    const AgentWeights &weights,
    std::span<double> out_scores) {
  size_t n = std::min({pa_span.size(), fit_span.size(), na_span.size(), ar_span.size(), out_scores.size()});

  double w_pa = weights.pa_weight;
  double w_fit = weights.fit_weight;
  double w_na = weights.num_authors_weight;
  double w_ar = weights.author_reputation_weight;

#pragma omp simd
  for (size_t i = 0; i < n; ++i) {
    out_scores[i] = pa_span[i] * w_pa +
                    fit_span[i] * w_fit +
                    na_span[i] * w_na +
                    ar_span[i] * w_ar;
  }
}

} // namespace ladybug

#ifndef METRICS_ENGINE_H
#define METRICS_ENGINE_H

#include "graph.h"
#include <span>
#include <unordered_map>
#include <vector>

class MetricsEngine {
public:
  /*
  Input: std::span<double> alpha_span, bool use_alpha, double alpha, double
  minimum_alpha Output: void (modifies alpha_span in-place) Description:
  Populates the provided span with alpha values representing the intrinsic
               attractiveness of nodes, optionally uniformly distributed or
  static.
  */
  static void PopulateAlphaSpan(std::span<double> alpha_span, bool use_alpha,
                                double alpha, double minimum_alpha);
  /*
  Input: std::span<int> fitness_lag_duration_span, std::span<int>
  fitness_peak_value_span, std::span<int> fitness_peak_duration_span, int
  lag_min, int lag_max, int peak_val_min, int peak_val_max, int peak_dur_min,
  int peak_dur_max, int fitness_alpha Output: void (modifies the three fitness
  spans in-place) Description: Initializes fitness characteristics (lag
  duration, peak value, and peak duration) for nodes by sampling from uniform
  distributions within the provided min/max bounds.
  */
  static void PopulateFitnessSpans(std::span<int> fitness_lag_duration_span,
                                   std::span<int> fitness_peak_value_span,
                                   std::span<int> fitness_peak_duration_span,
                                   int lag_min, int lag_max, int peak_val_min,
                                   int peak_val_max, int peak_dur_min,
                                   int peak_dur_max, int fitness_alpha);
  /*
  Input: Graph *graph, std::span<int> num_authors_span
  Output: void (modifies num_authors_span in-place)
  Description: Populates the number of authors for each node by fetching the
  next available value from the graph's pre-loaded author distribution.
  */
  static void PopulateNumAuthorsSpan(Graph *graph,
                                     std::span<int> num_authors_span);
  /*
  Input: std::span<int> out_degree_span, const std::vector<int>&
  out_degree_bag_vec Output: void (modifies out_degree_span in-place)
  Description: Populates the out-degree of nodes by uniformly sampling from a
  pre-defined bag of historical out-degree values.
  */
  static void PopulateOutDegreeSpan(std::span<int> out_degree_span,
                                    const std::vector<int> &out_degree_bag_vec);
  /*
  Input: Spans for pa_weight, fit_weight, num_authors_weight,
  author_reputation_weight, and their corresponding base weight bounds/values.
  Output: void (modifies all four weight spans in-place)
  Description: Calculates normalized weight distributions for preferential
  attachment, fitness, number of authors, and author reputation. Randomizes if
  bounds are not strictly provided.
  */
  static void PopulateWeightSpans(
      std::span<double> pa_weight_span, std::span<double> fit_weight_span,
      std::span<double> num_authors_weight_span,
      std::span<double> author_reputation_weight_span,
      double preferential_weight, double fitness_weight,
      double num_authors_weight, double author_reputation_weight);

  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, std::span<int> in_degree_span Output: void (modifies
  in_degree_span in-place) Description: Gathers the live in-degree (number of
  citations) from the graph for each node and maps them sequentially into the
  span for rapid contiguous access.
  */
  static void
  FillInDegreeSpan(Graph *graph,
                   const std::vector<int> &reverse_continuous_node_mapping,
                   std::span<int> in_degree_span, int current_graph_size);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, int current_year, std::span<int> fitness_span, int
  fitness_decay_alpha Output: void (modifies fitness_span in-place) Description:
  Evaluates the time-decaying fitness of each node based on the current
  simulation year, applying lag and peak durations to determine the node's
  current temporal relevance.
  */
  static void
  FillFitnessSpan(Graph *graph,
                  const std::vector<int> &reverse_continuous_node_mapping,
                  int current_year, std::span<int> fitness_span,
                  int fitness_decay_alpha, int current_graph_size);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, std::span<int> author_reputation_span Output: void
  (modifies author_reputation_span in-place) Description: Triggers author
  reputation computation on the graph and populates the continuous span with the
  calculated reputation values for each corresponding node.
  */
  static void FillAuthorReputationSpan(
      Graph *graph, const std::vector<int> &reverse_continuous_node_mapping,
      std::span<int> author_reputation_span, int current_graph_size);

  /*
  Input: std::unordered_map<int, double> &cached_results, std::span<int>
  src_span, std::span<double> dst_span, double peak_constant, double
  delay_constant Output: void (modifies dst_span in-place and conditionally
  updates cached_results) Description: Applies a non-linear hyperbolic tangent
  (tanh) scoring transformation to raw attributes, caching results to avoid
  redundant math computations for frequent values.
  */
  static void
  CalculateTanhScores(std::unordered_map<int, double> &cached_results,
                      std::span<int> src_span, std::span<double> dst_span,
                      double peak_constant, double delay_constant);
  /*
  Input: std::unordered_map<int, double> &cached_results, std::span<int>
  src_span, std::span<double> dst_span, double gamma Output: void (modifies
  dst_span in-place and conditionally updates cached_results) Description:
  Applies an exponential power scaling factor (gamma) to raw attributes, caching
               the output distribution for rapid probability generation in
  selection steps.
  */
  static void
  CalculateExpScores(std::unordered_map<int, double> &cached_results,
                     std::span<int> src_span, std::span<double> dst_span,
                     double gamma);
};

#endif

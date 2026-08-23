#ifndef STRUCTS_H
#define STRUCTS_H
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

struct alignas(32) NodeScoreComponents {
  double pa = 0.0;
  double fit = 0.0;
  double na = 0.0;
  double ar = 0.0;
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

struct ClonalCartelAgent {
  std::optional<int> num_authors;
  std::optional<double> pa_weight;
  std::optional<double> fit_weight;
  std::optional<double> num_authors_weight;
  std::optional<double> author_reputation_weight;
  std::optional<int> out_degree;
  std::optional<double> alpha;
  std::optional<int> fitness_lag_duration;
  std::optional<int> fitness_peak_value;
  std::optional<int> fitness_peak_duration;
};

#endif

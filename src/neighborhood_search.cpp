#include "neighborhood_search.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>

namespace {
class ThreadLocalVisitedTracker {
public:
  inline bool Contains(int node) const {
    if (node < 0 || (size_t)node >= (visited_words.size() << 6)) {
      return false;
    }
    return (visited_words[(size_t)node >> 6] & (1ULL << ((size_t)node & 63))) != 0;
  }

  inline void Insert(int node) {
    if (node < 0) {
      return;
    }
    size_t word_idx = (size_t)node >> 6;
    if (word_idx >= visited_words.size()) {
      visited_words.resize(word_idx + 1024, 0);
    }
    uint64_t mask = 1ULL << ((size_t)node & 63);
    if ((visited_words[word_idx] & mask) == 0) {
      visited_words[word_idx] |= mask;
      visited_list.push_back(node);
    }
  }

  inline void Clear() {
    for (int node : visited_list) {
      visited_words[(size_t)node >> 6] = 0;
    }
    visited_list.clear();
  }

private:
  std::vector<uint64_t> visited_words;
  std::vector<int> visited_list;
};
} // namespace

NeighborhoodSearch::NeighborhoodSearch(std::string recency_bins_str,
                                       bool use_alpha,
                                       int neighborhood_sample) {
  this->recency_bins_str = recency_bins_str;
  this->use_alpha = use_alpha;
  this->neighborhood_sample = neighborhood_sample;
  this->InitializeBinBoundaries();
}

void NeighborhoodSearch::InitializeBinBoundaries() {
  std::string bin_boundary_string = this->recency_bins_str;
  std::stringstream ss(bin_boundary_string);
  std::string current_value;
  int element_no = 0;
  while (std::getline(ss, current_value, ',')) {
    int current_bin_value = std::stoi(current_value);
    if (current_bin_value != -1) {
      this->bin_boundaries.push_back(current_bin_value);
    }
    element_no++;
  }
  this->num_bins = element_no;
  this->BuildBinLUT();
}

void NeighborhoodSearch::BuildBinLUT() {
  this->recency_bin_lut.resize(1024);
  for (int age = 0; age < 1024; ++age) {
    int bin = static_cast<int>(this->bin_boundaries.size() - 1);
    for (size_t i = 0; i < this->bin_boundaries.size() - 1; ++i) {
      if (this->bin_boundaries[i] <= age && age < this->bin_boundaries[i + 1]) {
        bin = static_cast<int>(i);
        break;
      }
    }
    this->recency_bin_lut[age] = bin;
  }
}

std::vector<int> NeighborhoodSearch::GetNumCitationsPerNeighborhood(
    double alpha, int total_num_citations_neighborhood,
    const std::vector<std::vector<int>> &n_hop_map) {
  std::vector<int> num_citations_per_neighborhood(n_hop_map.size(), 0);
  if (n_hop_map.size() >= 3) {
    int total_num_citations_neighborhood_clamped =
        std::min((int)(n_hop_map[1].size() + n_hop_map[2].size()),
                 total_num_citations_neighborhood);
    num_citations_per_neighborhood[1] =
        std::min((int)(total_num_citations_neighborhood_clamped * alpha),
                 (int)n_hop_map[1].size());
    num_citations_per_neighborhood[2] =
        std::min(total_num_citations_neighborhood_clamped -
                     num_citations_per_neighborhood[1],
                 (int)n_hop_map[2].size());
  } else if (n_hop_map.size() >= 2) {
    num_citations_per_neighborhood[1] = std::min(
        total_num_citations_neighborhood, (int)n_hop_map[1].size());
  }
  return num_citations_per_neighborhood;
}

std::vector<int> NeighborhoodSearch::BinOutdegrees(
    const std::vector<std::vector<int>> &binned_neighborhood,
    int total_outdegree,
    const std::vector<double> &binned_recency_probabilities) {
  std::vector<int> target_outdegree_per_bin_map(this->num_bins, 0);
  int remaining_outdegree = total_outdegree;
  for (int bin_index = 0; bin_index < this->num_bins; bin_index++) {
    if (remaining_outdegree == 0) {
      break;
    }
    double bin_probability = ((size_t)bin_index < binned_recency_probabilities.size())
                                 ? binned_recency_probabilities[bin_index]
                                 : 0.0;
    int current_bin_outdegree = std::round(total_outdegree * bin_probability);
    current_bin_outdegree =
        std::min(current_bin_outdegree, remaining_outdegree);
    target_outdegree_per_bin_map[bin_index] = current_bin_outdegree;
    remaining_outdegree -= current_bin_outdegree;
  }
  std::uniform_int_distribution<int> year_distribution{1, 5};
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  while (remaining_outdegree > 0) {
    int chosen_year = year_distribution(generator);
    int current_bin_index = this->GetBinIndex(chosen_year);
    if ((size_t)current_bin_index < target_outdegree_per_bin_map.size()) {
      target_outdegree_per_bin_map[current_bin_index] += 1;
    }
    remaining_outdegree--;
  }
  for (int bin_index = 0; bin_index < this->num_bins; bin_index++) {
    int binned_size = ((size_t)bin_index < binned_neighborhood.size())
                          ? static_cast<int>(binned_neighborhood[bin_index].size())
                          : 0;
    int current_uncited_num_nodes =
        target_outdegree_per_bin_map[bin_index] - binned_size;
    if (current_uncited_num_nodes > 0) {
      for (int sweep_index = bin_index - 1;
           sweep_index >= 0 && current_uncited_num_nodes > 0; sweep_index--) {
        int sweep_binned_size = static_cast<int>(binned_neighborhood[sweep_index].size());
        if (target_outdegree_per_bin_map[sweep_index] < sweep_binned_size) {
          int current_citable = std::min(
              current_uncited_num_nodes,
              sweep_binned_size - target_outdegree_per_bin_map[sweep_index]);
          current_uncited_num_nodes -= current_citable;
          target_outdegree_per_bin_map[sweep_index] += current_citable;
          target_outdegree_per_bin_map[bin_index] -= current_citable;
        }
      }
      for (int sweep_index = bin_index + 1;
           sweep_index < this->num_bins && current_uncited_num_nodes > 0;
           sweep_index++) {
        int sweep_binned_size = static_cast<int>(binned_neighborhood[sweep_index].size());
        if (target_outdegree_per_bin_map[sweep_index] < sweep_binned_size) {
          int current_citable = std::min(
              current_uncited_num_nodes,
              sweep_binned_size - target_outdegree_per_bin_map[sweep_index]);
          current_uncited_num_nodes -= current_citable;
          target_outdegree_per_bin_map[sweep_index] += current_citable;
          target_outdegree_per_bin_map[bin_index] -= current_citable;
        }
      }
    }
  }
  return target_outdegree_per_bin_map;
}

int NeighborhoodSearch::GetBinIndex(int year_diff) {
  if (year_diff >= 0 && (size_t)year_diff < this->recency_bin_lut.size()) {
    return this->recency_bin_lut[year_diff];
  }
  if (year_diff < 0) {
    return 0;
  }
  for (size_t i = 0; i < this->bin_boundaries.size() - 1; i++) {
    if (this->bin_boundaries[i] <= year_diff &&
        year_diff < this->bin_boundaries[i + 1]) {
      return static_cast<int>(i);
    }
  }
  return static_cast<int>(this->bin_boundaries.size() - 1);
}

std::vector<std::vector<int>>
NeighborhoodSearch::BinNeighborhood(Graph *graph, int current_year,
                                    const std::vector<int> &n_hop_list) {
  std::vector<std::vector<int>> binned_neighborhood(this->num_bins);
  for (size_t i = 0; i < n_hop_list.size(); i++) {
    int current_node = n_hop_list[i];
    int current_bin = GetBinIndex(graph, current_node, current_year);
    if ((size_t)current_bin < binned_neighborhood.size()) {
      binned_neighborhood[current_bin].push_back(current_node);
    }
  }
  return binned_neighborhood;
}

std::vector<std::vector<int>>
NeighborhoodSearch::GetNeighborhoodMap(Graph *graph, int current_year,
                                       const std::vector<int> &generator_nodes,
                                       int num_hops) {
  if (this->use_alpha) {
    // create distance 1 and distance 2 neighborhoods
    return this->GetOneAndTwoDistanceNeighborhoods(graph, current_year,
                                                   generator_nodes, num_hops);
  } else {
    return this->GetNHopNeighborhood(graph, current_year, generator_nodes,
                                     num_hops);
  }
}

std::vector<std::vector<int>>
NeighborhoodSearch::GetOneAndTwoDistanceNeighborhoods(
    Graph *graph, int current_year, const std::vector<int> &generator_nodes,
    int num_hops) {
  std::vector<std::vector<int>> n_hop_map(std::max(3, num_hops + 1));
  thread_local ThreadLocalVisitedTracker visited;
  visited.Clear();

  const auto &forward_adj = graph->GetForwardAdjList();
  const auto &backward_adj = graph->GetBackwardAdjList();

  if (this->neighborhood_sample == -1) {
    thread_local std::vector<std::pair<int, int>> to_visit;
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      int generator_node = generator_nodes[i];
      to_visit.clear();
      size_t head = 0;
      to_visit.push_back({generator_node, 0});
      visited.Insert(generator_node);
      while (head < to_visit.size()) {
        std::pair<int, int> current_pair = to_visit[head++];
        int current_node = current_pair.first;
        int current_distance = current_pair.second;
        if (current_distance > 0) {
          n_hop_map[current_distance].push_back(current_node);
        }
        if (current_distance < num_hops) {
          if (graph->GetOutDegree(current_node) > 0 &&
              (size_t)current_node < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[current_node]) {
              if (!visited.Contains(outgoing_neighbor)) {
                visited.Insert(outgoing_neighbor);
                to_visit.push_back({outgoing_neighbor, current_distance + 1});
              }
            }
          }
          if (graph->GetInDegree(current_node) > 0 &&
              (size_t)current_node < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[current_node]) {
              if (!visited.Contains(incoming_neighbor)) {
                visited.Insert(incoming_neighbor);
                to_visit.push_back({incoming_neighbor, current_distance + 1});
              }
            }
          }
        }
      }
    }
  } else {
    // NOTE: only supports randomly sampling from up to 2-hop
    n_hop_map[1].reserve(this->neighborhood_sample);
    n_hop_map[2].reserve(this->neighborhood_sample);
    size_t max_neighborhood_size = this->neighborhood_sample;
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      // get the 1-hop first
      int generator_node = generator_nodes[i];
      visited.Insert(generator_node);
      std::vector<int> current_one_hop_neighborhood;
      if (graph->GetOutDegree(generator_node) > 0 &&
          (size_t)generator_node < forward_adj.size()) {
        for (int outgoing_neighbor : forward_adj[generator_node]) {
          if (!visited.Contains(outgoing_neighbor)) {
            current_one_hop_neighborhood.push_back(outgoing_neighbor);
            visited.Insert(outgoing_neighbor);
          }
        }
      }
      if (graph->GetInDegree(generator_node) > 0 &&
          (size_t)generator_node < backward_adj.size()) {
        for (int incoming_neighbor : backward_adj[generator_node]) {
          if (!visited.Contains(incoming_neighbor)) {
            current_one_hop_neighborhood.push_back(incoming_neighbor);
            visited.Insert(incoming_neighbor);
          }
        }
      }
      // pick random nodes to get to 2-hop
      if (current_one_hop_neighborhood.size() > max_neighborhood_size) {
        std::vector<int> sampled_one_hop_neighborhood;
        std::sample(current_one_hop_neighborhood.begin(),
                    current_one_hop_neighborhood.end(),
                    std::back_inserter(sampled_one_hop_neighborhood),
                    max_neighborhood_size, generator);
        current_one_hop_neighborhood = std::move(sampled_one_hop_neighborhood);
      }
      // until here should be fast
      std::copy(current_one_hop_neighborhood.begin(),
                current_one_hop_neighborhood.end(),
                std::back_inserter(n_hop_map[1]));
      std::shuffle(current_one_hop_neighborhood.begin(),
                   current_one_hop_neighborhood.end(), generator);
      for (size_t j = 0; j < current_one_hop_neighborhood.size(); j++) {
        int u = current_one_hop_neighborhood[j];
        int current_two_hop_size = 0;
        if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
          current_two_hop_size += forward_adj[u].size();
        }
        if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
          current_two_hop_size += backward_adj[u].size();
        }
        // worst case for one node we might only grab the outgoing edges
        if (n_hop_map[2].size() + current_two_hop_size <
            max_neighborhood_size) {
          if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[u]) {
              if (!visited.Contains(outgoing_neighbor)) {
                visited.Insert(outgoing_neighbor);
                n_hop_map[2].push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[u]) {
              if (!visited.Contains(incoming_neighbor)) {
                visited.Insert(incoming_neighbor);
                n_hop_map[2].push_back(incoming_neighbor);
              }
            }
          }
        } else {
          std::vector<int> to_be_sampled_neighborhood;
          if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[u]) {
              if (!visited.Contains(outgoing_neighbor)) {
                to_be_sampled_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[u]) {
              if (!visited.Contains(incoming_neighbor)) {
                to_be_sampled_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
          std::vector<int> sampled_neighborhood;
          std::sample(to_be_sampled_neighborhood.begin(),
                      to_be_sampled_neighborhood.end(),
                      std::back_inserter(sampled_neighborhood),
                      max_neighborhood_size - n_hop_map[2].size(), generator);
          for (size_t k = 0; k < sampled_neighborhood.size(); k++) {
            int sampled_node = sampled_neighborhood[k];
            if (!visited.Contains(sampled_node)) {
              visited.Insert(sampled_node);
              n_hop_map[2].push_back(sampled_node);
            }
          }
          if (n_hop_map[2].size() == max_neighborhood_size) {
            visited.Clear();
            return n_hop_map;
          }
        }
      }
    }
  }
  visited.Clear();
  return n_hop_map;
}

std::vector<std::vector<int>>
NeighborhoodSearch::GetNHopNeighborhood(Graph *graph, int current_year,
                                        const std::vector<int> &generator_nodes,
                                        int num_hops) {
  std::vector<std::vector<int>> n_hop_map(std::max(2, num_hops + 1));
  std::vector<int> n_hop_neighborhood;
  thread_local ThreadLocalVisitedTracker visited;
  visited.Clear();

  const auto &forward_adj = graph->GetForwardAdjList();
  const auto &backward_adj = graph->GetBackwardAdjList();

  if (this->neighborhood_sample == -1) {
    thread_local std::vector<std::pair<int, int>> to_visit;
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      int generator_node = generator_nodes[i];
      to_visit.clear();
      size_t head = 0;
      to_visit.push_back({generator_node, 0});
      visited.Insert(generator_node);
      while (head < to_visit.size()) {
        std::pair<int, int> current_pair = to_visit[head++];
        int current_node = current_pair.first;
        int current_distance = current_pair.second;
        if (current_distance > 0) {
          n_hop_neighborhood.push_back(current_node);
        }
        if (current_distance < num_hops) {
          if (graph->GetOutDegree(current_node) > 0 &&
              (size_t)current_node < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[current_node]) {
              if (!visited.Contains(outgoing_neighbor)) {
                visited.Insert(outgoing_neighbor);
                to_visit.push_back({outgoing_neighbor, current_distance + 1});
              }
            }
          }
          if (graph->GetInDegree(current_node) > 0 &&
              (size_t)current_node < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[current_node]) {
              if (!visited.Contains(incoming_neighbor)) {
                visited.Insert(incoming_neighbor);
                to_visit.push_back({incoming_neighbor, current_distance + 1});
              }
            }
          }
        }
      }
    }
  } else {
    // NOTE: only supports randomly sampling from up to 2-hop
    n_hop_neighborhood.reserve(this->neighborhood_sample);
    size_t max_neighborhood_size = this->neighborhood_sample;
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      // get the 1-hop first
      int generator_node = generator_nodes[i];
      visited.Insert(generator_node);
      std::vector<int> current_one_hop_neighborhood;
      if (graph->GetOutDegree(generator_node) > 0 &&
          (size_t)generator_node < forward_adj.size()) {
        for (int outgoing_neighbor : forward_adj[generator_node]) {
          if (!visited.Contains(outgoing_neighbor)) {
            current_one_hop_neighborhood.push_back(outgoing_neighbor);
            visited.Insert(outgoing_neighbor);
          }
        }
      }
      if (graph->GetInDegree(generator_node) > 0 &&
          (size_t)generator_node < backward_adj.size()) {
        for (int incoming_neighbor : backward_adj[generator_node]) {
          if (!visited.Contains(incoming_neighbor)) {
            current_one_hop_neighborhood.push_back(incoming_neighbor);
            visited.Insert(incoming_neighbor);
          }
        }
      }
      // pick random nodes to get to 2-hop
      if (current_one_hop_neighborhood.size() > max_neighborhood_size) {
        std::vector<int> sampled_one_hop_neighborhood;
        std::sample(current_one_hop_neighborhood.begin(),
                    current_one_hop_neighborhood.end(),
                    std::back_inserter(sampled_one_hop_neighborhood),
                    max_neighborhood_size, generator);
        current_one_hop_neighborhood = std::move(sampled_one_hop_neighborhood);
      }
      // until here should be fast
      std::copy(current_one_hop_neighborhood.begin(),
                current_one_hop_neighborhood.end(),
                std::back_inserter(n_hop_neighborhood));
      std::shuffle(current_one_hop_neighborhood.begin(),
                   current_one_hop_neighborhood.end(), generator);
      for (size_t j = 0; j < current_one_hop_neighborhood.size(); j++) {
        int u = current_one_hop_neighborhood[j];
        int current_two_hop_size = 0;
        if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
          current_two_hop_size += forward_adj[u].size();
        }
        if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
          current_two_hop_size += backward_adj[u].size();
        }
        // worst case for one node we might only grab the outgoing edges
        if (n_hop_neighborhood.size() + current_two_hop_size <
            max_neighborhood_size) {
          if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[u]) {
              if (!visited.Contains(outgoing_neighbor)) {
                visited.Insert(outgoing_neighbor);
                n_hop_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[u]) {
              if (!visited.Contains(incoming_neighbor)) {
                visited.Insert(incoming_neighbor);
                n_hop_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
        } else {
          std::vector<int> to_be_sampled_neighborhood;
          if (graph->GetOutDegree(u) > 0 && (size_t)u < forward_adj.size()) {
            for (int outgoing_neighbor : forward_adj[u]) {
              if (!visited.Contains(outgoing_neighbor)) {
                to_be_sampled_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(u) > 0 && (size_t)u < backward_adj.size()) {
            for (int incoming_neighbor : backward_adj[u]) {
              if (!visited.Contains(incoming_neighbor)) {
                to_be_sampled_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
          std::vector<int> sampled_neighborhood;
          std::sample(to_be_sampled_neighborhood.begin(),
                      to_be_sampled_neighborhood.end(),
                      std::back_inserter(sampled_neighborhood),
                      max_neighborhood_size - n_hop_neighborhood.size(),
                      generator);
          for (size_t k = 0; k < sampled_neighborhood.size(); k++) {
            int sampled_node = sampled_neighborhood[k];
            if (!visited.Contains(sampled_node)) {
              visited.Insert(sampled_node);
              n_hop_neighborhood.push_back(sampled_node);
            }
          }
          if (n_hop_neighborhood.size() == max_neighborhood_size) {
            visited.Clear();
            n_hop_map[1] = std::move(n_hop_neighborhood);
            return n_hop_map;
          }
        }
      }
    }
  }
  visited.Clear();
  n_hop_map[1] = std::move(n_hop_neighborhood);
  return n_hop_map;
}

int NeighborhoodSearch::GetBinIndex(Graph *graph, int current_node,
                                    int current_year) {
  int current_diff = current_year - graph->GetYear(current_node);
  return this->GetBinIndex(current_diff);
}

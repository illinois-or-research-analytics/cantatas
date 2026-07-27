#include "neighborhood_search.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <sstream>

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
}

std::unordered_map<int, int> NeighborhoodSearch::GetNumCitationsPerNeighborhood(
    double alpha, int total_num_citations_neighborhood,
    const std::unordered_map<int, std::vector<int>> &n_hop_map) {
  std::unordered_map<int, int> num_citations_per_neighborhood;
  if (n_hop_map.size() > 0) {
    if (n_hop_map.size() == 2) {
      int total_num_citations_neighborhood_clamped =
          std::min((int)(n_hop_map.at(1).size() + n_hop_map.at(2).size()),
                   total_num_citations_neighborhood);
      num_citations_per_neighborhood[1] =
          std::min((int)(total_num_citations_neighborhood_clamped * alpha),
                   (int)n_hop_map.at(1).size());
      num_citations_per_neighborhood[2] =
          std::min(total_num_citations_neighborhood_clamped -
                       num_citations_per_neighborhood[1],
                   (int)n_hop_map.at(2).size());
    } else if (n_hop_map.size() == 1) {
      // Assuming if size is 1, the key is 1 since we explore distance 1 first
      if (n_hop_map.contains(1)) {
        num_citations_per_neighborhood[1] = std::min(
            total_num_citations_neighborhood, (int)n_hop_map.at(1).size());
      }
    }
  }
  return num_citations_per_neighborhood;
}

std::unordered_map<int, int> NeighborhoodSearch::BinOutdegrees(
    const std::unordered_map<int, std::vector<int>> &binned_neighborhood,
    int total_outdegree,
    const std::unordered_map<int, double> &binned_recency_probabilities) {
  // initially let's say total_outdegree (requested) = 100
  // [0.5, 0.2, 0.2, 0.1] recency probability that's the same for all agents
  // split 100 into proportions
  // [50, 20, 20, 10]
  // first check sum(E) == 100
  // [(6,7,9), (10,9,2), (19,1), (5 * )] actual neighborhood with node ids
  // [50 - 3, 20 - 3, 20 - 3, 10 - 1] -> "unfulfilled quota"
  // [47, 17, 17, 9] -> "unfulfilled quota"
  // look left and right to see if any bins are available
  // end up citing [3, 3, 2, x] things from each bin
  std::unordered_map<int, int> target_outdegree_per_bin_map;
  int remaining_outdegree = total_outdegree;
  for (int bin_index = 0; bin_index < this->num_bins; bin_index++) {
    if (remaining_outdegree == 0) {
      break;
    }
    double bin_probability = binned_recency_probabilities.contains(bin_index)
                                 ? binned_recency_probabilities.at(bin_index)
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
    target_outdegree_per_bin_map[current_bin_index] += 1;
    remaining_outdegree--;
  }
  for (int bin_index = 0; bin_index < this->num_bins; bin_index++) {
    int binned_size = binned_neighborhood.contains(bin_index)
                          ? binned_neighborhood.at(bin_index).size()
                          : 0;
    int current_uncited_num_nodes =
        target_outdegree_per_bin_map[bin_index] - binned_size;
    if (current_uncited_num_nodes > 0) {
      for (int sweep_index = bin_index - 1;
           sweep_index >= 0 && current_uncited_num_nodes > 0; sweep_index--) {
        int sweep_binned_size = binned_neighborhood.contains(sweep_index)
                                    ? binned_neighborhood.at(sweep_index).size()
                                    : 0;
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
        int sweep_binned_size = binned_neighborhood.contains(sweep_index)
                                    ? binned_neighborhood.at(sweep_index).size()
                                    : 0;
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
  /* 5,10,25 with explicit but ignored 1*/
  /* [1, 5) */
  /* [5, 10) */
  /* [10, 25) */
  /* [25, inf) */
  for (int i = 0; i < this->num_bins - 1; i++) {
    if (this->bin_boundaries.at(i) <= year_diff &&
        year_diff < this->bin_boundaries.at(i + 1)) {
      return i;
    }
  }
  return this->bin_boundaries.size() - 1;
}

std::unordered_map<int, std::vector<int>>
NeighborhoodSearch::BinNeighborhood(Graph *graph, int current_year,
                                    std::vector<int> n_hop_list) {
  std::unordered_map<int, std::vector<int>> binned_neighborhood;
  for (int i = 0; i < this->num_bins; i++) {
    binned_neighborhood[i] = {};
  }
  for (size_t i = 0; i < n_hop_list.size(); i++) {
    int current_node = n_hop_list[i];
    int current_bin = GetBinIndex(graph, current_node, current_year);
    binned_neighborhood[current_bin].push_back(current_node);
  }
  return binned_neighborhood;
}

std::unordered_map<int, std::vector<int>>
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

std::unordered_map<int, std::vector<int>>
NeighborhoodSearch::GetOneAndTwoDistanceNeighborhoods(
    Graph *graph, int current_year, const std::vector<int> &generator_nodes,
    int num_hops) {
  std::unordered_map<int, std::vector<int>> n_hop_map;
  if (this->neighborhood_sample == -1) {
    std::set<int> visited;
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      int generator_node = generator_nodes.at(i);
      std::queue<std::pair<int, int>> to_visit;
      to_visit.push({generator_node, 0});
      visited.insert(generator_node);
      while (!to_visit.empty()) {
        std::pair<int, int> current_pair = to_visit.front();
        to_visit.pop();
        int current_node = current_pair.first;
        int current_distance = current_pair.second;
        if (current_distance > 0) {
          n_hop_map[current_distance].push_back(current_node);
        }
        if (current_distance < num_hops) {
          if (graph->GetOutDegree(current_node) > 0) {
            for (auto const &outgoing_neighbor :
                 graph->GetForwardAdjList().at(current_node)) {
              if (!visited.contains(outgoing_neighbor)) {
                visited.insert(outgoing_neighbor);
                to_visit.push({outgoing_neighbor, current_distance + 1});
              }
            }
          }
          if (graph->GetInDegree(current_node) > 0) {
            for (auto const &incoming_neighbor :
                 graph->GetBackwardAdjList().at(current_node)) {
              if (!visited.contains(incoming_neighbor)) {
                visited.insert(incoming_neighbor);
                to_visit.push({incoming_neighbor, current_distance + 1});
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
    std::set<int> visited;
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      // get the 1-hop first
      int generator_node = generator_nodes.at(i);
      visited.insert(generator_node);
      std::vector<int> current_one_hop_neighborhood;
      if (graph->GetOutDegree(generator_node) > 0) {
        for (auto const &outgoing_neighbor :
             graph->GetForwardAdjList().at(generator_node)) {
          if (!visited.contains(outgoing_neighbor)) {
            current_one_hop_neighborhood.push_back(outgoing_neighbor);
            visited.insert(outgoing_neighbor);
          }
        }
      }
      if (graph->GetInDegree(generator_node) > 0) {
        for (auto const &incoming_neighbor :
             graph->GetBackwardAdjList().at(generator_node)) {
          if (!visited.contains(incoming_neighbor)) {
            current_one_hop_neighborhood.push_back(incoming_neighbor);
            visited.insert(incoming_neighbor);
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
        current_one_hop_neighborhood = sampled_one_hop_neighborhood;
      }
      // until here should be fast
      std::copy(current_one_hop_neighborhood.begin(),
                current_one_hop_neighborhood.end(),
                std::back_inserter(n_hop_map[1]));
      std::shuffle(current_one_hop_neighborhood.begin(),
                   current_one_hop_neighborhood.end(), generator);
      for (size_t j = 0; j < current_one_hop_neighborhood.size(); j++) {
        int current_two_hop_size = 0;
        if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
          current_two_hop_size += graph->GetForwardAdjList()
                                      .at(current_one_hop_neighborhood[j])
                                      .size();
        }
        if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
          current_two_hop_size += graph->GetBackwardAdjList()
                                      .at(current_one_hop_neighborhood[j])
                                      .size();
        }
        // worst case for one node we might only grab the outgoing edges
        if (n_hop_map[2].size() + current_two_hop_size <
            max_neighborhood_size) {
          if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &outgoing_neighbor : graph->GetForwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(outgoing_neighbor)) {
                visited.insert(outgoing_neighbor);
                n_hop_map[2].push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &incoming_neighbor : graph->GetBackwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(incoming_neighbor)) {
                visited.insert(incoming_neighbor);
                n_hop_map[2].push_back(incoming_neighbor);
              }
            }
          }
        } else {
          std::vector<int> to_be_sampled_neighborhood;
          if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &outgoing_neighbor : graph->GetForwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(outgoing_neighbor)) {
                to_be_sampled_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &incoming_neighbor : graph->GetBackwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(incoming_neighbor)) {
                to_be_sampled_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
          // std::sample(to_be_sampled_neighborhood.begin(),
          // to_be_sampled_neighborhood.end(), std::back_inserter(n_hop_map[2]),
          // max_neighborhood_size - n_hop_map[2].size(), generator); if
          // (n_hop_map[2].size() == max_neighborhood_size) {
          //     return n_hop_map;
          // }
          std::vector<int> sampled_neighborhood;
          std::sample(to_be_sampled_neighborhood.begin(),
                      to_be_sampled_neighborhood.end(),
                      std::back_inserter(sampled_neighborhood),
                      max_neighborhood_size - n_hop_map[2].size(), generator);
          for (size_t i = 0; i < sampled_neighborhood.size(); i++) {
            if (!visited.contains(sampled_neighborhood.at(i))) {
              visited.insert(sampled_neighborhood.at(i));
              n_hop_map[2].push_back(sampled_neighborhood.at(i));
            }
          }
          if (n_hop_map[2].size() == max_neighborhood_size) {
            return n_hop_map;
          }
        }
      }
    }
  }
  return n_hop_map;
}

std::unordered_map<int, std::vector<int>>
NeighborhoodSearch::GetNHopNeighborhood(Graph *graph, int current_year,
                                        const std::vector<int> &generator_nodes,
                                        int num_hops) {
  std::unordered_map<int, std::vector<int>> n_hop_map;
  std::vector<int> n_hop_neighborhood;
  if (this->neighborhood_sample == -1) {
    std::set<int> visited;
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      int generator_node = generator_nodes.at(i);
      std::queue<std::pair<int, int>> to_visit;
      to_visit.push({generator_node, 0});
      visited.insert(generator_node);
      while (!to_visit.empty()) {
        std::pair<int, int> current_pair = to_visit.front();
        to_visit.pop();
        int current_node = current_pair.first;
        int current_distance = current_pair.second;
        if (current_distance > 0) {
          n_hop_neighborhood.push_back(current_node);
        }
        if (current_distance < num_hops) {
          if (graph->GetOutDegree(current_node) > 0) {
            for (auto const &outgoing_neighbor :
                 graph->GetForwardAdjList().at(current_node)) {
              if (!visited.contains(outgoing_neighbor)) {
                visited.insert(outgoing_neighbor);
                to_visit.push({outgoing_neighbor, current_distance + 1});
              }
            }
          }
          if (graph->GetInDegree(current_node) > 0) {
            for (auto const &incoming_neighbor :
                 graph->GetBackwardAdjList().at(current_node)) {
              if (!visited.contains(incoming_neighbor)) {
                visited.insert(incoming_neighbor);
                to_visit.push({incoming_neighbor, current_distance + 1});
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
    std::set<int> visited;
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    for (size_t i = 0; i < generator_nodes.size(); i++) {
      // get the 1-hop first
      int generator_node = generator_nodes.at(i);
      visited.insert(generator_node);
      std::vector<int> current_one_hop_neighborhood;
      if (graph->GetOutDegree(generator_node) > 0) {
        for (auto const &outgoing_neighbor :
             graph->GetForwardAdjList().at(generator_node)) {
          if (!visited.contains(outgoing_neighbor)) {
            current_one_hop_neighborhood.push_back(outgoing_neighbor);
            visited.insert(outgoing_neighbor);
          }
        }
      }
      if (graph->GetInDegree(generator_node) > 0) {
        for (auto const &incoming_neighbor :
             graph->GetBackwardAdjList().at(generator_node)) {
          if (!visited.contains(incoming_neighbor)) {
            current_one_hop_neighborhood.push_back(incoming_neighbor);
            visited.insert(incoming_neighbor);
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
        current_one_hop_neighborhood = sampled_one_hop_neighborhood;
      }
      // until here should be fast
      std::copy(current_one_hop_neighborhood.begin(),
                current_one_hop_neighborhood.end(),
                std::back_inserter(n_hop_neighborhood));
      std::shuffle(current_one_hop_neighborhood.begin(),
                   current_one_hop_neighborhood.end(), generator);
      for (size_t j = 0; j < current_one_hop_neighborhood.size(); j++) {
        int current_two_hop_size = 0;
        if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
          current_two_hop_size += graph->GetForwardAdjList()
                                      .at(current_one_hop_neighborhood[j])
                                      .size();
        }
        if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
          current_two_hop_size += graph->GetBackwardAdjList()
                                      .at(current_one_hop_neighborhood[j])
                                      .size();
        }
        // worst case for one node we might only grab the outgoing edges
        if (n_hop_neighborhood.size() + current_two_hop_size <
            max_neighborhood_size) {
          if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &outgoing_neighbor : graph->GetForwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(outgoing_neighbor)) {
                visited.insert(outgoing_neighbor);
                n_hop_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &incoming_neighbor : graph->GetBackwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(incoming_neighbor)) {
                visited.insert(incoming_neighbor);
                n_hop_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
        } else {
          std::vector<int> to_be_sampled_neighborhood;
          if (graph->GetOutDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &outgoing_neighbor : graph->GetForwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(outgoing_neighbor)) {
                to_be_sampled_neighborhood.push_back(outgoing_neighbor);
              }
            }
          }
          if (graph->GetInDegree(current_one_hop_neighborhood[j]) > 0) {
            for (auto const &incoming_neighbor : graph->GetBackwardAdjList().at(
                     current_one_hop_neighborhood[j])) {
              if (!visited.contains(incoming_neighbor)) {
                to_be_sampled_neighborhood.push_back(incoming_neighbor);
              }
            }
          }
          // std::sample(to_be_sampled_neighborhood.begin(),
          // to_be_sampled_neighborhood.end(),
          // std::back_inserter(n_hop_neighborhood), max_neighborhood_size -
          // n_hop_neighborhood.size(), generator); if
          // (n_hop_neighborhood.size() == max_neighborhood_size) {
          //     n_hop_map[1] = n_hop_neighborhood;
          //     return n_hop_map;
          // }
          std::vector<int> sampled_neighborhood;
          // std::sample(to_be_sampled_neighborhood.begin(),
          // to_be_sampled_neighborhood.end(),
          // std::back_inserter(n_hop_neighborhood), max_neighborhood_size -
          // n_hop_neighborhood.size(), generator);
          std::sample(to_be_sampled_neighborhood.begin(),
                      to_be_sampled_neighborhood.end(),
                      std::back_inserter(sampled_neighborhood),
                      max_neighborhood_size - n_hop_neighborhood.size(),
                      generator);
          for (size_t i = 0; i < sampled_neighborhood.size(); i++) {
            if (!visited.contains(sampled_neighborhood.at(i))) {
              visited.insert(sampled_neighborhood.at(i));
              n_hop_neighborhood.push_back(sampled_neighborhood.at(i));
            }
          }
          if (n_hop_neighborhood.size() == max_neighborhood_size) {
            n_hop_map[1] = n_hop_neighborhood;
            return n_hop_map;
          }
        }
      }
    }
  }
  n_hop_map[1] = n_hop_neighborhood;
  return n_hop_map;
}

int NeighborhoodSearch::GetBinIndex(Graph *graph, int current_node,
                                    int current_year) {
  int current_diff = current_year - graph->GetYear(current_node);
  return this->GetBinIndex(current_diff);
}

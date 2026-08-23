#include "citation_engine.h"
#include "neighborhood_search.h"
#include "utils.h"
#include "vose_alias.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>

CitationEngine::CitationEngine(bool null_cartel, NeighborhoodSearch *ns) {
  this->null_cartel = null_cartel;
  this->neighborhood_search = ns;
}

int CitationEngine::MakeSameYearCitations(
    const std::set<int> &same_year_source_nodes, int num_new_nodes,
    const std::vector<int> &reverse_continuous_node_mapping,
    std::span<int> citations, int current_graph_size) {
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::uniform_int_distribution<int> int_uniform_distribution(0, num_new_nodes -
                                                                     1);
  int current_citation = int_uniform_distribution(generator);
  while (same_year_source_nodes.contains(current_citation)) {
    current_citation = int_uniform_distribution(generator);
  }
  citations[0] =
      reverse_continuous_node_mapping[current_graph_size + current_citation];
  return 1;
}

int CitationEngine::MakeUniformRandomCitationsFromGraph(
    Graph *graph, const std::vector<int> &reverse_continuous_node_mapping,
    std::vector<int> &generator_nodes, std::span<int> citations,
    int num_cited_so_far, int num_citations) {
  if (num_citations <= 0) {
    return 0;
  }
  int actual_num_cited = num_citations;
  std::set<int> selected;
  for (int i = 0; i < num_cited_so_far; i++) {
    selected.insert(citations[i]);
  }
  for (size_t i = 0; i < generator_nodes.size(); i++) {
    selected.insert(generator_nodes.at(i));
  }
  if ((int)graph->GetNodeSet().size() - (int)selected.size() <= num_citations) {
    actual_num_cited = (int)graph->GetNodeSet().size() - (int)selected.size();
    int current_citation_index = 0;
    for (auto const &node_id : graph->GetNodeSet()) {
      if (!selected.contains(node_id)) {
        citations[num_cited_so_far + current_citation_index] = node_id;
        selected.insert(node_id);
        current_citation_index++;
      }
    }
  } else {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    std::uniform_int_distribution<int> int_uniform_distribution(
        0, (int)(graph->GetNodeSet().size() - 1));
    int current_citation_index = 0;
    while (current_citation_index < actual_num_cited) {
      int current_citation = int_uniform_distribution(generator);
      int current_cited_node =
          reverse_continuous_node_mapping[current_citation];
      if (current_cited_node < 0) {
        std::cerr << "randomly selected negative node: "
                  << std::to_string(current_cited_node)
                  << " from continous index: "
                  << std::to_string(current_citation) << std::endl;
      }
      if (!selected.contains(current_cited_node)) {
        citations[num_cited_so_far + current_citation_index] =
            current_cited_node;
        selected.insert(current_cited_node);
        current_citation_index++;
      }
    }
  }
  return actual_num_cited;
}

int CitationEngine::MakeUniformRandomCitations(
    Graph *graph, const std::vector<int> &continuous_node_mapping,
    int current_year, const std::vector<int> &candidate_nodes,
    std::span<int> citations, int current_graph_size, int num_citations) {
  if (num_citations <= 0 || candidate_nodes.empty()) {
    return 0;
  }

  const size_t k = candidate_nodes.size();
  const size_t actual_num_cited = std::min(static_cast<size_t>(num_citations), k);
  if (actual_num_cited == k) {
    for (size_t i = 0; i < k; i++) {
      citations[i] = candidate_nodes[i];
    }
  } else if (actual_num_cited == 1) {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    std::uniform_int_distribution<size_t> dist(0, k - 1);
    citations[0] = candidate_nodes[dist(generator)];
  } else {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    std::uniform_int_distribution<size_t> dist(0, k - 1);
    size_t count = 0;
    if (k <= 64) {
      uint64_t seen_mask = 0;
      size_t attempts = 0;
      while (count < actual_num_cited && attempts < actual_num_cited * 50 + 100) {
        size_t idx = dist(generator);
        attempts++;
        uint64_t bit = 1ULL << idx;
        if ((seen_mask & bit) == 0) {
          seen_mask |= bit;
          citations[count++] = candidate_nodes[idx];
        }
      }
    } else {
      size_t attempts = 0;
      while (count < actual_num_cited && attempts < actual_num_cited * 50 + 100) {
        size_t idx = dist(generator);
        attempts++;
        int candidate = candidate_nodes[idx];
        bool duplicate = false;
        for (size_t i = 0; i < count; ++i) {
          if (citations[i] == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          citations[count++] = candidate;
        }
      }
    }
    if (count < actual_num_cited) {
      for (size_t i = 0; i < k && count < actual_num_cited; ++i) {
        int candidate = candidate_nodes[i];
        bool duplicate = false;
        for (size_t j = 0; j < count; ++j) {
          if (citations[j] == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          citations[count++] = candidate;
        }
      }
    }
  }
  return static_cast<int>(actual_num_cited);
}

int CitationEngine::MakeScoredCartelCitations(
    Graph *graph, const std::vector<int> &generator_nodes, int author_id,
    const std::vector<int> &continuous_node_mapping,
    const std::vector<std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations, int current_year,
    const std::vector<double> &binned_recency_probabilities,
    const NodeMetrics &metrics, const AgentWeights &weights,
    int current_graph_size) {
  // first handle within neighborhood
  std::set<int> generator_nodes_set(generator_nodes.begin(),
                                    generator_nodes.end());
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::vector<int> in_neighborhood_cartel_nodes;
  int cartel_id = graph->GetCartelID(author_id);
  std::set<int> current_cartel_authors = graph->GetCartelAuthors(cartel_id);
  for (size_t distance = 1; distance < n_hop_map.size(); ++distance) {
    std::vector<int> current_node_vec(n_hop_map[distance]);
    std::ranges::shuffle(current_node_vec, generator);
    for (auto const &node_id : current_node_vec) {
      int node_author_id = graph->GetAuthorId(node_id);
      int node_cartel_id = graph->GetCartelID(node_author_id);
      if (cartel_id == node_cartel_id && node_author_id != author_id &&
          !generator_nodes_set.contains(node_id)) {
        if (current_cartel_authors.contains(node_author_id)) {
          in_neighborhood_cartel_nodes.push_back(node_id);
          current_cartel_authors.erase(node_author_id);
        }
      }
    }
  }
  std::set<int> in_neighborhood_cartel_nodes_set(
      in_neighborhood_cartel_nodes.begin(), in_neighborhood_cartel_nodes.end());
  std::vector<std::vector<int>> binned_neighborhood =
      this->neighborhood_search->BinNeighborhood(graph, current_year,
                                                 in_neighborhood_cartel_nodes);
  std::vector<int> outdegree_per_bin_map =
      this->neighborhood_search->BinOutdegrees(binned_neighborhood,
                                               num_cartel_citations,
                                               binned_recency_probabilities);
  int num_locally_cited = 0;
  for (int bin_index = 0; bin_index < this->neighborhood_search->num_bins - 1;
       bin_index++) { // if there's only 1 bin then this is always false
    num_locally_cited += this->MakeCitations(
        graph, continuous_node_mapping, current_year,
        binned_neighborhood[bin_index], citations.subspan(num_locally_cited),
        metrics, weights, current_graph_size, outdegree_per_bin_map[bin_index]);
  }
  num_locally_cited += this->MakeUniformRandomCitations(
      graph, continuous_node_mapping, current_year,
      binned_neighborhood[this->neighborhood_search->num_bins - 1],
      citations.subspan(num_locally_cited), current_graph_size,
      outdegree_per_bin_map[this->neighborhood_search->num_bins - 1]);
  // next handle outside neighborhood
  if (num_locally_cited < num_cartel_citations) {
    int remaining_num_cartel_citations =
        num_cartel_citations - num_locally_cited;
    std::vector<int> outside_neighborhood_cartel_nodes;
    for (auto const &cartel_author_id : current_cartel_authors) {
      if (cartel_author_id == author_id) {
        continue;
      }
      std::vector<int> cartel_author_publications =
          graph->GetAuthorPublications(cartel_author_id);
      std::vector<int> already_published_cartel_author_publications;
      for (size_t i = 0; i < cartel_author_publications.size(); i++) {
        if (graph->GetNodeSet().contains(cartel_author_publications.at(i)) &&
            !in_neighborhood_cartel_nodes_set.contains(
                cartel_author_publications.at(i)) &&
            !generator_nodes_set.contains(cartel_author_publications.at(i))) {
          already_published_cartel_author_publications.push_back(
              cartel_author_publications.at(i));
        }
      }
      // assume that a cartel author has publications already
      if (already_published_cartel_author_publications.size() > 0) {
        std::ranges::shuffle(already_published_cartel_author_publications,
                             generator);
        outside_neighborhood_cartel_nodes.push_back(
            already_published_cartel_author_publications.at(0));
      }
    }
    std::vector<std::vector<int>> binned_neighborhood =
        this->neighborhood_search->BinNeighborhood(
            graph, current_year, outside_neighborhood_cartel_nodes);
    std::vector<int> outdegree_per_bin_map =
        this->neighborhood_search->BinOutdegrees(binned_neighborhood,
                                                 remaining_num_cartel_citations,
                                                 binned_recency_probabilities);
    for (int bin_index = 0; bin_index < this->neighborhood_search->num_bins - 1;
         bin_index++) { // if there's only 1 bin then this is always false
      num_locally_cited += this->MakeCitations(
          graph, continuous_node_mapping, current_year,
          binned_neighborhood[bin_index], citations.subspan(num_locally_cited),
          metrics, weights, current_graph_size,
          outdegree_per_bin_map[bin_index]);
    }
    num_locally_cited += this->MakeUniformRandomCitations(
        graph, continuous_node_mapping, current_year,
        binned_neighborhood[this->neighborhood_search->num_bins - 1],
        citations.subspan(num_locally_cited), current_graph_size,
        outdegree_per_bin_map[this->neighborhood_search->num_bins - 1]);
  }
  return num_locally_cited;
}

int CitationEngine::MakeCartelCitations(
    Graph *graph, const std::vector<int> &generator_nodes, int author_id,
    const std::vector<int> &continuous_node_mapping,
    const std::vector<std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations, int current_year,
    const std::vector<double> &binned_recency_probabilities,
    const NodeMetrics &metrics, const AgentWeights &weights,
    int current_graph_size) {
  if (num_cartel_citations == 0) {
    return 0;
  }
  if (this->null_cartel) {
    return this->MakeNullCartelCitations(graph, generator_nodes, author_id,
                                         continuous_node_mapping, n_hop_map,
                                         citations, num_cartel_citations);
  }
  return this->MakeScoredCartelCitations(
      graph, generator_nodes, author_id, continuous_node_mapping, n_hop_map,
      citations, num_cartel_citations, current_year,
      binned_recency_probabilities, metrics, weights, current_graph_size);
}

int CitationEngine::MakeNullCartelCitations(
    Graph *graph, const std::vector<int> &generator_nodes, int author_id,
    const std::vector<int> &continuous_node_mapping,
    const std::vector<std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations) {
  // assume at this point that we are a cartel author
  std::set<int> generator_nodes_set(generator_nodes.begin(),
                                    generator_nodes.end());
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::vector<int> in_neighborhood_cartel_nodes;
  int cartel_id = graph->GetCartelID(author_id);
  std::set<int> current_cartel_authors = graph->GetCartelAuthors(cartel_id);
  for (size_t distance = 1; distance < n_hop_map.size(); ++distance) {
    std::vector<int> current_node_vec(n_hop_map[distance]);
    std::ranges::shuffle(current_node_vec, generator);
    for (auto const &node_id : current_node_vec) {
      int node_author_id = graph->GetAuthorId(node_id);
      int node_cartel_id = graph->GetCartelID(node_author_id);
      if (cartel_id == node_cartel_id && author_id != node_author_id &&
          !generator_nodes_set.contains(node_id)) {
        if (current_cartel_authors.contains(node_author_id)) {
          in_neighborhood_cartel_nodes.push_back(node_id);
          current_cartel_authors.erase(node_author_id);
        }
      }
    }
  }
  int actual_num_cited = 0;
  std::ranges::shuffle(in_neighborhood_cartel_nodes, generator);
  for (size_t i = 0; i < in_neighborhood_cartel_nodes.size(); i++) {
    if (actual_num_cited == num_cartel_citations) {
      break;
    }
    citations[actual_num_cited] = in_neighborhood_cartel_nodes.at(i);
    actual_num_cited += 1;
  }
  std::set<int> in_neighborhood_cartel_nodes_set(
      in_neighborhood_cartel_nodes.begin(), in_neighborhood_cartel_nodes.end());
  if (in_neighborhood_cartel_nodes.size() !=
      in_neighborhood_cartel_nodes_set.size()) {
    std::cerr << "error: duplicate nodes inside in_neighborhood_cartel_nodes"
              << std::endl;
  }
  if (actual_num_cited < num_cartel_citations) {
    std::vector<int> cartel_nodes;
    // for(auto const& cartel_author_id : graph->GetCartelAuthors(cartel_id)) {
    for (auto const &cartel_author_id : current_cartel_authors) {
      if (cartel_author_id == author_id) {
        continue;
      }
      std::vector<int> cartel_author_publications =
          graph->GetAuthorPublications(cartel_author_id);
      std::vector<int> already_published_cartel_author_publications;
      for (size_t i = 0; i < cartel_author_publications.size(); i++) {
        if (graph->GetNodeSet().contains(cartel_author_publications.at(i)) &&
            !in_neighborhood_cartel_nodes_set.contains(
                cartel_author_publications.at(i)) &&
            !generator_nodes_set.contains(cartel_author_publications.at(i))) {
          already_published_cartel_author_publications.push_back(
              cartel_author_publications.at(i));
        }
      }
      // assume that a cartel author has publications already
      if (already_published_cartel_author_publications.size() > 0) {
        std::ranges::shuffle(already_published_cartel_author_publications,
                             generator);
        // cartel_nodes.push_back(already_published_cartel_author_publications.at(0));
        if (actual_num_cited == num_cartel_citations) {
          break;
        }
        if (in_neighborhood_cartel_nodes_set.contains(
                already_published_cartel_author_publications.at(0))) {
          std::cerr << "error: in_neighborhood_cartel_nodes_set already "
                       "contains outside neighborhood citation"
                    << std::endl;
        }
        citations[actual_num_cited] =
            already_published_cartel_author_publications.at(0);
        actual_num_cited += 1;
      }
    }
    // std::ranges::shuffle(cartel_nodes, generator);
    // for(size_t i = 0; i < cartel_nodes.size(); i ++) {
    //     if (actual_num_cited == num_cartel_citations) {
    //         break;
    //     }
    //     citations[actual_num_cited] = cartel_nodes.at(i);
    //     actual_num_cited += 1;
    // }
  }

  return actual_num_cited;
}
int CitationEngine::MakeCitations(
    Graph *graph, const std::vector<int> &continuous_node_mapping,
    int current_year, const std::vector<int> &candidate_nodes,
    std::span<int> citations, const NodeMetrics &metrics,
    const AgentWeights &weights, int current_graph_size, int num_citations) {
  if (num_citations <= 0 || candidate_nodes.empty()) {
    return 0;
  }
  const size_t k = candidate_nodes.size();
  const size_t actual_num_cited = std::min((size_t)num_citations, k);

  if (actual_num_cited == k) {
    for (size_t i = 0; i < k; ++i) {
      citations[i] = candidate_nodes[i];
    }
    return static_cast<int>(k);
  }

  // Calculate component sums across candidates
  double pa_sum = 0.0;
  double fit_sum = 0.0;
  double na_sum = 0.0;
  double ar_sum = 0.0;

  for (size_t i = 0; i < k; ++i) {
    int candidate_node = candidate_nodes[i];
    int continuous_node_id = ((size_t)candidate_node < continuous_node_mapping.size())
                                 ? continuous_node_mapping[candidate_node]
                                 : candidate_node;
    const auto &c = metrics.components[continuous_node_id];
    pa_sum += c.pa;
    fit_sum += c.fit;
    na_sum += c.na;
    ar_sum += c.ar;
  }

  const double scaled_pa_w = (pa_sum > 0.0) ? (weights.pa_weight / pa_sum) : 0.0;
  const double scaled_fit_w = (fit_sum > 0.0) ? (weights.fit_weight / fit_sum) : 0.0;
  const double scaled_na_w = (na_sum > 0.0) ? (weights.num_authors_weight / na_sum) : 0.0;
  const double scaled_ar_w = (ar_sum > 0.0) ? (weights.author_reputation_weight / ar_sum) : 0.0;

  // Thread-local scratchpad vectors to avoid per-call heap allocations
  thread_local std::vector<double> combined_weights;
  thread_local std::vector<int> sampled_indices;
  thread_local VoseAliasTable alias_table;

  combined_weights.resize(k);
  sampled_indices.resize(actual_num_cited);

  for (size_t i = 0; i < k; ++i) {
    int candidate_node = candidate_nodes[i];
    int continuous_node_id = ((size_t)candidate_node < continuous_node_mapping.size())
                                 ? continuous_node_mapping[candidate_node]
                                 : candidate_node;
    const auto &c = metrics.components[continuous_node_id];
    double score = c.pa * scaled_pa_w + c.fit * scaled_fit_w +
                   c.na * scaled_na_w + c.ar * scaled_ar_w;
    combined_weights[i] = (score > 0.0) ? score : 0.0;
  }

  pcg32 &generator = Utils::GetThreadLocalPRNG();
  alias_table.Init(combined_weights);

  int sampled_count = alias_table.SampleDistinct(
      actual_num_cited, std::span<int>(sampled_indices.data(), actual_num_cited),
      generator);

  for (int i = 0; i < sampled_count; ++i) {
    citations[i] = candidate_nodes[sampled_indices[i]];
  }

  return sampled_count;
}
int CitationEngine::GetNumCartelCitations(
    Graph *graph, int author_id,
    const std::vector<std::vector<int>> &n_hop_map,
    int total_num_citations_neighborhood) {
  std::set<int> cartel_authors_in_neighborhood;
  int current_cartel_id = graph->GetCartelID(author_id);
  if (current_cartel_id > 0) {
    for (size_t distance = 1; distance < n_hop_map.size(); ++distance) {
      const auto &node_vec = n_hop_map[distance];
      for (size_t i = 0; i < node_vec.size(); i++) {
        int node_author_id = graph->GetAuthorId(node_vec[i]);
        int node_cartel_id = graph->GetCartelID(node_author_id);
        if (current_cartel_id == node_cartel_id) {
          cartel_authors_in_neighborhood.insert(node_author_id);
        }
      }
    }
  }
  if ((size_t)total_num_citations_neighborhood >
      cartel_authors_in_neighborhood.size()) {
    return cartel_authors_in_neighborhood.size();
  }
  return total_num_citations_neighborhood;
}

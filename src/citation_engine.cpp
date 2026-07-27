#include "citation_engine.h"
#include "neighborhood_search.h"
#include "utils.h"
#include <Eigen/Dense>
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
    Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
    int current_year, const std::vector<int> &candidate_nodes,
    std::span<int> citations, int current_graph_size, int num_citations) {
  if (num_citations <= 0) {
    return 0;
  }
  if (candidate_nodes.size() <= 0) {
    return 0;
  }

  int actual_num_cited = num_citations;
  if (candidate_nodes.size() <= (size_t)num_citations) {
    actual_num_cited = candidate_nodes.size();
    for (int i = 0; i < actual_num_cited; i++) {
      citations[i] = candidate_nodes.at(i);
    }
  } else {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    std::uniform_int_distribution<int> int_uniform_distribution(
        0, (int)(candidate_nodes.size() - 1));
    std::set<int> selected;
    int current_citation_index = 0;
    while ((int)selected.size() != actual_num_cited) {
      int current_selected_index = int_uniform_distribution(generator);
      int current_node = candidate_nodes.at(current_selected_index);
      if (!selected.contains(current_node)) {
        citations[current_citation_index] = current_node;
        selected.insert(current_node);
        current_citation_index++;
      }
    }
  }
  return actual_num_cited;
}

int CitationEngine::MakeScoredCartelCitations(
    Graph *graph, const std::vector<int> &generator_nodes, int author_id,
    const std::unordered_map<int, int> &continuous_node_mapping,
    const std::unordered_map<int, std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations, int current_year,
    const std::unordered_map<int, double> &binned_recency_probabilities,
    const NodeMetrics &metrics, const AgentWeights &weights,
    int current_graph_size) {
  // first handle within neighborhood
  std::set<int> generator_nodes_set(generator_nodes.begin(),
                                    generator_nodes.end());
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::vector<int> in_neighborhood_cartel_nodes;
  int cartel_id = graph->GetCartelID(author_id);
  std::set<int> current_cartel_authors = graph->GetCartelAuthors(cartel_id);
  for (auto const &[distance, node_vec] : n_hop_map) {
    std::vector<int> current_node_vec(node_vec);
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
  std::unordered_map<int, std::vector<int>> binned_neighborhood =
      this->neighborhood_search->BinNeighborhood(graph, current_year,
                                                 in_neighborhood_cartel_nodes);
  std::unordered_map<int, int> outdegree_per_bin_map =
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
    std::unordered_map<int, std::vector<int>> binned_neighborhood =
        this->neighborhood_search->BinNeighborhood(
            graph, current_year, outside_neighborhood_cartel_nodes);
    std::unordered_map<int, int> outdegree_per_bin_map =
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
    const std::unordered_map<int, int> &continuous_node_mapping,
    const std::unordered_map<int, std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations, int current_year,
    const std::unordered_map<int, double> &binned_recency_probabilities,
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
    const std::unordered_map<int, int> &continuous_node_mapping,
    const std::unordered_map<int, std::vector<int>> &n_hop_map,
    std::span<int> citations, int num_cartel_citations) {
  // assume at this point that we are a cartel author
  std::set<int> generator_nodes_set(generator_nodes.begin(),
                                    generator_nodes.end());
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::vector<int> in_neighborhood_cartel_nodes;
  int cartel_id = graph->GetCartelID(author_id);
  std::set<int> current_cartel_authors = graph->GetCartelAuthors(cartel_id);
  for (auto const &[distance, node_vec] : n_hop_map) {
    std::vector<int> current_node_vec(node_vec);
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
    Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
    int current_year, const std::vector<int> &candidate_nodes,
    std::span<int> citations, const NodeMetrics &metrics,
    const AgentWeights &weights, int current_graph_size, int num_citations) {
  if (num_citations <= 0) {
    return 0;
  }
  if (candidate_nodes.size() <= 0) {
    return 0;
  }
  int actual_num_cited = num_citations;
  if (candidate_nodes.size() < (size_t)num_citations) {
    actual_num_cited = candidate_nodes.size();
  }
  std::vector<std::pair<double, int>> element_index_vec;
  pcg32 &generator = Utils::GetThreadLocalPRNG();

  /* begin weighted random sampling results */
  Eigen::MatrixXd current_scores(candidate_nodes.size(), 4);
  Eigen::Vector4d current_weights(weights.pa_weight, weights.fit_weight,
                                  weights.num_authors_weight,
                                  weights.author_reputation_weight);
  double pa_sum = 0.0;
  double fit_sum = 0.0;
  double na_sum = 0.0;
  double ar_sum = 0.0;
  std::vector<double> raw_pa_vec;
  std::vector<double> raw_fit_vec;
  std::vector<double> raw_na_vec;
  std::vector<double> raw_ar_vec;
  raw_pa_vec.reserve(candidate_nodes.size());
  raw_fit_vec.reserve(candidate_nodes.size());
  raw_na_vec.reserve(candidate_nodes.size());
  raw_ar_vec.reserve(candidate_nodes.size());
  for (size_t i = 0; i < candidate_nodes.size(); i++) {
    int continuous_node_id = continuous_node_mapping.at(candidate_nodes.at(i));
    double current_pa = metrics.pa_span[continuous_node_id];
    double current_fit = metrics.fit_span[continuous_node_id];
    double current_na = metrics.na_span[continuous_node_id];
    double current_ar = metrics.ar_span[continuous_node_id];
    pa_sum += current_pa;
    fit_sum += current_fit;
    na_sum += current_na;
    ar_sum += current_ar;
    raw_pa_vec.push_back(current_pa);
    raw_fit_vec.push_back(current_fit);
    raw_na_vec.push_back(current_na);
    raw_ar_vec.push_back(current_ar);
  }
  for (size_t i = 0; i < candidate_nodes.size(); i++) {
    current_scores(i, 0) = raw_pa_vec[i] / pa_sum;
    current_scores(i, 1) = raw_fit_vec[i] / fit_sum;
    current_scores(i, 2) = raw_na_vec[i] / na_sum;
    current_scores(i, 3) = raw_ar_vec[i] / ar_sum;
  }
  Eigen::MatrixXd current_weighted_scores = current_scores * current_weights;
  // double u = log(current_pa) + log(pa_weight);
  // double v = log(current_rec) + log(rec_weight);
  // double w = log(current_fit) + log(fit_weight);
  // double max_value = std::max(u, std::max(v, w));
  // current_scores[i] = max_value + log(exp(u - max_value) + exp(v - max_value)
  // + exp(w - max_value)); Eigen::ArrayXd
  // current_weighted_scores(candidate_nodes.size()); current_scores =
  // current_scores.array().log(); current_weights =
  // current_weights.array().log(); for(size_t i = 0; i <
  // candidate_nodes.size(); i ++) { double u = log(current_scores(i, 0)) +
  // log(current_weights(0)); double v = log(current_scores(i, 1)) +
  // log(current_weights(1)); double u = current_scores(i, 0) +
  // current_weights(0); double v = current_scores(i, 1) + current_weights(1);
  // double max_value = std::max(u, v);
  // current_weighted_scores(i) = max_value + log(exp(u - max_value) + exp(v -
  // max_value));
  // }

  std::uniform_real_distribution<double> wrs_uniform_distribution(0.0, 1.0);
  auto current_wrs_uniform = [&]() {
    return wrs_uniform_distribution(generator);
  };
  Eigen::ArrayXd current_bases =
      Eigen::ArrayXd::NullaryExpr(candidate_nodes.size(), current_wrs_uniform);
  // Eigen::ArrayXd weighted_random_sampling_results = current_bases.pow(1.0 /
  // current_weighted_scores.array());
  Eigen::ArrayXd weighted_random_sampling_results =
      current_bases.log() / current_weighted_scores.array();
  /* end weighted random sampling results */

  for (size_t i = 0; i < candidate_nodes.size(); i++) {
    element_index_vec.push_back(
        {weighted_random_sampling_results(i), candidate_nodes.at(i)});
  }
  std::ranges::shuffle(element_index_vec, generator);
  /* std::sort(element_index_vec.begin(), element_index_vec.end(), [](auto&
   * left, auto& right){ */
  if ((size_t)actual_num_cited == candidate_nodes.size()) {
    std::sort(element_index_vec.begin(), element_index_vec.end(),
              [](auto &left, auto &right) {
                return left.first > right.first; // read
              });
  } else {
    std::partial_sort(element_index_vec.begin(),
                      element_index_vec.begin() + actual_num_cited,
                      element_index_vec.end(), [](auto &left, auto &right) {
                        return left.first > right.first; // read
                      });
  }
  for (int i = 0; i < actual_num_cited; i++) {
    citations[i] = element_index_vec[i].second;
  }
  return actual_num_cited;
}
int CitationEngine::GetNumCartelCitations(
    Graph *graph, int author_id,
    const std::unordered_map<int, std::vector<int>> &n_hop_map,
    int total_num_citations_neighborhood) {
  std::set<int> cartel_authors_in_neighborhood;
  int current_cartel_id = graph->GetCartelID(author_id);
  if (current_cartel_id > 0) {
    for (auto const &[distance, node_vec] : n_hop_map) {
      for (size_t i = 0; i < node_vec.size(); i++) {
        int node_author_id = graph->GetAuthorId(node_vec.at(i));
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

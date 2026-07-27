#ifndef CITATION_ENGINE_H
#define CITATION_ENGINE_H

#include "graph.h"
#include "structs.h"
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

class NeighborhoodSearch;

class CitationEngine {
public:
  /*
  Input: bool null_cartel, NeighborhoodSearch *ns
  Output: CitationEngine object
  Description: Initializes a CitationEngine, injecting a dependency for
  neighborhood searches and a boolean flag determining whether cartel citations
  are random (null).
  */
  CitationEngine(bool null_cartel, NeighborhoodSearch *ns);

  /*
  Input: const std::set<int> &same_year_source_nodes, int num_new_nodes,
         const std::vector<int> &reverse_continuous_node_mapping,
         std::span<int> citations, int current_graph_size
  Output: int (number of citations actually made)
  Description: Simulates same-year citations by uniformly selecting among papers
  published in the same year to fulfill citation counts.
  */
  int MakeSameYearCitations(
      const std::set<int> &same_year_source_nodes, int num_new_nodes,
      const std::vector<int> &reverse_continuous_node_mapping,
      std::span<int> citations, int current_graph_size);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &reverse_continuous_node_mapping, std::vector<int> &generator_nodes,
  std::span<int> citations, int num_cited_so_far, int num_citations Output: int
  (number of citations made) Description: Randomly selects target nodes from the
  entire graph's history uniformly, ignoring metrics.
  */
  int MakeUniformRandomCitationsFromGraph(
      Graph *graph, const std::vector<int> &reverse_continuous_node_mapping,
      std::vector<int> &generator_nodes, std::span<int> citations,
      int num_cited_so_far, int num_citations);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, int current_year, const std::vector<int>
  &candidate_nodes, std::span<int> citations, int current_graph_size, int
  num_citations Output: int (number of citations made) Description: Makes
  uniform random citations selected exclusively from the provided candidate
  nodes.
  */
  int MakeUniformRandomCitations(
      Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
      int current_year, const std::vector<int> &candidate_nodes,
      std::span<int> citations, int current_graph_size, int num_citations);
  /*
  Input: Graph *graph, int author_id, const std::unordered_map<int,
  std::vector<int>> &n_hop_map, int total_num_citations_neighborhood Output: int
  (quota of cartel citations) Description: Calculates how many citations should
  be directed toward cartel co-authors based on neighborhood density and
  historical behavior.
  */
  int GetNumCartelCitations(
      Graph *graph, int author_id,
      const std::unordered_map<int, std::vector<int>> &n_hop_map,
      int total_num_citations_neighborhood);
  /*
  Input: Graph *graph, const std::vector<int> &generator_nodes, int author_id,
         const std::unordered_map<int, int> &continuous_node_mapping,
         const std::unordered_map<int, std::vector<int>> &n_hop_map,
  std::span<int> citations, int num_cartel_citations Output: int (number of
  cartel citations actually made) Description: Assigns citations uniformly at
  random within the author's cartel network (null model for cartel behavior).
  */
  int MakeNullCartelCitations(
      Graph *graph, const std::vector<int> &generator_nodes, int author_id,
      const std::unordered_map<int, int> &continuous_node_mapping,
      const std::unordered_map<int, std::vector<int>> &n_hop_map,
      std::span<int> citations, int num_cartel_citations);
  /*
  Input: Graph *graph, const std::vector<int> &generator_nodes, int author_id,
         const std::unordered_map<int, int> &continuous_node_mapping,
         const std::unordered_map<int, std::vector<int>> &n_hop_map,
  std::span<int> citations, int num_cartel_citations, int current_year, const
  std::unordered_map<int, double> &binned_recency_probabilities, const
  NodeMetrics &metrics, const AgentWeights &weights, int current_graph_size
  Output: int (number of citations made)
  Description: Distributes cartel citations intelligently based on node metrics
  (fitness, PA, etc.) and recency biases among co-authors.
  */
  int MakeScoredCartelCitations(
      Graph *graph, const std::vector<int> &generator_nodes, int author_id,
      const std::unordered_map<int, int> &continuous_node_mapping,
      const std::unordered_map<int, std::vector<int>> &n_hop_map,
      std::span<int> citations, int num_cartel_citations, int current_year,
      const std::unordered_map<int, double> &binned_recency_probabilities,
      const NodeMetrics &metrics, const AgentWeights &weights,
      int current_graph_size);
  /*
  Input: Same as MakeScoredCartelCitations
  Output: int (number of citations made)
  Description: Acts as a router to dispatch execution to either
  MakeNullCartelCitations or MakeScoredCartelCitations depending on the
  null_cartel engine flag.
  */
  int MakeCartelCitations(
      Graph *graph, const std::vector<int> &generator_nodes, int author_id,
      const std::unordered_map<int, int> &continuous_node_mapping,
      const std::unordered_map<int, std::vector<int>> &n_hop_map,
      std::span<int> citations, int num_cartel_citations, int current_year,
      const std::unordered_map<int, double> &binned_recency_probabilities,
      const NodeMetrics &metrics, const AgentWeights &weights,
      int current_graph_size);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, int current_year, const std::vector<int>
  &candidate_nodes, std::span<int> citations, const NodeMetrics &metrics, const
  AgentWeights &weights, int current_graph_size, int num_citations Output: int
  (number of citations made) Description: Orchestrates standard baseline
  citation logic across candidates using agent metric preferences (preferential
  attachment, fitness, reputation, etc.).
  */
  int MakeCitations(Graph *graph,
                    const std::unordered_map<int, int> &continuous_node_mapping,
                    int current_year, const std::vector<int> &candidate_nodes,
                    std::span<int> citations, const NodeMetrics &metrics,
                    const AgentWeights &weights, int current_graph_size,
                    int num_citations);

  bool null_cartel;
  NeighborhoodSearch *neighborhood_search;
};

#endif

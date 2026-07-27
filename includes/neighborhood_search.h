#ifndef NEIGHBORHOOD_SEARCH_H
#define NEIGHBORHOOD_SEARCH_H

#include "graph.h"
#include "utils.h"
#include <string>
#include <unordered_map>
#include <vector>

class NeighborhoodSearch {
public:
  /*
  Input: std::string recency_bins_str, bool use_alpha, int neighborhood_sample
  Output: NeighborhoodSearch object
  Description: Constructs a NeighborhoodSearch object and initializes recency
  bins for time-aware co-author network exploration.
  */
  NeighborhoodSearch(std::string recency_bins_str, bool use_alpha,
                     int neighborhood_sample);

  /*
  Input: Graph *graph, int current_year, const std::vector<int>
  &generator_nodes, int num_hops Output: std::unordered_map<int,
  std::vector<int>> (map of author ID to their neighborhood nodes) Description:
  Main entry point for gathering neighborhood nodes. Defers to specific N-hop
               strategies based on the configuration of the search instance.
  */
  std::unordered_map<int, std::vector<int>>
  GetNeighborhoodMap(Graph *graph, int current_year,
                     const std::vector<int> &generator_nodes, int num_hops);
  /*
  Input: Graph *graph, int current_year, const std::vector<int>
  &generator_nodes, int num_hops Output: std::unordered_map<int,
  std::vector<int>> (map of author ID to neighbor nodes) Description:
  Specifically computes the union of the 1-hop and 2-hop co-author network for
  the given generator nodes, filtering by the current year.
  */
  std::unordered_map<int, std::vector<int>>
  GetOneAndTwoDistanceNeighborhoods(Graph *graph, int current_year,
                                    const std::vector<int> &generator_nodes,
                                    int num_hops);
  /*
  Input: Graph *graph, int current_year, const std::vector<int>
  &generator_nodes, int num_hops Output: std::unordered_map<int,
  std::vector<int>> (map of author ID to neighbor nodes) Description: Computes
  the N-hop co-author network recursively or iteratively, expanding outward up
  to num_hops distances.
  */
  std::unordered_map<int, std::vector<int>>
  GetNHopNeighborhood(Graph *graph, int current_year,
                      const std::vector<int> &generator_nodes, int num_hops);

  /*
  Input: int year_diff
  Output: int (index of the corresponding recency bin)
  Description: Maps an absolute year difference to the appropriate index in the
  configured time bins (e.g., [0-1], [2-5], [6+]).
  */
  int GetBinIndex(int year_diff);
  /*
  Input: Graph *graph, int current_node, int current_year
  Output: int (index of the corresponding recency bin)
  Description: Extracts the publication year of a node from the graph and
  calculates its age relative to the current year, mapping it to a bin index.
  */
  int GetBinIndex(Graph *graph, int current_node, int current_year);
  /*
  Input: Graph *graph, int current_year, std::vector<int> n_hop_list
  Output: std::unordered_map<int, std::vector<int>> (bin index mapped to nodes
  in that age bin) Description: Partitions a flat list of neighborhood nodes
  into separate buckets depending on how recently the papers were published.
  */
  std::unordered_map<int, std::vector<int>>
  BinNeighborhood(Graph *graph, int current_year, std::vector<int> n_hop_list);
  /*
  Input: const std::unordered_map<int, std::vector<int>> &binned_neighborhood,
         int total_outdegree, const std::unordered_map<int, double>&
  binned_recency_probabilities Output: std::unordered_map<int, int> (number of
  citations allocated per bin) Description: Allocates a quota of out-degree
  (citations) across different age bins based on the configured probability
  distribution for recency.
  */
  std::unordered_map<int, int> BinOutdegrees(
      const std::unordered_map<int, std::vector<int>> &binned_neighborhood,
      int total_outdegree,
      const std::unordered_map<int, double> &binned_recency_probabilities);
  /*
  Input: double alpha, int total_num_citations_neighborhood,
         const std::unordered_map<int, std::vector<int>> &n_hop_map
  Output: std::unordered_map<int, int> (target node mapped to its allocated
  citation count) Description: Determines the precise number of citations to
  distribute across various distances or groups in the neighborhood based on an
  alpha bias factor.
  */
  std::unordered_map<int, int> GetNumCitationsPerNeighborhood(
      double alpha, int total_num_citations_neighborhood,
      const std::unordered_map<int, std::vector<int>> &n_hop_map);

  std::string recency_bins_str;
  bool use_alpha;
  int neighborhood_sample;
  int num_bins;
  std::vector<int> bin_boundaries;
  /*
  Input: None
  Output: void
  Description: Parses the recency_bins_str (e.g. "0-2,3-5,6-10") to establish
  internal integer boundary vectors for rapid bin lookups.
  */
  void InitializeBinBoundaries();
};

#endif

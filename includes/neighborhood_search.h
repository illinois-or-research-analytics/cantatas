#ifndef NEIGHBORHOOD_SEARCH_H
#define NEIGHBORHOOD_SEARCH_H

#include "ladybug/ladybug_db.h"
#include "pcg_random.hpp"

#include <span>
#include <string>
#include <vector>

/*
 * NeighborhoodSearch: candidate neighborhoods around an agent's generator,
 * and recency binning of candidates.
 *
 * With use_alpha the neighborhood is split into distance 1 (n_hop_map[1]) and
 * distance 2 (n_hop_map[2]); otherwise everything goes into n_hop_map[1].
 * Each distance is capped at neighborhood_sample nodes. Long neighbor lists
 * are sampled by drawing random positions instead of copying the list first,
 * so the cost per agent no longer grows with the size of the hubs it touches;
 * the sample is still uniform without replacement over unvisited entries.
 */
class NeighborhoodSearch {
public:
  NeighborhoodSearch(const std::string &recency_bins, bool use_alpha, int neighborhood_sample);

  // Fills n_hop_map (size 3) for the given generator. Reads only the CSR.
  void GetNeighborhoodMap(const ladybug::LadyBugDB &db, int32_t generator,
                          std::vector<std::vector<int>> &n_hop_map) const;

  int GetBinIndex(int year_diff) const;
  int GetBinIndex(const ladybug::LadyBugDB &db, int node, int current_year) const {
    return GetBinIndex(current_year - db.nodes.year[node]);
  }
  void BinNeighborhood(const ladybug::LadyBugDB &db, int current_year, std::span<const int> nodes,
                       std::vector<std::vector<int>> &bins) const;

  std::vector<int> GetNumCitationsPerNeighborhood(double alpha, int total, size_t hop1_size,
                                                  size_t hop2_size) const;
  std::vector<int> BinOutdegrees(const std::vector<int> &bin_sizes, int total_outdegree,
                                 const std::vector<double> &binned_recency_probabilities) const;

  const std::vector<int> &BinBoundaries() const { return bin_boundaries_; }

  int num_bins = 0;

private:
  std::vector<int> bin_boundaries_;
  std::vector<int> recency_bin_lut_;
  bool use_alpha_;
  int neighborhood_sample_;
};

#endif // NEIGHBORHOOD_SEARCH_H

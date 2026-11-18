#ifndef CITATION_ENGINE_H
#define CITATION_ENGINE_H

#include "structs.h"

#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

/*
 * CitationEngine: the four ways an agent picks the papers it cites.
 * All randomness comes from rng::Thread(), seeded per agent.
 */
class CitationEngine {
public:
  // One citation to a uniformly chosen paper of the same year that is not
  // itself a same-year source. New papers of the year are first_new_id ..
  // first_new_id + num_new_nodes - 1.
  static int MakeSameYearCitation(const std::vector<uint8_t> &is_same_year_source,
                                  int32_t first_new_id, int num_new_nodes, std::span<int> citations);

  // Uniformly random papers from the whole graph (existing nodes
  // [0, num_existing)), avoiding the papers already cited and the generator.
  static int MakeUniformRandomCitationsFromGraph(size_t num_existing, std::span<const int> generator_nodes,
                                                 std::span<int> citations, int num_cited_so_far,
                                                 int num_citations);

  // Uniformly random distinct papers from `candidates`.
  static int MakeUniformRandomCitations(std::span<const int> candidates, std::span<int> citations,
                                        int num_citations);

  // Papers from `candidates` drawn without replacement with probability
  // proportional to the agent's weighted preference score.
  static int MakeCitations(std::span<const int> candidates, std::span<int> citations,
                           const NodeMetrics &metrics, const AgentWeights &weights, int num_citations);
};

#endif // CITATION_ENGINE_H

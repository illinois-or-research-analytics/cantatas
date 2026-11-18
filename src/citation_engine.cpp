#include "citation_engine.h"
#include "fast_bitset.h"
#include "hash_set.h"
#include "rng.h"
#include "vectorized_scoring.h"
#include "vose_alias.h"

#include <algorithm>
#include <random>

int CitationEngine::MakeSameYearCitation(const std::vector<uint8_t> &is_same_year_source,
                                         int32_t first_new_id, int num_new_nodes,
                                         std::span<int> citations) {
  pcg32 &generator = rng::Thread();
  std::uniform_int_distribution<int> pick(0, num_new_nodes - 1);
  int index = pick(generator);
  while (is_same_year_source[index]) {
    index = pick(generator);
  }
  citations[0] = first_new_id + index;
  return 1;
}

int CitationEngine::MakeUniformRandomCitationsFromGraph(size_t num_existing,
                                                        std::span<const int> generator_nodes,
                                                        std::span<int> citations, int num_cited_so_far,
                                                        int num_citations) {
  if (num_citations <= 0) {
    return 0;
  }
  thread_local StampedHashSet<int32_t> selected;
  selected.Clear();
  for (int i = 0; i < num_cited_so_far; ++i) selected.Insert(citations[i]);
  for (int g : generator_nodes) selected.Insert(g);

  int count = num_citations;
  long long available = static_cast<long long>(num_existing) - static_cast<long long>(selected.size());
  if (available <= num_citations) {
    // Fewer candidates than requested: cite everything not yet selected.
    count = static_cast<int>(std::max(0LL, available));
    int written = 0;
    for (size_t u = 0; u < num_existing && written < count; ++u) {
      if (selected.Insert(static_cast<int32_t>(u))) {
        citations[num_cited_so_far + written++] = static_cast<int>(u);
      }
    }
    return written;
  }
  pcg32 &generator = rng::Thread();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(num_existing - 1));
  int written = 0;
  while (written < count) {
    int candidate = pick(generator);
    if (selected.Insert(candidate)) {
      citations[num_cited_so_far + written++] = candidate;
    }
  }
  return count;
}

int CitationEngine::MakeUniformRandomCitations(std::span<const int> candidates, std::span<int> citations,
                                               int num_citations) {
  if (num_citations <= 0 || candidates.empty()) {
    return 0;
  }
  const size_t k = candidates.size();
  const size_t want = std::min(static_cast<size_t>(num_citations), k);
  if (want == k) {
    std::copy(candidates.begin(), candidates.end(), citations.begin());
    return static_cast<int>(k);
  }
  pcg32 &generator = rng::Thread();
  std::uniform_int_distribution<size_t> pick(0, k - 1);
  if (want == 1) {
    citations[0] = candidates[pick(generator)];
    return 1;
  }
  thread_local StampedHashSet<uint64_t> used;
  used.Clear();
  size_t count = 0;
  size_t attempts = 0;
  while (count < want && attempts < want * 50 + 100) {
    size_t idx = pick(generator);
    ++attempts;
    if (used.Insert(idx)) {
      citations[count++] = candidates[idx];
    }
  }
  for (size_t idx = 0; idx < k && count < want; ++idx) {
    if (used.Insert(idx)) {
      citations[count++] = candidates[idx];
    }
  }
  return static_cast<int>(want);
}

int CitationEngine::MakeCitations(std::span<const int> candidates, std::span<int> citations,
                                  const NodeMetrics &metrics, const AgentWeights &weights,
                                  int num_citations) {
  if (num_citations <= 0 || candidates.empty()) {
    return 0;
  }
  const size_t k = candidates.size();
  const size_t want = std::min(static_cast<size_t>(num_citations), k);
  if (want == k) {
    std::copy(candidates.begin(), candidates.end(), citations.begin());
    return static_cast<int>(k);
  }

  thread_local ladybug::ScoringScratchpad score_scratch;
  thread_local ladybug::VoseScratchpad vose_scratch;
  thread_local ladybug::VoseAliasTable alias_table;
  thread_local ladybug::FastBitset visited;
  thread_local std::vector<int> sampled;

  sampled.resize(want);
  visited.Clear();
  std::span<double> weights_out;
  ladybug::VectorizedScoring::ComputeCandidateWeights(candidates, metrics, weights, score_scratch, weights_out);
  alias_table.Init(weights_out, vose_scratch);
  int got = alias_table.SampleWithoutReplacement(static_cast<int>(want), std::span<int>(sampled.data(), want),
                                                 rng::Thread(), visited);
  for (int i = 0; i < got; ++i) {
    citations[i] = candidates[sampled[i]];
  }
  return got;
}

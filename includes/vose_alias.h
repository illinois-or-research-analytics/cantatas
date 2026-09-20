#ifndef VOSE_ALIAS_H
#define VOSE_ALIAS_H

#include "fast_bitset.h"
#include "pcg_random.hpp"
#include <span>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <random>

namespace ladybug {

/*
 * Scratchpad buffer for Vose's Alias Table to eliminate all heap allocations
 * in multi-threaded parallel simulation regions.
 */
struct VoseScratchpad {
  std::vector<double> scaled_prob;
  std::vector<double> prob;
  std::vector<int> alias;
  std::vector<int> small_worklist;
  std::vector<int> large_worklist;

  void EnsureCapacity(size_t capacity) {
    if (scaled_prob.size() < capacity) {
      size_t new_cap = std::max(capacity, scaled_prob.size() * 2 + 128);
      scaled_prob.resize(new_cap);
      prob.resize(new_cap);
      alias.resize(new_cap);
      small_worklist.resize(new_cap);
      large_worklist.resize(new_cap);
    }
  }
};

/*
 * VoseAliasTable: O(1) weighted random sampling via Vose's Alias Method.
 * Replaces O(K log K) vector sorting with O(K) table initialization and O(1) constant-time draw.
 */
class VoseAliasTable {
public:
  VoseAliasTable() : size_(0) {}

  // Initialize alias table from non-negative weights using thread-local scratchpad
  void Init(std::span<const double> weights, VoseScratchpad &scratch);

  // Initialize alias table using internal storage
  void Init(std::span<const double> weights);

  // O(1) constant-time weighted random draw returning sampled index [0, size - 1]
  inline int Sample(pcg32 &rng) const noexcept {
    if (size_ <= 1) {
      return 0;
    }
    std::uniform_int_distribution<int> index_dist(0, static_cast<int>(size_ - 1));
    std::uniform_real_distribution<double> coin_dist(0.0, 1.0);

    int idx = index_dist(rng);
    double coin = coin_dist(rng);

    return (coin < prob_[idx]) ? idx : alias_[idx];
  }

  // Draw k unique items without replacement in expected O(k) time
  int SampleWithoutReplacement(int k, std::span<int> out_indices, pcg32 &rng, FastBitset &visited);

  size_t Size() const noexcept { return size_; }
  bool Empty() const noexcept { return size_ == 0; }

private:
  size_t size_;
  const double *prob_ = nullptr;
  const int *alias_ = nullptr;

  // Internal storage fallback if scratchpad is not provided
  std::vector<double> internal_prob_;
  std::vector<int> internal_alias_;
  VoseScratchpad internal_scratch_;
};

} // namespace ladybug

#endif // VOSE_ALIAS_H

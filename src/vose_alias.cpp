#include "vose_alias.h"
#include <algorithm>
#include <numeric>

namespace ladybug {

void VoseAliasTable::Init(std::span<const double> weights, VoseScratchpad &scratch) {
  size_ = weights.size();
  if (size_ == 0) {
    prob_ = nullptr;
    alias_ = nullptr;
    return;
  }

  if (size_ == 1) {
    scratch.EnsureCapacity(1);
    scratch.prob[0] = 1.0;
    scratch.alias[0] = 0;
    prob_ = scratch.prob.data();
    alias_ = scratch.alias.data();
    return;
  }

  scratch.EnsureCapacity(size_);

  double total_weight = 0.0;
  for (size_t i = 0; i < size_; ++i) {
    total_weight += std::max(0.0, weights[i]);
  }

  // If all weights are 0 or NaN, assign uniform distribution
  if (total_weight <= 0.0 || std::isnan(total_weight)) {
    for (size_t i = 0; i < size_; ++i) {
      scratch.prob[i] = 1.0;
      scratch.alias[i] = static_cast<int>(i);
    }
    prob_ = scratch.prob.data();
    alias_ = scratch.alias.data();
    return;
  }

  double scale = static_cast<double>(size_) / total_weight;
  int small_head = 0;
  int large_head = 0;

  for (size_t i = 0; i < size_; ++i) {
    double scaled = std::max(0.0, weights[i]) * scale;
    scratch.scaled_prob[i] = scaled;
    if (scaled < 1.0) {
      scratch.small_worklist[small_head++] = static_cast<int>(i);
    } else {
      scratch.large_worklist[large_head++] = static_cast<int>(i);
    }
  }

  int small_curr = 0;
  int large_curr = 0;

  while (small_curr < small_head && large_curr < large_head) {
    int s = scratch.small_worklist[small_curr++];
    int l = scratch.large_worklist[large_curr++];

    scratch.prob[s] = scratch.scaled_prob[s];
    scratch.alias[s] = l;

    scratch.scaled_prob[l] = (scratch.scaled_prob[l] + scratch.scaled_prob[s]) - 1.0;

    if (scratch.scaled_prob[l] < 1.0) {
      scratch.small_worklist[small_head++] = l;
    } else {
      scratch.large_worklist[large_head++] = l;
    }
  }

  while (large_curr < large_head) {
    int l = scratch.large_worklist[large_curr++];
    scratch.prob[l] = 1.0;
    scratch.alias[l] = l;
  }

  while (small_curr < small_head) {
    int s = scratch.small_worklist[small_curr++];
    scratch.prob[s] = 1.0;
    scratch.alias[s] = s;
  }

  prob_ = scratch.prob.data();
  alias_ = scratch.alias.data();
}

void VoseAliasTable::Init(std::span<const double> weights) {
  Init(weights, internal_scratch_);
}

int VoseAliasTable::SampleWithoutReplacement(int k, std::span<int> out_indices,
                                             pcg32 &rng, FastBitset &visited) {
  if (size_ == 0 || k <= 0) {
    return 0;
  }

  int target_k = std::min(k, static_cast<int>(size_));
  int count = 0;

  // When target_k equals total size, return all indices
  if (target_k == static_cast<int>(size_)) {
    for (size_t i = 0; i < size_; ++i) {
      out_indices[i] = static_cast<int>(i);
    }
    return target_k;
  }

  // Draw samples using Vose Alias table with fast bitset rejection
  int attempts = 0;
  int max_attempts = target_k * 20;

  while (count < target_k && attempts < max_attempts) {
    int sample = Sample(rng);
    attempts++;
    if (!visited.Test(sample)) {
      visited.Set(sample);
      out_indices[count++] = sample;
    }
  }

  // Fallback linear scan if rejection rate gets high
  if (count < target_k) {
    for (size_t i = 0; i < size_ && count < target_k; ++i) {
      int idx = static_cast<int>(i);
      if (!visited.Test(idx)) {
        visited.Set(idx);
        out_indices[count++] = idx;
      }
    }
  }

  return count;
}

} // namespace ladybug

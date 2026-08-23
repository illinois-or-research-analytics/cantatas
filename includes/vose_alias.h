#ifndef VOSE_ALIAS_H
#define VOSE_ALIAS_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

class VoseAliasTable {
public:
  VoseAliasTable() = default;

  /*
  Input: std::span<const double> weights
  Output: void
  Description: Initializes the Alias table in O(K) time using Vose's algorithm.
  Internal vectors retain capacity across invocations to avoid dynamic allocations.
  */
  void Init(std::span<const double> weights) {
    const size_t k = weights.size();
    if (k == 0) {
      prob.clear();
      alias.clear();
      return;
    }

    prob.resize(k);
    alias.resize(k);
    scaled_prob.resize(k);
    small_worklist.clear();
    large_worklist.clear();

    double total_weight = 0.0;
    for (size_t i = 0; i < k; ++i) {
      double w = (weights[i] > 0.0) ? weights[i] : 0.0;
      scaled_prob[i] = w;
      total_weight += w;
    }

    if (total_weight <= 0.0) {
      // Degenerate case: all weights are zero/non-positive, fallback to uniform
      std::fill(prob.begin(), prob.end(), 1.0);
      for (size_t i = 0; i < k; ++i) {
        alias[i] = static_cast<int>(i);
      }
      return;
    }

    // Scale probabilities so that average is 1.0
    const double scale = static_cast<double>(k) / total_weight;
    for (size_t i = 0; i < k; ++i) {
      scaled_prob[i] *= scale;
      if (scaled_prob[i] < 1.0) {
        small_worklist.push_back(static_cast<int>(i));
      } else {
        large_worklist.push_back(static_cast<int>(i));
      }
    }

    while (!small_worklist.empty() && !large_worklist.empty()) {
      int l = small_worklist.back();
      small_worklist.pop_back();
      int g = large_worklist.back();
      large_worklist.pop_back();

      prob[l] = scaled_prob[l];
      alias[l] = g;

      scaled_prob[g] = (scaled_prob[g] + scaled_prob[l]) - 1.0;
      if (scaled_prob[g] < 1.0) {
        small_worklist.push_back(g);
      } else {
        large_worklist.push_back(g);
      }
    }

    while (!large_worklist.empty()) {
      int g = large_worklist.back();
      large_worklist.pop_back();
      prob[g] = 1.0;
      alias[g] = g;
    }

    while (!small_worklist.empty()) {
      int l = small_worklist.back();
      small_worklist.pop_back();
      prob[l] = 1.0;
      alias[l] = l;
    }
  }

  /*
  Input: RNG& rng
  Output: int (sampled index in [0, K-1])
  Description: Generates a single sample in O(1) time using 1 uniform integer and 1 uniform float.
  */
  template <typename RNG>
  inline int Sample(RNG &rng) const {
    if (prob.empty()) {
      return 0;
    }
    std::uniform_int_distribution<size_t> dist(0, prob.size() - 1);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    size_t i = dist(rng);
    return (real_dist(rng) < prob[i]) ? static_cast<int>(i) : alias[i];
  }

  /*
  Input: size_t m, std::span<int> out_indices, RNG& rng
  Output: int (number of distinct sampled indices)
  Description: Samples M distinct indices without replacement into out_indices.
  Uses rejection sampling on alias draws, which is highly efficient when M << K.
  */
  template <typename RNG>
  int SampleDistinct(size_t m, std::span<int> out_indices, RNG &rng) const {
    const size_t k = prob.size();
    if (k == 0 || m == 0) {
      return 0;
    }
    const size_t actual_m = std::min(m, k);
    if (actual_m == k) {
      for (size_t i = 0; i < k; ++i) {
        out_indices[i] = static_cast<int>(i);
      }
      return static_cast<int>(k);
    }

    if (actual_m == 1) {
      out_indices[0] = Sample(rng);
      return 1;
    }

    size_t count = 0;
    size_t max_attempts = actual_m * 50 + 100;
    size_t attempts = 0;

    if (k <= 64) {
      uint64_t seen_mask = 0;
      while (count < actual_m && attempts < max_attempts) {
        int candidate = Sample(rng);
        attempts++;
        uint64_t bit = 1ULL << static_cast<unsigned>(candidate);
        if ((seen_mask & bit) == 0) {
          seen_mask |= bit;
          out_indices[count++] = candidate;
        }
      }
    } else {
      while (count < actual_m && attempts < max_attempts) {
        int candidate = Sample(rng);
        attempts++;
        bool duplicate = false;
        for (size_t i = 0; i < count; ++i) {
          if (out_indices[i] == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          out_indices[count++] = candidate;
        }
      }
    }

    // Deterministic fallback if rejection hit attempt limit (rare edge case)
    if (count < actual_m) {
      for (size_t i = 0; i < k && count < actual_m; ++i) {
        int candidate = static_cast<int>(i);
        bool duplicate = false;
        for (size_t j = 0; j < count; ++j) {
          if (out_indices[j] == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          out_indices[count++] = candidate;
        }
      }
    }

    return static_cast<int>(count);
  }

  inline size_t Size() const { return prob.size(); }

private:
  std::vector<double> prob;
  std::vector<int> alias;
  std::vector<double> scaled_prob;
  std::vector<int> small_worklist;
  std::vector<int> large_worklist;
};

#endif // VOSE_ALIAS_H

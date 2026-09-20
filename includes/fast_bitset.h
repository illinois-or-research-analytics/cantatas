#ifndef FAST_BITSET_H
#define FAST_BITSET_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace ladybug {

/*
 * FastBitset: High-performance flat bit array with sparse dirty-word tracking.
 * Provides O(1) bit insertion/lookups and O(k) Clear() operations where k is the number
 * of modified words, eliminating full-array scans in inner loops.
 */
class FastBitset {
public:
  FastBitset(size_t initial_capacity = 1024) {
    EnsureCapacity(initial_capacity);
  }

  inline void EnsureCapacity(size_t max_elements) {
    size_t required_words = (max_elements + 63) >> 6;
    if (required_words > words_.size()) {
      words_.resize(required_words + 256, 0ULL);
    }
  }

  inline bool Test(int node) const noexcept {
    if (node < 0) return false;
    size_t word_idx = static_cast<size_t>(node) >> 6;
    if (word_idx >= words_.size()) return false;
    return (words_[word_idx] & (1ULL << (static_cast<size_t>(node) & 63))) != 0ULL;
  }

  inline bool Contains(int node) const noexcept {
    return Test(node);
  }

  inline void Set(int node) noexcept {
    if (node < 0) return;
    size_t word_idx = static_cast<size_t>(node) >> 6;
    if (word_idx >= words_.size()) {
      EnsureCapacity(static_cast<size_t>(node) + 1024);
    }
    uint64_t mask = 1ULL << (static_cast<size_t>(node) & 63);
    if ((words_[word_idx] & mask) == 0ULL) {
      if (words_[word_idx] == 0ULL) {
        dirty_words_.push_back(word_idx);
      }
      words_[word_idx] |= mask;
      dirty_nodes_.push_back(node);
    }
  }

  inline void Insert(int node) noexcept {
    Set(node);
  }

  inline void Clear() noexcept {
    for (size_t word_idx : dirty_words_) {
      words_[word_idx] = 0ULL;
    }
    dirty_words_.clear();
    dirty_nodes_.clear();
  }

  inline size_t Size() const noexcept {
    return dirty_nodes_.size();
  }

  inline bool Empty() const noexcept {
    return dirty_nodes_.empty();
  }

  const std::vector<int> &GetSetNodes() const noexcept {
    return dirty_nodes_;
  }

private:
  std::vector<uint64_t> words_;
  std::vector<size_t> dirty_words_;
  std::vector<int> dirty_nodes_;
};

} // namespace ladybug

#endif // FAST_BITSET_H

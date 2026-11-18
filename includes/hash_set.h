#ifndef CANTATAS_HASH_SET_H
#define CANTATAS_HASH_SET_H

#include <cstdint>
#include <vector>

/*
 * StampedHashSet: open-addressing set with O(1) Clear().
 *
 * Used for per-agent "visited" sets. Memory follows the number of elements
 * actually inserted (typically tens of thousands), not the graph size.
 */
template <typename Key> class StampedHashSet {
public:
  explicit StampedHashSet(unsigned log2_capacity = 15) { Allocate(log2_capacity); }

  void Clear() {
    count_ = 0;
    if (log2_ > kDefaultLog2 + 5) {
      Allocate(kDefaultLog2); // give back memory after an unusually large agent
      return;
    }
    if (++generation_ == 0) {
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      generation_ = 1;
    }
  }

  bool Contains(Key key) const {
    size_t i = Slot(key);
    while (stamp_[i] == generation_) {
      if (keys_[i] == key) return true;
      i = (i + 1) & mask_;
    }
    return false;
  }

  // Inserts key; returns true if it was not already present.
  bool Insert(Key key) {
    if ((count_ + 1) * 2 > capacity_) {
      Grow();
    }
    size_t i = Slot(key);
    while (stamp_[i] == generation_) {
      if (keys_[i] == key) return false;
      i = (i + 1) & mask_;
    }
    stamp_[i] = generation_;
    keys_[i] = key;
    ++count_;
    return true;
  }

  size_t size() const { return count_; }

private:
  static constexpr unsigned kDefaultLog2 = 15;

  size_t Slot(Key key) const {
    uint64_t h = static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL;
    return static_cast<size_t>(h >> (64 - log2_));
  }

  void Allocate(unsigned log2) {
    log2_ = log2;
    capacity_ = size_t{1} << log2;
    mask_ = capacity_ - 1;
    stamp_.assign(capacity_, 0u);
    keys_.assign(capacity_, Key{});
    generation_ = 1;
    count_ = 0;
  }

  void Grow() {
    std::vector<uint32_t> old_stamp;
    std::vector<Key> old_keys;
    old_stamp.swap(stamp_);
    old_keys.swap(keys_);
    uint32_t old_generation = generation_;
    Allocate(log2_ + 1);
    for (size_t i = 0; i < old_stamp.size(); ++i) {
      if (old_stamp[i] == old_generation) {
        Insert(old_keys[i]);
      }
    }
  }

  std::vector<uint32_t> stamp_;
  std::vector<Key> keys_;
  uint32_t generation_ = 1;
  unsigned log2_ = kDefaultLog2;
  size_t capacity_ = 0;
  size_t mask_ = 0;
  size_t count_ = 0;
};

#endif // CANTATAS_HASH_SET_H

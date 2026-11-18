#include "check.h"
#include "hash_set.h"

#include <random>
#include <unordered_set>

int main() {
  StampedHashSet<int32_t> set(4); // tiny start to exercise growth
  std::unordered_set<int32_t> model;
  std::mt19937 gen(3);
  std::uniform_int_distribution<int32_t> value(-1000, 100000);
  for (int round = 0; round < 50; ++round) {
    set.Clear();
    model.clear();
    int inserts = 1 + round * 400;
    for (int i = 0; i < inserts; ++i) {
      int32_t x = value(gen);
      CHECK_EQ(set.Insert(x), model.insert(x).second);
    }
    CHECK_EQ(set.size(), model.size());
    for (int i = 0; i < 2000; ++i) {
      int32_t x = value(gen);
      CHECK_EQ(set.Contains(x), model.count(x) > 0);
    }
  }
  std::cout << "test_hash_set passed" << std::endl;
  return 0;
}

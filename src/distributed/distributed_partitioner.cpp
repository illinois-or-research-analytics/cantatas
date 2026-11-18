#include "distributed/distributed_partitioner.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>

namespace distributed {

void DistributedPartitioner::Partition(int world_size, int rank, const ladybug::LadyBugDB &db) {
  world_size_ = std::max(1, world_size);
  rank_ = rank;
  const size_t num_clusters = db.clusters.NumClusters();
  owner_.assign(num_clusters, 0);
  if (world_size_ == 1) {
    return;
  }
  std::vector<std::pair<size_t, int32_t>> sizes; // (size, cluster)
  sizes.reserve(num_clusters);
  for (size_t c = 0; c < num_clusters; ++c) {
    sizes.emplace_back(db.clusters.Members(static_cast<int32_t>(c)).size(), static_cast<int32_t>(c));
  }
  std::sort(sizes.begin(), sizes.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });
  using Load = std::pair<size_t, int>; // (load, rank)
  std::priority_queue<Load, std::vector<Load>, std::greater<Load>> loads;
  for (int r = 0; r < world_size_; ++r) loads.push({0, r});
  for (const auto &[size, cluster] : sizes) {
    auto [load, r] = loads.top();
    loads.pop();
    owner_[cluster] = r;
    loads.push({load + size, r});
  }
}

} // namespace distributed

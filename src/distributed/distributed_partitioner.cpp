#include "distributed/distributed_partitioner.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <queue>

namespace distributed {

DistributedPartitioner::DistributedPartitioner()
    : world_size_(1), my_rank_(0), assigned_node_count_(0) {}

DistributedPartitioner::~DistributedPartitioner() {}

void DistributedPartitioner::PartitionCommunities(
    int world_size, int my_rank,
    const ladybug::LadyBugClusterIndex &cluster_index,
    const ladybug::LadyBugNodeStore &node_store) {
  world_size_ = std::max(1, world_size);
  my_rank_ = my_rank;
  cluster_to_rank_.clear();
  my_clusters_.clear();
  assigned_node_count_ = 0;

  size_t total_nodes = node_store.GetNodeCount();
  if (total_nodes == 0) return;

  // Collect cluster sizes from node store
  std::map<int, size_t> cluster_sizes;
  for (size_t u = 0; u < total_nodes; ++u) {
    int cid = node_store.GetClusterId(static_cast<int>(u));
    cluster_sizes[cid]++;
  }

  if (world_size_ == 1) {
    for (const auto &[cid, sz] : cluster_sizes) {
      cluster_to_rank_[cid] = 0;
      my_clusters_.push_back(cid);
    }
    assigned_node_count_ = total_nodes;
    return;
  }

  // Multi-rank greedy LPT partition
  struct ClusterInfo {
    int cluster_id;
    size_t size;
  };

  std::vector<ClusterInfo> clusters;
  clusters.reserve(cluster_sizes.size());
  for (const auto &[cid, sz] : cluster_sizes) {
    clusters.push_back({cid, sz});
  }

  // Sort descending by size (deterministic tiebreak by cluster_id across all ranks)
  std::sort(clusters.begin(), clusters.end(), [](const ClusterInfo &a, const ClusterInfo &b) {
    if (a.size != b.size) return a.size > b.size;
    return a.cluster_id < b.cluster_id;
  });

  // Min-heap tracking (current_load, rank)
  using RankLoad = std::pair<size_t, int>;
  std::priority_queue<RankLoad, std::vector<RankLoad>, std::greater<RankLoad>> load_pq;
  for (int r = 0; r < world_size_; ++r) {
    load_pq.push({0, r});
  }

  for (const auto &c : clusters) {
    auto [current_load, r] = load_pq.top();
    load_pq.pop();

    cluster_to_rank_[c.cluster_id] = r;
    if (r == my_rank_) {
      my_clusters_.push_back(c.cluster_id);
      assigned_node_count_ += c.size;
    }
    load_pq.push({current_load + c.size, r});
  }
}

void DistributedPartitioner::SetClusterRank(int cluster_id, int rank) {
  cluster_to_rank_[cluster_id] = rank;
  auto it = std::find(my_clusters_.begin(), my_clusters_.end(), cluster_id);
  if (rank == my_rank_) {
    if (it == my_clusters_.end()) {
      my_clusters_.push_back(cluster_id);
    }
  } else {
    if (it != my_clusters_.end()) {
      my_clusters_.erase(it);
    }
  }
}

int DistributedPartitioner::GetOwnerRank(int cluster_id) const noexcept {
  if (world_size_ <= 1 || cluster_id < 0) return 0;
  auto it = cluster_to_rank_.find(cluster_id);
  if (it != cluster_to_rank_.end()) {
    return it->second;
  }
  return (cluster_id % world_size_);
}

int DistributedPartitioner::GetNodeOwnerRank(
    int node_id, const ladybug::LadyBugNodeStore &node_store) const noexcept {
  if (world_size_ <= 1 || node_id < 0) return 0;
  int cid = node_store.GetClusterId(node_id);
  return GetOwnerRank(cid);
}

bool DistributedPartitioner::IsLocalCluster(int cluster_id) const noexcept {
  return GetOwnerRank(cluster_id) == my_rank_;
}

bool DistributedPartitioner::IsLocalNode(
    int node_id, const ladybug::LadyBugNodeStore &node_store) const noexcept {
  return GetNodeOwnerRank(node_id, node_store) == my_rank_;
}

void DistributedPartitioner::RebalancePartitions(
    const std::vector<size_t> &cluster_weights) {
  if (world_size_ <= 1 || cluster_weights.empty()) return;

  struct ClusterWeight {
    int cluster_id;
    size_t weight;
  };

  std::vector<ClusterWeight> sorted_weights;
  sorted_weights.reserve(cluster_weights.size());
  for (size_t i = 0; i < cluster_weights.size(); ++i) {
    sorted_weights.push_back({static_cast<int>(i), cluster_weights[i]});
  }

  std::sort(sorted_weights.begin(), sorted_weights.end(),
            [](const ClusterWeight &a, const ClusterWeight &b) {
              if (a.weight != b.weight) return a.weight > b.weight;
              return a.cluster_id < b.cluster_id;
            });

  cluster_to_rank_.clear();
  my_clusters_.clear();
  assigned_node_count_ = 0;

  using RankLoad = std::pair<size_t, int>;
  std::priority_queue<RankLoad, std::vector<RankLoad>, std::greater<RankLoad>> load_pq;
  for (int r = 0; r < world_size_; ++r) {
    load_pq.push({0, r});
  }

  for (const auto &cw : sorted_weights) {
    auto [current_load, r] = load_pq.top();
    load_pq.pop();

    cluster_to_rank_[cw.cluster_id] = r;
    if (r == my_rank_) {
      my_clusters_.push_back(cw.cluster_id);
      assigned_node_count_ += cw.weight;
    }
    load_pq.push({current_load + cw.weight, r});
  }
}

} // namespace distributed

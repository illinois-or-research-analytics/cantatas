#ifndef DISTRIBUTED_PARTITIONER_H
#define DISTRIBUTED_PARTITIONER_H

#include "ladybug/ladybug_cluster_index.h"
#include "ladybug/ladybug_node_store.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace distributed {

/*
 * DistributedPartitioner: Shards the global citation network across MPI compute nodes
 * along dense Leiden research community boundaries.
 * Minimizes cross-rank network boundary traffic by keeping ~90% of intra-field
 * citation resolutions purely within local node memory.
 */
class DistributedPartitioner {
public:
  DistributedPartitioner();
  ~DistributedPartitioner();

  // Partition Leiden clusters across world_size ranks to balance node counts
  void PartitionCommunities(int world_size, int my_rank,
                            const ladybug::LadyBugClusterIndex &cluster_index,
                            const ladybug::LadyBugNodeStore &node_store);

  // Set manual mapping for testing
  void SetClusterRank(int cluster_id, int rank);

  // Lookups
  int GetOwnerRank(int cluster_id) const noexcept;
  int GetNodeOwnerRank(int node_id, const ladybug::LadyBugNodeStore &node_store) const noexcept;
  bool IsLocalCluster(int cluster_id) const noexcept;
  bool IsLocalNode(int node_id, const ladybug::LadyBugNodeStore &node_store) const noexcept;

  int GetMyRank() const noexcept { return my_rank_; }
  int GetWorldSize() const noexcept { return world_size_; }

  const std::vector<int> &GetMyAssignedClusters() const noexcept { return my_clusters_; }
  size_t GetAssignedNodeCount() const noexcept { return assigned_node_count_; }

  // Dynamic actor sub-partition rebalancing for asymmetric disciplinary growth
  void RebalancePartitions(const std::vector<size_t> &cluster_weights);

private:
  int world_size_;
  int my_rank_;
  size_t assigned_node_count_;

  std::unordered_map<int, int> cluster_to_rank_;
  std::vector<int> my_clusters_;
};

} // namespace distributed

#endif // DISTRIBUTED_PARTITIONER_H

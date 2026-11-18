#ifndef DISTRIBUTED_PARTITIONER_H
#define DISTRIBUTED_PARTITIONER_H

#include "ladybug/ladybug_db.h"

#include <cstdint>
#include <vector>

namespace distributed {

/*
 * DistributedPartitioner: decides which rank simulates which agents.
 *
 * Clusters are assigned to ranks by greedy longest-processing-time on cluster
 * size, so each rank simulates the agents of its clusters and builds recency
 * bins only for those clusters. Agents without a cluster are spread by node
 * ID. Because every agent has its own random stream, the assignment affects
 * only load balance, never results.
 */
class DistributedPartitioner {
public:
  void Partition(int world_size, int rank, const ladybug::LadyBugDB &db);
  bool IsLocal(int32_t cluster, int32_t node) const noexcept {
    if (world_size_ <= 1) return true;
    if (cluster >= 0 && static_cast<size_t>(cluster) < owner_.size()) {
      return owner_[cluster] == rank_;
    }
    return (node % world_size_) == rank_;
  }

private:
  int world_size_ = 1;
  int rank_ = 0;
  std::vector<int32_t> owner_; // owner_[cluster] = rank
};

} // namespace distributed

#endif // DISTRIBUTED_PARTITIONER_H

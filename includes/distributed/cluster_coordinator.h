#ifndef CLUSTER_COORDINATOR_H
#define CLUSTER_COORDINATOR_H

#include "agent_factory.h"

#include <cstdint>
#include <vector>

namespace distributed {

/*
 * ClusterCoordinator: the MPI layer. Every rank holds a full copy of the
 * graph; each year rank 0 broadcasts the new agents, ranks simulate their own
 * share, and all new citations are gathered on every rank. Without MPI (or
 * with one rank) every call is a no-op.
 */
class ClusterCoordinator {
public:
  ClusterCoordinator() = default;
  ~ClusterCoordinator();
  ClusterCoordinator(const ClusterCoordinator &) = delete;
  ClusterCoordinator &operator=(const ClusterCoordinator &) = delete;

  void Init(int &argc, char **&argv);
  void Finalize();
  [[noreturn]] void Abort(int code);

  int Rank() const noexcept { return rank_; }
  int WorldSize() const noexcept { return world_size_; }
  bool IsMaster() const noexcept { return rank_ == 0; }
  bool Enabled() const noexcept { return enabled_; }

  void Barrier();
  bool GlobalAny(bool local);
  void BroadcastU64(uint64_t &value);
  void BroadcastBatch(AgentBatch &batch);
  // Concatenates every rank's buffer, in rank order, on every rank.
  void AllgatherInts(const std::vector<int32_t> &local, std::vector<int32_t> &all);
  void AllreduceMaxInts(std::vector<int32_t> &values);

private:
  int rank_ = 0;
  int world_size_ = 1;
  bool enabled_ = false;
  [[maybe_unused]] bool initialized_by_me_ = false;
};

} // namespace distributed

#endif // CLUSTER_COORDINATOR_H

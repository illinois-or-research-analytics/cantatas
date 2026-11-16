#ifndef CLUSTER_COORDINATOR_H
#define CLUSTER_COORDINATOR_H

#include "distributed/actor_exchange.h"
#include "distributed/distributed_partitioner.h"
#include "ladybug/ladybug_db.h"
#include <cstdint>
#include <string>
#include <vector>

namespace distributed {

struct AgentBatchAttributes {
  std::vector<double> pa_weights;
  std::vector<double> fit_weights;
  std::vector<double> num_authors_weights;
  std::vector<double> author_reputation_weights;
  std::vector<double> alphas;
  std::vector<int> out_degrees;
  std::vector<int> fitness_lag_durations;
  std::vector<int> fitness_peak_values;
  std::vector<int> fitness_peak_durations;
  std::vector<int> num_authors;
  std::vector<int> is_same_year_source;
};

/*
 * ClusterCoordinator: Coordinates the distributed simulation lifecycle across MPI ranks.
 * - Manages MPI rank environment, process topology, and collective barriers.
 * - Coordinates annual BSP supersteps, cross-rank edge reductions, and distributed checkpoint commits.
 * - Aggregates global simulation metrics across compute nodes.
 */
class ClusterCoordinator {
public:
  ClusterCoordinator();
  ~ClusterCoordinator();

  ClusterCoordinator(const ClusterCoordinator &) = delete;
  ClusterCoordinator &operator=(const ClusterCoordinator &) = delete;

  ClusterCoordinator(ClusterCoordinator &&other) noexcept;
  ClusterCoordinator &operator=(ClusterCoordinator &&other) noexcept;

  // Initialize MPI runtime environment
  void Init(int &argc, char **&argv);

  // Shut down MPI runtime
  void Finalize();

  // Collective barrier synchronization across all ranks
  void Barrier();

  // Annual BSP Superstep Synchronization:
  // 1. Exchanges all inter-partition citations generated in the current year.
  // 2. Merges local + remote delta incoming citations into backward CSR.
  // 3. Atomically commits local database checkpoint and updates manifest.
  size_t SynchronizeSuperstep(int current_year, ladybug::LadyBugDB &db,
                              ActorExchange &exchange);

  // Broadcast manifest metadata from root rank to all worker ranks
  void BroadcastManifest(ladybug::DBManifest &manifest, int root = 0);

  // Broadcast new node assignments and agent attributes from root rank to all worker ranks
  void BroadcastNewNodeMetadata(std::vector<int> &new_nodes,
                                std::vector<int> &cluster_ids,
                                std::vector<int> &author_ids,
                                std::vector<int> &generator_nodes,
                                AgentBatchAttributes &attrs,
                                int root = 0);

  // Synchronize newly created edges across all ranks via MPI_Allgatherv
  void SynchronizeNewEdges(std::vector<std::pair<int, int>> &new_edges_vec);

  // Synchronize sampled neighborhood sizes and fully random citations across all ranks
  void SynchronizeNodeDiagnostics(std::vector<int> &sampled_neighborhood_sizes,
                                  std::vector<int> &fully_random_citations);

  // Global reduction helpers
  uint64_t GlobalSum(uint64_t local_val) const;
  uint64_t GlobalMax(uint64_t local_val) const;
  bool GlobalAny(bool local_val) const;

  // Accessors
  int GetRank() const noexcept { return rank_; }
  int GetWorldSize() const noexcept { return world_size_; }
  bool IsMaster() const noexcept { return rank_ == 0; }
  bool IsMPIEnabled() const noexcept { return mpi_enabled_; }

private:
  int rank_;
  int world_size_;
  bool mpi_enabled_;
  bool initialized_by_me_;
};

} // namespace distributed

#endif // CLUSTER_COORDINATOR_H

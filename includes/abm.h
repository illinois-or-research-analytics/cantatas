#ifndef ABM_H
#define ABM_H

#include "agent_factory.h"
#include "config.h"
#include "distributed/cluster_coordinator.h"
#include "distributed/distributed_partitioner.h"
#include "ladybug/ladybug_db.h"
#include "neighborhood_search.h"
#include "signal_supervisor.h"
#include "structs.h"
#include "utils.h"

#include <chrono>
#include <memory>
#include <vector>

enum class ResumeMode { No, Yes, Auto };

// Process exit codes.
constexpr int kExitFinished = 0;
constexpr int kExitError = 1;
constexpr int kExitPaused = 3; // a snapshot was written; run again to continue

/*
 * ABM: the simulation driver.
 *
 * Each simulated year (bulk-synchronous):
 *   1. rank 0 creates the year's agents and broadcasts them;
 *   2. every rank computes scores and simulates the agents of its clusters,
 *      reading only its copy of LadyBugDB (unchanged during the year);
 *   3. all new citations are gathered on every rank and applied in the same
 *      order, so every copy of the graph stays identical;
 *   4. ranks agree whether to pause; snapshots are written by rank 0.
 */
class ABM {
public:
  ABM(const SimulationConfig &config, distributed::ClusterCoordinator &coordinator);
  int Run(ResumeMode resume);

private:
  bool Initialize(ResumeMode resume);
  void SimulateYear(int32_t year);
  void ComputeScores(int32_t year, size_t num_nodes);
  void ApplyYear(const AgentBatch &batch, const std::vector<int32_t> &edges,
                 const std::vector<int32_t> &sampled_sizes, const std::vector<int32_t> &fully_random);
  void WriteCheckpoint(const char *reason);
  double EstimatedSnapshotSeconds() const;
  double EstimatedExportSeconds() const;
  void Export();
  void LogMemory(int32_t year);

  const SimulationConfig &config_;
  distributed::ClusterCoordinator &coordinator_;
  SimLogger logger_;
  ladybug::LadyBugDB db_;
  NeighborhoodSearch search_;
  distributed::DistributedPartitioner partitioner_;
  std::unique_ptr<AgentFactory> factory_;
  SignalSupervisor supervisor_;
  std::vector<double> binned_recency_probabilities_;
  ladybug::Column<NodeScoreComponents> scores_;
  double last_snapshot_seconds_ = -1;
  std::chrono::steady_clock::time_point last_snapshot_time_ = std::chrono::steady_clock::now();
};

#endif // ABM_H

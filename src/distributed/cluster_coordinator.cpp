#include "distributed/cluster_coordinator.h"
#include <iostream>

#if __has_include(<mpi.h>)
#define DISTRIBUTED_HAS_MPI 1
#include <mpi.h>
#else
#define DISTRIBUTED_HAS_MPI 0
#endif

namespace distributed {

ClusterCoordinator::ClusterCoordinator()
    : rank_(0), world_size_(1), mpi_enabled_(false), initialized_by_me_(false) {}

ClusterCoordinator::~ClusterCoordinator() {
  Finalize();
}

void ClusterCoordinator::Init(int &argc, char **&argv) {
  // TODO: Step 1 - Initialize MPI Runtime Environment
  // 1. Set environment variables for fallback transport (e.g., TCP / shared memory)
  // 2. Check if MPI is already initialized (MPI_Initialized).
  // 3. If not initialized:
  //    - Call MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //    - Mark initialized_by_me_ = true;
  // 4. Query rank and world size:
  //    - MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  //    - MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  // 5. mpi_enabled_ = (world_size_ > 1);
#if DISTRIBUTED_HAS_MPI
  setenv("OMPI_MCA_pml", "ob1", 0);
  setenv("OMPI_MCA_btl", "self,vader,tcp", 0);

  int flag = 0;
  MPI_Initialized(&flag);
  if (!flag) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    initialized_by_me_ = true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  mpi_enabled_ = (world_size_ > 1);
#else
  (void)argc;
  (void)argv;
  rank_ = 0;
  world_size_ = 1;
  mpi_enabled_ = false;
  initialized_by_me_ = false;
#endif
}

void ClusterCoordinator::Finalize() {
  // TODO: Step 2 - Shutdown MPI runtime if initialized by this instance
#if DISTRIBUTED_HAS_MPI
  if (initialized_by_me_) {
    int flag = 0;
    MPI_Finalized(&flag);
    if (!flag) {
      MPI_Finalize();
    }
    initialized_by_me_ = false;
  }
#endif
  rank_ = 0;
  world_size_ = 1;
  mpi_enabled_ = false;
}

void ClusterCoordinator::Barrier() {
  // TODO: Step 3 - Collective barrier across all compute ranks
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

void ClusterCoordinator::SynchronizeSuperstep(int current_year,
                                              ladybug::LadyBugDB &db,
                                              ActorExchange &exchange) {
  // TODO: Step 4 - Annual BSP Superstep Synchronization
  // 1. Bulk exchange inter-partition citations across cluster ranks
  exchange.ExchangeAnnualCitations(rank_, world_size_, db);

  // 2. Merge local + received remote delta incoming citations into Backward CSR
  db.CSR().MergeAnnualDeltaBuffer();

  // 3. Collective barrier before writing persistent checkpoint
  Barrier();

  // 4. Atomically commit checkpoint for current year
  db.Checkpoint(current_year);

  // 5. Collective barrier to ensure all nodes committed to disk
  Barrier();
}

void ClusterCoordinator::BroadcastManifest(ladybug::DBManifest &manifest,
                                           int root) {
  // TODO: Step 5 - Broadcast master manifest metadata across all worker ranks
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_) {
    int buf[4];
    if (rank_ == root) {
      buf[0] = manifest.version;
      buf[1] = manifest.current_year;
      buf[2] = static_cast<int>(manifest.node_count);
      buf[3] = static_cast<int>(manifest.edge_count);
    }
    MPI_Bcast(buf, 4, MPI_INT, root, MPI_COMM_WORLD);
    if (rank_ != root) {
      manifest.version = buf[0];
      manifest.current_year = buf[1];
      manifest.node_count = static_cast<size_t>(buf[2]);
      manifest.edge_count = static_cast<size_t>(buf[3]);
    }
  }
#else
  (void)manifest;
  (void)root;
#endif
}

uint64_t ClusterCoordinator::GlobalSum(uint64_t local_val) const {
  // TODO: Step 6 - Global all-reduce sum across all compute ranks
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_) {
    uint64_t global_val = 0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_val;
  }
#endif
  return local_val;
}

uint64_t ClusterCoordinator::GlobalMax(uint64_t local_val) const {
  // TODO: Step 7 - Global all-reduce max across all compute ranks
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_) {
    uint64_t global_val = 0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_UINT64_T, MPI_MAX,
                  MPI_COMM_WORLD);
    return global_val;
  }
#endif
  return local_val;
}

} // namespace distributed

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

ClusterCoordinator::ClusterCoordinator(ClusterCoordinator &&other) noexcept
    : rank_(other.rank_),
      world_size_(other.world_size_),
      mpi_enabled_(other.mpi_enabled_),
      initialized_by_me_(other.initialized_by_me_) {
  other.initialized_by_me_ = false;
  other.rank_ = 0;
  other.world_size_ = 1;
  other.mpi_enabled_ = false;
}

ClusterCoordinator &ClusterCoordinator::operator=(ClusterCoordinator &&other) noexcept {
  if (this != &other) {
    Finalize();
    rank_ = other.rank_;
    world_size_ = other.world_size_;
    mpi_enabled_ = other.mpi_enabled_;
    initialized_by_me_ = other.initialized_by_me_;
    other.initialized_by_me_ = false;
    other.rank_ = 0;
    other.world_size_ = 1;
    other.mpi_enabled_ = false;
  }
  return *this;
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
  setenv("UCX_TLS", "sm,self,tcp", 0);
  setenv("UCX_NET_DEVICES", "all", 0);

  bool in_mpi_env = (getenv("OMPI_COMM_WORLD_SIZE") != nullptr ||
                     getenv("PMIX_RANK") != nullptr ||
                     getenv("SLURM_PROCID") != nullptr ||
                     getenv("MPI_LOCALRANKID") != nullptr);

  int flag = 0;
  MPI_Initialized(&flag);
  if (flag || in_mpi_env) {
    if (!flag) {
      int provided = 0;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
      initialized_by_me_ = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    mpi_enabled_ = (world_size_ > 1);
    return;
  }
#endif
  rank_ = 0;
  world_size_ = 1;
  mpi_enabled_ = false;
  initialized_by_me_ = false;
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

size_t ClusterCoordinator::SynchronizeSuperstep(int current_year,
                                                ladybug::LadyBugDB &db,
                                                ActorExchange &exchange) {
  // TODO: Step 4 - Annual BSP Superstep Synchronization
  // 1. Bulk exchange inter-partition citations across cluster ranks
  size_t routed = exchange.ExchangeAnnualCitations(rank_, world_size_, db);

  // 2. Merge local + received remote delta incoming citations into Backward CSR
  db.CSR().MergeAnnualDeltaBuffer();

  // 3. Collective barrier before writing persistent checkpoint
  Barrier();

  // 4. Atomically commit checkpoint for current year
  db.Checkpoint(current_year);

  // 5. Collective barrier to ensure all nodes committed to disk
  Barrier();
  return routed;
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

void ClusterCoordinator::BroadcastNewNodeMetadata(
    std::vector<int> &new_nodes,
    std::vector<int> &cluster_ids,
    std::vector<int> &author_ids,
    std::vector<int> &generator_nodes,
    AgentBatchAttributes &attrs,
    int root) {
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_ && world_size_ > 1) {
    int count = static_cast<int>(new_nodes.size());
    MPI_Bcast(&count, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank_ != root) {
      new_nodes.resize(count);
      cluster_ids.resize(count);
      author_ids.resize(count);
      generator_nodes.resize(count);
      attrs.pa_weights.resize(count);
      attrs.fit_weights.resize(count);
      attrs.num_authors_weights.resize(count);
      attrs.author_reputation_weights.resize(count);
      attrs.alphas.resize(count);
      attrs.out_degrees.resize(count);
      attrs.fitness_lag_durations.resize(count);
      attrs.fitness_peak_values.resize(count);
      attrs.fitness_peak_durations.resize(count);
      attrs.num_authors.resize(count);
      attrs.is_same_year_source.resize(count);
    }
    if (count > 0) {
      MPI_Bcast(new_nodes.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(cluster_ids.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(author_ids.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(generator_nodes.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.pa_weights.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.fit_weights.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.num_authors_weights.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.author_reputation_weights.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.alphas.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.out_degrees.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.fitness_lag_durations.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.fitness_peak_values.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.fitness_peak_durations.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.num_authors.data(), count, MPI_INT, root, MPI_COMM_WORLD);
      MPI_Bcast(attrs.is_same_year_source.data(), count, MPI_INT, root, MPI_COMM_WORLD);
    }
  }
#else
  (void)new_nodes;
  (void)cluster_ids;
  (void)author_ids;
  (void)generator_nodes;
  (void)attrs;
  (void)root;
#endif
}

void ClusterCoordinator::SynchronizeNodeDiagnostics(
    std::vector<int> &sampled_neighborhood_sizes,
    std::vector<int> &fully_random_citations) {
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_ && world_size_ > 1) {
    if (!sampled_neighborhood_sizes.empty()) {
      MPI_Allreduce(MPI_IN_PLACE, sampled_neighborhood_sizes.data(),
                    static_cast<int>(sampled_neighborhood_sizes.size()),
                    MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    if (!fully_random_citations.empty()) {
      MPI_Allreduce(MPI_IN_PLACE, fully_random_citations.data(),
                    static_cast<int>(fully_random_citations.size()),
                    MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
  }
#else
  (void)sampled_neighborhood_sizes;
  (void)fully_random_citations;
#endif
}

void ClusterCoordinator::SynchronizeNewEdges(
    std::vector<std::pair<int, int>> &new_edges_vec) {
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_ && world_size_ > 1) {
    int local_edge_count = static_cast<int>(new_edges_vec.size());
    std::vector<int> edge_counts(world_size_, 0);
    MPI_Allgather(&local_edge_count, 1, MPI_INT, edge_counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> recv_counts(world_size_, 0);
    std::vector<int> displs(world_size_, 0);
    int total_ints = 0;
    for (int r = 0; r < world_size_; ++r) {
      recv_counts[r] = edge_counts[r] * 2;
      displs[r] = total_ints;
      total_ints += recv_counts[r];
    }

    std::vector<int32_t> flat_send(local_edge_count * 2);
    for (int i = 0; i < local_edge_count; ++i) {
      flat_send[2 * i] = new_edges_vec[i].first;
      flat_send[2 * i + 1] = new_edges_vec[i].second;
    }

    std::vector<int32_t> flat_recv(total_ints);
    MPI_Allgatherv(flat_send.data(), local_edge_count * 2, MPI_INT32_T,
                   flat_recv.data(), recv_counts.data(), displs.data(),
                   MPI_INT32_T, MPI_COMM_WORLD);

    new_edges_vec.clear();
    new_edges_vec.reserve(total_ints / 2);
    for (int i = 0; i < total_ints; i += 2) {
      new_edges_vec.push_back({flat_recv[i], flat_recv[i + 1]});
    }
  }
#else
  (void)new_edges_vec;
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

bool ClusterCoordinator::GlobalAny(bool local_val) const {
#if DISTRIBUTED_HAS_MPI
  if (mpi_enabled_) {
    int in = local_val ? 1 : 0;
    int out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return out != 0;
  }
#endif
  return local_val;
}

} // namespace distributed

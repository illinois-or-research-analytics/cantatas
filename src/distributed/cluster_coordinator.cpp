#include "distributed/cluster_coordinator.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>

#if __has_include(<mpi.h>)
#define CANTATAS_HAS_MPI 1
#include <mpi.h>
#else
#define CANTATAS_HAS_MPI 0
#endif

namespace distributed {

namespace {
#if CANTATAS_HAS_MPI
// Largest element count passed to a single MPI call (int-sized counts).
constexpr size_t kMaxChunk = size_t{1} << 30;

template <typename T> MPI_Datatype Type();
template <> MPI_Datatype Type<int32_t>() { return MPI_INT32_T; }
template <> MPI_Datatype Type<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype Type<uint8_t>() { return MPI_UINT8_T; }

template <typename T> void BcastVector(std::vector<T> &v, size_t n) {
  v.resize(n);
  for (size_t off = 0; off < n; off += kMaxChunk) {
    int len = static_cast<int>(std::min(kMaxChunk, n - off));
    MPI_Bcast(v.data() + off, len, Type<T>(), 0, MPI_COMM_WORLD);
  }
}
#endif
} // namespace

ClusterCoordinator::~ClusterCoordinator() { Finalize(); }

void ClusterCoordinator::Init(int &argc, char **&argv) {
#if CANTATAS_HAS_MPI
  // Keep the transport choices from earlier versions unless the job sets them.
  setenv("OMPI_MCA_pml", "ob1", 0);
  setenv("OMPI_MCA_btl", "self,vader,tcp", 0);
  setenv("UCX_TLS", "sm,self,tcp", 0);
  setenv("UCX_NET_DEVICES", "all", 0);
  bool launched = std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr || std::getenv("PMIX_RANK") != nullptr ||
                  std::getenv("PMI_RANK") != nullptr || std::getenv("MPI_LOCALRANKID") != nullptr;
  int already = 0;
  MPI_Initialized(&already);
  if (already || launched) {
    if (!already) {
      int provided = 0;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
      initialized_by_me_ = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    enabled_ = world_size_ > 1;
    return;
  }
#endif
  (void)argc;
  (void)argv;
  rank_ = 0;
  world_size_ = 1;
  enabled_ = false;
}

void ClusterCoordinator::Finalize() {
#if CANTATAS_HAS_MPI
  if (initialized_by_me_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
    initialized_by_me_ = false;
  }
#endif
}

void ClusterCoordinator::Abort(int code) {
#if CANTATAS_HAS_MPI
  if (enabled_) {
    MPI_Abort(MPI_COMM_WORLD, code);
  }
#endif
  std::exit(code);
}

void ClusterCoordinator::Barrier() {
#if CANTATAS_HAS_MPI
  if (enabled_) MPI_Barrier(MPI_COMM_WORLD);
#endif
}

bool ClusterCoordinator::GlobalAny(bool local) {
#if CANTATAS_HAS_MPI
  if (enabled_) {
    int in = local ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return out != 0;
  }
#endif
  return local;
}

void ClusterCoordinator::BroadcastU64(uint64_t &value) {
#if CANTATAS_HAS_MPI
  if (enabled_) MPI_Bcast(&value, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#else
  (void)value;
#endif
}

void ClusterCoordinator::BroadcastBatch(AgentBatch &batch) {
#if CANTATAS_HAS_MPI
  if (!enabled_) return;
  int32_t header[4] = {batch.year, batch.first_id, batch.count, batch.next_author_id};
  MPI_Bcast(header, 4, MPI_INT32_T, 0, MPI_COMM_WORLD);
  batch.year = header[0];
  batch.first_id = header[1];
  batch.count = header[2];
  batch.next_author_id = header[3];
  const size_t n = static_cast<size_t>(batch.count);
  for (auto *v : {&batch.author, &batch.generator, &batch.cluster, &batch.out_degree, &batch.fitness_lag,
                  &batch.fitness_peak, &batch.fitness_duration, &batch.num_authors}) {
    BcastVector(*v, n);
  }
  for (auto *v : {&batch.pa_weight, &batch.fit_weight, &batch.num_authors_weight,
                  &batch.author_reputation_weight, &batch.alpha}) {
    BcastVector(*v, n);
  }
  BcastVector(batch.same_year_source, n);
#else
  (void)batch;
#endif
}

void ClusterCoordinator::AllgatherInts(const std::vector<int32_t> &local, std::vector<int32_t> &all) {
#if CANTATAS_HAS_MPI
  if (enabled_) {
    // Sizes as 64-bit values, then the data in rounds of at most kMaxChunk
    // elements per rank so no count overflows an int.
    long long mine = static_cast<long long>(local.size());
    std::vector<long long> sizes(world_size_);
    MPI_Allgather(&mine, 1, MPI_LONG_LONG, sizes.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    std::vector<size_t> starts(world_size_ + 1, 0);
    for (int r = 0; r < world_size_; ++r) starts[r + 1] = starts[r] + static_cast<size_t>(sizes[r]);
    all.resize(starts[world_size_]);
    long long max_size = 0;
    for (long long s : sizes) max_size = std::max(max_size, s);
    std::vector<int> counts(world_size_), displs(world_size_);
    std::vector<int32_t> recv;
    for (size_t off = 0; off < static_cast<size_t>(max_size); off += kMaxChunk / world_size_ + 1) {
      size_t chunk = kMaxChunk / world_size_ + 1;
      int total = 0;
      for (int r = 0; r < world_size_; ++r) {
        size_t have = static_cast<size_t>(sizes[r]);
        counts[r] = static_cast<int>(off < have ? std::min(chunk, have - off) : 0);
        displs[r] = total;
        total += counts[r];
      }
      recv.resize(static_cast<size_t>(total));
      int send_count = counts[rank_];
      MPI_Allgatherv(local.data() + std::min(off, local.size()), send_count, MPI_INT32_T, recv.data(),
                     counts.data(), displs.data(), MPI_INT32_T, MPI_COMM_WORLD);
      for (int r = 0; r < world_size_; ++r) {
        std::copy(recv.begin() + displs[r], recv.begin() + displs[r] + counts[r], all.begin() + starts[r] + off);
      }
    }
    return;
  }
#endif
  all = local;
}

void ClusterCoordinator::AllreduceMaxInts(std::vector<int32_t> &values) {
#if CANTATAS_HAS_MPI
  if (enabled_ && !values.empty()) {
    for (size_t off = 0; off < values.size(); off += kMaxChunk) {
      int len = static_cast<int>(std::min(kMaxChunk, values.size() - off));
      MPI_Allreduce(MPI_IN_PLACE, values.data() + off, len, MPI_INT32_T, MPI_MAX, MPI_COMM_WORLD);
    }
  }
#else
  (void)values;
#endif
}

} // namespace distributed

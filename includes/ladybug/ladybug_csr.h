#ifndef LADYBUG_CSR_H
#define LADYBUG_CSR_H

#include "ladybug_mmap.h"
#include <atomic>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace ladybug {

struct DeltaEdge {
  int32_t target; // Cited node v (receives incoming citation)
  int32_t source; // Citing node u (creates reference)
};

/*
 * LadyBugBidirectionalCSR: Embedded out-of-core bidirectional graph engine.
 * - Forward CSR (u -> v): Fast O(1) out-citation traversal and dynamic O(1) appends.
 * - Backward CSR / CSC (v <- u): Fast O(1) in-citation and reputation lookups.
 * - Delta Buffer: Lock-free thread-local buffer accumulating incoming citations during active year.
 * - Annual Superstep Merge: Merges delta incoming edges into Backward CSR in parallel at year boundary.
 */
class LadyBugBidirectionalCSR {
public:
  LadyBugBidirectionalCSR();
  ~LadyBugBidirectionalCSR();

  LadyBugBidirectionalCSR(const LadyBugBidirectionalCSR &) = delete;
  LadyBugBidirectionalCSR &operator=(const LadyBugBidirectionalCSR &) = delete;

  LadyBugBidirectionalCSR(LadyBugBidirectionalCSR &&) noexcept;
  LadyBugBidirectionalCSR &operator=(LadyBugBidirectionalCSR &&) noexcept;

  void Open(const std::string &db_dir, MmapMode mode,
            size_t initial_nodes = 0, size_t initial_edges = 0);
  void Close();
  void Sync(bool async = false);

  size_t GetNodeCount() const noexcept { return num_nodes_; }
  size_t GetEdgeCount() const noexcept { return num_edges_; }

  // Forward CSR (u -> v)
  inline std::span<const int32_t> GetOutNeighbors(int u) const noexcept {
    if (static_cast<size_t>(u) >= num_nodes_ || u < 0) {
      return std::span<const int32_t>();
    }
    uint64_t start = fwd_offsets_[u];
    uint64_t end = fwd_offsets_[u + 1];
    if (start >= end || end > fwd_edges_.size()) {
      return std::span<const int32_t>();
    }
    return fwd_edges_.subspan(start, end - start);
  }

  inline size_t GetOutDegree(int u) const noexcept {
    if (static_cast<size_t>(u) >= num_nodes_ || u < 0) {
      return 0;
    }
    return fwd_offsets_[u + 1] - fwd_offsets_[u];
  }

  // Backward CSR / CSC (v <- u)
  inline std::span<const int32_t> GetInNeighbors(int v) const noexcept {
    if (static_cast<size_t>(v) >= num_nodes_ || v < 0) {
      return std::span<const int32_t>();
    }
    uint64_t start = bwd_offsets_[v];
    uint64_t end = bwd_offsets_[v + 1];
    if (start >= end || end > bwd_edges_.size()) {
      return std::span<const int32_t>();
    }
    return bwd_edges_.subspan(start, end - start);
  }

  inline size_t GetInDegree(int v) const noexcept {
    if (static_cast<size_t>(v) >= num_nodes_ || v < 0) {
      return 0;
    }
    return bwd_offsets_[v + 1] - bwd_offsets_[v];
  }

  // Append new node with its out-edges (O(1) sequential append)
  void AppendNodeOutEdges(int u, std::span<const int32_t> targets);

  // Thread-local delta logging for incoming citations generated during active simulation year
  void RecordDeltaIncomingCitation(int thread_id, int32_t target_v, int32_t citing_u);
  void RecordDeltaBatch(int thread_id, std::span<const std::pair<int32_t, int32_t>> edges);

  // Initialize thread-local delta buffers for max_threads
  void SetThreadCount(int num_threads);

  // Annual Superstep Merge: Rebuilds / updates Backward CSR by merging accumulated delta buffers
  void MergeAnnualDeltaBuffer();

  // Ingest seed edgelist CSV into bidirectional CSR
  void IngestCSV(const std::string &edgelist_csv, size_t total_nodes);

  // Export to CSV
  void ExportCSV(const std::string &output_csv) const;

  // Direct buffer access for IPC / MPI transport
  const MmapBuffer<uint64_t> &GetFwdOffsets() const noexcept { return fwd_offsets_; }
  const MmapBuffer<int32_t> &GetFwdEdges() const noexcept { return fwd_edges_; }
  const MmapBuffer<uint64_t> &GetBwdOffsets() const noexcept { return bwd_offsets_; }
  const MmapBuffer<int32_t> &GetBwdEdges() const noexcept { return bwd_edges_; }

private:
  std::string db_dir_;
  size_t num_nodes_;
  size_t num_edges_;
  MmapMode mode_;

  // Forward CSR (u -> v)
  MmapBuffer<uint64_t> fwd_offsets_;
  MmapBuffer<int32_t> fwd_edges_;

  // Backward CSR / CSC (v <- u)
  MmapBuffer<uint64_t> bwd_offsets_;
  MmapBuffer<int32_t> bwd_edges_;

  // Thread-local delta buffers for incoming edge accumulation
  std::vector<std::vector<DeltaEdge>> thread_delta_buffers_;
  std::mutex delta_mutex_;

  void EnsureNodeCapacity(size_t node_count);
  void EnsureEdgeCapacity(size_t edge_count);
};

} // namespace ladybug

#endif // LADYBUG_CSR_H

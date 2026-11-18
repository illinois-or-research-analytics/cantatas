#ifndef LADYBUG_CSR_H
#define LADYBUG_CSR_H

#include "ladybug/ladybug_column.h"
#include <cstdint>
#include <span>

namespace ladybug {

/*
 * BidirectionalCSR: the citation graph, in both directions.
 *
 *   Forward  (u -> v): the papers u cites.      Rows are appended once per year.
 *   Backward (v <- u): the papers that cite v.  New in-edges are merged once per year.
 *
 * Every rank holds an identical copy. The model reads only this structure, and
 * during a simulated year it is read-only (bulk-synchronous: agents of year Y
 * see the graph as it stood at the start of Y).
 */
class BidirectionalCSR {
public:
  BidirectionalCSR();

  size_t NumNodes() const noexcept { return num_nodes_; }
  uint64_t NumEdges() const noexcept { return fwd_targets_.size(); }

  std::span<const int32_t> Out(int32_t u) const noexcept {
    return {fwd_targets_.data() + fwd_offsets_[u],
            static_cast<size_t>(fwd_offsets_[u + 1] - fwd_offsets_[u])};
  }
  std::span<const int32_t> In(int32_t v) const noexcept {
    return {bwd_sources_.data() + bwd_offsets_[v],
            static_cast<size_t>(bwd_offsets_[v + 1] - bwd_offsets_[v])};
  }
  uint32_t OutDegree(int32_t u) const noexcept {
    return static_cast<uint32_t>(fwd_offsets_[u + 1] - fwd_offsets_[u]);
  }
  uint32_t InDegree(int32_t v) const noexcept {
    return static_cast<uint32_t>(bwd_offsets_[v + 1] - bwd_offsets_[v]);
  }

  // Reserves address space for the projected final size (no memory is used
  // until written).
  void Reserve(size_t nodes, uint64_t edges);

  // Builds both directions from an edge list src[i] -> dst[i] over nodes
  // [0, num_nodes). Row contents keep input order. Duplicate edges are kept.
  void Build(size_t num_nodes, std::span<const int32_t> src, std::span<const int32_t> dst);

  // Appends rows for nodes [NumNodes(), NumNodes() + count). row_ptr holds
  // count + 1 relative offsets into targets. New nodes start with no in-edges.
  void AppendRows(size_t count, std::span<const uint64_t> row_ptr,
                  std::span<const int32_t> targets);

  // Adds in-edges sources[i] -> targets[i] to the backward direction. Within
  // each row the new entries follow input order, so the result is
  // deterministic for a deterministic input order.
  void MergeInEdges(std::span<const int32_t> sources, std::span<const int32_t> targets);

  // Raw columns, for snapshots.
  Column<uint64_t> &FwdOffsets() noexcept { return fwd_offsets_; }
  Column<int32_t> &FwdTargets() noexcept { return fwd_targets_; }
  Column<uint64_t> &BwdOffsets() noexcept { return bwd_offsets_; }
  Column<int32_t> &BwdSources() noexcept { return bwd_sources_; }
  const Column<uint64_t> &FwdOffsets() const noexcept { return fwd_offsets_; }
  const Column<int32_t> &FwdTargets() const noexcept { return fwd_targets_; }
  const Column<uint64_t> &BwdOffsets() const noexcept { return bwd_offsets_; }
  const Column<int32_t> &BwdSources() const noexcept { return bwd_sources_; }

  // Re-derives NumNodes() after the columns were loaded from a snapshot.
  void SyncNodeCountFromColumns();

private:
  size_t num_nodes_ = 0;
  Column<uint64_t> fwd_offsets_;
  Column<int32_t> fwd_targets_;
  Column<uint64_t> bwd_offsets_;
  Column<int32_t> bwd_sources_;
};

} // namespace ladybug

#endif // LADYBUG_CSR_H

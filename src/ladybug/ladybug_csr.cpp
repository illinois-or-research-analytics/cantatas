#include "ladybug/ladybug_csr.h"
#include "ladybug/ladybug_parallel.h"

#include <omp.h>

#include <cstring>
#include <stdexcept>

namespace ladybug {

BidirectionalCSR::BidirectionalCSR() {
  fwd_offsets_.Resize(1, 0);
  bwd_offsets_.Resize(1, 0);
}

void BidirectionalCSR::Reserve(size_t nodes, uint64_t edges) {
  fwd_offsets_.Reserve(nodes + 1);
  bwd_offsets_.Reserve(nodes + 1);
  fwd_targets_.Reserve(edges);
  bwd_sources_.Reserve(edges);
}

void BidirectionalCSR::Build(size_t num_nodes, std::span<const int32_t> src,
                             std::span<const int32_t> dst) {
  if (src.size() != dst.size()) {
    throw std::invalid_argument("BidirectionalCSR::Build: src/dst size mismatch");
  }
  const size_t m = src.size();
  num_nodes_ = num_nodes;
  fwd_offsets_.Clear();
  bwd_offsets_.Clear();
  fwd_offsets_.Resize(num_nodes + 1, 0);
  bwd_offsets_.Resize(num_nodes + 1, 0);

  uint64_t *fwd_off = fwd_offsets_.data();
  uint64_t *bwd_off = bwd_offsets_.data();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; ++i) {
    __atomic_fetch_add(&fwd_off[src[i]], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&bwd_off[dst[i]], 1, __ATOMIC_RELAXED);
  }
  ExclusiveScanInPlace(fwd_off, num_nodes + 1);
  ExclusiveScanInPlace(bwd_off, num_nodes + 1);

  fwd_targets_.Clear();
  bwd_sources_.Clear();
  fwd_targets_.ResizeUninitialized(m);
  bwd_sources_.ResizeUninitialized(m);

  // Placement walks the edge list in order so rows keep input order; the two
  // directions are filled concurrently.
#pragma omp parallel sections
  {
#pragma omp section
    {
      Column<uint64_t> cursor;
      cursor.ResizeUninitialized(num_nodes);
      std::memcpy(cursor.data(), fwd_off, num_nodes * sizeof(uint64_t));
      int32_t *targets = fwd_targets_.data();
      for (size_t i = 0; i < m; ++i) {
        targets[cursor[src[i]]++] = dst[i];
      }
    }
#pragma omp section
    {
      Column<uint64_t> cursor;
      cursor.ResizeUninitialized(num_nodes);
      std::memcpy(cursor.data(), bwd_off, num_nodes * sizeof(uint64_t));
      int32_t *sources = bwd_sources_.data();
      for (size_t i = 0; i < m; ++i) {
        sources[cursor[dst[i]]++] = src[i];
      }
    }
  }
}

void BidirectionalCSR::AppendRows(size_t count, std::span<const uint64_t> row_ptr,
                                  std::span<const int32_t> targets) {
  if (row_ptr.size() != count + 1) {
    throw std::invalid_argument("BidirectionalCSR::AppendRows: row_ptr must have count + 1 entries");
  }
  const uint64_t base = fwd_targets_.size();
  fwd_targets_.Append(targets);
  fwd_offsets_.Reserve(fwd_offsets_.size() + count);
  bwd_offsets_.Reserve(bwd_offsets_.size() + count);
  const uint64_t bwd_end = bwd_offsets_.back();
  for (size_t k = 1; k <= count; ++k) {
    fwd_offsets_.PushBack(base + row_ptr[k]);
    bwd_offsets_.PushBack(bwd_end);
  }
  num_nodes_ += count;
}

void BidirectionalCSR::MergeInEdges(std::span<const int32_t> sources,
                                    std::span<const int32_t> targets) {
  if (sources.size() != targets.size()) {
    throw std::invalid_argument("BidirectionalCSR::MergeInEdges: size mismatch");
  }
  const size_t n = num_nodes_;
  const size_t m = targets.size();
  if (m == 0) {
    return;
  }

  // 1. Count new in-edges per node.
  Column<uint32_t> added;
  added.Resize(n, 0);
  uint32_t *added_ptr = added.data();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; ++i) {
    __atomic_fetch_add(&added_ptr[targets[i]], 1u, __ATOMIC_RELAXED);
  }

  // 2. New offsets.
  Column<uint64_t> new_offsets;
  new_offsets.ResizeUninitialized(n + 1);
  const uint64_t *old_off = bwd_offsets_.data();
#pragma omp parallel for schedule(static)
  for (size_t v = 0; v < n; ++v) {
    new_offsets[v] = (old_off[v + 1] - old_off[v]) + added_ptr[v];
  }
  new_offsets[n] = 0;
  ExclusiveScanInPlace(new_offsets.data(), n + 1);

  // 3. Copy existing rows into the new layout (double buffer).
  Column<int32_t> new_sources;
  new_sources.ResizeUninitialized(bwd_sources_.size() + m);
  const int32_t *old_src = bwd_sources_.data();
  int32_t *dst_src = new_sources.data();
#pragma omp parallel for schedule(dynamic, 4096)
  for (size_t v = 0; v < n; ++v) {
    uint64_t len = old_off[v + 1] - old_off[v];
    if (len > 0) {
      std::memcpy(dst_src + new_offsets[v], old_src + old_off[v], len * sizeof(int32_t));
    }
  }

  // 4. Append the new in-edges after each row's existing entries, in input order.
#pragma omp parallel for schedule(static)
  for (size_t v = 0; v < n; ++v) {
    added_ptr[v] = static_cast<uint32_t>(old_off[v + 1] - old_off[v]);
  }
  for (size_t i = 0; i < m; ++i) {
    int32_t v = targets[i];
    dst_src[new_offsets[v] + added_ptr[v]++] = sources[i];
  }

  bwd_sources_ = std::move(new_sources);
  bwd_offsets_ = std::move(new_offsets);
}

void BidirectionalCSR::SyncNodeCountFromColumns() {
  if (fwd_offsets_.size() == 0 || bwd_offsets_.size() != fwd_offsets_.size()) {
    throw std::runtime_error("BidirectionalCSR: inconsistent offset columns");
  }
  num_nodes_ = fwd_offsets_.size() - 1;
  if (fwd_offsets_.back() != fwd_targets_.size() || bwd_offsets_.back() != bwd_sources_.size()) {
    throw std::runtime_error("BidirectionalCSR: offsets do not match edge columns");
  }
}

} // namespace ladybug

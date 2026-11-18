#include "ladybug/ladybug_db.h"
#include "ladybug/ladybug_parallel.h"

#include <omp.h>

#include <algorithm>
#include <cstring>

namespace ladybug {

void ClusterIndex::Rebuild(std::span<const int32_t> cluster_of_node, size_t num_nodes) {
  num_nodes = std::min(num_nodes, cluster_of_node.size());
  int32_t max_cluster = -1;
#pragma omp parallel for reduction(max : max_cluster) schedule(static)
  for (size_t v = 0; v < num_nodes; ++v) {
    max_cluster = std::max(max_cluster, cluster_of_node[v]);
  }
  const size_t num_clusters = static_cast<size_t>(max_cluster + 1);

  offsets_.Clear();
  offsets_.Resize(num_clusters + 1, 0);
  uint64_t *off = offsets_.data();
#pragma omp parallel for schedule(static)
  for (size_t v = 0; v < num_nodes; ++v) {
    int32_t c = cluster_of_node[v];
    if (c >= 0) {
      __atomic_fetch_add(&off[c], 1, __ATOMIC_RELAXED);
    }
  }
  const uint64_t total = ExclusiveScanInPlace(off, num_clusters + 1);

  members_.Clear();
  members_.ResizeUninitialized(total);
  Column<uint64_t> cursor;
  cursor.ResizeUninitialized(num_clusters);
  if (num_clusters > 0) {
    std::memcpy(cursor.data(), off, num_clusters * sizeof(uint64_t));
  }
  // Sequential so each member list is in increasing node order.
  for (size_t v = 0; v < num_nodes; ++v) {
    int32_t c = cluster_of_node[v];
    if (c >= 0) {
      members_[cursor[c]++] = static_cast<int32_t>(v);
    }
  }
}

void LadyBugDB::ReserveFinalSize(size_t nodes_hint, uint64_t edges_hint) {
  csr.Reserve(nodes_hint, edges_hint);
  nodes.ForEach([&](const char *, auto &column) { column.Reserve(nodes_hint); });
}

} // namespace ladybug

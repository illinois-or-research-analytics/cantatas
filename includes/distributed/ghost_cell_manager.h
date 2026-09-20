#ifndef GHOST_CELL_MANAGER_H
#define GHOST_CELL_MANAGER_H

#include "ladybug/ladybug_csr.h"
#include "ladybug/ladybug_node_store.h"
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace distributed {

struct GhostNodeData {
  int32_t node_id;
  uint16_t year;
  int32_t author_id;
  int32_t cluster_id;
  uint32_t in_degree;
  std::vector<int32_t> out_edges;
};

/*
 * GhostCellManager: Replicates top high-degree foundational papers locally as read-only
 * "ghost cells" and maintains a local 2-hop Ghost Halo cache for frequently traversed
 * cross-rank pathways, eliminating network RPC round-trips for high-traffic hubs.
 */
class GhostCellManager {
public:
  GhostCellManager();
  ~GhostCellManager();

  // Register foundational hub nodes as ghost cells
  void RegisterHubs(const std::vector<int32_t> &hub_node_ids,
                    const ladybug::LadyBugNodeStore &node_store,
                    const ladybug::LadyBugBidirectionalCSR &csr);

  // Cache 2-hop cross-disciplinary pathway
  void CacheGhostHalo(int32_t node_id, std::span<const int32_t> nbrs);

  // Lookups
  bool IsGhost(int32_t node_id) const noexcept;
  std::span<const int32_t> GetGhostOutNeighbors(int32_t node_id) const noexcept;
  uint32_t GetGhostInDegree(int32_t node_id) const noexcept;
  uint16_t GetGhostYear(int32_t node_id) const noexcept;

  size_t GetGhostCount() const noexcept { return ghost_nodes_.size(); }
  size_t GetHaloCount() const noexcept { return halo_cache_.size(); }

  void Clear();

private:
  std::unordered_map<int32_t, GhostNodeData> ghost_nodes_;
  std::unordered_map<int32_t, std::vector<int32_t>> halo_cache_;
  static const std::vector<int32_t> empty_edges_;
};

} // namespace distributed

#endif // GHOST_CELL_MANAGER_H

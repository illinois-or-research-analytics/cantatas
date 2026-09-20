#include "distributed/ghost_cell_manager.h"

namespace distributed {

const std::vector<int32_t> GhostCellManager::empty_edges_ = {};

GhostCellManager::GhostCellManager() {}

GhostCellManager::~GhostCellManager() {
  Clear();
}

void GhostCellManager::Clear() {
  ghost_nodes_.clear();
  halo_cache_.clear();
}

void GhostCellManager::RegisterHubs(
    const std::vector<int32_t> &hub_node_ids,
    const ladybug::LadyBugNodeStore &node_store,
    const ladybug::LadyBugBidirectionalCSR &csr) {
  for (int32_t u : hub_node_ids) {
    if (u < 0 || static_cast<size_t>(u) >= node_store.GetNodeCount()) {
      continue;
    }
    GhostNodeData ghost_data;
    ghost_data.node_id = u;
    ghost_data.year = node_store.GetYear(u);
    ghost_data.author_id = node_store.GetAuthorId(u);
    ghost_data.cluster_id = node_store.GetClusterId(u);
    ghost_data.in_degree = node_store.GetInDegree(u);

    auto out_nbrs = csr.GetOutNeighbors(u);
    ghost_data.out_edges.assign(out_nbrs.begin(), out_nbrs.end());

    ghost_nodes_[u] = std::move(ghost_data);
  }
}

void GhostCellManager::CacheGhostHalo(int32_t node_id,
                                      std::span<const int32_t> nbrs) {
  halo_cache_[node_id].assign(nbrs.begin(), nbrs.end());
}

bool GhostCellManager::IsGhost(int32_t node_id) const noexcept {
  return ghost_nodes_.contains(node_id);
}

std::span<const int32_t>
GhostCellManager::GetGhostOutNeighbors(int32_t node_id) const noexcept {
  auto it = ghost_nodes_.find(node_id);
  if (it != ghost_nodes_.end()) {
    return it->second.out_edges;
  }
  auto hit = halo_cache_.find(node_id);
  if (hit != halo_cache_.end()) {
    return hit->second;
  }
  return empty_edges_;
}

uint32_t GhostCellManager::GetGhostInDegree(int32_t node_id) const noexcept {
  auto it = ghost_nodes_.find(node_id);
  if (it != ghost_nodes_.end()) {
    return it->second.in_degree;
  }
  return 0;
}

uint16_t GhostCellManager::GetGhostYear(int32_t node_id) const noexcept {
  auto it = ghost_nodes_.find(node_id);
  if (it != ghost_nodes_.end()) {
    return it->second.year;
  }
  return 0;
}

} // namespace distributed

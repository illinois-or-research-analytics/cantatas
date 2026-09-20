#ifndef LADYBUG_CLUSTER_INDEX_H
#define LADYBUG_CLUSTER_INDEX_H

#include "ladybug_mmap.h"
#include "ladybug_node_store.h"
#include <algorithm>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ladybug {

/*
 * LadyBugClusterIndex: Manages Leiden community partition mappings and fast inverted lookups.
 * Supports year-bounded lower_bound cluster size lookups in O(log K) time for theta thresholding.
 */
class LadyBugClusterIndex {
public:
  LadyBugClusterIndex();
  ~LadyBugClusterIndex();

  void Open(const std::string &db_dir, MmapMode mode);
  void Close();
  void Sync(bool async = false);

  void AddNodeToCluster(int node, int cluster_id);
  const std::vector<int32_t> &GetClusterNodes(int cluster_id) const;
  int GetClusterSize(int cluster_id, const LadyBugNodeStore &node_store, int current_year = -1) const;

  void IngestCSV(const std::string &cluster_csv);

private:
  std::string db_dir_;
  MmapMode mode_;
  std::unordered_map<int32_t, std::vector<int32_t>> cluster_to_nodes_;
  static const std::vector<int32_t> empty_cluster_;
};

} // namespace ladybug

#endif // LADYBUG_CLUSTER_INDEX_H

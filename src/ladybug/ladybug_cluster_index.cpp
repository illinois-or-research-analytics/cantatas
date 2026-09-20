#include "ladybug/ladybug_cluster_index.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ladybug {

const std::vector<int32_t> LadyBugClusterIndex::empty_cluster_ = {};

LadyBugClusterIndex::LadyBugClusterIndex() : mode_(MmapMode::ReadWrite) {}

LadyBugClusterIndex::~LadyBugClusterIndex() {
  Close();
}

void LadyBugClusterIndex::Open(const std::string &db_dir, MmapMode mode) {
  Close();
  db_dir_ = db_dir;
  mode_ = mode;
  cluster_to_nodes_.clear();
}

void LadyBugClusterIndex::Close() {
  cluster_to_nodes_.clear();
}

void LadyBugClusterIndex::Sync(bool /*async*/) {
  // Can serialize cluster map to disk if needed
}

void LadyBugClusterIndex::AddNodeToCluster(int node, int cluster_id) {
  if (cluster_id >= 0) {
    cluster_to_nodes_[cluster_id].push_back(node);
  }
}

const std::vector<int32_t> &LadyBugClusterIndex::GetClusterNodes(int cluster_id) const {
  auto it = cluster_to_nodes_.find(cluster_id);
  if (it != cluster_to_nodes_.end()) {
    return it->second;
  }
  return empty_cluster_;
}

int LadyBugClusterIndex::GetClusterSize(int cluster_id, const LadyBugNodeStore &node_store, int current_year) const {
  auto it = cluster_to_nodes_.find(cluster_id);
  if (it == cluster_to_nodes_.end()) {
    return 0;
  }
  const auto &nodes = it->second;
  if (current_year == -1) {
    return static_cast<int>(nodes.size());
  }
  auto comp = [&node_store](int node, int yr) {
    return node_store.GetYear(node) < yr;
  };
  auto lower = std::lower_bound(nodes.begin(), nodes.end(), current_year, comp);
  return static_cast<int>(lower - nodes.begin());
}

void LadyBugClusterIndex::IngestCSV(const std::string &cluster_csv) {
  std::ifstream file(cluster_csv);
  if (!file.is_open()) return;

  std::string line;
  // Check header
  if (!std::getline(file, line)) return;

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string val1, val2;
    if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
      if (!val1.empty() && val1.back() == '\r') val1.pop_back();
      if (!val2.empty() && val2.back() == '\r') val2.pop_back();
      int node_id = std::stoi(val1);
      int cluster_id = std::stoi(val2);
      AddNodeToCluster(node_id, cluster_id);
    }
  }
}

} // namespace ladybug

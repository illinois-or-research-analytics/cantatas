#include "ladybug/ladybug_csr.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>

namespace ladybug {

namespace {
char DetectDelimiter(const std::string &filepath) {
  std::ifstream f(filepath);
  std::string line;
  if (std::getline(f, line)) {
    if (line.find(',') != std::string::npos) return ',';
    if (line.find('\t') != std::string::npos) return '\t';
    if (line.find(' ') != std::string::npos) return ' ';
  }
  return ',';
}
} // namespace

LadyBugBidirectionalCSR::LadyBugBidirectionalCSR()
    : num_nodes_(0), num_edges_(0), mode_(MmapMode::ReadWrite) {
  SetThreadCount(omp_get_max_threads());
}

LadyBugBidirectionalCSR::~LadyBugBidirectionalCSR() {
  Close();
}

LadyBugBidirectionalCSR::LadyBugBidirectionalCSR(LadyBugBidirectionalCSR &&other) noexcept
    : db_dir_(std::move(other.db_dir_)),
      num_nodes_(other.num_nodes_),
      num_edges_(other.num_edges_),
      mode_(other.mode_),
      fwd_offsets_(std::move(other.fwd_offsets_)),
      fwd_edges_(std::move(other.fwd_edges_)),
      bwd_offsets_(std::move(other.bwd_offsets_)),
      bwd_edges_(std::move(other.bwd_edges_)),
      thread_delta_buffers_(std::move(other.thread_delta_buffers_)) {
  other.num_nodes_ = 0;
  other.num_edges_ = 0;
}

LadyBugBidirectionalCSR &LadyBugBidirectionalCSR::operator=(LadyBugBidirectionalCSR &&other) noexcept {
  if (this != &other) {
    Close();
    db_dir_ = std::move(other.db_dir_);
    num_nodes_ = other.num_nodes_;
    num_edges_ = other.num_edges_;
    mode_ = other.mode_;
    fwd_offsets_ = std::move(other.fwd_offsets_);
    fwd_edges_ = std::move(other.fwd_edges_);
    bwd_offsets_ = std::move(other.bwd_offsets_);
    bwd_edges_ = std::move(other.bwd_edges_);
    thread_delta_buffers_ = std::move(other.thread_delta_buffers_);

    other.num_nodes_ = 0;
    other.num_edges_ = 0;
  }
  return *this;
}

void LadyBugBidirectionalCSR::Open(const std::string &db_dir, MmapMode mode,
                                   size_t initial_nodes, size_t initial_edges) {
  Close();
  db_dir_ = db_dir;
  mode_ = mode;

  std::filesystem::create_directories(db_dir_);

  size_t offset_cap = std::max(initial_nodes + 1, static_cast<size_t>(1024));
  size_t edge_cap = std::max(initial_edges, static_cast<size_t>(4096));

  fwd_offsets_.Open(db_dir_ + "/csr_fwd_offsets.bin", mode_, offset_cap);
  fwd_edges_.Open(db_dir_ + "/csr_fwd_edges.bin", mode_, edge_cap);
  bwd_offsets_.Open(db_dir_ + "/csr_bwd_offsets.bin", mode_, offset_cap);
  bwd_edges_.Open(db_dir_ + "/csr_bwd_edges.bin", mode_, edge_cap);

  num_nodes_ = 0;
  num_edges_ = 0;
  SetThreadCount(omp_get_max_threads());
}

void LadyBugBidirectionalCSR::Close() {
  Sync(false);
  fwd_offsets_.Close();
  fwd_edges_.Close();
  bwd_offsets_.Close();
  bwd_edges_.Close();
  for (auto &buf : thread_delta_buffers_) {
    buf.clear();
  }
  num_nodes_ = 0;
  num_edges_ = 0;
}

void LadyBugBidirectionalCSR::Sync(bool async) {
  fwd_offsets_.Sync(async);
  fwd_edges_.Sync(async);
  bwd_offsets_.Sync(async);
  bwd_edges_.Sync(async);
}

void LadyBugBidirectionalCSR::SetThreadCount(int num_threads) {
  int count = std::max(1, num_threads);
  thread_delta_buffers_.resize(count);
  for (auto &buf : thread_delta_buffers_) {
    buf.reserve(8192);
  }
}

void LadyBugBidirectionalCSR::EnsureNodeCapacity(size_t node_count) {
  size_t required_offsets = node_count + 1;
  if (required_offsets > fwd_offsets_.capacity()) {
    size_t new_cap = std::max(required_offsets, fwd_offsets_.capacity() * 2);
    fwd_offsets_.Reserve(new_cap);
    bwd_offsets_.Reserve(new_cap);
  }
}

void LadyBugBidirectionalCSR::EnsureEdgeCapacity(size_t edge_count) {
  if (edge_count > fwd_edges_.capacity()) {
    size_t new_cap = std::max(edge_count, fwd_edges_.capacity() * 2);
    fwd_edges_.Reserve(new_cap);
    bwd_edges_.Reserve(new_cap);
  }
}

void LadyBugBidirectionalCSR::AppendNodeOutEdges(int u, std::span<const int32_t> targets) {
  size_t node_idx = static_cast<size_t>(u);
  EnsureNodeCapacity(node_idx + 1);

  // If there are skipped nodes between num_nodes_ and u, initialize their offsets
  while (num_nodes_ <= node_idx) {
    fwd_offsets_[num_nodes_] = fwd_edges_.size();
    bwd_offsets_[num_nodes_] = bwd_edges_.size();
    num_nodes_++;
  }

  // Append targets to forward edges
  if (!targets.empty()) {
    EnsureEdgeCapacity(fwd_edges_.size() + targets.size());
    fwd_edges_.AppendSpan(targets);
    num_edges_ += targets.size();
  }

  fwd_offsets_[node_idx + 1] = fwd_edges_.size();
  bwd_offsets_[node_idx + 1] = bwd_edges_.size();
  num_nodes_ = node_idx + 1;

  fwd_offsets_.SetSize(num_nodes_ + 1);
  bwd_offsets_.SetSize(num_nodes_ + 1);
}

void LadyBugBidirectionalCSR::RecordDeltaIncomingCitation(int thread_id, int32_t target_v, int32_t citing_u) {
  if (thread_id < 0 || static_cast<size_t>(thread_id) >= thread_delta_buffers_.size()) {
    std::lock_guard<std::mutex> lock(delta_mutex_);
    if (thread_delta_buffers_.empty()) thread_delta_buffers_.resize(1);
    thread_delta_buffers_[0].push_back({target_v, citing_u});
    return;
  }
  thread_delta_buffers_[thread_id].push_back({target_v, citing_u});
}

void LadyBugBidirectionalCSR::RecordDeltaBatch(int thread_id,
                                               std::span<const std::pair<int32_t, int32_t>> edges) {
  if (thread_id < 0 || static_cast<size_t>(thread_id) >= thread_delta_buffers_.size()) {
    std::lock_guard<std::mutex> lock(delta_mutex_);
    if (thread_delta_buffers_.empty()) thread_delta_buffers_.resize(1);
    for (const auto &e : edges) {
      thread_delta_buffers_[0].push_back({e.second, e.first}); // e.second is target, e.first is citing
    }
    return;
  }
  auto &buf = thread_delta_buffers_[thread_id];
  for (const auto &e : edges) {
    buf.push_back({e.second, e.first});
  }
}

void LadyBugBidirectionalCSR::MergeAnnualDeltaBuffer() {
  // 1. Calculate total delta count
  size_t total_deltas = 0;
  for (const auto &buf : thread_delta_buffers_) {
    total_deltas += buf.size();
  }

  if (total_deltas == 0) {
    return;
  }

  // 2. Count delta incoming edges per node v
  std::vector<uint32_t> delta_in_degree(num_nodes_ + 1, 0);
  for (const auto &buf : thread_delta_buffers_) {
    for (const auto &d : buf) {
      if (d.target >= 0 && static_cast<size_t>(d.target) < num_nodes_) {
        delta_in_degree[d.target]++;
      }
    }
  }

  // 3. Construct new Backward CSR layout
  size_t new_total_bwd_edges = bwd_edges_.size() + total_deltas;
  EnsureEdgeCapacity(new_total_bwd_edges);
  EnsureNodeCapacity(num_nodes_);

  // Compute new offsets and insertion write pointers
  std::vector<uint64_t> new_bwd_offsets(num_nodes_ + 1, 0);
  std::vector<uint64_t> write_pos(num_nodes_ + 1, 0);

  uint64_t running_offset = 0;
  for (size_t v = 0; v < num_nodes_; ++v) {
    new_bwd_offsets[v] = running_offset;
    write_pos[v] = running_offset;
    size_t old_degree = bwd_offsets_[v + 1] - bwd_offsets_[v];
    running_offset += old_degree + delta_in_degree[v];
  }
  new_bwd_offsets[num_nodes_] = running_offset;

  // Allocate temporary buffer for merged bwd edges
  std::vector<int32_t> merged_bwd_edges(running_offset);

  // Copy existing historical in-edges into new layout
#pragma omp parallel for schedule(dynamic, 1024)
  for (size_t v = 0; v < num_nodes_; ++v) {
    uint64_t old_start = bwd_offsets_[v];
    uint64_t old_end = bwd_offsets_[v + 1];
    uint64_t pos = write_pos[v];
    for (uint64_t i = old_start; i < old_end; ++i) {
      merged_bwd_edges[pos++] = bwd_edges_[i];
    }
    write_pos[v] = pos; // points to start of delta append region
  }

  // Append delta incoming edges to each node v
  for (const auto &buf : thread_delta_buffers_) {
    for (const auto &d : buf) {
      if (d.target >= 0 && static_cast<size_t>(d.target) < num_nodes_) {
        uint64_t pos = write_pos[d.target]++;
        merged_bwd_edges[pos] = d.source;
      }
    }
  }

  // Copy back to memory-mapped bwd_edges_ and bwd_offsets_
  bwd_edges_.SetSize(running_offset);
  std::memcpy(bwd_edges_.data(), merged_bwd_edges.data(), running_offset * sizeof(int32_t));

  bwd_offsets_.SetSize(num_nodes_ + 1);
  std::memcpy(bwd_offsets_.data(), new_bwd_offsets.data(), (num_nodes_ + 1) * sizeof(uint64_t));

  // Clear delta buffers
  for (auto &buf : thread_delta_buffers_) {
    buf.clear();
  }

  Sync(false);
}

void LadyBugBidirectionalCSR::IngestCSV(const std::string &edgelist_csv, size_t total_nodes) {
  char delim = DetectDelimiter(edgelist_csv);
  std::ifstream file(edgelist_csv);
  if (!file.is_open()) {
    throw std::runtime_error("LadyBugBidirectionalCSR::IngestCSV failed to open " + edgelist_csv);
  }

  std::string line;
  if (!std::getline(file, line)) {
    return;
  }

  struct EdgePair {
    int32_t source;
    int32_t target;
  };

  std::vector<EdgePair> edges;
  edges.reserve(262144);

  int max_node_id = static_cast<int>(total_nodes > 0 ? total_nodes - 1 : 0);

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string val1, val2;
    if (std::getline(ss, val1, delim) && std::getline(ss, val2, delim)) {
      if (!val1.empty() && val1.back() == '\r') val1.pop_back();
      if (!val2.empty() && val2.back() == '\r') val2.pop_back();
      int32_t u = std::stoi(val1);
      int32_t v = std::stoi(val2);
      edges.push_back({u, v});
      max_node_id = std::max(max_node_id, std::max(u, v));
    }
  }

  size_t n_nodes = std::max(total_nodes, static_cast<size_t>(max_node_id + 1));
  size_t n_edges = edges.size();

  EnsureNodeCapacity(n_nodes);
  EnsureEdgeCapacity(n_edges);

  // 1. Build Forward CSR
  std::vector<uint32_t> out_degrees(n_nodes + 1, 0);
  for (const auto &e : edges) {
    if (e.source >= 0 && static_cast<size_t>(e.source) < n_nodes) {
      out_degrees[e.source]++;
    }
  }

  fwd_offsets_.SetSize(n_nodes + 1);
  uint64_t fwd_running = 0;
  std::vector<uint64_t> fwd_write_pos(n_nodes + 1, 0);
  for (size_t u = 0; u < n_nodes; ++u) {
    fwd_offsets_[u] = fwd_running;
    fwd_write_pos[u] = fwd_running;
    fwd_running += out_degrees[u];
  }
  fwd_offsets_[n_nodes] = fwd_running;

  fwd_edges_.SetSize(fwd_running);
  for (const auto &e : edges) {
    if (e.source >= 0 && static_cast<size_t>(e.source) < n_nodes) {
      uint64_t pos = fwd_write_pos[e.source]++;
      fwd_edges_[pos] = e.target;
    }
  }

  // 2. Build Backward CSR (CSC)
  std::vector<uint32_t> in_degrees(n_nodes + 1, 0);
  for (const auto &e : edges) {
    if (e.target >= 0 && static_cast<size_t>(e.target) < n_nodes) {
      in_degrees[e.target]++;
    }
  }

  bwd_offsets_.SetSize(n_nodes + 1);
  uint64_t bwd_running = 0;
  std::vector<uint64_t> bwd_write_pos(n_nodes + 1, 0);
  for (size_t v = 0; v < n_nodes; ++v) {
    bwd_offsets_[v] = bwd_running;
    bwd_write_pos[v] = bwd_running;
    bwd_running += in_degrees[v];
  }
  bwd_offsets_[n_nodes] = bwd_running;

  bwd_edges_.SetSize(bwd_running);
  for (const auto &e : edges) {
    if (e.target >= 0 && static_cast<size_t>(e.target) < n_nodes) {
      uint64_t pos = bwd_write_pos[e.target]++;
      bwd_edges_[pos] = e.source;
    }
  }

  num_nodes_ = n_nodes;
  num_edges_ = n_edges;

  Sync(false);
}

void LadyBugBidirectionalCSR::ExportCSV(const std::string &output_csv) const {
  std::ofstream out(output_csv);
  if (!out.is_open()) {
    throw std::runtime_error("LadyBugBidirectionalCSR::ExportCSV failed to open " + output_csv);
  }

  out << "source,target\n";
  for (size_t u = 0; u < num_nodes_; ++u) {
    uint64_t start = fwd_offsets_[u];
    uint64_t end = fwd_offsets_[u + 1];
    for (uint64_t i = start; i < end; ++i) {
      out << u << "," << fwd_edges_[i] << "\n";
    }
  }
}

} // namespace ladybug

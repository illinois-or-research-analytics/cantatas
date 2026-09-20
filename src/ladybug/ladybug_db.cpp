#include "ladybug/ladybug_db.h"
#include "utils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace ladybug {

LadyBugDB::LadyBugDB() : mode_(MmapMode::ReadWrite) {}

LadyBugDB::~LadyBugDB() {
  Close();
}

LadyBugDB::LadyBugDB(LadyBugDB &&other) noexcept
    : db_dir_(std::move(other.db_dir_)),
      mode_(other.mode_),
      manifest_(other.manifest_),
      node_store_(std::move(other.node_store_)),
      csr_(std::move(other.csr_)),
      cluster_index_(std::move(other.cluster_index_)),
      author_store_(std::move(other.author_store_)),
      journal_(std::move(other.journal_)),
      num_authors_bag_vec_(std::move(other.num_authors_bag_vec_)) {}

LadyBugDB &LadyBugDB::operator=(LadyBugDB &&other) noexcept {
  if (this != &other) {
    Close();
    db_dir_ = std::move(other.db_dir_);
    mode_ = other.mode_;
    manifest_ = other.manifest_;
    node_store_ = std::move(other.node_store_);
    csr_ = std::move(other.csr_);
    cluster_index_ = std::move(other.cluster_index_);
    author_store_ = std::move(other.author_store_);
    journal_ = std::move(other.journal_);
    num_authors_bag_vec_ = std::move(other.num_authors_bag_vec_);
  }
  return *this;
}

void LadyBugDB::Open(const std::string &db_dir, MmapMode mode,
                     size_t initial_nodes, size_t initial_edges,
                     int author_max_lifetime) {
  Close();
  db_dir_ = db_dir;
  mode_ = mode;

  std::filesystem::create_directories(db_dir_);

  node_store_.Open(db_dir_ + "/nodes", mode, initial_nodes);
  csr_.Open(db_dir_ + "/graph", mode, initial_nodes, initial_edges);
  cluster_index_.Open(db_dir_ + "/clusters", mode);
  author_store_.Open(db_dir_ + "/authors", mode, author_max_lifetime);
  journal_.Open(db_dir_ + "/delta_journal.bin", mode);

  LoadManifest();
}

void LadyBugDB::Close() {
  Sync(false);
  node_store_.Close();
  csr_.Close();
  cluster_index_.Close();
  author_store_.Close();
  journal_.Close();
}

void LadyBugDB::Sync(bool async) {
  node_store_.Sync(async);
  csr_.Sync(async);
  cluster_index_.Sync(async);
  author_store_.Sync(async);
  journal_.Sync(async);
}

void LadyBugDB::ReadNumAuthorsBag(const std::string &bag_csv) {
  if (bag_csv.empty()) return;
  std::ifstream file(bag_csv);
  if (!file.is_open()) return;

  num_authors_bag_vec_.clear();
  std::string line;
  int line_no = 0;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string val1, val2;
    if (std::getline(ss, val1, ',') && std::getline(ss, val2, ',')) {
      if (!val2.empty() && val2.back() == '\r') val2.pop_back();
      if (line_no != 0) {
        num_authors_bag_vec_.push_back(std::stoi(val2));
      }
    }
    line_no++;
  }
}

int LadyBugDB::GetNextNumAuthors() {
  if (num_authors_bag_vec_.empty()) {
    return 1;
  }
  std::uniform_int_distribution<int> dist{0, static_cast<int>(num_authors_bag_vec_.size() - 1)};
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  return num_authors_bag_vec_[dist(generator)];
}

void LadyBugDB::IngestSeedData(const std::string &nodelist_csv,
                               const std::string &edgelist_csv,
                               const std::string &cluster_csv,
                               const std::string &num_authors_bag_csv,
                               bool is_checkpoint) {
  if (!num_authors_bag_csv.empty()) {
    ReadNumAuthorsBag(num_authors_bag_csv);
  }

  // 1. Ingest Nodelist
  node_store_.IngestCSV(nodelist_csv, is_checkpoint);

  // 2. Ingest Edgelist
  csr_.IngestCSV(edgelist_csv, node_store_.GetNodeCount());

  // Update in/out degrees in node store based on CSR
  for (size_t u = 0; u < node_store_.GetNodeCount(); ++u) {
    node_store_.SetInDegree(u, static_cast<uint32_t>(csr_.GetInDegree(u)));
    node_store_.SetOutDegree(u, static_cast<uint32_t>(csr_.GetOutDegree(u)));
  }

  // 3. Ingest Clusters
  if (!cluster_csv.empty()) {
    cluster_index_.IngestCSV(cluster_csv);
    for (size_t u = 0; u < node_store_.GetNodeCount(); ++u) {
      int cid = node_store_.GetClusterId(u);
      if (cid >= 0) {
        cluster_index_.AddNodeToCluster(u, cid);
      }
    }
  }

  // 4. Initialize Authorship
  if (!is_checkpoint) {
    std::vector<std::pair<int, int>> node_years;
    node_years.reserve(node_store_.GetNodeCount());
    for (size_t u = 0; u < node_store_.GetNodeCount(); ++u) {
      node_years.push_back({static_cast<int>(u), node_store_.GetYear(u)});
    }
    std::sort(node_years.begin(), node_years.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.second < b.second;
              });

    size_t prev_idx = 0;
    int prev_yr = node_years.empty() ? 0 : node_years[0].second;

    for (size_t i = 0; i < node_years.size(); ++i) {
      int u = node_years[i].first;
      int yr = node_years[i].second;
      int author_id = author_store_.GetNextAuthor(yr, {});
      node_store_.SetAuthorId(u, author_id);
      author_store_.UpdateAuthorPublication(author_id, u);

      if (prev_yr != yr) {
        author_store_.ComputeAuthorReputations(node_store_);
        for (size_t j = prev_idx; j < i; ++j) {
          int n = node_years[j].first;
          int auth = node_store_.GetAuthorId(n);
          node_store_.SetInitialAuthorReputation(n, author_store_.GetAuthorReputation(auth));
        }
        prev_idx = i;
        prev_yr = yr;
      }
    }

    if (prev_idx < node_years.size()) {
      author_store_.ComputeAuthorReputations(node_store_);
      for (size_t j = prev_idx; j < node_years.size(); ++j) {
        int n = node_years[j].first;
        int auth = node_store_.GetAuthorId(n);
        node_store_.SetInitialAuthorReputation(n, author_store_.GetAuthorReputation(auth));
      }
    }
  } else {
    for (size_t u = 0; u < node_store_.GetNodeCount(); ++u) {
      int auth = node_store_.GetAuthorId(u);
      int yr = node_store_.GetYear(u);
      if (auth >= 0) {
        author_store_.SetAuthorBirthYear(auth, yr);
        author_store_.UpdateAuthorPublication(auth, u);
        int cart = node_store_.GetCartelId(u);
        if (cart != -1) {
          author_store_.SetCartelID(auth, cart);
        }
        author_store_.SetNextAuthorId(std::max(author_store_.GetNextAuthorId(), auth + 1));
      }
    }
    author_store_.ComputeAuthorReputations(node_store_);
  }

  manifest_.node_count = node_store_.GetNodeCount();
  manifest_.edge_count = csr_.GetEdgeCount();
  manifest_.next_author_id = author_store_.GetNextAuthorId();
  manifest_.last_commit_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();

  Sync(false);
  SaveManifest();
}

void LadyBugDB::Checkpoint(int current_year) {
  // 1. Merge incoming delta buffer into Backward CSR
  csr_.MergeAnnualDeltaBuffer();

  // 2. Recompute author reputations
  author_store_.ComputeAuthorReputations(node_store_);

  // 3. Log commit to journal and sync
  journal_.AppendCommit(current_year, node_store_.GetNodeCount(), csr_.GetEdgeCount());
  Sync(false);

  // 4. Update and atomically write manifest.json
  manifest_.current_year = current_year;
  manifest_.node_count = node_store_.GetNodeCount();
  manifest_.edge_count = csr_.GetEdgeCount();
  manifest_.next_author_id = author_store_.GetNextAuthorId();
  manifest_.last_commit_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();
  manifest_.status = "valid";

  SaveManifest();

  // 5. Prune / reset delta journal on successful commit (zero-bloat)
  journal_.Truncate();
}

bool LadyBugDB::Recover(int target_year) {
  if (!LoadManifest()) {
    return false;
  }
  if (manifest_.status != "valid") {
    return false;
  }

  // If a target year is requested, verify match
  if (target_year >= 0 && manifest_.current_year != target_year) {
    std::cout << "[LadyBugDB::Recover] Warning: Manifest current_year ("
              << manifest_.current_year << ") != target_year (" << target_year << ")\n";
  }

  // Restore node and CSR counts from manifest
  if (manifest_.node_count > 0) {
    node_store_.SetNodeCount(manifest_.node_count);
    csr_.Restore(manifest_.node_count, manifest_.edge_count);
  }

  return true;
}

void LadyBugDB::SaveManifest() {
  std::string manifest_path = db_dir_ + "/manifest.json";
  std::string tmp_path = db_dir_ + "/manifest.json.tmp";

  std::ofstream f(tmp_path);
  if (!f.is_open()) return;

  f << "{\n"
    << "  \"version\": " << manifest_.version << ",\n"
    << "  \"current_year\": " << manifest_.current_year << ",\n"
    << "  \"node_count\": " << manifest_.node_count << ",\n"
    << "  \"edge_count\": " << manifest_.edge_count << ",\n"
    << "  \"next_author_id\": " << manifest_.next_author_id << ",\n"
    << "  \"last_commit_timestamp\": " << manifest_.last_commit_timestamp << ",\n"
    << "  \"status\": \"" << manifest_.status << "\"\n"
    << "}\n";
  f.close();

  std::filesystem::rename(tmp_path, manifest_path);
}

bool LadyBugDB::LoadManifest() {
  std::string manifest_path = db_dir_ + "/manifest.json";
  std::ifstream f(manifest_path);
  if (!f.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(f, line)) {
    if (line.find("\"current_year\"") != std::string::npos) {
      size_t pos = line.find(':');
      if (pos != std::string::npos) manifest_.current_year = std::stoi(line.substr(pos + 1));
    } else if (line.find("\"node_count\"") != std::string::npos) {
      size_t pos = line.find(':');
      if (pos != std::string::npos) manifest_.node_count = std::stoull(line.substr(pos + 1));
    } else if (line.find("\"edge_count\"") != std::string::npos) {
      size_t pos = line.find(':');
      if (pos != std::string::npos) manifest_.edge_count = std::stoull(line.substr(pos + 1));
    } else if (line.find("\"next_author_id\"") != std::string::npos) {
      size_t pos = line.find(':');
      if (pos != std::string::npos) manifest_.next_author_id = std::stoi(line.substr(pos + 1));
    } else if (line.find("\"last_commit_timestamp\"") != std::string::npos) {
      size_t pos = line.find(':');
      if (pos != std::string::npos) manifest_.last_commit_timestamp = std::stoull(line.substr(pos + 1));
    } else if (line.find("\"status\"") != std::string::npos) {
      if (line.find("valid") != std::string::npos) manifest_.status = "valid";
    }
  }
  return true;
}

} // namespace ladybug

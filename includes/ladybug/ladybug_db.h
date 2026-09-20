#ifndef LADYBUG_DB_H
#define LADYBUG_DB_H

#include "ladybug_author_store.h"
#include "ladybug_cluster_index.h"
#include "ladybug_csr.h"
#include "ladybug_mmap.h"
#include "ladybug_node_store.h"
#include <string>
#include <vector>

namespace ladybug {

struct DBManifest {
  int version = 1;
  int current_year = 0;
  size_t node_count = 0;
  size_t edge_count = 0;
  int next_author_id = 0;
  std::string status = "valid";
};

/*
 * LadyBugDB: Unified high-performance out-of-core embedded database engine for CantataS.
 * Encapsulates:
 * - LadyBugNodeStore (Columnar SoA node attributes)
 * - LadyBugBidirectionalCSR (Forward & Backward CSR with annual superstep delta merge)
 * - LadyBugClusterIndex (Leiden community partition indexing)
 * - LadyBugAuthorStore (Author reputations and Lotka distribution)
 * - Zero-bloat atomic checkpointing & manifest recovery
 */
class LadyBugDB {
public:
  LadyBugDB();
  ~LadyBugDB();

  LadyBugDB(const LadyBugDB &) = delete;
  LadyBugDB &operator=(const LadyBugDB &) = delete;

  LadyBugDB(LadyBugDB &&) noexcept;
  LadyBugDB &operator=(LadyBugDB &&) noexcept;

  void Open(const std::string &db_dir, MmapMode mode = MmapMode::ReadWrite,
            size_t initial_nodes = 0, size_t initial_edges = 0,
            int author_max_lifetime = 30);
  void Close();
  void Sync(bool async = false);

  // Ingest seed files from CSV
  void IngestSeedData(const std::string &nodelist_csv,
                       const std::string &edgelist_csv,
                       const std::string &cluster_csv = "",
                       const std::string &num_authors_bag_csv = "",
                       bool is_checkpoint = false);

  // Checkpointing & Persistence
  void Checkpoint(int current_year);
  bool Recover(int target_year);

  // Bag sampling for author counts
  void ReadNumAuthorsBag(const std::string &bag_csv);
  int GetNextNumAuthors();

  // Core component accessors
  LadyBugNodeStore &Nodes() noexcept { return node_store_; }
  const LadyBugNodeStore &Nodes() const noexcept { return node_store_; }

  LadyBugBidirectionalCSR &CSR() noexcept { return csr_; }
  const LadyBugBidirectionalCSR &CSR() const noexcept { return csr_; }

  LadyBugClusterIndex &Clusters() noexcept { return cluster_index_; }
  const LadyBugClusterIndex &Clusters() const noexcept { return cluster_index_; }

  LadyBugAuthorStore &Authors() noexcept { return author_store_; }
  const LadyBugAuthorStore &Authors() const noexcept { return author_store_; }

  const std::string &GetDbDir() const noexcept { return db_dir_; }
  const DBManifest &GetManifest() const noexcept { return manifest_; }

private:
  std::string db_dir_;
  MmapMode mode_;
  DBManifest manifest_;

  LadyBugNodeStore node_store_;
  LadyBugBidirectionalCSR csr_;
  LadyBugClusterIndex cluster_index_;
  LadyBugAuthorStore author_store_;

  std::vector<int> num_authors_bag_vec_;

  void SaveManifest();
  bool LoadManifest();
};

} // namespace ladybug

#endif // LADYBUG_DB_H

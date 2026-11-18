#ifndef LADYBUG_DB_H
#define LADYBUG_DB_H

#include "ladybug/ladybug_authors.h"
#include "ladybug/ladybug_column.h"
#include "ladybug/ladybug_csr.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace ladybug {

enum class NodeType : uint8_t { Seed = 0, Agent = 1 };

// Per-node attributes, one entry per node, in internal node-ID order.
// Internal IDs are contiguous: seeds are 0..seed_count-1 (sorted by their
// original ID), agents follow in creation order.
struct NodeColumns {
  // Read by the model.
  Column<int16_t> year;
  Column<uint8_t> type;
  Column<int32_t> cluster;
  Column<int32_t> author;
  Column<int32_t> num_authors;
  Column<int32_t> fitness_lag;
  Column<int32_t> fitness_peak;
  Column<int32_t> fitness_duration;
  Column<int32_t> generator;
  // Written once when a node is created; used only for output.
  Column<float> alpha;
  Column<float> pa_weight;
  Column<float> fit_weight;
  Column<float> num_authors_weight;
  Column<float> author_reputation_weight;
  Column<int32_t> assigned_out_degree;
  Column<int32_t> sampled_neighborhood_size;
  Column<int32_t> fully_random_citations;
  Column<int32_t> initial_author_reputation;

  size_t size() const noexcept { return year.size(); }

  // Visits every column (name, column) for snapshots.
  template <typename F> void ForEach(F &&f) {
    f("node_year", year); f("node_type", type); f("node_cluster", cluster);
    f("node_author", author); f("node_num_authors", num_authors);
    f("node_fitness_lag", fitness_lag); f("node_fitness_peak", fitness_peak);
    f("node_fitness_duration", fitness_duration); f("node_generator", generator);
    f("node_alpha", alpha); f("node_pa_weight", pa_weight); f("node_fit_weight", fit_weight);
    f("node_num_authors_weight", num_authors_weight);
    f("node_author_reputation_weight", author_reputation_weight);
    f("node_assigned_out_degree", assigned_out_degree);
    f("node_sampled_neighborhood_size", sampled_neighborhood_size);
    f("node_fully_random_citations", fully_random_citations);
    f("node_initial_author_reputation", initial_author_reputation);
  }
};

// Members of each cluster, rebuilt from NodeColumns::cluster after each year.
// Members are in node-ID order, so seeds come first, then agents by year.
class ClusterIndex {
public:
  void Rebuild(std::span<const int32_t> cluster_of_node, size_t num_nodes);
  std::span<const int32_t> Members(int32_t cluster) const noexcept {
    if (cluster < 0 || static_cast<size_t>(cluster) + 1 >= offsets_.size()) {
      return {};
    }
    return {members_.data() + offsets_[cluster],
            static_cast<size_t>(offsets_[cluster + 1] - offsets_[cluster])};
  }
  size_t NumClusters() const noexcept { return offsets_.empty() ? 0 : offsets_.size() - 1; }

private:
  Column<uint64_t> offsets_;
  Column<int32_t> members_;
};

struct RunMetadata {
  uint64_t run_seed = 0;
  int32_t next_year = 0;     // first year that still has to be simulated
  int32_t end_year = 0;      // simulation stops before this year
  int32_t start_year = 0;    // first simulated year of the run
  uint64_t seed_count = 0;   // number of seed nodes
  int64_t max_orig_id = -1;  // largest original node ID in the seed
  uint64_t model_fingerprint = 0; // hash of the model parameters
};

/*
 * LadyBugDB: the single graph store the simulation reads.
 *
 * Holds the citation graph (BidirectionalCSR), per-node attributes, the
 * cluster index and the author registry. Every MPI rank holds an identical
 * copy and applies every year's new nodes and edges in the same order.
 * Snapshots (binary, checksummed) make runs restartable across jobs.
 */
class LadyBugDB {
public:
  BidirectionalCSR csr;
  NodeColumns nodes;
  ClusterIndex clusters;
  AuthorRegistry authors;
  Column<int32_t> seed_original_id; // original ID of each seed node
  RunMetadata meta;

  // Nodes with attribute rows (includes agents created this year).
  size_t NumNodes() const noexcept { return nodes.size(); }
  // Nodes that are part of the graph (have CSR rows).
  size_t NumGraphNodes() const noexcept { return csr.NumNodes(); }

  int64_t OriginalId(int32_t node) const noexcept {
    return (static_cast<uint64_t>(node) < meta.seed_count)
               ? static_cast<int64_t>(seed_original_id[node])
               : meta.max_orig_id + 1 + (static_cast<int64_t>(node) - static_cast<int64_t>(meta.seed_count));
  }

  void ReserveFinalSize(size_t nodes_hint, uint64_t edges_hint);
  void RebuildClusterIndex() { clusters.Rebuild(nodes.cluster.span(), NumGraphNodes()); }
};

// --- Ingest (ladybug_ingest.cpp) ---------------------------------------------
struct IngestOptions {
  std::string nodelist;
  std::string edgelist;
  std::string community_assignment; // optional
  std::vector<int> num_authors_bag;
  int author_max_lifetime = 30;
  uint64_t run_seed = 0;
};
// Builds the store from the seed CSV files. Deterministic for a given seed.
void IngestSeed(const IngestOptions &options, LadyBugDB &db,
                const std::function<void(const std::string &)> &log);

// --- Snapshots (ladybug_snapshot.cpp) -----------------------------------------
// Writes db into <dir>/snap_<next_year>/ atomically and points <dir>/LATEST at
// it. Keeps the newest `keep` snapshots. Returns the snapshot directory.
std::string WriteSnapshot(LadyBugDB &db, const std::string &dir, int keep);
// Loads the snapshot named in <dir>/LATEST. Returns false if there is none.
bool LoadLatestSnapshot(LadyBugDB &db, const std::string &dir, bool verify_checksums,
                        std::string *loaded_from);
bool HasSnapshot(const std::string &dir);

// --- Export (ladybug_export.cpp) ------------------------------------------------
void ExportEdgelist(const LadyBugDB &db, const std::string &path);
void ExportNodelist(const LadyBugDB &db, const std::string &path);

} // namespace ladybug

#endif // LADYBUG_DB_H

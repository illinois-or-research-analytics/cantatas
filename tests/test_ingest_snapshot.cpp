// Seed ingest (ID mapping, years, clusters, authors) and snapshot round trip.
#include "check.h"
#include "ladybug/ladybug_db.h"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void WriteFile(const fs::path &p, const std::string &text) {
  std::ofstream out(p);
  out << text;
}

template <typename T> static void SameColumn(const ladybug::Column<T> &a, const ladybug::Column<T> &b) {
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK(a[i] == b[i]);
}

int main() {
  fs::path dir = fs::temp_directory_path() / ("cantatas_test_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  // Original IDs 3, 5, 7, 10; node 3 appears only in the edge list.
  WriteFile(dir / "nodes.csv", "node_id,year\n10,1990\n5,1991\n7,1990\n");
  WriteFile(dir / "edges.csv", "source,target\n10,5\n7,5\r\n5,3\n\n10,7\n");
  WriteFile(dir / "comm.csv", "node_id,cluster_id\n10,0\n5,1\n7,0\n99,4\n");

  ladybug::IngestOptions options;
  options.nodelist = (dir / "nodes.csv").string();
  options.edgelist = (dir / "edges.csv").string();
  options.community_assignment = (dir / "comm.csv").string();
  options.num_authors_bag = {1, 2, 3};
  options.run_seed = 42;
  ladybug::LadyBugDB db;
  ladybug::IngestSeed(options, db, [](const std::string &) {});

  CHECK_EQ(db.NumNodes(), 4u);
  CHECK_EQ(db.csr.NumEdges(), 4u);
  // Internal IDs follow ascending original ID: 3->0, 5->1, 7->2, 10->3.
  CHECK_EQ(db.OriginalId(0), 3);
  CHECK_EQ(db.OriginalId(3), 10);
  CHECK_EQ(db.nodes.year[0], -1);
  CHECK_EQ(db.nodes.year[1], 1991);
  CHECK_EQ(db.nodes.cluster[0], -1);
  CHECK_EQ(db.nodes.cluster[2], 0);
  CHECK_EQ(db.nodes.cluster[1], 1);
  CHECK_EQ(db.csr.InDegree(1), 2u); // 10->5 and 7->5
  CHECK_EQ(db.csr.Out(3).size(), 2u);
  CHECK_EQ(db.nodes.author[0], -1); // edge-only node has no author
  CHECK(db.nodes.author[1] >= 0 && db.nodes.author[3] >= 0);
  CHECK_EQ(db.meta.start_year, 1992);
  CHECK_EQ(db.clusters.Members(0).size(), 2u);
  // Agents start after the largest original ID.
  db.meta.seed_count = 4;
  CHECK_EQ(db.OriginalId(4), 11);

  // Determinism: a second ingest with the same seed is identical.
  ladybug::LadyBugDB again;
  ladybug::IngestSeed(options, again, [](const std::string &) {});
  SameColumn(db.nodes.num_authors, again.nodes.num_authors);
  SameColumn(db.nodes.author, again.nodes.author);

  // Snapshot round trip.
  db.meta.next_year = 1992;
  db.meta.end_year = 2000;
  fs::path ckpt = dir / "ckpt";
  ladybug::WriteSnapshot(db, ckpt.string(), 2);
  CHECK(ladybug::HasSnapshot(ckpt.string()));
  ladybug::LadyBugDB loaded;
  std::string from;
  CHECK(ladybug::LoadLatestSnapshot(loaded, ckpt.string(), true, &from));
  SameColumn(db.csr.FwdOffsets(), loaded.csr.FwdOffsets());
  SameColumn(db.csr.FwdTargets(), loaded.csr.FwdTargets());
  SameColumn(db.csr.BwdOffsets(), loaded.csr.BwdOffsets());
  SameColumn(db.csr.BwdSources(), loaded.csr.BwdSources());
  SameColumn(db.nodes.year, loaded.nodes.year);
  SameColumn(db.nodes.cluster, loaded.nodes.cluster);
  SameColumn(db.nodes.author, loaded.nodes.author);
  SameColumn(db.seed_original_id, loaded.seed_original_id);
  CHECK_EQ(loaded.meta.end_year, 2000);
  CHECK_EQ(loaded.meta.run_seed, db.meta.run_seed);
  CHECK_EQ(loaded.authors.NumAuthors(), db.authors.NumAuthors());
  CHECK_EQ(loaded.authors.Buckets().size(), db.authors.Buckets().size());

  // A later snapshot becomes LATEST; old ones beyond `keep` are removed.
  db.meta.next_year = 1993;
  std::string newest = ladybug::WriteSnapshot(db, ckpt.string(), 1);
  size_t snapshot_dirs = 0;
  for (const auto &entry : fs::directory_iterator(ckpt)) {
    if (entry.is_directory()) ++snapshot_dirs;
  }
  CHECK_EQ(snapshot_dirs, 1u);
  CHECK(fs::path(newest).filename().string().rfind("snap_1993_", 0) == 0);
  {
    ladybug::LadyBugDB latest;
    CHECK(ladybug::LoadLatestSnapshot(latest, ckpt.string(), true, nullptr));
    CHECK_EQ(latest.meta.next_year, 1993);
  }

  // Corruption is detected.
  {
    std::fstream f(fs::path(newest) / "node_year.bin", std::ios::in | std::ios::out | std::ios::binary);
    char c = 0x7f;
    f.seekp(0);
    f.write(&c, 1);
  }
  bool threw = false;
  try {
    ladybug::LadyBugDB broken;
    ladybug::LoadLatestSnapshot(broken, ckpt.string(), true, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);

  // Node-list papers without edges are not graph nodes but still get authors.
  WriteFile(dir / "nodes2.csv", "node_id,year\n1,1990\n2,1990\n3,1995\n4,1996\n");
  WriteFile(dir / "edges2.csv", "source,target\n2,1\n");
  options.nodelist = (dir / "nodes2.csv").string();
  options.edgelist = (dir / "edges2.csv").string();
  options.community_assignment.clear();
  ladybug::LadyBugDB sparse;
  ladybug::IngestSeed(options, sparse, [](const std::string &) {});
  CHECK_EQ(sparse.NumNodes(), 2u);              // only papers 1 and 2
  CHECK_EQ(sparse.meta.start_year, 1991);      // max year over graph nodes, + 1
  CHECK_EQ(sparse.meta.max_orig_id, 4);        // agents' IDs start after every seed ID
  CHECK_EQ(sparse.authors.NumAuthors() >= 2, true);
  int listed_authors = sparse.authors.NumAuthors();
  CHECK(listed_authors <= 4);

  fs::remove_all(dir);
  std::cout << "test_ingest_snapshot passed" << std::endl;
  return 0;
}

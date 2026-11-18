#include "ladybug/ladybug_db.h"
#include "rng.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <tuple>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace ladybug {
namespace {

constexpr int kSnapshotVersion = 1;

// Parallel 64-bit content hash: deterministic regardless of thread count.
uint64_t HashBytes(const void *data, size_t bytes) {
  const size_t block = size_t{1} << 20;
  const size_t num_blocks = (bytes + block - 1) / block;
  std::vector<uint64_t> block_hash(num_blocks, 0);
  const unsigned char *p = static_cast<const unsigned char *>(data);
#pragma omp parallel for schedule(static)
  for (size_t b = 0; b < num_blocks; ++b) {
    size_t lo = b * block;
    size_t hi = std::min(bytes, lo + block);
    uint64_t h = 0x84222325CBF29CE4ULL ^ b;
    size_t i = lo;
    for (; i + 8 <= hi; i += 8) {
      uint64_t w;
      std::memcpy(&w, p + i, 8);
      h = (h ^ (w * 0x9E3779B97F4A7C15ULL));
      h = ((h << 31) | (h >> 33)) * 0xBF58476D1CE4E5B9ULL;
    }
    for (; i < hi; ++i) {
      h = (h ^ p[i]) * 0x100000001B3ULL;
    }
    block_hash[b] = rng::Mix64(h);
  }
  uint64_t h = rng::Mix64(bytes);
  for (size_t b = 0; b < num_blocks; ++b) {
    h = rng::Combine(h, block_hash[b]);
  }
  return h;
}

struct FileEntry {
  std::string name;
  size_t element_size = 0;
  uint64_t count = 0;
  uint64_t hash = 0;
};

void FsyncPath(const std::string &path, bool directory) {
  int fd = open(path.c_str(), directory ? O_RDONLY : O_WRONLY);
  if (fd >= 0) {
    fsync(fd);
    close(fd);
  }
}

void WriteRaw(const std::string &path, const void *data, size_t bytes) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    throw std::runtime_error("Snapshot: cannot create " + path + ": " + std::strerror(errno));
  }
  std::setvbuf(f, nullptr, _IOFBF, 8 << 20);
  const char *p = static_cast<const char *>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    size_t chunk = std::min(remaining, size_t{256} << 20);
    if (std::fwrite(p, 1, chunk, f) != chunk) {
      std::fclose(f);
      throw std::runtime_error("Snapshot: write failed for " + path + ": " + std::strerror(errno));
    }
    p += chunk;
    remaining -= chunk;
  }
  if (std::fflush(f) != 0 || fsync(fileno(f)) != 0) {
    std::fclose(f);
    throw std::runtime_error("Snapshot: flush failed for " + path);
  }
  std::fclose(f);
}

void ReadRaw(const std::string &path, void *data, size_t bytes) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    throw std::runtime_error("Snapshot: cannot open " + path);
  }
  std::setvbuf(f, nullptr, _IOFBF, 8 << 20);
  char *p = static_cast<char *>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    size_t chunk = std::min(remaining, size_t{256} << 20);
    if (std::fread(p, 1, chunk, f) != chunk) {
      std::fclose(f);
      throw std::runtime_error("Snapshot: short read from " + path);
    }
    p += chunk;
    remaining -= chunk;
  }
  std::fclose(f);
}

class Writer {
public:
  explicit Writer(std::string dir) : dir_(std::move(dir)) {}

  template <typename T> void Put(const std::string &name, const T *data, uint64_t count) {
    FileEntry e{name, sizeof(T), count, HashBytes(data, count * sizeof(T))};
    WriteRaw(dir_ + "/" + name + ".bin", data, count * sizeof(T));
    entries_.push_back(e);
  }

  void WriteManifest(const std::map<std::string, std::string> &values) {
    std::ostringstream out;
    out << "version=" << kSnapshotVersion << "\n";
    for (const auto &[k, v] : values) out << k << "=" << v << "\n";
    for (const auto &e : entries_) {
      out << "file " << e.name << " " << e.element_size << " " << e.count << " " << std::hex
          << e.hash << std::dec << "\n";
    }
    std::string text = out.str();
    WriteRaw(dir_ + "/manifest.txt", text.data(), text.size());
  }

private:
  std::string dir_;
  std::vector<FileEntry> entries_;
};

class Reader {
public:
  Reader(std::string dir, bool verify) : dir_(std::move(dir)), verify_(verify) {
    std::ifstream in(dir_ + "/manifest.txt");
    if (!in.is_open()) {
      throw std::runtime_error("Snapshot: missing manifest in " + dir_);
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("file ", 0) == 0) {
        std::istringstream ss(line.substr(5));
        FileEntry e;
        ss >> e.name >> e.element_size >> e.count >> std::hex >> e.hash;
        files_[e.name] = e;
      } else if (auto eq = line.find('='); eq != std::string::npos) {
        values_[line.substr(0, eq)] = line.substr(eq + 1);
      }
    }
    if (Get("version") != std::to_string(kSnapshotVersion)) {
      throw std::runtime_error("Snapshot: unsupported version in " + dir_);
    }
  }

  std::string Get(const std::string &key) const {
    auto it = values_.find(key);
    if (it == values_.end()) {
      throw std::runtime_error("Snapshot: manifest lacks " + key);
    }
    return it->second;
  }

  const FileEntry &Entry(const std::string &name, size_t element_size) const {
    auto it = files_.find(name);
    if (it == files_.end()) {
      throw std::runtime_error("Snapshot: manifest lacks file " + name);
    }
    if (it->second.element_size != element_size) {
      throw std::runtime_error("Snapshot: element size mismatch for " + name);
    }
    return it->second;
  }

  template <typename T> void Into(const std::string &name, Column<T> &column) {
    const FileEntry &e = Entry(name, sizeof(T));
    column.Clear();
    column.ResizeUninitialized(e.count);
    ReadRaw(dir_ + "/" + name + ".bin", column.data(), e.count * sizeof(T));
    Check(e, column.data());
  }

  template <typename T> void Into(const std::string &name, std::vector<T> &vec) {
    const FileEntry &e = Entry(name, sizeof(T));
    vec.resize(e.count);
    ReadRaw(dir_ + "/" + name + ".bin", vec.data(), e.count * sizeof(T));
    Check(e, vec.data());
  }

private:
  void Check(const FileEntry &e, const void *data) const {
    if (verify_ && HashBytes(data, e.count * e.element_size) != e.hash) {
      throw std::runtime_error("Snapshot: checksum mismatch for " + e.name + " in " + dir_);
    }
  }

  std::string dir_;
  bool verify_;
  std::map<std::string, std::string> values_;
  std::map<std::string, FileEntry> files_;
};

// Published snapshots are snap_<year>_<epoch ms>; oldest first.
std::vector<std::string> ListSnapshots(const std::string &dir) {
  std::vector<std::tuple<long, long long, std::string>> found;
  if (!fs::exists(dir)) return {};
  for (const auto &entry : fs::directory_iterator(dir)) {
    std::string name = entry.path().filename().string();
    if (!entry.is_directory() || name.rfind("snap_", 0) != 0 || name.find(".tmp") != std::string::npos) {
      continue;
    }
    try {
      size_t sep = name.find('_', 5);
      long year = std::stol(name.substr(5, sep == std::string::npos ? std::string::npos : sep - 5));
      long long stamp = (sep == std::string::npos) ? 0 : std::stoll(name.substr(sep + 1));
      found.emplace_back(year, stamp, name);
    } catch (...) {
    }
  }
  std::sort(found.begin(), found.end());
  std::vector<std::string> names;
  for (auto &f : found) names.push_back(std::get<2>(f));
  return names;
}

std::string ReadLatestName(const std::string &dir) {
  std::ifstream latest(dir + "/LATEST");
  std::string name;
  if (latest.is_open()) std::getline(latest, name);
  return name;
}

// Removes leftovers of snapshot writes that were killed part-way.
void RemoveStaleTemporaries(const std::string &dir) {
  for (const auto &entry : fs::directory_iterator(dir)) {
    std::string name = entry.path().filename().string();
    if ((name.rfind("snap_", 0) == 0 || name.rfind("LATEST", 0) == 0) && name.find(".tmp-") != std::string::npos) {
      fs::remove_all(entry.path());
    }
  }
}

} // namespace

std::string WriteSnapshot(LadyBugDB &db, const std::string &dir, int keep) {
  fs::create_directories(dir);
  RemoveStaleTemporaries(dir);
  const long long stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
  const std::string final_name = "snap_" + std::to_string(db.meta.next_year) + "_" + std::to_string(stamp);
  const std::string tmp_dir = dir + "/" + final_name + ".tmp-" + std::to_string(getpid());
  fs::create_directories(tmp_dir);

  Writer w(tmp_dir);
  w.Put("csr_fwd_offsets", db.csr.FwdOffsets().data(), db.csr.FwdOffsets().size());
  w.Put("csr_fwd_targets", db.csr.FwdTargets().data(), db.csr.FwdTargets().size());
  w.Put("csr_bwd_offsets", db.csr.BwdOffsets().data(), db.csr.BwdOffsets().size());
  w.Put("csr_bwd_sources", db.csr.BwdSources().data(), db.csr.BwdSources().size());
  db.nodes.ForEach([&](const char *name, auto &column) { w.Put(name, column.data(), column.size()); });
  w.Put("seed_original_id", db.seed_original_id.data(), db.seed_original_id.size());

  const auto &buckets = db.authors.Buckets();
  std::vector<uint64_t> bucket_sizes(buckets.size());
  std::vector<int32_t> bucket_members;
  for (size_t k = 0; k < buckets.size(); ++k) {
    bucket_sizes[k] = buckets[k].size();
    bucket_members.insert(bucket_members.end(), buckets[k].begin(), buckets[k].end());
  }
  w.Put("author_birth_year", db.authors.BirthYears().data(), db.authors.BirthYears().size());
  w.Put("author_reputation", db.authors.Reputations().data(), db.authors.Reputations().size());
  w.Put("author_bucket_sizes", bucket_sizes.data(), bucket_sizes.size());
  w.Put("author_bucket_members", bucket_members.data(), bucket_members.size());

  std::map<std::string, std::string> values{
      {"run_seed", std::to_string(db.meta.run_seed)},
      {"next_year", std::to_string(db.meta.next_year)},
      {"end_year", std::to_string(db.meta.end_year)},
      {"start_year", std::to_string(db.meta.start_year)},
      {"seed_count", std::to_string(db.meta.seed_count)},
      {"max_orig_id", std::to_string(db.meta.max_orig_id)},
      {"model_fingerprint", std::to_string(db.meta.model_fingerprint)},
      {"next_author_id", std::to_string(db.authors.NumAuthors())},
      {"node_count", std::to_string(db.NumNodes())},
      {"edge_count", std::to_string(db.csr.NumEdges())},
  };
  w.WriteManifest(values);
  FsyncPath(tmp_dir, true);

  // Publish: the directory name is new, so the snapshot LATEST points to is
  // never touched until LATEST has been switched to this one.
  const std::string final_dir = dir + "/" + final_name;
  fs::rename(tmp_dir, final_dir);
  FsyncPath(dir, true);

  const std::string latest_tmp = dir + "/LATEST.tmp-" + std::to_string(getpid());
  WriteRaw(latest_tmp, final_name.data(), final_name.size());
  fs::rename(latest_tmp, dir + "/LATEST");
  FsyncPath(dir, true);

  std::vector<std::string> existing = ListSnapshots(dir);
  for (size_t i = 0; i + static_cast<size_t>(std::max(keep, 1)) < existing.size(); ++i) {
    if (existing[i] != final_name) {
      fs::remove_all(dir + "/" + existing[i]);
    }
  }
  return final_dir;
}

bool HasSnapshot(const std::string &dir) {
  std::ifstream latest(dir + "/LATEST");
  return latest.is_open();
}

bool LoadLatestSnapshot(LadyBugDB &db, const std::string &dir, bool verify_checksums,
                        std::string *loaded_from) {
  const std::string name = ReadLatestName(dir);
  if (name.empty()) {
    return false;
  }
  const std::string snap_dir = dir + "/" + name;
  Reader r(snap_dir, verify_checksums);

  r.Into("csr_fwd_offsets", db.csr.FwdOffsets());
  r.Into("csr_fwd_targets", db.csr.FwdTargets());
  r.Into("csr_bwd_offsets", db.csr.BwdOffsets());
  r.Into("csr_bwd_sources", db.csr.BwdSources());
  db.csr.SyncNodeCountFromColumns();
  db.nodes.ForEach([&](const char *col_name, auto &column) { r.Into(col_name, column); });
  r.Into("seed_original_id", db.seed_original_id);

  r.Into("author_birth_year", db.authors.BirthYears());
  r.Into("author_reputation", db.authors.Reputations());
  std::vector<uint64_t> bucket_sizes;
  std::vector<int32_t> bucket_members;
  r.Into("author_bucket_sizes", bucket_sizes);
  r.Into("author_bucket_members", bucket_members);
  auto &buckets = db.authors.Buckets();
  buckets.assign(bucket_sizes.size(), {});
  size_t pos = 0;
  for (size_t k = 0; k < bucket_sizes.size(); ++k) {
    buckets[k].assign(bucket_members.begin() + pos, bucket_members.begin() + pos + bucket_sizes[k]);
    pos += bucket_sizes[k];
  }
  db.authors.SetNextAuthorId(static_cast<int32_t>(std::stoll(r.Get("next_author_id"))));

  db.meta.run_seed = std::stoull(r.Get("run_seed"));
  db.meta.next_year = std::stoi(r.Get("next_year"));
  db.meta.end_year = std::stoi(r.Get("end_year"));
  db.meta.start_year = std::stoi(r.Get("start_year"));
  db.meta.seed_count = std::stoull(r.Get("seed_count"));
  db.meta.max_orig_id = std::stoll(r.Get("max_orig_id"));
  db.meta.model_fingerprint = std::stoull(r.Get("model_fingerprint"));

  if (db.NumNodes() != db.csr.NumNodes()) {
    throw std::runtime_error("Snapshot: node columns and CSR disagree in " + snap_dir);
  }
  db.RebuildClusterIndex();
  if (loaded_from != nullptr) {
    *loaded_from = snap_dir;
  }
  return true;
}

} // namespace ladybug

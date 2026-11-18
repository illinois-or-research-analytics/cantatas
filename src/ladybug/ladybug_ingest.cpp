#include "ladybug/ladybug_db.h"
#include "ladybug/ladybug_parallel.h"
#include "rng.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ladybug {
namespace {

// Read-only memory map of a whole file.
class MappedFile {
public:
  explicit MappedFile(const std::string &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Could not open " + path);
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
      close(fd_);
      throw std::runtime_error("Could not stat " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ > 0) {
      void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (p == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Could not map " + path);
      }
      data_ = static_cast<const char *>(p);
      madvise(const_cast<char *>(data_), size_, MADV_SEQUENTIAL);
    }
  }
  ~MappedFile() {
    if (data_ != nullptr) {
      munmap(const_cast<char *>(data_), size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  std::string_view view() const { return {data_, size_}; }
  const std::string &path() const { return path_; }

private:
  std::string path_;
  int fd_ = -1;
  const char *data_ = nullptr;
  size_t size_ = 0;
};

char DetectDelimiter(std::string_view header) {
  if (header.find(',') != std::string_view::npos) return ',';
  if (header.find('\t') != std::string_view::npos) return '\t';
  if (header.find(' ') != std::string_view::npos) return ' ';
  return ',';
}

std::string_view Trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

std::vector<std::string> SplitHeader(std::string_view header, char delim) {
  std::vector<std::string> names;
  size_t start = 0;
  while (start <= header.size()) {
    size_t end = header.find(delim, start);
    if (end == std::string_view::npos) end = header.size();
    names.emplace_back(Trim(header.substr(start, end - start)));
    start = end + 1;
  }
  return names;
}

int FindColumn(const std::vector<std::string> &names, std::initializer_list<const char *> candidates) {
  for (const char *candidate : candidates) {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == candidate) return static_cast<int>(i);
    }
  }
  return -1;
}

bool ParseInt64(std::string_view field, int64_t &value) {
  field = Trim(field);
  if (field.empty()) return false;
  auto result = std::from_chars(field.data(), field.data() + field.size(), value);
  return result.ec == std::errc();
}

// Parses two integer columns (col_a, col_b) of every data line, in parallel
// and in file order. Lines missing either column are skipped.
void ParseTwoColumns(const MappedFile &file, int col_a, int col_b,
                     std::vector<int64_t> &out_a, std::vector<int64_t> &out_b,
                     std::vector<std::string> *header_names = nullptr) {
  std::string_view text = file.view();
  size_t header_end = text.find('\n');
  std::string_view header = text.substr(0, header_end == std::string_view::npos ? text.size() : header_end);
  char delim = DetectDelimiter(header);
  if (header_names != nullptr) {
    *header_names = SplitHeader(header, delim);
  }
  if (header_end == std::string_view::npos) {
    out_a.clear();
    out_b.clear();
    return;
  }
  const size_t body = header_end + 1;
  const size_t size = text.size();
  const int max_col = std::max(col_a, col_b);

  // Chunk boundaries aligned to line starts.
  int num_chunks = std::max(1, omp_get_max_threads() * 4);
  if (size - body < (1u << 20)) num_chunks = 1;
  std::vector<size_t> bounds(num_chunks + 1);
  bounds[0] = body;
  bounds[num_chunks] = size;
  for (int c = 1; c < num_chunks; ++c) {
    size_t pos = body + (size - body) * static_cast<size_t>(c) / static_cast<size_t>(num_chunks);
    pos = std::max(pos, bounds[c - 1]);
    while (pos < size && text[pos - 1] != '\n') ++pos;
    bounds[c] = pos;
  }

  // Pass 1: upper bound on lines per chunk.
  std::vector<size_t> line_start(num_chunks + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
  for (int c = 0; c < num_chunks; ++c) {
    size_t count = 0;
    for (size_t i = bounds[c]; i < bounds[c + 1]; ++i) {
      if (text[i] == '\n') ++count;
    }
    if (bounds[c + 1] == size && bounds[c + 1] > bounds[c] && text[size - 1] != '\n') ++count;
    line_start[c + 1] = count;
  }
  for (int c = 0; c < num_chunks; ++c) line_start[c + 1] += line_start[c];

  out_a.assign(line_start[num_chunks], 0);
  out_b.assign(line_start[num_chunks], 0);
  std::vector<size_t> parsed(num_chunks, 0);

  // Pass 2: parse each chunk into its slot.
#pragma omp parallel for schedule(dynamic, 1)
  for (int c = 0; c < num_chunks; ++c) {
    size_t write = line_start[c];
    size_t i = bounds[c];
    const size_t end = bounds[c + 1];
    while (i < end) {
      size_t eol = i;
      while (eol < end && text[eol] != '\n') ++eol;
      std::string_view line = text.substr(i, eol - i);
      i = eol + 1;
      if (Trim(line).empty()) continue;
      int64_t a = 0, b = 0;
      bool have_a = false, have_b = false;
      int col = 0;
      size_t start = 0;
      while (col <= max_col && start <= line.size()) {
        size_t stop = line.find(delim, start);
        if (stop == std::string_view::npos) stop = line.size();
        std::string_view field = line.substr(start, stop - start);
        // Treat runs of spaces after the delimiter as part of the gap.
        if (col == col_a) have_a = ParseInt64(field, a);
        if (col == col_b) have_b = ParseInt64(field, b);
        ++col;
        start = stop + 1;
      }
      if (have_a && have_b) {
        out_a[write] = a;
        out_b[write] = b;
        ++write;
      }
    }
    parsed[c] = write - line_start[c];
  }

  // Compact (only moves data when some lines were skipped).
  size_t total = 0;
  for (int c = 0; c < num_chunks; ++c) {
    if (total != line_start[c] && parsed[c] > 0) {
      std::memmove(out_a.data() + total, out_a.data() + line_start[c], parsed[c] * sizeof(int64_t));
      std::memmove(out_b.data() + total, out_b.data() + line_start[c], parsed[c] * sizeof(int64_t));
    }
    total += parsed[c];
  }
  out_a.resize(total);
  out_b.resize(total);
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

void IngestSeed(const IngestOptions &options, LadyBugDB &db,
                const std::function<void(const std::string &)> &log) {
  auto t0 = std::chrono::steady_clock::now();
  if (options.num_authors_bag.empty()) {
    throw std::runtime_error("IngestSeed: num_authors bag is empty");
  }

  // 1. Read the node list (node_id, year) and edge list (first two columns).
  std::vector<int64_t> node_ids, node_years;
  {
    MappedFile file(options.nodelist);
    std::vector<std::string> names;
    {
      std::string_view text = file.view();
      size_t eol = text.find('\n');
      names = SplitHeader(text.substr(0, eol == std::string_view::npos ? text.size() : eol),
                          DetectDelimiter(text.substr(0, eol == std::string_view::npos ? text.size() : eol)));
    }
    int id_col = FindColumn(names, {"node_id", "id"});
    int year_col = FindColumn(names, {"year"});
    if (id_col < 0 || year_col < 0) {
      throw std::runtime_error("nodelist " + options.nodelist + " needs node_id and year columns");
    }
    ParseTwoColumns(file, id_col, year_col, node_ids, node_years);
  }
  log("Read " + std::to_string(node_ids.size()) + " node rows (" + std::to_string(Seconds(t0)) + " s)");

  std::vector<int64_t> edge_src, edge_dst;
  {
    MappedFile file(options.edgelist);
    ParseTwoColumns(file, 0, 1, edge_src, edge_dst);
  }
  log("Read " + std::to_string(edge_src.size()) + " edges (" + std::to_string(Seconds(t0)) + " s)");

  // 2. Map original IDs to contiguous internal IDs (ascending original ID).
  int64_t max_id = -1;
  bool negative = false;
#pragma omp parallel for reduction(max : max_id) reduction(|| : negative)
  for (size_t i = 0; i < node_ids.size(); ++i) {
    max_id = std::max(max_id, node_ids[i]);
    negative = negative || node_ids[i] < 0;
  }
#pragma omp parallel for reduction(max : max_id) reduction(|| : negative)
  for (size_t i = 0; i < edge_src.size(); ++i) {
    max_id = std::max(max_id, std::max(edge_src[i], edge_dst[i]));
    negative = negative || edge_src[i] < 0 || edge_dst[i] < 0;
  }
  if (negative) {
    throw std::runtime_error("IngestSeed: node IDs must be non-negative");
  }
  if (max_id >= std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("IngestSeed: node IDs must fit in a 32-bit int");
  }
  // As in the original model, a paper is part of the graph only if it has at
  // least one citation edge. Node-list papers without edges still receive an
  // author below (they take part in the author-assignment sequence) but are not
  // nodes: they are never generators, random-citation targets or output rows.
  const size_t id_space = static_cast<size_t>(max_id + 1);
  std::vector<int32_t> to_internal(id_space, 0);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < edge_src.size(); ++i) {
    __atomic_store_n(&to_internal[edge_src[i]], 1, __ATOMIC_RELAXED);
    __atomic_store_n(&to_internal[edge_dst[i]], 1, __ATOMIC_RELAXED);
  }
  std::vector<uint8_t> has_edges(id_space, 0);
#pragma omp parallel for schedule(static)
  for (size_t orig = 0; orig < id_space; ++orig) {
    has_edges[orig] = static_cast<uint8_t>(to_internal[orig]);
  }
  const int32_t num_nodes = ExclusiveScanInPlace(to_internal.data(), id_space);
  // to_internal[orig] is now the internal ID of every original with edges.

  db.seed_original_id.Clear();
  db.seed_original_id.ResizeUninitialized(num_nodes);
#pragma omp parallel for schedule(static)
  for (size_t orig = 0; orig < id_space; ++orig) {
    if (has_edges[orig]) {
      db.seed_original_id[to_internal[orig]] = static_cast<int32_t>(orig);
    }
  }
  db.meta.seed_count = static_cast<uint64_t>(num_nodes);
  db.meta.max_orig_id = max_id;

  // 3. Graph.
  {
    std::vector<int32_t> src(edge_src.size()), dst(edge_dst.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < edge_src.size(); ++i) {
      src[i] = to_internal[edge_src[i]];
      dst[i] = to_internal[edge_dst[i]];
    }
    std::vector<int64_t>().swap(edge_src);
    std::vector<int64_t>().swap(edge_dst);
    db.csr.Build(static_cast<size_t>(num_nodes), src, dst);
  }
  log("Built CSR: " + std::to_string(num_nodes) + " nodes, " +
      std::to_string(db.csr.NumEdges()) + " edges (" + std::to_string(Seconds(t0)) + " s)");

  // 4. Node attributes. Seeds have static fitness (lag 0, peak 1, duration 1000).
  NodeColumns &n = db.nodes;
  n.ForEach([&](const char *, auto &column) { column.Clear(); });
  n.year.Resize(num_nodes, -1);
  n.type.Resize(num_nodes, static_cast<uint8_t>(NodeType::Seed));
  n.cluster.Resize(num_nodes, -1);
  n.author.Resize(num_nodes, -1);
  n.num_authors.Resize(num_nodes, 0);
  n.fitness_lag.Resize(num_nodes, 0);
  n.fitness_peak.Resize(num_nodes, 1);
  n.fitness_duration.Resize(num_nodes, 1000);
  n.generator.Resize(num_nodes, -1);
  n.alpha.Resize(num_nodes, -1.0f);
  n.pa_weight.Resize(num_nodes, -1.0f);
  n.fit_weight.Resize(num_nodes, -1.0f);
  n.num_authors_weight.Resize(num_nodes, -1.0f);
  n.author_reputation_weight.Resize(num_nodes, -1.0f);
  n.assigned_out_degree.Resize(num_nodes, -1);
  n.sampled_neighborhood_size.Resize(num_nodes, -1);
  n.fully_random_citations.Resize(num_nodes, -1);
  n.initial_author_reputation.Resize(num_nodes, -1);

  // Years from the node list (later rows win). Papers without edges are kept
  // aside for author assignment only.
  std::vector<int16_t> isolated_year(id_space, 0);
  std::vector<uint8_t> isolated_listed(id_space, 0);
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const size_t orig = static_cast<size_t>(node_ids[i]);
    if (has_edges[orig]) {
      n.year[to_internal[orig]] = static_cast<int16_t>(node_years[i]);
    } else {
      isolated_year[orig] = static_cast<int16_t>(node_years[i]);
      isolated_listed[orig] = 1;
    }
  }
  std::vector<uint8_t> listed(num_nodes, 0);
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const size_t orig = static_cast<size_t>(node_ids[i]);
    if (has_edges[orig]) listed[to_internal[orig]] = 1;
  }
  size_t num_isolated = 0;
  for (size_t orig = 0; orig < id_space; ++orig) num_isolated += isolated_listed[orig];
  if (num_isolated > 0) {
    log("Node-list papers without any citation edge: " + std::to_string(num_isolated) +
        " (not added to the graph, as in previous versions; they still receive authors)");
  }

  // num_authors: one draw from the bag per seed (per-node stream).
  const std::vector<int> &bag = options.num_authors_bag;
#pragma omp parallel for schedule(static)
  for (int32_t v = 0; v < num_nodes; ++v) {
    rng::SeedThread(options.run_seed, rng::Stream::IngestNumAuthors, static_cast<uint64_t>(v));
    std::uniform_int_distribution<size_t> pick(0, bag.size() - 1);
    n.num_authors[v] = bag[pick(rng::Thread())];
  }

  // 5. Clusters from the community assignment file (node_id, cluster_id).
  if (!options.community_assignment.empty()) {
    MappedFile file(options.community_assignment);
    std::string_view text = file.view();
    size_t eol = text.find('\n');
    std::string_view header = text.substr(0, eol == std::string_view::npos ? text.size() : eol);
    std::vector<std::string> names = SplitHeader(header, DetectDelimiter(header));
    int id_col = FindColumn(names, {"node_id"});
    int cluster_col = FindColumn(names, {"cluster_id", "comm_id"});
    if (id_col < 0 || cluster_col < 0) {
      throw std::runtime_error("community_assignment needs node_id and cluster_id (or comm_id) columns");
    }
    std::vector<int64_t> ids, clusters;
    ParseTwoColumns(file, id_col, cluster_col, ids, clusters);
    size_t assigned = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 0 || ids[i] > max_id || !has_edges[ids[i]]) continue;
      int32_t v = to_internal[ids[i]];
      n.cluster[v] = static_cast<int32_t>(clusters[i]);
      ++assigned;
    }
    log("Assigned clusters to " + std::to_string(assigned) + " nodes");
  }

  // 6. Authors in publication-year order (Lotka's law), as in the original:
  // every node-list paper gets an author (including papers without edges);
  // edge-only nodes have none. Each cohort's initial author reputation is
  // computed after the cohort.
  //
  // Within a year the papers go in a random (seeded) order. The order matters:
  // an author who gets a second paper in the same year gets one from later in
  // the sequence, so any order that tracks citation counts changes which
  // papers share an author. Original-ID order does track them (low IDs are
  // cited more in real data) and gave seed authors measurably lower h-indices
  // than the original code, whose order was effectively scrambled by an
  // unstable sort; that thinned the in-degree tail of the whole run.
  {
    struct Paper {
      int16_t year;
      int32_t node; // internal ID, or -1 for a paper without edges
    };
    std::vector<Paper> order;
    order.reserve(static_cast<size_t>(num_nodes) + num_isolated);
    for (size_t orig = 0; orig < id_space; ++orig) {
      if (has_edges[orig]) {
        int32_t v = to_internal[orig];
        if (listed[v]) order.push_back({n.year[v], v});
      } else if (isolated_listed[orig]) {
        order.push_back({isolated_year[orig], -1});
      }
    }
    std::stable_sort(order.begin(), order.end(), [](const Paper &a, const Paper &b) { return a.year < b.year; });
    rng::SeedThread(options.run_seed, rng::Stream::IngestAuthorOrder, 0);
    for (size_t lo = 0; lo < order.size();) {
      size_t hi = lo;
      while (hi < order.size() && order[hi].year == order[lo].year) ++hi;
      std::shuffle(order.begin() + static_cast<std::ptrdiff_t>(lo), order.begin() + static_cast<std::ptrdiff_t>(hi),
                   rng::Thread());
      lo = hi;
    }
    db.authors.Configure(options.author_max_lifetime);
    rng::SeedThread(options.run_seed, rng::Stream::IngestAuthors, 0);
    pcg32 &generator = rng::Thread();
    std::vector<std::vector<uint32_t>> author_degrees;
    std::vector<uint32_t> freq;
    auto h_index = [&](const std::vector<uint32_t> &degrees) {
      size_t m = degrees.size();
      freq.assign(m + 1, 0);
      for (uint32_t d : degrees) freq[std::min<size_t>(m, d)]++;
      size_t h = m;
      size_t at_least = freq[m];
      while (h > at_least) {
        --h;
        at_least += freq[h];
      }
      return static_cast<int32_t>(h);
    };
    size_t cohort_start = 0;
    for (size_t i = 0; i <= order.size(); ++i) {
      bool cohort_ends = (i == order.size()) || (i > cohort_start && order[i].year != order[cohort_start].year);
      if (cohort_ends) {
        for (size_t j = cohort_start; j < i; ++j) {
          int32_t v = order[j].node;
          if (v >= 0) n.initial_author_reputation[v] = h_index(author_degrees[n.author[v]]);
        }
        cohort_start = i;
      }
      if (i == order.size()) break;
      const Paper &paper = order[i];
      int32_t author = db.authors.NextAuthor(paper.year, generator);
      if (static_cast<size_t>(author) >= author_degrees.size()) author_degrees.resize(author + 1);
      if (paper.node >= 0) {
        n.author[paper.node] = author;
        author_degrees[author].push_back(db.csr.InDegree(paper.node));
      } else {
        author_degrees[author].push_back(0);
      }
    }
  }
  db.authors.ComputeReputations(n.author.span(), db.csr);
  db.RebuildClusterIndex();

  int32_t max_year = std::numeric_limits<int16_t>::min();
  for (int32_t v = 0; v < num_nodes; ++v) max_year = std::max<int32_t>(max_year, n.year[v]);
  db.meta.start_year = max_year + 1;
  db.meta.next_year = max_year + 1;
  log("Ingest finished: " + std::to_string(db.authors.NumAuthors()) + " authors, first simulated year " +
      std::to_string(db.meta.start_year) + " (" + std::to_string(Seconds(t0)) + " s)");
}

} // namespace ladybug

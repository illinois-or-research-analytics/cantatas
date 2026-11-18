#include "ladybug/ladybug_db.h"

#include <omp.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ladybug {
namespace {

class Line {
public:
  explicit Line(std::string &out) : out_(out) {}
  Line &Int(int64_t v) {
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    out_.append(buf, r.ptr);
    return *this;
  }
  // Same text as std::ostream << double with default precision (%g, 6 digits).
  Line &Real(double v) {
    char buf[32];
    int len = std::snprintf(buf, sizeof(buf), "%g", v);
    out_.append(buf, static_cast<size_t>(len));
    return *this;
  }
  Line &Str(const char *s) {
    out_.append(s);
    return *this;
  }
  Line &Comma() {
    out_.push_back(',');
    return *this;
  }
  void End() { out_.push_back('\n'); }

private:
  std::string &out_;
};

// Formats [0, num_items) in parallel blocks and writes them in order.
template <typename Format>
void WriteInBlocks(const std::string &path, const std::string &header, size_t num_items,
                   size_t items_per_block, Format &&format) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    throw std::runtime_error("Cannot open " + path + " for writing");
  }
  std::setvbuf(f, nullptr, _IOFBF, 16 << 20);
  std::fwrite(header.data(), 1, header.size(), f);

  const size_t num_blocks = (num_items + items_per_block - 1) / items_per_block;
  const size_t batch = static_cast<size_t>(std::max(1, omp_get_max_threads())) * 4;
  std::vector<std::string> buffers(batch);
  for (size_t first = 0; first < num_blocks; first += batch) {
    size_t count = std::min(batch, num_blocks - first);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t b = 0; b < count; ++b) {
      std::string &buf = buffers[b];
      buf.clear();
      size_t lo = (first + b) * items_per_block;
      size_t hi = std::min(num_items, lo + items_per_block);
      for (size_t i = lo; i < hi; ++i) {
        format(i, buf);
      }
    }
    for (size_t b = 0; b < count; ++b) {
      if (std::fwrite(buffers[b].data(), 1, buffers[b].size(), f) != buffers[b].size()) {
        std::fclose(f);
        throw std::runtime_error("Write failed for " + path);
      }
    }
  }
  if (std::fclose(f) != 0) {
    throw std::runtime_error("Close failed for " + path);
  }
}

} // namespace

void ExportEdgelist(const LadyBugDB &db, const std::string &path) {
  WriteInBlocks(path, "source,target\n", db.NumGraphNodes(), 1 << 14, [&](size_t u, std::string &buf) {
    const int64_t src = db.OriginalId(static_cast<int32_t>(u));
    for (int32_t v : db.csr.Out(static_cast<int32_t>(u))) {
      Line(buf).Int(src).Comma().Int(db.OriginalId(v)).End();
    }
  });
}

void ExportNodelist(const LadyBugDB &db, const std::string &path) {
  const std::string header =
      "node_id,type,year,alpha,pa_weight,fit_weight,num_authors_weight,"
      "author_reputation_weight,fit_lag_duration,fit_peak_value,fit_peak_duration,"
      "in_degree,out_degree,assigned_out_degree,planted_nodes_line_number,"
      "generator_node_string,sampled_neighborhood_size,fully_random_citations,"
      "author_id,num_authors,initial_author_reputation,final_author_reputation,"
      "cartel_id,cluster_id\n";
  const NodeColumns &n = db.nodes;
  WriteInBlocks(path, header, db.NumGraphNodes(), 1 << 14, [&](size_t i, std::string &buf) {
    const int32_t v = static_cast<int32_t>(i);
    const bool seed = n.type[v] == static_cast<uint8_t>(NodeType::Seed);
    Line line(buf);
    line.Int(db.OriginalId(v)).Comma().Str(seed ? "seed" : "agent").Comma().Int(n.year[v]).Comma();
    line.Real(n.alpha[v]).Comma().Real(n.pa_weight[v]).Comma().Real(n.fit_weight[v]).Comma();
    line.Real(n.num_authors_weight[v]).Comma().Real(n.author_reputation_weight[v]).Comma();
    line.Int(n.fitness_lag[v]).Comma().Int(n.fitness_peak[v]).Comma().Int(n.fitness_duration[v]).Comma();
    line.Int(db.csr.InDegree(v)).Comma().Int(db.csr.OutDegree(v)).Comma();
    line.Int(n.assigned_out_degree[v]).Comma().Int(-1).Comma();
    if (n.generator[v] >= 0) {
      line.Int(db.OriginalId(n.generator[v])).Comma();
    } else {
      line.Str("no_generators").Comma();
    }
    line.Int(n.sampled_neighborhood_size[v]).Comma().Int(n.fully_random_citations[v]).Comma();
    line.Int(n.author[v]).Comma().Int(n.num_authors[v]).Comma();
    line.Int(n.initial_author_reputation[v]).Comma().Int(db.authors.Reputation(n.author[v])).Comma();
    line.Int(-1).Comma().Int(n.cluster[v]).End();
  });
}

} // namespace ladybug

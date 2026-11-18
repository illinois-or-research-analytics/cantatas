// run_summary: reduces one run's output (edgelist.csv + nodelist.csv) to the
// macroscopic statistics used to compare model versions, written as JSON.
//
//   run_summary <run_dir> [out.json]
//
// Works on output from both the old and the new simulator (same CSV layout).
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::string_view> Split(std::string_view line, char delim) {
  std::vector<std::string_view> fields;
  size_t start = 0;
  while (true) {
    size_t end = line.find(delim, start);
    fields.push_back(line.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return fields;
}

long long ToInt(std::string_view s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ')) s.remove_suffix(1);
  long long v = 0;
  std::from_chars(s.data(), s.data() + s.size(), v);
  return v;
}

void WriteHist(std::ostream &out, const char *name, const std::map<long long, long long> &h, bool last = false) {
  out << "  \"" << name << "\": {";
  bool first = true;
  for (const auto &[k, v] : h) {
    out << (first ? "" : ", ") << "\"" << k << "\": " << v;
    first = false;
  }
  out << "}" << (last ? "\n" : ",\n");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: run_summary <run_dir> [out.json]" << std::endl;
    return 2;
  }
  std::string dir = argv[1];
  std::string out_path = argc > 2 ? argv[2] : dir + "/summary.json";

  // Nodes: id -> (is_agent, year, cluster).
  std::ifstream nodes(dir + "/nodelist.csv");
  if (!nodes) {
    std::cerr << "cannot open " << dir << "/nodelist.csv" << std::endl;
    return 1;
  }
  std::string line;
  std::getline(nodes, line);
  auto header = Split(line, ',');
  int c_id = -1, c_type = -1, c_year = -1, c_cluster = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    std::string h(header[i]);
    while (!h.empty() && h.back() == '\r') h.pop_back();
    if (h == "node_id") c_id = static_cast<int>(i);
    if (h == "type") c_type = static_cast<int>(i);
    if (h == "year") c_year = static_cast<int>(i);
    if (h == "cluster_id") c_cluster = static_cast<int>(i);
  }
  if (c_id < 0 || c_type < 0 || c_year < 0) {
    std::cerr << "nodelist needs node_id, type, year columns" << std::endl;
    return 1;
  }
  std::vector<int8_t> agent;
  std::vector<int16_t> year;
  std::vector<int32_t> cluster;
  long long num_nodes = 0, num_agents = 0;
  while (std::getline(nodes, line)) {
    if (line.empty()) continue;
    auto f = Split(line, ',');
    long long id = ToInt(f[c_id]);
    if (id < 0) continue;
    if (static_cast<size_t>(id) >= agent.size()) {
      size_t n = static_cast<size_t>(id) + 1;
      agent.resize(n, -1);
      year.resize(n, -1);
      cluster.resize(n, -1);
    }
    bool is_agent = f[c_type].substr(0, 5) == "agent";
    agent[id] = is_agent ? 1 : 0;
    year[id] = static_cast<int16_t>(ToInt(f[c_year]));
    cluster[id] = c_cluster >= 0 && static_cast<size_t>(c_cluster) < f.size() ? static_cast<int32_t>(ToInt(f[c_cluster])) : -1;
    ++num_nodes;
    num_agents += is_agent;
  }

  // Edges.
  std::ifstream edges(dir + "/edgelist.csv");
  if (!edges) {
    std::cerr << "cannot open " << dir << "/edgelist.csv" << std::endl;
    return 1;
  }
  std::getline(edges, line);
  std::vector<int32_t> in_degree(agent.size(), 0), out_degree(agent.size(), 0);
  std::map<long long, long long> age_hist, per_year;
  long long num_edges = 0, agent_edges = 0, intra = 0, clustered_src = 0, to_seed = 0, unknown_age = 0;
  while (std::getline(edges, line)) {
    if (line.empty()) continue;
    size_t comma = line.find(',');
    if (comma == std::string::npos) continue;
    long long s = ToInt(std::string_view(line).substr(0, comma));
    long long t = ToInt(std::string_view(line).substr(comma + 1));
    if (s < 0 || t < 0 || static_cast<size_t>(std::max(s, t)) >= agent.size()) continue;
    ++num_edges;
    in_degree[t]++;
    out_degree[s]++;
    if (agent[s] == 1) {
      ++agent_edges;
      per_year[year[s]]++;
      if (agent[t] == 0) ++to_seed;
      if (year[t] >= 0) {
        age_hist[year[s] - year[t]]++;
      } else {
        ++unknown_age;
      }
      if (cluster[s] >= 0) {
        ++clustered_src;
        if (cluster[t] == cluster[s]) ++intra;
      }
    }
  }

  std::map<long long, long long> in_all, in_agents, in_seeds, out_agents;
  std::vector<int32_t> top;
  for (size_t v = 0; v < agent.size(); ++v) {
    if (agent[v] < 0) continue;
    in_all[in_degree[v]]++;
    if (agent[v] == 1) {
      in_agents[in_degree[v]]++;
      out_agents[out_degree[v]]++;
    } else {
      in_seeds[in_degree[v]]++;
    }
    top.push_back(in_degree[v]);
  }
  std::partial_sort(top.begin(), top.begin() + std::min<size_t>(20, top.size()), top.end(), std::greater<>());
  top.resize(std::min<size_t>(20, top.size()));

  std::ofstream out(out_path);
  out << "{\n";
  out << "  \"run\": \"" << dir << "\",\n";
  out << "  \"nodes\": " << num_nodes << ",\n  \"agents\": " << num_agents << ",\n";
  out << "  \"edges\": " << num_edges << ",\n  \"agent_edges\": " << agent_edges << ",\n";
  out << "  \"agent_edges_to_seeds\": " << to_seed << ",\n";
  out << "  \"agent_edges_unknown_age\": " << unknown_age << ",\n";
  out << "  \"agent_edges_clustered_source\": " << clustered_src << ",\n";
  out << "  \"agent_edges_intra_cluster\": " << intra << ",\n";
  out << "  \"top_in_degrees\": [";
  for (size_t i = 0; i < top.size(); ++i) out << (i ? ", " : "") << top[i];
  out << "],\n";
  WriteHist(out, "in_degree_all", in_all);
  WriteHist(out, "in_degree_agents", in_agents);
  WriteHist(out, "in_degree_seeds", in_seeds);
  WriteHist(out, "out_degree_agents", out_agents);
  WriteHist(out, "citation_age", age_hist);
  WriteHist(out, "agent_edges_per_year", per_year, true);
  out << "}\n";
  std::cout << "wrote " << out_path << ": " << num_nodes << " nodes, " << num_edges << " edges" << std::endl;
  return 0;
}

#include "ladybug/ladybug_node_store.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

std::unordered_map<std::string, int> ParseHeader(const std::string &header_line, char delim) {
  std::unordered_map<std::string, int> col_map;
  std::stringstream ss(header_line);
  std::string item;
  int idx = 0;
  while (std::getline(ss, item, delim)) {
    // Strip trailing \r if present
    if (!item.empty() && item.back() == '\r') {
      item.pop_back();
    }
    col_map[item] = idx++;
  }
  return col_map;
}
} // namespace

LadyBugNodeStore::LadyBugNodeStore()
    : node_count_(0), capacity_(0), mode_(MmapMode::ReadWrite) {}

LadyBugNodeStore::~LadyBugNodeStore() {
  Close();
}

LadyBugNodeStore::LadyBugNodeStore(LadyBugNodeStore &&other) noexcept
    : db_dir_(std::move(other.db_dir_)),
      node_count_(other.node_count_),
      capacity_(other.capacity_),
      mode_(other.mode_),
      year_(std::move(other.year_)),
      type_(std::move(other.type_)),
      cluster_id_(std::move(other.cluster_id_)),
      author_id_(std::move(other.author_id_)),
      in_degree_(std::move(other.in_degree_)),
      out_degree_(std::move(other.out_degree_)),
      alpha_(std::move(other.alpha_)),
      pa_weight_(std::move(other.pa_weight_)),
      fit_weight_(std::move(other.fit_weight_)),
      num_authors_weight_(std::move(other.num_authors_weight_)),
      author_rep_weight_(std::move(other.author_rep_weight_)),
      fitness_lag_duration_(std::move(other.fitness_lag_duration_)),
      fitness_peak_value_(std::move(other.fitness_peak_value_)),
      fitness_peak_duration_(std::move(other.fitness_peak_duration_)),
      assigned_out_degree_(std::move(other.assigned_out_degree_)),
      planted_nodes_line_number_(std::move(other.planted_nodes_line_number_)),
      sampled_neighborhood_size_(std::move(other.sampled_neighborhood_size_)),
      fully_random_citations_(std::move(other.fully_random_citations_)),
      num_authors_(std::move(other.num_authors_)),
      initial_author_reputation_(std::move(other.initial_author_reputation_)),
      cartel_id_(std::move(other.cartel_id_)),
      generator_node_id_(std::move(other.generator_node_id_)) {
  other.node_count_ = 0;
  other.capacity_ = 0;
}

LadyBugNodeStore &LadyBugNodeStore::operator=(LadyBugNodeStore &&other) noexcept {
  if (this != &other) {
    Close();
    db_dir_ = std::move(other.db_dir_);
    node_count_ = other.node_count_;
    capacity_ = other.capacity_;
    mode_ = other.mode_;
    year_ = std::move(other.year_);
    type_ = std::move(other.type_);
    cluster_id_ = std::move(other.cluster_id_);
    author_id_ = std::move(other.author_id_);
    in_degree_ = std::move(other.in_degree_);
    out_degree_ = std::move(other.out_degree_);
    alpha_ = std::move(other.alpha_);
    pa_weight_ = std::move(other.pa_weight_);
    fit_weight_ = std::move(other.fit_weight_);
    num_authors_weight_ = std::move(other.num_authors_weight_);
    author_rep_weight_ = std::move(other.author_rep_weight_);
    fitness_lag_duration_ = std::move(other.fitness_lag_duration_);
    fitness_peak_value_ = std::move(other.fitness_peak_value_);
    fitness_peak_duration_ = std::move(other.fitness_peak_duration_);
    assigned_out_degree_ = std::move(other.assigned_out_degree_);
    planted_nodes_line_number_ = std::move(other.planted_nodes_line_number_);
    sampled_neighborhood_size_ = std::move(other.sampled_neighborhood_size_);
    fully_random_citations_ = std::move(other.fully_random_citations_);
    num_authors_ = std::move(other.num_authors_);
    initial_author_reputation_ = std::move(other.initial_author_reputation_);
    cartel_id_ = std::move(other.cartel_id_);
    generator_node_id_ = std::move(other.generator_node_id_);

    other.node_count_ = 0;
    other.capacity_ = 0;
  }
  return *this;
}

void LadyBugNodeStore::Open(const std::string &db_dir, MmapMode mode, size_t initial_node_capacity) {
  Close();
  db_dir_ = db_dir;
  mode_ = mode;

  std::filesystem::create_directories(db_dir_);
  InitializeColumns(initial_node_capacity);
}

void LadyBugNodeStore::InitializeColumns(size_t initial_node_capacity) {
  size_t cap = std::max(initial_node_capacity, static_cast<size_t>(1024));

  year_.Open(db_dir_ + "/node_year.bin", mode_, cap);
  type_.Open(db_dir_ + "/node_type.bin", mode_, cap);
  cluster_id_.Open(db_dir_ + "/node_cluster.bin", mode_, cap);
  author_id_.Open(db_dir_ + "/node_author.bin", mode_, cap);
  in_degree_.Open(db_dir_ + "/node_in_degree.bin", mode_, cap);
  out_degree_.Open(db_dir_ + "/node_out_degree.bin", mode_, cap);
  alpha_.Open(db_dir_ + "/node_alpha.bin", mode_, cap);
  pa_weight_.Open(db_dir_ + "/node_pa_weight.bin", mode_, cap);
  fit_weight_.Open(db_dir_ + "/node_fit_weight.bin", mode_, cap);
  num_authors_weight_.Open(db_dir_ + "/node_num_authors_weight.bin", mode_, cap);
  author_rep_weight_.Open(db_dir_ + "/node_author_rep_weight.bin", mode_, cap);
  fitness_lag_duration_.Open(db_dir_ + "/node_fitness_lag.bin", mode_, cap);
  fitness_peak_value_.Open(db_dir_ + "/node_fitness_peak_val.bin", mode_, cap);
  fitness_peak_duration_.Open(db_dir_ + "/node_fitness_peak_dur.bin", mode_, cap);
  assigned_out_degree_.Open(db_dir_ + "/node_assigned_out_degree.bin", mode_, cap);
  planted_nodes_line_number_.Open(db_dir_ + "/node_planted_line.bin", mode_, cap);
  sampled_neighborhood_size_.Open(db_dir_ + "/node_sampled_nbr_size.bin", mode_, cap);
  fully_random_citations_.Open(db_dir_ + "/node_fully_random_cites.bin", mode_, cap);
  num_authors_.Open(db_dir_ + "/node_num_authors.bin", mode_, cap);
  initial_author_reputation_.Open(db_dir_ + "/node_initial_author_rep.bin", mode_, cap);
  cartel_id_.Open(db_dir_ + "/node_cartel_id.bin", mode_, cap);
  generator_node_id_.Open(db_dir_ + "/node_generator_node.bin", mode_, cap);

  capacity_ = year_.capacity();
  node_count_ = 0;
}

void LadyBugNodeStore::Close() {
  Sync(false);
  year_.Close();
  type_.Close();
  cluster_id_.Close();
  author_id_.Close();
  in_degree_.Close();
  out_degree_.Close();
  alpha_.Close();
  pa_weight_.Close();
  fit_weight_.Close();
  num_authors_weight_.Close();
  author_rep_weight_.Close();
  fitness_lag_duration_.Close();
  fitness_peak_value_.Close();
  fitness_peak_duration_.Close();
  assigned_out_degree_.Close();
  planted_nodes_line_number_.Close();
  sampled_neighborhood_size_.Close();
  fully_random_citations_.Close();
  num_authors_.Close();
  initial_author_reputation_.Close();
  cartel_id_.Close();
  generator_node_id_.Close();

  node_count_ = 0;
  capacity_ = 0;
}

void LadyBugNodeStore::Sync(bool async) {
  year_.Sync(async);
  type_.Sync(async);
  cluster_id_.Sync(async);
  author_id_.Sync(async);
  in_degree_.Sync(async);
  out_degree_.Sync(async);
  alpha_.Sync(async);
  pa_weight_.Sync(async);
  fit_weight_.Sync(async);
  num_authors_weight_.Sync(async);
  author_rep_weight_.Sync(async);
  fitness_lag_duration_.Sync(async);
  fitness_peak_value_.Sync(async);
  fitness_peak_duration_.Sync(async);
  assigned_out_degree_.Sync(async);
  planted_nodes_line_number_.Sync(async);
  sampled_neighborhood_size_.Sync(async);
  fully_random_citations_.Sync(async);
  num_authors_.Sync(async);
  initial_author_reputation_.Sync(async);
  cartel_id_.Sync(async);
  generator_node_id_.Sync(async);
}

void LadyBugNodeStore::EnsureNodeCapacity(size_t node_count) {
  if (node_count > capacity_) {
    size_t new_cap = std::max(node_count, capacity_ == 0 ? 1024 : capacity_ * 2);
    year_.Reserve(new_cap);
    type_.Reserve(new_cap);
    cluster_id_.Reserve(new_cap);
    author_id_.Reserve(new_cap);
    in_degree_.Reserve(new_cap);
    out_degree_.Reserve(new_cap);
    alpha_.Reserve(new_cap);
    pa_weight_.Reserve(new_cap);
    fit_weight_.Reserve(new_cap);
    num_authors_weight_.Reserve(new_cap);
    author_rep_weight_.Reserve(new_cap);
    fitness_lag_duration_.Reserve(new_cap);
    fitness_peak_value_.Reserve(new_cap);
    fitness_peak_duration_.Reserve(new_cap);
    assigned_out_degree_.Reserve(new_cap);
    planted_nodes_line_number_.Reserve(new_cap);
    sampled_neighborhood_size_.Reserve(new_cap);
    fully_random_citations_.Reserve(new_cap);
    num_authors_.Reserve(new_cap);
    initial_author_reputation_.Reserve(new_cap);
    cartel_id_.Reserve(new_cap);
    generator_node_id_.Reserve(new_cap);
    capacity_ = year_.capacity();
  }
}

void LadyBugNodeStore::SetNodeCount(size_t count) {
  EnsureNodeCapacity(count);
  node_count_ = count;
  year_.SetSize(count);
  type_.SetSize(count);
  cluster_id_.SetSize(count);
  author_id_.SetSize(count);
  in_degree_.SetSize(count);
  out_degree_.SetSize(count);
  alpha_.SetSize(count);
  pa_weight_.SetSize(count);
  fit_weight_.SetSize(count);
  num_authors_weight_.SetSize(count);
  author_rep_weight_.SetSize(count);
  fitness_lag_duration_.SetSize(count);
  fitness_peak_value_.SetSize(count);
  fitness_peak_duration_.SetSize(count);
  assigned_out_degree_.SetSize(count);
  planted_nodes_line_number_.SetSize(count);
  sampled_neighborhood_size_.SetSize(count);
  fully_random_citations_.SetSize(count);
  num_authors_.SetSize(count);
  initial_author_reputation_.SetSize(count);
  cartel_id_.SetSize(count);
  generator_node_id_.SetSize(count);
}

void LadyBugNodeStore::IngestCSV(const std::string &nodelist_csv, bool is_checkpoint) {
  char delim = DetectDelimiter(nodelist_csv);
  std::ifstream file(nodelist_csv);
  if (!file.is_open()) {
    throw std::runtime_error("LadyBugNodeStore::IngestCSV failed to open " + nodelist_csv);
  }

  std::string line;
  if (!std::getline(file, line)) {
    return;
  }

  auto header_map = ParseHeader(line, delim);

  int max_node_id = -1;
  struct RowData {
    int node_id;
    int year = 0;
    NodeType type = NodeType::Seed;
    double alpha = -1.0;
    double pa_weight = -1.0;
    double fit_weight = -1.0;
    double num_authors_weight = -1.0;
    double author_rep_weight = -1.0;
    int fit_lag = 0;
    int fit_peak_val = 1;
    int fit_peak_dur = 1000;
    int in_degree = 0;
    int out_degree = 0;
    int assigned_out_degree = -1;
    int planted_line = -1;
    int sampled_nbr_size = -1;
    int fully_random_cites = -1;
    int author_id = -1;
    int num_authors = 1;
    int initial_author_rep = 0;
    int cartel_id = -1;
    int cluster_id = -1;
    int generator_node_id = -1;
  };

  std::vector<RowData> rows;
  rows.reserve(65536);

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string val;
    std::vector<std::string> tokens;
    while (std::getline(ss, val, delim)) {
      if (!val.empty() && val.back() == '\r') val.pop_back();
      tokens.push_back(val);
    }
    if (tokens.empty()) continue;

    RowData r;
    if (header_map.contains("node_id")) {
      r.node_id = std::stoi(tokens[header_map["node_id"]]);
    } else {
      continue;
    }

    if (header_map.contains("year")) {
      r.year = std::stoi(tokens[header_map["year"]]);
    }

    if (is_checkpoint) {
      if (header_map.contains("type")) {
        r.type = (tokens[header_map["type"]] == "seed") ? NodeType::Seed : NodeType::Agent;
      }
      if (header_map.contains("alpha")) r.alpha = std::stod(tokens[header_map["alpha"]]);
      if (header_map.contains("pa_weight")) r.pa_weight = std::stod(tokens[header_map["pa_weight"]]);
      if (header_map.contains("fit_weight")) r.fit_weight = std::stod(tokens[header_map["fit_weight"]]);
      if (header_map.contains("num_authors_weight")) r.num_authors_weight = std::stod(tokens[header_map["num_authors_weight"]]);
      if (header_map.contains("author_reputation_weight")) r.author_rep_weight = std::stod(tokens[header_map["author_reputation_weight"]]);
      if (header_map.contains("fit_lag_duration")) r.fit_lag = std::stoi(tokens[header_map["fit_lag_duration"]]);
      if (header_map.contains("fit_peak_value")) r.fit_peak_val = std::stoi(tokens[header_map["fit_peak_value"]]);
      if (header_map.contains("fit_peak_duration")) r.fit_peak_dur = std::stoi(tokens[header_map["fit_peak_duration"]]);
      if (header_map.contains("in_degree")) r.in_degree = std::stoi(tokens[header_map["in_degree"]]);
      if (header_map.contains("out_degree")) r.out_degree = std::stoi(tokens[header_map["out_degree"]]);
      if (header_map.contains("assigned_out_degree")) r.assigned_out_degree = std::stoi(tokens[header_map["assigned_out_degree"]]);
      if (header_map.contains("planted_nodes_line_number")) r.planted_line = std::stoi(tokens[header_map["planted_nodes_line_number"]]);
      if (header_map.contains("sampled_neighborhood_size")) r.sampled_nbr_size = std::stoi(tokens[header_map["sampled_neighborhood_size"]]);
      if (header_map.contains("fully_random_citations")) r.fully_random_cites = std::stoi(tokens[header_map["fully_random_citations"]]);
      if (header_map.contains("author_id")) r.author_id = std::stoi(tokens[header_map["author_id"]]);
      if (header_map.contains("num_authors")) r.num_authors = std::stoi(tokens[header_map["num_authors"]]);
      if (header_map.contains("initial_author_reputation")) r.initial_author_rep = std::stoi(tokens[header_map["initial_author_reputation"]]);
      if (header_map.contains("cartel_id")) r.cartel_id = std::stoi(tokens[header_map["cartel_id"]]);
      if (header_map.contains("cluster_id")) r.cluster_id = std::stoi(tokens[header_map["cluster_id"]]);
    } else {
      r.type = NodeType::Seed;
      r.fit_lag = 0;
      r.fit_peak_val = 1;
      r.fit_peak_dur = 1000;
    }

    max_node_id = std::max(max_node_id, r.node_id);
    rows.push_back(r);
  }

  size_t total_nodes = max_node_id + 1;
  SetNodeCount(total_nodes);

  for (const auto &r : rows) {
    int u = r.node_id;
    SetYear(u, r.year);
    SetType(u, r.type);
    SetAlpha(u, r.alpha);
    SetPaWeight(u, r.pa_weight);
    SetFitWeight(u, r.fit_weight);
    SetNumAuthorsWeight(u, r.num_authors_weight);
    SetAuthorRepWeight(u, r.author_rep_weight);
    SetFitnessLagDuration(u, r.fit_lag);
    SetFitnessPeakValue(u, r.fit_peak_val);
    SetFitnessPeakDuration(u, r.fit_peak_dur);
    SetInDegree(u, r.in_degree);
    SetOutDegree(u, r.out_degree);
    SetAssignedOutDegree(u, r.assigned_out_degree);
    SetPlantedLineNumber(u, r.planted_line);
    SetSampledNeighborhoodSize(u, r.sampled_nbr_size);
    SetFullyRandomCitations(u, r.fully_random_cites);
    SetAuthorId(u, r.author_id);
    SetNumAuthors(u, r.num_authors);
    SetInitialAuthorReputation(u, r.initial_author_rep);
    SetCartelId(u, r.cartel_id);
    SetClusterId(u, r.cluster_id);
    SetGeneratorNode(u, r.generator_node_id);
  }

  Sync(false);
}

void LadyBugNodeStore::ExportCSV(const std::string &output_csv) const {
  std::ofstream out(output_csv);
  if (!out.is_open()) {
    throw std::runtime_error("LadyBugNodeStore::ExportCSV failed to open " + output_csv);
  }

  out << "node_id,type,year,alpha,pa_weight,fit_weight,num_authors_weight,"
         "author_reputation_weight,fit_lag_duration,fit_peak_value,fit_peak_"
         "duration,in_degree,out_degree,assigned_out_degree,planted_nodes_line_"
         "number,generator_node_string,sampled_neighborhood_size,fully_random_"
         "citations,author_id,num_authors,initial_author_reputation,final_"
         "author_reputation,cartel_id,cluster_id\n";

  for (size_t i = 0; i < node_count_; ++i) {
    int u = static_cast<int>(i);
    std::string type_str = (GetType(u) == NodeType::Seed) ? "seed" : "agent";
    out << u << "," << type_str << "," << GetYear(u) << ","
        << GetAlpha(u) << "," << GetPaWeight(u) << "," << GetFitWeight(u) << ","
        << GetNumAuthorsWeight(u) << "," << GetAuthorRepWeight(u) << ","
        << GetFitnessLagDuration(u) << "," << GetFitnessPeakValue(u) << ","
        << GetFitnessPeakDuration(u) << "," << GetInDegree(u) << ","
        << GetOutDegree(u) << "," << GetAssignedOutDegree(u) << ","
        << GetPlantedLineNumber(u) << "," << "no_generators" << ","
        << GetSampledNeighborhoodSize(u) << "," << GetFullyRandomCitations(u) << ","
        << GetAuthorId(u) << "," << GetNumAuthors(u) << ","
        << GetInitialAuthorReputation(u) << "," << 0 << ","
        << GetCartelId(u) << "," << GetClusterId(u) << "\n";
  }
}

} // namespace ladybug

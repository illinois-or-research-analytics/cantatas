#include "graph.h"
#include "utils.h"
#include <charconv>
#include <string_view>
#if defined(__linux__)
#include <malloc.h>
#endif

Graph::Graph(std::string edgelist, std::string nodelist,
             bool start_from_checkpoint, std::string num_authors_bag,
             int author_max_lifetime)
    : edgelist(edgelist), nodelist(nodelist),
      start_from_checkpoint(start_from_checkpoint),
      num_authors_bag(num_authors_bag),
      author_max_lifetime(author_max_lifetime) {
  this->ReadNumAuthorsBag();
  this->ParseEdgelist();
  this->ParseNodelist();
}

void Graph::ParseEdgelist() {
  char delimiter = Graph::GetDelimiter(this->edgelist);
  std::ifstream input_edgelist(this->edgelist);
  if (!input_edgelist.is_open()) {
    std::cerr << "Error: Could not open edgelist file: " << this->edgelist << std::endl;
    throw std::runtime_error("Could not open edgelist file: " + this->edgelist);
  }
  std::string line;
  int line_no = 0;
  while (std::getline(input_edgelist, line)) {
    if (line.empty()) continue;
    if (line_no != 0) {
      const char *p = line.c_str();
      char *end = nullptr;
      int integer_citing = static_cast<int>(std::strtol(p, &end, 10));
      while (*end == delimiter || *end == ' ' || *end == '\t' || *end == '\r') {
        end++;
      }
      int integer_cited = static_cast<int>(std::strtol(end, nullptr, 10));
      this->AddEdge({integer_citing, integer_cited});
    }
    line_no++;
  }
}

void Graph::ParseNodelist() {
  char delimiter = Graph::GetDelimiter(this->nodelist);
  std::unordered_map<std::string, int> header_to_index_map =
      Graph::GetHeaderToIndexMap(delimiter, this->nodelist);
  std::ifstream input_nodelist(this->nodelist);
  if (!input_nodelist.is_open()) {
    std::cerr << "Error: Could not open nodelist file: " << this->nodelist << std::endl;
    throw std::runtime_error("Could not open nodelist file: " + this->nodelist);
  }

  auto get_col = [&](const std::string &col) -> int {
    auto it = header_to_index_map.find(col);
    return it != header_to_index_map.end() ? it->second : -1;
  };
  int node_id_idx = get_col("node_id");
  int year_idx = get_col("year");
  int type_idx = get_col("type");
  int alpha_idx = get_col("alpha");
  int pa_weight_idx = get_col("pa_weight");
  int fit_weight_idx = get_col("fit_weight");
  int fit_lag_idx = get_col("fit_lag_duration");
  int fit_peak_val_idx = get_col("fit_peak_value");
  int fit_peak_dur_idx = get_col("fit_peak_duration");
  int assigned_out_deg_idx = get_col("assigned_out_degree");
  int planted_nodes_line_idx = get_col("planted_nodes_line_number");
  int sampled_nbr_idx = get_col("sampled_neighborhood_size");
  int gen_node_str_idx = get_col("generator_node_string");
  int fully_rand_idx = get_col("fully_random_citations");
  int author_idx = get_col("author_id");
  int init_author_rep_idx = get_col("initial_author_reputation");
  int num_authors_idx = get_col("num_authors");
  int cartel_id_idx = get_col("cartel_id");
  int cluster_id_idx = get_col("cluster_id");

  std::string line;
  int line_no = 0;
  std::vector<std::pair<int, int>> node_year_vec;
  std::vector<std::string_view> current_line;
  current_line.reserve(32);

  auto split_line = [&](std::string_view s) {
    current_line.clear();
    size_t start = 0;
    while (start < s.size()) {
      size_t end = s.find(delimiter, start);
      if (end == std::string_view::npos) {
        std::string_view tok = s.substr(start);
        if (!tok.empty() && tok.back() == '\r') tok.remove_suffix(1);
        current_line.push_back(tok);
        break;
      }
      std::string_view tok = s.substr(start, end - start);
      if (!tok.empty() && tok.back() == '\r') tok.remove_suffix(1);
      current_line.push_back(tok);
      start = end + 1;
    }
  };

  auto parse_int = [](std::string_view sv) -> int {
    int val = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), val);
    return val;
  };
  auto parse_double = [](std::string_view sv) -> double {
    return std::stod(std::string(sv));
  };

  while (std::getline(input_nodelist, line)) {
    if (line.empty()) continue;
    if (line_no != 0) {
      split_line(line);
      if (current_line.empty()) break;
      if (node_id_idx < 0 || static_cast<size_t>(node_id_idx) >= current_line.size()) continue;

      int integer_node = parse_int(current_line[node_id_idx]);
      int integer_year = (year_idx >= 0 && static_cast<size_t>(year_idx) < current_line.size())
                             ? parse_int(current_line[year_idx]) : 0;
      this->AddNode(integer_node);
      this->SetYear(integer_node, integer_year);
      node_year_vec.push_back({integer_node, integer_year});

      if (this->start_from_checkpoint) {
        if (type_idx >= 0 && static_cast<size_t>(type_idx) < current_line.size()) {
          this->SetType(integer_node, current_line[type_idx] == "seed" ? NodeType::Seed : NodeType::Agent);
        }
        if (alpha_idx >= 0 && static_cast<size_t>(alpha_idx) < current_line.size()) {
          this->SetAlpha(integer_node, parse_double(current_line[alpha_idx]));
        }
        if (pa_weight_idx >= 0 && static_cast<size_t>(pa_weight_idx) < current_line.size()) {
          this->SetPaWeight(integer_node, parse_double(current_line[pa_weight_idx]));
        }
        if (fit_weight_idx >= 0 && static_cast<size_t>(fit_weight_idx) < current_line.size()) {
          this->SetFitWeight(integer_node, parse_double(current_line[fit_weight_idx]));
        }
        if (fit_lag_idx >= 0 && static_cast<size_t>(fit_lag_idx) < current_line.size()) {
          this->SetFitnessLagDuration(integer_node, parse_int(current_line[fit_lag_idx]));
        }
        if (fit_peak_val_idx >= 0 && static_cast<size_t>(fit_peak_val_idx) < current_line.size()) {
          this->SetFitnessPeakValue(integer_node, parse_int(current_line[fit_peak_val_idx]));
        }
        if (fit_peak_dur_idx >= 0 && static_cast<size_t>(fit_peak_dur_idx) < current_line.size()) {
          this->SetFitnessPeakDuration(integer_node, parse_int(current_line[fit_peak_dur_idx]));
        }
        if (assigned_out_deg_idx >= 0 && static_cast<size_t>(assigned_out_deg_idx) < current_line.size()) {
          this->SetAssignedOutDegree(integer_node, parse_int(current_line[assigned_out_deg_idx]));
        }
        if (planted_nodes_line_idx >= 0 && static_cast<size_t>(planted_nodes_line_idx) < current_line.size()) {
          this->SetPlantedNodesLineNumber(integer_node, parse_int(current_line[planted_nodes_line_idx]));
        }
        if (sampled_nbr_idx >= 0 && static_cast<size_t>(sampled_nbr_idx) < current_line.size()) {
          this->SetSampledNeighborhoodSize(integer_node, parse_int(current_line[sampled_nbr_idx]));
        }
        if (gen_node_str_idx >= 0 && static_cast<size_t>(gen_node_str_idx) < current_line.size()) {
          std::string_view sv = current_line[gen_node_str_idx];
          if (sv.empty() || sv == "no_generators") {
            this->SetGeneratorNode(integer_node, -1);
          } else {
            size_t pos = sv.find(';');
            std::string_view first_gen = (pos == std::string_view::npos) ? sv : sv.substr(0, pos);
            this->SetGeneratorNode(integer_node, parse_int(first_gen));
          }
        }
        if (fully_rand_idx >= 0 && static_cast<size_t>(fully_rand_idx) < current_line.size()) {
          this->SetFullyRandomCitations(integer_node, parse_int(current_line[fully_rand_idx]));
        }
        int author = -1;
        if (author_idx >= 0 && static_cast<size_t>(author_idx) < current_line.size()) {
          author = parse_int(current_line[author_idx]);
          this->SetAuthorId(integer_node, author);
        }
        if (init_author_rep_idx >= 0 && static_cast<size_t>(init_author_rep_idx) < current_line.size()) {
          this->SetInitialAuthorReputation(integer_node, parse_int(current_line[init_author_rep_idx]));
        }
        if (num_authors_idx >= 0 && static_cast<size_t>(num_authors_idx) < current_line.size()) {
          this->SetNumAuthors(integer_node, parse_int(current_line[num_authors_idx]));
        }
        if (cartel_id_idx >= 0 && static_cast<size_t>(cartel_id_idx) < current_line.size()) {
          int cartel_id = parse_int(current_line[cartel_id_idx]);
          if (cartel_id != -1) {
            this->SetCartelIdAttr(integer_node, cartel_id);
            this->cartel_set.insert(cartel_id);
            if (author >= 0) this->SetCartelID(author, cartel_id);
          }
        }
        if (cluster_id_idx >= 0 && static_cast<size_t>(cluster_id_idx) < current_line.size()) {
          int cluster_id = parse_int(current_line[cluster_id_idx]);
          this->SetCommunityAssignment(integer_node, cluster_id);
          if (cluster_id >= 0) {
            this->AddNodeToCluster(integer_node, cluster_id);
          }
        }
        if (author >= 0) {
          if (this->author_birth_year_map.contains(author)) {
            this->author_birth_year_map[author] =
                std::min(this->author_birth_year_map[author], integer_year);
          } else {
            this->author_birth_year_map[author] = integer_year;
          }
          this->UpdateAuthorPublicationMap(author, integer_node);
          this->next_author_id = std::max(this->next_author_id, author);
        }
      } else {
        int fitness_lag_uniform = 0; // MARK: hard coded to be static fitness
        int fitness_peak_uniform = 1000;
        int fitness_power = 1;
        this->SetType(integer_node, NodeType::Seed);
        this->SetFitnessLagDuration(integer_node, fitness_lag_uniform);
        this->SetFitnessPeakDuration(integer_node, fitness_peak_uniform);
        this->SetFitnessPeakValue(integer_node, fitness_power);
      }
    }
    line_no++;
  }

  if (this->start_from_checkpoint) {
    this->next_author_id++;
    for (const auto &[author_id, birth_year] : author_birth_year_map) {
      int author_id_publication_count =
          this->author_publication_map.at(author_id).size();
      this->publication_count_to_author_map[author_id_publication_count]
          .push_back(author_id);
    }
  } else {
    std::sort(
        node_year_vec.begin(), node_year_vec.end(),
        [](const std::pair<int, int> &left, const std::pair<int, int> &right) {
          return left.second < right.second;
        });
    size_t previous_index = 0;
    int previous_year = node_year_vec.empty() ? 0 : node_year_vec[0].second;
    for (size_t i = 0; i < node_year_vec.size(); i++) {
      int current_node_id = node_year_vec[i].first;
      int current_year = node_year_vec[i].second;
      int author_id = this->GetNextAuthor(current_year, std::set<int>());
      this->SetAuthorId(current_node_id, author_id);
      this->UpdateAuthorPublicationMap(author_id, current_node_id);
      if (previous_year != current_year) {
        for (size_t j = previous_index; j < i; j++) {
          int node_id = node_year_vec[j].first;
          int auth = this->GetAuthorId(node_id);
          this->SetInitialAuthorReputation(
              node_id, this->ComputeSingleAuthorReputation(auth));
        }
        previous_index = i;
        previous_year = current_year;
      }
    }
    if (previous_index < node_year_vec.size()) {
      for (size_t j = previous_index; j < node_year_vec.size(); j++) {
        int node_id = node_year_vec[j].first;
        int auth = this->GetAuthorId(node_id);
        this->SetInitialAuthorReputation(
            node_id, this->ComputeSingleAuthorReputation(auth));
      }
    }
  }
}

std::set<int> Graph::GetCartelSet() const {
  if (db_) {
    return db_->Authors().GetCartelSet();
  }
  return this->cartel_set;
}

void Graph::SetCartelID(int author, int cartel_id) {
  if (db_) {
    db_->Authors().SetCartelID(author, cartel_id);
    return;
  }
  this->author_cartel_map[author] = cartel_id;
  this->cartel_author_map[cartel_id].insert(author);
}

std::set<int> Graph::GetCartelAuthors(int cartel_id) const {
  if (db_) {
    return db_->Authors().GetCartelAuthors(cartel_id);
  }
  if (cartel_id == -1) {
    return std::set<int>();
  }
  return this->cartel_author_map.at(cartel_id);
}

std::vector<int> Graph::GetAuthorPublications(int author_id) const {
  if (db_) {
    return db_->Authors().GetAuthorPublications(author_id);
  }
  static const std::vector<int> empty_vec;
  if (!this->author_publication_map.contains(author_id)) {
    return empty_vec;
  }
  return this->author_publication_map.at(author_id);
}

int Graph::GetCartelID(int author) const {
  if (db_) {
    return db_->Authors().GetCartelID(author);
  }
  if (this->author_cartel_map.contains(author)) {
    return this->author_cartel_map.at(author);
  }
  return -1;
}

void Graph::UpdateAuthorPublicationMap(int author, int node) {
  if (db_) {
    db_->Authors().UpdateAuthorPublication(author, node);
    return;
  }
  this->author_publication_map[author].push_back(node);
}

void Graph::ComputeAuthorReputations() {
  if (db_) {
    db_->Authors().ComputeAuthorReputations(db_->CSR());
    return;
  }
  std::vector<int> freq;
  for (const auto &[author_id, birth_year] : this->author_birth_year_map) {
    auto it = this->author_publication_map.find(author_id);
    if (it == this->author_publication_map.end()) {
      std::cerr << "Missing author_id in author_publication_map: " << author_id
                << "\n";
      throw std::out_of_range("Missing author_id in author_publication_map");
    }
    const std::vector<int> &publication_vec = it->second;
    int h_index = 0;
    if (!publication_vec.empty()) {
      size_t n = publication_vec.size();
      if (freq.size() <= n) {
        freq.resize(n + 1, 0);
      } else {
        std::fill(freq.begin(), freq.begin() + n + 1, 0);
      }
      for (size_t i = 0; i < publication_vec.size(); i++) {
        int current_publication = publication_vec[i];
        size_t current_publication_in_degree =
            this->GetInDegree(current_publication);
        freq[std::min(n, current_publication_in_degree)]++;
      }
      h_index = static_cast<int>(n);
      int num_candidate_papers = freq[h_index];
      while (h_index > num_candidate_papers) {
        h_index--;
        num_candidate_papers += freq[h_index];
      }
    }
    this->author_reputation_map[author_id] = h_index;
  }
}

int Graph::ComputeSingleAuthorReputation(int author_id) {
  if (db_) {
    return db_->Authors().ComputeSingleAuthorReputation(author_id, db_->CSR());
  }
  auto it = this->author_publication_map.find(author_id);
  if (it == this->author_publication_map.end() || it->second.empty()) {
    this->author_reputation_map[author_id] = 0;
    return 0;
  }
  const auto &pubs = it->second;
  size_t n = pubs.size();
  std::vector<int> freq(n + 1, 0);
  for (int p : pubs) {
    size_t in_deg = this->GetInDegree(p);
    freq[std::min(n, in_deg)]++;
  }
  int h = static_cast<int>(n);
  int cum = freq[h];
  while (h > cum) {
    h--;
    cum += freq[h];
  }
  this->author_reputation_map[author_id] = h;
  return h;
}

void Graph::SaveInitialAuthorReputations() {
  if (db_) {
    size_t n = db_->Nodes().GetNodeCount();
    for (size_t node_id = 0; node_id < n; ++node_id) {
      int auth = db_->Nodes().GetAuthorId(static_cast<int>(node_id));
      db_->Nodes().SetInitialAuthorReputation(static_cast<int>(node_id), db_->Authors().GetAuthorReputation(auth));
    }
    return;
  }
  for (const auto &node_id : this->GetNodeSet()) {
    int author_id = this->GetAuthorId(node_id);
    auto it = this->author_reputation_map.find(author_id);
    this->SetInitialAuthorReputation(node_id, (it != this->author_reputation_map.end()) ? it->second : 0);
  }
}

int Graph::GetAuthorReputationForNode(int node) const {
  if (db_) {
    int auth = db_->Nodes().GetAuthorId(node);
    return db_->Authors().GetAuthorReputation(auth);
  }
  int author_id = this->GetAuthorId(node);
  auto it = this->author_reputation_map.find(author_id);
  if (it != this->author_reputation_map.end()) {
    return it->second;
  }
  return 0;
}

int Graph::GetNextNumAuthors() {
  if (this->num_authors_bag_vec.empty()) {
    return 1;
  }
  std::uniform_int_distribution<int> num_authors_uniform_distribution{
      0, (int)(this->num_authors_bag_vec.size() - 1)};
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  int index_uniform = num_authors_uniform_distribution(generator);
  return this->num_authors_bag_vec[index_uniform];
}

void Graph::ReadNumAuthorsBag() {
  char delimiter = ',';
  std::ifstream out_degree_bag_stream(this->num_authors_bag);
  if (!out_degree_bag_stream.is_open()) {
    std::cerr << "Error: Could not open num_authors_bag file: " << this->num_authors_bag << std::endl;
    throw std::runtime_error("Could not open num_authors_bag file: " + this->num_authors_bag);
  }
  std::string line;
  int line_no = 0;
  while (std::getline(out_degree_bag_stream, line)) {
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.size() == 0) {
      break;
    }
    if (line_no != 0) {
      this->num_authors_bag_vec.push_back(std::stoi(current_line[1]));
    }
    line_no++;
  }
  if (this->num_authors_bag_vec.empty()) {
    std::cerr << "Error: num_authors_bag file is empty or formatted incorrectly: " << this->num_authors_bag << std::endl;
    throw std::runtime_error("num_authors_bag file is empty: " + this->num_authors_bag);
  }
}

void Graph::UpdateAuthorManual(int author_id) {
  if (db_) {
    db_->Authors().UpdateAuthorManual(author_id);
    return;
  }
  int num_publications_by_author =
      this->author_publication_map.contains(author_id)
          ? this->author_publication_map.at(author_id).size()
          : 0;
  auto &vec = this->publication_count_to_author_map[num_publications_by_author];
  auto it = std::find(vec.begin(), vec.end(), author_id);
  if (it != vec.end()) {
    *it = vec.back();
    vec.pop_back();
  }
  this->publication_count_to_author_map[num_publications_by_author + 1]
      .push_back(author_id);
}

int Graph::GetNextAuthor(int current_year, const std::set<int> &exclusion_set) {
  if (db_) {
    return db_->Authors().GetNextAuthor(current_year, exclusion_set);
  }
  int num_authors_with_one_paper =
      this->publication_count_to_author_map[1].size();
  bool found_valid_place = false;
  int proposed_publication_count_for_author = 2;
  int return_author = this->next_author_id;

  while (true) {
    double denom = (this->lotka_exponent == 2)
                       ? (static_cast<double>(proposed_publication_count_for_author) *
                          proposed_publication_count_for_author)
                       : std::pow(proposed_publication_count_for_author, this->lotka_exponent);
    int expected_num =
        static_cast<int>(std::round(num_authors_with_one_paper / denom));
    if (expected_num < 1) {
      break;
    }
    int actual_num =
        this->publication_count_to_author_map[proposed_publication_count_for_author]
            .size();
    int deficit = expected_num - actual_num;
    if (deficit > 1) {
      found_valid_place = true;
      break;
    }
    proposed_publication_count_for_author++;
  }

  if (found_valid_place) {
    auto &cand = this->publication_count_to_author_map[proposed_publication_count_for_author - 1];
    if (cand.empty()) {
      found_valid_place = false;
    } else {
      int upgraded_author_id = -1;
      size_t picked_idx = 0;
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::uniform_int_distribution<size_t> dist(0, cand.size() - 1);

      // Fast rejection sampling: try up to 32 random attempts
      for (int attempt = 0; attempt < 32; ++attempt) {
        size_t idx = dist(generator);
        int a = cand[idx];
        int birth_year = this->author_birth_year_map[a];
        if (current_year - birth_year < this->author_max_lifetime &&
            (exclusion_set.empty() || !exclusion_set.contains(a))) {
          upgraded_author_id = a;
          picked_idx = idx;
          break;
        }
      }

      // Fallback if rejection sampling didn't succeed (e.g. >90% dead or excluded)
      if (upgraded_author_id == -1) {
        std::vector<size_t> valid_indices;
        for (size_t i = 0; i < cand.size(); ++i) {
          int a = cand[i];
          int birth_year = this->author_birth_year_map[a];
          if (current_year - birth_year < this->author_max_lifetime &&
              (exclusion_set.empty() || !exclusion_set.contains(a))) {
            valid_indices.push_back(i);
          }
        }
        if (!valid_indices.empty()) {
          std::uniform_int_distribution<size_t> vdist(0, valid_indices.size() - 1);
          picked_idx = valid_indices[vdist(generator)];
          upgraded_author_id = cand[picked_idx];
        }
      }

      if (upgraded_author_id != -1) {
        return_author = upgraded_author_id;
        // O(1) swap-and-pop from candidate list
        cand[picked_idx] = cand.back();
        cand.pop_back();
        this->publication_count_to_author_map[proposed_publication_count_for_author]
            .push_back(upgraded_author_id);
      } else {
        found_valid_place = false;
      }
    }
  }

  if (!found_valid_place) {
    this->publication_count_to_author_map[1].push_back(this->next_author_id);
    this->author_birth_year_map[this->next_author_id] = current_year;
    return_author = this->next_author_id;
    this->next_author_id++;
  }

  return return_author;
}

void Graph::AddEdge(std::pair<int, int> edge) {
  if (db_) {
    return;
  }
  this->EnsureNodeCapacity(edge.first);
  this->forward_adj_list[edge.first].push_back(edge.second);
  this->SetOutDegree(edge.first, this->GetOutDegree(edge.first) + 1);
  this->EnsureNodeCapacity(edge.second);
  this->backward_adj_list[edge.second].push_back(edge.first);
  this->SetInDegree(edge.second, this->GetInDegree(edge.second) + 1);
  this->AddNode(edge.first);
  this->AddNode(edge.second);
}

void Graph::AddNode(int u) {
  if (db_) {
    return;
  }
  this->EnsureNodeCapacity(u);
  if (!this->node_exists_[u]) {
    this->node_exists_[u] = 1;
    this->node_list_.push_back(u);
    this->node_count_++;
    this->node_attributes[u].is_valid = true;
  }
}

void Graph::ReleaseHeavyStructures() {
  std::vector<std::vector<int>>().swap(this->forward_adj_list);
  std::vector<std::vector<int>>().swap(this->backward_adj_list);
  std::vector<NodeAttributes>().swap(this->node_attributes);
  std::vector<int>().swap(this->node_list_);
  std::vector<uint8_t>().swap(this->node_exists_);
  this->node_count_ = 0;
  std::unordered_map<int, std::vector<int>>().swap(this->cluster_nodes_map);
  std::unordered_map<int, std::vector<int>>().swap(this->author_publication_map);
  std::unordered_map<int, std::vector<int>>().swap(this->publication_count_to_author_map);
  std::unordered_map<int, int>().swap(this->author_birth_year_map);
  std::unordered_map<int, int>().swap(this->author_reputation_map);
  std::unordered_map<int, int>().swap(this->author_cartel_map);
  std::unordered_map<int, std::set<int>>().swap(this->cartel_author_map);
  std::set<int>().swap(this->cartel_set);
#if defined(__linux__)
  malloc_trim(0);
#endif
}

int Graph::GetNextAuthorId() const noexcept {
  if (db_) {
    return db_->Authors().GetNextAuthorId();
  }
  return this->next_author_id;
}

std::string Graph::GetGeneratorNodeString(int node) const {
  int gen = this->GetGeneratorNode(node);
  if (gen < 0) {
    return "no_generators";
  }
  return std::to_string(gen);
}

void Graph::SetGeneratorNodeString(int node, const std::string &val) {
  if (val.empty() || val == "no_generators") {
    this->SetGeneratorNode(node, -1);
  } else {
    size_t pos = val.find(';');
    int gen = std::stoi((pos == std::string::npos) ? val : val.substr(0, pos));
    this->SetGeneratorNode(node, gen);
  }
}

void Graph::PrintGraph() const {
  for (size_t u = 0; u < this->GetForwardAdjList().size(); u++) {
    const auto &u_neighbors = this->GetForwardAdjList()[u];
    for (const int &v : u_neighbors) {
      /* if (u < v) { */
      std::cout << u << "-" << v << "\n";
      /* } */
    }
  }
}

void Graph::WriteGraph(std::string output_file) const {
  if (db_) {
    db_->CSR().ExportCSV(output_file);
    return;
  }
  std::ofstream output_filehandle(output_file);
  output_filehandle << "source,target\n";
  for (size_t u = 0; u < this->GetForwardAdjList().size(); u++) {
    const auto &u_neighbors = this->GetForwardAdjList()[u];
    for (const int &v : u_neighbors) {
      /* if (u < v) { */
      output_filehandle << u << "," << v << "\n";
      /* } */
    }
  }
  output_filehandle.close();
}

void Graph::WriteAttributes(std::string auxiliary_information_file) const {
  std::ofstream auxiliary_information_filehandle(auxiliary_information_file);
  auxiliary_information_filehandle
      << "node_id,type,year,alpha,pa_weight,fit_weight,num_authors_weight,"
         "author_reputation_weight,fit_lag_duration,fit_peak_value,fit_peak_"
         "duration,in_degree,out_degree,assigned_out_degree,planted_nodes_line_"
         "number,generator_node_string,sampled_neighborhood_size,fully_random_"
         "citations,author_id,num_authors,initial_author_reputation,final_"
         "author_reputation,cartel_id,cluster_id\n";

  if (db_) {
    size_t n = db_->Nodes().GetNodeCount();
    for (size_t i = 0; i < n; ++i) {
      int node_id = static_cast<int>(i);
      std::string node_type =
          (db_->Nodes().GetType(node_id) == ladybug::NodeType::Seed ? "seed" : "agent");
      int year = db_->Nodes().GetYear(node_id);
      double pa_weight = -1;
      double fit_weight = -1;
      double num_authors_weight = -1;
      double author_reputation_weight = -1;
      double alpha = -1;
      int fit_lag_duration = db_->Nodes().GetFitnessLagDuration(node_id);
      int fit_peak_value = db_->Nodes().GetFitnessPeakValue(node_id);
      int fit_peak_duration = db_->Nodes().GetFitnessPeakDuration(node_id);
      int out_degree = static_cast<int>(db_->CSR().GetOutDegree(node_id));
      int assigned_out_degree = -1;
      int in_degree = static_cast<int>(db_->CSR().GetInDegree(node_id));
      int author = db_->Nodes().GetAuthorId(node_id);
      int num_authors = db_->Nodes().GetNumAuthors(node_id);
      int planted_nodes_line_number = -1;
      std::string generator_node_string = "no_generators";
      int neighborhood_size = -1;
      int fully_random_citations = -1;
      int initial_author_reputation = db_->Nodes().GetInitialAuthorReputation(node_id);
      int final_author_reputation = db_->Authors().GetAuthorReputation(author);
      int cartel_id = db_->Authors().GetCartelID(author);
      int cluster_id = db_->Nodes().GetClusterId(node_id);

      if (db_->Nodes().GetType(node_id) != ladybug::NodeType::Seed) {
        alpha = db_->Nodes().GetAlpha(node_id);
        pa_weight = db_->Nodes().GetPaWeight(node_id);
        fit_weight = db_->Nodes().GetFitWeight(node_id);
        num_authors_weight = db_->Nodes().GetNumAuthorsWeight(node_id);
        author_reputation_weight = db_->Nodes().GetAuthorRepWeight(node_id);
        assigned_out_degree = db_->Nodes().GetAssignedOutDegree(node_id);
        int gen = db_->Nodes().GetGeneratorNode(node_id);
        if (gen >= 0) {
          generator_node_string = std::to_string(gen);
        }
        planted_nodes_line_number = db_->Nodes().GetPlantedLineNumber(node_id);
        neighborhood_size = db_->Nodes().GetSampledNeighborhoodSize(node_id);
        fully_random_citations = db_->Nodes().GetFullyRandomCitations(node_id);
      }

      auxiliary_information_filehandle
          << node_id << "," << node_type << "," << year << "," << alpha << ","
          << pa_weight << "," << fit_weight << "," << num_authors_weight << ","
          << author_reputation_weight << "," << fit_lag_duration << ","
          << fit_peak_value << "," << fit_peak_duration << "," << in_degree << ","
          << out_degree << "," << assigned_out_degree << ","
          << planted_nodes_line_number << "," << generator_node_string << ","
          << neighborhood_size << "," << fully_random_citations << "," << author
          << "," << num_authors << "," << initial_author_reputation << ","
          << final_author_reputation << "," << cartel_id << "," << cluster_id
          << "\n";
    }
    auxiliary_information_filehandle.close();
    return;
  }

  for (const auto &node_id : this->GetNodeSet()) {
    std::string node_type =
        (this->GetType(node_id) == NodeType::Seed ? "seed" : "agent");
    int year = this->GetYear(node_id);
    double pa_weight = -1;
    double fit_weight = -1;
    double num_authors_weight = -1;
    double author_reputation_weight = -1;
    double alpha = -1;
    int fit_lag_duration = this->GetFitnessLagDuration(node_id);
    int fit_peak_value = this->GetFitnessPeakValue(node_id);
    int fit_peak_duration = this->GetFitnessPeakDuration(node_id);
    int out_degree = this->GetOutDegree(node_id);
    int assigned_out_degree = -1;
    int in_degree = this->GetInDegree(node_id);
    int author = this->GetAuthorId(node_id);
    int num_authors = this->GetNumAuthors(node_id);
    int planted_nodes_line_number = -1;
    std::string generator_node_string = "no_generators";
    int neighborhood_size = -1;
    int fully_random_citations = -1;
    int initial_author_reputation = this->GetInitialAuthorReputation(node_id);
    int final_author_reputation = this->GetAuthorReputationForNode(node_id);
    int cartel_id = this->GetCartelID(author);
    int cluster_id = this->GetCommunityAssignment(node_id);
    if (this->GetType(node_id) != NodeType::Seed) {
      alpha = this->GetAlpha(node_id);
      pa_weight = this->GetPaWeight(node_id);
      fit_weight = this->GetFitWeight(node_id);
      num_authors_weight = this->GetNumAuthorsWeight(node_id);
      author_reputation_weight = this->GetAuthorReputationWeight(node_id);
      assigned_out_degree = this->GetAssignedOutDegree(node_id);
      generator_node_string = this->GetGeneratorNodeString(node_id);
      if (this->HasPlantedNodesLineNumber(node_id)) {
        planted_nodes_line_number = this->GetPlantedNodesLineNumber(node_id);
      }
      neighborhood_size = this->GetSampledNeighborhoodSize(node_id);
      fully_random_citations = this->GetFullyRandomCitations(node_id);
    }
    auxiliary_information_filehandle
        << node_id << "," << node_type << "," << year << "," << alpha << ","
        << pa_weight << "," << fit_weight << "," << num_authors_weight << ","
        << author_reputation_weight << "," << fit_lag_duration << ","
        << fit_peak_value << "," << fit_peak_duration << "," << in_degree << ","
        << out_degree << "," << assigned_out_degree << ","
        << planted_nodes_line_number << "," << generator_node_string << ","
        << neighborhood_size << "," << fully_random_citations << "," << author
        << "," << num_authors << "," << initial_author_reputation << ","
        << final_author_reputation << "," << cartel_id << "," << cluster_id
        << "\n";
  }
  auxiliary_information_filehandle.close();
}

void Graph::AddNodeToCluster(int node, int cluster_id) {
  if (cluster_id >= 0) {
    if (db_) {
      db_->Clusters().AddNodeToCluster(node, cluster_id);
      return;
    }
    this->cluster_nodes_map[cluster_id].push_back(node);
  }
}

const std::vector<int> &Graph::GetClusterNodes(int cluster_id) const {
  if (db_) {
    return db_->Clusters().GetClusterNodes(cluster_id);
  }
  static const std::vector<int> empty_cluster;
  auto it = this->cluster_nodes_map.find(cluster_id);
  if (it != this->cluster_nodes_map.end()) {
    return it->second;
  }
  return empty_cluster;
}

int Graph::GetClusterSize(int cluster_id, int current_year) const {
  if (db_) {
    return db_->Clusters().GetClusterSize(cluster_id, db_->Nodes(), current_year);
  }
  auto it = this->cluster_nodes_map.find(cluster_id);
  if (it != this->cluster_nodes_map.end()) {
    if (current_year == -1) {
      return static_cast<int>(it->second.size());
    }
    const auto &nodes = it->second;
    auto comp = [this](int node, int yr) {
      return this->GetYear(node) < yr;
    };
    auto lower = std::lower_bound(nodes.begin(), nodes.end(), current_year, comp);
    return static_cast<int>(lower - nodes.begin());
  }
  return 0;
}

#include "graph.h"
#include "utils.h"

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
  std::string line;
  int line_no = 0;
  while (std::getline(input_edgelist, line)) {
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
      int integer_citing = std::stoi(current_line[0]);
      int integer_cited = std::stoi(current_line[1]);
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
  std::string line;
  int line_no = 0;
  std::vector<std::pair<int, int>> node_year_vec;
  while (std::getline(input_nodelist, line)) {
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
      int integer_node =
          std::stoi(current_line[header_to_index_map["node_id"]]);
      int integer_year = std::stoi(current_line[header_to_index_map["year"]]);
      this->SetYear(integer_node, integer_year);
      node_year_vec.push_back({integer_node, integer_year});
      if (this->start_from_checkpoint) {
        std::string type_string = current_line[header_to_index_map["type"]];
        this->SetType(integer_node,
                      type_string == "seed" ? NodeType::Seed : NodeType::Agent);
        double alpha = std::stod(current_line[header_to_index_map["alpha"]]);
        this->SetAlpha(integer_node, alpha);
        double pa_weight =
            std::stod(current_line[header_to_index_map["pa_weight"]]);
        this->SetPaWeight(integer_node, pa_weight);
        double fit_weight =
            std::stod(current_line[header_to_index_map["fit_weight"]]);
        this->SetFitWeight(integer_node, fit_weight);
        int fit_lag_duration =
            std::stoi(current_line[header_to_index_map["fit_lag_duration"]]);
        this->SetFitnessLagDuration(integer_node, fit_lag_duration);
        int fit_peak_value =
            std::stoi(current_line[header_to_index_map["fit_peak_value"]]);
        this->SetFitnessPeakValue(integer_node, fit_peak_value);
        int fit_peak_duration =
            std::stoi(current_line[header_to_index_map["fit_peak_duration"]]);
        this->SetFitnessPeakDuration(integer_node, fit_peak_duration);
        int assigned_out_degree =
            std::stoi(current_line[header_to_index_map["assigned_out_degree"]]);
        this->SetAssignedOutDegree(integer_node, assigned_out_degree);
        int planted_nodes_line_number = std::stoi(
            current_line[header_to_index_map["planted_nodes_line_number"]]);
        this->SetPlantedNodesLineNumber(integer_node,
                                        planted_nodes_line_number);
        int sampled_neighborhood_size = std::stoi(
            current_line[header_to_index_map["sampled_neighborhood_size"]]);
        this->SetSampledNeighborhoodSize(integer_node,
                                         sampled_neighborhood_size);
        std::string generator_node_string =
            current_line[header_to_index_map["generator_node_string"]];
        this->SetGeneratorNodeString(integer_node, generator_node_string);
        int fully_random_citations = std::stoi(
            current_line[header_to_index_map["fully_random_citations"]]);
        this->SetFullyRandomCitations(integer_node, fully_random_citations);
        int author = std::stoi(current_line[header_to_index_map["author_id"]]);
        this->SetAuthorId(integer_node, author);
        int initial_author_reputation = std::stoi(
            current_line[header_to_index_map["initial_author_reputation"]]);
        this->SetInitialAuthorReputation(integer_node,
                                         initial_author_reputation);
        int num_authors =
            std::stoi(current_line[header_to_index_map["num_authors"]]);
        this->SetNumAuthors(integer_node, num_authors);
        int cartel_id =
            std::stoi(current_line[header_to_index_map["cartel_id"]]);
        if (cartel_id != -1) {
          this->SetCartelIdAttr(integer_node, cartel_id);
          this->cartel_set.insert(cartel_id);
          this->SetCartelID(author, cartel_id);
        }
        if (this->author_birth_year_map.contains(author)) {
          this->author_birth_year_map[author] =
              std::min(this->author_birth_year_map[author], integer_year);
        } else {
          this->author_birth_year_map[author] = integer_year;
        }
        this->UpdateAuthorPublicationMap(author, integer_node);
        this->next_author_id = std::max(this->next_author_id, author);
      } else {
        int fitness_lag_uniform = 0; // MARK: hard coded to be static fitness
        int fitness_peak_uniform =
            1000; // MARK: hard coded to be static fitness
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
    int previous_year = node_year_vec.at(previous_index).second;
    for (size_t i = 0; i < node_year_vec.size(); i++) {
      int current_node_id = node_year_vec[i].first;
      int current_year = node_year_vec[i].second;
      int author_id = this->GetNextAuthor(current_year, std::set<int>());
      this->SetAuthorId(current_node_id, author_id);
      this->UpdateAuthorPublicationMap(author_id, current_node_id);
      if (previous_year != current_year) {
        this->ComputeAuthorReputations();
        for (size_t j = previous_index; j < i; j++) {
          int node_id = node_year_vec.at(j).first;
          int author_id = this->GetAuthorId(node_id);
          this->SetInitialAuthorReputation(
              node_id, this->author_reputation_map.at(author_id));
        }
        previous_index = i;
        previous_year = node_year_vec.at(previous_index).second;
      }
    }
    if (previous_index < node_year_vec.size()) {
      this->ComputeAuthorReputations();
      for (size_t j = previous_index; j < node_year_vec.size(); j++) {
        int node_id = node_year_vec.at(j).first;
        int author_id = this->GetAuthorId(node_id);
        this->SetInitialAuthorReputation(
            node_id, this->author_reputation_map.at(author_id));
      }
    }
  }
}

std::set<int> Graph::GetCartelSet() const { return this->cartel_set; }

void Graph::SetCartelID(int author, int cartel_id) {
  this->author_cartel_map[author] = cartel_id;
  this->cartel_author_map[cartel_id].insert(author);
}

std::set<int> Graph::GetCartelAuthors(int cartel_id) const {
  if (cartel_id == -1) {
    return std::set<int>();
  }
  return this->cartel_author_map.at(cartel_id);
}

std::vector<int> Graph::GetAuthorPublications(int author_id) const {
  static const std::vector<int> empty_vec;
  if (!this->author_publication_map.contains(author_id)) {
    return empty_vec;
  }
  return this->author_publication_map.at(author_id);
}

int Graph::GetCartelID(int author) const {
  if (this->author_cartel_map.contains(author)) {
    return this->author_cartel_map.at(author);
  }
  return -1;
}

void Graph::UpdateAuthorPublicationMap(int author, int node) {
  this->author_publication_map[author].push_back(node);
}

void Graph::ComputeAuthorReputations() {
  for (const auto &[author_id, birth_year] : this->author_birth_year_map) {
    if (!this->author_publication_map.contains(author_id)) {
      std::cerr << "Missing author_id in author_publication_map: " << author_id
                << "\n";
      throw std::out_of_range("Missing author_id in author_publication_map");
    }
    const std::vector<int> &publication_vec =
        this->author_publication_map.at(author_id);
    int h_index = 0;
    if (!publication_vec.empty()) {
      std::unordered_map<int, int> freq_map;
      for (size_t i = 0; i < publication_vec.size(); i++) {
        int current_publication = publication_vec.at(i);
        size_t current_publication_in_degree =
            this->GetInDegree(current_publication);
        freq_map[std::min(publication_vec.size(),
                          current_publication_in_degree)]++;
      }
      h_index = publication_vec.size();
      int num_candidate_papers = freq_map[h_index];
      while (h_index > num_candidate_papers) {
        h_index--;
        num_candidate_papers += freq_map[h_index];
      }
    }
    this->author_reputation_map[author_id] = h_index;
  }
}

void Graph::SaveInitialAuthorReputations() {
  for (const auto &node_id : this->GetNodeSet()) {
    int author_id = this->GetAuthorId(node_id);
    this->SetInitialAuthorReputation(node_id,
                                     this->author_reputation_map.at(author_id));
  }
}

int Graph::GetAuthorReputationForNode(int node) const {
  int author_id = this->GetAuthorId(node);
  return this->author_reputation_map.at(author_id);
}

int Graph::GetNextNumAuthors() {
  std::uniform_int_distribution<int> num_authors_uniform_distribution{
      0, (int)(this->num_authors_bag_vec.size() - 1)};
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  int index_uniform = num_authors_uniform_distribution(generator);
  return this->num_authors_bag_vec[index_uniform];
}

void Graph::ReadNumAuthorsBag() {
  char delimiter = ',';
  std::ifstream out_degree_bag_stream(this->num_authors_bag);
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
}

void Graph::UpdateAuthorManual(int author_id) {
  int num_publications_by_author =
      this->author_publication_map.contains(author_id)
          ? this->author_publication_map.at(author_id).size()
          : 0;
  std::erase(this->publication_count_to_author_map[num_publications_by_author],
             author_id);
  this->publication_count_to_author_map[num_publications_by_author + 1]
      .push_back(author_id);
}

int Graph::GetNextAuthor(int current_year, const std::set<int> &exclusion_set) {
  int num_authors_with_one_paper =
      this->publication_count_to_author_map[1].size();
  bool found_valid_place = false;
  int proposed_publication_count_for_author = 2;
  int return_author = this->next_author_id;
  while (std::round(num_authors_with_one_paper /
                    pow(proposed_publication_count_for_author,
                        this->lotka_exponent)) >= 1) {
    int expected_num_authors_with_proposed_publication_count = std::round(
        num_authors_with_one_paper /
        pow(proposed_publication_count_for_author, this->lotka_exponent));
    int actual_num_authors_with_proposed_publication_count =
        this->publication_count_to_author_map
            [proposed_publication_count_for_author]
                .size();
    int deficit = expected_num_authors_with_proposed_publication_count -
                  actual_num_authors_with_proposed_publication_count;
    if (deficit > 1) {
      found_valid_place = true;
      break;
    }
    proposed_publication_count_for_author++;
  }
  if (found_valid_place) {
    std::vector<int> living_authors;
    for (size_t i = 0; i < this->publication_count_to_author_map
                               [proposed_publication_count_for_author - 1]
                                   .size();
         i++) {
      int current_author = this->publication_count_to_author_map
                               [proposed_publication_count_for_author - 1][i];
      if (current_year - this->author_birth_year_map[current_author] <
              this->author_max_lifetime &&
          !exclusion_set.contains(current_author)) {
        living_authors.push_back(current_author);
      }
    }
    if (living_authors.size() > 0) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::ranges::shuffle(living_authors, generator);
      int upgraded_author_id = living_authors.back();
      return_author = upgraded_author_id;
      std::erase(this->publication_count_to_author_map
                     [proposed_publication_count_for_author - 1],
                 upgraded_author_id);
      this->publication_count_to_author_map
          [proposed_publication_count_for_author]
              .push_back(upgraded_author_id);
    } else {
      found_valid_place = false;
    }
  }
  if (!found_valid_place) {
    this->publication_count_to_author_map[1].push_back(this->next_author_id);
    this->author_birth_year_map[this->next_author_id] = current_year;
    this->next_author_id++;
  }
  return return_author;
}

void Graph::AddEdge(std::pair<int, int> edge) {
  this->EnsureNodeCapacity(edge.first);
  this->forward_adj_list[edge.first].push_back(edge.second);
  this->SetOutDegree(edge.first, this->GetOutDegree(edge.first) + 1);
  this->EnsureNodeCapacity(edge.second);
  this->backward_adj_list[edge.second].push_back(edge.first);
  this->SetInDegree(edge.second, this->GetInDegree(edge.second) + 1);
  this->AddNode(edge.first);
  this->AddNode(edge.second);
}

void Graph::AddNode(int u) { this->node_set.insert(u); }

const std::set<int> &Graph::GetNodeSet() const { return this->node_set; }
const std::vector<std::vector<int>> &Graph::GetForwardAdjList() const {
  return this->forward_adj_list;
}

const std::vector<std::vector<int>> &Graph::GetBackwardAdjList() const {
  return this->backward_adj_list;
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
    this->cluster_nodes_map[cluster_id].push_back(node);
  }
}

const std::vector<int> &Graph::GetClusterNodes(int cluster_id) const {
  static const std::vector<int> empty_cluster;
  auto it = this->cluster_nodes_map.find(cluster_id);
  if (it != this->cluster_nodes_map.end()) {
    return it->second;
  }
  return empty_cluster;
}

int Graph::GetClusterSize(int cluster_id) const {
  auto it = this->cluster_nodes_map.find(cluster_id);
  if (it != this->cluster_nodes_map.end()) {
    return it->second.size();
  }
  return 0;
}

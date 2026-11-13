#include "abm.h"
#include "metrics_engine.h"
#include "utils.h"
#include <algorithm>
#include <iomanip>
#include <span>
#pragma omp declare reduction(                                                 \
        merge_int_pair_vecs : std::vector<std::pair<int, int>> : omp_out       \
            .insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(                                                 \
        merge_str_int_pair_vecs : std::vector<                                 \
                std::pair<std::string, int>> : omp_out                         \
            .insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(                                                 \
        merge_int_vecs : std::vector<int> : omp_out.insert(                    \
                omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp declare reduction(                                                 \
        custom_merge_vec_int : std::vector<std::pair<int, int>> : omp_out      \
            .insert(omp_out.end(), omp_in.begin(), omp_in.end()))              \
    initializer(omp_priv = decltype(omp_orig){})

std::vector<double> ABM::GetBinnedRecencyProbabilities() {
  std::vector<double> binned_recency_probabilities(this->neighborhood_search->num_bins, 0.0);
  double binned_recency_sum = 0;
  for (const auto &[year_diff, count] : this->recency_counts_map) {
    int current_bin_index = this->neighborhood_search->GetBinIndex(year_diff);
    if (current_bin_index >= 0 && (size_t)current_bin_index < binned_recency_probabilities.size()) {
      binned_recency_probabilities[current_bin_index] += count;
      binned_recency_sum += count;
    }
  }
  if (binned_recency_sum > 0) {
    for (size_t i = 0; i < binned_recency_probabilities.size(); ++i) {
      binned_recency_probabilities[i] /= binned_recency_sum;
    }
  }
  return binned_recency_probabilities;
}

std::vector<int> ABM::BuildContinuousNodeMapping(Graph *graph) {
  int max_node = 0;
  for (int node : graph->GetNodeSet()) {
    if (node > max_node) {
      max_node = node;
    }
  }
  int estimated_final_size = this->GetFinalGraphSize(graph);
  int capacity = std::max(estimated_final_size * 2, max_node + estimated_final_size + 1000);
  std::vector<int> continuous_node_mapping(capacity, -1);
  int current_idx = 0;
  for (auto const &node : graph->GetNodeSet()) {
    if ((size_t)node >= continuous_node_mapping.size()) {
      continuous_node_mapping.resize(node + 10000, -1);
    }
    continuous_node_mapping[node] = current_idx;
    current_idx++;
  }
  this->next_node_id = current_idx;
  return continuous_node_mapping;
}

std::vector<int> ABM::ReverseMapping(const std::vector<int> &mapping) {
  int max_val = 0;
  for (int val : mapping) {
    if (val > max_val) {
      max_val = val;
    }
  }
  std::vector<int> reverse_mapping(max_val + 1, -1);
  for (size_t key = 0; key < mapping.size(); ++key) {
    int val = mapping[key];
    if (val >= 0) {
      if ((size_t)val >= reverse_mapping.size()) {
        reverse_mapping.resize(val + 1024, -1);
      }
      reverse_mapping[val] = static_cast<int>(key);
    }
  }
  return reverse_mapping;
}

void ABM::InitializeFitness(Graph *graph) {
  this->AssignPeakFitnessValues(graph, graph->GetNodeSet());
  this->AssignFitnessLagDuration(graph, graph->GetNodeSet());
  this->AssignFitnessPeakDuration(graph, graph->GetNodeSet());
}

std::unordered_map<int, int>
ABM::PlantNodes(Graph *graph, std::span<double> pa_weight_span,
                std::span<double> fit_weight_span,
                std::span<double> num_authors_weight_span,
                std::span<double> author_reputation_weight_span,
                std::span<int> out_degree_span, std::span<double> alpha_span,
                std::span<int> fitness_lag_duration_span,
                std::span<int> fitness_peak_value_span,
                std::span<int> fitness_peak_duration_span,
                std::span<int> num_authors_span,
                std::span<int> planted_author_id_span,
                std::span<int> planted_cluster_id_span) {
  int current_graph_size = graph->GetNodeSet().size();
  this->initial_graph_size = current_graph_size;
  const std::unordered_map<std::string, std::pair<std::string, void *>>
      column_header_to_type_span_map = {
          {"pa_weight", {"double", (void *)this->pa_weight_vec.data()}},
          {"fit_weight", {"double", (void *)this->fit_weight_vec.data()}},
          {"num_authors_weight",
           {"double", (void *)this->num_authors_weight_vec.data()}},
          {"author_reputation_weight",
           {"double", (void *)this->author_reputation_weight_vec.data()}},
          {"out_degree", {"int", (void *)this->out_degree_vec.data()}},
          {"alpha", {"double", (void *)this->alpha_vec.data()}},
          {"fit_lag_duration",
           {"int", (void *)this->fitness_lag_duration_vec.data()}},
          {"fit_peak_value",
           {"int", (void *)this->fitness_peak_value_vec.data()}},
          {"fit_peak_duration",
           {"int", (void *)this->fitness_peak_duration_vec.data()}},
          {"num_authors",
           {"int",
            (void *)(this->num_authors_vec.data() + this->initial_graph_size)}},
          {"author_id", {"int", (void *)(this->planted_author_id_vec.data())}},
          {"cluster_id",
           {"int", (void *)(this->planted_cluster_id_vec.data())}},
          {"community_assignment",
           {"int", (void *)(this->planted_cluster_id_vec.data())}}};
  this->planted_nodes_line_number_map.clear();
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  for (int current_relative_year = 0;
       current_relative_year < this->num_cycles; current_relative_year++) {
    int num_new_nodes = std::ceil(current_graph_size * this->growth_rate);
    int year_start_index = current_graph_size;
    int year_end_index = current_graph_size + num_new_nodes - 1;
    if (this->planted_nodes_map.count(current_relative_year) && num_new_nodes > 0) {
      std::unordered_set<int> selected;
      std::uniform_int_distribution<int> new_nodes_distribution{
          year_start_index, year_end_index};
      std::unordered_map<int, std::unordered_map<std::string, std::string>>
          current_year_map = this->planted_nodes_map.at(current_relative_year);
      for (auto const &[line_no, line_map] : current_year_map) {
        int chosen_agent_index = new_nodes_distribution(generator);
        while (selected.contains(chosen_agent_index)) {
          chosen_agent_index = new_nodes_distribution(generator);
        }
        selected.insert(chosen_agent_index);
        this->planted_nodes_line_number_map[chosen_agent_index -
                                            this->initial_graph_size] = line_no;
        for (auto const &[planted_feature_name, planted_feature_value] :
             line_map) {
          if (column_header_to_type_span_map.contains(planted_feature_name)) {
            std::string current_variable_type =
                column_header_to_type_span_map.at(planted_feature_name).first;
            void *current_variable_span =
                column_header_to_type_span_map.at(planted_feature_name).second;
            if (current_variable_type == "double") {
              ((double *)current_variable_span)[chosen_agent_index -
                                                this->initial_graph_size] =
                  std::stod(planted_feature_value);
            } else if (current_variable_type == "int") {
              ((int *)current_variable_span)[chosen_agent_index -
                                             this->initial_graph_size] =
                  std::stoi(planted_feature_value);
            }
          }
        }
      }
    }
    current_graph_size += num_new_nodes;
  }
  return this->planted_nodes_line_number_map;
}

int ABM::GetMaxYear(Graph *graph) {
  int max_year = -1;
  bool is_first = true;
  for (auto const &node : graph->GetNodeSet()) {
    int current_node_year = graph->GetYear(node);
    if (is_first) {
      max_year = current_node_year;
      is_first = false;
    }
    if (current_node_year > max_year) {
      max_year = current_node_year;
    }
  }
  return max_year;
}

int ABM::GetMaxNode(Graph *graph) {
  int max_node = -1;
  bool is_first = true;
  for (auto const &node : graph->GetNodeSet()) {
    if (is_first) {
      max_node = node;
      is_first = false;
    }
    if (node > max_node) {
      max_node = node;
    }
  }
  return max_node;
}

int ABM::GetFinalGraphSize(Graph *graph) {
  int current_graph_size = graph->GetNodeSet().size();
  for (int i = 0; i < this->num_cycles; i++) {
    int num_new_nodes = std::ceil(current_graph_size * this->growth_rate);
    current_graph_size += num_new_nodes;
  }
  return current_graph_size;
}

void ABM::UpdateGraphAttributesWeights(
    Graph *graph, int next_node_id, std::span<double> pa_weight_span,
    std::span<double> fit_weight_span,
    std::span<double> num_authors_weight_span,
    std::span<double> author_reputation_weight_span, int len) {
  for (int i = 0; i < len; i++) {
    int current_node_id = next_node_id + i;
    graph->SetPaWeight(current_node_id, pa_weight_span[i]);
    graph->SetFitWeight(current_node_id, fit_weight_span[i]);
    graph->SetNumAuthorsWeight(current_node_id, num_authors_weight_span[i]);
    graph->SetAuthorReputationWeight(current_node_id,
                                     author_reputation_weight_span[i]);
  }
}

void ABM::UpdateGraphAttributesNumAuthors(
    Graph *graph, const std::vector<int> &continuous_node_mapping,
    std::span<int> num_authors_span) {
  for (auto const &node_id : graph->GetNodeSet()) {
    int continuous_id = ((size_t)node_id < continuous_node_mapping.size())
                            ? continuous_node_mapping[node_id]
                            : node_id;
    graph->SetNumAuthors(node_id, num_authors_span[continuous_id]);
  }
}

void ABM::UpdateGraphAttributesInitialAuthorReputations(
    Graph *graph, const std::vector<int> &new_nodes_vec) {
  for (size_t i = 0; i < new_nodes_vec.size(); i++) {
    int current_node_id = new_nodes_vec.at(i);
    int current_author_reputation =
        graph->GetAuthorReputationForNode(current_node_id);
    graph->SetInitialAuthorReputation(current_node_id,
                                      current_author_reputation);
  }
}

void ABM::UpdateGraphAttributesFitnesses(
    Graph *graph, const std::vector<int> &new_nodes_vec,
    const std::vector<int> &continuous_node_mapping,
    std::span<int> fitness_lag_duration_span,
    std::span<int> fitness_peak_value_span,
    std::span<int> fitness_peak_duration_span, int initial_graph_size) {
  for (size_t i = 0; i < new_nodes_vec.size(); i++) {
    int current_node_id = new_nodes_vec.at(i);
    int continuous_id = ((size_t)current_node_id < continuous_node_mapping.size())
                            ? continuous_node_mapping[current_node_id]
                            : current_node_id;
    int current_weight_span_index = continuous_id - initial_graph_size;
    graph->SetFitnessLagDuration(
        current_node_id, fitness_lag_duration_span[current_weight_span_index]);
    graph->SetFitnessPeakValue(
        current_node_id, fitness_peak_value_span[current_weight_span_index]);
    graph->SetFitnessPeakDuration(
        current_node_id, fitness_peak_duration_span[current_weight_span_index]);
  }
}

void ABM::UpdateGraphAttributesPlantedNodesLineNumbers(
    Graph *graph, int next_node_id,
    const std::unordered_map<int, int> &planted_nodes_line_number_map) {
  for (auto const &[weight_span_index, line_no] :
       planted_nodes_line_number_map) {
    int current_node_id = next_node_id + weight_span_index;
    graph->SetPlantedNodesLineNumber(current_node_id, line_no);
  }
}

void ABM::UpdateGraphAttributesAlphas(Graph *graph, int next_node_id,
                                      std::span<double> alpha_span, int len) {
  for (int i = 0; i < len; i++) {
    int current_node_id = next_node_id + i;
    graph->SetAlpha(current_node_id, alpha_span[i]);
  }
}

void ABM::UpdateGraphAttributesOutDegrees(Graph *graph, int next_node_id,
                                          std::span<int> out_degree_span,
                                          int len) {
  for (int i = 0; i < len; i++) {
    int current_node_id = next_node_id + i;
    graph->SetAssignedOutDegree(current_node_id, out_degree_span[i]);
  }
}

std::vector<int> ABM::GetGraphAttributesGeneratorNodes(Graph *graph,
                                                       int new_node) const {
  int gen = graph->GetGeneratorNode(new_node);
  if (gen >= 0) return {gen};
  return {};
}

void ABM::UpdateGraphAttributesAuthors(Graph *graph, int new_node,
                                       int author_id) {
  graph->SetAuthorId(new_node, author_id);
  graph->UpdateAuthorPublicationMap(author_id, new_node);
}

void ABM::UpdateGraphAttributesGeneratorNodes(
    Graph *graph, int new_node, const std::vector<int> &generator_nodes,
    int explicit_cluster_id) {
  int gen = generator_nodes.empty() ? -1 : generator_nodes[0];
  int cluster_id = explicit_cluster_id;
  if (cluster_id < 0 && gen >= 0) {
    cluster_id = graph->GetCommunityAssignment(gen);
  }
  graph->SetGeneratorNode(new_node, gen);
  graph->SetCommunityAssignment(new_node, cluster_id);
  if (cluster_id >= 0) {
    graph->AddNodeToCluster(new_node, cluster_id);
  }
}

void ABM::FillSameYearSourceNodes(std::set<int> &same_year_source_nodes,
                                  int current_year_new_nodes) {
  size_t num_same_year_source_nodes =
      (size_t)std::floor(current_year_new_nodes * this->same_year_citations);
  if (num_same_year_source_nodes == 0 || current_year_new_nodes == 0) {
    return;
  }
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::uniform_int_distribution<int> int_uniform_distribution(
      0, current_year_new_nodes - 1);
  while (same_year_source_nodes.size() != num_same_year_source_nodes) {
    int current_source = int_uniform_distribution(generator);
    if (same_year_source_nodes.count(current_source) == 0) {
      same_year_source_nodes.insert(current_source);
    }
  }
}

std::vector<int> ABM::GetCartelGeneratorNodes(Graph *graph, int author_id) {
  std::vector<int> cartel_generator_nodes;
  int cartel_id = graph->GetCartelID(author_id);
  for (auto const &cartel_author_id : graph->GetCartelAuthors(cartel_id)) {
    if (cartel_author_id == author_id) {
      continue;
    }
    std::vector<int> cartel_author_publications =
        graph->GetAuthorPublications(cartel_author_id);
    for (const auto &cartel_author_publication : cartel_author_publications) {
      if (!graph->HasNode(cartel_author_publication)) {
        continue;
      }
      cartel_generator_nodes.push_back(cartel_author_publication);
    }
  }
  return cartel_generator_nodes;
}

std::vector<int> ABM::GetEligibleGeneratorNodes(
    Graph *graph, int graph_size,
    const std::vector<int> &reverse_continuous_node_mapping,
    std::span<int> in_degree_span, std::span<int> fitness_span,
    int in_degree_threshold, int fitness_threshold, int start_year,
    int current_year, int recency_threshold) {
  std::vector<std::pair<double, int>> in_degree_eligible_generator_nodes;
  std::vector<std::pair<double, int>> fitness_eligible_generator_nodes;
  std::vector<int> eligible_generator_nodes;
  if (current_year - start_year <= recency_threshold) {
    for (int i = graph_size - 1; i >= 0; i--) {
      int current_node_id = reverse_continuous_node_mapping[i];
      if ((current_year - graph->GetYear(current_node_id)) >
          recency_threshold) {
      } else {
        in_degree_eligible_generator_nodes.push_back(
            {in_degree_span[i], current_node_id});
        if (graph->GetType(current_node_id) == Graph::NodeType::Seed) {
          fitness_eligible_generator_nodes.push_back(
              {this->fitness_value_max, current_node_id});
        } else {
          fitness_eligible_generator_nodes.push_back(
              {fitness_span[i], current_node_id});
        }
      }
    }
  } else {
    for (int i = graph_size - 1; i >= 0; i--) {
      int current_node_id = reverse_continuous_node_mapping[i];
      if ((current_year - graph->GetYear(current_node_id)) >
          recency_threshold) {
        break;
      }
      in_degree_eligible_generator_nodes.push_back(
          {in_degree_span[i], current_node_id});
      fitness_eligible_generator_nodes.push_back(
          {fitness_span[i], current_node_id});
    }
  }
  int in_degree_top_n_nodes_index =
      (int)((in_degree_eligible_generator_nodes.size() - 1) *
            ((double)in_degree_threshold / 100));
  int fitness_top_n_nodes_index =
      (int)((fitness_eligible_generator_nodes.size() - 1) *
            ((double)fitness_threshold / 100));

  std::nth_element(
      in_degree_eligible_generator_nodes.begin(),
      in_degree_eligible_generator_nodes.begin() + in_degree_top_n_nodes_index,
      in_degree_eligible_generator_nodes.end(), [](auto &left, auto &right) {
        return left.first > right.first; // read
      });
  std::nth_element(
      fitness_eligible_generator_nodes.begin(),
      fitness_eligible_generator_nodes.begin() + fitness_top_n_nodes_index,
      fitness_eligible_generator_nodes.end(), [](auto &left, auto &right) {
        return left.first > right.first; // read
      });
  std::set<int> eligible_in_degree_node_ids;
  std::set<int> eligible_fitness_node_ids;
  for (size_t i = 0; i < in_degree_eligible_generator_nodes.size(); i++) {
    if (in_degree_eligible_generator_nodes.at(i).first >=
        in_degree_eligible_generator_nodes.at(in_degree_top_n_nodes_index)
            .first) {
      eligible_in_degree_node_ids.insert(
          in_degree_eligible_generator_nodes.at(i).second);
    }
    if (fitness_eligible_generator_nodes.at(i).first >=
        fitness_eligible_generator_nodes.at(fitness_top_n_nodes_index).first) {
      eligible_fitness_node_ids.insert(
          fitness_eligible_generator_nodes.at(i).second);
    }
  }
  std::set_intersection(
      eligible_in_degree_node_ids.begin(), eligible_in_degree_node_ids.end(),
      eligible_fitness_node_ids.begin(), eligible_fitness_node_ids.end(),
      std::back_inserter(eligible_generator_nodes));
  return eligible_generator_nodes;
}

std::vector<int>
ABM::GetGeneratorNodesFromSet(std::vector<int> &eligible_generator_nodes) {
  std::vector<int> generator_nodes;
  if (eligible_generator_nodes.empty()) {
    return generator_nodes;
  }
  std::uniform_int_distribution<int> generator_uniform_distribution{
      0, (int)(eligible_generator_nodes.size() - 1)};
  int num_generator_nodes = 1;
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  for (int i = 0; i < num_generator_nodes; i++) {
    int continuous_generator_node = generator_uniform_distribution(generator);
    int generator_node = eligible_generator_nodes.at(continuous_generator_node);
    generator_nodes.push_back(generator_node);
  }
  return generator_nodes;
}

std::vector<int> ABM::GetGeneratorNodes(
    Graph *graph, const std::vector<int> &reverse_continuous_node_mapping) {
  std::vector<int> generator_nodes;
  std::uniform_int_distribution<int> generator_uniform_distribution{
      0, (int)(graph->GetNodeCount() - 1)};
  int num_generator_nodes = 1;
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  for (int i = 0; i < num_generator_nodes; i++) {
    int continuous_generator_node = generator_uniform_distribution(generator);
    int generator_node =
        reverse_continuous_node_mapping[continuous_generator_node];
    generator_nodes.push_back(generator_node);
  }
  return generator_nodes;
}

bool ABM::ValidateBinBoundaries() {
  this->logger.WriteToLogFile(
      std::to_string(this->neighborhood_search->bin_boundaries.size()) +
          " bins have been created",
      Log::info);
  if (this->neighborhood_search->bin_boundaries.size() == 0) {
    this->logger.WriteToLogFile("At least one bin is required", Log::error);
    return false;
  }
  if (this->neighborhood_search->bin_boundaries.at(0) != 1) {
    this->logger.WriteToLogFile("The first bin must start with year 1",
                                Log::error);
    return false;
  }
  std::string recency_bin_string;
  for (size_t i = 0; i < this->neighborhood_search->bin_boundaries.size() - 1;
       i++) {
    recency_bin_string +=
        ("[" + std::to_string(this->neighborhood_search->bin_boundaries.at(i)) +
         "," +
         std::to_string(this->neighborhood_search->bin_boundaries.at(i + 1)) +
         "), ");
  }
  recency_bin_string +=
      ("[" +
       std::to_string(this->neighborhood_search->bin_boundaries.at(
           this->neighborhood_search->bin_boundaries.size() - 1)) +
       ",infinity)");
  this->logger.WriteToLogFile("Here are the bins: " + recency_bin_string,
                              Log::info);
  return true;
}

bool ABM::ValidateArguments() {
  if (!this->ValidateArgument("Environment", "edgelist", this->edgelist, "")) {
    return false;
  }
  if (!this->ValidateArgument("Environment", "nodelist", this->nodelist, "")) {
    return false;
  }
  if (!this->ValidateArgument("Environment", "growth_rate", this->growth_rate,
                              -42)) {
    return false;
  }
  if (!this->ValidateArgument("Environment", "num_cycles", this->num_cycles,
                              -42)) {
    return false;
  }
  if (!this->ValidateArgument("Environment", "out_degree_bag",
                              this->out_degree_bag, "")) {
    return false;
  }
  if (!this->ValidateArgument("Environment", "recency_table",
                              this->recency_table, "")) {
    return false;
  }
  if (!this->community_assignment.empty()) {
    this->logger.WriteToLogFile(
        "community_assignment: " + this->community_assignment +
            " was read successfully from the config file",
        Log::info);
  }
  if (this->planted_nodes == "") {
    this->logger.WriteToLogFile("No agents will be planted", Log::info);
  } else {
    this->logger.WriteToLogFile("planted_nodes: " + this->planted_nodes,
                                Log::info);
  }
  if (this->clonal_cartel_agent_file == "") {
    this->logger.WriteToLogFile("No clonal cartel agents will be created",
                                Log::info);
  } else {
    this->logger.WriteToLogFile(
        "clonal agents: " + this->clonal_cartel_agent_file, Log::info);
  }
  if (!this->ValidateArgument("Agent", "fully_random_citations",
                              this->fully_random_citations, -42)) {
    return false;
  }
  if (this->preferential_weight == -42) {
    this->logger.WriteToLogFile(
        "Required parameter 'preferential_weight' was not "
        "found in the 'Agent' section",
        Log::error);
    return false;
  } else if (this->preferential_weight == -1) {
    this->logger.WriteToLogFile("preferential_weight: randomized", Log::info);
  } else {
    this->logger.WriteToLogFile("preferential_weight: " +
                                    std::to_string(this->preferential_weight),
                                Log::info);
  }
  if (this->fitness_weight == -42) {
    this->logger.WriteToLogFile(
        "Required parameter 'fitness_weight' was not found in "
        "the 'Agent' section",
        Log::error);
    return false;
  } else if (this->fitness_weight == -1) {
    this->logger.WriteToLogFile("fitness_weight: randomized", Log::info);
  } else {
    this->logger.WriteToLogFile(
        "fitness_weight: " + std::to_string(this->fitness_weight), Log::info);
  }
  if (this->num_authors_weight == -42) {
    this->logger.WriteToLogFile(
        "Required parameter 'num_authors_weight' was not "
        "found in the 'Agent' section",
        Log::error);
    return false;
  } else if (this->num_authors_weight == -1) {
    this->logger.WriteToLogFile("num_authors_weight: randomized", Log::info);
  } else {
    this->logger.WriteToLogFile("num_authors_weight: " +
                                    std::to_string(this->num_authors_weight),
                                Log::info);
  }
  if (this->author_reputation_weight == -42) {
    this->logger.WriteToLogFile(
        "Required parameter 'author_reputation_weight' was "
        "not found in the 'Agent' section",
        Log::error);
    return false;
  } else if (this->author_reputation_weight == -1) {
    this->logger.WriteToLogFile("author_reputation_weight: randomized",
                                Log::info);
  } else {
    this->logger.WriteToLogFile(
        "author_reputation_weight: " +
            std::to_string(this->author_reputation_weight),
        Log::info);
  }
  if (!this->ValidateArgument("Agent", "fitness_value_min",
                              this->fitness_value_min, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_value_max",
                              this->fitness_value_max, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_lag_duration_min",
                              this->fitness_lag_duration_min, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_lag_duration_max",
                              this->fitness_lag_duration_max, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_peak_duration_min",
                              this->fitness_peak_duration_min, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_peak_duration_max",
                              this->fitness_peak_duration_max, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "same_year_citations",
                              this->same_year_citations, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "neighborhood_sample",
                              this->neighborhood_sample, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "num_authors_bag", this->num_authors_bag,
                              "")) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "author_max_lifetime",
                              this->author_max_lifetime, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "cartel_outdegree_proportion",
                              this->cartel_outdegree_proportion, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "in_degree_threshold",
                              this->in_degree_threshold, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "fitness_threshold",
                              this->fitness_threshold, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "recency_threshold",
                              this->recency_threshold, -42)) {
    return false;
  }
  if (!this->ValidateArgument("Agent", "non_random_generator_probability",
                              this->non_random_generator_probability, -42)) {
    return false;
  }
  if (this->null_cartel) {
    this->logger.WriteToLogFile("Using a null cartel model", Log::info);
  } else {
    this->logger.WriteToLogFile("Not using a null cartel model", Log::info);
  }
  if (this->use_alpha) {
    if (this->alpha == -42) {
      this->logger.WriteToLogFile(
          "Required parameter 'alpha' was not found in the "
          "'Agent' section while 'use_alpha' was true",
          Log::error);
      return false;
    } else if (this->alpha == -1) {
      this->logger.WriteToLogFile("alpha: randomized", Log::info);
    } else {
      this->logger.WriteToLogFile("alpha: " + std::to_string(this->alpha),
                                  Log::info);
    }
  } else {
    if (this->alpha == -42) {
      this->logger.WriteToLogFile(
          "Alpha ignored. Agents will not split the "
          "neighborhood into 1 and 2 distance neighborhoods",
          Log::info);
    } else {
      this->logger.WriteToLogFile(
          "'use_alpha' is false but a value of" + std::to_string(this->alpha) +
              " for 'alpha' was provided. Leave the alpha "
              "line as 'alpha=' with an empty string as the "
              "alpha value if alpha should be ignored.",
          Log::error);
      return false;
    }
  }
  if (this->start_from_checkpoint) {
    this->logger.WriteToLogFile(
        "Starting from a checkpoint. Make sure the edgelist and nodelist "
        "provided are the result of a previous simulation",
        Log::info);
  } else {
    this->logger.WriteToLogFile(
        "Not using checkpointing. Starting simulation from the first year.",
        Log::info);
  }
  if (!this->community_assignment.empty()) {
    if (!this->ValidateArgument("Agent", "theta", this->theta, -42)) {
      return false;
    }
    if (this->theta <= 0) {
      this->logger.WriteToLogFile(
          "theta must be greater than zero when community_assignment is used",
          Log::error);
      return false;
    }
  } else {
    if (this->theta != -42) {
      this->logger.WriteToLogFile(
          "theta is provided but community_assignment is not provided. Theta "
          "will be ignored.",
          Log::info);
    }
  }
  if (!this->ValidateArgument("General", "output_file", this->output_file,
                              "")) {
    return false;
  }
  if (!this->ValidateArgument("General", "recency_bins", this->recency_bins,
                              "")) {
    return false;
  }
  if (!this->ValidateArgument("General", "auxiliary_information_file",
                              this->auxiliary_information_file, "")) {
    return false;
  }
  if (!this->ValidateArgument("General", "log_file", this->log_file, "")) {
    return false;
  }
  if (!this->ValidateArgument("General", "num_processors", this->num_processors,
                              -42)) {
    return false;
  }
  if (!this->ValidateArgument("General", "log_level", this->log_level, -42)) {
    return false;
  }
  return true;
}

void ABM::InitializeSimulation() {
  this->graph =
      new Graph(this->edgelist, this->nodelist, this->start_from_checkpoint,
                this->num_authors_bag, this->author_max_lifetime);
  if (!this->start_from_checkpoint) {
    this->InitializeSeedFitness(this->graph);
  }
  this->ReadCommunityAssignment();
  this->logger.WriteToLogFile("loaded this->graph", Log::info);
  /* node ids to continous integer from 0 */
  this->continuous_node_mapping = this->BuildContinuousNodeMapping(this->graph);

  /* continous integer from 0 to node ids*/
  this->reverse_continuous_node_mapping =
      this->ReverseMapping(this->continuous_node_mapping);

  this->start_year = this->GetMaxYear(this->graph) + 1;
  this->next_node_id = this->GetMaxNode(this->graph) + 1;
  this->initial_next_node_id = this->next_node_id;

  /* get input to score arrays based on this->continuous_node_mapping */
  this->initial_graph_size = this->graph->GetNodeSet().size();
  this->final_graph_size = this->GetFinalGraphSize(this->graph);
  this->reverse_continuous_node_mapping.resize(this->final_graph_size);
  this->logger.WriteToLogFile("final this->graph size is " +
                                  std::to_string(this->final_graph_size),
                              Log::info);
  this->in_degree_vec.resize(this->final_graph_size);
  this->fitness_vec.resize(this->final_graph_size);
  this->num_authors_vec.resize(this->final_graph_size);
  this->author_reputation_vec.resize(this->final_graph_size);
  this->pa_vec.resize(this->final_graph_size);
  this->fit_vec.resize(this->final_graph_size);
  this->na_vec.resize(this->final_graph_size);
  this->ar_vec.resize(this->final_graph_size);
  this->node_score_components_vec.resize(this->final_graph_size);
  this->random_weight_vec.resize(this->final_graph_size);
  this->current_score_vec.resize(this->final_graph_size);
  // this->initial_graph_size in the continuous mapping
  int added_size = this->final_graph_size - this->initial_graph_size;
  this->pa_weight_vec.assign(added_size, -1);
  this->fit_weight_vec.assign(added_size, -1);
  this->num_authors_weight_vec.assign(added_size, -1);
  this->author_reputation_weight_vec.assign(added_size, -1);
  this->out_degree_vec.assign(added_size, -1);
  this->alpha_vec.assign(added_size, -1);
  this->fitness_lag_duration_vec.assign(added_size, -1);
  this->fitness_peak_value_vec.assign(added_size, -1);
  this->fitness_peak_duration_vec.assign(added_size, -1);
  this->planted_author_id_vec.assign(added_size, -1);
  this->planted_cluster_id_vec.assign(added_size, -1);
  MetricsEngine::PopulateWeightSpans(
      this->pa_weight_vec, this->fit_weight_vec, this->num_authors_weight_vec,
      this->author_reputation_weight_vec, this->preferential_weight,
      this->fitness_weight, this->num_authors_weight,
      this->author_reputation_weight);
  MetricsEngine::PopulateAlphaSpan(this->alpha_vec, this->use_alpha,
                                   this->alpha, this->minimum_alpha);
  MetricsEngine::PopulateOutDegreeSpan(this->out_degree_vec,
                                       this->out_degree_bag_vec);
  MetricsEngine::PopulateFitnessSpans(
      this->fitness_lag_duration_vec, this->fitness_peak_value_vec,
      this->fitness_peak_duration_vec, this->fitness_lag_duration_min,
      this->fitness_lag_duration_max, this->fitness_value_min,
      this->fitness_value_max, this->fitness_peak_duration_min,
      this->fitness_peak_duration_max, this->fitness_alpha);
  MetricsEngine::PopulateNumAuthorsSpan(this->graph, this->num_authors_vec);
  this->planted_nodes_line_number_map = this->PlantNodes(
      this->graph, this->pa_weight_vec, this->fit_weight_vec,
      this->num_authors_weight_vec, this->author_reputation_weight_vec,
      this->out_degree_vec, this->alpha_vec, this->fitness_lag_duration_vec,
      this->fitness_peak_value_vec, this->fitness_peak_duration_vec,
      this->num_authors_vec, this->planted_author_id_vec,
      this->planted_cluster_id_vec);

  if (this->storage_backend == "ladybug") {
    this->logger.WriteToLogFile("Initializing LadyBugDB storage engine at " + this->db_path, Log::info);
    this->db.Open(this->db_path, ladybug::MmapMode::ReadWrite, this->final_graph_size,
                  static_cast<size_t>(this->final_graph_size) * 50, this->author_max_lifetime);
    this->db.CSR().SetThreadCount(this->num_processors);

    if (this->start_from_checkpoint) {
      bool ok = this->db.Recover();
      if (!ok) {
        this->logger.WriteToLogFile("LadyBugDB checkpoint recovery not found on disk, ingesting seed files as checkpoint", Log::info);
        this->db.IngestSeedData(this->nodelist, this->edgelist, this->community_assignment, this->num_authors_bag, true);
      } else {
        this->logger.WriteToLogFile("LadyBugDB successfully recovered from checkpoint at year " + std::to_string(this->db.GetManifest().current_year), Log::info);
      }
    } else {
      this->db.IngestFromGraph(*this->graph);
    }

    this->db.Nodes().EnsureNodeCapacity(this->final_graph_size);
    this->graph->AttachLadyBugDB(&this->db);
    this->graph->ReleaseHeavyStructures();
    this->logger.WriteToLogFile("Attached LadyBugDB to Graph and retired duplicate in-memory graph structures", Log::info);

    // Setup Distributed Partitioner & Ghost Cells if running under MPI
    if (this->coordinator.IsMPIEnabled() || this->use_distributed) {
      this->partitioner.PartitionCommunities(this->coordinator.GetWorldSize(),
                                             this->coordinator.GetRank(),
                                             this->db.Clusters(),
                                             this->db.Nodes());
      std::vector<int32_t> hub_node_ids;
      for (size_t u = 0; u < this->db.Nodes().GetNodeCount(); ++u) {
        if (this->db.CSR().GetInDegree(u) >= static_cast<size_t>(this->ghost_hub_threshold)) {
          hub_node_ids.push_back(static_cast<int32_t>(u));
        }
      }
      this->ghost_mgr.RegisterHubs(hub_node_ids, this->db.Nodes(), this->db.CSR());
      this->logger.WriteToLogFile(
          "Distributed partitioning initialized: Rank " +
              std::to_string(this->coordinator.GetRank()) + "/" +
              std::to_string(this->coordinator.GetWorldSize()) +
              " with " + std::to_string(this->ghost_mgr.GetGhostCount()) + " ghost cells",
          Log::info);
    }

    // Register Signal Supervisor
    ladybug::SignalSupervisor::Register();
    if (this->wall_clock_limit > 0.0) {
      this->signal_supervisor.SetWallClockLimit(this->wall_clock_limit);
      this->logger.WriteToLogFile(
          "Signal supervisor active with wall clock limit: " +
              std::to_string(this->wall_clock_limit) + " seconds",
          Log::info);
    }
  }
}

void ABM::RunSimulationLoop() {
  std::vector<int> new_nodes_vec;
  std::vector<std::pair<int, int>> new_edges_vec;
  std::set<int> same_year_source_nodes;
  std::unordered_map<int, double> exp_cached_results;
  for (int i = 0; i < 1000; i++) {
    exp_cached_results[i] = std::max(pow(i, this->gamma), 1.0) + 1;
  }
  std::vector<double> binned_recency_probabilities =
      GetBinnedRecencyProbabilities();
  Eigen::setNbThreads(1);
  pcg32 &generator = Utils::GetThreadLocalPRNG();
  std::uniform_real_distribution<double>
      non_random_generator_uniform_distribution{0, 1};
  for (int current_year = this->start_year;
       current_year < this->start_year + this->num_cycles; current_year++) {
    this->logger.prev_time = std::chrono::steady_clock::now();
    int current_graph_size = static_cast<int>(this->graph->GetNodeCount());
    this->logger.WriteToLogFile(
        "current year is: " + std::to_string(current_year) +
            " and the graph is " + std::to_string(current_graph_size) +
            " nodes large",
        Log::info);
    MetricsEngine::FillInDegreeSpan(this->graph,
                                    this->reverse_continuous_node_mapping,
                                    this->in_degree_vec, current_graph_size);
    this->logger.LogTime(current_year, "Fill in-degree array");
    MetricsEngine::FillFitnessSpan(
        this->graph, this->reverse_continuous_node_mapping, current_year,
        this->fitness_vec, this->fitness_decay_alpha, current_graph_size);
    this->logger.LogTime(current_year, "Fill fitness array");
    MetricsEngine::CalculateExpScores(
        exp_cached_results,
        std::span<int>{this->in_degree_vec.data(), (size_t)current_graph_size},
        std::span<double>{this->pa_vec.data(), (size_t)current_graph_size},
        this->gamma);
    this->logger.LogTime(current_year, "Process in-degree array");
    MetricsEngine::CalculateExpScores(
        exp_cached_results,
        std::span<int>{this->fitness_vec.data(), (size_t)current_graph_size},
        std::span<double>{this->fit_vec.data(), (size_t)current_graph_size},
        this->gamma);
    this->logger.LogTime(current_year, "Process fitness array");
    MetricsEngine::CalculateExpScores(
        exp_cached_results,
        std::span<int>{this->num_authors_vec.data(),
                       (size_t)current_graph_size},
        std::span<double>{this->na_vec.data(), (size_t)current_graph_size},
        this->gamma);
    this->logger.LogTime(current_year, "Process num authors array");

    /* initialize new nodes */
    int num_new_nodes = std::ceil(current_graph_size * this->growth_rate);
    this->logger.WriteToLogFile("making " + std::to_string(num_new_nodes) +
                                    " nodes this year",
                                Log::info);
    if (this->storage_backend == "ladybug") {
      this->db.Nodes().EnsureNodeCapacity(this->next_node_id + num_new_nodes);
      this->db.Nodes().SetNodeCount(this->next_node_id + num_new_nodes);
    }
    for (int i = 0; i < num_new_nodes; i++) {
      if ((size_t)this->next_node_id >= this->continuous_node_mapping.size()) {
        this->continuous_node_mapping.resize(this->next_node_id + 10000, -1);
      }
      this->continuous_node_mapping[this->next_node_id] =
          current_graph_size + i;
      if ((size_t)(current_graph_size + i) >= this->reverse_continuous_node_mapping.size()) {
        this->reverse_continuous_node_mapping.resize(current_graph_size + i + 10000, -1);
      }
      this->reverse_continuous_node_mapping[current_graph_size + i] =
          this->next_node_id;
      new_nodes_vec.push_back(this->next_node_id);
      this->graph->SetYear(this->next_node_id, current_year);
      this->graph->SetType(this->next_node_id, Graph::NodeType::Agent);
      this->next_node_id++;
    }
    this->logger.LogTime(current_year, "Create new node ids");
    if (!this->coordinator.IsMPIEnabled() || this->coordinator.IsMaster()) {
      this->FillSameYearSourceNodes(same_year_source_nodes, new_nodes_vec.size());
    }
    this->logger.LogTime(current_year, "Pick same year nodes");

    if (!this->coordinator.IsMPIEnabled() || this->coordinator.IsMaster()) {
      std::vector<int> eligible_generator_nodes = this->GetEligibleGeneratorNodes(
          this->graph, current_graph_size, this->reverse_continuous_node_mapping,
          this->in_degree_vec, this->fitness_vec, this->in_degree_threshold,
          this->fitness_threshold, this->start_year, current_year,
          this->recency_threshold);
      std::set<int> cartel_author_ids;
      for (auto const &cartel_id : this->graph->GetCartelSet()) {
        std::set<int> current_cartel_authors =
            this->graph->GetCartelAuthors(cartel_id);
        cartel_author_ids.insert(current_cartel_authors.begin(),
                                 current_cartel_authors.end());
      }
      std::set<int> initial_cartel_author_ids = cartel_author_ids;
      for (size_t i = 0; i < new_nodes_vec.size(); i++) {
        int author_id = -1;
        int new_node = new_nodes_vec[i];
        int weight_span_index =
            this->continuous_node_mapping[new_node] - this->initial_graph_size;
        int graph_span_index = this->continuous_node_mapping[new_node];
        if (this->planted_nodes_line_number_map.contains(weight_span_index) &&
            this->planted_author_id_vec[weight_span_index] != -1) {
          author_id = this->planted_author_id_vec[weight_span_index];
          this->graph->UpdateAuthorManual(author_id);
          this->UpdateGraphAttributesAuthors(this->graph, new_node, author_id);
        } else if (!this->planted_nodes_line_number_map.contains(
                       weight_span_index) &&
                   !cartel_author_ids.empty()) {
          author_id = *(cartel_author_ids.begin());
          this->graph->UpdateAuthorManual(author_id);
          cartel_author_ids.erase(cartel_author_ids.begin());
          this->UpdateGraphAttributesAuthors(this->graph, new_node, author_id);
          if (this->clonal_cartel_agent.num_authors) {
            this->num_authors_vec[graph_span_index] =
                this->clonal_cartel_agent.num_authors.value();
          }
          if (this->clonal_cartel_agent.pa_weight) {
            this->pa_weight_vec[weight_span_index] =
                this->clonal_cartel_agent.pa_weight.value();
          }
          if (this->clonal_cartel_agent.fit_weight) {
            this->fit_weight_vec[weight_span_index] =
                this->clonal_cartel_agent.fit_weight.value();
          }
          if (this->clonal_cartel_agent.num_authors_weight) {
            this->num_authors_weight_vec[weight_span_index] =
                this->clonal_cartel_agent.num_authors_weight.value();
          }
          if (this->clonal_cartel_agent.author_reputation_weight) {
            this->author_reputation_weight_vec[weight_span_index] =
                this->clonal_cartel_agent.author_reputation_weight.value();
          }
          if (this->clonal_cartel_agent.out_degree) {
            this->out_degree_vec[weight_span_index] =
                this->clonal_cartel_agent.out_degree.value();
          }
          if (this->clonal_cartel_agent.alpha) {
            this->alpha_vec[weight_span_index] =
                this->clonal_cartel_agent.alpha.value();
          }
          if (this->clonal_cartel_agent.fitness_lag_duration) {
            this->fitness_lag_duration_vec[weight_span_index] =
                this->clonal_cartel_agent.fitness_lag_duration.value();
          }
          if (this->clonal_cartel_agent.fitness_peak_value) {
            this->fitness_peak_value_vec[weight_span_index] =
                this->clonal_cartel_agent.fitness_peak_value.value();
          }
          if (this->clonal_cartel_agent.fitness_peak_duration) {
            this->fitness_peak_duration_vec[weight_span_index] =
                this->clonal_cartel_agent.fitness_peak_duration.value();
          }
        } else {
          author_id =
              this->graph->GetNextAuthor(current_year, initial_cartel_author_ids);
          this->UpdateGraphAttributesAuthors(this->graph, new_node, author_id);
        }

        int explicit_cluster_id = -1;
        if (this->planted_nodes_line_number_map.contains(weight_span_index) &&
            this->planted_cluster_id_vec[weight_span_index] != -1) {
          explicit_cluster_id = this->planted_cluster_id_vec[weight_span_index];
        }

        double new_node_non_random_draw =
            non_random_generator_uniform_distribution(generator);
        // if author part of cartel, pick randomly from cartel list
        if (this->graph->GetCartelID(author_id) > 0) {
          std::vector<int> cartel_generator_nodes =
              this->GetCartelGeneratorNodes(this->graph, author_id);
          std::vector<int> generator_nodes =
              this->GetGeneratorNodesFromSet(cartel_generator_nodes);
          this->UpdateGraphAttributesGeneratorNodes(
              this->graph, new_node, generator_nodes, explicit_cluster_id);
        } else if (new_node_non_random_draw <
                   this->non_random_generator_probability) {
          std::vector<int> generator_nodes =
              this->GetGeneratorNodesFromSet(eligible_generator_nodes);
          this->UpdateGraphAttributesGeneratorNodes(
              this->graph, new_node, generator_nodes, explicit_cluster_id);
        } else {
          std::vector<int> generator_nodes = this->GetGeneratorNodes(
              this->graph, this->reverse_continuous_node_mapping);
          this->UpdateGraphAttributesGeneratorNodes(
              this->graph, new_node, generator_nodes, explicit_cluster_id);
        }
      }
    }
    this->logger.LogTime(current_year, "Pick generator nodes");

    // Broadcast new node assignments across MPI ranks so all compute nodes share identical cluster mappings and weights
    if (this->coordinator.IsMPIEnabled()) {
      std::vector<int> cluster_ids(new_nodes_vec.size());
      std::vector<int> author_ids(new_nodes_vec.size());
      std::vector<int> primary_gen_nodes(new_nodes_vec.size());
      distributed::AgentBatchAttributes attrs;

      if (this->coordinator.IsMaster()) {
        attrs.pa_weights.resize(new_nodes_vec.size());
        attrs.fit_weights.resize(new_nodes_vec.size());
        attrs.num_authors_weights.resize(new_nodes_vec.size());
        attrs.author_reputation_weights.resize(new_nodes_vec.size());
        attrs.alphas.resize(new_nodes_vec.size());
        attrs.out_degrees.resize(new_nodes_vec.size());
        attrs.fitness_lag_durations.resize(new_nodes_vec.size());
        attrs.fitness_peak_values.resize(new_nodes_vec.size());
        attrs.fitness_peak_durations.resize(new_nodes_vec.size());
        attrs.num_authors.resize(new_nodes_vec.size());
        attrs.is_same_year_source.resize(new_nodes_vec.size());

        for (size_t i = 0; i < new_nodes_vec.size(); ++i) {
          int n = new_nodes_vec[i];
          cluster_ids[i] = this->graph->GetCommunityAssignment(n);
          author_ids[i] = this->graph->GetAuthorId(n);
          primary_gen_nodes[i] = this->graph->GetGeneratorNode(n);

          int continuous_id = ((size_t)n < this->continuous_node_mapping.size())
                                  ? this->continuous_node_mapping[n]
                                  : n;
          int weight_span_index = continuous_id - this->initial_graph_size;

          attrs.pa_weights[i] = this->pa_weight_vec[weight_span_index];
          attrs.fit_weights[i] = this->fit_weight_vec[weight_span_index];
          attrs.num_authors_weights[i] = this->num_authors_weight_vec[weight_span_index];
          attrs.author_reputation_weights[i] = this->author_reputation_weight_vec[weight_span_index];
          attrs.alphas[i] = this->alpha_vec[weight_span_index];
          attrs.out_degrees[i] = this->out_degree_vec[weight_span_index];
          attrs.fitness_lag_durations[i] = this->fitness_lag_duration_vec[weight_span_index];
          attrs.fitness_peak_values[i] = this->fitness_peak_value_vec[weight_span_index];
          attrs.fitness_peak_durations[i] = this->fitness_peak_duration_vec[weight_span_index];
          attrs.num_authors[i] = this->num_authors_vec[continuous_id];
          attrs.is_same_year_source[i] = same_year_source_nodes.count(i) ? 1 : 0;
        }
      }

      this->coordinator.BroadcastNewNodeMetadata(
          new_nodes_vec, cluster_ids, author_ids, primary_gen_nodes, attrs, 0);

      if (!this->coordinator.IsMaster()) {
        same_year_source_nodes.clear();
        for (size_t i = 0; i < new_nodes_vec.size(); ++i) {
          int n = new_nodes_vec[i];
          int continuous_id = ((size_t)n < this->continuous_node_mapping.size())
                                  ? this->continuous_node_mapping[n]
                                  : n;
          int weight_span_index = continuous_id - this->initial_graph_size;

          this->pa_weight_vec[weight_span_index] = attrs.pa_weights[i];
          this->fit_weight_vec[weight_span_index] = attrs.fit_weights[i];
          this->num_authors_weight_vec[weight_span_index] = attrs.num_authors_weights[i];
          this->author_reputation_weight_vec[weight_span_index] = attrs.author_reputation_weights[i];
          this->alpha_vec[weight_span_index] = attrs.alphas[i];
          this->out_degree_vec[weight_span_index] = attrs.out_degrees[i];
          this->fitness_lag_duration_vec[weight_span_index] = attrs.fitness_lag_durations[i];
          this->fitness_peak_value_vec[weight_span_index] = attrs.fitness_peak_values[i];
          this->fitness_peak_duration_vec[weight_span_index] = attrs.fitness_peak_durations[i];
          this->num_authors_vec[continuous_id] = attrs.num_authors[i];

          if (attrs.is_same_year_source[i]) {
            same_year_source_nodes.insert(i);
          }

          this->graph->SetAuthorId(n, author_ids[i]);
          this->graph->UpdateAuthorPublicationMap(author_ids[i], n);
          this->graph->SetAuthorBirthYear(author_ids[i], current_year);
          this->graph->SetCommunityAssignment(n, cluster_ids[i]);
          if (cluster_ids[i] >= 0) {
            this->graph->AddNodeToCluster(n, cluster_ids[i]);
          }
          if (primary_gen_nodes[i] >= 0) {
            this->graph->SetGeneratorNode(n, primary_gen_nodes[i]);
          }
        }
      }
    }

    // Tally clusters and count of new agents added by this compute node
    std::vector<size_t> local_agent_indices;
    local_agent_indices.reserve(new_nodes_vec.size() / (this->coordinator.IsMPIEnabled() ? this->coordinator.GetWorldSize() : 1));
    std::map<int, int> local_cluster_new_agents;
    int local_new_agents_count = 0;
    for (size_t i = 0; i < new_nodes_vec.size(); i++) {
      int new_node = new_nodes_vec[i];
      int cluster_id = this->graph->GetCommunityAssignment(new_node);
      if (!this->coordinator.IsMPIEnabled() || this->partitioner.IsLocalClusterOrNode(cluster_id, new_node)) {
        local_agent_indices.push_back(i);
        local_cluster_new_agents[cluster_id]++;
        local_new_agents_count++;
      }
    }

    std::string rank_prefix = "Rank " + std::to_string(this->coordinator.GetRank());
    std::string timing_summary = rank_prefix + " added " + std::to_string(local_new_agents_count) +
                                 " agents across " + std::to_string(local_cluster_new_agents.size()) +
                                 " clusters";
    this->logger.LogTime(current_year, timing_summary, local_new_agents_count);

    // Record per-cluster agent additions in timing log file
    for (const auto &[cid, count] : local_cluster_new_agents) {
      std::string cluster_label = "Cluster " + std::to_string(cid) + " (" + rank_prefix + ") new agents";
      this->logger.LogTime(current_year, cluster_label, count);
    }

    std::stringstream cluster_ss;
    cluster_ss << "[" << rank_prefix << "] Added " << local_new_agents_count << " new agents in "
               << local_cluster_new_agents.size() << " local clusters: [";
    int c_shown = 0;
    for (const auto &[cid, count] : local_cluster_new_agents) {
      if (c_shown++ < 15) {
        cluster_ss << "C" << cid << ":" << count << " ";
      }
    }
    if (local_cluster_new_agents.size() > 15) {
      cluster_ss << "... (" << local_cluster_new_agents.size() << " total)]";
    } else {
      cluster_ss << "]";
    }
    this->logger.WriteToLogFile(cluster_ss.str(), Log::info);

    MetricsEngine::FillAuthorReputationSpan(
        this->graph, this->reverse_continuous_node_mapping,
        this->author_reputation_vec, current_graph_size);
    this->logger.LogTime(current_year, "Fill author reputation array");
    MetricsEngine::CalculateExpScores(
        exp_cached_results,
        std::span<int>{this->author_reputation_vec.data(),
                       (size_t)current_graph_size},
        std::span<double>{this->ar_vec.data(), (size_t)current_graph_size},
        this->gamma);
    this->logger.LogTime(current_year, "Process author reputation array");

    std::vector<int> sampled_neighborhood_sizes_map(new_nodes_vec.size(), -1);
    std::vector<int> fully_random_citations_map(new_nodes_vec.size(), -1);
    std::vector<std::vector<int>> sampled_binned_neighborhood_sizes_map(
        new_nodes_vec.size());

#pragma omp parallel for simd
    for (int idx = 0; idx < current_graph_size; ++idx) {
      this->node_score_components_vec[idx].pa = this->pa_vec[idx];
      this->node_score_components_vec[idx].fit = this->fit_vec[idx];
      this->node_score_components_vec[idx].na = this->na_vec[idx];
      this->node_score_components_vec[idx].ar = this->ar_vec[idx];
    }

    NodeMetrics metrics = {
        std::span<const NodeScoreComponents>{this->node_score_components_vec.data(),
                                             (size_t)current_graph_size}};

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<std::pair<int, int>>> thread_local_new_edges_vec(
        max_threads);
    std::vector<std::vector<std::pair<std::string, int>>>
        thread_local_parallel_stage_time_vec(max_threads);
    std::vector<std::vector<int>> thread_citations_vec(max_threads);
    std::vector<std::vector<int>> thread_scratch_bin_vec(max_threads);
    for (int t = 0; t < max_threads; ++t) {
      thread_local_new_edges_vec[t].reserve(100);
      thread_local_parallel_stage_time_vec[t].reserve(20);
      thread_citations_vec[t].resize(this->max_out_degree + 1, 0);
    }

    // Precompute recency bins for historical cluster nodes (year < current_year)
    // to avoid redundant O(|cluster|) binning per agent.
    std::unordered_map<int, std::vector<std::vector<int>>> cluster_binned_cache;
    std::unordered_map<int, int> cluster_historical_sizes;
    if (this->use_alpha) {
      for (const auto &[cid, _] : local_cluster_new_agents) {
        if (cid >= 0) {
          const auto &cluster_nodes = this->graph->GetClusterNodes(cid);
          auto &bins = cluster_binned_cache[cid];
          bins.resize(this->neighborhood_search->num_bins);
          int hist_count = 0;
          for (int node : cluster_nodes) {
            if (this->graph->GetYear(node) < current_year) {
              int bin_idx = this->neighborhood_search->GetBinIndex(this->graph, node, current_year);
              if (bin_idx >= 0 && bin_idx < this->neighborhood_search->num_bins) {
                bins[bin_idx].push_back(node);
                hist_count++;
              }
            }
          }
          cluster_historical_sizes[cid] = hist_count;
        }
      }
    }

    std::vector<std::pair<std::string, int>> parallel_stage_time_vec;
#pragma omp parallel for schedule(dynamic, 64)                                 \
    reduction(custom_merge_vec_int : new_edges_vec)                           \
    reduction(merge_str_int_pair_vecs : parallel_stage_time_vec)
    for (size_t k = 0; k < local_agent_indices.size(); k++) {
      size_t i = local_agent_indices[k];
      int new_node = new_nodes_vec[i];
      int cluster_id = this->graph->GetCommunityAssignment(new_node);

      int thread_id = omp_get_thread_num();
      std::chrono::time_point<std::chrono::steady_clock> local_prev_time =
          std::chrono::steady_clock::now();

      auto &local_new_edges_vec = thread_local_new_edges_vec[thread_id];
      local_new_edges_vec.clear();

      auto &local_parallel_stage_time_vec =
          thread_local_parallel_stage_time_vec[thread_id];
      local_parallel_stage_time_vec.clear();

      auto &citations_vec = thread_citations_vec[thread_id];
      std::fill(citations_vec.begin(), citations_vec.end(), 0);
      std::span<int> citations(citations_vec);
      int continuous_id = ((size_t)new_node < this->continuous_node_mapping.size())
                              ? this->continuous_node_mapping[new_node]
                              : new_node;
      int weight_span_index = continuous_id - this->initial_graph_size;
      double pa_weight = this->pa_weight_vec[weight_span_index];
      double fit_weight = this->fit_weight_vec[weight_span_index];
      double num_authors_weight =
          this->num_authors_weight_vec[weight_span_index];
      double author_reputation_weight =
          this->author_reputation_weight_vec[weight_span_index];
      AgentWeights weights = {pa_weight, fit_weight, num_authors_weight,
                              author_reputation_weight};
      double alpha = this->alpha_vec[weight_span_index];
      int author_id = this->graph->GetAuthorId(new_node);
      std::vector<int> generator_nodes =
          this->GetGraphAttributesGeneratorNodes(this->graph, new_node);
      int num_hops = 2;
      // if use alpha then vector has size 3 (indices 1 and 2)
      // if use alpha false then vector has size 2 (index 1)
      std::vector<std::vector<int>> n_hop_map =
          this->neighborhood_search->GetNeighborhoodMap(
              this->graph, current_year, generator_nodes, num_hops);

      bool use_cluster_fastpath = false;
      int cluster_hist_size = 0;
      if (alpha > 0 && cluster_id >= 0 &&
          this->graph->GetClusterSize(cluster_id, current_year) >= this->theta) {
        auto hist_it = cluster_historical_sizes.find(cluster_id);
        if (hist_it != cluster_historical_sizes.end()) {
          cluster_hist_size = hist_it->second;
          use_cluster_fastpath = true;
          if (n_hop_map.size() > 2) {
            std::vector<int> pruned_2_hop;
            for (int node : n_hop_map[2]) {
              if (this->graph->GetCommunityAssignment(node) != cluster_id) {
                pruned_2_hop.push_back(node);
              }
            }
            n_hop_map[2] = std::move(pruned_2_hop);
          }
        }
      }

      if (!use_cluster_fastpath && alpha > 0 && cluster_id >= 0 &&
          this->graph->GetClusterSize(cluster_id, current_year) >= this->theta) {
        const std::vector<int> &cluster_nodes =
            this->graph->GetClusterNodes(cluster_id);
        std::vector<int> filtered_cluster_nodes;
        filtered_cluster_nodes.reserve(cluster_nodes.size());
        for (int node : cluster_nodes) {
          if (std::find(generator_nodes.begin(), generator_nodes.end(), node) ==
                  generator_nodes.end() &&
              this->graph->GetYear(node) < current_year) {
            filtered_cluster_nodes.push_back(node);
          }
        }
        if (n_hop_map.size() > 1) {
          n_hop_map[1] = std::move(filtered_cluster_nodes);
        }

        if (n_hop_map.size() > 2) {
          std::vector<int> pruned_2_hop;
          for (int node : n_hop_map[2]) {
            if (this->graph->GetCommunityAssignment(node) != cluster_id) {
              pruned_2_hop.push_back(node);
            }
          }
          n_hop_map[2] = std::move(pruned_2_hop);
        }
      }

      int num_cartel_citations_limit =
          std::round(this->cartel_outdegree_proportion *
                     this->out_degree_vec[weight_span_index]);
      int num_fully_random_cited_reserved = std::floor(
          this->fully_random_citations *
          this->out_degree_vec[weight_span_index]); // e.g., 5% of out-degree.
                                                    // some small number
      int remaining_citation_quota = this->out_degree_vec[weight_span_index];
      remaining_citation_quota -=
          generator_nodes.size(); // should be 1 and always decremented
      remaining_citation_quota -=
          same_year_source_nodes.count(i); // could be 0 or 1
      int current_cartel_size = 0;
      if (this->graph->GetCartelID(author_id) > 0) {
        current_cartel_size =
            this->graph->GetCartelAuthors(this->graph->GetCartelID(author_id))
                .size() -
            1; // -1 since we cite others only
      }
      int num_cartel_citations =
          std::min(num_cartel_citations_limit, current_cartel_size);
      remaining_citation_quota -= num_cartel_citations;

      if (remaining_citation_quota > num_fully_random_cited_reserved) {
        remaining_citation_quota -= num_fully_random_cited_reserved;
      } else {
        num_fully_random_cited_reserved = 0;
      }

      int num_actually_cited = 0;
      if (same_year_source_nodes.count(i)) {
        num_actually_cited += this->citation_engine->MakeSameYearCitations(
            same_year_source_nodes, new_nodes_vec.size(),
            this->reverse_continuous_node_mapping, citations,
            current_graph_size);
      }
      local_prev_time = this->logger.LocalLogTime(local_parallel_stage_time_vec,
                                                  local_prev_time,
                                                  "make same year citations");

      num_actually_cited += this->citation_engine->MakeCartelCitations(
          this->graph, generator_nodes, author_id,
          this->continuous_node_mapping, n_hop_map,
          citations.subspan(num_actually_cited), num_cartel_citations,
          current_year, binned_recency_probabilities, metrics, weights,
          current_graph_size);
      // remove cartel cited nodes from n_hop_map
      std::set<int> cited_elements(citations.begin(),
                                   citations.begin() + num_actually_cited);
      std::vector<std::vector<int>> filtered_n_hop_map(n_hop_map.size());
      if (use_cluster_fastpath) {
        // Distance 1 is handled via pre-binned cache; only filter distance >= 2
        for (size_t distance = 2; distance < n_hop_map.size(); ++distance) {
          filtered_n_hop_map[distance].reserve(this->neighborhood_sample);
          for (int node_id : n_hop_map[distance]) {
            if (!cited_elements.contains(node_id)) {
              filtered_n_hop_map[distance].push_back(node_id);
            }
          }
        }
      } else {
        for (size_t distance = 1; distance < n_hop_map.size(); ++distance) {
          filtered_n_hop_map[distance].reserve(this->neighborhood_sample);
          for (int node_id : n_hop_map[distance]) {
            if (!cited_elements.contains(node_id)) {
              filtered_n_hop_map[distance].push_back(node_id);
            }
          }
        }
      }
      n_hop_map = std::move(filtered_n_hop_map);
      if (remaining_citation_quota > 0) {
        std::vector<int> num_citations_per_neighborhood;
        size_t hop1_size = 0;
        size_t hop2_size = (n_hop_map.size() > 2) ? n_hop_map[2].size() : 0;
        if (use_cluster_fastpath) {
          int excluded_count_hop1 = 0;
          for (int g : generator_nodes) {
            if (this->graph->GetCommunityAssignment(g) == cluster_id && this->graph->GetYear(g) < current_year) {
              excluded_count_hop1++;
            }
          }
          for (int c : cited_elements) {
            if (std::find(generator_nodes.begin(), generator_nodes.end(), c) == generator_nodes.end() &&
                this->graph->GetCommunityAssignment(c) == cluster_id && this->graph->GetYear(c) < current_year) {
              excluded_count_hop1++;
            }
          }
          hop1_size = static_cast<size_t>(std::max(0, cluster_hist_size - excluded_count_hop1));
          num_citations_per_neighborhood =
              this->neighborhood_search->GetNumCitationsPerNeighborhood(
                  alpha, remaining_citation_quota, hop1_size, hop2_size);
        } else {
          num_citations_per_neighborhood =
              this->neighborhood_search->GetNumCitationsPerNeighborhood(
                  alpha, remaining_citation_quota, n_hop_map);
        }

        for (size_t current_neighborhood_index = 1;
             current_neighborhood_index < n_hop_map.size();
             current_neighborhood_index++) { // 2 iter if use alpha true

          if (use_cluster_fastpath && current_neighborhood_index == 1) {
            sampled_neighborhood_sizes_map[i] += hop1_size;

            std::vector<int> bin_sizes(this->neighborhood_search->num_bins, 0);
            const auto &cached_bins = cluster_binned_cache[cluster_id];
            for (int b = 0; b < this->neighborhood_search->num_bins; ++b) {
              bin_sizes[b] = static_cast<int>(cached_bins[b].size());
            }

            std::vector<int> excluded_nodes;
            for (int g : generator_nodes) {
              if (this->graph->GetCommunityAssignment(g) == cluster_id && this->graph->GetYear(g) < current_year) {
                int b = this->neighborhood_search->GetBinIndex(this->graph, g, current_year);
                if (b >= 0 && b < this->neighborhood_search->num_bins && bin_sizes[b] > 0) {
                  bin_sizes[b]--;
                  excluded_nodes.push_back(g);
                }
              }
            }
            for (int c : cited_elements) {
              if (std::find(generator_nodes.begin(), generator_nodes.end(), c) == generator_nodes.end() &&
                  this->graph->GetCommunityAssignment(c) == cluster_id && this->graph->GetYear(c) < current_year) {
                int b = this->neighborhood_search->GetBinIndex(this->graph, c, current_year);
                if (b >= 0 && b < this->neighborhood_search->num_bins && bin_sizes[b] > 0) {
                  bin_sizes[b]--;
                  excluded_nodes.push_back(c);
                }
              }
            }

            local_prev_time =
                this->logger.LocalLogTime(local_parallel_stage_time_vec,
                                          local_prev_time, "bin neighborhood");

            int neighborhood_quota = (current_neighborhood_index < num_citations_per_neighborhood.size())
                                         ? num_citations_per_neighborhood[current_neighborhood_index]
                                         : 0;
            std::vector<int> outdegree_per_bin_map =
                this->neighborhood_search->BinOutdegrees(
                    bin_sizes,
                    neighborhood_quota,
                    binned_recency_probabilities);

            for (int bin_index = 0;
                 bin_index < this->neighborhood_search->num_bins - 1;
                 bin_index++) {
              int quota = outdegree_per_bin_map[bin_index];
              if (quota <= 0) continue;

              const std::vector<int> *cand_ptr = &cached_bins[bin_index];
              auto &scratch_bin = thread_scratch_bin_vec[thread_id];
              if (bin_sizes[bin_index] < (int)cached_bins[bin_index].size()) {
                scratch_bin.clear();
                scratch_bin.reserve(bin_sizes[bin_index]);
                for (int node : cached_bins[bin_index]) {
                  bool exclude = false;
                  for (int ex : excluded_nodes) {
                    if (node == ex) {
                      exclude = true;
                      break;
                    }
                  }
                  if (!exclude) {
                    scratch_bin.push_back(node);
                  }
                }
                cand_ptr = &scratch_bin;
              }

              num_actually_cited += this->citation_engine->MakeCitations(
                  this->graph, this->continuous_node_mapping, current_year,
                  *cand_ptr,
                  citations.subspan(num_actually_cited), metrics, weights,
                  current_graph_size, quota);
            }

            int last_bin = this->neighborhood_search->num_bins - 1;
            int last_quota = outdegree_per_bin_map[last_bin];
            if (last_quota > 0) {
              const std::vector<int> *cand_ptr = &cached_bins[last_bin];
              auto &scratch_bin = thread_scratch_bin_vec[thread_id];
              if (bin_sizes[last_bin] < (int)cached_bins[last_bin].size()) {
                scratch_bin.clear();
                scratch_bin.reserve(bin_sizes[last_bin]);
                for (int node : cached_bins[last_bin]) {
                  bool exclude = false;
                  for (int ex : excluded_nodes) {
                    if (node == ex) {
                      exclude = true;
                      break;
                    }
                  }
                  if (!exclude) {
                    scratch_bin.push_back(node);
                  }
                }
                cand_ptr = &scratch_bin;
              }

              num_actually_cited +=
                  this->citation_engine->MakeUniformRandomCitations(
                      this->graph, this->continuous_node_mapping, current_year,
                      *cand_ptr,
                      citations.subspan(num_actually_cited), current_graph_size,
                      last_quota);
            }

          } else {
            sampled_neighborhood_sizes_map[i] +=
                n_hop_map[current_neighborhood_index].size();

            std::vector<std::vector<int>> binned_neighborhood =
                this->neighborhood_search->BinNeighborhood(
                    this->graph, current_year,
                    n_hop_map[current_neighborhood_index]);

            local_prev_time =
                this->logger.LocalLogTime(local_parallel_stage_time_vec,
                                          local_prev_time, "bin neighborhood");

            int neighborhood_quota = (current_neighborhood_index < num_citations_per_neighborhood.size())
                                         ? num_citations_per_neighborhood[current_neighborhood_index]
                                         : 0;
            std::vector<int> outdegree_per_bin_map =
                this->neighborhood_search->BinOutdegrees(
                    binned_neighborhood,
                    neighborhood_quota,
                    binned_recency_probabilities);

            for (int bin_index = 0;
                 bin_index < this->neighborhood_search->num_bins - 1;
                 bin_index++) { // if there's only 1 bin then this is always false
              num_actually_cited += this->citation_engine->MakeCitations(
                  this->graph, this->continuous_node_mapping, current_year,
                  binned_neighborhood[bin_index],
                  citations.subspan(num_actually_cited), metrics, weights,
                  current_graph_size, outdegree_per_bin_map[bin_index]);
            }
            num_actually_cited +=
                this->citation_engine->MakeUniformRandomCitations(
                    this->graph, this->continuous_node_mapping, current_year,
                    binned_neighborhood[this->neighborhood_search->num_bins - 1],
                    citations.subspan(num_actually_cited), current_graph_size,
                    outdegree_per_bin_map[this->neighborhood_search->num_bins -
                                          1]);
          }
        }
      }
      for (int j = 0; j < num_actually_cited; j++) {
        if (citations[j] < 0) {
          std::cerr
              << "[checking before uniform random to this->graph] regular "
                 "citation negative: "
              << std::to_string(citations[j]) << std::endl;
        }
      }

      local_prev_time = this->logger.LocalLogTime(
          local_parallel_stage_time_vec, local_prev_time,
          "make neighborhood citations");

      int num_fully_random_cited = this->out_degree_vec[weight_span_index] -
                                   num_actually_cited -
                                   generator_nodes.size(); // finalized later
      fully_random_citations_map[i] = num_fully_random_cited;
      num_actually_cited +=
          this->citation_engine->MakeUniformRandomCitationsFromGraph(
              this->graph, this->reverse_continuous_node_mapping,
              generator_nodes, citations, num_actually_cited,
              num_fully_random_cited);
      local_prev_time =
          this->logger.LocalLogTime(local_parallel_stage_time_vec,
                                    local_prev_time, "make random citations");

      std::set<int> unique_cited_nodes;
      for (size_t j = 0; j < generator_nodes.size(); j++) {
        if (generator_nodes[j] < 0) {
          std::cerr << "generator node negative: "
                    << std::to_string(generator_nodes[j]) << std::endl;
        }
        if (unique_cited_nodes.find(generator_nodes[j]) == unique_cited_nodes.end()) {
          local_new_edges_vec.push_back({new_node, generator_nodes[j]});
          unique_cited_nodes.insert(generator_nodes[j]);
        }
      }
      for (int j = 0; j < num_actually_cited; j++) {
        if (citations[j] < 0) {
          std::cerr << "regular citation negative: "
                    << std::to_string(citations[j]) << " at index "
                    << std::to_string(j) << std::endl;
          std::cerr << "num_actually_cited: "
                    << std::to_string(num_actually_cited) << std::endl;
        }
        if (unique_cited_nodes.find(citations[j]) == unique_cited_nodes.end()) {
          local_new_edges_vec.push_back({new_node, citations[j]});
          unique_cited_nodes.insert(citations[j]);
        }
      }

      if (this->storage_backend == "ladybug") {
        for (const auto &edge : local_new_edges_vec) {
          int citing = edge.first;
          int cited = edge.second;
          if (this->coordinator.IsMPIEnabled()) {
            int target_rank = this->partitioner.GetNodeOwnerRank(cited, this->db.Nodes());
            if (target_rank == this->coordinator.GetRank()) {
              this->db.CSR().RecordDeltaIncomingCitation(thread_id, cited, citing);
            } else {
              this->exchange.QueueCitation(target_rank, citing, cited, static_cast<uint16_t>(current_year));
            }
          } else {
            this->db.CSR().RecordDeltaIncomingCitation(thread_id, cited, citing);
          }
        }
      }

      new_edges_vec.insert(new_edges_vec.end(), local_new_edges_vec.begin(),
                           local_new_edges_vec.end());
      local_prev_time = this->logger.LocalLogTime(
          local_parallel_stage_time_vec, local_prev_time, "record edges");
      parallel_stage_time_vec.insert(parallel_stage_time_vec.end(),
                                     local_parallel_stage_time_vec.begin(),
                                     local_parallel_stage_time_vec.end());
    } // end of omp parallel loop
    std::map<std::string, int> per_stage_time_map;
    for (size_t i = 0; i < parallel_stage_time_vec.size(); i++) {
      per_stage_time_map[parallel_stage_time_vec[i].first] +=
          parallel_stage_time_vec[i].second;
    }
    this->logger.LogTime(current_year, "find neighborhood",
                         per_stage_time_map["retrieve neighborhood"]);
    this->logger.LogTime(current_year, "bin neighborhood",
                         per_stage_time_map["bin neighborhood"]);
    this->logger.LogTime(current_year, "make same year citations",
                         per_stage_time_map["make same year citations"]);
    this->logger.LogTime(current_year, "make neighborhood citations",
                         per_stage_time_map["make neighborhood citations"]);
    this->logger.LogTime(current_year, "make random citations",
                         per_stage_time_map["make random citations"]);
    this->logger.LogTime(current_year, "record edges",
                         per_stage_time_map["record edges"]);
    if (this->coordinator.IsMPIEnabled()) {
      this->coordinator.SynchronizeNewEdges(new_edges_vec);
      this->coordinator.SynchronizeNodeDiagnostics(
          sampled_neighborhood_sizes_map, fully_random_citations_map);
      this->logger.LogTime(current_year, "Synchronize distributed new edges across ranks",
                           static_cast<int>(new_edges_vec.size()));
    }

    for (size_t i = 0; i < new_nodes_vec.size(); i++) {
      if (sampled_neighborhood_sizes_map[i] < 0) {
        sampled_neighborhood_sizes_map[i] = 0;
      }
      if (fully_random_citations_map[i] < 0) {
        fully_random_citations_map[i] = 0;
      }
    }

    if (this->storage_backend == "ladybug") {
      int min_new_node = new_nodes_vec.empty() ? 0 : new_nodes_vec.front();
      int num_new = static_cast<int>(new_nodes_vec.size());
      std::vector<std::vector<int32_t>> new_node_out_edges(num_new);
      for (const auto &[citing, cited] : new_edges_vec) {
        int offset = citing - min_new_node;
        if (offset >= 0 && offset < num_new) {
          new_node_out_edges[offset].push_back(cited);
        }
      }

      for (size_t k = 0; k < local_agent_indices.size(); k++) {
        size_t i = local_agent_indices[k];
        int new_node = new_nodes_vec[i];
        int cluster_id = this->graph->GetCommunityAssignment(new_node);

        int weight_span_index = this->continuous_node_mapping[new_node] - this->initial_graph_size;
        int offset = new_node - min_new_node;
        const auto &out_edges = new_node_out_edges[offset];

        this->db.CSR().AppendNodeOutEdges(new_node, std::span<const int32_t>(out_edges.data(), out_edges.size()));
        this->db.Nodes().SetOutDegree(new_node, static_cast<uint32_t>(out_edges.size()));

        this->db.Nodes().SetYear(new_node, current_year);
        this->db.Nodes().SetType(new_node, ladybug::NodeType::Agent);
        this->db.Nodes().SetAuthorId(new_node, this->graph->GetAuthorId(new_node));
        this->db.Nodes().SetClusterId(new_node, cluster_id);
        this->db.Nodes().SetAssignedOutDegree(new_node, this->out_degree_vec[weight_span_index]);
        this->db.Nodes().SetPaWeight(new_node, this->pa_weight_vec[weight_span_index]);
        this->db.Nodes().SetFitWeight(new_node, this->fit_weight_vec[weight_span_index]);
        this->db.Nodes().SetNumAuthorsWeight(new_node, this->num_authors_weight_vec[weight_span_index]);
        this->db.Nodes().SetAuthorRepWeight(new_node, this->author_reputation_weight_vec[weight_span_index]);
        this->db.Nodes().SetAlpha(new_node, this->alpha_vec[weight_span_index]);
        this->db.Nodes().SetFitnessLagDuration(new_node, this->fitness_lag_duration_vec[weight_span_index]);
        this->db.Nodes().SetFitnessPeakValue(new_node, this->fitness_peak_value_vec[weight_span_index]);
        this->db.Nodes().SetFitnessPeakDuration(new_node, this->fitness_peak_duration_vec[weight_span_index]);
        this->db.Nodes().SetSampledNeighborhoodSize(new_node, sampled_neighborhood_sizes_map[i]);
        this->db.Nodes().SetFullyRandomCitations(new_node, fully_random_citations_map[i]);
        this->db.Nodes().SetNumAuthors(new_node, this->num_authors_vec[this->continuous_node_mapping[new_node]]);
        if (this->planted_nodes_line_number_map.contains(weight_span_index)) {
          this->db.Nodes().SetPlantedLineNumber(new_node, this->planted_nodes_line_number_map[weight_span_index]);
        }
      }

      this->logger.LogTime(current_year, "Append new nodes and edges to LadyBugDB");
      size_t remote_routed = this->coordinator.SynchronizeSuperstep(current_year, this->db, this->exchange);
      this->logger.LogTime(current_year, "Synchronize LadyBug superstep & checkpoint (inter-rank citations routed: " + std::to_string(remote_routed) + ")", static_cast<int>(remote_routed));
      this->UpdateGraphAttributesInitialAuthorReputations(this->graph, new_nodes_vec);

      if (this->signal_supervisor.ShouldStop()) {
        this->logger.WriteToLogFile(
            "Signal or wall-clock threshold reached. Gracefully pausing simulation at year " +
                std::to_string(current_year),
            Log::info);
        break;
      }
    } else {
      for (size_t i = 0; i < new_edges_vec.size(); i++) {
        int new_node = new_edges_vec[i].first;
        int destination_id = new_edges_vec[i].second;
        this->graph->AddEdge({new_node, destination_id});
        if (new_node < 0 || destination_id < 0) {
          std::cerr << "making edge" << std::to_string(new_node) << " -> "
                    << std::to_string(destination_id) << std::endl;
        }
      }
      this->logger.LogTime(current_year, "Add edges to this->graph");

      for (size_t i = 0; i < new_nodes_vec.size(); i++) {
        int new_node = new_nodes_vec[i];
        this->graph->SetSampledNeighborhoodSize(
            new_node, sampled_neighborhood_sizes_map[i]);
        this->graph->SetFullyRandomCitations(new_node,
                                             fully_random_citations_map[i]);
      }

      this->logger.LogTime(current_year,
                           "Update this->graph attributes (neighborhood sizes)");
      this->UpdateGraphAttributesFitnesses(
          this->graph, new_nodes_vec, this->continuous_node_mapping,
          this->fitness_lag_duration_vec, this->fitness_peak_value_vec,
          this->fitness_peak_duration_vec, this->initial_graph_size);
      this->logger.LogTime(current_year, "Assign fitness values to new nodes");
      this->graph->ComputeAuthorReputations();
      this->UpdateGraphAttributesInitialAuthorReputations(this->graph,
                                                           new_nodes_vec);
    }

    new_nodes_vec.clear();
    new_edges_vec.clear();
    same_year_source_nodes.clear();
  }
}

void ABM::FinalizeSimulation() {
  this->logger.WriteToLogFile("finished sim", Log::info);

  if (this->storage_backend == "ladybug") {
    this->coordinator.Barrier();
    this->db.Sync(false);
    this->logger.WriteToLogFile("LadyBugDB storage engine synchronized", Log::info);
  }

  if (this->coordinator.IsMaster()) {
    if (this->storage_backend == "ladybug") {
      this->graph->WriteGraph(this->output_file);
      this->logger.WriteToLogFile("wrote edgelist from LadyBugDB", Log::info);
      this->graph->WriteAttributes(this->auxiliary_information_file);
      this->logger.WriteToLogFile("wrote nodelist from LadyBugDB", Log::info);
    } else {
      this->graph->WriteGraph(this->output_file);
      this->logger.WriteToLogFile("wrote this->graph", Log::info);

      this->UpdateGraphAttributesWeights(
          this->graph, this->initial_next_node_id, this->pa_weight_vec,
          this->fit_weight_vec, this->num_authors_weight_vec,
          this->author_reputation_weight_vec,
          this->final_graph_size - this->initial_graph_size);
      this->logger.WriteToLogFile("updated weights", Log::info);
      this->UpdateGraphAttributesOutDegrees(
          this->graph, this->initial_next_node_id, this->out_degree_vec,
          this->final_graph_size - this->initial_graph_size);
      this->logger.WriteToLogFile("updated out-degrees", Log::info);
      this->UpdateGraphAttributesAlphas(
          this->graph, this->initial_next_node_id, this->alpha_vec,
          this->final_graph_size - this->initial_graph_size);
      this->logger.WriteToLogFile("updated alphas", Log::info);
      this->UpdateGraphAttributesNumAuthors(
          this->graph, this->continuous_node_mapping, this->num_authors_vec);
      this->logger.WriteToLogFile("updated num authors", Log::info);
      this->UpdateGraphAttributesPlantedNodesLineNumbers(
          this->graph, this->initial_next_node_id,
          this->planted_nodes_line_number_map);
      this->logger.WriteToLogFile("updated planted nodes line numbers", Log::info);

      for (auto const &node_id : this->graph->GetNodeSet()) {
        this->graph->SetInDegree(node_id, this->graph->GetInDegree(node_id));
        this->graph->SetOutDegree(node_id, this->graph->GetOutDegree(node_id));
      }
      this->logger.WriteToLogFile("computed in-degree and out-degrees", Log::info);
      this->graph->ComputeAuthorReputations();
      this->logger.WriteToLogFile("computed author reputations", Log::info);
      this->graph->WriteAttributes(this->auxiliary_information_file);
      this->logger.WriteToLogFile("wrote nodelist", Log::info);
    }
  }
  delete this->graph;
}

void ABM::ReadCommunityAssignment() {
  if (this->community_assignment.empty()) {
    return;
  }

  this->logger.WriteToLogFile("Attempting to read community assignment file: " +
                                  this->community_assignment,
                              Log::info);
  std::ifstream file(this->community_assignment);
  if (!file.is_open()) {
    this->logger.WriteToLogFile(
        "Error: Could not open community assignment file " +
            this->community_assignment,
        Log::error);
    throw std::runtime_error("Could not open community assignment file");
  }

  char delimiter = Utils::GetDelimiter(this->community_assignment);
  auto header_map =
      Utils::GetHeaderToIndexMap(delimiter, this->community_assignment);

  if (!header_map.contains("node_id")) {
    this->logger.WriteToLogFile(
        "Error: Community assignment file must have 'node_id' column",
        Log::error);
    throw std::runtime_error("Invalid community assignment file schema");
  }

  int cluster_id_idx = -1;
  if (header_map.contains("cluster_id")) {
    cluster_id_idx = header_map["cluster_id"];
  } else if (header_map.contains("comm_id")) {
    cluster_id_idx = header_map["comm_id"];
  } else {
    this->logger.WriteToLogFile("Error: Community assignment file must have "
                                "'cluster_id' or 'comm_id' column",
                                Log::error);
    throw std::runtime_error("Invalid community assignment file schema");
  }

  int node_id_idx = header_map["node_id"];

  std::string line;
  int line_no = 0;

  // Need to read the file again since GetDelimiter / GetHeaderToIndexMap open
  // and close it internally
  std::ifstream data_file(this->community_assignment);
  while (std::getline(data_file, line)) {
    if (line_no == 0) {
      line_no++;
      continue;
    }
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.empty()) {
      break;
    }

    // Some lines might be malformed
    if (current_line.size() <= (size_t)std::max(node_id_idx, cluster_id_idx)) {
      continue;
    }

    int node_id = std::stoi(current_line[node_id_idx]);
    int cluster_id = std::stoi(current_line[cluster_id_idx]);

    this->graph->SetCommunityAssignment(node_id, cluster_id);
    this->graph->AddNodeToCluster(node_id, cluster_id);
    line_no++;
  }
  this->logger.WriteToLogFile(
      "Successfully read community assignment file and assigned clusters to " +
          std::to_string(line_no - 1) + " nodes.",
      Log::info);
}

int ABM::main() {
  if (!this->ValidateBinBoundaries()) {
    return 1;
  }

  this->InitializeSimulation();
  this->RunSimulationLoop();
  this->FinalizeSimulation();

  return 0;
}

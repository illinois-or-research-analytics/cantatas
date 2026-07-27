#ifndef ABM_H
#define ABM_H

#include "citation_engine.h"
#include "graph.h"
#include "neighborhood_search.h"
#include "pcg_random.hpp"
#include "structs.h"
#include "utils.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <map>
#include <omp.h>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

struct SimulationConfig {
  std::string edgelist;
  std::string nodelist;
  std::string out_degree_bag;
  std::string recency_table;
  std::string recency_bins;
  double alpha;
  double minimum_alpha;
  bool use_alpha;
  bool start_from_checkpoint;
  std::string planted_nodes;
  std::string community_assignment;
  double fully_random_citations;
  double preferential_weight;
  double fitness_weight;
  double num_authors_weight;
  double author_reputation_weight;
  int fitness_value_min;
  int fitness_value_max;
  int fitness_lag_duration_min;
  int fitness_lag_duration_max;
  int fitness_peak_duration_min;
  int fitness_peak_duration_max;
  double minimum_preferential_weight;
  double minimum_fitness_weight;
  int in_degree_threshold;
  int fitness_threshold;
  int recency_threshold;
  double non_random_generator_probability;
  int theta;
  double growth_rate;
  int num_cycles;
  double same_year_citations;
  int neighborhood_sample;
  std::string num_authors_bag;
  int author_max_lifetime;
  double cartel_outdegree_proportion;
  bool null_cartel;
  std::string output_file;
  std::string clonal_cartel_agent_file;
  std::string auxiliary_information_file;
  std::string log_file;
  int num_processors;
  int log_level;
};

class ABM {
public:
  SimLogger logger;

  Graph *graph;
  std::unordered_map<int, int> continuous_node_mapping;
  std::vector<int> reverse_continuous_node_mapping;
  int start_year;
  int next_node_id;
  int initial_next_node_id;
  int initial_graph_size;
  int final_graph_size;
  std::unordered_map<int, int> planted_nodes_line_number_map;

  std::vector<int> in_degree_vec;
  std::vector<int> fitness_vec;
  std::vector<int> num_authors_vec;
  std::vector<int> author_reputation_vec;
  std::vector<double> pa_vec;
  std::vector<double> fit_vec;
  std::vector<double> na_vec;
  std::vector<double> ar_vec;
  std::vector<double> random_weight_vec;
  std::vector<double> current_score_vec;

  std::vector<double> pa_weight_vec;
  std::vector<double> fit_weight_vec;
  std::vector<double> num_authors_weight_vec;
  std::vector<double> author_reputation_weight_vec;
  std::vector<int> out_degree_vec;
  std::vector<double> alpha_vec;
  std::vector<int> fitness_lag_duration_vec;
  std::vector<int> fitness_peak_value_vec;
  std::vector<int> fitness_peak_duration_vec;
  std::vector<int> planted_author_id_vec;

public:
  ABM(const SimulationConfig &config)
      : edgelist(config.edgelist), nodelist(config.nodelist),
        out_degree_bag(config.out_degree_bag),
        recency_table(config.recency_table), recency_bins(config.recency_bins),
        community_assignment(config.community_assignment), alpha(config.alpha),
        minimum_alpha(config.minimum_alpha), use_alpha(config.use_alpha),
        start_from_checkpoint(config.start_from_checkpoint),
        planted_nodes(config.planted_nodes),
        fully_random_citations(config.fully_random_citations),
        preferential_weight(config.preferential_weight),
        fitness_weight(config.fitness_weight),
        num_authors_weight(config.num_authors_weight),
        author_reputation_weight(config.author_reputation_weight),
        fitness_value_min(config.fitness_value_min),
        fitness_value_max(config.fitness_value_max),
        fitness_lag_duration_min(config.fitness_lag_duration_min),
        fitness_lag_duration_max(config.fitness_lag_duration_max),
        fitness_peak_duration_min(config.fitness_peak_duration_min),
        fitness_peak_duration_max(config.fitness_peak_duration_max),
        minimum_preferential_weight(config.minimum_preferential_weight),
        minimum_fitness_weight(config.minimum_fitness_weight),
        in_degree_threshold(config.in_degree_threshold),
        fitness_threshold(config.fitness_threshold),
        recency_threshold(config.recency_threshold),
        non_random_generator_probability(
            config.non_random_generator_probability),
        theta(config.theta), growth_rate(config.growth_rate),
        num_cycles(config.num_cycles),
        same_year_citations(config.same_year_citations),
        neighborhood_sample(config.neighborhood_sample),
        num_authors_bag(config.num_authors_bag),
        author_max_lifetime(config.author_max_lifetime),
        cartel_outdegree_proportion(config.cartel_outdegree_proportion),
        null_cartel(config.null_cartel), output_file(config.output_file),
        clonal_cartel_agent_file(config.clonal_cartel_agent_file),
        auxiliary_information_file(config.auxiliary_information_file),
        log_file(config.log_file), num_processors(config.num_processors),
        log_level(config.log_level) {
    // initial validation
    if (this->log_file == "") {
      std::cerr << "Log file is required" << std::endl;
      exit(1);
    }
    if (this->log_level == -42) {
      std::cerr << "Log level is required" << std::endl;
      exit(1);
    }
    this->logger.Init(this->log_file, this->log_file + "_timing",
                      (int)this->log_level);
    if (!this->ValidateArguments()) {
      exit(1);
    }

    this->out_degree_bag_vec = Utils::ParseOutDegreeBag(this->out_degree_bag);
    this->recency_counts_map =
        Utils::ParseRecencyProbabilities(this->recency_table);
    if (this->planted_nodes != "") {
      this->planted_nodes_map = Utils::ParsePlantedNodes(this->planted_nodes);
    }

    if (this->clonal_cartel_agent_file != "") {
      this->clonal_cartel_agent =
          Utils::ParseClonalCartelAgentStruct(this->clonal_cartel_agent_file);
    }

    this->neighborhood_search = new NeighborhoodSearch(
        this->recency_bins, this->use_alpha, this->neighborhood_sample);
    this->citation_engine =
        new CitationEngine(this->null_cartel, this->neighborhood_search);
    omp_set_num_threads(this->num_processors);
  };

  ~ABM() {
    delete this->neighborhood_search;
    delete this->citation_engine;
  }

  int main();
  /*
  Input: None
  Output: void
  Description: Initializes the entire simulation state, parses configuration,
  loads the initial graph, and pre-allocates vectors for node metrics.
  */
  void InitializeSimulation();
  /*
  Input: None
  Output: void
  Description: The core simulation loop that iteratively advances the graph by
  adding new nodes, resolving citations, and updating temporal attributes each
  simulated year.
  */
  void RunSimulationLoop();
  /*
  Input: None
  Output: void
  Description: Cleans up the simulation, writing out generated graph topologies
  and flushing remaining metrics to disk.
  */
  void FinalizeSimulation();
  /*
  Input: std::string message, Log message_type
  Output: int (status code)
  Description: Delegates logging out to the centralized SimLogger, standardizing
  outputs based on severity level.
  */
  int WriteToLogFile(std::string message, Log message_type);
  /*
  Input: None
  Output: bool (true if valid)
  Description: Verifies that all parsed command-line or config file arguments
  meet structural and logical requirements before execution begins.
  */
  bool ValidateArguments();
  /*
  Input: None
  Output: bool (true if valid)
  Description: Ensures that the parsed string for recency bins (e.g. "0-2,3-5")
  translates into a valid, monotonic set of integers.
  */
  bool ValidateBinBoundaries();
  /*
  Input: None
  Output: void
  Description: Reads the community cluster assignments from the CSV file
  specified in the configuration and populates the graph with this inherited
  clustering data.
  */
  void ReadCommunityAssignment();
  /*
  Input: Graph *graph
  Output: std::unordered_map<int, int> (continuous node mapping)
  Description: Builds a mapping from potentially non-contiguous graph node IDs
  to a continuous, 0-indexed integer range, which is required for array
  allocations.
  */
  std::unordered_map<int, int> BuildContinuousNodeMapping(Graph *graph);
  /*
  Input: const std::unordered_map<int, int> &mapping
  Output: std::vector<int> (reverse mapping)
  Description: Inverts the continuous node mapping, returning a vector where the
  index is the continuous ID and the value is the original graph node ID.
  */
  std::vector<int> ReverseMapping(const std::unordered_map<int, int> &mapping);
  /*
  Input: Graph *graph
  Output: int (total projected size)
  Description: Estimates the final expected graph size based on the configured
  growth rate and the number of simulation cycles, used for memory
  pre-allocation.
  */
  int GetFinalGraphSize(Graph *graph);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &reverse_continuous_node_mapping Output: std::vector<int> (generator
  candidates) Description: Fetches nodes that are eligible to act as
  'generators' (sources of citations) for the current simulation step based on
  configuration.
  */
  std::vector<int>
  GetGeneratorNodes(Graph *graph,
                    const std::vector<int> &reverse_continuous_node_mapping);
  /*
  Input: Graph *graph, int graph_size, const std::unordered_map<int, int>
  &reverse_continuous_node_mapping, std::span<int> in_degree_span,
  std::span<int> fitness_span, int in_degree_threshold, int fitness_threshold,
  int start_year, int current_year, int recency_threshold Output:
  std::vector<int> (eligible candidates) Description: Aggressively filters
  generator nodes based on strict criteria such as minimum in-degree, fitness
  thresholds, and recency limits.
  */
  std::vector<int> GetEligibleGeneratorNodes(
      Graph *graph, int graph_size,
      const std::vector<int> &reverse_continuous_node_mapping,
      std::span<int> in_degree_span, std::span<int> fitness_span,
      int in_degree_threshold, int fitness_threshold, int start_year,
      int current_year, int recency_threshold);
  std::
      vector<int>
      /*
      Input: std::vector<int> &eligible_generator_nodes
      Output: std::vector<int> (sampled nodes)
      Description: Extracts a uniform sample or the entirety of eligible
      generator nodes depending on whether a cap is enforced by the
      configuration.
      */
      GetGeneratorNodesFromSet(std::vector<int> &eligible_generator_nodes);
  /*
  Input: Graph *graph, int new_node
  Output: std::vector<int>
  Description: Obtains generator nodes explicitly linked to a specific newly
  injected node, primarily for assigning attribute inheritance.
  */
  std::vector<int> GetGraphAttributesGeneratorNodes(Graph *graph,
                                                    int new_node) const;
  NeighborhoodSearch *neighborhood_search;
  CitationEngine *citation_engine;
  std::unordered_map<int, double> GetBinnedRecencyProbabilities();
  /*
  Input: Graph *graph
  Output: void
  Description: Triggers the initial population of fitness metadata (peak value,
  lag duration) for all nodes pre-existing in the empirical base graph.
  */
  void InitializeFitness(Graph *graph);
  /*
  Input: Graph *graph, const std::vector<int> &new_nodes_vec
  Output: void
  Description: Assigns baseline author reputations to newly introduced nodes in
  the graph at their time of creation.
  */
  void UpdateGraphAttributesInitialAuthorReputations(
      Graph *graph, const std::vector<int> &new_nodes_vec);
  /*
  Input: Graph *graph
  Output: int (highest year found)
  Description: Scans the graph to identify the maximum publication year among
  all existing nodes, used to align the simulation clock.
  */
  int GetMaxYear(Graph *graph);
  /*
  Input: Graph *graph
  Output: int (highest node ID found)
  Description: Scans the graph to identify the highest unique node ID,
  establishing the starting point for injecting simulated nodes.
  */
  int GetMaxNode(Graph *graph);

  /*
  Input: std::set<int> &same_year_source_nodes, int current_year_new_nodes
  Output: void
  Description: Populates a set representing nodes published in the exact same
  year to simulate horizontal citation behaviors (e.g., citing contemporaneous
  preprints).
  */
  void FillSameYearSourceNodes(std::set<int> &same_year_source_nodes,
                               int current_year_new_nodes);

  /*
  Input: Graph *graph, int next_node_id, std::span<double> pa_weight_span,
  std::span<double> fit_weight_span, std::span<double> num_authors_weight_span,
  std::span<double> author_reputation_weight_span, int len Output: void
  Description: Bulk-updates the graph's internal attribute storage with the
  generated weights for PA, fitness, authors, and reputation.
  */
  void UpdateGraphAttributesWeights(
      Graph *graph, int next_node_id, std::span<double> pa_weight_span,
      std::span<double> fit_weight_span,
      std::span<double> num_authors_weight_span,
      std::span<double> author_reputation_weight_span, int len);
  /*
  Input: Graph *graph, int next_node_id, std::span<int> out_degree_span, int len
  Output: void
  Description: Bulk-updates the graph's internal attribute storage with the
  generated out-degrees for new nodes.
  */
  void UpdateGraphAttributesOutDegrees(Graph *graph, int next_node_id,
                                       std::span<int> out_degree_span, int len);
  void
  UpdateGraphAttributesGeneratorNodes(Graph *graph, int new_node,
                                      const std::vector<int> &generator_nodes);
  /*
  Input: Graph *graph, int new_node, int author_id
  Output: void
  Description: Links a newly generated node to a specific author ID, especially
  relevant when tracking cartel behaviors.
  */
  void UpdateGraphAttributesAuthors(Graph *graph, int new_node, int author_id);
  /*
  Input: Graph *graph, int next_node_id, std::span<double> alpha_span, int len
  Output: void
  Description: Bulk-updates the graph's internal attribute storage with the
  intrinsic attractiveness (alpha) assigned to each new node.
  */
  void UpdateGraphAttributesAlphas(Graph *graph, int next_node_id,
                                   std::span<double> alpha_span, int len);
  /*
  Input: Graph *graph, int next_node_id, const std::unordered_map<int, int>
  &planted_nodes_line_number_map Output: void Description: Marks specific nodes
  in the graph with their originating configuration line number, strictly for
  auditability of planted cartel members.
  */
  void UpdateGraphAttributesPlantedNodesLineNumbers(
      Graph *graph, int next_node_id,
      const std::unordered_map<int, int> &planted_nodes_line_number_map);
  /*
  Input: Graph *graph, const std::vector<int> &new_nodes_vec, const
  std::unordered_map<int, int> &continuous_node_mapping, std::span<int>
  fitness_lag_duration_span, std::span<int> fitness_peak_value_span,
  std::span<int> fitness_peak_duration_span, int initial_graph_size Output: void
  Description: Bulk-updates the temporal fitness trajectories (lag, peak,
  duration) into the graph attributes for new nodes.
  */
  void UpdateGraphAttributesFitnesses(
      Graph *graph, const std::vector<int> &new_nodes_vec,
      const std::unordered_map<int, int> &continuous_node_mapping,
      std::span<int> fitness_lag_duration_span,
      std::span<int> fitness_peak_value_span,
      std::span<int> fitness_peak_duration_span, int initial_graph_size);
  /*
  Input: Graph *graph, const std::unordered_map<int, int>
  &continuous_node_mapping, std::span<int> num_authors_span Output: void
  Description: Bulk-updates the author count properties into the graph for new
  nodes.
  */
  void UpdateGraphAttributesNumAuthors(
      Graph *graph, const std::unordered_map<int, int> &continuous_node_mapping,
      std::span<int> num_authors_span);

  /*
  Input: int current_year, std::string label (optional: int time_elapsed)
  Output: void
  Description: Utility function to emit elapsed timing benchmarks for specific
  simulation stages at a given simulated year.
  */
  void LogTime(int current_year, std::string label);
  void LogTime(int current_year, std::string label, int time_elapsed);
  /*
  Input: int start_year, int end_year
  Output: void
  Description: Dumps all recorded simulation step timings into a persistent CSV
  file for post-run performance profiling.
  */
  void WriteTimingFile(int start_year, int end_year);
  std::
      unordered_map<int, int>
      /*
      Input: Graph *graph, various spans for weights and attributes (e.g.,
      pa_weight_span) Output: std::unordered_map<int, int> (mapping of planted
      nodes) Description: Intentionally injects pre-configured 'cartel' nodes
      into the simulation state with predefined malicious attributes before the
      main random generation happens.
      */
      PlantNodes(Graph *graph, std::span<double> pa_weight_span,
                 std::span<double> fit_weight_span,
                 std::span<double> num_authors_weight_span,
                 std::span<double> author_reputation_weight_span,
                 std::span<int> out_degree_span, std::span<double> alpha_span,
                 std::span<int> fitness_lag_duration_span,
                 std::span<int> fitness_peak_value_span,
                 std::span<int> fitness_peak_duration_span,
                 std::span<int> num_authors_span,
                 std::span<int> planted_author_id_span);

  /*
  Input: Graph *graph, int author_id
  Output: std::vector<int> (list of node IDs)
  Description: Finds all nodes historically published by a specific cartel
  author to act as the nucleus for future cartel citations.
  */
  std::vector<int> GetCartelGeneratorNodes(Graph *graph, int author_id);
  /*
  Input: Graph *graph
  Output: void
  Description: Inline helper that hardcodes baseline static fitness values for
  the very first seed nodes in the graph.
  */
  void InitializeSeedFitness(Graph *graph) {
    for (auto const &node : graph->GetNodeSet()) {
      int fitness_lag_uniform = 0;     // MARK: hard coded to be static fitness
      int fitness_peak_uniform = 1000; // MARK: hard coded to be static fitness
      int fitness_power = 1;

      graph->SetFitnessLagDuration(node, fitness_lag_uniform);
      graph->SetFitnessPeakDuration(node, fitness_peak_uniform);
      graph->SetFitnessPeakValue(node, fitness_power);
    }
  }

  template <typename T1, typename T2>
  /*
  Input: std::string section, std::string argument_name, T1 argument_value, T2
  default_value Output: bool (true if valid) Description: Generic template to
  assert that a provided argument differs from its uninitialized default state.
  */
  bool ValidateArgument(std::string section, std::string argument_name,
                        T1 argument_value, T2 default_value) {
    if (argument_value == default_value) {
      this->logger.WriteToLogFile("Required parameter '" + argument_name +
                                      "' was not found in the '" + section +
                                      "' section",
                                  Log::error);
      return false;
    }
    std::ostringstream oss;
    oss << argument_name << ": " << argument_value;
    this->logger.WriteToLogFile(oss.str(), Log::info);
    return true;
  }

  template <typename T>
  void AssignFitnessLagDuration(Graph *graph, const T &container) {
    for (auto const &node : container) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      int fitness_lag_uniform =
          this->fitness_lag_duration_uniform_distribution(generator);
      // int fitness_lag_uniform = 0; // MARK: hard coded to be static fitness
      graph->SetFitnessLagDuration(node, fitness_lag_uniform);
    }
  }

  template <typename T>
  void AssignFitnessPeakDuration(Graph *graph, const T &container) {
    for (auto const &node : container) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      int fitness_peak_uniform =
          this->fitness_peak_duration_uniform_distribution(generator);
      // int fitness_peak_uniform = 1000; // MARK: hard coded to be static
      // fitness
      graph->SetFitnessPeakDuration(node, fitness_peak_uniform);
    }
  }

  template <typename T>
  void AssignPeakFitnessValues(Graph *graph, const T &container) {
    pcg32 &generator = Utils::GetThreadLocalPRNG();
    /*
    std::vector<double> fitness_probabilities;
    double fitness_probabilities_sum = 0.0;
    for(int i = this->fitness_value_min; i <  this->fitness_value_max + 1; i ++)
    { double scale_factor = 6.3742991333; double constant = 0.072; double
    exponent = -1.634; double current_fitness_probability = scale_factor *
    constant * pow(i, exponent);
        fitness_probabilities.push_back(current_fitness_probability);
        fitness_probabilities_sum += current_fitness_probability;
    }

    for(size_t i = 0; i < fitness_probabilities.size(); i ++) {
        fitness_probabilities[i] /= fitness_probabilities_sum;
    }

    std::discrete_distribution<int>
    int_discrete_distribution(fitness_probabilities.begin(),
    fitness_probabilities.end());
    */
    for (auto const &node : container) {
      double fitness_uniform =
          this->fitness_value_uniform_distribution(generator);
      double adjusted_alpha = this->fitness_alpha + 1;
      double base_left = (pow(this->fitness_value_max, adjusted_alpha) -
                          pow(this->fitness_value_min, adjusted_alpha)) *
                         fitness_uniform;
      double base_right = pow(this->fitness_value_min, adjusted_alpha);
      double exponent = 1.0 / adjusted_alpha;
      int fitness_power = std::round(pow(base_left + base_right, exponent));
      graph->SetFitnessPeakValue(node, fitness_power);
    }
  }

protected:
  std::string edgelist;
  std::string nodelist;
  std::string out_degree_bag;
  std::string recency_table;
  std::string recency_bins;
  std::string community_assignment;
  double alpha;
  double minimum_alpha;
  bool use_alpha;
  bool start_from_checkpoint;
  std::string planted_nodes;
  double fully_random_citations;
  double preferential_weight;
  double fitness_weight;
  double num_authors_weight;
  double author_reputation_weight;
  int fitness_value_min;
  int fitness_value_max;
  int fitness_lag_duration_min;
  int fitness_lag_duration_max;
  int fitness_peak_duration_min;
  int fitness_peak_duration_max;
  double minimum_preferential_weight;
  double minimum_fitness_weight;
  int in_degree_threshold;
  int fitness_threshold;
  int recency_threshold;
  double non_random_generator_probability;
  int theta;
  double growth_rate;
  int num_cycles;
  double same_year_citations;
  int neighborhood_sample;
  std::string num_authors_bag;
  int author_max_lifetime;
  double cartel_outdegree_proportion;
  bool null_cartel;
  std::string output_file;
  std::string clonal_cartel_agent_file;
  std::string auxiliary_information_file;
  std::string log_file;
  int num_processors;
  int log_level;

  const int fitness_alpha = -3;
  const int fitness_decay_alpha = 3;
  const int gamma = 3;
  const int max_author_lifetime = 30;
  const int k = 2;
  const int recency_limit = 3;
  const int peak_constant = 2;
  const int delay_constant = 500;
  const int max_out_degree = 1500;
  int next_author_id = 0;
  int num_bins;
  bool clonal_agent = false;
  std::uniform_real_distribution<double> fitness_value_uniform_distribution{0,
                                                                            1};
  std::uniform_real_distribution<double> weights_uniform_distribution{0, 1};
  std::uniform_real_distribution<double> wrs_uniform_distribution{0, 1};
  std::uniform_real_distribution<double> alpha_uniform_distribution{0, 1};
  std::uniform_int_distribution<int> fitness_lag_duration_uniform_distribution{
      fitness_lag_duration_min, fitness_lag_duration_max};
  std::uniform_int_distribution<int> fitness_peak_duration_uniform_distribution{
      fitness_peak_duration_min, fitness_peak_duration_max};
  std::vector<int> out_degree_bag_vec;
  std::vector<int> bin_boundaries;
  std::unordered_map<int, int> recency_counts_map;
  std::unordered_map<
      int,
      std::unordered_map<int, std::unordered_map<std::string, std::string>>>
      planted_nodes_map;

  ClonalCartelAgent clonal_cartel_agent;
};

#endif

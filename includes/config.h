#ifndef CANTATAS_CONFIG_H
#define CANTATAS_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

/*
 * SimulationConfig: everything read from the .ini file.
 *
 * Existing config files keep working. Keys that no longer do anything
 * (storage_backend, db_path, use_distributed, ghost_hub_threshold, cartel
 * settings) are accepted and ignored; planted_nodes and
 * clonal_cartel_agent_file must be empty because cartels and planted agents
 * are not supported in this version.
 */
struct SimulationConfig {
  // [Environment]
  std::string nodelist;
  std::string edgelist;
  std::string community_assignment;
  std::string recency_table;
  std::string out_degree_bag;
  double growth_rate = 0.0;
  int num_cycles = 0;
  std::string recency_bins;
  bool start_from_checkpoint = false;

  // [Agent]
  double alpha = -42;
  double minimum_alpha = -42; // > 0 with a fixed alpha: draw alpha uniformly from [minimum_alpha, 1)
  bool use_alpha = false;
  int theta = -42;
  double same_year_citations = 0.0;
  double fully_random_citations = 0.0;
  double preferential_weight = -1;
  double fitness_weight = -1;
  double num_authors_weight = -1;
  double author_reputation_weight = -1;
  int fitness_value_min = 1;
  int fitness_value_max = 1000;
  int fitness_lag_duration_min = 0;
  int fitness_lag_duration_max = 0;
  int fitness_peak_duration_min = 1000;
  int fitness_peak_duration_max = 1000;
  int neighborhood_sample = 10000;
  int in_degree_threshold = 5;
  int fitness_threshold = 20;
  int recency_threshold = 10;
  double non_random_generator_probability = 0.0;
  int author_max_lifetime = 30;
  std::string num_authors_bag;
  double attachment_exponent = 3.0; // exponent applied to all four score components

  // [General]
  std::string output_file;
  std::string auxiliary_information_file;
  std::string log_file;
  int num_processors = 1;
  int log_level = 1;
  std::string checkpoint_dir;          // default: <output dir>/checkpoints
  double checkpoint_interval_minutes = 120.0; // 0 = only when pausing
  int checkpoints_to_keep = 2;
  uint64_t random_seed = 0;            // 0 = draw one at start
  bool verify_snapshots = true;
  double wall_clock_limit = 0.0;       // seconds from process start; 0 = none
  double safety_margin_seconds = 1200.0;

  // Set from the command line (--allow-model-change): resume even if the
  // model parameters differ from the run that wrote the snapshot.
  bool allow_model_change = false;

  // Hash of every parameter that affects the model; stored in snapshots so a
  // resumed run can warn when the model changed.
  uint64_t ModelFingerprint() const;
};

// Parses and validates the config. Throws std::runtime_error with a readable
// message on any problem.
SimulationConfig LoadConfig(const std::string &path);

#endif // CANTATAS_CONFIG_H

#ifndef AGENT_FACTORY_H
#define AGENT_FACTORY_H

#include "config.h"
#include "ladybug/ladybug_db.h"

#include <cstdint>
#include <vector>

/*
 * AgentBatch: one year's new agents, created on rank 0 and broadcast to all
 * ranks. Agent i gets node ID first_id + i.
 */
struct AgentBatch {
  int32_t year = 0;
  int32_t first_id = 0;
  int32_t count = 0;
  int32_t next_author_id = 0; // author count after this batch (new authors are born this year)
  std::vector<int32_t> author;
  std::vector<int32_t> generator;
  std::vector<int32_t> cluster;
  std::vector<int32_t> out_degree;
  std::vector<int32_t> fitness_lag;
  std::vector<int32_t> fitness_peak;
  std::vector<int32_t> fitness_duration;
  std::vector<int32_t> num_authors;
  std::vector<double> pa_weight;
  std::vector<double> fit_weight;
  std::vector<double> num_authors_weight;
  std::vector<double> author_reputation_weight;
  std::vector<double> alpha;
  std::vector<uint8_t> same_year_source;

  void Resize(int32_t n);
};

/*
 * AgentFactory: draws every new agent's attributes, author and generator.
 * Runs on rank 0 only, from one random stream per year, so the result does
 * not depend on the number of ranks or threads.
 */
class AgentFactory {
public:
  AgentFactory(const SimulationConfig &config, std::vector<int> out_degree_bag,
               std::vector<int> num_authors_bag);

  void Create(ladybug::LadyBugDB &db, int32_t year, int32_t count, uint64_t run_seed,
              AgentBatch &batch) const;

  int MaxOutDegree() const { return max_out_degree_; }

private:
  std::vector<int32_t> EligibleGenerators(const ladybug::LadyBugDB &db, int32_t year) const;

  const SimulationConfig &config_;
  std::vector<int> out_degree_bag_;
  std::vector<int> num_authors_bag_;
  int max_out_degree_ = 0;
};

// Fitness of a node in a given year (the original FillFitnessSpan formula,
// including its truncation to an integer).
int FitnessAt(int published_year, int lag, int peak_value, int peak_duration, int current_year,
              int decay_alpha);

#endif // AGENT_FACTORY_H

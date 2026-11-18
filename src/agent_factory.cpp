#include "agent_factory.h"
#include "rng.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace {
constexpr int kFitnessAlpha = -3;      // power-law exponent of peak fitness
constexpr int kFitnessDecayAlpha = 3;  // decay exponent after the peak
} // namespace

int FitnessAt(int published_year, int lag, int peak_value, int peak_duration, int current_year,
              int decay_alpha) {
  if (published_year + lag > current_year) {
    return 1;
  }
  if (published_year + lag + peak_duration >= current_year) {
    return peak_value;
  }
  double decayed = peak_value / std::pow(current_year - published_year - lag - peak_duration + 1, decay_alpha);
  return static_cast<int>(decayed);
}

void AgentBatch::Resize(int32_t n) {
  count = n;
  for (auto *v : {&author, &generator, &cluster, &out_degree, &fitness_lag, &fitness_peak,
                  &fitness_duration, &num_authors}) {
    v->assign(n, 0);
  }
  for (auto *v : {&pa_weight, &fit_weight, &num_authors_weight, &author_reputation_weight, &alpha}) {
    v->assign(n, 0.0);
  }
  same_year_source.assign(n, 0);
}

AgentFactory::AgentFactory(const SimulationConfig &config, std::vector<int> out_degree_bag,
                           std::vector<int> num_authors_bag)
    : config_(config), out_degree_bag_(std::move(out_degree_bag)),
      num_authors_bag_(std::move(num_authors_bag)) {
  if (out_degree_bag_.empty() || num_authors_bag_.empty()) {
    throw std::runtime_error("AgentFactory: empty out-degree or num_authors bag");
  }
  max_out_degree_ = *std::max_element(out_degree_bag_.begin(), out_degree_bag_.end());
}

std::vector<int32_t> AgentFactory::EligibleGenerators(const ladybug::LadyBugDB &db, int32_t year) const {
  // Port of the original GetEligibleGeneratorNodes: among recent papers, the
  // intersection of the top in_degree_threshold% by in-degree and the top
  // fitness_threshold% by fitness (seeds count as maximum fitness).
  const size_t n = db.NumGraphNodes();
  std::vector<std::pair<double, int32_t>> by_degree, by_fitness;
  const bool early = (year - db.meta.start_year) <= config_.recency_threshold;
  for (size_t i = n; i-- > 0;) {
    int32_t v = static_cast<int32_t>(i);
    int pub = db.nodes.year[v];
    if (year - pub > config_.recency_threshold) {
      if (early) continue;
      break;
    }
    by_degree.emplace_back(db.csr.InDegree(v), v);
    double fit = (db.nodes.type[v] == static_cast<uint8_t>(ladybug::NodeType::Seed) && early)
                     ? config_.fitness_value_max
                     : FitnessAt(pub, db.nodes.fitness_lag[v], db.nodes.fitness_peak[v],
                                 db.nodes.fitness_duration[v], year, kFitnessDecayAlpha);
    by_fitness.emplace_back(fit, v);
  }
  if (by_degree.empty()) {
    return {};
  }
  auto top = [](std::vector<std::pair<double, int32_t>> &vec, int percent) {
    size_t idx = static_cast<size_t>((vec.size() - 1) * (static_cast<double>(percent) / 100));
    std::nth_element(vec.begin(), vec.begin() + idx, vec.end(),
                     [](const auto &a, const auto &b) { return a.first > b.first; });
    double cutoff = vec[idx].first;
    std::vector<int32_t> ids;
    for (const auto &[score, id] : vec) {
      if (score >= cutoff) ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  };
  std::vector<int32_t> a = top(by_degree, config_.in_degree_threshold);
  std::vector<int32_t> b = top(by_fitness, config_.fitness_threshold);
  std::vector<int32_t> both;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(both));
  return both;
}

void AgentFactory::Create(ladybug::LadyBugDB &db, int32_t year, int32_t count, uint64_t run_seed,
                          AgentBatch &batch) const {
  batch.Resize(count);
  batch.year = year;
  batch.first_id = static_cast<int32_t>(db.NumGraphNodes());
  batch.next_author_id = db.authors.NumAuthors();
  if (count == 0) {
    return;
  }
  rng::SeedThread(run_seed, rng::Stream::AgentCreation, static_cast<uint64_t>(year));
  pcg32 &g = rng::Thread();
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  // Same-year sources: floor(count * same_year_citations) distinct agents.
  size_t num_sources = static_cast<size_t>(std::floor(count * config_.same_year_citations));
  if (num_sources > 0) {
    std::uniform_int_distribution<int32_t> pick(0, count - 1);
    size_t chosen = 0;
    while (chosen < num_sources) {
      int32_t i = pick(g);
      if (!batch.same_year_source[i]) {
        batch.same_year_source[i] = 1;
        ++chosen;
      }
    }
  }

  std::vector<int32_t> eligible;
  if (config_.non_random_generator_probability > 0) {
    eligible = EligibleGenerators(db, year);
  }

  const bool fixed_weights = config_.preferential_weight != -1 && config_.fitness_weight != -1 &&
                             config_.num_authors_weight != -1 && config_.author_reputation_weight != -1;
  std::uniform_int_distribution<size_t> pick_out(0, out_degree_bag_.size() - 1);
  std::uniform_int_distribution<size_t> pick_authors(0, num_authors_bag_.size() - 1);
  std::uniform_int_distribution<int> pick_lag(config_.fitness_lag_duration_min, config_.fitness_lag_duration_max);
  std::uniform_int_distribution<int> pick_duration(config_.fitness_peak_duration_min,
                                                   config_.fitness_peak_duration_max);
  std::uniform_int_distribution<int32_t> pick_any(0, batch.first_id - 1);
  const double a1 = kFitnessAlpha + 1;

  for (int32_t i = 0; i < count; ++i) {
    // Preference weights: fixed from the config, or four uniforms normalized.
    double w[4];
    if (fixed_weights) {
      w[0] = config_.preferential_weight;
      w[1] = config_.fitness_weight;
      w[2] = config_.num_authors_weight;
      w[3] = config_.author_reputation_weight;
    } else {
      for (double &x : w) x = unit(g);
    }
    double sum = w[0] + w[1] + w[2] + w[3];
    batch.pa_weight[i] = w[0] / sum;
    batch.fit_weight[i] = w[1] / sum;
    batch.num_authors_weight[i] = w[2] / sum;
    batch.author_reputation_weight[i] = w[3] / sum;

    if (!config_.use_alpha) {
      batch.alpha[i] = -1;
    } else if (config_.alpha == -1) {
      batch.alpha[i] = std::round(unit(g) * 1000.0) / 1000.0;
    } else if (config_.minimum_alpha > 0) {
      batch.alpha[i] = std::uniform_real_distribution<double>(config_.minimum_alpha, 1.0)(g);
    } else {
      batch.alpha[i] = config_.alpha;
    }

    batch.out_degree[i] = out_degree_bag_[pick_out(g)];
    batch.fitness_lag[i] = pick_lag(g);
    double base_left = (std::pow(config_.fitness_value_max, a1) - std::pow(config_.fitness_value_min, a1)) * unit(g);
    double base_right = std::pow(config_.fitness_value_min, a1);
    batch.fitness_peak[i] = static_cast<int32_t>(std::pow(base_left + base_right, 1.0 / a1));
    batch.fitness_duration[i] = pick_duration(g);
    batch.num_authors[i] = num_authors_bag_[pick_authors(g)];

    batch.author[i] = db.authors.NextAuthor(year, g);

    double draw = unit(g);
    int32_t generator;
    if (draw < config_.non_random_generator_probability && !eligible.empty()) {
      std::uniform_int_distribution<size_t> pick_eligible(0, eligible.size() - 1);
      generator = eligible[pick_eligible(g)];
    } else {
      generator = pick_any(g);
    }
    batch.generator[i] = generator;
    batch.cluster[i] = db.nodes.cluster[generator];
  }
  batch.next_author_id = db.authors.NumAuthors();
}

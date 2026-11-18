#include "config.h"
#include "INIReader.h"
#include "rng.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace {

const char *kMissing = "\x01missing";

class Reader {
public:
  explicit Reader(const std::string &path) : ini_(path), path_(path) {
    if (ini_.ParseError() < 0) {
      throw std::runtime_error("Cannot open config file " + path);
    }
    if (ini_.ParseError() > 0) {
      throw std::runtime_error("Parse error in " + path + " on line " + std::to_string(ini_.ParseError()));
    }
  }

  std::string Required(const char *section, const char *key) {
    std::string v = ini_.Get(section, key, kMissing);
    if (v == kMissing) {
      throw std::runtime_error(std::string("Required value '") + key + "' is missing from section [" +
                               section + "] in " + path_);
    }
    return v;
  }
  std::string Optional(const char *section, const char *key, const std::string &def = "") {
    std::string v = ini_.Get(section, key, kMissing);
    return v == kMissing ? def : v;
  }
  bool Has(const char *section, const char *key) { return ini_.Get(section, key, kMissing) != kMissing; }

  double Real(const char *section, const char *key) { return ToReal(key, Required(section, key)); }
  double Real(const char *section, const char *key, double def) {
    std::string v = Optional(section, key);
    return v.empty() ? def : ToReal(key, v);
  }
  long long Int(const char *section, const char *key) { return ToInt(key, Required(section, key)); }
  long long Int(const char *section, const char *key, long long def) {
    std::string v = Optional(section, key);
    return v.empty() ? def : ToInt(key, v);
  }
  bool Bool(const char *section, const char *key) { return ToBool(key, Required(section, key)); }
  bool Bool(const char *section, const char *key, bool def) {
    std::string v = Optional(section, key);
    return v.empty() ? def : ToBool(key, v);
  }

private:
  static double ToReal(const char *key, const std::string &v) {
    try {
      size_t used = 0;
      double d = std::stod(v, &used);
      return d;
    } catch (...) {
      throw std::runtime_error(std::string("Value of '") + key + "' is not a number: '" + v + "'");
    }
  }
  static long long ToInt(const char *key, const std::string &v) {
    try {
      return std::stoll(v);
    } catch (...) {
      throw std::runtime_error(std::string("Value of '") + key + "' is not an integer: '" + v + "'");
    }
  }
  static bool ToBool(const char *key, const std::string &v) {
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    throw std::runtime_error(std::string("Value of '") + key + "' must be true or false, got '" + v + "'");
  }

  INIReader ini_;
  std::string path_;
};

void Require(bool ok, const std::string &message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

} // namespace

uint64_t SimulationConfig::ModelFingerprint() const {
  std::ostringstream s;
  s << growth_rate << '|' << recency_bins << '|' << alpha << '|' << minimum_alpha << '|' << use_alpha << '|'
    << theta << '|'
    << same_year_citations << '|' << fully_random_citations << '|' << preferential_weight << '|'
    << fitness_weight << '|' << num_authors_weight << '|' << author_reputation_weight << '|'
    << fitness_value_min << '|' << fitness_value_max << '|' << fitness_lag_duration_min << '|'
    << fitness_lag_duration_max << '|' << fitness_peak_duration_min << '|' << fitness_peak_duration_max
    << '|' << neighborhood_sample << '|' << in_degree_threshold << '|' << fitness_threshold << '|'
    << recency_threshold << '|' << non_random_generator_probability << '|' << author_max_lifetime << '|'
    << attachment_exponent << '|' << out_degree_bag << '|' << num_authors_bag << '|' << recency_table;
  uint64_t h = 0x1234ABCDULL;
  for (char c : s.str()) h = rng::Combine(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
  return h;
}

SimulationConfig LoadConfig(const std::string &path) {
  Reader r(path);
  SimulationConfig c;

  c.nodelist = r.Required("Environment", "nodelist");
  c.edgelist = r.Required("Environment", "edgelist");
  c.community_assignment = r.Optional("Environment", "community_assignment");
  c.recency_table = r.Required("Environment", "recency_table");
  c.out_degree_bag = r.Required("Environment", "out_degree_bag");
  c.growth_rate = r.Real("Environment", "growth_rate");
  c.num_cycles = static_cast<int>(r.Int("Environment", "num_cycles"));
  c.recency_bins = r.Required("Environment", "recency_bins");
  c.start_from_checkpoint = r.Bool("Environment", "start_from_checkpoint", false);
  Require(r.Optional("Environment", "planted_nodes").empty(),
          "planted_nodes is not supported in this version; leave it empty");

  c.use_alpha = r.Bool("Agent", "use_alpha");
  std::string alpha = r.Optional("Agent", "alpha");
  if (c.use_alpha) {
    Require(!alpha.empty(), "alpha is required when use_alpha=true (-1 draws a random alpha per agent)");
    c.alpha = r.Real("Agent", "alpha");
    Require(c.alpha == -1 || (c.alpha >= 0 && c.alpha <= 1), "alpha must be -1 or in [0, 1]");
  } else {
    Require(alpha.empty(), "use_alpha=false but alpha is set; leave the line as 'alpha='");
  }
  c.minimum_alpha = r.Real("Agent", "minimum_alpha", -42);
  c.theta = static_cast<int>(r.Int("Agent", "theta", -42));
  if (!c.community_assignment.empty()) {
    Require(c.theta > 0, "theta must be greater than zero when community_assignment is used");
  }
  c.same_year_citations = r.Real("Agent", "same_year_citations");
  c.fully_random_citations = r.Real("Agent", "fully_random_citations");
  c.preferential_weight = r.Real("Agent", "preferential_weight");
  c.fitness_weight = r.Real("Agent", "fitness_weight");
  c.num_authors_weight = r.Real("Agent", "num_authors_weight");
  c.author_reputation_weight = r.Real("Agent", "author_reputation_weight");
  c.fitness_value_min = static_cast<int>(r.Int("Agent", "fitness_value_min"));
  c.fitness_value_max = static_cast<int>(r.Int("Agent", "fitness_value_max"));
  c.fitness_lag_duration_min = static_cast<int>(r.Int("Agent", "fitness_lag_duration_min"));
  c.fitness_lag_duration_max = static_cast<int>(r.Int("Agent", "fitness_lag_duration_max"));
  c.fitness_peak_duration_min = static_cast<int>(r.Int("Agent", "fitness_peak_duration_min"));
  c.fitness_peak_duration_max = static_cast<int>(r.Int("Agent", "fitness_peak_duration_max"));
  c.neighborhood_sample = static_cast<int>(r.Int("Agent", "neighborhood_sample"));
  c.in_degree_threshold = static_cast<int>(r.Int("Agent", "in_degree_threshold"));
  c.fitness_threshold = static_cast<int>(r.Int("Agent", "fitness_threshold"));
  c.recency_threshold = static_cast<int>(r.Int("Agent", "recency_threshold"));
  c.non_random_generator_probability = r.Real("Agent", "non_random_generator_probability");
  c.author_max_lifetime = static_cast<int>(r.Int("Agent", "author_max_lifetime"));
  c.num_authors_bag = r.Required("Agent", "num_authors_bag");
  c.attachment_exponent = r.Real("Agent", "attachment_exponent", 3.0);
  Require(r.Optional("Agent", "clonal_cartel_agent_file").empty(),
          "clonal_cartel_agent_file is not supported in this version; leave it empty");

  c.output_file = r.Required("General", "output_file");
  c.auxiliary_information_file = r.Required("General", "auxiliary_information_file");
  c.log_file = r.Required("General", "log_file");
  c.num_processors = static_cast<int>(r.Int("General", "num_processors"));
  c.log_level = static_cast<int>(r.Int("General", "log_level"));
  std::string default_ckpt =
      (std::filesystem::path(c.output_file).parent_path() / "checkpoints").string();
  c.checkpoint_dir = r.Optional("General", "checkpoint_dir", default_ckpt);
  if (c.checkpoint_dir.empty()) c.checkpoint_dir = default_ckpt;
  c.checkpoint_interval_minutes = r.Real("General", "checkpoint_interval_minutes", 120.0);
  c.checkpoints_to_keep = static_cast<int>(r.Int("General", "checkpoints_to_keep", 2));
  c.random_seed = static_cast<uint64_t>(r.Int("General", "random_seed", 0));
  c.verify_snapshots = r.Bool("General", "verify_snapshots", true);
  c.wall_clock_limit = r.Real("General", "wall_clock_limit", 0.0);
  c.safety_margin_seconds = r.Real("General", "safety_margin_seconds", 1200.0);

  // Sanity checks.
  Require(c.growth_rate > 0, "growth_rate must be positive");
  Require(c.num_cycles > 0, "num_cycles must be positive");
  Require(c.same_year_citations >= 0 && c.same_year_citations <= 1, "same_year_citations must be in [0, 1]");
  Require(c.fully_random_citations >= 0 && c.fully_random_citations <= 1, "fully_random_citations must be in [0, 1]");
  bool all_fixed = c.preferential_weight != -1 && c.fitness_weight != -1 && c.num_authors_weight != -1 &&
                   c.author_reputation_weight != -1;
  if (all_fixed) {
    Require(c.preferential_weight + c.fitness_weight + c.num_authors_weight + c.author_reputation_weight > 0,
            "the four preference weights must not all be zero");
  }
  Require(c.fitness_value_min >= 1 && c.fitness_value_min <= c.fitness_value_max,
          "need 1 <= fitness_value_min <= fitness_value_max");
  Require(c.fitness_lag_duration_min <= c.fitness_lag_duration_max, "fitness_lag_duration_min > max");
  Require(c.fitness_peak_duration_min <= c.fitness_peak_duration_max, "fitness_peak_duration_min > max");
  Require(c.neighborhood_sample == -1 || c.neighborhood_sample > 0, "neighborhood_sample must be -1 or positive");
  Require(c.num_processors > 0, "num_processors must be positive");
  Require(c.non_random_generator_probability >= 0 && c.non_random_generator_probability <= 1,
          "non_random_generator_probability must be in [0, 1]");
  Require(c.attachment_exponent > 0, "attachment_exponent must be positive");
  Require(c.checkpoints_to_keep >= 1, "checkpoints_to_keep must be at least 1");
  return c;
}

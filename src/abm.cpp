#include "abm.h"
#include "citation_engine.h"
#include "hash_set.h"
#include "rng.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr int kFitnessDecayAlpha = 3;

double SecondsSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::string RankSuffix(const distributed::ClusterCoordinator &c) {
  return c.WorldSize() > 1 ? "_rank_" + std::to_string(c.Rank()) : "";
}

// Per-thread timing of the stages inside the agent loop (summed thread time).
enum Stage { kRetrieve, kSameYear, kBin, kNeighborhood, kRandom, kRecord, kNumStages };
const char *kStageLabels[kNumStages] = {"retrieve neighborhood", "make same year citations", "bin neighborhood",
                                        "make neighborhood citations", "make random citations", "record edges"};

// Recency bins of one cluster's members (all members predate the current year).
struct ClusterBins {
  std::vector<std::vector<int>> bins;
  int historical_size = 0;
};

} // namespace

ABM::ABM(const SimulationConfig &config, distributed::ClusterCoordinator &coordinator)
    : config_(config), coordinator_(coordinator),
      search_(config.recency_bins, config.use_alpha, config.neighborhood_sample) {
  omp_set_num_threads(config_.num_processors);

  // Recency probabilities per bin, from the empirical recency table.
  binned_recency_probabilities_.assign(search_.num_bins, 0.0);
  double total = 0;
  for (const auto &[diff, count] : Utils::ParseRecencyCounts(config_.recency_table)) {
    int bin = search_.GetBinIndex(diff);
    if (bin >= 0 && bin < search_.num_bins) {
      binned_recency_probabilities_[bin] += static_cast<double>(count);
      total += static_cast<double>(count);
    }
  }
  if (total > 0) {
    for (double &p : binned_recency_probabilities_) p /= total;
  }
  factory_ = std::make_unique<AgentFactory>(config_, Utils::ParseIntBag(config_.out_degree_bag, "out_degree_bag"),
                                            Utils::ParseIntBag(config_.num_authors_bag, "num_authors_bag"));
  SignalSupervisor::Register();
  supervisor_.Configure(config_.wall_clock_limit, config_.safety_margin_seconds);
}

bool ABM::Initialize(ResumeMode resume) {
  auto t0 = std::chrono::steady_clock::now();
  // Rank 0 decides whether this is a resume so that every rank takes the same
  // path (and reaches the same collective calls).
  uint64_t resume_flag = 0;
  bool have_snapshot = false;
  if (coordinator_.IsMaster()) {
    have_snapshot = ladybug::HasSnapshot(config_.checkpoint_dir);
    resume_flag = (resume == ResumeMode::Yes || (resume == ResumeMode::Auto && have_snapshot)) ? 1 : 0;
  }
  coordinator_.BroadcastU64(resume_flag);
  const bool resuming = resume_flag != 0;

  std::filesystem::path log_path(config_.log_file + RankSuffix(coordinator_));
  if (log_path.has_parent_path()) {
    std::filesystem::create_directories(log_path.parent_path());
  }
  logger_.Init(log_path.string(), log_path.string() + "_timing", config_.log_level, resuming);
  logger_.Write(std::string("CantataS 9.0.0 starting (") + (resuming ? "resume" : "new run") + ", " +
                std::to_string(coordinator_.WorldSize()) + " rank(s), " + std::to_string(config_.num_processors) +
                " threads per rank)");
  db_.authors.Configure(config_.author_max_lifetime);

  if (resuming) {
    std::string from;
    if (!ladybug::LoadLatestSnapshot(db_, config_.checkpoint_dir, config_.verify_snapshots, &from)) {
      logger_.Write("Resume requested but no snapshot found in " + config_.checkpoint_dir, Log::error);
      return false;
    }
    logger_.Write("Resumed from " + from + ": next year " + std::to_string(db_.meta.next_year) + ", end year " +
                  std::to_string(db_.meta.end_year) + ", " + std::to_string(db_.NumNodes()) + " nodes, " +
                  std::to_string(db_.csr.NumEdges()) + " edges (" + std::to_string(SecondsSince(t0)) + " s)");
    if (db_.meta.model_fingerprint != config_.ModelFingerprint()) {
      if (!config_.allow_model_change) {
        logger_.Write("Model parameters in the config differ from the run that wrote this snapshot. Not resuming: "
                      "restore the original config, or pass --allow-model-change to continue with the new one.",
                      Log::error);
        return false;
      }
      logger_.Write("Model parameters changed since the snapshot; continuing with the new config "
                    "(--allow-model-change)",
                    Log::error);
      db_.meta.model_fingerprint = config_.ModelFingerprint();
    }
  } else {
    if (have_snapshot) {
      logger_.Write("Starting a new run; existing snapshots in " + config_.checkpoint_dir +
                        " will be replaced as new ones are written",
                    Log::info);
    }
    uint64_t seed = config_.random_seed;
    if (seed == 0 && coordinator_.IsMaster()) {
      std::random_device rd;
      seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
      if (seed == 0) seed = 1;
    }
    coordinator_.BroadcastU64(seed);
    ladybug::IngestOptions options;
    options.nodelist = config_.nodelist;
    options.edgelist = config_.edgelist;
    options.community_assignment = config_.community_assignment;
    options.num_authors_bag = Utils::ParseIntBag(config_.num_authors_bag, "num_authors_bag");
    options.author_max_lifetime = config_.author_max_lifetime;
    options.run_seed = seed;
    ladybug::IngestSeed(options, db_, [&](const std::string &m) { logger_.Write(m); });
    db_.meta.run_seed = seed;
    db_.meta.end_year = db_.meta.start_year + config_.num_cycles;
    db_.meta.model_fingerprint = config_.ModelFingerprint();
  }
  logger_.Write("Run seed " + std::to_string(db_.meta.run_seed) + "; deadline source: " +
                supervisor_.DeadlineSource());

  // Reserve address space for the projected final size (no memory is used yet).
  size_t nodes = db_.NumGraphNodes();
  double mean_out = 0;
  {
    std::vector<int> bag = Utils::ParseIntBag(config_.out_degree_bag, "out_degree_bag");
    double s = 0;
    for (int d : bag) s += d;
    mean_out = s / static_cast<double>(bag.size());
  }
  size_t final_nodes = nodes;
  for (int32_t y = db_.meta.next_year; y < db_.meta.end_year; ++y) {
    final_nodes += static_cast<size_t>(std::ceil(final_nodes * config_.growth_rate));
  }
  uint64_t final_edges = db_.csr.NumEdges() + static_cast<uint64_t>((final_nodes - nodes) * (mean_out + 1) * 1.05);
  db_.ReserveFinalSize(final_nodes, final_edges);
  logger_.Write("Projected final size: " + std::to_string(final_nodes) + " nodes, ~" + std::to_string(final_edges) +
                " edges");

  partitioner_.Partition(coordinator_.WorldSize(), coordinator_.Rank(), db_);
  last_snapshot_time_ = std::chrono::steady_clock::now();
  return true;
}

void ABM::ComputeScores(int32_t year, size_t num_nodes) {
  scores_.ResizeUninitialized(num_nodes);
  const double gamma = config_.attachment_exponent;
  const ladybug::NodeColumns &n = db_.nodes;
  auto score = [gamma](double x) { return std::max(std::pow(x, gamma), 1.0) + 1; };
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < num_nodes; ++i) {
    const int32_t v = static_cast<int32_t>(i);
    int fitness = FitnessAt(n.year[v], n.fitness_lag[v], n.fitness_peak[v], n.fitness_duration[v], year,
                            kFitnessDecayAlpha);
    NodeScoreComponents &c = scores_[i];
    c.pa = score(static_cast<double>(db_.csr.InDegree(v)));
    c.fit = score(static_cast<double>(fitness));
    c.na = score(static_cast<double>(n.num_authors[v]));
    c.ar = score(static_cast<double>(db_.authors.Reputation(n.author[v])));
  }
}

void ABM::SimulateYear(int32_t year) {
  const size_t num_existing = db_.NumGraphNodes();
  const int32_t count = static_cast<int32_t>(std::ceil(num_existing * config_.growth_rate));
  logger_.ResetLap();
  logger_.Write("current year is: " + std::to_string(year) + " and the graph is " + std::to_string(num_existing) +
                " nodes large");
  logger_.Write("making " + std::to_string(count) + " nodes this year");

  // 1. Rank 0 creates the agents; everyone receives them.
  AgentBatch batch;
  if (coordinator_.IsMaster()) {
    factory_->Create(db_, year, count, db_.meta.run_seed, batch);
  }
  logger_.Lap(year, "Create agents");
  coordinator_.BroadcastBatch(batch);
  db_.authors.SyncNewAuthors(batch.next_author_id, year);
  logger_.Lap(year, "Broadcast agents");

  // Node rows for the new agents (their graph rows are added after the year).
  ladybug::NodeColumns &n = db_.nodes;
  for (int32_t i = 0; i < batch.count; ++i) {
    n.year.PushBack(static_cast<int16_t>(year));
    n.type.PushBack(static_cast<uint8_t>(ladybug::NodeType::Agent));
    n.cluster.PushBack(batch.cluster[i]);
    n.author.PushBack(batch.author[i]);
    n.num_authors.PushBack(batch.num_authors[i]);
    n.fitness_lag.PushBack(batch.fitness_lag[i]);
    n.fitness_peak.PushBack(batch.fitness_peak[i]);
    n.fitness_duration.PushBack(batch.fitness_duration[i]);
    n.generator.PushBack(batch.generator[i]);
    n.alpha.PushBack(static_cast<float>(batch.alpha[i]));
    n.pa_weight.PushBack(static_cast<float>(batch.pa_weight[i]));
    n.fit_weight.PushBack(static_cast<float>(batch.fit_weight[i]));
    n.num_authors_weight.PushBack(static_cast<float>(batch.num_authors_weight[i]));
    n.author_reputation_weight.PushBack(static_cast<float>(batch.author_reputation_weight[i]));
    n.assigned_out_degree.PushBack(batch.out_degree[i]);
    n.sampled_neighborhood_size.PushBack(-1);
    n.fully_random_citations.PushBack(-1);
    n.initial_author_reputation.PushBack(-1);
  }

  // 2. Scores of the existing papers for this year.
  ComputeScores(year, num_existing);
  logger_.Lap(year, "Compute scores");
  const NodeMetrics metrics{scores_.span()};

  // 3. This rank's agents, and recency bins of their clusters.
  std::vector<int32_t> local;
  local.reserve(batch.count / std::max(1, coordinator_.WorldSize()) + 16);
  for (int32_t i = 0; i < batch.count; ++i) {
    if (partitioner_.IsLocal(batch.cluster[i], batch.first_id + i)) local.push_back(i);
  }
  logger_.Value(year, "Rank " + std::to_string(coordinator_.Rank()) + " agents", static_cast<long long>(local.size()));

  std::unordered_map<int32_t, size_t> cluster_slot;
  std::vector<ClusterBins> cluster_bins;
  if (config_.use_alpha) {
    std::vector<int32_t> active;
    for (int32_t i : local) {
      int32_t c = batch.cluster[i];
      if (c >= 0 && batch.alpha[i] > 0) active.push_back(c);
    }
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    cluster_bins.resize(active.size());
    for (size_t s = 0; s < active.size(); ++s) cluster_slot[active[s]] = s;
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t s = 0; s < active.size(); ++s) {
      ClusterBins &cb = cluster_bins[s];
      cb.bins.assign(search_.num_bins, {});
      for (int32_t node : db_.clusters.Members(active[s])) {
        int bin = search_.GetBinIndex(db_, node, year);
        if (bin >= 0 && bin < search_.num_bins) {
          cb.bins[bin].push_back(node);
          cb.historical_size++;
        }
      }
    }
  }
  logger_.Lap(year, "Build cluster bins");

  // 4. Simulate this rank's agents.
  const int num_threads = omp_get_max_threads();
  std::vector<std::vector<int32_t>> thread_edges(num_threads);
  std::vector<std::array<long long, kNumStages>> thread_stage_ms(num_threads);
  for (auto &a : thread_stage_ms) a.fill(0);
  std::vector<int32_t> sampled_sizes(batch.count, -1);
  std::vector<int32_t> fully_random(batch.count, -1);
  int num_sources = 0;
  for (uint8_t s : batch.same_year_source) num_sources += s;
  const bool same_year_possible = num_sources < batch.count;
  const size_t citation_capacity = static_cast<size_t>(factory_->MaxOutDegree()) + 2;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::vector<int32_t> &edges_out = thread_edges[tid];
    auto &stage_ms = thread_stage_ms[tid];
    std::vector<int> citations(citation_capacity);
    std::vector<std::vector<int>> n_hop_map(3);
    std::vector<std::vector<int>> filtered(3);
    std::vector<std::vector<int>> bins;
    std::vector<int> scratch_bin;
    std::vector<int> excluded_nodes;
    std::vector<int> bin_sizes;
    StampedHashSet<int32_t> cited;
    StampedHashSet<int32_t> unique_cited;

#pragma omp for schedule(dynamic, 64)
    for (size_t k = 0; k < local.size(); ++k) {
      const int32_t i = local[k];
      const int32_t agent = batch.first_id + i;
      rng::SeedThread(db_.meta.run_seed, rng::Stream::Agent, static_cast<uint64_t>(year),
                      static_cast<uint64_t>(agent));
      auto lap = std::chrono::steady_clock::now();
      auto stage = [&](Stage s) {
        auto now = std::chrono::steady_clock::now();
        stage_ms[s] += std::chrono::duration_cast<std::chrono::milliseconds>(now - lap).count();
        lap = now;
      };

      const int32_t cluster_id = batch.cluster[i];
      const double alpha = batch.alpha[i];
      const int32_t generator = batch.generator[i];
      const AgentWeights weights{batch.pa_weight[i], batch.fit_weight[i], batch.num_authors_weight[i],
                                 batch.author_reputation_weight[i]};
      const int generator_nodes[1] = {generator};

      search_.GetNeighborhoodMap(db_, generator, n_hop_map);
      stage(kRetrieve);

      // Cluster fast path: distance 1 becomes the cluster's history, and
      // same-cluster nodes are removed from distance 2.
      bool fastpath = false;
      const ClusterBins *cb = nullptr;
      if (alpha > 0 && cluster_id >= 0 &&
          static_cast<int>(db_.clusters.Members(cluster_id).size()) >= config_.theta) {
        auto it = cluster_slot.find(cluster_id);
        if (it != cluster_slot.end()) {
          cb = &cluster_bins[it->second];
          fastpath = true;
          std::vector<int> &hop2 = n_hop_map[2];
          hop2.erase(std::remove_if(hop2.begin(), hop2.end(),
                                    [&](int node) { return db_.nodes.cluster[node] == cluster_id; }),
                     hop2.end());
        }
      }

      const int out_degree = batch.out_degree[i];
      int reserved_random = static_cast<int>(std::floor(config_.fully_random_citations * out_degree));
      int remaining = out_degree - 1 - batch.same_year_source[i];
      if (remaining > reserved_random) {
        remaining -= reserved_random;
      } else {
        reserved_random = 0;
      }

      std::span<int> cites(citations);
      int num_cited = 0;
      if (batch.same_year_source[i] && same_year_possible) {
        num_cited += CitationEngine::MakeSameYearCitation(batch.same_year_source, batch.first_id, batch.count,
                                                          cites);
      }
      stage(kSameYear);

      // Remove already-cited papers from the neighborhoods.
      cited.Clear();
      for (int j = 0; j < num_cited; ++j) cited.Insert(citations[j]);
      for (auto &level : filtered) level.clear();
      for (size_t d = fastpath ? 2 : 1; d < n_hop_map.size(); ++d) {
        for (int node : n_hop_map[d]) {
          if (!cited.Contains(node)) filtered[d].push_back(node);
        }
      }
      std::swap(n_hop_map, filtered);

      auto in_cluster_history = [&](int node) {
        return db_.nodes.cluster[node] == cluster_id && db_.nodes.year[node] < year;
      };

      if (remaining > 0) {
        std::vector<int> quotas;
        size_t hop1_size = 0;
        if (fastpath) {
          int excluded = in_cluster_history(generator) ? 1 : 0;
          for (int j = 0; j < num_cited; ++j) {
            if (citations[j] != generator && in_cluster_history(citations[j])) ++excluded;
          }
          hop1_size = static_cast<size_t>(std::max(0, cb->historical_size - excluded));
          quotas = search_.GetNumCitationsPerNeighborhood(alpha, remaining, hop1_size, n_hop_map[2].size());
        } else if (config_.use_alpha) {
          quotas = search_.GetNumCitationsPerNeighborhood(alpha, remaining, n_hop_map[1].size(),
                                                          n_hop_map[2].size());
        } else {
          // Without alpha the neighborhood is one pool (distances 1 and 2).
          quotas = {0, std::min(remaining, static_cast<int>(n_hop_map[1].size())), 0};
        }

        for (size_t d = 1; d < n_hop_map.size(); ++d) {
          const int quota = (d < quotas.size()) ? quotas[d] : 0;
          if (fastpath && d == 1) {
            sampled_sizes[i] += static_cast<int32_t>(hop1_size);
            bin_sizes.assign(search_.num_bins, 0);
            for (int b = 0; b < search_.num_bins; ++b) bin_sizes[b] = static_cast<int>(cb->bins[b].size());
            excluded_nodes.clear();
            auto exclude = [&](int node) {
              int b = search_.GetBinIndex(db_, node, year);
              if (b >= 0 && b < search_.num_bins && bin_sizes[b] > 0) {
                bin_sizes[b]--;
                excluded_nodes.push_back(node);
              }
            };
            if (in_cluster_history(generator)) exclude(generator);
            for (int j = 0; j < num_cited; ++j) {
              if (citations[j] != generator && in_cluster_history(citations[j])) exclude(citations[j]);
            }
            stage(kBin);
            std::vector<int> per_bin = search_.BinOutdegrees(bin_sizes, quota, binned_recency_probabilities_);
            auto candidates = [&](int b) -> std::span<const int> {
              const std::vector<int> &full = cb->bins[b];
              if (bin_sizes[b] >= static_cast<int>(full.size())) return full;
              scratch_bin.clear();
              for (int node : full) {
                if (std::find(excluded_nodes.begin(), excluded_nodes.end(), node) == excluded_nodes.end()) {
                  scratch_bin.push_back(node);
                }
              }
              return scratch_bin;
            };
            const int last = search_.num_bins - 1;
            for (int b = 0; b < last; ++b) {
              if (per_bin[b] <= 0) continue;
              num_cited += CitationEngine::MakeCitations(candidates(b), cites.subspan(num_cited), metrics, weights,
                                                         per_bin[b]);
            }
            if (per_bin[last] > 0) {
              num_cited += CitationEngine::MakeUniformRandomCitations(candidates(last), cites.subspan(num_cited),
                                                                      per_bin[last]);
            }
          } else {
            sampled_sizes[i] += static_cast<int32_t>(n_hop_map[d].size());
            search_.BinNeighborhood(db_, year, n_hop_map[d], bins);
            stage(kBin);
            bin_sizes.assign(search_.num_bins, 0);
            for (int b = 0; b < search_.num_bins; ++b) bin_sizes[b] = static_cast<int>(bins[b].size());
            std::vector<int> per_bin = search_.BinOutdegrees(bin_sizes, quota, binned_recency_probabilities_);
            const int last = search_.num_bins - 1;
            for (int b = 0; b < last; ++b) {
              num_cited += CitationEngine::MakeCitations(bins[b], cites.subspan(num_cited), metrics, weights,
                                                         per_bin[b]);
            }
            num_cited += CitationEngine::MakeUniformRandomCitations(bins[last], cites.subspan(num_cited),
                                                                    per_bin[last]);
          }
        }
      }
      stage(kNeighborhood);

      const int num_random = out_degree - num_cited - 1;
      fully_random[i] = num_random;
      num_cited += CitationEngine::MakeUniformRandomCitationsFromGraph(num_existing, generator_nodes, cites,
                                                                       num_cited, num_random);
      stage(kRandom);

      // Edges: the generator first, then each distinct citation.
      unique_cited.Clear();
      if (unique_cited.Insert(generator)) {
        edges_out.push_back(agent);
        edges_out.push_back(generator);
      }
      for (int j = 0; j < num_cited; ++j) {
        if (unique_cited.Insert(citations[j])) {
          edges_out.push_back(agent);
          edges_out.push_back(citations[j]);
        }
      }
      stage(kRecord);
    }
  }
  logger_.Lap(year, "Simulate agents");
  for (int s = 0; s < kNumStages; ++s) {
    long long total = 0;
    for (const auto &a : thread_stage_ms) total += a[s];
    logger_.Value(year, kStageLabels[s], total);
  }

  // 5. Gather every rank's citations and diagnostics on every rank.
  std::vector<int32_t> local_edges;
  {
    size_t total = 0;
    for (const auto &e : thread_edges) total += e.size();
    local_edges.reserve(total);
    for (auto &e : thread_edges) {
      local_edges.insert(local_edges.end(), e.begin(), e.end());
      std::vector<int32_t>().swap(e);
    }
  }
  std::vector<int32_t> all_edges;
  coordinator_.AllgatherInts(local_edges, all_edges);
  std::vector<int32_t>().swap(local_edges);
  coordinator_.AllreduceMaxInts(sampled_sizes);
  coordinator_.AllreduceMaxInts(fully_random);
  logger_.Lap(year, "Gather new citations");
  logger_.Value(year, "New citations", static_cast<long long>(all_edges.size() / 2));

  // 6. Apply the year to the graph, identically on every rank.
  ApplyYear(batch, all_edges, sampled_sizes, fully_random);
}

void ABM::ApplyYear(const AgentBatch &batch, const std::vector<int32_t> &edges,
                    const std::vector<int32_t> &sampled_sizes, const std::vector<int32_t> &fully_random) {
  const int32_t year = batch.year;
  const size_t count = static_cast<size_t>(batch.count);
  const size_t m = edges.size() / 2;

  // Group by citing agent; each agent's own order is kept.
  std::vector<uint64_t> row_ptr(count + 1, 0);
  for (size_t e = 0; e < m; ++e) {
    row_ptr[static_cast<size_t>(edges[2 * e] - batch.first_id) + 1]++;
  }
  for (size_t i = 0; i < count; ++i) row_ptr[i + 1] += row_ptr[i];
  std::vector<int32_t> targets(m), sources(m);
  {
    std::vector<uint64_t> cursor(row_ptr.begin(), row_ptr.end() - 1);
    for (size_t e = 0; e < m; ++e) {
      size_t i = static_cast<size_t>(edges[2 * e] - batch.first_id);
      size_t pos = cursor[i]++;
      targets[pos] = edges[2 * e + 1];
      sources[pos] = edges[2 * e];
    }
  }
  db_.csr.AppendRows(count, row_ptr, targets);
  logger_.Lap(year, "Append new rows");
  db_.csr.MergeInEdges(sources, targets);
  logger_.Lap(year, "Merge in-edges");

  ladybug::NodeColumns &n = db_.nodes;
  for (size_t i = 0; i < count; ++i) {
    size_t v = static_cast<size_t>(batch.first_id) + i;
    n.sampled_neighborhood_size[v] = std::max(0, sampled_sizes[i]);
    n.fully_random_citations[v] = std::max(0, fully_random[i]);
  }
  db_.RebuildClusterIndex();
  logger_.Lap(year, "Rebuild cluster index");
  db_.authors.ComputeReputations(n.author.span(), db_.csr);
  for (size_t i = 0; i < count; ++i) {
    size_t v = static_cast<size_t>(batch.first_id) + i;
    n.initial_author_reputation[v] = db_.authors.Reputation(n.author[v]);
  }
  logger_.Lap(year, "Author reputations");
  db_.meta.next_year = year + 1;
}

double ABM::EstimatedSnapshotSeconds() const {
  if (last_snapshot_seconds_ >= 0) return last_snapshot_seconds_ * 1.2 + 30;
  double bytes = static_cast<double>(db_.csr.NumEdges()) * 8 + static_cast<double>(db_.NumNodes()) * 80;
  return bytes / 300e6 + 60; // conservative first guess: 300 MB/s
}

double ABM::EstimatedExportSeconds() const {
  double bytes = static_cast<double>(db_.csr.NumEdges()) * 22 + static_cast<double>(db_.NumNodes()) * 160;
  return bytes / 150e6 + 60;
}

void ABM::WriteCheckpoint(const char *reason) {
  auto t0 = std::chrono::steady_clock::now();
  if (coordinator_.IsMaster()) {
    std::string dir = ladybug::WriteSnapshot(db_, config_.checkpoint_dir, config_.checkpoints_to_keep);
    logger_.Write(std::string("Wrote snapshot (") + reason + ") to " + dir + " in " +
                  std::to_string(SecondsSince(t0)) + " s");
  }
  coordinator_.Barrier();
  last_snapshot_seconds_ = SecondsSince(t0);
  last_snapshot_time_ = std::chrono::steady_clock::now();
}

void ABM::Export() {
  if (!coordinator_.IsMaster()) return;
  auto t0 = std::chrono::steady_clock::now();
  for (const std::string &path : {config_.output_file, config_.auxiliary_information_file}) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  }
  ladybug::ExportEdgelist(db_, config_.output_file);
  logger_.Write("wrote edgelist (" + std::to_string(SecondsSince(t0)) + " s)");
  ladybug::ExportNodelist(db_, config_.auxiliary_information_file);
  logger_.Write("wrote nodelist (" + std::to_string(SecondsSince(t0)) + " s)");
}

void ABM::LogMemory(int32_t year) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0) {
      long long kb = std::stoll(line.substr(6));
      logger_.Value(year, line.substr(0, 5) + " (MB)", kb / 1024);
    }
  }
}

int ABM::Run(ResumeMode resume) {
  const bool ok = Initialize(resume);
  if (coordinator_.GlobalAny(!ok)) {
    return kExitError;
  }
  coordinator_.Barrier();

  int years_this_job = 0;
  auto pause = [&](const char *reason, bool write_snapshot) {
    if (write_snapshot) WriteCheckpoint(reason);
    logger_.Write("Pausing at the start of year " + std::to_string(db_.meta.next_year) +
                  "; run again with --resume auto to continue");
    return kExitPaused;
  };

  while (db_.meta.next_year < db_.meta.end_year) {
    const int32_t year = db_.meta.next_year;
    auto t0 = std::chrono::steady_clock::now();
    SimulateYear(year);
    ++years_this_job;
    const double cycle = SecondsSince(t0);
    supervisor_.RecordCycle(cycle);
    logger_.Write("finished year " + std::to_string(year) + " in " + std::to_string(cycle) + " s; " +
                  std::to_string(db_.NumNodes()) + " nodes, " + std::to_string(db_.csr.NumEdges()) + " edges");
    logger_.Value(year, "Cycle seconds", static_cast<long long>(cycle));
    LogMemory(year);

    if (db_.meta.next_year >= db_.meta.end_year) break;
    if (coordinator_.GlobalAny(supervisor_.ShouldStop(EstimatedSnapshotSeconds()))) {
      return pause("pausing before the job ends", true);
    }
    bool periodic = config_.checkpoint_interval_minutes > 0 &&
                    SecondsSince(last_snapshot_time_) >= config_.checkpoint_interval_minutes * 60;
    if (coordinator_.GlobalAny(periodic)) {
      WriteCheckpoint("periodic");
      // The save used time; check again before starting another year.
      if (coordinator_.GlobalAny(supervisor_.ShouldStop(EstimatedSnapshotSeconds()))) {
        return pause("periodic snapshot is the resume point", false);
      }
    }
  }

  // All years simulated. Export now, unless the job would end first (a job
  // that only has the export left always tries it).
  if (years_this_job > 0) {
    bool short_on_time = supervisor_.HasDeadline() &&
                         supervisor_.SecondsRemaining() < EstimatedExportSeconds() + EstimatedSnapshotSeconds();
    if (coordinator_.GlobalAny(short_on_time || SignalSupervisor::SignalCaught())) {
      WriteCheckpoint("export deferred to the next job");
      logger_.Write("All years simulated; not enough time left to export. Run again with --resume auto to export.");
      return kExitPaused;
    }
    if (config_.checkpoint_interval_minutes > 0) {
      WriteCheckpoint("final");
    }
  }
  Export();
  coordinator_.Barrier();
  logger_.Write("finished sim");
  return kExitFinished;
}

#include "neighborhood_search.h"
#include "hash_set.h"
#include "rng.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

using NodeSet = StampedHashSet<int32_t>;

// Adds up to `need` distinct entries of (a ++ b) that are not yet visited,
// uniformly without replacement, and marks them visited. Returns how many
// were added.
//
// Short lists use the original method (collect the unvisited entries, then
// std::sample). Long lists draw random positions and reject repeats and
// visited nodes, which costs O(need) instead of O(list length); this is what
// keeps hub neighbor lists (millions of citers) from dominating run time.
size_t SampleUnvisited(std::span<const int32_t> a, std::span<const int32_t> b, size_t need,
                       NodeSet &visited, std::vector<int> &out, pcg32 &generator) {
  const size_t d = a.size() + b.size();
  if (need == 0 || d == 0) {
    return 0;
  }
  auto at = [&](size_t i) { return i < a.size() ? a[i] : b[i - a.size()]; };
  thread_local std::vector<int> pool;
  thread_local std::vector<int> picked;
  size_t added = 0;

  auto exact = [&](auto keep_position) {
    pool.clear();
    for (size_t i = 0; i < d; ++i) {
      if (!keep_position(i)) continue;
      int x = at(i);
      if (!visited.Contains(x)) pool.push_back(x);
    }
    picked.clear();
    std::sample(pool.begin(), pool.end(), std::back_inserter(picked), need - added, generator);
    for (int x : picked) {
      if (visited.Insert(x)) {
        out.push_back(x);
        ++added;
      }
    }
  };

  // CANTATAS_EXACT_2HOP=1 forces the original method (diagnostics only).
  static const bool force_exact = std::getenv("CANTATAS_EXACT_2HOP") != nullptr;
  if (force_exact || d <= 4 * need + 64) {
    exact([](size_t) { return true; });
    return added;
  }

  thread_local StampedHashSet<uint64_t> drawn;
  drawn.Clear();
  std::uniform_int_distribution<size_t> pick(0, d - 1);
  const size_t max_draws = 8 * need + 256;
  size_t draws = 0;
  while (added < need && draws < max_draws) {
    size_t pos = pick(generator);
    ++draws;
    if (!drawn.Insert(pos)) continue;
    int x = at(pos);
    if (visited.Insert(x)) {
      out.push_back(x);
      ++added;
    }
  }
  if (added < need) {
    // Most of the list was already visited: finish exactly over the rest.
    exact([&](size_t i) { return !drawn.Contains(i); });
  }
  return added;
}

} // namespace

NeighborhoodSearch::NeighborhoodSearch(const std::string &recency_bins, bool use_alpha,
                                       int neighborhood_sample)
    : use_alpha_(use_alpha), neighborhood_sample_(neighborhood_sample) {
  std::stringstream ss(recency_bins);
  std::string value;
  int elements = 0;
  while (std::getline(ss, value, ',')) {
    int bin = std::stoi(value);
    if (bin != -1) {
      bin_boundaries_.push_back(bin);
    }
    ++elements;
  }
  if (bin_boundaries_.empty() || bin_boundaries_[0] != 1) {
    throw std::runtime_error("recency_bins must be non-empty and start with 1");
  }
  num_bins = elements;
  recency_bin_lut_.resize(1024);
  for (int age = 0; age < 1024; ++age) {
    int bin = static_cast<int>(bin_boundaries_.size()) - 1;
    for (size_t i = 0; i + 1 < bin_boundaries_.size(); ++i) {
      if (bin_boundaries_[i] <= age && age < bin_boundaries_[i + 1]) {
        bin = static_cast<int>(i);
        break;
      }
    }
    recency_bin_lut_[age] = bin;
  }
}

int NeighborhoodSearch::GetBinIndex(int year_diff) const {
  if (year_diff >= 0 && static_cast<size_t>(year_diff) < recency_bin_lut_.size()) {
    return recency_bin_lut_[year_diff];
  }
  if (year_diff < 0) {
    return 0;
  }
  for (size_t i = 0; i + 1 < bin_boundaries_.size(); ++i) {
    if (bin_boundaries_[i] <= year_diff && year_diff < bin_boundaries_[i + 1]) {
      return static_cast<int>(i);
    }
  }
  return static_cast<int>(bin_boundaries_.size()) - 1;
}

void NeighborhoodSearch::BinNeighborhood(const ladybug::LadyBugDB &db, int current_year,
                                         std::span<const int> nodes,
                                         std::vector<std::vector<int>> &bins) const {
  bins.resize(num_bins);
  for (auto &bin : bins) bin.clear();
  for (int node : nodes) {
    int bin = GetBinIndex(db, node, current_year);
    if (bin >= 0 && bin < num_bins) {
      bins[bin].push_back(node);
    }
  }
}

void NeighborhoodSearch::GetNeighborhoodMap(const ladybug::LadyBugDB &db, int32_t generator,
                                            std::vector<std::vector<int>> &n_hop_map) const {
  n_hop_map.resize(3);
  for (auto &level : n_hop_map) level.clear();
  thread_local NodeSet visited;
  thread_local std::vector<int> one_hop;
  visited.Clear();
  one_hop.clear();
  const ladybug::BidirectionalCSR &csr = db.csr;
  std::vector<int> &first = n_hop_map[1];
  std::vector<int> &second = use_alpha_ ? n_hop_map[2] : n_hop_map[1];

  visited.Insert(generator);
  for (int32_t x : csr.Out(generator)) {
    if (visited.Insert(x)) one_hop.push_back(x);
  }
  for (int32_t x : csr.In(generator)) {
    if (visited.Insert(x)) one_hop.push_back(x);
  }

  if (neighborhood_sample_ == -1) {
    // Unbounded breadth-first search to distance 2.
    first.insert(first.end(), one_hop.begin(), one_hop.end());
    for (int u : one_hop) {
      for (int32_t x : csr.Out(u)) {
        if (visited.Insert(x)) second.push_back(x);
      }
      for (int32_t x : csr.In(u)) {
        if (visited.Insert(x)) second.push_back(x);
      }
    }
    return;
  }

  const size_t cap = static_cast<size_t>(neighborhood_sample_);
  pcg32 &rng = rng::Thread();
  if (one_hop.size() > cap) {
    thread_local std::vector<int> sampled;
    sampled.clear();
    std::sample(one_hop.begin(), one_hop.end(), std::back_inserter(sampled), cap, rng);
    one_hop.swap(sampled);
  }
  first.insert(first.end(), one_hop.begin(), one_hop.end());
  std::shuffle(one_hop.begin(), one_hop.end(), rng);

  for (int u : one_hop) {
    std::span<const int32_t> out = csr.Out(u);
    std::span<const int32_t> in = csr.In(u);
    size_t degree = out.size() + in.size();
    if (second.size() + degree < cap) {
      for (int32_t x : out) {
        if (visited.Insert(x)) second.push_back(x);
      }
      for (int32_t x : in) {
        if (visited.Insert(x)) second.push_back(x);
      }
    } else {
      size_t need = cap - std::min(cap, second.size());
      SampleUnvisited(out, in, need, visited, second, rng);
      if (second.size() == cap) {
        return;
      }
    }
  }
}

std::vector<int> NeighborhoodSearch::GetNumCitationsPerNeighborhood(double alpha, int total,
                                                                    size_t hop1_size,
                                                                    size_t hop2_size) const {
  std::vector<int> per_neighborhood(3, 0);
  int clamped = std::min(static_cast<int>(hop1_size + hop2_size), total);
  per_neighborhood[1] = std::min(static_cast<int>(clamped * alpha), static_cast<int>(hop1_size));
  per_neighborhood[2] = std::min(clamped - per_neighborhood[1], static_cast<int>(hop2_size));
  return per_neighborhood;
}

std::vector<int> NeighborhoodSearch::BinOutdegrees(
    const std::vector<int> &bin_sizes, int total_outdegree,
    const std::vector<double> &binned_recency_probabilities) const {
  std::vector<int> target(num_bins, 0);
  int remaining = total_outdegree;
  for (int b = 0; b < num_bins; ++b) {
    if (remaining == 0) break;
    double p = (static_cast<size_t>(b) < binned_recency_probabilities.size())
                   ? binned_recency_probabilities[b]
                   : 0.0;
    int quota = static_cast<int>(std::round(total_outdegree * p));
    quota = std::min(quota, remaining);
    target[b] = quota;
    remaining -= quota;
  }
  std::uniform_int_distribution<int> year_distribution{1, 5};
  pcg32 &generator = rng::Thread();
  while (remaining > 0) {
    int bin = GetBinIndex(year_distribution(generator));
    if (static_cast<size_t>(bin) < target.size()) {
      target[bin] += 1;
    }
    remaining--;
  }
  auto size_of = [&](int b) {
    return (static_cast<size_t>(b) < bin_sizes.size()) ? bin_sizes[b] : 0;
  };
  for (int b = 0; b < num_bins; ++b) {
    int uncited = target[b] - size_of(b);
    if (uncited <= 0) continue;
    for (int s = b - 1; s >= 0 && uncited > 0; --s) {
      if (target[s] < size_of(s)) {
        int citable = std::min(uncited, size_of(s) - target[s]);
        uncited -= citable;
        target[s] += citable;
        target[b] -= citable;
      }
    }
    for (int s = b + 1; s < num_bins && uncited > 0; ++s) {
      if (target[s] < size_of(s)) {
        int citable = std::min(uncited, size_of(s) - target[s]);
        uncited -= citable;
        target[s] += citable;
        target[b] -= citable;
      }
    }
  }
  return target;
}

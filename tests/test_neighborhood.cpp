// Bounded 2-hop sampling around a hub: size caps, distinctness, uniformity.
#include "check.h"
#include "ladybug/ladybug_db.h"
#include "neighborhood_search.h"
#include "rng.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

int main() {
  // Node 0: generator; it cites hub 1 and nodes 2..21. Hub 1 is cited by
  // 50,000 papers (22 .. 50,021), so the 2-hop list is mostly the hub's citers.
  const int32_t hub_citers = 50000;
  const int32_t n = 22 + hub_citers;
  std::vector<int32_t> src, dst;
  src.push_back(0);
  dst.push_back(1);
  for (int32_t k = 2; k < 22; ++k) {
    src.push_back(0);
    dst.push_back(k);
  }
  for (int32_t c = 22; c < n; ++c) {
    src.push_back(c);
    dst.push_back(1);
  }
  ladybug::LadyBugDB db;
  db.csr.Build(n, src, dst);
  db.nodes.year.Resize(n, 2000);

  const int cap = 1000;
  NeighborhoodSearch search("1,2,3,4,5,10,20,30", /*use_alpha=*/true, cap);
  std::vector<std::vector<int>> map;
  std::vector<int> hits(n, 0);
  const int trials = 400;
  for (int t = 0; t < trials; ++t) {
    rng::SeedThread(123, rng::Stream::Agent, 0, static_cast<uint64_t>(t));
    search.GetNeighborhoodMap(db, 0, map);
    CHECK_EQ(map[1].size(), 21u);
    CHECK(map[2].size() <= static_cast<size_t>(cap));
    CHECK(map[2].size() >= static_cast<size_t>(cap) - 25);
    std::set<int> seen(map[1].begin(), map[1].end());
    seen.insert(0);
    for (int x : map[2]) {
      CHECK(seen.insert(x).second); // distinct, and not the generator or a 1-hop node
      if (x >= 22) hits[x]++;
    }
  }
  // Each citer should be picked with probability ~ cap / hub_citers.
  double expected = static_cast<double>(trials) * cap / hub_citers;
  double total = 0, sq = 0;
  for (int32_t c = 22; c < n; ++c) {
    total += hits[c];
    sq += (hits[c] - expected) * (hits[c] - expected);
  }
  double mean = total / hub_citers;
  double chi2_per_cell = sq / expected / hub_citers; // ~1 for a uniform sample
  CHECK(std::abs(mean - expected) / expected < 0.1);
  CHECK(chi2_per_cell > 0.8 && chi2_per_cell < 1.2);

  // Unbounded search returns every 2-hop node.
  NeighborhoodSearch full("1,2,3,4,5,10,20,30", true, -1);
  full.GetNeighborhoodMap(db, 0, map);
  CHECK_EQ(map[2].size(), static_cast<size_t>(hub_citers));
  std::cout << "test_neighborhood passed (chi2/cell " << chi2_per_cell << ")" << std::endl;
  return 0;
}

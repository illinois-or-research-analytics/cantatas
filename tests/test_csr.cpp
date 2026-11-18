// BidirectionalCSR against a naive adjacency-list model.
#include "check.h"
#include "ladybug/ladybug_csr.h"

#include <random>
#include <vector>

using ladybug::BidirectionalCSR;

static void Compare(const BidirectionalCSR &csr, const std::vector<std::vector<int>> &out,
                    const std::vector<std::vector<int>> &in) {
  CHECK_EQ(csr.NumNodes(), out.size());
  uint64_t edges = 0;
  for (size_t u = 0; u < out.size(); ++u) {
    auto o = csr.Out(static_cast<int32_t>(u));
    auto i = csr.In(static_cast<int32_t>(u));
    CHECK_EQ(o.size(), out[u].size());
    CHECK_EQ(i.size(), in[u].size());
    for (size_t k = 0; k < o.size(); ++k) CHECK_EQ(o[k], out[u][k]);
    for (size_t k = 0; k < i.size(); ++k) CHECK_EQ(i[k], in[u][k]);
    CHECK_EQ(csr.InDegree(static_cast<int32_t>(u)), in[u].size());
    edges += out[u].size();
  }
  CHECK_EQ(csr.NumEdges(), edges);
}

int main() {
  std::mt19937 gen(7);
  const int n = 300;
  std::uniform_int_distribution<int> node(0, n - 1);
  std::vector<int32_t> src, dst;
  std::vector<std::vector<int>> out(n), in(n);
  for (int e = 0; e < 4000; ++e) {
    int a = node(gen), b = node(gen);
    src.push_back(a);
    dst.push_back(b);
    out[a].push_back(b);
    in[b].push_back(a);
  }
  BidirectionalCSR csr;
  csr.Build(n, src, dst);
  Compare(csr, out, in);

  // Three simulated years: append rows for new nodes, then merge in-edges.
  for (int year = 0; year < 3; ++year) {
    const int first = static_cast<int>(csr.NumNodes());
    const int count = 40;
    std::vector<uint64_t> row_ptr(count + 1, 0);
    std::vector<int32_t> targets, sources;
    out.resize(first + count);
    in.resize(first + count);
    for (int i = 0; i < count; ++i) {
      std::uniform_int_distribution<int> any(0, first + count - 1); // may cite same-year nodes
      int k = 1 + (i % 7);
      for (int j = 0; j < k; ++j) {
        int t = any(gen);
        targets.push_back(t);
        sources.push_back(first + i);
        out[first + i].push_back(t);
        in[t].push_back(first + i);
      }
      row_ptr[i + 1] = targets.size();
    }
    csr.AppendRows(count, row_ptr, targets);
    csr.MergeInEdges(sources, targets);
    Compare(csr, out, in);
  }
  std::cout << "test_csr passed" << std::endl;
  return 0;
}

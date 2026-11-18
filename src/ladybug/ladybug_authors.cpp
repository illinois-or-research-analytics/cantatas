#include "ladybug/ladybug_authors.h"
#include "ladybug/ladybug_column.h"
#include "ladybug/ladybug_parallel.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace ladybug {

int32_t AuthorRegistry::NextAuthor(int current_year, pcg32 &rng) {
  const size_t one_paper = Bucket(1).size();
  bool found_valid_place = false;
  size_t proposed = 2;

  while (true) {
    double denom = (lotka_exponent_ == 2)
                       ? static_cast<double>(proposed) * static_cast<double>(proposed)
                       : std::pow(static_cast<double>(proposed), lotka_exponent_);
    long expected = std::lround(static_cast<double>(one_paper) / denom);
    if (expected < 1) {
      break;
    }
    long actual = static_cast<long>(Bucket(proposed).size());
    if (expected - actual > 1) {
      found_valid_place = true;
      break;
    }
    proposed++;
  }

  if (found_valid_place) {
    std::vector<int32_t> &candidates = Bucket(proposed - 1);
    if (!candidates.empty()) {
      auto active = [&](int32_t author) {
        return current_year - birth_year_[author] < max_lifetime_;
      };
      int32_t chosen = -1;
      size_t chosen_index = 0;
      std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
      // Rejection sampling first; it is uniform over active authors.
      for (int attempt = 0; attempt < 32; ++attempt) {
        size_t idx = pick(rng);
        if (active(candidates[idx])) {
          chosen = candidates[idx];
          chosen_index = idx;
          break;
        }
      }
      if (chosen < 0) {
        std::vector<size_t> valid;
        for (size_t i = 0; i < candidates.size(); ++i) {
          if (active(candidates[i])) {
            valid.push_back(i);
          }
        }
        if (!valid.empty()) {
          std::uniform_int_distribution<size_t> vpick(0, valid.size() - 1);
          chosen_index = valid[vpick(rng)];
          chosen = candidates[chosen_index];
        }
      }
      if (chosen >= 0) {
        candidates[chosen_index] = candidates.back();
        candidates.pop_back();
        Bucket(proposed).push_back(chosen);
        return chosen;
      }
    }
  }

  int32_t author = next_author_id_++;
  Bucket(1).push_back(author);
  birth_year_.push_back(static_cast<int16_t>(current_year));
  return author;
}

void AuthorRegistry::ComputeReputations(std::span<const int32_t> paper_author,
                                        const BidirectionalCSR &csr) {
  const size_t num_authors = static_cast<size_t>(next_author_id_);
  const size_t num_papers = std::min(paper_author.size(), csr.NumNodes());
  reputation_.assign(num_authors, 0);
  if (num_authors == 0 || num_papers == 0) {
    return;
  }

  // Group papers by author (counting sort), then compute each h-index from
  // the in-degrees of that author's papers.
  Column<uint64_t> offsets;
  offsets.Resize(num_authors + 1, 0);
  uint64_t *off = offsets.data();
  bool out_of_range = false;
#pragma omp parallel for schedule(static) reduction(|| : out_of_range)
  for (size_t p = 0; p < num_papers; ++p) {
    int32_t a = paper_author[p];
    if (a >= 0) {
      if (static_cast<size_t>(a) >= num_authors) {
        out_of_range = true;
        continue;
      }
      __atomic_fetch_add(&off[a], 1, __ATOMIC_RELAXED);
    }
  }
  if (out_of_range) {
    throw std::runtime_error("AuthorRegistry: a paper refers to an unknown author ID");
  }
  const uint64_t total = ExclusiveScanInPlace(off, num_authors + 1);

  Column<uint32_t> degrees;
  degrees.ResizeUninitialized(total);
  {
    Column<uint64_t> cursor;
    cursor.ResizeUninitialized(num_authors);
    std::copy(off, off + num_authors, cursor.data());
    for (size_t p = 0; p < num_papers; ++p) {
      int32_t a = paper_author[p];
      if (a >= 0) {
        degrees[cursor[a]++] = csr.InDegree(static_cast<int32_t>(p));
      }
    }
  }

#pragma omp parallel
  {
    std::vector<uint32_t> freq;
#pragma omp for schedule(dynamic, 1024)
    for (size_t a = 0; a < num_authors; ++a) {
      const uint64_t lo = off[a];
      const uint64_t hi = off[a + 1];
      const size_t n = static_cast<size_t>(hi - lo);
      if (n == 0) {
        continue;
      }
      freq.assign(n + 1, 0);
      for (uint64_t i = lo; i < hi; ++i) {
        freq[std::min<size_t>(n, degrees[i])]++;
      }
      size_t h = n;
      uint64_t at_least = freq[n];
      while (h > at_least) {
        --h;
        at_least += freq[h];
      }
      reputation_[a] = static_cast<int32_t>(h);
    }
  }
}

} // namespace ladybug

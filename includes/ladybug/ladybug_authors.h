#ifndef LADYBUG_AUTHORS_H
#define LADYBUG_AUTHORS_H

#include "ladybug/ladybug_csr.h"
#include "pcg_random.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace ladybug {

/*
 * AuthorRegistry: author lifecycle and reputation.
 *
 * - NextAuthor() assigns an author to a new paper following Lotka's law
 *   (a port of the original GetNextAuthor): it either promotes an active
 *   author from the k-paper bucket to k+1, or creates a new author.
 *   Only the rank that creates agents (rank 0) calls it.
 * - ComputeReputations() recomputes every author's h-index from the current
 *   in-degrees; every rank runs it on its identical copy of the graph.
 */
class AuthorRegistry {
public:
  void Configure(int author_max_lifetime, int lotka_exponent = 2) {
    max_lifetime_ = author_max_lifetime;
    lotka_exponent_ = lotka_exponent;
  }

  int32_t NextAuthor(int current_year, pcg32 &rng);

  int32_t NumAuthors() const noexcept { return next_author_id_; }

  // On ranks that do not create agents: registers the authors rank 0 created
  // this year (IDs up to next_id), all born in `year`.
  void SyncNewAuthors(int32_t next_id, int year) {
    if (next_id > next_author_id_) {
      birth_year_.resize(static_cast<size_t>(next_id), static_cast<int16_t>(year));
      next_author_id_ = next_id;
    }
  }

  // h-index of every author from the in-degrees of their papers.
  // paper_author[p] is the author of paper p (-1 for none).
  void ComputeReputations(std::span<const int32_t> paper_author,
                          const BidirectionalCSR &csr);

  int32_t Reputation(int32_t author) const noexcept {
    return (author >= 0 && static_cast<size_t>(author) < reputation_.size())
               ? reputation_[author]
               : 0;
  }

  // Snapshot access.
  std::vector<int16_t> &BirthYears() noexcept { return birth_year_; }
  std::vector<int32_t> &Reputations() noexcept { return reputation_; }
  std::vector<std::vector<int32_t>> &Buckets() noexcept { return buckets_; }
  const std::vector<int16_t> &BirthYears() const noexcept { return birth_year_; }
  const std::vector<int32_t> &Reputations() const noexcept { return reputation_; }
  const std::vector<std::vector<int32_t>> &Buckets() const noexcept { return buckets_; }
  void SetNextAuthorId(int32_t id) noexcept { next_author_id_ = id; }

private:
  std::vector<int32_t> &Bucket(size_t count) {
    if (buckets_.size() <= count) {
      buckets_.resize(count + 1);
    }
    return buckets_[count];
  }

  int max_lifetime_ = 30;
  int lotka_exponent_ = 2;
  int32_t next_author_id_ = 0;
  std::vector<int16_t> birth_year_;                // per author
  std::vector<int32_t> reputation_;                // per author (h-index)
  std::vector<std::vector<int32_t>> buckets_;      // buckets_[k]: authors with k papers
};

} // namespace ladybug

#endif // LADYBUG_AUTHORS_H

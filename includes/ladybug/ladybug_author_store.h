#ifndef LADYBUG_AUTHOR_STORE_H
#define LADYBUG_AUTHOR_STORE_H

#include "ladybug_csr.h"
#include "ladybug_mmap.h"
#include "ladybug_node_store.h"
#include "pcg_random.hpp"
#include "utils.h"
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ladybug {

/*
 * LadyBugAuthorStore: Author lifecycle, Lotka's Law publication distribution,
 * dynamic h-index reputation tracking, and cartel syndicate registry.
 */
class LadyBugAuthorStore {
public:
  LadyBugAuthorStore();
  ~LadyBugAuthorStore();

  void Open(const std::string &db_dir, MmapMode mode, int author_max_lifetime = 30);
  void Close();
  void Sync(bool async = false);

  int GetNextAuthor(int current_year, const std::set<int> &exclusion_set);
  void UpdateAuthorPublication(int author_id, int node_id);
  void UpdateAuthorManual(int author_id);
  void SetAuthorBirthYear(int author_id, int year);
  int GetAuthorBirthYear(int author_id) const;

  void ComputeAuthorReputations(const LadyBugNodeStore &node_store);
  int GetAuthorReputation(int author_id) const;

  // Cartel operations
  void SetCartelID(int author_id, int cartel_id);
  int GetCartelID(int author_id) const;
  std::set<int> GetCartelAuthors(int cartel_id) const;
  const std::set<int> &GetCartelSet() const noexcept { return cartel_set_; }

  const std::vector<int> &GetAuthorPublications(int author_id) const;

  int GetNextAuthorId() const noexcept { return next_author_id_; }
  void SetNextAuthorId(int next_id) noexcept { next_author_id_ = next_id; }

private:
  std::string db_dir_;
  MmapMode mode_;
  int author_max_lifetime_;
  int next_author_id_;
  int lotka_exponent_;

  std::unordered_map<int, int> author_birth_year_map_;
  std::unordered_map<int, std::vector<int>> author_publication_map_;
  std::unordered_map<int, int> author_reputation_map_;
  std::unordered_map<int, std::vector<int>> publication_count_to_author_map_;

  std::unordered_map<int, int> author_cartel_map_;
  std::unordered_map<int, std::set<int>> cartel_author_map_;
  std::set<int> cartel_set_;

  static const std::vector<int> empty_pub_vec_;
};

} // namespace ladybug

#endif // LADYBUG_AUTHOR_STORE_H

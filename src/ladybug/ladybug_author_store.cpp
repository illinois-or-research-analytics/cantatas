#include "ladybug/ladybug_author_store.h"
#include <algorithm>
#include <iostream>

namespace ladybug {

const std::vector<int> LadyBugAuthorStore::empty_pub_vec_ = {};

LadyBugAuthorStore::LadyBugAuthorStore()
    : mode_(MmapMode::ReadWrite), author_max_lifetime_(30),
      next_author_id_(0), lotka_exponent_(2) {}

LadyBugAuthorStore::~LadyBugAuthorStore() {
  Close();
}

void LadyBugAuthorStore::Open(const std::string &db_dir, MmapMode mode, int author_max_lifetime) {
  Close();
  db_dir_ = db_dir;
  mode_ = mode;
  author_max_lifetime_ = author_max_lifetime;
  next_author_id_ = 0;
  lotka_exponent_ = 2;

  author_birth_year_map_.clear();
  author_publication_map_.clear();
  author_reputation_map_.clear();
  publication_count_to_author_map_.clear();
  author_cartel_map_.clear();
  cartel_author_map_.clear();
  cartel_set_.clear();
}

void LadyBugAuthorStore::Close() {
  author_birth_year_map_.clear();
  author_publication_map_.clear();
  author_reputation_map_.clear();
  publication_count_to_author_map_.clear();
  author_cartel_map_.clear();
  cartel_author_map_.clear();
  cartel_set_.clear();
}

void LadyBugAuthorStore::Sync(bool /*async*/) {
  // Sync state if persisted
}

void LadyBugAuthorStore::SetAuthorBirthYear(int author_id, int year) {
  if (author_birth_year_map_.contains(author_id)) {
    author_birth_year_map_[author_id] = std::min(author_birth_year_map_[author_id], year);
  } else {
    author_birth_year_map_[author_id] = year;
  }
}

int LadyBugAuthorStore::GetAuthorBirthYear(int author_id) const {
  auto it = author_birth_year_map_.find(author_id);
  if (it != author_birth_year_map_.end()) {
    return it->second;
  }
  return -1;
}

void LadyBugAuthorStore::UpdateAuthorPublication(int author_id, int node_id) {
  author_publication_map_[author_id].push_back(node_id);
}

void LadyBugAuthorStore::UpdateAuthorManual(int author_id) {
  int num_publications = author_publication_map_.contains(author_id)
                             ? static_cast<int>(author_publication_map_.at(author_id).size())
                             : 0;
  std::erase(publication_count_to_author_map_[num_publications], author_id);
  publication_count_to_author_map_[num_publications + 1].push_back(author_id);
}

int LadyBugAuthorStore::GetNextAuthor(int current_year, const std::set<int> &exclusion_set) {
  int num_authors_with_one_paper = publication_count_to_author_map_[1].size();
  bool found_valid_place = false;
  int proposed_publication_count = 2;
  int return_author = next_author_id_;

  while (std::round(num_authors_with_one_paper /
                    std::pow(proposed_publication_count, lotka_exponent_)) >= 1) {
    int expected_num = std::round(num_authors_with_one_paper /
                                  std::pow(proposed_publication_count, lotka_exponent_));
    int actual_num = publication_count_to_author_map_[proposed_publication_count].size();
    int deficit = expected_num - actual_num;
    if (deficit > 1) {
      found_valid_place = true;
      break;
    }
    proposed_publication_count++;
  }

  if (found_valid_place) {
    std::vector<int> living_authors;
    const auto &cand = publication_count_to_author_map_[proposed_publication_count - 1];
    for (int current_author : cand) {
      int birth_yr = author_birth_year_map_.contains(current_author)
                         ? author_birth_year_map_[current_author]
                         : current_year;
      if (current_year - birth_yr < author_max_lifetime_ &&
          !exclusion_set.contains(current_author)) {
        living_authors.push_back(current_author);
      }
    }

    if (!living_authors.empty()) {
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::ranges::shuffle(living_authors, generator);
      int upgraded_author_id = living_authors.back();
      return_author = upgraded_author_id;
      std::erase(publication_count_to_author_map_[proposed_publication_count - 1], upgraded_author_id);
      publication_count_to_author_map_[proposed_publication_count].push_back(upgraded_author_id);
    } else {
      found_valid_place = false;
    }
  }

  if (!found_valid_place) {
    publication_count_to_author_map_[1].push_back(next_author_id_);
    author_birth_year_map_[next_author_id_] = current_year;
    return_author = next_author_id_;
    next_author_id_++;
  }

  return return_author;
}

void LadyBugAuthorStore::ComputeAuthorReputations(const LadyBugNodeStore &node_store) {
  for (const auto &[author_id, birth_year] : author_birth_year_map_) {
    auto it = author_publication_map_.find(author_id);
    if (it == author_publication_map_.end()) {
      author_reputation_map_[author_id] = 0;
      continue;
    }
    const auto &publications = it->second;
    int h_index = 0;
    if (!publications.empty()) {
      std::unordered_map<int, int> freq_map;
      for (int node : publications) {
        size_t deg = node_store.GetInDegree(node);
        freq_map[std::min(publications.size(), deg)]++;
      }
      h_index = static_cast<int>(publications.size());
      int num_cand = freq_map[h_index];
      while (h_index > num_cand) {
        h_index--;
        num_cand += freq_map[h_index];
      }
    }
    author_reputation_map_[author_id] = h_index;
  }
}

int LadyBugAuthorStore::GetAuthorReputation(int author_id) const {
  auto it = author_reputation_map_.find(author_id);
  if (it != author_reputation_map_.end()) {
    return it->second;
  }
  return 0;
}

void LadyBugAuthorStore::SetCartelID(int author_id, int cartel_id) {
  author_cartel_map_[author_id] = cartel_id;
  cartel_author_map_[cartel_id].insert(author_id);
  cartel_set_.insert(cartel_id);
}

int LadyBugAuthorStore::GetCartelID(int author_id) const {
  auto it = author_cartel_map_.find(author_id);
  if (it != author_cartel_map_.end()) {
    return it->second;
  }
  return -1;
}

std::set<int> LadyBugAuthorStore::GetCartelAuthors(int cartel_id) const {
  auto it = cartel_author_map_.find(cartel_id);
  if (it != cartel_author_map_.end()) {
    return it->second;
  }
  return {};
}

const std::vector<int> &LadyBugAuthorStore::GetAuthorPublications(int author_id) const {
  auto it = author_publication_map_.find(author_id);
  if (it != author_publication_map_.end()) {
    return it->second;
  }
  return empty_pub_vec_;
}

} // namespace ladybug

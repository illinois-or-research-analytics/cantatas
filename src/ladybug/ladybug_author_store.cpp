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
  auto &vec = publication_count_to_author_map_[num_publications];
  auto it = std::find(vec.begin(), vec.end(), author_id);
  if (it != vec.end()) {
    *it = vec.back();
    vec.pop_back();
  }
  publication_count_to_author_map_[num_publications + 1].push_back(author_id);
}

int LadyBugAuthorStore::GetNextAuthor(int current_year, const std::set<int> &exclusion_set) {
  int num_authors_with_one_paper = publication_count_to_author_map_[1].size();
  bool found_valid_place = false;
  int proposed_publication_count = 2;
  int return_author = next_author_id_;

  while (true) {
    double denom = (lotka_exponent_ == 2)
                       ? (static_cast<double>(proposed_publication_count) * proposed_publication_count)
                       : std::pow(proposed_publication_count, lotka_exponent_);
    int expected_num = static_cast<int>(std::round(num_authors_with_one_paper / denom));
    if (expected_num < 1) {
      break;
    }
    int actual_num = publication_count_to_author_map_[proposed_publication_count].size();
    int deficit = expected_num - actual_num;
    if (deficit > 1) {
      found_valid_place = true;
      break;
    }
    proposed_publication_count++;
  }

  if (found_valid_place) {
    auto &cand = publication_count_to_author_map_[proposed_publication_count - 1];
    if (cand.empty()) {
      found_valid_place = false;
    } else {
      int upgraded_author_id = -1;
      size_t picked_idx = 0;
      pcg32 &generator = Utils::GetThreadLocalPRNG();
      std::uniform_int_distribution<size_t> dist(0, cand.size() - 1);

      // Fast rejection sampling: try up to 32 random attempts
      for (int attempt = 0; attempt < 32; ++attempt) {
        size_t idx = dist(generator);
        int a = cand[idx];
        auto it_by = author_birth_year_map_.find(a);
        int birth_yr = (it_by != author_birth_year_map_.end()) ? it_by->second : current_year;
        if (current_year - birth_yr < author_max_lifetime_ &&
            (exclusion_set.empty() || !exclusion_set.contains(a))) {
          upgraded_author_id = a;
          picked_idx = idx;
          break;
        }
      }

      // Fallback scan
      if (upgraded_author_id == -1) {
        std::vector<size_t> valid_indices;
        for (size_t i = 0; i < cand.size(); ++i) {
          int a = cand[i];
          auto it_by = author_birth_year_map_.find(a);
          int birth_yr = (it_by != author_birth_year_map_.end()) ? it_by->second : current_year;
          if (current_year - birth_yr < author_max_lifetime_ &&
              (exclusion_set.empty() || !exclusion_set.contains(a))) {
            valid_indices.push_back(i);
          }
        }
        if (!valid_indices.empty()) {
          std::uniform_int_distribution<size_t> vdist(0, valid_indices.size() - 1);
          picked_idx = valid_indices[vdist(generator)];
          upgraded_author_id = cand[picked_idx];
        }
      }

      if (upgraded_author_id != -1) {
        return_author = upgraded_author_id;
        // O(1) swap-and-pop
        cand[picked_idx] = cand.back();
        cand.pop_back();
        publication_count_to_author_map_[proposed_publication_count].push_back(upgraded_author_id);
      } else {
        found_valid_place = false;
      }
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
  std::vector<int> freq;
  for (const auto &[author_id, birth_year] : author_birth_year_map_) {
    auto it = author_publication_map_.find(author_id);
    if (it == author_publication_map_.end()) {
      author_reputation_map_[author_id] = 0;
      continue;
    }
    const auto &publications = it->second;
    int h_index = 0;
    if (!publications.empty()) {
      size_t n = publications.size();
      if (freq.size() <= n) {
        freq.resize(n + 1, 0);
      } else {
        std::fill(freq.begin(), freq.begin() + n + 1, 0);
      }
      for (int node : publications) {
        size_t deg = node_store.GetInDegree(node);
        freq[std::min(n, deg)]++;
      }
      h_index = static_cast<int>(n);
      int num_cand = freq[h_index];
      while (h_index > num_cand) {
        h_index--;
        num_cand += freq[h_index];
      }
    }
    author_reputation_map_[author_id] = h_index;
  }
}

void LadyBugAuthorStore::ComputeAuthorReputations(const LadyBugBidirectionalCSR &csr) {
  std::vector<int> freq;
  for (const auto &[author_id, birth_year] : author_birth_year_map_) {
    auto it = author_publication_map_.find(author_id);
    if (it == author_publication_map_.end()) {
      author_reputation_map_[author_id] = 0;
      continue;
    }
    const auto &publications = it->second;
    int h_index = 0;
    if (!publications.empty()) {
      size_t n = publications.size();
      if (freq.size() <= n) {
        freq.resize(n + 1, 0);
      } else {
        std::fill(freq.begin(), freq.begin() + n + 1, 0);
      }
      for (int node : publications) {
        size_t deg = csr.GetInDegree(node);
        freq[std::min(n, deg)]++;
      }
      h_index = static_cast<int>(n);
      int num_cand = freq[h_index];
      while (h_index > num_cand) {
        h_index--;
        num_cand += freq[h_index];
      }
    }
    author_reputation_map_[author_id] = h_index;
  }
}

int LadyBugAuthorStore::ComputeSingleAuthorReputation(int author_id, const LadyBugNodeStore &node_store) {
  auto it = author_publication_map_.find(author_id);
  if (it == author_publication_map_.end() || it->second.empty()) {
    author_reputation_map_[author_id] = 0;
    return 0;
  }
  const auto &publications = it->second;
  size_t n = publications.size();
  std::vector<int> freq(n + 1, 0);
  for (int node : publications) {
    size_t deg = node_store.GetInDegree(node);
    freq[std::min(n, deg)]++;
  }
  int h_index = static_cast<int>(n);
  int num_cand = freq[h_index];
  while (h_index > num_cand) {
    h_index--;
    num_cand += freq[h_index];
  }
  author_reputation_map_[author_id] = h_index;
  return h_index;
}

int LadyBugAuthorStore::ComputeSingleAuthorReputation(int author_id, const LadyBugBidirectionalCSR &csr) {
  auto it = author_publication_map_.find(author_id);
  if (it == author_publication_map_.end() || it->second.empty()) {
    author_reputation_map_[author_id] = 0;
    return 0;
  }
  const auto &publications = it->second;
  size_t n = publications.size();
  std::vector<int> freq(n + 1, 0);
  for (int node : publications) {
    size_t deg = csr.GetInDegree(node);
    freq[std::min(n, deg)]++;
  }
  int h_index = static_cast<int>(n);
  int num_cand = freq[h_index];
  while (h_index > num_cand) {
    h_index--;
    num_cand += freq[h_index];
  }
  author_reputation_map_[author_id] = h_index;
  return h_index;
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

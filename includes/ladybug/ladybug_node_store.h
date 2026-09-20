#ifndef LADYBUG_NODE_STORE_H
#define LADYBUG_NODE_STORE_H

#include "ladybug_mmap.h"
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ladybug {

enum class NodeType : uint8_t {
  None = 0,
  Seed = 1,
  Agent = 2
};

/*
 * LadyBugNodeStore: Contiguous Struct-of-Arrays (SoA) columnar table for node attributes.
 * Backed by NVMe memory-mapped files (vmcache), allowing direct SIMD AVX2/AVX-512 operations
 * on contiguous spans with zero RAM bloat.
 */
class LadyBugNodeStore {
public:
  LadyBugNodeStore();
  ~LadyBugNodeStore();

  LadyBugNodeStore(const LadyBugNodeStore &) = delete;
  LadyBugNodeStore &operator=(const LadyBugNodeStore &) = delete;

  LadyBugNodeStore(LadyBugNodeStore &&) noexcept;
  LadyBugNodeStore &operator=(LadyBugNodeStore &&) noexcept;

  void Open(const std::string &db_dir, MmapMode mode, size_t initial_node_capacity = 0);
  void Close();
  void Sync(bool async = false);

  void EnsureNodeCapacity(size_t node_count);
  void SetNodeCount(size_t count);
  size_t GetNodeCount() const noexcept { return node_count_; }
  size_t GetCapacity() const noexcept { return capacity_; }

  // --- COLUMN ACCESSORS (Direct Spans for SIMD/Vectorized Kernels) ---
  std::span<uint16_t> GetYearSpan() noexcept { return year_.span(); }
  std::span<const uint16_t> GetYearSpan() const noexcept { return year_.span(); }

  std::span<uint8_t> GetTypeSpan() noexcept { return type_.span(); }
  std::span<const uint8_t> GetTypeSpan() const noexcept { return type_.span(); }

  std::span<int32_t> GetClusterSpan() noexcept { return cluster_id_.span(); }
  std::span<const int32_t> GetClusterSpan() const noexcept { return cluster_id_.span(); }

  std::span<int32_t> GetAuthorSpan() noexcept { return author_id_.span(); }
  std::span<const int32_t> GetAuthorSpan() const noexcept { return author_id_.span(); }

  std::span<uint32_t> GetInDegreeSpan() noexcept { return in_degree_.span(); }
  std::span<const uint32_t> GetInDegreeSpan() const noexcept { return in_degree_.span(); }

  std::span<uint32_t> GetOutDegreeSpan() noexcept { return out_degree_.span(); }
  std::span<const uint32_t> GetOutDegreeSpan() const noexcept { return out_degree_.span(); }

  std::span<float> GetAlphaSpan() noexcept { return alpha_.span(); }
  std::span<const float> GetAlphaSpan() const noexcept { return alpha_.span(); }

  std::span<float> GetPaWeightSpan() noexcept { return pa_weight_.span(); }
  std::span<const float> GetPaWeightSpan() const noexcept { return pa_weight_.span(); }

  std::span<float> GetFitWeightSpan() noexcept { return fit_weight_.span(); }
  std::span<const float> GetFitWeightSpan() const noexcept { return fit_weight_.span(); }

  std::span<float> GetNumAuthorsWeightSpan() noexcept { return num_authors_weight_.span(); }
  std::span<const float> GetNumAuthorsWeightSpan() const noexcept { return num_authors_weight_.span(); }

  std::span<float> GetAuthorRepWeightSpan() noexcept { return author_rep_weight_.span(); }
  std::span<const float> GetAuthorRepWeightSpan() const noexcept { return author_rep_weight_.span(); }

  std::span<int32_t> GetFitnessLagSpan() noexcept { return fitness_lag_duration_.span(); }
  std::span<const int32_t> GetFitnessLagSpan() const noexcept { return fitness_lag_duration_.span(); }

  std::span<int32_t> GetFitnessPeakValSpan() noexcept { return fitness_peak_value_.span(); }
  std::span<const int32_t> GetFitnessPeakValSpan() const noexcept { return fitness_peak_value_.span(); }

  std::span<int32_t> GetFitnessPeakDurSpan() noexcept { return fitness_peak_duration_.span(); }
  std::span<const int32_t> GetFitnessPeakDurSpan() const noexcept { return fitness_peak_duration_.span(); }

  std::span<int32_t> GetAssignedOutDegreeSpan() noexcept { return assigned_out_degree_.span(); }
  std::span<const int32_t> GetAssignedOutDegreeSpan() const noexcept { return assigned_out_degree_.span(); }

  std::span<int32_t> GetPlantedLineSpan() noexcept { return planted_nodes_line_number_.span(); }
  std::span<const int32_t> GetPlantedLineSpan() const noexcept { return planted_nodes_line_number_.span(); }

  std::span<int32_t> GetSampledNbrSizeSpan() noexcept { return sampled_neighborhood_size_.span(); }
  std::span<const int32_t> GetSampledNbrSizeSpan() const noexcept { return sampled_neighborhood_size_.span(); }

  std::span<int32_t> GetFullyRandomCitationsSpan() noexcept { return fully_random_citations_.span(); }
  std::span<const int32_t> GetFullyRandomCitationsSpan() const noexcept { return fully_random_citations_.span(); }

  std::span<int32_t> GetNumAuthorsSpan() noexcept { return num_authors_.span(); }
  std::span<const int32_t> GetNumAuthorsSpan() const noexcept { return num_authors_.span(); }

  std::span<int32_t> GetInitialAuthorRepSpan() noexcept { return initial_author_reputation_.span(); }
  std::span<const int32_t> GetInitialAuthorRepSpan() const noexcept { return initial_author_reputation_.span(); }

  std::span<int32_t> GetCartelIdSpan() noexcept { return cartel_id_.span(); }
  std::span<const int32_t> GetCartelIdSpan() const noexcept { return cartel_id_.span(); }

  std::span<int32_t> GetGeneratorNodeSpan() noexcept { return generator_node_id_.span(); }
  std::span<const int32_t> GetGeneratorNodeSpan() const noexcept { return generator_node_id_.span(); }

  // --- ELEMENT-WISE GETTERS / SETTERS ---
  inline int GetYear(int node) const noexcept { return year_[node]; }
  inline void SetYear(int node, int val) noexcept { year_[node] = static_cast<uint16_t>(val); }

  inline NodeType GetType(int node) const noexcept { return static_cast<NodeType>(type_[node]); }
  inline void SetType(int node, NodeType val) noexcept { type_[node] = static_cast<uint8_t>(val); }

  inline int GetClusterId(int node) const noexcept { return cluster_id_[node]; }
  inline void SetClusterId(int node, int val) noexcept { cluster_id_[node] = val; }

  inline int GetAuthorId(int node) const noexcept { return author_id_[node]; }
  inline void SetAuthorId(int node, int val) noexcept { author_id_[node] = val; }

  inline uint32_t GetInDegree(int node) const noexcept { return in_degree_[node]; }
  inline void SetInDegree(int node, uint32_t val) noexcept { in_degree_[node] = val; }
  inline void IncrementInDegree(int node) noexcept { in_degree_[node]++; }

  inline uint32_t GetOutDegree(int node) const noexcept { return out_degree_[node]; }
  inline void SetOutDegree(int node, uint32_t val) noexcept { out_degree_[node] = val; }
  inline void IncrementOutDegree(int node) noexcept { out_degree_[node]++; }

  inline double GetAlpha(int node) const noexcept { return alpha_[node]; }
  inline void SetAlpha(int node, double val) noexcept { alpha_[node] = static_cast<float>(val); }

  inline double GetPaWeight(int node) const noexcept { return pa_weight_[node]; }
  inline void SetPaWeight(int node, double val) noexcept { pa_weight_[node] = static_cast<float>(val); }

  inline double GetFitWeight(int node) const noexcept { return fit_weight_[node]; }
  inline void SetFitWeight(int node, double val) noexcept { fit_weight_[node] = static_cast<float>(val); }

  inline double GetNumAuthorsWeight(int node) const noexcept { return num_authors_weight_[node]; }
  inline void SetNumAuthorsWeight(int node, double val) noexcept { num_authors_weight_[node] = static_cast<float>(val); }

  inline double GetAuthorRepWeight(int node) const noexcept { return author_rep_weight_[node]; }
  inline void SetAuthorRepWeight(int node, double val) noexcept { author_rep_weight_[node] = static_cast<float>(val); }

  inline int GetFitnessLagDuration(int node) const noexcept { return fitness_lag_duration_[node]; }
  inline void SetFitnessLagDuration(int node, int val) noexcept { fitness_lag_duration_[node] = val; }

  inline int GetFitnessPeakValue(int node) const noexcept { return fitness_peak_value_[node]; }
  inline void SetFitnessPeakValue(int node, int val) noexcept { fitness_peak_value_[node] = val; }

  inline int GetFitnessPeakDuration(int node) const noexcept { return fitness_peak_duration_[node]; }
  inline void SetFitnessPeakDuration(int node, int val) noexcept { fitness_peak_duration_[node] = val; }

  inline int GetAssignedOutDegree(int node) const noexcept { return assigned_out_degree_[node]; }
  inline void SetAssignedOutDegree(int node, int val) noexcept { assigned_out_degree_[node] = val; }

  inline int GetPlantedLineNumber(int node) const noexcept { return planted_nodes_line_number_[node]; }
  inline void SetPlantedLineNumber(int node, int val) noexcept { planted_nodes_line_number_[node] = val; }

  inline int GetSampledNeighborhoodSize(int node) const noexcept { return sampled_neighborhood_size_[node]; }
  inline void SetSampledNeighborhoodSize(int node, int val) noexcept { sampled_neighborhood_size_[node] = val; }

  inline int GetFullyRandomCitations(int node) const noexcept { return fully_random_citations_[node]; }
  inline void SetFullyRandomCitations(int node, int val) noexcept { fully_random_citations_[node] = val; }

  inline int GetNumAuthors(int node) const noexcept { return num_authors_[node]; }
  inline void SetNumAuthors(int node, int val) noexcept { num_authors_[node] = val; }

  inline int GetInitialAuthorReputation(int node) const noexcept { return initial_author_reputation_[node]; }
  inline void SetInitialAuthorReputation(int node, int val) noexcept { initial_author_reputation_[node] = val; }

  inline int GetCartelId(int node) const noexcept { return cartel_id_[node]; }
  inline void SetCartelId(int node, int val) noexcept { cartel_id_[node] = val; }

  inline int GetGeneratorNode(int node) const noexcept { return generator_node_id_[node]; }
  inline void SetGeneratorNode(int node, int val) noexcept { generator_node_id_[node] = val; }

  // Ingest from seed nodelist CSV
  void IngestCSV(const std::string &nodelist_csv, bool is_checkpoint = false);

  // Export to CSV
  void ExportCSV(const std::string &output_csv) const;

private:
  std::string db_dir_;
  size_t node_count_;
  size_t capacity_;
  MmapMode mode_;

  // Columnar buffers (SoA)
  MmapBuffer<uint16_t> year_;
  MmapBuffer<uint8_t> type_;
  MmapBuffer<int32_t> cluster_id_;
  MmapBuffer<int32_t> author_id_;
  MmapBuffer<uint32_t> in_degree_;
  MmapBuffer<uint32_t> out_degree_;
  MmapBuffer<float> alpha_;
  MmapBuffer<float> pa_weight_;
  MmapBuffer<float> fit_weight_;
  MmapBuffer<float> num_authors_weight_;
  MmapBuffer<float> author_rep_weight_;
  MmapBuffer<int32_t> fitness_lag_duration_;
  MmapBuffer<int32_t> fitness_peak_value_;
  MmapBuffer<int32_t> fitness_peak_duration_;
  MmapBuffer<int32_t> assigned_out_degree_;
  MmapBuffer<int32_t> planted_nodes_line_number_;
  MmapBuffer<int32_t> sampled_neighborhood_size_;
  MmapBuffer<int32_t> fully_random_citations_;
  MmapBuffer<int32_t> num_authors_;
  MmapBuffer<int32_t> initial_author_reputation_;
  MmapBuffer<int32_t> cartel_id_;
  MmapBuffer<int32_t> generator_node_id_;

  void InitializeColumns(size_t capacity);
};

} // namespace ladybug

#endif // LADYBUG_NODE_STORE_H

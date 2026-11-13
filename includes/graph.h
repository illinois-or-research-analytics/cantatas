#ifndef GRAPH_H
#define GRAPH_H

#include "ladybug/ladybug_db.h"
#include "pcg_random.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Graph {
public:
  enum class NodeType { None, Seed, Agent };
  struct NodeAttributes {
    bool is_valid = false;
    int year = -1;
    double alpha = -1.0;
    double pa_weight = -1.0;
    double fit_weight = -1.0;
    double num_authors_weight = -1.0;
    double author_reputation_weight = -1.0;
    int fitness_lag_duration = -1;
    int fitness_peak_value = -1;
    int fitness_peak_duration = -1;
    int assigned_out_degree = -1;
    int planted_nodes_line_number = -1;
    int sampled_neighborhood_size = -1;
    std::string generator_node_string = "no_generators";
    int fully_random_citations = -1;
    int author_id = -1;
    int initial_author_reputation = -1;
    int num_authors = -1;
    int cartel_id = -1;
    NodeType type = NodeType::None;
    int in_degree = 0;
    int out_degree = 0;
    int cluster_id = -1;
  };
  /*
  Input: std::string edgelist, std::string nodelist, bool start_from_checkpoint,
  std::string num_authors_bag, int author_max_lifetime Output: Graph object
  Description: Initializes the in-memory graph representation by parsing
  edgelist and nodelist files, setting up historical authorship data and
  structural topology.
  */
  Graph(std::string edgelist, std::string nodelist, bool start_from_checkpoint,
        std::string num_authors_bag, int author_max_lifetime);
  /*
  Input: std::pair<int, int> edge
  Output: void
  Description: Adds a directed edge between two nodes, updating the forward and
  backward adjacency maps as well as in/out degrees.
  */
  void AddEdge(std::pair<int, int> edge);

  /*
  Input: std::string filepath
  Output: char
  Description: Automatically sniffs the column delimiter (comma, tab, space) of
  a given CSV/TSV file by reading its first line.
  */
  static inline char GetDelimiter(std::string filepath) {
    std::ifstream edgelist(filepath);
    std::string line;
    getline(edgelist, line);
    if (line.find(',') != std::string::npos) {
      return ',';
    } else if (line.find('\t') != std::string::npos) {
      return '\t';
    } else if (line.find(' ') != std::string::npos) {
      return ' ';
    }
    throw std::invalid_argument("Could not detect filetype for " + filepath);
  }

  static inline std::
      unordered_map<std::string, int>
      /*
      Input: char delimiter, std::string filepath
      Output: std::unordered_map<std::string, int>
      Description: Parses the header row of a dataset and maps column string
      names to their respective integer indices for robust parsing.
      */
      GetHeaderToIndexMap(char delimiter, std::string filepath) {
    std::unordered_map<std::string, int> header_to_index_map;
    std::ifstream input_nodelist(filepath);
    std::string line;
    std::getline(input_nodelist, line);
    std::stringstream ss(line);
    std::string current_value;
    int index = 0;
    while (std::getline(ss, current_value, delimiter)) {
      header_to_index_map[current_value] = index;
      index++;
    }
    return header_to_index_map;
  }

  // --- LADYBUG DB INTEGRATION & OUT-OF-CORE DELEGATION ---
  void AttachLadyBugDB(ladybug::LadyBugDB *db) noexcept { db_ = db; }
  bool HasLadyBugDB() const noexcept { return db_ != nullptr; }
  ladybug::LadyBugDB *GetLadyBugDB() noexcept { return db_; }
  const ladybug::LadyBugDB *GetLadyBugDB() const noexcept { return db_; }
  void ReleaseHeavyStructures();

  inline size_t GetNodeCount() const noexcept {
    if (db_) {
      return db_->Nodes().GetNodeCount();
    }
    return node_set.size();
  }

  inline bool HasNode(int node) const noexcept {
    if (node < 0) return false;
    if (db_) {
      return static_cast<size_t>(node) < db_->Nodes().GetNodeCount();
    }
    return IsNodeValid(node);
  }

  inline std::span<const int32_t> GetOutNeighbors(int u) const noexcept {
    if (db_) {
      return db_->CSR().GetOutNeighbors(u);
    }
    if (u >= 0 && static_cast<size_t>(u) < forward_adj_list.size()) {
      return std::span<const int32_t>(
          reinterpret_cast<const int32_t *>(forward_adj_list[u].data()),
          forward_adj_list[u].size());
    }
    return {};
  }

  inline std::span<const int32_t> GetInNeighbors(int v) const noexcept {
    if (db_) {
      return db_->CSR().GetInNeighbors(v);
    }
    if (v >= 0 && static_cast<size_t>(v) < backward_adj_list.size()) {
      return std::span<const int32_t>(
          reinterpret_cast<const int32_t *>(backward_adj_list[v].data()),
          backward_adj_list[v].size());
    }
    return {};
  }

  inline const std::set<int> &GetNodeSet() const { return node_set; }
  inline const std::vector<std::vector<int>> &GetForwardAdjList() const { return forward_adj_list; }
  inline const std::vector<std::vector<int>> &GetBackwardAdjList() const { return backward_adj_list; }
  /*
  Input: None
  Output: void
  Description: Caches the initial historical reputation score of all existing
  authors at the beginning of the simulation.
  */
  void SaveInitialAuthorReputations();
  /*
  Input: None
  Output: void
  Description: Consumes the empirical nodelist file, seeding the graph with its
  base nodes and their initial attributes (year, author, etc.).
  */
  void ParseNodelist();
  /*
  Input: None
  Output: void
  Description: Consumes the empirical edgelist file to construct the initial
  historical citation topology.
  */
  void ParseEdgelist();
  /*
  Input: None
  Output: void
  Description: Aggregates the in-degrees of all papers published by each author
  to compute a real-time reputation score.
  */
  void ComputeAuthorReputations();
  int ComputeSingleAuthorReputation(int author_id);
  inline bool IsNodeValid(int node) const {
    if (node < 0) return false;
    if (db_) {
      return static_cast<size_t>(node) < db_->Nodes().GetNodeCount();
    }
    return static_cast<size_t>(node) < node_attributes.size() && node_attributes[node].is_valid;
  }
  int GetNextAuthorId() const noexcept;
  const std::unordered_map<int, int> &GetAuthorBirthYearMap() const noexcept {
    return author_birth_year_map;
  }
  const std::unordered_map<int, std::vector<int>> &GetAuthorPublicationMap() const noexcept {
    return author_publication_map;
  }
  const std::unordered_map<int, std::vector<int>> &GetPublicationCountToAuthorMap() const noexcept {
    return publication_count_to_author_map;
  }
  /*
  Input: int node
  Output: int
  Description: Looks up the author(s) of a given node and returns their current
  aggregated reputation score.
  */
  int GetAuthorReputationForNode(int node) const;
  inline void SetAuthorBirthYear(int author_id, int year) {
    if (db_) {
      db_->Authors().SetAuthorBirthYear(author_id, year);
      return;
    }
    author_birth_year_map[author_id] = year;
  }
  /*
  Input: int u
  Output: void
  Description: Injects a new empty node into the graph, initializing its
  attribute struct.
  */

  // --- DIRECT ATTRIBUTE ACCESSORS ---
  inline NodeAttributes &GetNodeAttr(int node) { return node_attributes[node]; }
  inline const NodeAttributes &GetNodeAttr(int node) const {
    return node_attributes[node];
  }

  inline int GetYear(int node) const {
    if (db_) return db_->Nodes().GetYear(node);
    return node_attributes[node].year;
  }
  inline void SetYear(int node, int val) {
    if (db_) {
      db_->Nodes().SetYear(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].year = val;
  }

  inline double GetAlpha(int node) const {
    if (db_) return db_->Nodes().GetAlpha(node);
    return node_attributes[node].alpha;
  }
  inline void SetAlpha(int node, double val) {
    if (db_) {
      db_->Nodes().SetAlpha(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].alpha = val;
  }

  inline double GetPaWeight(int node) const {
    if (db_) return db_->Nodes().GetPaWeight(node);
    return node_attributes[node].pa_weight;
  }
  inline void SetPaWeight(int node, double val) {
    if (db_) {
      db_->Nodes().SetPaWeight(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].pa_weight = val;
  }

  inline double GetFitWeight(int node) const {
    if (db_) return db_->Nodes().GetFitWeight(node);
    return node_attributes[node].fit_weight;
  }
  inline void SetFitWeight(int node, double val) {
    if (db_) {
      db_->Nodes().SetFitWeight(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].fit_weight = val;
  }

  inline double GetNumAuthorsWeight(int node) const {
    if (db_) return db_->Nodes().GetNumAuthorsWeight(node);
    return node_attributes[node].num_authors_weight;
  }
  inline void SetNumAuthorsWeight(int node, double val) {
    if (db_) {
      db_->Nodes().SetNumAuthorsWeight(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].num_authors_weight = val;
  }

  inline double GetAuthorReputationWeight(int node) const {
    if (db_) return db_->Nodes().GetAuthorRepWeight(node);
    return node_attributes[node].author_reputation_weight;
  }
  inline void SetAuthorReputationWeight(int node, double val) {
    if (db_) {
      db_->Nodes().SetAuthorRepWeight(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].author_reputation_weight = val;
  }

  inline int GetFitnessLagDuration(int node) const {
    if (db_) return db_->Nodes().GetFitnessLagDuration(node);
    return node_attributes[node].fitness_lag_duration;
  }
  inline void SetFitnessLagDuration(int node, int val) {
    if (db_) {
      db_->Nodes().SetFitnessLagDuration(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].fitness_lag_duration = val;
  }

  inline int GetFitnessPeakValue(int node) const {
    if (db_) return db_->Nodes().GetFitnessPeakValue(node);
    return node_attributes[node].fitness_peak_value;
  }
  inline void SetFitnessPeakValue(int node, int val) {
    if (db_) {
      db_->Nodes().SetFitnessPeakValue(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].fitness_peak_value = val;
  }

  inline int GetFitnessPeakDuration(int node) const {
    if (db_) return db_->Nodes().GetFitnessPeakDuration(node);
    return node_attributes[node].fitness_peak_duration;
  }
  inline void SetFitnessPeakDuration(int node, int val) {
    if (db_) {
      db_->Nodes().SetFitnessPeakDuration(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].fitness_peak_duration = val;
  }

  inline int GetAssignedOutDegree(int node) const {
    if (db_) return db_->Nodes().GetAssignedOutDegree(node);
    return node_attributes[node].assigned_out_degree;
  }
  inline void SetAssignedOutDegree(int node, int val) {
    if (db_) {
      db_->Nodes().SetAssignedOutDegree(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].assigned_out_degree = val;
  }

  inline int GetPlantedNodesLineNumber(int node) const {
    if (db_) return db_->Nodes().GetPlantedLineNumber(node);
    return node_attributes[node].planted_nodes_line_number;
  }
  inline void SetPlantedNodesLineNumber(int node, int val) {
    if (db_) {
      db_->Nodes().SetPlantedLineNumber(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].planted_nodes_line_number = val;
  }
  inline bool HasPlantedNodesLineNumber(int node) const {
    if (db_) return db_->Nodes().GetPlantedLineNumber(node) != -1;
    return node_attributes[node].planted_nodes_line_number != -1;
  }

  inline int GetSampledNeighborhoodSize(int node) const {
    if (db_) return db_->Nodes().GetSampledNeighborhoodSize(node);
    return node_attributes[node].sampled_neighborhood_size;
  }
  inline void SetSampledNeighborhoodSize(int node, int val) {
    if (db_) {
      db_->Nodes().SetSampledNeighborhoodSize(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].sampled_neighborhood_size = val;
  }

  const std::string &GetGeneratorNodeString(int node) const;
  void SetGeneratorNodeString(int node, const std::string &val);

  inline int GetCommunityAssignment(int node) const {
    if (db_) return db_->Nodes().GetClusterId(node);
    return node_attributes[node].cluster_id;
  }
  inline void SetCommunityAssignment(int node, int val) {
    if (db_) {
      db_->Nodes().SetClusterId(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].cluster_id = val;
  }

  inline int GetFullyRandomCitations(int node) const {
    if (db_) return db_->Nodes().GetFullyRandomCitations(node);
    return node_attributes[node].fully_random_citations;
  }
  inline void SetFullyRandomCitations(int node, int val) {
    if (db_) {
      db_->Nodes().SetFullyRandomCitations(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].fully_random_citations = val;
  }

  inline int GetAuthorId(int node) const {
    if (db_) return db_->Nodes().GetAuthorId(node);
    return node_attributes[node].author_id;
  }
  inline void SetAuthorId(int node, int val) {
    if (db_) {
      db_->Nodes().SetAuthorId(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].author_id = val;
  }

  inline int GetInitialAuthorReputation(int node) const {
    if (db_) return db_->Nodes().GetInitialAuthorReputation(node);
    return node_attributes[node].initial_author_reputation;
  }
  inline void SetInitialAuthorReputation(int node, int val) {
    if (db_) {
      db_->Nodes().SetInitialAuthorReputation(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].initial_author_reputation = val;
  }

  inline int GetNumAuthors(int node) const {
    if (db_) return db_->Nodes().GetNumAuthors(node);
    return node_attributes[node].num_authors;
  }
  inline void SetNumAuthors(int node, int val) {
    if (db_) {
      db_->Nodes().SetNumAuthors(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].num_authors = val;
  }

  inline int GetCartelIdAttr(int node) const {
    if (db_) return db_->Nodes().GetCartelId(node);
    return node_attributes[node].cartel_id;
  }
  inline void SetCartelIdAttr(int node, int val) {
    if (db_) {
      db_->Nodes().SetCartelId(node, val);
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].cartel_id = val;
  }

  inline NodeType GetType(int node) const {
    if (db_) return static_cast<NodeType>(db_->Nodes().GetType(node));
    return node_attributes[node].type;
  }
  inline void SetType(int node, NodeType val) {
    if (db_) {
      db_->Nodes().SetType(node, static_cast<ladybug::NodeType>(val));
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].type = val;
  }

  inline int GetInDegree(int node) const {
    if (db_) return static_cast<int>(db_->CSR().GetInDegree(node));
    return node_attributes[node].in_degree;
  }
  inline void SetInDegree(int node, int val) {
    if (db_) {
      db_->Nodes().SetInDegree(node, static_cast<uint32_t>(val));
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].in_degree = val;
  }

  inline int GetOutDegree(int node) const {
    if (db_) return static_cast<int>(db_->CSR().GetOutDegree(node));
    return node_attributes[node].out_degree;
  }
  inline void SetOutDegree(int node, int val) {
    if (db_) {
      db_->Nodes().SetOutDegree(node, static_cast<uint32_t>(val));
      return;
    }
    EnsureNodeCapacity(node);
    node_attributes[node].out_degree = val;
  }
  // --- END DIRECT ATTRIBUTE ACCESSORS ---

  void AddNode(int u);
  /*
  Input: None
  Output: void
  Description: Dumps the entire structural topology of the graph to standard
  output for debugging purposes.
  */
  void PrintGraph() const;
  /*
  Input: std::string output_file
  Output: void
  Description: Flushes the complete generated citation topology (edgelist) to a
  specified file on disk.
  */
  void WriteGraph(std::string output_file) const;
  /*
  Input: std::string auxiliary_information_file
  Output: void
  Description: Flushes all accumulated node metadata (attributes) into a CSV
  file for subsequent analysis.
  */
  void WriteAttributes(std::string auxiliary_information_file) const;
  /*
  Input: int current_year, const std::set<int> &exclusion_set
  Output: int
  Description: Generates or selects an author ID for a newly simulated paper,
  modeling birth/death rates and Lotka's law, avoiding any excluded IDs.
  */
  int GetNextAuthor(int current_year, const std::set<int> &exclusion_set);
  /*
  Input: None
  Output: int
  Description: Randomly samples from the historical empirical distribution to
  determine how many authors a new paper should have.
  */
  int GetNextNumAuthors();
  /*
  Input: None
  Output: void
  Description: Pre-loads the bag of empirical author counts from disk into
  memory for fast O(1) sampling.
  */
  void ReadNumAuthorsBag();
  /*
  Input: int author
  Output: int (0 or 1)
  Description: Evaluates whether a specific author ID is considered to have
  received funding based on graph parameters.
  */
  int IsAuthorFunded(int author) const;
  /*
  Input: int author, int node
  Output: void
  Description: Registers a new publication (node) against an author's historical
  record, updating internal maps.
  */
  void UpdateAuthorPublicationMap(int author, int node);
  /*
  Input: int author, int cartel_id
  Output: void
  Description: Marks an author as belonging to a specific cartel syndicate.
  */
  void SetCartelID(int author, int cartel_id);
  /*
  Input: int author
  Output: int
  Description: Retrieves the cartel identifier for an author, returning -1 if
  they are not in a cartel.
  */
  int GetCartelID(int author) const;
  /*
  Input: int cartel_id
  Output: std::set<int>
  Description: Returns all author IDs that are members of the given cartel.
  */
  std::set<int> GetCartelAuthors(int cartel_id) const;
  /*
  Input: int author_id
  Output: std::vector<int>
  Description: Retrieves the list of all node IDs (papers) authored by a
  specific individual.
  */
  std::vector<int> GetAuthorPublications(int author_id) const;
  /*
  Input: int author_id
  Output: void
  Description: Manually injects an author into the tracking maps, primarily used
  during cartel injection or specialized setups.
  */
  void UpdateAuthorManual(int author_id);
  /*
  Input: None
  Output: std::set<int>
  Description: Returns the IDs of all authors currently participating in any
  cartel.
  */
  std::set<int> GetCartelSet() const;

  /*
  Input: int node, int cluster_id
  Output: void
  Description: Appends a node to the internal tracking list for a specific
  cluster assignment.
  */
  void AddNodeToCluster(int node, int cluster_id);
  /*
  Input: int cluster_id
  Output: const std::vector<int>&
  Description: Returns all node IDs that belong to the given cluster.
  */
  const std::vector<int> &GetClusterNodes(int cluster_id) const;
  /*
  Input: int cluster_id
  Output: int
  Description: Retrieves the number of nodes currently assigned to the given
  cluster.
  */
  int GetClusterSize(int cluster_id, int current_year = -1) const;

private:
  ladybug::LadyBugDB *db_ = nullptr;
  std::set<int> node_set;
  std::string edgelist;
  std::string nodelist;
  std::set<int> cartel_set;
  bool start_from_checkpoint;
  std::string num_authors_bag;
  int author_max_lifetime;
  int IsNextAuthorFunded();
  std::vector<int> num_authors_bag_vec;

protected:
  std::unordered_map<int, std::vector<int>> publication_count_to_author_map;
  int next_author_id = 0;
  int lotka_exponent = 2;
  std::vector<std::vector<int>> forward_adj_list;
  std::vector<std::vector<int>> backward_adj_list;
  std::unordered_map<int, int> author_birth_year_map;
  std::unordered_map<int, std::vector<int>> author_publication_map;
  std::unordered_map<int, int> author_reputation_map;
  std::vector<NodeAttributes> node_attributes;
  inline void EnsureNodeCapacity(int node) {
    if (db_) {
      db_->Nodes().EnsureNodeCapacity(node + 1);
      return;
    }
    if (static_cast<size_t>(node) >= node_attributes.size()) {
      size_t new_size =
          std::max((size_t)node + 1, node_attributes.size() * 2 + 1);
      node_attributes.resize(new_size);
      forward_adj_list.resize(new_size);
      backward_adj_list.resize(new_size);
    }
  }
  std::unordered_map<int, int> author_cartel_map;
  std::unordered_map<int, std::set<int>> cartel_author_map;
  std::unordered_map<int, std::vector<int>> cluster_nodes_map;
};

#endif

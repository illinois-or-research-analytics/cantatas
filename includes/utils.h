#ifndef UTILS_H
#define UTILS_H

#include "pcg_random.hpp"
#include "structs.h"
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class Log { error = -1, debug = 0, info = 1 };
#include <chrono>

class Utils {
public:
  static inline pcg32 &GetThreadLocalPRNG() {
    thread_local pcg_extras::seed_seq_from<std::random_device> rand_dev;
    thread_local pcg32 generator(rand_dev);
    return generator;
  }
  /*
  Input: std::string filepath
  Output: char
  Description: Detects the CSV/TSV delimiter (comma, tab, space) by reading the
  first line of the file.
  */
  static inline char GetDelimiter(std::string_view filepath) {
    std::ifstream edgelist{std::string(filepath)};
    std::string line;
    getline(edgelist, line);
    if (line.find(',') != std::string::npos) {
      return ',';
    } else if (line.find('\t') != std::string::npos) {
      return '\t';
    } else if (line.find(' ') != std::string::npos) {
      return ' ';
    }
    throw std::invalid_argument("Could not detect filetype for " +
                                std::string(filepath));
  }

  static inline std::
      unordered_map<int, std::string>
      /*
      Input: char delimiter, std::string filepath
      Output: std::unordered_map<int, std::string>
      Description: Parses the header of a file and maps the integer column index
      to the string column name.
      */
      GetIndexToHeaderMap(char delimiter, std::string_view filepath) {
    std::unordered_map<int, std::string> index_to_header_map;
    std::ifstream input_nodelist{std::string(filepath)};
    std::string line;
    std::getline(input_nodelist, line);
    std::stringstream ss(line);
    std::string current_value;
    int index = 0;
    while (std::getline(ss, current_value, delimiter)) {
      index_to_header_map[index] = current_value;
      index++;
    }
    return index_to_header_map;
  }

  static inline std::
      unordered_map<std::string, int>
      /*
      Input: char delimiter, std::string filepath
      Output: std::unordered_map<std::string, int>
      Description: Parses the header of a file and maps the string column name
      to its integer column index.
      */
      GetHeaderToIndexMap(char delimiter, std::string_view filepath) {
    std::unordered_map<std::string, int> header_to_index_map;
    std::ifstream input_nodelist{std::string(filepath)};
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

  /*
  Input: std::string filepath
  Output: ClonalCartelAgent
  Description: Parses a JSON or structured configuration file describing the
  intrinsic properties of a planted cartel agent.
  */
  static ClonalCartelAgent
  ParseClonalCartelAgentStruct(std::string_view filepath);
  /*
  Input: std::string filepath
  Output: std::vector<int>
  Description: Parses historical empirical out-degree frequencies into a
  flattened vector for constant-time uniform sampling.
  */
  static std::vector<int> ParseOutDegreeBag(std::string_view filepath);
  static std::unordered_map<
      int,
      std::unordered_map<int, std::unordered_map<std::string, std::string>>>
  ParsePlantedNodes(std::string_view filepath);
  static std::unordered_map<int, int>
  ParseRecencyProbabilities(std::string_view filepath);
};

class SimLogger {
public:
  int log_level;
  std::ofstream log_file_handle;
  std::ofstream timing_file_handle;
  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> prev_time;
  std::chrono::time_point<std::chrono::steady_clock> global_prev_time;
  int num_calls_to_log_write;
  std::unordered_map<int, std::vector<std::pair<std::string, int>>> timing_map;

  /*
  Input: None
  Output: SimLogger object
  Description: Initializes an empty logger object meant to be configured via
  Init() before use.
  */
  SimLogger() : log_level(0), num_calls_to_log_write(0) {}

  /*
  Input: std::string log_file, std::string timing_file, int log_level
  Output: void
  Description: Opens the necessary file streams and establishes the verbosity
  baseline for the simulation logger.
  */
  void Init(std::string log_file, std::string timing_file, int log_level);

  /*
  Input: std::string message, Log message_type
  Output: int (status code)
  Description: Writes a formatted message string to the central log file if it
  meets the minimum configured verbosity level.
  */
  int WriteToLogFile(std::string message, Log message_type);
  /*
  Input: int current_year, std::string label, (optional) int time_elapsed
  Output: void
  Description: Records execution duration for a specific high-level step during
  a specific simulation year.
  */
  void LogTime(int current_year, std::string label);
  void LogTime(int current_year, std::string label, int time_elapsed);
  /*
  Input: local parallel stage vector, previous time point, label
  Output: time_point (the new updated time point)
  Description: Records execution times inside fine-grained OpenMP parallel
  blocks without locking the global logger state.
  */
  std::chrono::time_point<std::chrono::steady_clock> LocalLogTime(
      std::vector<std::pair<std::string, int>> &local_parallel_stage_time_vec,
      std::chrono::time_point<std::chrono::steady_clock> local_prev_time,
      std::string label);
  /*
  Input: int start_year, int end_year
  Output: void
  Description: Aggregates all recorded timing checkpoints and flushes them to
  the designated CSV performance log.
  */
  void WriteTimingFile(int start_year, int end_year);
};

#endif

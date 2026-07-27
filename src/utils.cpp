#include "utils.h"

ClonalCartelAgent
Utils::ParseClonalCartelAgentStruct(std::string_view filepath) {
  ClonalCartelAgent agent;
  char delimiter = Utils::GetDelimiter(filepath);
  std::unordered_map<int, std::string> index_to_header_map =
      Utils::GetIndexToHeaderMap(delimiter, filepath);
  std::unordered_map<std::string, int> header_to_index_map =
      Utils::GetHeaderToIndexMap(delimiter, filepath);
  std::ifstream clonal_cartel_agent_stream{std::string(filepath)};
  std::string line;
  int line_no = 1;
  while (std::getline(clonal_cartel_agent_stream, line)) {
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.size() == 0) {
      break;
    }
    if (line_no > 1) {
      for (size_t i = 0; i < current_line.size(); i++) {
        std::string current_header = index_to_header_map[i];
        if (current_header == "num_authors") {
          agent.num_authors = std::stoi(current_line[i]);
        }
        if (current_header == "pa_weight") {
          agent.pa_weight = std::stod(current_line[i]);
        }
        if (current_header == "fit_weight") {
          agent.fit_weight = std::stod(current_line[i]);
        }
        if (current_header == "num_authors_weight") {
          agent.num_authors_weight = std::stod(current_line[i]);
        }
        if (current_header == "author_reputation_weight") {
          agent.author_reputation_weight = std::stod(current_line[i]);
        }
        if (current_header == "out_degree") {
          agent.out_degree = std::stoi(current_line[i]);
        }
        if (current_header == "alpha") {
          agent.alpha = std::stod(current_line[i]);
        }
        if (current_header == "fitness_lag_duration") {
          agent.fitness_lag_duration = std::stoi(current_line[i]);
        }
        if (current_header == "fitness_peak_value") {
          agent.fitness_peak_value = std::stoi(current_line[i]);
        }
        if (current_header == "fitness_peak_duration") {
          agent.fitness_peak_duration = std::stoi(current_line[i]);
        }
      }
    }
    line_no += 1;
  }
  return agent;
}

std::vector<int> Utils::ParseOutDegreeBag(std::string_view filepath) {
  std::vector<int> vec;
  char delimiter = ',';
  std::ifstream out_degree_bag_stream{std::string(filepath)};
  std::string line;
  int line_no = 0;
  while (std::getline(out_degree_bag_stream, line)) {
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.size() == 0) {
      break;
    }
    if (line_no != 0) {
      int current_out_degree = std::stoi(current_line[1]);
      if (current_out_degree > 1500) {
        std::cerr
            << "Maximum out-degrree defined in the header file is less than "
               "the out-degree read from the input out-degree bag. Aborting."
            << std::endl;
        exit(1);
      }
      vec.push_back(current_out_degree);
    }
    line_no++;
  }
  return vec;
}

std::unordered_map<
    int, std::unordered_map<int, std::unordered_map<std::string, std::string>>>
Utils::ParsePlantedNodes(std::string_view filepath) {
  std::unordered_map<
      int,
      std::unordered_map<int, std::unordered_map<std::string, std::string>>>
      planted_nodes_map;
  char delimiter = Utils::GetDelimiter(filepath);
  std::unordered_map<int, std::string> index_to_header_map =
      Utils::GetIndexToHeaderMap(delimiter, filepath);
  std::unordered_map<std::string, int> header_to_index_map =
      Utils::GetHeaderToIndexMap(delimiter, filepath);
  std::ifstream planted_nodes_stream{std::string(filepath)};
  std::string line;
  int line_no = 1;
  while (std::getline(planted_nodes_stream, line)) {
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.size() == 0) {
      break;
    }
    if (line_no > 1) {
      int year_header_index = header_to_index_map["year"];
      int current_year = std::stoi(current_line[year_header_index]);
      for (size_t i = 0; i < current_line.size(); i++) {
        if ((int)i != year_header_index) {
          planted_nodes_map[current_year][line_no][index_to_header_map[i]] =
              current_line[i];
        }
      }
    }
    line_no += 1;
  }
  return planted_nodes_map;
}

std::unordered_map<int, int>
Utils::ParseRecencyProbabilities(std::string_view filepath) {
  std::unordered_map<int, int> counts_map;
  char delimiter = ',';
  std::ifstream stream{std::string(filepath)};
  std::string line;
  int line_no = 0;
  while (std::getline(stream, line)) {
    std::stringstream ss(line);
    std::string current_value;
    std::vector<std::string> current_line;
    while (std::getline(ss, current_value, delimiter)) {
      current_line.push_back(current_value);
    }
    if (current_line.size() == 0) {
      break;
    }
    if (line_no != 0) {
      int integer_year_diff = std::stoi(current_line[0]);
      int count = std::stoi(current_line[1]);
      if (integer_year_diff > 0) {
        counts_map[integer_year_diff] = count;
      }
    }
    line_no++;
  }
  return counts_map;
}

void SimLogger::Init(std::string log_file, std::string timing_file, int level) {
  this->log_level = level;
  if (this->log_level > 0) {
    this->log_file_handle.open(log_file, std::ios::out);
    this->timing_file_handle.open(timing_file, std::ios::out);
  }
  this->start_time = std::chrono::steady_clock::now();
  this->prev_time = this->start_time;
  this->global_prev_time = this->start_time;
  this->num_calls_to_log_write = 0;
}

int SimLogger::WriteToLogFile(std::string message, Log message_type) {
  if (this->log_level >= (int)message_type) {
    std::chrono::time_point<std::chrono::steady_clock> now =
        std::chrono::steady_clock::now();
    std::string log_message_prefix;
    if (message_type == Log::info) {
      log_message_prefix = "[INFO]";
    } else if (message_type == Log::debug) {
      log_message_prefix = "[DEBUG]";
    } else if (message_type == Log::error) {
      log_message_prefix = "[ERROR]";
    }
    auto days_elapsed =
        std::chrono::duration_cast<std::chrono::days>(now - this->start_time);
    auto hours_elapsed = std::chrono::duration_cast<std::chrono::hours>(
        now - this->start_time - days_elapsed);
    auto minutes_elapsed = std::chrono::duration_cast<std::chrono::minutes>(
        now - this->start_time - days_elapsed - hours_elapsed);
    auto seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - this->start_time - days_elapsed - hours_elapsed -
        minutes_elapsed);
    auto total_seconds_elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now -
                                                         this->start_time);
    log_message_prefix += "[";
    log_message_prefix += std::to_string(days_elapsed.count());
    log_message_prefix += "-";
    log_message_prefix += std::to_string(hours_elapsed.count());
    log_message_prefix += ":";
    log_message_prefix += std::to_string(minutes_elapsed.count());
    log_message_prefix += ":";
    log_message_prefix += std::to_string(seconds_elapsed.count());
    log_message_prefix += "]";

    log_message_prefix += "(t=";
    log_message_prefix += std::to_string(total_seconds_elapsed.count());
    log_message_prefix += "s)";
    this->log_file_handle << log_message_prefix << " " << message << '\n';

    if (this->num_calls_to_log_write % 1 == 0) {
      std::flush(this->log_file_handle);
    }
    this->num_calls_to_log_write++;
  }
  return 0;
}

void SimLogger::LogTime(int current_year, std::string label) {
  if (this->log_level == 1) {
    std::chrono::time_point<std::chrono::steady_clock> current_time =
        std::chrono::steady_clock::now();
    auto milliseconds_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time -
                                                              this->prev_time);
    this->timing_map[current_year].push_back(
        {label, (int)milliseconds_elapsed.count()});
    this->prev_time = current_time;
    this->timing_file_handle
        << std::to_string(current_year) << "," << label << ","
        << std::to_string(milliseconds_elapsed.count()) << "\n";
    std::flush(this->timing_file_handle);
  }
}

void SimLogger::LogTime(int current_year, std::string label, int time_elapsed) {
  if (this->log_level == 1) {
    this->timing_file_handle << std::to_string(current_year) << "," << label
                             << "," << std::to_string(time_elapsed) << "\n";
    std::flush(this->timing_file_handle);
  }
}

std::chrono::time_point<std::chrono::steady_clock> SimLogger::LocalLogTime(
    std::vector<std::pair<std::string, int>> &local_parallel_stage_time_vec,
    std::chrono::time_point<std::chrono::steady_clock> local_prev_time,
    std::string label) {
  std::chrono::time_point<std::chrono::steady_clock> current_time =
      std::chrono::steady_clock::now();
  if (this->log_level == 1) {
    auto milliseconds_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time -
                                                              local_prev_time);
    local_parallel_stage_time_vec.push_back(
        {label, (int)milliseconds_elapsed.count()});
  }
  return current_time;
}

void SimLogger::WriteTimingFile(int start_year, int end_year) {
  for (int i = start_year; i < end_year; i++) {
    for (size_t j = 0; j < this->timing_map[i].size(); j++) {
      this->timing_file_handle
          << std::to_string(i) << "," << (this->timing_map[i][j]).first << ","
          << std::to_string((this->timing_map[i][j]).second) << "\n";
    }
  }
}

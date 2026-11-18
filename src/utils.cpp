#include "utils.h"

#include <sstream>
#include <stdexcept>

namespace Utils {

namespace {
std::vector<std::string> SplitLine(const std::string &line, char delim) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, delim)) {
    fields.push_back(field);
  }
  return fields;
}
} // namespace

std::vector<int> ParseIntBag(std::string_view filepath, const char *what) {
  std::ifstream in{std::string(filepath)};
  if (!in.is_open()) {
    throw std::runtime_error(std::string("Could not open ") + what + " file: " + std::string(filepath));
  }
  std::vector<int> values;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) {
      header = false;
      continue;
    }
    if (line.empty() || line == "\r") {
      continue;
    }
    std::vector<std::string> fields = SplitLine(line, ',');
    if (fields.size() < 2) {
      continue;
    }
    values.push_back(std::stoi(fields[1]));
  }
  if (values.empty()) {
    throw std::runtime_error(std::string(what) + " file is empty or malformed: " + std::string(filepath));
  }
  return values;
}

std::map<int, long long> ParseRecencyCounts(std::string_view filepath) {
  std::ifstream in{std::string(filepath)};
  if (!in.is_open()) {
    throw std::runtime_error("Could not open recency_table file: " + std::string(filepath));
  }
  std::map<int, long long> counts;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (header) {
      header = false;
      continue;
    }
    std::vector<std::string> fields = SplitLine(line, ',');
    if (fields.size() < 2) {
      continue;
    }
    int diff = std::stoi(fields[0]);
    long long count = std::stoll(fields[1]);
    if (diff > 0) {
      counts[diff] = count;
    }
  }
  return counts;
}

} // namespace Utils

void SimLogger::Init(const std::string &log_file, const std::string &timing_file, int level, bool append) {
  level_ = level;
  if (level_ > 0) {
    auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
    log_.open(log_file, mode);
    timing_.open(timing_file, mode);
  }
  start_ = std::chrono::steady_clock::now();
  lap_ = start_;
}

void SimLogger::Write(const std::string &message, Log type) {
  if (level_ < static_cast<int>(type) || !log_.is_open()) {
    return;
  }
  auto now = std::chrono::steady_clock::now();
  long long total = std::chrono::duration_cast<std::chrono::seconds>(now - start_).count();
  long long days = total / 86400;
  long long hours = (total % 86400) / 3600;
  long long minutes = (total % 3600) / 60;
  long long seconds = total % 60;
  const char *prefix = (type == Log::info) ? "[INFO]" : (type == Log::debug) ? "[DEBUG]" : "[ERROR]";
  log_ << prefix << "[" << days << "-" << hours << ":" << minutes << ":" << seconds << "](t=" << total
       << "s) " << message << '\n';
  log_.flush();
}

void SimLogger::Lap(int year, const std::string &label) {
  auto now = std::chrono::steady_clock::now();
  long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lap_).count();
  lap_ = now;
  if (level_ == 1 && timing_.is_open()) {
    timing_ << year << "," << label << "," << ms << "\n";
    timing_.flush();
  }
}

void SimLogger::ResetLap() { lap_ = std::chrono::steady_clock::now(); }

void SimLogger::Value(int year, const std::string &label, long long value) {
  if (level_ == 1 && timing_.is_open()) {
    timing_ << year << "," << label << "," << value << "\n";
    timing_.flush();
  }
}

double SimLogger::ElapsedSeconds() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

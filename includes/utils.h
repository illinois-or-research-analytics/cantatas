#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum class Log { error = -1, debug = 0, info = 1 };

namespace Utils {
// Second column of a CSV with a header row (out-degree bag, author-count bag).
std::vector<int> ParseIntBag(std::string_view filepath, const char *what);
// Recency table: header row, then "year difference, count" rows.
std::map<int, long long> ParseRecencyCounts(std::string_view filepath);
} // namespace Utils

/*
 * SimLogger: the run log (log.txt) and the per-stage timing CSV
 * (log.txt_timing: year,label,value). Formats match the previous versions.
 */
class SimLogger {
public:
  // append = true keeps earlier content (used when a run resumes).
  void Init(const std::string &log_file, const std::string &timing_file, int level, bool append);
  void Write(const std::string &message, Log type = Log::info);
  // Wall-clock ms since the previous call (or since ResetLap()), then resets.
  void Lap(int year, const std::string &label);
  void ResetLap();
  // Records a value (count, or summed per-thread milliseconds) without
  // touching the lap clock.
  void Value(int year, const std::string &label, long long value);
  double ElapsedSeconds() const;

private:
  int level_ = 0;
  std::ofstream log_;
  std::ofstream timing_;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point lap_ = start_;
};

#endif // UTILS_H

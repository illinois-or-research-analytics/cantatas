#include "signal_supervisor.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <ctime>

std::atomic<bool> SignalSupervisor::signal_caught_{false};

namespace {
const std::chrono::system_clock::time_point kProcessStart = std::chrono::system_clock::now();

double NowEpoch() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double ParseEpoch(const char *value) {
  if (value == nullptr || *value == '\0') return 0;
  char *end = nullptr;
  double v = std::strtod(value, &end);
  return (end != value && v > 0) ? v : 0;
}
} // namespace

void SignalSupervisor::Handler(int) {
  // Only an atomic store: async-signal-safe.
  signal_caught_.store(true, std::memory_order_relaxed);
}

void SignalSupervisor::Register() {
  std::signal(SIGTERM, Handler);
  std::signal(SIGINT, Handler);
#ifdef SIGUSR1
  std::signal(SIGUSR1, Handler);
#endif
}

void SignalSupervisor::Configure(double wall_clock_limit_seconds, double safety_margin_seconds) {
  safety_margin_seconds_ = safety_margin_seconds;
  if (double v = ParseEpoch(std::getenv("CANTATAS_DEADLINE_EPOCH")); v > 0) {
    deadline_epoch_ = v;
    deadline_source_ = "CANTATAS_DEADLINE_EPOCH";
  } else if (double s = ParseEpoch(std::getenv("SLURM_JOB_END_TIME")); s > 0) {
    deadline_epoch_ = s;
    deadline_source_ = "SLURM_JOB_END_TIME";
  } else if (wall_clock_limit_seconds > 0) {
    deadline_epoch_ =
        std::chrono::duration<double>(kProcessStart.time_since_epoch()).count() + wall_clock_limit_seconds;
    deadline_source_ = "wall_clock_limit";
  }
}

double SignalSupervisor::SecondsRemaining() const {
  if (deadline_epoch_ <= 0) return 1e18;
  return deadline_epoch_ - NowEpoch();
}

bool SignalSupervisor::ShouldStop(double save_seconds) const {
  if (SignalCaught()) return true;
  if (deadline_epoch_ <= 0) return false;
  double needed = std::max(safety_margin_seconds_, 1.3 * last_cycle_seconds_) + save_seconds;
  return SecondsRemaining() <= needed;
}

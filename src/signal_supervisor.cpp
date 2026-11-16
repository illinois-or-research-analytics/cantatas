#include "signal_supervisor.h"
#include <algorithm>
#include <iostream>

namespace ladybug {

std::atomic<bool> SignalSupervisor::signal_caught_{false};
std::chrono::time_point<std::chrono::steady_clock> SignalSupervisor::process_start_time_ =
    std::chrono::steady_clock::now();

SignalSupervisor::SignalSupervisor()
    : limit_seconds_(0.0),
      safety_margin_seconds_(1200.0),
      last_cycle_duration_seconds_(0.0) {}

SignalSupervisor::~SignalSupervisor() {}

void SignalSupervisor::RecordProcessStartTime() noexcept {
  process_start_time_ = std::chrono::steady_clock::now();
}

void SignalSupervisor::SignalHandler(int signum) {
  signal_caught_.store(true, std::memory_order_relaxed);
  std::cerr << "\n[SignalSupervisor] Caught termination signal (" << signum
            << "). Requesting clean annual BSP flush before exit.\n" << std::endl;
}

void SignalSupervisor::Register() {
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);
#if defined(SIGUSR1)
  std::signal(SIGUSR1, SignalHandler);
#endif
}

void SignalSupervisor::SetWallClockLimit(double limit_seconds, double safety_margin_seconds) {
  limit_seconds_ = limit_seconds;
  safety_margin_seconds_ = safety_margin_seconds;
}

void SignalSupervisor::RecordCycleDuration(double seconds) noexcept {
  last_cycle_duration_seconds_ = seconds;
}

bool SignalSupervisor::IsSignalCaught() noexcept {
  return signal_caught_.load(std::memory_order_relaxed);
}

void SignalSupervisor::ResetSignal() noexcept {
  signal_caught_.store(false, std::memory_order_relaxed);
}

double SignalSupervisor::GetElapsedSeconds() const noexcept {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - process_start_time_).count();
}

double SignalSupervisor::GetRemainingSeconds() const noexcept {
  if (limit_seconds_ <= 0.0) {
    return 1e9; // Unlimited
  }
  double elapsed = GetElapsedSeconds();
  return std::max(0.0, limit_seconds_ - elapsed);
}

bool SignalSupervisor::ShouldStop() const {
  if (IsSignalCaught()) {
    return true;
  }
  if (limit_seconds_ > 0.0) {
    double remaining = GetRemainingSeconds();
    double required_buffer =
        std::max(safety_margin_seconds_, last_cycle_duration_seconds_ * 1.3);
    if (remaining <= required_buffer) {
      return true;
    }
  }
  return false;
}

} // namespace ladybug

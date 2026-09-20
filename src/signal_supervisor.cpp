#include "signal_supervisor.h"
#include <iostream>

namespace ladybug {

std::atomic<bool> SignalSupervisor::signal_caught_{false};

SignalSupervisor::SignalSupervisor()
    : start_time_(std::chrono::steady_clock::now()),
      limit_seconds_(0.0),
      safety_margin_seconds_(1200.0) {}

SignalSupervisor::~SignalSupervisor() {}

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
  start_time_ = std::chrono::steady_clock::now();
}

bool SignalSupervisor::IsSignalCaught() noexcept {
  return signal_caught_.load(std::memory_order_relaxed);
}

void SignalSupervisor::ResetSignal() noexcept {
  signal_caught_.store(false, std::memory_order_relaxed);
}

double SignalSupervisor::GetElapsedSeconds() const noexcept {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - start_time_).count();
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
    if (remaining <= safety_margin_seconds_) {
      return true;
    }
  }
  return false;
}

} // namespace ladybug

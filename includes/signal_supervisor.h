#ifndef SIGNAL_SUPERVISOR_H
#define SIGNAL_SUPERVISOR_H

#include <atomic>
#include <chrono>
#include <csignal>

namespace ladybug {

/*
 * SignalSupervisor: Monitors Slurm wall-clock execution limits and traps termination signals
 * (SIGTERM, SIGUSR1, SIGINT) to guarantee clean annual BSP checkpoint flushes before job expiration.
 */
class SignalSupervisor {
public:
  SignalSupervisor();
  ~SignalSupervisor();

  // Register POSIX signal handlers
  static void Register();

  // Configure wall-clock limits (e.g. 4-hour Slurm window with 20-minute safety buffer)
  void SetWallClockLimit(double limit_seconds, double safety_margin_seconds = 1200.0);

  // Evaluates whether the simulation should gracefully pause at the next annual boundary
  bool ShouldStop() const;

  // Signal status accessors
  static bool IsSignalCaught() noexcept;
  static void ResetSignal() noexcept;

  double GetElapsedSeconds() const noexcept;
  double GetRemainingSeconds() const noexcept;

private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  double limit_seconds_;
  double safety_margin_seconds_;

  static std::atomic<bool> signal_caught_;
  static void SignalHandler(int signum);
};

} // namespace ladybug

#endif // SIGNAL_SUPERVISOR_H

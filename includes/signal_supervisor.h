#ifndef SIGNAL_SUPERVISOR_H
#define SIGNAL_SUPERVISOR_H

#include <atomic>
#include <chrono>
#include <string>

/*
 * SignalSupervisor: decides when a run must pause so it can save a snapshot
 * before the job ends.
 *
 * The deadline comes from, in order: the CANTATAS_DEADLINE_EPOCH environment
 * variable (UNIX time, set by the job script from squeue), SLURM_JOB_END_TIME
 * if Slurm provides it, or wall_clock_limit seconds after process start.
 * Before each year the run continues only if the time left covers
 * max(safety margin, 1.3 x last year) plus the time to write a snapshot.
 * SIGTERM, SIGINT and SIGUSR1 request a pause at the next year boundary.
 */
class SignalSupervisor {
public:
  static void Register();
  static bool SignalCaught() noexcept { return signal_caught_.load(std::memory_order_relaxed); }

  void Configure(double wall_clock_limit_seconds, double safety_margin_seconds);
  // Seconds until the deadline; a very large number when there is none.
  double SecondsRemaining() const;
  bool HasDeadline() const noexcept { return deadline_epoch_ > 0; }
  const std::string &DeadlineSource() const noexcept { return deadline_source_; }

  void RecordCycle(double seconds) noexcept { last_cycle_seconds_ = seconds; }
  // True if another year plus `save_seconds` would not fit before the deadline.
  bool ShouldStop(double save_seconds) const;

private:
  static void Handler(int signum);
  static std::atomic<bool> signal_caught_;

  double deadline_epoch_ = 0; // UNIX seconds; 0 = none
  std::string deadline_source_ = "none";
  double safety_margin_seconds_ = 1200.0;
  double last_cycle_seconds_ = 0.0;
};

#endif // SIGNAL_SUPERVISOR_H

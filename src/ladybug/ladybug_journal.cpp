#include "ladybug/ladybug_journal.h"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace ladybug {

LadyBugJournal::LadyBugJournal()
    : mode_(MmapMode::ReadWrite), fd_(-1), entry_count_(0) {}

LadyBugJournal::~LadyBugJournal() {
  Close();
}

void LadyBugJournal::Open(const std::string &journal_path, MmapMode mode) {
  Close();
  journal_path_ = journal_path;
  mode_ = mode;

  int flags = (mode == MmapMode::ReadOnly)
                  ? O_RDONLY
                  : (O_RDWR | O_CREAT | O_APPEND);
  mode_t create_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

  fd_ = open(journal_path_.c_str(), flags, create_mode);
  if (fd_ < 0) {
    throw std::runtime_error("LadyBugJournal::Open failed to open '" + journal_path_ +
                             "': " + std::string(strerror(errno)));
  }

  struct stat st;
  if (fstat(fd_, &st) == 0) {
    entry_count_ = st.st_size / sizeof(JournalNodeEntry);
  }
}

void LadyBugJournal::Close() {
  if (fd_ >= 0) {
    Sync(false);
    close(fd_);
    fd_ = -1;
  }
  entry_count_ = 0;
}

void LadyBugJournal::Sync(bool /*async*/) {
  if (fd_ >= 0 && mode_ != MmapMode::ReadOnly) {
    fdatasync(fd_);
  }
}

void LadyBugJournal::AppendNode(int32_t node_id, uint16_t year, int32_t author_id, int32_t cluster_id) {
  if (fd_ < 0) return;
  JournalNodeEntry entry;
  entry.node_id = node_id;
  entry.year = year;
  entry.author_id = author_id;
  entry.cluster_id = cluster_id;

  ssize_t written = write(fd_, &entry, sizeof(entry));
  if (written > 0) {
    entry_count_++;
  }
}

void LadyBugJournal::AppendEdge(int32_t source, int32_t target) {
  if (fd_ < 0) return;
  JournalEdgeEntry entry;
  entry.source = source;
  entry.target = target;

  ssize_t written = write(fd_, &entry, sizeof(entry));
  if (written > 0) {
    entry_count_++;
  }
}

void LadyBugJournal::AppendCommit(int32_t year, uint64_t node_count, uint64_t edge_count) {
  if (fd_ < 0) return;
  JournalCommitEntry entry;
  entry.year = year;
  entry.node_count = node_count;
  entry.edge_count = edge_count;
  entry.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

  ssize_t written = write(fd_, &entry, sizeof(entry));
  if (written > 0) {
    entry_count_++;
  }
  Sync(false);
}

void LadyBugJournal::Truncate() {
  if (fd_ >= 0 && mode_ != MmapMode::ReadOnly) {
    if (ftruncate(fd_, 0) == 0) {
      lseek(fd_, 0, SEEK_SET);
      entry_count_ = 0;
    }
  }
}

} // namespace ladybug

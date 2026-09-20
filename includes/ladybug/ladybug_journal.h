#ifndef LADYBUG_JOURNAL_H
#define LADYBUG_JOURNAL_H

#include "ladybug_mmap.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ladybug {

enum class JournalEntryType : uint8_t {
  Node = 1,
  Edge = 2,
  SuperstepCommit = 3
};

#pragma pack(push, 1)
struct JournalNodeEntry {
  uint8_t type = static_cast<uint8_t>(JournalEntryType::Node);
  int32_t node_id;
  uint16_t year;
  int32_t author_id;
  int32_t cluster_id;
};

struct JournalEdgeEntry {
  uint8_t type = static_cast<uint8_t>(JournalEntryType::Edge);
  int32_t source;
  int32_t target;
};

struct JournalCommitEntry {
  uint8_t type = static_cast<uint8_t>(JournalEntryType::SuperstepCommit);
  int32_t year;
  uint64_t node_count;
  uint64_t edge_count;
  uint64_t timestamp;
};
#pragma pack(pop)

/*
 * LadyBugJournal: High-speed append-only binary journal on NVMe storage.
 * Maintains transactional durability during active simulation years and is atomically
 * committed to the master manifest and truncated at annual BSP boundaries (zero-bloat).
 */
class LadyBugJournal {
public:
  LadyBugJournal();
  ~LadyBugJournal();

  void Open(const std::string &journal_path, MmapMode mode = MmapMode::ReadWrite);
  void Close();
  void Sync(bool async = false);

  void AppendNode(int32_t node_id, uint16_t year, int32_t author_id, int32_t cluster_id);
  void AppendEdge(int32_t source, int32_t target);
  void AppendCommit(int32_t year, uint64_t node_count, uint64_t edge_count);

  void Truncate();
  size_t GetEntryCount() const noexcept { return entry_count_; }

private:
  std::string journal_path_;
  MmapMode mode_;
  int fd_;
  size_t entry_count_;
};

} // namespace ladybug

#endif // LADYBUG_JOURNAL_H
